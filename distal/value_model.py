"""Distributional value function using SmolVLM backbone.

Initializes from a trained SmolVLA policy checkpoint, extracting just the VLM
(vision encoder + language model). Uses the last token's hidden state as the
value representation, fed through an MLP value head that predicts returns as a
categorical distribution over discrete bins (RECAP-style from pi0.6).

Supports partial unfreezing: vision encoder is always frozen, and optionally
only the top N language model layers are unfrozen for fine-tuning.
"""

import math
from dataclasses import dataclass, field
from typing import cast

import torch
import torch.nn.functional as F
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig, OptimizerConfig
from lerobot.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
    LRSchedulerConfig,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.modeling_smolvla import (
    pad_vector,
    resize_with_pad,
)
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)
from torch import Tensor, nn
from transformers import AutoModelForImageTextToText, AutoProcessor


@PreTrainedConfig.register_subclass("value")
@dataclass
class ValueConfig(PreTrainedConfig):
    pretrained_path: str = "reece-omahoney/adv-libero-base"
    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    resize_imgs_with_padding: tuple[int, int] = (256, 256)
    max_state_dim: int = 8
    n_bins: int = 50
    tokenizer_max_length: int = 48
    num_vlm_layers: int = 16
    freeze_vision_encoder: bool = True
    num_unfrozen_layers: int = 3
    hl_gauss_sigma: float = 0.0

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
        }
    )

    @property
    def observation_delta_indices(self) -> list | None:
        return None

    @property
    def action_delta_indices(self) -> list | None:
        return None

    @property
    def reward_delta_indices(self) -> list | None:
        return None

    def get_optimizer_preset(self) -> OptimizerConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    def validate_features(self) -> None:
        pass


class ValueFunction(PreTrainedPolicy):
    name = "value"
    config_class = ValueConfig
    config: ValueConfig

    def __init__(self, config: ValueConfig, **kwargs):
        super().__init__(config)

        self.vlm = AutoModelForImageTextToText.from_pretrained(
            config.vlm_model_name,
            torch_dtype="bfloat16",
            low_cpu_mem_usage=True,
        )
        self.processor = AutoProcessor.from_pretrained(config.vlm_model_name)

        # Truncate VLM layers if needed
        if config.num_vlm_layers > 0:
            text_model = self.vlm.model.text_model
            text_model.layers = text_model.layers[: config.num_vlm_layers]

        vlm_hidden = self.vlm.config.text_config.hidden_size

        # State projection (same as SmolVLA)
        self.state_proj = nn.Linear(config.max_state_dim, vlm_hidden)

        # Value head: vlm_hidden → logits over bins
        self.value_head = nn.Sequential(
            nn.Linear(vlm_hidden, vlm_hidden),
            nn.SiLU(),
            nn.Linear(vlm_hidden, config.n_bins),
        )

        # Bin centers for computing expected value
        self.register_buffer("bin_centers", torch.linspace(-1.0, 0.0, config.n_bins))

        # Special tokens for image wrapping
        self.fake_image_token = self.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)

        self.set_requires_grad()

    def load_vlm_from_policy(self, policy_path: str):
        """Load VLM weights from a trained SmolVLA policy checkpoint.

        Extracts the VLM and state_proj weights from the policy's
        model.vlm_with_expert state dict. Accepts a local path or a
        HuggingFace Hub repo ID.
        """
        from pathlib import Path

        from safetensors.torch import load_file

        path = Path(policy_path)
        if path.exists():
            safetensors_path = (
                str(path)
                if path.suffix == ".safetensors"
                else str(path / "model.safetensors")
            )
        else:
            from huggingface_hub import hf_hub_download

            safetensors_path = hf_hub_download(
                policy_path, "model.safetensors", repo_type="model"
            )

        policy_weights = load_file(safetensors_path)

        vlm_state = {}
        state_proj_state = {}
        for key, value in policy_weights.items():
            # SmolVLA stores VLM as model.vlm_with_expert.vlm.*
            if key.startswith("model.vlm_with_expert.vlm."):
                new_key = key.removeprefix("model.vlm_with_expert.vlm.")
                vlm_state[new_key] = value
            # State projection
            elif key.startswith("model.state_proj."):
                new_key = key.removeprefix("model.state_proj.")
                state_proj_state[new_key] = value

        # Handle layer truncation: filter out layers beyond num_vlm_layers
        if self.config.num_vlm_layers > 0:
            filtered_vlm_state = {}
            for key, value in vlm_state.items():
                if "text_model.layers." in key:
                    parts = key.split(".")
                    layer_idx_pos = parts.index("layers") + 1
                    layer_idx = int(parts[layer_idx_pos])
                    if layer_idx >= self.config.num_vlm_layers:
                        continue
                filtered_vlm_state[key] = value
            vlm_state = filtered_vlm_state

        missing, unexpected = self.vlm.load_state_dict(vlm_state, strict=False)
        # lm_head is expected to be missing since we don't use it
        missing = [k for k in missing if not k.startswith("lm_head")]
        if missing:
            print(f"Warning: missing VLM keys: {missing[:10]}")
        if unexpected:
            print(f"Warning: unexpected VLM keys: {unexpected[:10]}")

        if state_proj_state:
            self.state_proj.load_state_dict(state_proj_state)
            print("Loaded state_proj from policy checkpoint")

        print(f"Loaded VLM weights from {policy_path}")

    def set_requires_grad(self):
        """Freeze vision encoder and optionally freeze all but top N LM layers."""
        # Freeze vision encoder
        vision_model = self.vlm.model.vision_model
        vision_model.eval()
        for p in vision_model.parameters():
            p.requires_grad = False

        # Freeze connector (multimodal projector) - keep frozen like vision
        for p in self.vlm.model.connector.parameters():
            p.requires_grad = False

        # Freeze lm_head (unused for value prediction)
        if hasattr(self.vlm, "lm_head"):
            for p in self.vlm.lm_head.parameters():
                p.requires_grad = False

        # Freeze LM layers except the top N
        text_model = self.vlm.model.text_model
        num_layers = len(text_model.layers)
        num_frozen = num_layers - self.config.num_unfrozen_layers

        # Freeze embeddings
        if text_model.get_input_embeddings() is not None:
            for p in text_model.get_input_embeddings().parameters():
                p.requires_grad = False

        # Freeze bottom layers
        for i in range(num_frozen):
            for p in text_model.layers[i].parameters():
                p.requires_grad = False

        # Final norm stays trainable (it's after the unfrozen layers)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(
            f"Value model: {trainable:,} trainable / {total:,} total params "
            f"({num_frozen} frozen + {self.config.num_unfrozen_layers} "
            f"unfrozen LM layers)"
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep vision encoder in eval mode
        self.vlm.model.vision_model.eval()
        return self

    def embed_image(self, image: Tensor) -> Tensor:
        """Embed image through vision encoder + connector."""
        vision_model = self.vlm.model.vision_model
        image_hidden_states = vision_model(
            pixel_values=image.to(dtype=vision_model.dtype),
            patch_attention_mask=None,
        ).last_hidden_state
        image_hidden_states = self.vlm.model.connector(image_hidden_states)
        return image_hidden_states

    def embed_language_tokens(self, tokens: Tensor) -> Tensor:
        """Embed language tokens through the VLM's text embedding layer."""
        return self.vlm.model.text_model.get_input_embeddings()(tokens)

    def prepare_images(
        self, batch: dict[str, Tensor]
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Resize, pad, and normalize images for SigLIP."""
        img_keys = sorted(k for k in batch if k.startswith("observation.images."))
        images = []
        img_masks = []
        for key in img_keys:
            img = batch[key]
            img = img[:, -1] if img.ndim == 5 else img
            w, h = self.config.resize_imgs_with_padding
            img = resize_with_pad(img, w, h, pad_value=-1)
            img = img * 2.0 - 1.0
            images.append(img)
            mask = torch.ones(img.shape[0], dtype=torch.bool, device=img.device)
            img_masks.append(mask)
        return images, img_masks

    def prepare_state(self, batch: dict[str, Tensor]) -> Tensor:
        """Pad state to max_state_dim."""
        state = batch[OBS_STATE]
        state = state[:, -1] if state.ndim > 2 else state
        return pad_vector(state, self.config.max_state_dim)

    def compute_logits(self, batch: dict[str, Tensor]) -> Tensor:
        """Forward pass returning logits over value bins.

        Embeds images, language, and state into a single sequence, runs through
        the VLM text model, and extracts the last token's hidden state for the
        value head.
        """
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        # Build sequence: [images... | language | state]
        embs: list[Tensor] = []
        pad_masks: list[Tensor] = []

        for img, img_mask in zip(images, img_masks, strict=False):
            img_emb = self.embed_image(img)
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(
                img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device
            )
            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)
            embs.append(img_emb)
            pad_masks.append(img_mask)

        # Language tokens
        lang_emb = self.embed_language_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # State (last tokens — value will be read from here)
        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        bsize = state_emb.shape[0]
        device = state_emb.device
        state_mask = torch.ones(
            bsize, state_emb.shape[1], dtype=torch.bool, device=device
        )
        embs.append(state_emb)
        pad_masks.append(state_mask)

        # Concatenate
        inputs_embeds = torch.cat(embs, dim=1)
        pad_mask = torch.cat(pad_masks, dim=1)

        # Build causal attention mask (1 = attend, 0 = ignore)
        attention_mask = pad_mask.long()

        # Position IDs from attention mask
        position_ids = torch.cumsum(attention_mask, dim=1) - 1
        position_ids = position_ids.clamp(min=0)

        # Forward through VLM text model
        text_model = self.vlm.model.text_model
        outputs = text_model(
            inputs_embeds=inputs_embeds.to(
                dtype=text_model.layers[0].self_attn.q_proj.weight.dtype
            ),
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        # Extract last real token per sample
        seq_lengths = attention_mask.sum(dim=1) - 1  # (B,)
        hidden_states = outputs.last_hidden_state  # (B, L, H)
        last_hidden = hidden_states[
            torch.arange(bsize, device=device), seq_lengths
        ]  # (B, H)

        logits = self.value_head(last_hidden.to(dtype=torch.float32))
        return logits

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Training forward pass: compute cross-entropy loss over value bins.

        Expects batch to contain a "returns" key with (B,) float tensor in [-1, 0].
        When hl_gauss_sigma > 0, uses soft Gaussian targets (HL-Gauss) instead of
        one-hot targets.
        """
        logits = self.compute_logits(batch)
        returns = batch["returns"]

        if self.config.hl_gauss_sigma > 0:
            targets = self.returns_to_hl_gauss(
                returns, self.config.n_bins, self.config.hl_gauss_sigma
            )
        else:
            targets = self.returns_to_bins(returns, self.config.n_bins)

        loss = F.cross_entropy(logits, targets)

        with torch.no_grad():
            pred_values = self.logits_to_value(logits)
            mae = (pred_values - returns).abs().mean().item()

        return loss, {"mae": mae}

    def predict_value(self, batch: dict[str, Tensor]) -> Tensor:
        """Compute expected value for a batch of observations."""
        logits = self.compute_logits(batch)
        return self.logits_to_value(logits)

    def logits_to_value(self, logits: Tensor) -> Tensor:
        """Expected value from logits via softmax + dot with bin centers."""
        probs = F.softmax(logits, dim=-1)
        return (probs * cast(Tensor, self.bin_centers)).sum(dim=-1)

    @staticmethod
    def returns_to_bins(returns: Tensor, n_bins: int = 50) -> Tensor:
        """Convert return values in [-1, 0] to one-hot bin targets."""
        returns = returns.clamp(-1.0, 0.0)
        bin_indices = ((returns + 1.0) * (n_bins - 1)).long()
        bin_indices = bin_indices.clamp(0, n_bins - 1)
        return F.one_hot(bin_indices, num_classes=n_bins).float()

    def returns_to_hl_gauss(self, returns: Tensor, n_bins: int, sigma: float) -> Tensor:
        """Convert returns to soft Gaussian targets over bins (HL-Gauss)."""
        returns = returns.clamp(-1.0, 0.0)
        bin_centers = cast(Tensor, self.bin_centers)
        diff = returns.unsqueeze(-1) - bin_centers.unsqueeze(0)
        log_probs = -0.5 * (diff / sigma) ** 2
        return F.softmax(log_probs, dim=-1)

    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        raise NotImplementedError("ValueFunction does not produce actions.")

    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        raise NotImplementedError("ValueFunction does not produce actions.")

    def reset(self):
        pass

    def get_optim_params(self) -> dict:
        return {"params": [p for p in self.parameters() if p.requires_grad]}
