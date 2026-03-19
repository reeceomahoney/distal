"""Advantage-conditioned policy model.

Extends SmolVLAPolicy with a learned advantage embedding injected between the
prefix (images+language+state) and suffix (noisy actions+time) during training,
and always set to positive (advantage=1) during inference.
"""

import logging
from typing import cast

import torch
from lerobot.policies.smolvla.modeling_smolvla import (
    SmolVLAPolicy,
    VLAFlowMatching,
    make_att_2d_masks,
)
from torch import Tensor, nn

from .configuration_advantage import AdvantageConfig


class AdvantageEmbedding(nn.Module):
    """Learned embedding for binary advantage indicator (negative=0, positive=1)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(2, hidden_size)

    def forward(self, labels: Tensor) -> Tensor:
        return self.embedding(labels).unsqueeze(1)


class AdvantageVLAFlowMatching(VLAFlowMatching):
    """VLAFlowMatching with advantage token injection after the prefix."""

    def __init__(self, config: AdvantageConfig, **kwargs):
        super().__init__(config, **kwargs)
        vlm_hidden = self.vlm_with_expert.config.text_config.hidden_size
        self.adv_embedding = AdvantageEmbedding(vlm_hidden)
        self.advantage_dropout = config.advantage_dropout

    def inject_advantage(
        self, prefix_embs, prefix_pad_masks, prefix_att_masks, adv_labels
    ):
        """Inject advantage embedding after prefix with optional dropout."""
        bsize = prefix_embs.shape[0]
        device = prefix_embs.device

        adv_embs = self.adv_embedding(adv_labels).to(dtype=prefix_embs.dtype)

        # Zero out the embedding for dropped samples (classifier-free guidance)
        if self.training and self.advantage_dropout > 0:
            keep = (torch.rand(bsize, device=device) > self.advantage_dropout).float()
            adv_embs = adv_embs * keep[:, None, None]

        adv_pad = torch.ones(bsize, 1, dtype=prefix_pad_masks.dtype, device=device)
        adv_att = torch.ones(bsize, 1, dtype=prefix_att_masks.dtype, device=device)

        prefix_embs = torch.cat([prefix_embs, adv_embs], dim=1)
        prefix_pad_masks = torch.cat([prefix_pad_masks, adv_pad], dim=1)
        prefix_att_masks = torch.cat([prefix_att_masks, adv_att], dim=1)
        return prefix_embs, prefix_pad_masks, prefix_att_masks

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        noise=None,
        time=None,
        adv_labels=None,
    ) -> Tensor:
        """Training forward with advantage token injected after prefix."""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )

        if adv_labels is not None:
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.inject_advantage(
                prefix_embs, prefix_pad_masks, prefix_att_masks, adv_labels
            )

        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=cast(torch.LongTensor, position_ids),
            past_key_values=None,
            inputs_embeds=cast(list[torch.FloatTensor], [prefix_embs, suffix_embs]),
            use_cache=False,
            fill_kv_cache=False,
        )

        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        losses = torch.nn.functional.mse_loss(u_t, v_t, reduction="none")
        return losses

    def sample_actions(
        self, images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs
    ) -> Tensor:
        """Inference with positive advantage token always injected."""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )

        # Always inject positive advantage at inference
        pos_labels = torch.ones(bsize, dtype=torch.long, device=device)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.inject_advantage(
            prefix_embs, prefix_pad_masks, prefix_att_masks, pos_labels
        )

        # KV cache forward
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=cast(torch.LongTensor, prefix_position_ids),
            past_key_values=None,
            inputs_embeds=cast(list[torch.FloatTensor], [prefix_embs, None]),
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        # Denoising loop
        num_steps = self.config.num_steps
        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(
                bsize
            )
            v_t = self.denoise_step(
                x_t=x_t,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                timestep=time_tensor,
            )
            x_t = x_t + dt * v_t

        return x_t


class AdvantagePolicy(SmolVLAPolicy):
    """Advantage-conditioned policy extending SmolVLA."""

    config_class = AdvantageConfig
    name = "advantage"

    def __init__(self, config: AdvantageConfig, **kwargs):
        super(SmolVLAPolicy, self).__init__(config, **kwargs)
        config.validate_features()
        self.config: AdvantageConfig = config
        self.init_rtc_processor()
        self.model: AdvantageVLAFlowMatching = AdvantageVLAFlowMatching(
            config, rtc_processor=self.rtc_processor
        )
        self.reset()

        self.advantage_labels: torch.Tensor | None = None
        if not config.fixed_advantage:
            dataset_meta = kwargs.get("dataset_meta")
            assert (
                dataset_meta is not None
            ), "dataset_meta is required to load advantage labels"
            self.advantage_labels = self.load_advantage_labels(dataset_meta.repo_id)

    @staticmethod
    def load_advantage_labels(repo_id: str) -> torch.Tensor:
        """Load advantage labels from a HuggingFace dataset repo."""
        from datasets import load_dataset

        logging.info(f"Loading advantage labels from {repo_id}...")
        ds = load_dataset(repo_id, split="train")
        labels = torch.tensor(ds["advantage_label"], dtype=torch.long)
        logging.info(f"Loaded {len(labels)} advantage labels")
        return labels

    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"
    ):
        """Training forward: resolve advantage labels then delegate to model."""
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch["observation.language_tokens"]
        lang_masks = batch["observation.language_attention_mask"]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")

        # Resolve advantage labels
        device = actions.device
        if self.config.fixed_advantage:
            adv_labels = torch.ones(actions.shape[0], dtype=torch.long, device=device)
        else:
            assert self.advantage_labels is not None
            indices = batch["index"].long().cpu()
            adv_labels = self.advantage_labels[indices].to(device)

        losses = AdvantageVLAFlowMatching.forward(
            self.model,
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            actions,
            noise,
            time,
            adv_labels=adv_labels,
        )

        assert self.config.action_feature is not None
        original_action_dim = self.config.action_feature.shape[0]
        losses = losses[:, :, :original_action_dim]

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)

        losses = losses[:, :, : self.config.max_action_dim]

        loss_dict = {}
        if not self.config.fixed_advantage:
            loss_dict["pct_positive"] = adv_labels.float().mean().item()

        if reduction == "none":
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            loss = losses.mean()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict
