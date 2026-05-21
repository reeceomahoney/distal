"""VAE-likelihood-based per-step rewards for value training.

A small MLP VAE is fit (by ``distal/rewards/train_vae.py``) to the pooled
SigLIP embeddings of the base training distribution. This module loads that
VAE and scores every frame in a value-training dataset by its negative ELBO
(an upper bound on the negative log-likelihood). A high negative ELBO means
the frame is unlikely under the base distribution — out-of-distribution — so
it plays the same role as the Mahalanobis / kNN distance: high value → low
reward after ``normalize_distances_to_rewards``.

The trained VAE is stored as a single safetensors file: weight tensors plus
the input standardisation ``input_mean`` / ``input_std``, with the
architecture and embedding type in the file's metadata.
"""

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from safetensors import safe_open
from safetensors.torch import load_file
from torch.utils.data import Subset

from distal.rewards.knn import embed_dataset

LOG_2PI = math.log(2.0 * math.pi)


@dataclass
class VaeConfig:
    """Architecture of the SigLIP-embedding VAE."""

    input_dim: int = 1152
    hidden_dims: list[int] = field(default_factory=lambda: [512, 256])
    latent_dim: int = 64


def build_mlp(dims: list[int], final_activation: bool) -> nn.Sequential:
    """Linear stack over ``dims``; LayerNorm+SiLU after every layer except
    optionally the last."""
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        is_last = i == len(dims) - 2
        if final_activation or not is_last:
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class SiglipVAE(nn.Module):
    """MLP VAE with a Gaussian decoder (learned scalar output variance)."""

    def __init__(self, config: VaeConfig):
        super().__init__()
        self.config = config
        hidden = list(config.hidden_dims)
        self.encoder = build_mlp([config.input_dim, *hidden], final_activation=True)
        self.fc_mu = nn.Linear(hidden[-1], config.latent_dim)
        self.fc_logvar = nn.Linear(hidden[-1], config.latent_dim)
        self.decoder = build_mlp(
            [config.latent_dim, *reversed(hidden), config.input_dim],
            final_activation=False,
        )
        self.decoder_logvar = nn.Parameter(torch.zeros(()))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h).clamp(-8.0, 8.0)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(
            std.shape, device=std.device, dtype=std.dtype, generator=generator
        )
        return mu + eps * std

    def recon_nll(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """Per-sample Gaussian negative log-likelihood, summed over dims."""
        var = torch.exp(self.decoder_logvar)
        nll = 0.5 * ((x - x_hat) ** 2 / var + self.decoder_logvar + LOG_2PI)
        return nll.sum(dim=-1)

    @staticmethod
    def kl_div(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

    def loss_terms(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-sample (recon_nll, kl) for a single reparameterised draw."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return self.recon_nll(x, x_hat), self.kl_div(mu, logvar)

    @torch.no_grad()
    def neg_elbo(
        self,
        x: torch.Tensor,
        num_samples: int = 1,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Per-sample negative ELBO, averaging the recon term over draws."""
        num_samples = max(1, num_samples)
        mu, logvar = self.encode(x)
        kl = self.kl_div(mu, logvar)
        recon = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for _ in range(num_samples):
            z = self.reparameterize(mu, logvar, generator)
            recon = recon + self.recon_nll(x, self.decode(z))
        return recon / num_samples + kl


def resolve_vae_path(vae_path: str) -> str:
    """Resolve ``vae_path`` as a local file or an HF dataset repo id."""
    local = Path(vae_path)
    if local.is_file():
        return str(local)
    return hf_hub_download(
        repo_id=vae_path, filename="vae.safetensors", repo_type="dataset"
    )


def load_vae(
    vae_path: str, device: torch.device
) -> tuple[SiglipVAE, torch.Tensor, torch.Tensor, str]:
    """Load a trained VAE, returning (vae, input_mean, input_std, embedding_type)."""
    resolved = resolve_vae_path(vae_path)
    tensors = load_file(resolved)
    with safe_open(resolved, framework="pt") as f:
        metadata = f.metadata() or {}

    vae_config = VaeConfig(**json.loads(metadata["vae_config"]))
    embedding_type = metadata.get("embedding_type", "siglip")

    vae = SiglipVAE(vae_config)
    state = {
        key[len("vae.") :]: value
        for key, value in tensors.items()
        if key.startswith("vae.")
    }
    vae.load_state_dict(state)
    vae = vae.to(device).eval()

    input_mean = tensors["input_mean"].to(device=device, dtype=torch.float32)
    input_std = tensors["input_std"].to(device=device, dtype=torch.float32)
    return vae, input_mean, input_std, embedding_type


def compute_vae_distances_for_dataset(
    dataset: LeRobotDataset,
    policy_path: str,
    vae_path: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    *,
    eval_samples: int,
    eval_seed: int,
    chunk_size: int = 8192,
    frame_indices: list[int] | None = None,
) -> np.ndarray:
    """Return raw per-frame negative-ELBO scores for the dataset.

    If ``frame_indices`` is provided, only those frames are embedded (e.g. for
    AUROC over a sampled subset). The full ``dataset`` is still used to build
    the policy/preprocessor (which need ``dataset.meta``).
    """
    vae, input_mean, input_std, embedding_type = load_vae(vae_path, device)
    logging.info(
        f"Loaded VAE from {vae_path} "
        f"(latent_dim={vae.config.latent_dim}, embedding_type={embedding_type})"
    )

    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = Path(policy_path)
    policy_cfg.device = str(device)
    policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta)
    assert isinstance(policy, PI05Policy)
    policy.eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg, pretrained_path=str(policy_cfg.pretrained_path)
    )

    loader_ds: LeRobotDataset | Subset = (
        Subset(dataset, frame_indices) if frame_indices is not None else dataset
    )

    try:
        embeddings = embed_dataset(
            policy=policy,
            preprocessor=preprocessor,
            dataset=loader_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            max_frames=None,
            subsample_seed=0,
            desc="Embedding for VAE reward",
            embedding_type=embedding_type,
        )
    finally:
        del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    x = torch.from_numpy(embeddings).to(device=device, dtype=torch.float32)
    x = (x - input_mean) / input_std

    generator = torch.Generator(device=device).manual_seed(eval_seed)
    distances: list[np.ndarray] = []
    for i in range(0, x.shape[0], chunk_size):
        neg_elbo = vae.neg_elbo(
            x[i : i + chunk_size], num_samples=eval_samples, generator=generator
        )
        distances.append(neg_elbo.double().cpu().numpy())
    return np.concatenate(distances)
