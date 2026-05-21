"""Train a VAE on pooled SigLIP embeddings of a base dataset.

Embeds every (subsampled) frame of ``dataset_repo_id`` with the policy's
SigLIP vision tower, standardises the embeddings, and fits an MLP VAE
(``distal/rewards/vae.py``). The trained VAE is the artifact consumed by the
``vae`` reward type — its negative ELBO scores how out-of-distribution a frame
is relative to this base distribution.

Saved as a single safetensors file (weights + input standardisation) with the
architecture and embedding type in the file metadata, mirrored to the HF Hub.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import draccus
import torch
from huggingface_hub import HfApi
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.processor import rename_stats
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging
from safetensors.torch import save_file

from distal.rewards.knn import embed_dataset
from distal.rewards.vae import SiglipVAE, VaeConfig


@dataclass
class VaeTrainConfig:
    policy_path: str = "lerobot/pi05-libero"
    dataset_repo_id: str = "lerobot/libero_plus"
    hub_repo_id: str = "reece-omahoney/pi05-libero-plus-vae"
    output_path: str = "outputs/vae/vae.safetensors"
    device: str = "cuda"

    # Embedding pass
    embed_batch_size: int = 256
    embed_num_workers: int = 4
    max_frames: int | None = 50_000
    subsample_seed: int = 0
    embedding_type: str = "siglip"  # "siglip" or "post_lm"
    rename_map: dict[str, str] = field(
        default_factory=lambda: {
            "observation.images.front": "observation.images.image",
            "observation.images.wrist": "observation.images.image2",
        }
    )

    # VAE architecture
    latent_dim: int = 64
    hidden_dims: list[int] = field(default_factory=lambda: [512, 256])

    # VAE training
    epochs: int = 500
    train_batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    kl_weight: float = 1.0
    val_split: float = 0.1
    eval_samples: int = 16
    seed: int = 0
    log_every: int = 10

    # Early stopping on val neg-ELBO, counted in validation events (which
    # coincide with log steps). patience=20 with log_every=10 is ~200 epochs
    # without improvement. Set patience to null to disable.
    early_stopping_patience: int | None = 20
    early_stopping_min_delta: float = 1e-3

    push_to_hub: bool = True


def embed_base_dataset(cfg: VaeTrainConfig, device: torch.device) -> torch.Tensor:
    """Run the SigLIP embedding pass over the base dataset."""
    dataset = LeRobotDataset(repo_id=cfg.dataset_repo_id, vcodec="auto")
    print(f"Loaded dataset {cfg.dataset_repo_id} with {len(dataset)} frames")

    policy_cfg = PreTrainedConfig.from_pretrained(cfg.policy_path)
    policy_cfg.pretrained_path = Path(cfg.policy_path)
    policy_cfg.device = str(device)

    policy = make_policy(
        cfg=policy_cfg, ds_meta=dataset.meta, rename_map=cfg.rename_map
    )
    assert isinstance(policy, PI05Policy)
    policy.eval()

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(policy_cfg.pretrained_path),
        dataset_stats=rename_stats(dataset.meta.stats or {}, cfg.rename_map),
        preprocessor_overrides={
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )

    try:
        embeddings = embed_dataset(
            policy=policy,
            preprocessor=preprocessor,
            dataset=dataset,
            batch_size=cfg.embed_batch_size,
            num_workers=cfg.embed_num_workers,
            device=device,
            max_frames=cfg.max_frames,
            subsample_seed=cfg.subsample_seed,
            desc="Embedding base dataset",
            embedding_type=cfg.embedding_type,
        )
    finally:
        del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.from_numpy(embeddings).float()


def train_vae(
    cfg: VaeTrainConfig,
    vae: SiglipVAE,
    train_x: torch.Tensor,
    val_x: torch.Tensor,
) -> None:
    """Fit the VAE in-place by minimising recon_nll + kl_weight * kl.

    Restores the weights from the lowest-val-neg-ELBO validation before
    returning, and stops early once ``early_stopping_patience`` validations
    pass without improvement.
    """
    optimizer = torch.optim.Adam(
        vae.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    device = train_x.device
    can_early_stop = cfg.early_stopping_patience is not None and val_x.shape[0] > 0

    best_val_neg_elbo = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0

    for epoch in range(1, cfg.epochs + 1):
        vae.train()
        perm = torch.randperm(train_x.shape[0], device=device)
        recon_sum = 0.0
        kl_sum = 0.0
        for i in range(0, train_x.shape[0], cfg.train_batch_size):
            batch = train_x[perm[i : i + cfg.train_batch_size]]
            recon, kl = vae.loss_terms(batch)
            loss = recon.mean() + cfg.kl_weight * kl.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            recon_sum += float(recon.sum().item())
            kl_sum += float(kl.sum().item())

        if epoch % cfg.log_every != 0 and epoch != cfg.epochs:
            continue

        train_recon = recon_sum / train_x.shape[0]
        train_kl = kl_sum / train_x.shape[0]
        vae.eval()
        if val_x.shape[0] > 0:
            val_neg_elbo = float(
                vae.neg_elbo(val_x, num_samples=cfg.eval_samples).mean().item()
            )
        else:
            val_neg_elbo = float("nan")
        print(
            f"[epoch {epoch:4d}/{cfg.epochs}] "
            f"train recon_nll={train_recon:.4f} kl={train_kl:.4f} "
            f"neg_elbo={train_recon + train_kl:.4f} | "
            f"val neg_elbo={val_neg_elbo:.4f} "
            f"decoder_logvar={float(vae.decoder_logvar.item()):.4f}"
        )

        if not can_early_stop:
            continue
        if best_val_neg_elbo - val_neg_elbo > cfg.early_stopping_min_delta:
            best_val_neg_elbo = val_neg_elbo
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in vae.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"  no val improvement ({patience_counter}/"
                f"{cfg.early_stopping_patience}); best neg_elbo="
                f"{best_val_neg_elbo:.4f}"
            )
            if patience_counter >= cfg.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    if best_state is not None:
        vae.load_state_dict(best_state)
        print(f"Restored best VAE (val neg_elbo={best_val_neg_elbo:.4f}).")


@draccus.wrap()
def main(cfg: VaeTrainConfig):
    init_logging()
    register_third_party_plugins()

    device = get_safe_torch_device(cfg.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    embeddings = embed_base_dataset(cfg, device)
    print(f"Embedded {embeddings.shape[0]} frames, dim={embeddings.shape[1]}")

    # Standardise embeddings (zero mean, unit variance per dim) for stable training.
    input_mean = embeddings.mean(dim=0)
    input_std = embeddings.std(dim=0).clamp_min(1e-6)
    normalized = ((embeddings - input_mean) / input_std).to(device)

    # Train/val split.
    generator = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(normalized.shape[0], generator=generator)
    n_val = int(normalized.shape[0] * cfg.val_split)
    val_x = normalized[perm[:n_val]]
    train_x = normalized[perm[n_val:]]
    print(f"Train frames: {train_x.shape[0]}, val frames: {val_x.shape[0]}")

    torch.manual_seed(cfg.seed)
    vae_config = VaeConfig(
        input_dim=embeddings.shape[1],
        hidden_dims=list(cfg.hidden_dims),
        latent_dim=cfg.latent_dim,
    )
    vae = SiglipVAE(vae_config).to(device)
    n_params = sum(p.numel() for p in vae.parameters())
    print(f"VAE: {n_params:,} parameters, config={vae_config}")

    train_vae(cfg, vae, train_x, val_x)

    # Save weights + input standardisation, architecture in file metadata.
    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, torch.Tensor] = {
        f"vae.{key}": value.detach().cpu() for key, value in vae.state_dict().items()
    }
    state["input_mean"] = input_mean.cpu()
    state["input_std"] = input_std.cpu()
    save_file(
        state,
        str(output_path),
        metadata={
            "vae_config": json.dumps(asdict(vae_config)),
            "embedding_type": cfg.embedding_type,
        },
    )
    print(f"Saved VAE to {output_path}")

    if cfg.push_to_hub:
        api = HfApi()
        api.create_repo(cfg.hub_repo_id, repo_type="dataset", exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(output_path),
            path_in_repo="vae.safetensors",
            repo_id=cfg.hub_repo_id,
            repo_type="dataset",
        )
        print(f"Pushed VAE to https://huggingface.co/datasets/{cfg.hub_repo_id}")


if __name__ == "__main__":
    main()
