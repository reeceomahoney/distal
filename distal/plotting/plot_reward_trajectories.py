"""Plot kNN-siglip, kNN-post_lm, and Mahalanobis-siglip rewards over LIBERO-10
success vs failure trajectories.

Samples ``num_episodes`` success and ``num_episodes`` failure episodes from
``reece-omahoney/pi05-libero-10`` and computes raw distances
(``compute_distances``) for only those episodes' frames, normalising them to
rewards over the sampled subset, then resamples each episode onto a common
normalised-progress axis and saves a 1×3 figure of normalised rewards
(columns = methods), each subplot comparing mean ± std success vs failure
curves. Saved as both .png and .pdf.

The final per-episode curves are cached as an ``.npz`` next to the output,
content-addressed by config, so re-running with the same config skips all
embedding and just redraws the figure.
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import draccus
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins

from distal.rewards.configs import KnnRewardConfig, MahaRewardConfig, RewardConfig


@dataclass
class PlotRewardTrajectoriesConfig:
    dataset_repo_id: str = "reece-omahoney/pi05-libero-10"
    base_policy: str = "lerobot/pi05-libero"
    demo_dataset_repo_id: str = "lerobot/libero_10"
    maha_stats_path: str = "reece-omahoney/pi05-libero-10-maha-stats-siglip"
    num_episodes: int = 10
    num_progress_points: int = 100
    seed: int = 0
    device: str = "cuda"
    output_path: str = "outputs/reward_trajectories.png"
    cache: bool = True


def resample_to_progress(values: np.ndarray, num_points: int) -> np.ndarray:
    n = len(values)
    if n == 0:
        return np.full(num_points, np.nan)
    if n == 1:
        return np.full(num_points, float(values[0]))
    src = np.linspace(0.0, 1.0, n)
    tgt = np.linspace(0.0, 1.0, num_points)
    return np.interp(tgt, src, values)


def normalize_distances(distances: np.ndarray) -> np.ndarray:
    """Clip to [p1, p99], map to [-1, 0], rescale to mean -1.

    Mirrors ``distal.rewards.maha.normalize_distances_to_rewards`` but operates
    on the sampled-episode subset rather than the full dataset.
    """
    p1 = float(np.percentile(distances, 1))
    p99 = float(np.percentile(distances, 99))
    if p99 <= p1:
        return np.zeros_like(distances)
    clipped = np.clip(distances, p1, p99)
    normalized = -(clipped - p1) / (p99 - p1)
    mean_reward = float(normalized.mean())
    if mean_reward < 0:
        normalized = normalized * (-1.0 / mean_reward)
    return normalized


def select_episodes(
    unique_eps: np.ndarray,
    episode_index: np.ndarray,
    success: np.ndarray,
    want_success: bool,
    n: int,
    rng: np.random.Generator,
) -> list[int]:
    candidates = [
        int(e)
        for e in unique_eps
        if bool(success[episode_index == e][0]) is want_success
    ]
    label = "success" if want_success else "failure"
    if len(candidates) < n:
        raise RuntimeError(
            f"Only {len(candidates)} {label} episodes available, need {n}."
        )
    chosen = rng.choice(np.array(candidates), size=n, replace=False)
    return sorted(int(e) for e in chosen)


def plot_methods(
    method_labels: list[str],
    success_rewards: dict[str, np.ndarray],
    failure_rewards: dict[str, np.ndarray],
    output_path: Path,
    num_progress_points: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(method_labels)
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(5 * n, 4.5),
        constrained_layout=True,
    )
    if n == 1:
        axes = np.array([axes])
    t = np.linspace(0.0, 1.0, num_progress_points)

    for col_idx, label in enumerate(method_labels):
        ax = axes[col_idx]
        for group_label, curves, color in (
            ("success", success_rewards[label], "tab:green"),
            ("failure", failure_rewards[label], "tab:red"),
        ):
            mean = curves.mean(axis=0)
            std = curves.std(axis=0)
            ax.plot(t, mean, linewidth=1.6, color=color, label=group_label)
            ax.fill_between(t, mean - std, mean + std, alpha=0.2, color=color)
        ax.set_title(label, fontsize=16)
        ax.set_xlabel("normalized episode progress", fontsize=14)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left")
    axes[0].set_ylabel("normalized reward", fontsize=14)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"saved -> {output_path}")
    print(f"saved -> {pdf_path}")


CACHE_KINDS = (
    "success_rewards",
    "failure_rewards",
)


def build_sources(cfg: PlotRewardTrajectoriesConfig) -> list[tuple[str, RewardConfig]]:
    return [
        (
            "kNN siglip",
            KnnRewardConfig(
                base_policy=cfg.base_policy,
                demo_dataset_repo_id=cfg.demo_dataset_repo_id,
                embedding_type="siglip",
            ),
        ),
        (
            "kNN post_lm",
            KnnRewardConfig(
                base_policy=cfg.base_policy,
                demo_dataset_repo_id=cfg.demo_dataset_repo_id,
                embedding_type="post_lm",
            ),
        ),
        (
            "Mahalanobis siglip",
            MahaRewardConfig(
                base_policy=cfg.base_policy,
                stats_path=cfg.maha_stats_path,
            ),
        ),
    ]


def cache_path_for(
    cfg: PlotRewardTrajectoriesConfig, sources: list[tuple[str, RewardConfig]]
) -> Path:
    """Content-addressed ``.npz`` path next to the output figure."""
    payload = {
        "dataset_repo_id": cfg.dataset_repo_id,
        "base_policy": cfg.base_policy,
        "demo_dataset_repo_id": cfg.demo_dataset_repo_id,
        "maha_stats_path": cfg.maha_stats_path,
        "num_episodes": cfg.num_episodes,
        "num_progress_points": cfg.num_progress_points,
        "seed": cfg.seed,
        "sources": [(label, repr(reward)) for label, reward in sources],
    }
    sig = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    out = Path(cfg.output_path)
    return out.with_name(f".{out.stem}_cache_{sig}.npz")


def save_plot_cache(
    path: Path, method_labels: list[str], data: dict[str, dict[str, np.ndarray]]
) -> None:
    arrays: dict[str, np.ndarray] = {"__method_labels__": np.array(method_labels)}
    for kind in CACHE_KINDS:
        for label, arr in data[kind].items():
            arrays[f"{kind}|{label}"] = arr
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)  # ty: ignore[invalid-argument-type]
    print(f"cached plot data -> {path}")


def load_plot_cache(path: Path) -> tuple[list[str], dict[str, dict[str, np.ndarray]]]:
    npz = np.load(path, allow_pickle=False)
    method_labels = [str(x) for x in npz["__method_labels__"]]
    data: dict[str, dict[str, np.ndarray]] = {kind: {} for kind in CACHE_KINDS}
    for key in npz.files:
        if key == "__method_labels__":
            continue
        kind, label = key.split("|", 1)
        if kind in data:
            data[kind][label] = npz[key]
    return method_labels, data


def compute_plot_data(
    cfg: PlotRewardTrajectoriesConfig,
    sources: list[tuple[str, RewardConfig]],
    device: torch.device,
) -> dict[str, dict[str, np.ndarray]]:
    dataset = LeRobotDataset(repo_id=cfg.dataset_repo_id, vcodec="auto")
    episode_index = np.array(dataset.hf_dataset["episode_index"])
    success = np.array(dataset.hf_dataset["success"])
    unique_eps = np.unique(episode_index)

    rng = np.random.default_rng(cfg.seed)
    success_eps = select_episodes(
        unique_eps, episode_index, success, True, cfg.num_episodes, rng
    )
    failure_eps = select_episodes(
        unique_eps, episode_index, success, False, cfg.num_episodes, rng
    )
    print(f"success episodes: {success_eps}")
    print(f"failure episodes: {failure_eps}")

    selected_eps = success_eps + failure_eps
    abs_index = np.array(dataset.hf_dataset["index"])
    episode_abs_indices: dict[int, np.ndarray] = {
        ep: abs_index[np.where(episode_index == ep)[0]] for ep in selected_eps
    }

    # Positional row indices of every frame in the sampled episodes; only
    # these frames are embedded (compute_distances supports frame_indices).
    sample_frame_indices: list[int] = [
        int(i) for ep in selected_eps for i in np.where(episode_index == ep)[0]
    ]
    sample_abs_indices: list[int] = [int(abs_index[i]) for i in sample_frame_indices]

    success_rewards: dict[str, np.ndarray] = {}
    failure_rewards: dict[str, np.ndarray] = {}

    def per_ep_curves(eps: list[int], lookup: dict[int, float]) -> np.ndarray:
        rows = []
        for ep in eps:
            values = np.array(
                [lookup[int(a)] for a in episode_abs_indices[ep]], dtype=np.float64
            )
            rows.append(resample_to_progress(values, cfg.num_progress_points))
        return np.stack(rows, axis=0)

    for label, reward in sources:
        print(
            f"\n=== Computing distances for {len(selected_eps)} sampled "
            f"episodes: {label} ==="
        )
        distances = reward.compute_distances(
            dataset=dataset, device=device, frame_indices=sample_frame_indices
        )
        rewards_dict = dict(
            zip(sample_abs_indices, normalize_distances(distances).tolist())
        )

        success_rewards[label] = per_ep_curves(success_eps, rewards_dict)
        failure_rewards[label] = per_ep_curves(failure_eps, rewards_dict)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "success_rewards": success_rewards,
        "failure_rewards": failure_rewards,
    }


@draccus.wrap()
def main(cfg: PlotRewardTrajectoriesConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    sources = build_sources(cfg)
    method_labels = [label for label, _ in sources]
    cache_path = cache_path_for(cfg, sources)

    if cfg.cache and cache_path.is_file():
        print(f"loading cached plot data <- {cache_path}")
        method_labels, data = load_plot_cache(cache_path)
    else:
        register_third_party_plugins()
        device = get_safe_torch_device(cfg.device, log=True)
        data = compute_plot_data(cfg, sources, device)
        if cfg.cache:
            save_plot_cache(cache_path, method_labels, data)

    plot_methods(
        method_labels=method_labels,
        success_rewards=data["success_rewards"],
        failure_rewards=data["failure_rewards"],
        output_path=Path(cfg.output_path),
        num_progress_points=cfg.num_progress_points,
    )


if __name__ == "__main__":
    main()
