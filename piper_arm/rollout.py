"""Environment rollout with Mahalanobis trace capture and plotting."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION
from lerobot.utils.utils import inside_slurm
from tqdm import tqdm

from piper_arm.embedding import embed_prefix_pooled
from piper_arm.mahalanobis import compute_mahalanobis_np

MA_WINDOW = 10


def build_frame(
    obs: dict[str, Any], action: torch.Tensor, features: dict[str, dict]
) -> dict[str, Any]:
    """Extract dataset-relevant fields from a preprocessed observation."""
    frame = {"action": action}

    for key in features:
        val = obs.get(key)
        if val is None:
            continue

        if key.startswith("observation.images."):
            # (C, H, W) float [0,1] -> (H, W, C) uint8
            frame[key] = (val.permute(1, 2, 0) * 255).to(torch.uint8)
        else:
            frame[key] = val

    return frame


def rollout(
    policy: PI05Policy | SmolVLAPolicy,
    vec_env: Any,
    preprocessor: Any,
    postprocessor: Any,
    env_preprocessor: Any,
    env_postprocessor: Any,
    gauss_mean: np.ndarray,
    gauss_cov_inv: np.ndarray,
    seed: int,
    desc: str = "",
) -> dict[str, Any]:
    """Run a single episode, capturing full observations, actions, and
    Mahalanobis traces.

    No interventions — the episode runs to completion or truncation.

    Returns:
        Dict with keys: success, trace_distances,
        actions (list of Tensor), observations (list of dicts).
    """
    max_steps = vec_env.call("_max_episode_steps")[0]
    observation, info = vec_env.reset(seed=[seed])
    policy.reset()

    success = False
    done = False
    trace_distances: list[float] = []
    actions: list[torch.Tensor] = []
    observations: list[dict[str, torch.Tensor]] = []

    for _ in tqdm(
        range(max_steps),
        desc=desc,
        leave=False,
        disable=inside_slurm(),
    ):
        if done:
            break

        observation = preprocess_observation(observation)
        observation = add_envs_task(vec_env, observation)
        observation = env_preprocessor(observation)
        raw_obs = deepcopy(observation)
        observation = preprocessor(observation)

        with torch.inference_mode():
            emb = embed_prefix_pooled(policy, observation)
            emb_np = emb.cpu().numpy()
            dist = compute_mahalanobis_np(emb_np, gauss_mean, gauss_cov_inv)
            dist_val = dist[0].item()
            trace_distances.append(dist_val)
            action = policy.select_action(observation)

        # Save frame data
        raw_obs = {
            k: v[0].cpu() if isinstance(v, torch.Tensor) else v
            for k, v in raw_obs.items()
        }
        raw_obs["maha_distance"] = np.array([dist_val], dtype=np.float32)
        observations.append(raw_obs)

        action = postprocessor(action)
        actions.append(action[0].cpu())
        action_transition = {ACTION: action}
        action_transition = env_postprocessor(action_transition)
        action_np = action_transition[ACTION].to("cpu").numpy()

        observation, _, terminated, truncated, info = vec_env.step(action_np)

        if "final_info" in info:
            if info["final_info"]["is_success"].item():
                success = True

        done = bool(terminated | truncated)

    return {
        "success": success,
        "trace_distances": trace_distances,
        "actions": actions,
        "observations": observations,
    }


def plot_traces(results: list[dict], output_dir: Path) -> None:
    """Plot per-episode Mahalanobis distance traces colored by success/failure."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(12, 6))

    for result in results:
        steps = np.arange(len(result["trace_distances"]))
        dists = np.array(result["trace_distances"])
        color = "#2ecc71" if result["success"] else "#e74c3c"

        if len(dists) >= MA_WINDOW:
            kernel = np.ones(MA_WINDOW) / MA_WINDOW
            ma_dists = np.convolve(dists, kernel, mode="valid")
            ma_steps = steps[MA_WINDOW - 1 :]
        else:
            ma_dists = dists
            ma_steps = steps

        ax.plot(ma_steps, ma_dists, color=color, alpha=0.7, linewidth=1.0)

    handles = [
        Line2D([0], [0], color="#2ecc71", label="Success"),
        Line2D([0], [0], color="#e74c3c", label="Failure"),
    ]
    ax.legend(handles=handles, fontsize=8)
    ax.set_xlabel("Timestep")
    ax.set_ylabel(f"Mahalanobis Distance (MA, w={MA_WINDOW})")
    ax.set_title("Mahalanobis Distance of VLM Prefix Embeddings")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = output_dir / "eval_dist.png"
    fig.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")
    plt.close(fig)
