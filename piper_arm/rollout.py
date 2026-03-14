"""Environment rollout loop."""

from copy import deepcopy
from typing import Any

import torch
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION
from lerobot.utils.utils import inside_slurm
from tqdm import tqdm


def build_frame(
    obs: dict[str, Any], action: torch.Tensor, features: dict[str, dict]
) -> dict[str, Any]:
    """Extract dataset-relevant fields from a preprocessed observation."""
    frame = {"action": action}

    for key in features:
        if (val := obs.get(key)) is None:
            continue
        # (C, H, W) float [0,1] -> (H, W, C) uint8
        frame[key] = (
            (val.permute(1, 2, 0) * 255).to(torch.uint8)
            if key.startswith("observation.images.")
            else val
        )

    return frame


def rollout(
    policy: PI05Policy | SmolVLAPolicy,
    vec_env: Any,
    preprocessor: Any,
    postprocessor: Any,
    env_preprocessor: Any,
    env_postprocessor: Any,
    seeds: list[int],
    desc: str = "",
) -> list[dict[str, Any]]:
    """Run episodes for all envs in parallel, capturing observations and actions.

    Each env runs independently; collection stops per-env when it terminates.

    Args:
        seeds: One seed per env. len(seeds) must equal the number of envs in vec_env.

    Returns:
        List of dicts (one per env) with keys: success, actions (list of Tensor),
        observations (list of dicts).
    """
    n_envs = len(seeds)
    max_steps = vec_env.call("_max_episode_steps")[0]
    observation, _ = vec_env.reset(seed=seeds)
    policy.reset()

    active = [True] * n_envs
    results: list[dict[str, Any]] = [
        {"success": False, "actions": [], "observations": []} for _ in range(n_envs)
    ]

    for _ in tqdm(range(max_steps), desc=desc, leave=False, disable=inside_slurm()):
        if not any(active):
            break

        observation = preprocess_observation(observation)
        observation = add_envs_task(vec_env, observation)
        observation = env_preprocessor(observation)
        raw_obs = deepcopy(observation)
        observation = preprocessor(observation)
        observation = {
            k: v.to(policy.config.device) if isinstance(v, torch.Tensor) else v
            for k, v in observation.items()
        }

        with torch.inference_mode():
            action = policy.select_action(observation)

        for i in range(n_envs):
            if not active[i]:
                continue
            obs_i = {
                k: v[i].cpu() if isinstance(v, torch.Tensor) else v
                for k, v in raw_obs.items()
            }
            results[i]["observations"].append(obs_i)

        action = postprocessor(action)
        for i in range(n_envs):
            if active[i]:
                results[i]["actions"].append(action[i].cpu())

        action_transition = {ACTION: action}
        action_transition = env_postprocessor(action_transition)
        action_np = action_transition[ACTION].to("cpu").numpy()

        observation, _, terminated, truncated, info = vec_env.step(action_np)

        if "final_info" in info:
            is_success = info["final_info"].get("is_success")
            if is_success is not None:
                for i in range(n_envs):
                    if active[i] and (bool(terminated[i]) or bool(truncated[i])):
                        val = (
                            is_success[i]
                            if hasattr(is_success, "__len__")
                            else is_success
                        )
                        if hasattr(val, "item"):
                            val = val.item()
                        if val:
                            results[i]["success"] = True

        for i in range(n_envs):
            if active[i] and (bool(terminated[i]) or bool(truncated[i])):
                active[i] = False

    return results
