"""Content-addressed local cache for precomputed advantage labels.

The cache is keyed by a signature over every input that influences the
computed advantages (dataset/value-network repo ids, scalar hyperparams,
schema version). Files live at
``$HF_ASSETS_CACHE/distal/advantages/<signature>.json``.
"""

import hashlib
import json
import logging
from pathlib import Path

from huggingface_hub.constants import HF_ASSETS_CACHE

ADVANTAGE_CACHE_DIR = Path(HF_ASSETS_CACHE) / "distal" / "advantages"

# Bump when the cache file format changes in a way file hashes can't detect.
CACHE_SCHEMA_VERSION = 2


def compute_signature(
    *,
    dataset_repo_id: str,
    value_network_pretrained_path: str,
    c_fail: float,
    num_value_bins: int,
    reward_mode: str,
    maha_stats_path: str | None = None,
) -> str:
    """Build a deterministic 16-hex-char signature over all cache-affecting inputs."""
    sig_dict = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "dataset_repo_id": dataset_repo_id,
        "value_network_pretrained_path": value_network_pretrained_path,
        "c_fail": c_fail,
        "num_value_bins": num_value_bins,
        "reward_mode": reward_mode,
        "maha_stats_path": maha_stats_path if reward_mode == "maha" else None,
    }
    canonical = json.dumps(sig_dict, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def cache_path(signature: str) -> Path:
    """Local path where a signature-keyed advantage cache lives."""
    return ADVANTAGE_CACHE_DIR / f"{signature}.json"


def save(
    path: str | Path,
    advantage_lookup: dict[int, float],
    episode_lookup: dict[int, int] | None = None,
    metadata: dict | None = None,
) -> None:
    """Save pre-computed advantages to a JSON file for reuse across runs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "advantages": {str(k): v for k, v in advantage_lookup.items()},
        "num_frames": len(advantage_lookup),
        "mean_advantage": sum(advantage_lookup.values())
        / max(1, len(advantage_lookup)),
    }
    if episode_lookup is not None:
        payload["episode_labels"] = {str(k): v for k, v in episode_lookup.items()}
    if metadata:
        payload["metadata"] = metadata
    with open(path, "w") as f:
        json.dump(payload, f)
    logging.info(f"Saved advantage cache ({len(advantage_lookup)} frames) to {path}")


def load(
    path: str | Path,
) -> tuple[dict[int, float], dict[int, int] | None]:
    """Load pre-computed advantages from a JSON cache file."""
    path = Path(path)
    with open(path) as f:
        payload = json.load(f)
    lookup = {int(k): float(v) for k, v in payload["advantages"].items()}
    episode_lookup = None
    if "episode_labels" in payload:
        episode_lookup = {int(k): int(v) for k, v in payload["episode_labels"].items()}
    logging.info(
        f"Loaded advantage cache from {path}: {len(lookup)} frames, "
        f"mean={sum(lookup.values()) / max(1, len(lookup)):.4f}"
    )
    return lookup, episode_lookup
