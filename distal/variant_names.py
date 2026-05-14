"""Sidecar mapping ``episode_index -> LIBERO-plus variant_name``.

The LeRobot dataset only stores the rewritten natural-language task string per
episode, which loses the variant identity for language perturbations. This
sidecar lets downstream scripts recover the variant (and from there the base
task via ``distal.collect_libero_plus.base_task_name``) without re-running the
collect-time replay each time.

File layout: ``{dataset.root}/meta/variant_names.json`` with
``{"<episode_index>": "<variant_name>", ...}``. Pushed to the Hub alongside
the rest of the dataset's ``meta/`` directory.
"""

import json
from pathlib import Path

VARIANT_NAMES_REPO_PATH = "meta/variant_names.json"


def variant_names_path(dataset_root: str | Path) -> Path:
    return Path(dataset_root) / VARIANT_NAMES_REPO_PATH


def save_variant_names(dataset_root: str | Path, ep_to_variant: dict[int, str]) -> Path:
    """Write the episode_index -> variant_name mapping to the dataset's meta dir."""
    path = variant_names_path(dataset_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({str(k): v for k, v in sorted(ep_to_variant.items())}, f, indent=2)
    return path


def load_variant_names(dataset) -> dict[int, str]:
    """Load the sidecar for an already-constructed ``LeRobotDataset``.

    Tries the dataset's local ``meta/variant_names.json`` first; on miss,
    fetches the file from the Hub via ``hf_hub_download`` (which caches under
    ``~/.cache/huggingface/hub`` for subsequent calls). Raises if the sidecar
    is absent in both places — use ``try_load_variant_names`` for the
    "missing is OK" path (e.g. base LIBERO datasets without perturbations).
    """
    local = variant_names_path(dataset.root)
    if local.is_file():
        path: str | Path = local
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=str(dataset.repo_id),
            filename=VARIANT_NAMES_REPO_PATH,
            repo_type="dataset",
        )
    with open(path) as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


def try_load_variant_names(dataset) -> dict[int, str] | None:
    """Load the sidecar if present, else return None (no error)."""
    from huggingface_hub.errors import (
        EntryNotFoundError,
        RepositoryNotFoundError,
    )

    try:
        return load_variant_names(dataset)
    except (EntryNotFoundError, RepositoryNotFoundError, FileNotFoundError):
        return None
