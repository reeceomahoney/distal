"""Pre-compute binary advantage labels and add them to a LeRobot dataset.

Loads a trained value model, computes per-task advantage thresholds, then
binarizes per-sample advantages. Adds an `advantage_label` column to the
dataset's underlying HuggingFace dataset and saves/pushes the result.

Usage:
    python -m piper_arm.compute_advantage_labels --config_path configs/advantage.yaml
"""

from dataclasses import dataclass

import draccus
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader

from piper_arm.train_advantage import (
    _prepare_value_inputs,
    compute_advantage_thresholds,
    load_value_model,
)
from piper_arm.train_value import compute_returns
from piper_arm.value_model import ValueModel


@dataclass
class ComputeAdvantageLabelsConfig:
    value_checkpoint: str = "outputs/value/checkpoint_final.pt"
    dataset_repo_id: str = "reece-omahoney/libero-10-maha"
    dataset_root: str | None = None
    c_fail: float = 1000.0
    advantage_percentile: float = 0.3
    batch_size: int = 64
    num_workers: int = 4
    push_to_hub: bool = False
    output_repo_id: str | None = None


def compute_all_labels(
    dataset: LeRobotDataset,
    value_model: ValueModel,
    thresholds: dict[str, float],
    max_episode_length: int,
    c_fail: float,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
) -> list[int]:
    """Compute binary advantage labels for every sample in the dataset.

    Returns:
        List of integer labels (0 or 1), one per dataset frame.
    """
    tokenizer = value_model.vlm_with_expert.processor.tokenizer

    loader = DataLoader(
        dataset,  # type: ignore[arg-type]
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    all_labels: list[int] = []

    for batch in loader:
        images, img_masks, lang_tokens, lang_masks, state = _prepare_value_inputs(
            batch, value_model, tokenizer, device
        )

        returns = compute_returns(
            batch["steps_remaining"].to(device),
            batch["success"].to(device),
            max_episode_length,
            c_fail,
        )

        with torch.no_grad():
            logits = value_model(images, img_masks, lang_tokens, lang_masks, state)
            values = value_model.predict_value(logits)

        advantages = (returns - values).cpu()
        task_texts = batch["task"]

        for i, task in enumerate(task_texts):
            threshold = thresholds.get(task, 0.0)
            label = 1 if advantages[i].item() > threshold else 0
            all_labels.append(label)

    return all_labels


@draccus.wrap()  # type: ignore[misc]
def main(cfg: ComputeAdvantageLabelsConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    ds_kwargs: dict = {"repo_id": cfg.dataset_repo_id}
    if cfg.dataset_root:
        ds_kwargs["root"] = cfg.dataset_root
    dataset = LeRobotDataset(**ds_kwargs)

    all_steps = dataset.hf_dataset["steps_remaining"]
    max_episode_length = max(s.item() for s in all_steps) + 1
    print(f"Dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames")
    print(f"Max episode length: {max_episode_length}")

    # Load value model (return type annotation is wrong — it returns just the model)
    print("Loading value model...")
    value_model: ValueModel = load_value_model(cfg.value_checkpoint, device)  # type: ignore[assignment]

    # Compute per-task thresholds
    print("Computing per-task advantage thresholds...")
    thresholds = compute_advantage_thresholds(dataset, value_model, cfg.c_fail, device)

    # Compute labels for all samples
    print("Computing advantage labels for all samples...")
    labels = compute_all_labels(
        dataset,
        value_model,
        thresholds,
        max_episode_length,
        cfg.c_fail,
        device,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    assert (
        len(labels) == dataset.num_frames
    ), f"Expected {dataset.num_frames} labels, got {len(labels)}"

    pct_positive = sum(labels) / len(labels) * 100
    print(
        f"Labels computed: {pct_positive:.1f}% positive ({sum(labels)}/{len(labels)})"
    )

    # Add column to dataset
    print("Adding advantage_label column to dataset...")
    dataset.hf_dataset = dataset.hf_dataset.add_column("advantage_label", labels)

    # Save locally
    save_path = dataset.root
    print(f"Saving dataset to {save_path}...")
    dataset.hf_dataset.save_to_disk(str(save_path / "train"))

    # Push to hub
    if cfg.push_to_hub:
        repo_id = cfg.output_repo_id or cfg.dataset_repo_id
        print(f"Pushing dataset to {repo_id}...")
        dataset.push_to_hub(repo_id=repo_id)

    print("Done.")


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
