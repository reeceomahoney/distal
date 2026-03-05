"""Advantage policy pre/post processors.

Delegates to SmolVLA's processors since the advantage policy uses the same
observation/action space. Wraps the to_transition converter to preserve extra
dataset columns (advantage_label, etc.) through the preprocessing pipeline.
"""

from typing import Any

import torch
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.processor.converters import batch_to_transition
from lerobot.processor.pipeline import EnvTransition, TransitionKey

from .configuration_advantage import AdvantageConfig

ADVANTAGE_KEYS = ("advantage_label", "steps_remaining", "success", "maha_distance")


def batch_to_transition_with_extras(batch: dict[str, Any]) -> EnvTransition:
    """Wrap batch_to_transition to preserve extra dataset columns.

    The default converter only extracts known keys (task, index, etc.) into
    complementary_data. This version also preserves advantage-related keys.
    """
    transition = batch_to_transition(batch)
    comp = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
    for key in ADVANTAGE_KEYS:
        if key in batch and key not in comp:
            comp[key] = batch[key]
    transition[TransitionKey.COMPLEMENTARY_DATA] = comp
    return transition


def make_advantage_pre_post_processors(
    config: AdvantageConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build pre/post processors by delegating to SmolVLA's processor factory.

    Replaces the default to_transition converter to preserve advantage_label
    and other extra columns through the pipeline.
    """
    smolvla_config = config.to_smolvla_config()
    preprocessor, postprocessor = make_smolvla_pre_post_processors(
        smolvla_config, dataset_stats
    )

    # Replace the to_transition converter to preserve extra keys
    preprocessor.to_transition = batch_to_transition_with_extras

    return preprocessor, postprocessor
