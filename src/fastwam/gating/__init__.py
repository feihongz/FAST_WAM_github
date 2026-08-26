"""Independent Stage 2 Gate training utilities."""

from .contracts import (
    EpisodeSplitLookup,
    build_episode_split,
    build_episode_split_lookup,
    dataset_id,
    dataset_id_from_lookup,
    derive_pair_seeds,
    sample_id,
    sample_id_from_lookup,
    split_for_identity,
    validate_episode_split,
    validate_sample_identity,
    validate_sample_identity_with_lookup,
)
from .inference import PairedActionRollouts, run_paired_action_rollouts
from .labels import GateLabelStatistics, paired_gate_label_statistics

__all__ = [
    "GateLabelStatistics",
    "EpisodeSplitLookup",
    "PairedActionRollouts",
    "build_episode_split",
    "build_episode_split_lookup",
    "dataset_id",
    "dataset_id_from_lookup",
    "derive_pair_seeds",
    "paired_gate_label_statistics",
    "sample_id",
    "sample_id_from_lookup",
    "run_paired_action_rollouts",
    "split_for_identity",
    "validate_episode_split",
    "validate_sample_identity",
    "validate_sample_identity_with_lookup",
]
