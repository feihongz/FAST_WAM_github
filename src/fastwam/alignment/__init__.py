"""Stage 3 action/video alignment training utilities."""

from .data_identity import build_data_manifest, validate_data_manifest
from .losses import Stage3LossOutput, stage3_alignment_loss
from .rollout import (
    PreparedStage3Batch,
    SolverPanel,
    Stage3FrozenPanel,
    Stage3VelocityPanel,
    build_solver_panel,
    complete_stage3_velocity_panel,
    compute_stage3_frozen_panel,
    compute_stage3_velocity_panel,
    prepare_stage3_batch,
    rollout_self_video,
    validate_video_only_joint_equivalence,
)
from .trainer import AlignmentTrainer, AlignmentVelocityModule
from .formal_trainer import Stage3AlignmentTrainer

__all__ = [
    "AlignmentTrainer",
    "AlignmentVelocityModule",
    "PreparedStage3Batch",
    "SolverPanel",
    "Stage3FrozenPanel",
    "Stage3LossOutput",
    "Stage3AlignmentTrainer",
    "Stage3VelocityPanel",
    "build_solver_panel",
    "build_data_manifest",
    "complete_stage3_velocity_panel",
    "compute_stage3_frozen_panel",
    "compute_stage3_velocity_panel",
    "prepare_stage3_batch",
    "rollout_self_video",
    "stage3_alignment_loss",
    "validate_video_only_joint_equivalence",
    "validate_data_manifest",
]
