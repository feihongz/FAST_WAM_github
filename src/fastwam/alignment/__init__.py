"""Stage 3 action/video alignment training utilities."""

from .losses import Stage3LossOutput, stage3_alignment_loss
from .rollout import (
    PreparedStage3Batch,
    SolverPanel,
    Stage3VelocityPanel,
    build_solver_panel,
    compute_stage3_velocity_panel,
    prepare_stage3_batch,
    rollout_self_video,
    validate_video_only_joint_equivalence,
)
from .trainer import AlignmentTrainer

__all__ = [
    "AlignmentTrainer",
    "PreparedStage3Batch",
    "SolverPanel",
    "Stage3LossOutput",
    "Stage3VelocityPanel",
    "build_solver_panel",
    "compute_stage3_velocity_panel",
    "prepare_stage3_batch",
    "rollout_self_video",
    "stage3_alignment_loss",
    "validate_video_only_joint_equivalence",
]
