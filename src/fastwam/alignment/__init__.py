"""Stage 3 action/video alignment training utilities."""

from .losses import Stage3LossOutput, stage3_alignment_loss
from .trainer import AlignmentTrainer

__all__ = ["AlignmentTrainer", "Stage3LossOutput", "stage3_alignment_loss"]
