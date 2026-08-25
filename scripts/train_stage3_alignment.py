"""Hydra entrypoint for the independent Stage 3 Adapter training chain."""

import hydra
from omegaconf import DictConfig

from fastwam.alignment.runtime import run_stage3_alignment_training
from fastwam.utils.config_resolvers import register_default_resolvers


register_default_resolvers()


@hydra.main(
    config_path="../configs",
    config_name="train_stage3_alignment",
    version_base="1.3",
)
def main(config: DictConfig) -> None:
    run_stage3_alignment_training(config)


if __name__ == "__main__":
    main()
