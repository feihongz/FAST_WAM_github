#!/usr/bin/env python3
"""Build the immutable selected-data manifest for formal Stage 3 training."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from fastwam.alignment.checkpointing import (
    resolve_base_checkpoint,
    write_json_atomic,
)
from fastwam.alignment.data_identity import build_data_manifest
from fastwam.alignment.runtime import (
    _canonicalize_data_paths,
    _repo_dir,
    _resolved_config,
    _validate_formal_dataset,
    _validate_required_environment,
)
from fastwam.runtime import build_datasets
from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.pytorch_utils import set_global_seed


register_default_resolvers()


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="train_stage3_alignment",
)
def main(config: DictConfig) -> None:
    resolved = _resolved_config(config)
    runtime_config = dict(resolved["runtime"])
    _validate_required_environment(runtime_config)
    resolved["data"] = _canonicalize_data_paths(
        dict(resolved["data"]),
        repo_dir=_repo_dir(runtime_config),
    )

    manifest_path_value = resolved["data_manifest"].get("path")
    if not manifest_path_value:
        raise ValueError(
            "set data_manifest.path to the output JSON for manifest generation"
        )
    manifest_path = Path(manifest_path_value).expanduser().resolve()
    misc.register_work_dir(manifest_path.parent)

    stats_spec = dict(resolved["assets"]["normalization_stats"])
    stats_identity = resolve_base_checkpoint(
        stats_spec["path"],
        expected_sha256=str(stats_spec["expected_sha256"]),
    )
    set_global_seed(int(resolved["training"]["seed"]))
    train_dataset, _ = build_datasets(OmegaConf.create(resolved["data"]))
    cardinality = _validate_formal_dataset(
        train_dataset,
        runtime_config=runtime_config,
    )
    manifest = build_data_manifest(
        train_dataset,
        normalization_stats_path=stats_identity.path,
    )
    write_json_atomic(manifest_path, manifest)
    print(
        "Stage 3 data manifest written:\n"
        f"  path: {manifest_path}\n"
        f"  canonical_sha256: {manifest['manifest_sha256']}\n"
        f"  frames: {cardinality['dataset_length']}\n"
        f"  episodes: {cardinality['dataset_episodes']}"
    )


if __name__ == "__main__":
    main()
