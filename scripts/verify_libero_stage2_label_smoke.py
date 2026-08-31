#!/usr/bin/env python3
"""Strict acceptance check for the one-sample LIBERO Stage 2 label smoke."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from fastwam.alignment.checkpointing import (
    canonical_json_sha256,
    sha256_file,
    write_json_atomic,
)
from fastwam.gating.artifacts import load_complete_label_chunk


DATA_MANIFEST_PATH = Path(
    "/root/feihong/FastWAM/formal_runs/contracts/stage3/"
    "libero_current_273465f_1693e/libero_stage3_data_manifest.json"
)
DATA_MANIFEST_SHA256 = (
    "08da49109a57b55c67f3fa4ac31fbfa44e44dd541a194a5d3420838537d0d320"
)
BASE_SHA256 = (
    "17a5588cc2b8d162219c9daf818614f614ee4a7921933a4a26c5d678111330e9"
)
ADAPTER_SHA256 = (
    "cbc593bc6ce99c0249a65e5c7cef754c9a1d7ea602f81fdae2b8cb158a25858c"
)
STATS_SHA256 = (
    "30f81ad7d5076e97323e3328bce003e01a04cb21327b5bacd21bb72846768638"
)
VAE_SHA256 = (
    "20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36"
)
DATA_CONFIG_SHA256 = (
    "44dc596c6700e02e69ba12823ed899d12d25c6980263f9bf3ac85cb73d53daa4"
)
SPLIT_ASSIGNMENT_SHA256 = (
    "78bd013dcd49dcafb01898e4c1e8ac5d00c26bee81536a1b5ff40aebd2098704"
)
NUM_SHARDS = 1_048_576
SHARD_INDEX = 780_575
SAMPLE_ID = (
    "11a8900dcffbe91f4cd0b56128430af4e45cdb61f76864be00317747db3dcc4c"
)
SAMPLE_IDENTITY = {
    "global_sample_index": 32,
    "dataset_index": 0,
    "episode_id": 0,
    "frame_id": 32,
    "dataset_frame_index": 32,
}


def _load_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable: {path}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_smoke(
    job_dir: Path,
    *,
    expected_git_commit: str,
    data_manifest_path: Path = DATA_MANIFEST_PATH,
) -> dict[str, Any]:
    root = job_dir.expanduser().resolve()
    manifest_path = data_manifest_path.expanduser().resolve()
    contract_path = root / "label_contract.json"
    runtime_path = root / "label_runtime_config.json"
    split_path = root / "episode_split.json"
    chunk_path = (
        root
        / f"shard-{SHARD_INDEX:05d}"
        / "chunk-00000000.json"
    )

    contract = _load_mapping(contract_path, label="label contract")
    runtime = _load_mapping(runtime_path, label="label runtime config")
    split = _load_mapping(split_path, label="episode split")
    manifest = _load_mapping(manifest_path, label="data manifest")

    _require(
        manifest.get("manifest_sha256") == DATA_MANIFEST_SHA256,
        "LIBERO data manifest identity drifted",
    )
    _require(
        split.get("assignment_sha256") == SPLIT_ASSIGNMENT_SHA256,
        "LIBERO episode split assignment drifted",
    )
    _require(split.get("validation_fraction") == 0.1, "split fraction drifted")
    _require(split.get("split_seed") == 42, "split seed drifted")
    _require(
        contract.get("data_manifest_sha256") == DATA_MANIFEST_SHA256,
        "label contract data manifest SHA drifted",
    )
    _require(
        contract.get("base_checkpoint_sha256") == BASE_SHA256,
        "label contract base SHA drifted",
    )
    _require(
        contract.get("adapter_checkpoint_sha256") == ADAPTER_SHA256,
        "label contract Adapter SHA drifted",
    )
    _require(
        contract.get("normalization_stats_sha256") == STATS_SHA256,
        "label contract normalization SHA drifted",
    )
    _require(
        contract.get("vae_sha256") == VAE_SHA256,
        "label contract VAE SHA drifted",
    )
    _require(
        contract.get("data_config_sha256") == DATA_CONFIG_SHA256,
        "label contract data config SHA drifted",
    )
    _require(
        contract.get("git_identity", {}).get("commit") == expected_git_commit,
        "label contract Git commit drifted",
    )
    _require(
        contract.get("git_identity", {}).get("tracked_dirty") is False,
        "label contract recorded a dirty tracked worktree",
    )
    _require(
        contract.get("git_identity", {}).get("untracked_source_files") == [],
        "label contract recorded untracked source files",
    )
    _require(contract.get("num_shards") == NUM_SHARDS, "smoke shard count drifted")
    _require(contract.get("chunk_size") == 1, "smoke chunk size drifted")
    _require(contract.get("base_seed") == 42, "base seed drifted")
    _require(contract.get("num_seed_pairs") == 2, "seed-pair count drifted")
    _require(contract.get("num_inference_steps") == 10, "solver steps drifted")
    _require(contract.get("relative_margin") == 0.05, "label margin drifted")
    _require(
        contract.get("relative_gain_epsilon") == 1.0e-12,
        "relative-gain epsilon drifted",
    )
    _require(contract.get("sigma_shift") is None, "sigma shift drifted")
    _require(contract.get("rand_device") == "cpu", "seed device drifted")
    _require(contract.get("tiled") is False, "tiled setting drifted")

    _require(
        runtime.get("kind") == "stage2_label_runtime_config",
        "runtime config kind drifted",
    )
    _require(runtime.get("mixed_precision") == "bf16", "mixed precision drifted")
    _require(
        canonical_json_sha256(runtime)
        == contract.get("label_runtime_config_sha256"),
        "runtime config SHA disagrees with the label contract",
    )
    _require(
        runtime.get("model", {}).get("_target_")
        == "fastwam.runtime.create_fastwam_unified_aligned",
        "runtime model target drifted",
    )

    chunk = load_complete_label_chunk(
        chunk_path,
        contract=contract,
        data_manifest=manifest,
        episode_split=split,
        planned_sample_ids=[SAMPLE_ID],
    )
    _require(chunk["shard_index"] == SHARD_INDEX, "chunk shard index drifted")
    _require(chunk["chunk_index"] == 0, "chunk index drifted")
    _require(chunk["planned_row_count"] == 1, "planned row count drifted")
    _require(chunk["row_count"] == 1, "row count drifted")

    row = chunk["rows"][0]
    _require(row["sample_id"] == SAMPLE_ID, "smoke sample ID drifted")
    for field, expected in SAMPLE_IDENTITY.items():
        _require(row[field] == expected, f"smoke sample {field} drifted")
    _require(row["shard_index"] == SHARD_INDEX, "row shard index drifted")
    _require(row["num_inference_steps"] == 10, "row solver steps drifted")
    _require(row["num_video_frames"] == 9, "row video-frame count drifted")
    _require(len(row["seeds"]) == 2, "row seed count drifted")
    _require(row["margin"] == 0.05, "row margin drifted")
    _require(math.isfinite(row["e0"]) and row["e0"] >= 0.0, "row E0 is invalid")
    _require(
        math.isfinite(row["e10"]) and row["e10"] >= 0.0,
        "row E10 is invalid",
    )
    _require(
        row["label"] == (row["e10"] < 0.95 * row["e0"]),
        "row label disagrees with E0/E10",
    )
    _require(
        math.isclose(row["sample_weight"], 1.0, rel_tol=0.0, abs_tol=0.0),
        "row sample weight drifted",
    )

    observed_chunks = sorted(root.glob("shard-*/chunk-*.json"))
    _require(
        observed_chunks == [chunk_path],
        f"smoke output contains unexpected label chunks: {observed_chunks}",
    )

    artifact_paths = {
        "episode_split": split_path,
        "label_runtime_config": runtime_path,
        "label_contract": contract_path,
        "label_chunk": chunk_path,
    }
    return {
        "schema_version": 1,
        "kind": "libero_stage2_label_smoke_receipt",
        "status": "pass",
        "formal_merge_allowed": False,
        "reason": "singleton-shard smoke contract differs from formal 64-shard contract",
        "git_commit": expected_git_commit,
        "contract_sha256": contract["contract_sha256"],
        "chunk_sha256": chunk["chunk_sha256"],
        "sample_id": SAMPLE_ID,
        "sample_identity": dict(SAMPLE_IDENTITY),
        "e0": row["e0"],
        "e10": row["e10"],
        "label": row["label"],
        "sample_weight": row["sample_weight"],
        "artifact_sha256": {
            name: sha256_file(path) for name, path in artifact_paths.items()
        },
        "data_manifest_path": str(manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--data-manifest", type=Path, default=DATA_MANIFEST_PATH)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()

    receipt = verify_smoke(
        args.job_dir,
        expected_git_commit=args.expected_git_commit,
        data_manifest_path=args.data_manifest,
    )
    if args.receipt is not None:
        write_json_atomic(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
