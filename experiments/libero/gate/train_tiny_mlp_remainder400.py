"""CLI for the sealed remainder-400 exact-V1 follow-up trainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from experiments.libero.gate.offline_tiny_mlp import sha256_file
from experiments.libero.gate.offline_tiny_mlp_remainder400 import (
    run_remainder400_followup,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit Target5-only remainder-400 models and seal predictions for the "
            "label-free locked original-100 panel. No Validation4 argument exists."
        )
    )
    parser.add_argument("--remainder-target-dir", type=Path, required=True)
    parser.add_argument("--remainder-target-manifest-sha256", required=True)
    parser.add_argument("--remainder-target-targets-sha256", required=True)
    parser.add_argument("--remainder-feature-dir", type=Path, required=True)
    parser.add_argument("--remainder-feature-completion-sha256", required=True)
    parser.add_argument("--original-feature-dir", type=Path, required=True)
    parser.add_argument("--original-feature-completion-sha256", required=True)
    parser.add_argument("--original-fold-source-dir", type=Path, required=True)
    parser.add_argument("--original-fold-source-completion-sha256", required=True)
    parser.add_argument("--protocol-doc-path", type=Path, required=True)
    parser.add_argument("--protocol-doc-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = run_remainder400_followup(
        remainder_target_dir=args.remainder_target_dir,
        remainder_target_manifest_sha256=args.remainder_target_manifest_sha256,
        remainder_target_targets_sha256=args.remainder_target_targets_sha256,
        remainder_feature_dir=args.remainder_feature_dir,
        remainder_feature_completion_sha256=args.remainder_feature_completion_sha256,
        original_feature_dir=args.original_feature_dir,
        original_feature_completion_sha256=args.original_feature_completion_sha256,
        original_fold_source_dir=args.original_fold_source_dir,
        original_fold_source_completion_sha256=(
            args.original_fold_source_completion_sha256
        ),
        protocol_doc_path=args.protocol_doc_path,
        protocol_doc_sha256=args.protocol_doc_sha256,
        output_dir=args.output_dir,
    )
    completion_path = output / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "completion_path": str(completion_path),
                "completion_file_sha256": sha256_file(completion_path),
                "completion_payload_sha256": completion["completion_sha256"],
                "num_predictions": completion["num_predictions"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
