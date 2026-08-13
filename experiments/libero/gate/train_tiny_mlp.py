"""CLI for the preregistered Target5-only offline Tiny-MLP experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from experiments.libero.gate.offline_tiny_mlp import (
    run_offline_feasibility,
    sha256_file,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train and seal task/suite-held-out Gate OOF predictions. "
            "This command intentionally has no Validation4 argument."
        )
    )
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--target-manifest-sha256", required=True)
    parser.add_argument("--target-targets-sha256", required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--feature-completion-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = run_offline_feasibility(
        target_dir=args.target_dir,
        target_manifest_sha256=args.target_manifest_sha256,
        target_targets_sha256=args.target_targets_sha256,
        feature_dir=args.feature_dir,
        feature_completion_sha256=args.feature_completion_sha256,
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
