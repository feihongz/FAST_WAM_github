"""Summarize a LIBERO Demo Utility Collector JSONL file.

The analysis intentionally stops at label diagnostics.  It neither constructs
features nor trains a Gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


LOGGER = logging.getLogger(__name__)
ANALYSIS_SCHEMA_VERSION = 1
QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_utility_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            sample_id = str(record.get("sample_id", ""))
            if not sample_id:
                raise ValueError(f"Missing sample_id at {path}:{line_number}")
            if sample_id in sample_ids:
                raise ValueError(f"Duplicate sample_id={sample_id!r} in {path}")
            sample_ids.add(sample_id)

            if "collector_record_schema_version" in record:
                from experiments.libero.gate.collect_demo_utility import (
                    _validate_completed_record,
                )

                try:
                    full_steps = int(record["num_inference_steps"])
                    _validate_completed_record(record, full_steps=full_steps)
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid production collector record at {path}:{line_number}: {exc}"
                    ) from exc

            if "utility" not in record:
                raise ValueError(f"Missing utility at {path}:{line_number}")
            utility = float(record["utility"])
            if not math.isfinite(utility):
                raise ValueError(f"Non-finite utility at {path}:{line_number}")
            if "e0" in record and "efull" in record:
                expected = float(record["e0"]) - float(record["efull"])
                if not math.isclose(utility, expected, rel_tol=1e-6, abs_tol=1e-8):
                    raise ValueError(
                        f"Utility != E0-Efull at {path}:{line_number}: {utility} vs {expected}"
                    )
            records.append(record)
    if not records:
        raise ValueError(f"No utility records found in {path}")
    return records


def validate_manifest_coverage(
    records: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    allow_incomplete: bool,
) -> dict[str, Any]:
    """Validate that records are exactly the manifest plan, or an allowed subset.

    Even in ``allow_incomplete`` mode, records outside the immutable plan,
    duplicate source indices, and missing/mismatched fingerprints are rejected.
    """

    if not isinstance(manifest, Mapping):
        raise ValueError("manifest.json must contain a JSON object")
    expected_fingerprint = manifest.get("compatibility_fingerprint")
    if not isinstance(expected_fingerprint, str) or not expected_fingerprint:
        raise ValueError("manifest.json is missing a non-empty compatibility_fingerprint")

    selection = manifest.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("manifest.json is missing mapping-valued selection")
    raw_plan = selection.get("ordered_selected_source_indices")
    if not isinstance(raw_plan, list):
        raise ValueError(
            "manifest selection is missing ordered_selected_source_indices list"
        )

    plan: list[int] = []
    for position, value in enumerate(raw_plan):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                "manifest selected source indices must be non-negative integers; "
                f"position={position}, value={value!r}"
            )
        plan.append(value)
    if len(plan) != len(set(plan)):
        raise ValueError("manifest selection contains duplicate source indices")
    declared_count = selection.get("num_samples")
    if (
        isinstance(declared_count, bool)
        or not isinstance(declared_count, int)
        or declared_count != len(plan)
    ):
        raise ValueError(
            "manifest selection num_samples does not match its ordered plan: "
            f"declared={declared_count!r}, actual={len(plan)}"
        )
    declared_plan_sha256 = selection.get("ordered_selected_source_indices_sha256")
    actual_plan_sha256 = _sha256_json(plan)
    if declared_plan_sha256 is not None and declared_plan_sha256 != actual_plan_sha256:
        raise ValueError(
            "manifest ordered-selected-indices SHA-256 mismatch: "
            f"declared={declared_plan_sha256!r}, actual={actual_plan_sha256!r}"
        )

    completed: list[int] = []
    for record_number, record in enumerate(records, start=1):
        actual_fingerprint = record.get("manifest_compatibility_fingerprint")
        if actual_fingerprint != expected_fingerprint:
            raise ValueError(
                "Record/manifest compatibility fingerprint mismatch at record "
                f"{record_number}: record={actual_fingerprint!r}, "
                f"manifest={expected_fingerprint!r}"
            )
        metadata = record.get("source_metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(
                f"Record {record_number} is missing mapping-valued source_metadata"
            )
        source_index = metadata.get("requested_sample_idx")
        if isinstance(source_index, bool) or not isinstance(source_index, int) or source_index < 0:
            raise ValueError(
                f"Record {record_number} has invalid "
                f"source_metadata.requested_sample_idx={source_index!r}"
            )
        completed.append(source_index)

    if len(completed) != len(set(completed)):
        raise ValueError("records contain duplicate requested source indices")
    plan_set = set(plan)
    completed_set = set(completed)
    out_of_plan = sorted(completed_set - plan_set)
    if out_of_plan:
        raise ValueError(
            "records contain source indices outside the immutable manifest plan: "
            f"{out_of_plan[:20]}"
        )
    missing = [source_index for source_index in plan if source_index not in completed_set]
    is_complete = not missing and len(completed) == len(plan)
    if not is_complete and not allow_incomplete:
        raise ValueError(
            "Collection is incomplete: "
            f"completed={len(completed)}/{len(plan)}, missing={len(missing)}. "
            "Finish/resume collection first, or pass --allow-incomplete for an explicit "
            "plan-subset diagnostic."
        )

    return {
        "verified_against_manifest": True,
        "allow_incomplete": bool(allow_incomplete),
        "is_complete": is_complete,
        "status": "complete" if is_complete else "incomplete_allowed",
        "expected_count": len(plan),
        "completed_count": len(completed),
        "missing_count": len(missing),
        "coverage_fraction": (float(len(completed) / len(plan)) if plan else 1.0),
        "missing_source_index_examples": missing[:20],
        "selection_sha256": actual_plan_sha256,
    }


def utility_statistics(values: Sequence[float], *, near_zero_epsilon: float) -> dict[str, Any]:
    if near_zero_epsilon < 0:
        raise ValueError("near_zero_epsilon must be non-negative")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("utility_statistics expects a non-empty finite 1D sequence")

    positive = array > near_zero_epsilon
    negative = array < -near_zero_epsilon
    nearzero = np.abs(array) <= near_zero_epsilon
    strict_positive = array > 0.0
    strict_negative = array < 0.0
    count = int(array.size)
    result: dict[str, Any] = {
        "count": count,
        "positive_count": int(positive.sum()),
        "negative_count": int(negative.sum()),
        "nearzero_count": int(nearzero.sum()),
        "positive_fraction": float(positive.mean()),
        "negative_fraction": float(negative.mean()),
        "nearzero_fraction": float(nearzero.mean()),
        "strict_positive_count": int(strict_positive.sum()),
        "strict_negative_count": int(strict_negative.sum()),
        "strict_positive_fraction": float(strict_positive.mean()),
        "strict_negative_fraction": float(strict_negative.mean()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "mean_abs": float(np.abs(array).mean()),
        "min": float(array.min()),
        "max": float(array.max()),
    }
    quantile_values = np.quantile(array, QUANTILES)
    for quantile, value in zip(QUANTILES, quantile_values, strict=True):
        result[f"q{int(round(quantile * 100)):02d}"] = float(value)
    if result["positive_count"] + result["negative_count"] + result["nearzero_count"] != count:
        raise AssertionError("Positive/negative/near-zero categories do not partition records")
    return result


def _record_suite(record: Mapping[str, Any]) -> str:
    suite = record.get("suite")
    if suite is None and isinstance(record.get("source_metadata"), dict):
        dataset_name = str(record["source_metadata"].get("dataset_name", "unknown"))
        for candidate in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
            if candidate in dataset_name:
                return candidate
    return str(suite if suite is not None else "unknown")


def _record_task(record: Mapping[str, Any]) -> tuple[str, str, str]:
    suite = _record_suite(record)
    task_index = record.get("task_index")
    task = record.get("task")
    metadata = record.get("source_metadata")
    if isinstance(metadata, dict):
        if task_index is None:
            task_index = metadata.get("task_index")
        if task is None:
            task = metadata.get("task")
    return suite, str(task_index if task_index is not None else "unknown"), str(task or "unknown")


def grouped_statistics(
    records: Sequence[Mapping[str, Any]],
    *,
    group: str,
    near_zero_epsilon: float,
) -> list[dict[str, Any]]:
    buckets: dict[Any, list[float]] = {}
    if group == "suite":
        for record in records:
            buckets.setdefault(_record_suite(record), []).append(float(record["utility"]))
        return [
            {"suite": key, **utility_statistics(values, near_zero_epsilon=near_zero_epsilon)}
            for key, values in sorted(buckets.items(), key=lambda item: str(item[0]))
        ]
    if group == "task":
        for record in records:
            buckets.setdefault(_record_task(record), []).append(float(record["utility"]))
        return [
            {
                "suite": key[0],
                "task_index": key[1],
                "task": key[2],
                **utility_statistics(values, near_zero_epsilon=near_zero_epsilon),
            }
            for key, values in sorted(
                buckets.items(), key=lambda item: tuple(str(part) for part in item[0])
            )
        ]
    raise ValueError(f"Unknown group={group!r}")


def histogram_rows(values: Sequence[float], *, bins: int) -> list[dict[str, Any]]:
    if bins <= 0:
        raise ValueError("bins must be positive")
    array = np.asarray(values, dtype=np.float64)
    low = float(array.min())
    high = float(array.max())
    if low == high:
        half_width = max(abs(low) * 0.01, 1e-12)
        low -= half_width
        high += half_width
    counts, edges = np.histogram(array, bins=bins, range=(low, high))
    return [
        {
            "bin_index": index,
            "left": float(edges[index]),
            "right": float(edges[index + 1]),
            "count": int(counts[index]),
            "fraction": float(counts[index] / array.size),
        }
        for index in range(len(counts))
    ]


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def _maybe_plot_histogram(
    values: Sequence[float], *, bins: int, epsilon: float, output_path: Path
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("matplotlib is not installed; skipping optional histogram PNG")
        return False

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(np.asarray(values, dtype=np.float64), bins=bins, color="#3568a8", alpha=0.9)
    axis.axvline(0.0, color="black", linewidth=1.0, label="U=0")
    if epsilon > 0:
        axis.axvspan(-epsilon, epsilon, color="#efb949", alpha=0.2, label="near zero")
    axis.set_xlabel("Demo video utility U = E0 - Efull")
    axis.set_ylabel("Number of states")
    axis.set_title("LIBERO demonstration video utility")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return True


def analyze(
    records_path: Path,
    output_dir: Path,
    *,
    near_zero_epsilon: float,
    bins: int,
    make_plot: bool,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    records_path = records_path.resolve()
    if not records_path.is_file():
        raise FileNotFoundError(records_path)
    records = load_utility_records(records_path)
    values = [float(record["utility"]) for record in records]
    overall = utility_statistics(values, near_zero_epsilon=near_zero_epsilon)
    by_suite = grouped_statistics(
        records, group="suite", near_zero_epsilon=near_zero_epsilon
    )
    by_task = grouped_statistics(records, group="task", near_zero_epsilon=near_zero_epsilon)
    histogram = histogram_rows(values, bins=bins)

    manifest_path = records_path.parent / "manifest.json"
    manifest_info: dict[str, Any] | None = None
    completeness: dict[str, Any] = {
        "verified_against_manifest": False,
        "allow_incomplete": bool(allow_incomplete),
        "is_complete": None,
        "status": "unverified_no_manifest",
        "expected_count": None,
        "completed_count": len(records),
        "missing_count": None,
        "coverage_fraction": None,
        "missing_source_index_examples": [],
        "selection_sha256": None,
    }
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed manifest JSON: {manifest_path}") from exc
        production_records = any(
            "collector_record_schema_version" in record for record in records
        )
        if production_records:
            from experiments.libero.gate.collect_demo_utility import (
                _validate_manifest_integrity,
            )

            try:
                _validate_manifest_integrity(manifest)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid production collector manifest {manifest_path}: {exc}"
                ) from exc
        completeness = validate_manifest_coverage(
            records,
            manifest,
            allow_incomplete=allow_incomplete,
        )
        manifest_info = {
            "path": str(manifest_path),
            "sha256": _sha256_file(manifest_path),
            "compatibility_fingerprint": manifest.get("compatibility_fingerprint"),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "overall.csv", [overall])
    _write_csv(output_dir / "by_suite.csv", by_suite)
    _write_csv(output_dir / "by_task.csv", by_task)
    _write_csv(output_dir / "histogram.csv", histogram)
    plot_written = False
    if make_plot:
        plot_written = _maybe_plot_histogram(
            values,
            bins=bins,
            epsilon=near_zero_epsilon,
            output_path=output_dir / "utility_histogram.png",
        )

    report = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "input": {
            "records_path": str(records_path),
            "records_sha256": _sha256_file(records_path),
            "manifest": manifest_info,
        },
        "settings": {
            "near_zero_epsilon": float(near_zero_epsilon),
            "histogram_bins": int(bins),
            "allow_incomplete": bool(allow_incomplete),
        },
        "completeness": completeness,
        "overall": overall,
        "by_suite": by_suite,
        "by_task": by_task,
        "histogram": histogram,
        "plot_written": plot_written,
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True, help="Collector records.jsonl")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <records parent>/analysis",
    )
    parser.add_argument("--near-zero-epsilon", type=float, default=1e-4)
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "Allow analysis of a strict subset of manifest-selected source indices; "
            "out-of-plan records and fingerprint mismatches remain errors"
        ),
    )
    parser.add_argument("--no-plot", action="store_true", help="Do not attempt optional PNG")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = _parse_args()
    output_dir = args.output_dir or (args.records.resolve().parent / "analysis")
    report = analyze(
        args.records,
        output_dir,
        near_zero_epsilon=float(args.near_zero_epsilon),
        bins=int(args.bins),
        make_plot=not bool(args.no_plot),
        allow_incomplete=bool(args.allow_incomplete),
    )
    LOGGER.info(
        "Analyzed %d records into %s (mean U=%.8g, mean |U|=%.8g)",
        report["overall"]["count"],
        output_dir,
        report["overall"]["mean"],
        report["overall"]["mean_abs"],
    )


if __name__ == "__main__":
    main()
