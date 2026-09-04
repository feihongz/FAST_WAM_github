from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest
from hydra import compose, initialize_config_dir

from experiments.libero.summarize_results import (
    aggregate_routing_results,
    summarize_results,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts/jihe/eval_libero_gate_closed_loop_smoke_1xh100.sh"


def _query(route, steps, *, gate, policy, total, timing_included=True):
    return {
        "selected_mode": route,
        "actual_video_steps": steps,
        "gate_latency_s": gate,
        "policy_latency_s": policy,
        "total_latency_s": total,
        "timing_included": timing_included,
    }


def _task_result(*, task_id, successes, episodes):
    for episode in episodes:
        episode["total_actual_video_steps"] = sum(
            query["actual_video_steps"] for query in episode["queries"]
        )
    return {
        "task_suite": "libero_spatial",
        "task_id": task_id,
        "successes": successes,
        "total_episodes": len(episodes),
        "duration": 1.0,
        "task_description": f"task {task_id}",
        "routing": {"episodes": episodes},
    }


def test_sim_libero_declares_gate_fields_and_preserves_static_default():
    with initialize_config_dir(
        config_dir=str(REPO_ROOT / "configs"),
        version_base="1.3",
    ):
        default_cfg = compose(config_name="sim_libero")
        gate_cfg = compose(
            config_name="sim_libero",
            overrides=[
                "EVALUATION.routing_mode=gate",
                "EVALUATION.use_manifest_text_cache=true",
                "EVALUATION.gate_checkpoint=/tmp/gate.pt",
                f"EVALUATION.gate_checkpoint_sha256={'a' * 64}",
                "EVALUATION.gate_threshold=0.25",
                f"EVALUATION.gate_expected_label_manifest_sha256={'b' * 64}",
                f"EVALUATION.gate_expected_episode_split_assignment_sha256={'c' * 64}",
                f"EVALUATION.gate_expected_training_config_sha256={'d' * 64}",
                f"EVALUATION.gate_expected_git_commit={'e' * 40}",
                "EVALUATION.gate_expected_git_tracked_dirty=false",
                "EVALUATION.gate_expected_git_untracked_source_files=[]",
                "EVALUATION.random_video_probability=0.4",
                "EVALUATION.random_seed=7",
                "EVALUATION.timing_enabled=true",
                "EVALUATION.timing_warmup_queries=3",
                "EVALUATION.save_query_metrics=true",
            ],
        )

    assert default_cfg.EVALUATION.routing_mode == "static"
    assert default_cfg.EVALUATION.inference_mode == "wo"
    assert default_cfg.EVALUATION.use_manifest_text_cache is False
    assert default_cfg.EVALUATION.timing_enabled is True
    assert default_cfg.EVALUATION.save_query_metrics is True
    assert gate_cfg.EVALUATION.routing_mode == "gate"
    assert gate_cfg.EVALUATION.gate_threshold == pytest.approx(0.25)


def test_routing_summary_uses_all_queries_for_compute_but_skips_timing_warmup():
    results = [
        _task_result(
            task_id=0,
            successes=1,
            episodes=[
                {
                    "episode_index": 0,
                    "queries": [
                        _query(
                            "wo",
                            0,
                            gate=0.01,
                            policy=0.10,
                            total=0.11,
                            timing_included=False,
                        )
                    ],
                }
            ],
        ),
        _task_result(
            task_id=1,
            successes=0,
            episodes=[
                {
                    "episode_index": 0,
                    "queries": [_query("w", 10, gate=0.02, policy=0.20, total=0.22)],
                },
                {
                    "episode_index": 1,
                    "queries": [_query("w", 10, gate=0.04, policy=0.40, total=0.44)],
                },
            ],
        ),
    ]

    summary = aggregate_routing_results(results)

    assert summary["source"] == "query_records"
    assert summary["counts"] == {"total": 3, "wo": 1, "w": 2}
    assert summary["with_rate"] == pytest.approx(2 / 3)
    assert summary["actual_total_video_nfe"] == 20
    assert summary["avg_video_nfe_per_query"] == pytest.approx(20 / 3)
    assert summary["per_episode_video_nfe"]["mean"] == pytest.approx(20 / 3)
    assert summary["latency_s"]["total"] == {
        "count": 2,
        "mean": pytest.approx(0.33),
        "p50": pytest.approx(0.33),
        "p95": pytest.approx(0.429),
    }
    assert summary["latency_percentiles_exact"] is True


def test_routing_summary_fallback_weights_task_means_by_query_count():
    results = [
        {
            "routing": {
                "episodes": [{"episode_index": 0, "total_actual_video_steps": 0}],
                "summary": {
                    "counts": {"total": 1, "wo": 1, "w": 0},
                    "effective_video_steps": {"total": 0, "mean": 0.0},
                    "latency_s": {
                        "policy": {"mean": 1.0},
                        "total": {"mean": 1.1},
                    },
                }
            }
        },
        {
            "routing": {
                "episodes": [{"episode_index": 0, "total_actual_video_steps": 90}],
                "summary": {
                    "counts": {"total": 9, "wo": 0, "w": 9},
                    "effective_video_steps": {"total": 90, "mean": 10.0},
                    "latency_s": {
                        "policy": {"mean": 3.0},
                        "total": {"mean": 3.1},
                    },
                }
            }
        },
    ]

    summary = aggregate_routing_results(results)

    assert summary["source"] == "task_summaries_fallback"
    assert summary["with_rate"] == pytest.approx(0.9)
    assert summary["avg_video_nfe_per_query"] == pytest.approx(9.0)
    assert summary["latency_s"]["policy"]["mean"] == pytest.approx(2.8)
    assert summary["latency_s"]["policy"]["p50"] is None
    assert summary["total_video_nfe_per_episode"]["count"] == 2
    assert summary["total_video_nfe_per_episode"]["mean"] == pytest.approx(45.0)
    assert summary["latency_percentiles_exact"] is False


def test_summarize_results_writes_query_weighted_routing_json(tmp_path):
    suite_dir = tmp_path / "libero_spatial"
    suite_dir.mkdir()
    results = [
        _task_result(
            task_id=0,
            successes=1,
            episodes=[{"episode_index": 0, "queries": [_query("wo", 0, gate=0.01, policy=0.1, total=0.11)]}],
        ),
        _task_result(
            task_id=1,
            successes=0,
            episodes=[
                {"episode_index": 0, "queries": [_query("w", 10, gate=0.02, policy=0.2, total=0.22)]},
                {"episode_index": 1, "queries": [_query("w", 10, gate=0.04, policy=0.4, total=0.44)]},
            ],
        ),
    ]
    for result in results:
        path = suite_dir / f"gpu0_task{result['task_id']}_results.json"
        path.write_text(json.dumps(result), encoding="utf-8")

    summarize_results(str(tmp_path))

    payload = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    overall = payload["overall"]["routing"]
    assert overall["counts"] == {"total": 3, "wo": 1, "w": 2}
    assert overall["actual_total_video_nfe"] == 20
    assert overall["avg_video_nfe_per_query"] == pytest.approx(20 / 3)


def test_gate_closed_loop_launcher_is_one_h100_dry_run(tmp_path):
    completed = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "FASTWAM_DRY_RUN": "1",
            "RUN_ID": "pytest",
            "OUTPUT_DIR": str(tmp_path / "unused"),
        },
        check=True,
        capture_output=True,
        text=True,
    )

    output = completed.stdout
    assert "gpu_count=1" in output
    assert "EVALUATION.routing_mode=gate" in output
    assert "model.load_text_encoder=false" in output
    assert "EVALUATION.use_manifest_text_cache=true" in output
    assert "EVALUATION.num_inference_steps=10" in output
    assert "EVALUATION.visualize_future_video=false" in output
    assert "67db6f46abe67f5c6a4417b60864f0ad0535edf8f911d9e4d11eaed137b9b722" in output
    assert "cbc593bc6ce99c0249a65e5c7cef754c9a1d7ea602f81fdae2b8cb158a25858c" in output
