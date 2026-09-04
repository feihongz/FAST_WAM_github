from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
LIBERO_ROOT = Path("/root/feihong/FastWAM/third_party/LIBERO")
os.environ.setdefault("FASTWAM_LIBERO_ROOT", str(LIBERO_ROOT))
LIBERO_EVAL_DIR = str(REPO_ROOT / "experiments" / "libero")
if LIBERO_EVAL_DIR not in sys.path:
    sys.path.insert(0, LIBERO_EVAL_DIR)

from experiments.libero import eval_libero_single as evaluator
from experiments.libero.summarize_results import (
    aggregate_routing_results,
    summarize_results,
)

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


def test_evaluation_protocol_identity_is_self_hashed_and_task_specific():
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "mixed_precision": "bf16",
            "data": {
                "train": {
                    "num_frames": 33,
                    "concat_multi_camera": "horizontal",
                    "action_video_freq_ratio": 1,
                }
            },
            "EVALUATION": {
                "task_suite_name": "libero_10",
                "task_id": 7,
                "num_trials": 50,
                "num_steps_wait": 30,
                "replan_steps": 32,
                "num_inference_steps": 10,
                "binarize_gripper": True,
                "use_action_ensembler": False,
                "visualize_future_video": False,
                "save_videos": False,
                "retry_invalid_episodes": False,
                "max_invalid_episode_retries": 20,
                "black_screen_filter": True,
                "timing_enabled": True,
                "timing_warmup_queries": 3,
                "save_query_metrics": True,
                "sigma_shift": None,
                "text_cfg_scale": 1.0,
                "negative_prompt": "",
                "rand_device": "cpu",
                "tiled": False,
            },
        }
    )
    routing_identity = {"schema_version": 1, "routing_mode": "gate"}
    environment_assets = {
        "task_bddl": {
            "path": "/tmp/task.bddl",
            "sha256": "a" * 64,
            "size_bytes": 10,
        },
        "initial_states": {
            "path": "/tmp/task.pruned_init",
            "sha256": "b" * 64,
            "size_bytes": 20,
        },
    }

    protocol = evaluator._evaluation_protocol_identity(
        cfg,
        action_horizon=32,
        input_h=224,
        input_w=448,
        source_initial_state_count=50,
        routing_runtime_identity=routing_identity,
        environment_assets=environment_assets,
        simulator_runtime_identity=None,
    )

    recorded_sha = protocol.pop("protocol_sha256")
    assert recorded_sha == evaluator.canonical_json_sha256(protocol)
    assert protocol["max_environment_steps"] == 700
    assert protocol["env_num"] == 1
    assert protocol["render_resolution"] == 256
    assert protocol["environment_assets"] == environment_assets
    assert protocol["num_trials"] == 50
    assert protocol["model_input"]["width"] == 448
    assert protocol["routing_runtime_identity_sha256"] == (
        evaluator.canonical_json_sha256(routing_identity)
    )
    assert protocol["simulator_runtime_identity_sha256"] is None


def test_formal_simulator_preflight_binds_physical_egl_device(monkeypatch):
    identity = {"identity_sha256": "a" * 64}
    cfg = OmegaConf.create(
        {
            "EVALUATION": {
                "simulator_runtime_identity_sha256": "a" * 64
            }
        }
    )
    monkeypatch.setenv("FASTWAM_LIBERO_ROOT", "/frozen/LIBERO")
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "egl")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "5")
    monkeypatch.setattr(
        evaluator,
        "capture_libero_simulator_runtime_identity",
        lambda root: identity,
    )
    monkeypatch.setattr(
        evaluator, "verify_loaded_libero_module_location", lambda *_args: None
    )

    assert evaluator._prepare_simulator_runtime_identity(cfg) == identity

    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "0")
    with pytest.raises(ValueError, match="must equal"):
        evaluator._prepare_simulator_runtime_identity(cfg)


def test_evaluator_git_identity_end_check_detects_drift(monkeypatch):
    expected = {
        "commit": "a" * 40,
        "tracked_dirty": False,
        "untracked_source_files": [],
    }

    class Identity:
        def __init__(self, payload):
            self.payload = payload

        def as_dict(self):
            return self.payload

    monkeypatch.setattr(
        evaluator, "read_git_identity", lambda _root: Identity(expected)
    )
    evaluator._verify_evaluation_git_identity_unchanged(
        expected, require_clean=True
    )
    changed = {**expected, "commit": "b" * 40}
    monkeypatch.setattr(
        evaluator, "read_git_identity", lambda _root: Identity(changed)
    )
    with pytest.raises(RuntimeError, match="changed"):
        evaluator._verify_evaluation_git_identity_unchanged(
            expected, require_clean=True
        )


def test_atomic_result_writer_replaces_complete_json_without_temp_file(tmp_path):
    output = tmp_path / "gpu0_task0_results.json"

    evaluator._write_json_atomic(output, {"value": 1, "array": np.array([2, 3])})
    evaluator._write_json_atomic(output, {"value": 4})

    assert json.loads(output.read_text(encoding="utf-8")) == {"value": 4}
    assert list(tmp_path.glob(".*.tmp")) == []
