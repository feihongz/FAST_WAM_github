from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import pytest

from fastwam.alignment.libero_simulator_identity import (
    capture_libero_simulator_runtime_identity,
)
from experiments.libero.formal_eval_validation import (
    CONDITION_CONTRACT_KIND,
    aggregate_condition_result_files,
    condition_contract_sha256,
    validate_condition_contract,
    validate_task_result,
    validate_task_result_file,
)


SHA = {
    "base": "1" * 64,
    "adapter": "2" * 64,
    "data": "3" * 64,
    "stats": "4" * 64,
    "vae": "5" * 64,
    "gate": "6" * 64,
    "calibration": "7" * 64,
    "complete": "8" * 64,
}
GIT = {
    "commit": "9" * 40,
    "tracked_dirty": False,
    "untracked_source_files": [],
}
_ENVIRONMENT_ASSETS = None
_SIMULATOR_RUNTIME_IDENTITY = None


@pytest.fixture(scope="session")
def _simulator_runtime_identity(tmp_path_factory):
    libero_root = tmp_path_factory.mktemp("libero-runtime") / "LIBERO"
    source_root = libero_root / "libero" / "libero"
    source_root.mkdir(parents=True)
    (source_root / "__init__.py").write_text("# package\n", encoding="utf-8")
    (source_root / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source_root / "arena.xml").write_text("<mujoco/>\n", encoding="utf-8")
    runtime_environ = {
        "FASTWAM_LIBERO_ROOT": str(libero_root),
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
    }
    return capture_libero_simulator_runtime_identity(
        libero_root, environ=runtime_environ
    )


@pytest.fixture(autouse=True)
def _environment_asset_files(tmp_path, monkeypatch, _simulator_runtime_identity):
    global _ENVIRONMENT_ASSETS, _SIMULATOR_RUNTIME_IDENTITY
    _SIMULATOR_RUNTIME_IDENTITY = _simulator_runtime_identity
    runtime_variables = _simulator_runtime_identity["runtime_environment"][
        "variables"
    ]
    for key, value in runtime_variables.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    for key in (
        "FASTWAM_CONTAINER_IMAGE_DIGEST",
        "FASTWAM_JIHE_IMAGE_DIGEST",
        "JIHE_IMAGE_DIGEST",
        "CONTAINER_IMAGE_DIGEST",
        "FASTWAM_JIHE_ENVIRONMENT_IDENTITY",
        "JIHE_ENVIRONMENT_IDENTITY",
    ):
        monkeypatch.delenv(key, raising=False)
    task_bddl = tmp_path / "task.bddl"
    initial_states = tmp_path / "task.init"
    task_bddl.write_bytes(b"(define task)\n")
    initial_states.write_bytes(b"official-init-states\n")
    _ENVIRONMENT_ASSETS = {
        "task_bddl": {
            "path": str(task_bddl.resolve()),
            "sha256": hashlib.sha256(task_bddl.read_bytes()).hexdigest(),
            "size_bytes": task_bddl.stat().st_size,
        },
        "initial_states": {
            "path": str(initial_states.resolve()),
            "sha256": hashlib.sha256(initial_states.read_bytes()).hexdigest(),
            "size_bytes": initial_states.stat().st_size,
        },
    }
    yield
    _ENVIRONMENT_ASSETS = None
    _SIMULATOR_RUNTIME_IDENTITY = None


def _canonical_sha(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()


def _runtime(mode="static", inference_mode="wo", threshold=None):
    gate = None
    calibration = None
    if mode == "gate":
        gate = {
            "path": "/gate.pt",
            "sha256": SHA["gate"],
            "adapter_checkpoint_sha256": SHA["adapter"],
            "base_checkpoint_sha256": SHA["base"],
            "data_manifest_sha256": SHA["data"],
        }
        calibration = {
            "complete_file": {"path": "/COMPLETE", "sha256": SHA["complete"]},
            "manifest_file": {
                "path": "/calibration_manifest.json",
                "semantic_sha256": SHA["calibration"],
            },
            "configured_video_steps": 10,
            "gate_checkpoint_sha256": SHA["gate"],
            "selected_point": {"threshold": threshold},
            "source_identities": {
                "adapter_checkpoint_sha256": SHA["adapter"],
                "base_checkpoint_sha256": SHA["base"],
                "data_manifest_sha256": SHA["data"],
                "normalization_stats_sha256": SHA["stats"],
            },
        }
    return {
        "schema_version": 1,
        "kind": "binary_video_routing_runtime",
        "routing_mode": mode,
        "inference_mode": inference_mode,
        "configured_video_steps": 10,
        "gate_threshold": threshold,
        "gate_decision_rule": (
            "sigmoid(logit) >= gate_threshold -> w" if mode == "gate" else None
        ),
        "random_video_probability": None,
        "random_seed": None,
        "use_manifest_text_cache": mode == "gate",
        "prompt_template": "In: What action should the robot take to {task}? Out:",
        "gate_input_preprocessing": "exact" if mode == "gate" else None,
        "timing": {"enabled": True, "warmup_queries_per_task": 1, "save_query_metrics": True},
        "gate_checkpoint": gate,
        "gate_calibration": calibration,
    }


def _protocol_shared(runtime, *, trials):
    return {
        "schema_version": 1,
        "kind": "libero_closed_loop_evaluation_protocol",
        "benchmark": "LIBERO",
        "env_num": 1,
        "render_resolution": 256,
        "seed": 42,
        "mixed_precision": "bf16",
        "num_trials": trials,
        "initial_state_order": "official",
        "num_steps_wait": 5,
        "action_horizon": 16,
        "replan_steps": 5,
        "num_inference_steps": 10,
        "binarize_gripper": False,
        "use_action_ensembler": False,
        "visualize_future_video": False,
        "save_videos": False,
        "retry_invalid_episodes": False,
        "max_invalid_episode_retries": 0,
        "black_screen_filter": False,
        "black_screen_thresholds": {"mean": 5.0, "std": 2.0, "minimum_frame_fraction": 0.8},
        "timing": {"enabled": True, "warmup_queries_per_task": 1, "save_query_metrics": True},
        "sampling": {"sigma_shift": None, "text_cfg_scale": 1.0, "negative_prompt": "", "rand_device": "cpu", "tiled": False},
        "model_input": {"height": 224, "width": 448, "num_frames": 17, "concat_multi_camera": "horizontal", "action_video_freq_ratio": 2},
        "prompt_template": "In: What action should the robot take to {task}? Out:",
        "routing_runtime_identity_sha256": _canonical_sha(runtime),
        "simulator_runtime_identity_sha256": _SIMULATOR_RUNTIME_IDENTITY[
            "identity_sha256"
        ],
    }


def _contract(*, mode="static", inference_mode="wo", threshold=None, trials=2, tasks=None):
    runtime = _runtime(mode, inference_mode, threshold)
    if tasks is None:
        tasks = [{"task_suite_name": "libero_spatial", "task_id": 0}]
    tasks = [
        {
            **deepcopy(task),
            "source_initial_state_count": task.get(
                "source_initial_state_count", 50
            ),
            "environment_assets": deepcopy(
                task.get("environment_assets", _ENVIRONMENT_ASSETS)
            ),
        }
        for task in tasks
    ]
    return {
        "schema_version": 1,
        "kind": CONDITION_CONTRACT_KIND,
        "condition_id": inference_mode if mode == "static" else "gate_050",
        "expected_tasks": tasks,
        "num_trials_per_task": trials,
        "simulator_runtime_identity": deepcopy(_SIMULATOR_RUNTIME_IDENTITY),
        "routing": {
            "routing_mode": mode,
            "configured_video_steps": 10,
            "inference_mode": inference_mode,
            "gate_threshold": threshold,
            "calibration_complete_sha256": SHA["complete"] if mode == "gate" else None,
        },
        "expected_identities": {
            "evaluation_git_identity": GIT,
            "base_checkpoint_sha256": SHA["base"],
            "alignment_export_sha256": SHA["adapter"],
            "data_manifest_sha256": SHA["data"],
            "normalization_stats_sha256": SHA["stats"],
            "vae_sha256": SHA["vae"],
            "gate_checkpoint_sha256": SHA["gate"] if mode == "gate" else None,
            "calibration_manifest_sha256": SHA["calibration"] if mode == "gate" else None,
        },
        "protocol_shared": _protocol_shared(runtime, trials=trials),
    }


def _summary(queries):
    total = len(queries)
    wo = sum(query["selected_mode"] == "wo" for query in queries)
    w = total - wo
    steps = 10 * w
    timed = sum(query["timing_included"] for query in queries)
    return {
        "counts": {"total": total, "wo": wo, "w": w},
        "with_rate": w / total if total else 0.0,
        "effective_video_steps": {"total": steps, "mean": steps / total if total else 0.0},
        "latency_s": {},
        "by_route": {
            "wo": {"count": wo, "effective_video_steps": {"total": 0, "mean": 0.0}, "latency_s": {}},
            "w": {"count": w, "effective_video_steps": {"total": steps, "mean": 10.0 if w else 0.0}, "latency_s": {}},
        },
        "timing": {"query_count": timed, "warmup_query_count": total - timed, "cuda_synchronized": True},
    }


def _query(*, suite, task_id, episode, replan, global_index, attempt, route, mode, threshold):
    if mode == "gate":
        probability = 0.75 if route == "w" else 0.25
        logit = math.log(probability / (1 - probability))
        gate_latency = 0.01
    else:
        probability = logit = gate_latency = None
    return {
        "episode_index": episode,
        "replan_index": replan,
        "global_query_index": global_index,
        "attempt_index": attempt,
        "environment_step": 5 + 5 * replan,
        "query_id": f"{suite}/{task_id}/{episode}/{replan}",
        "selected_mode": route,
        "configured_video_steps": 10,
        "actual_video_steps": 10 if route == "w" else 0,
        "logit": logit,
        "probability": probability,
        "gate_latency_s": gate_latency,
        "policy_latency_s": 0.2,
        "preprocess_plus_gate_latency_s": 0.02,
        "total_latency_s": 0.22,
        "timing_included": global_index > 0,
        "timing_synchronized": True,
    }


def _model_identity():
    return {
        "schema_version": 1,
        "kind": "stage3_aligned_model_identity",
        "base_checkpoint": {"sha256": SHA["base"]},
        "alignment_export": {
            "sha256": SHA["adapter"],
            "export_metadata": {
                "base_checkpoint_sha256": SHA["base"],
                "data_manifest_sha256": SHA["data"],
            },
        },
        "data_manifest_sha256": SHA["data"],
        "runtime_assets": {
            "normalization_stats": {"sha256": SHA["stats"]},
            "vae": {"sha256": SHA["vae"]},
        },
    }


def _result(contract, *, suite="libero_spatial", task_id=0, successes=1):
    mode = contract["routing"]["routing_mode"]
    static_route = contract["routing"]["inference_mode"]
    threshold = contract["routing"]["gate_threshold"]
    trials = contract["num_trials_per_task"]
    episodes = []
    all_queries = []
    for episode in range(trials):
        route = static_route if mode == "static" else ("wo" if episode == 0 else "w")
        query = _query(
            suite=suite,
            task_id=task_id,
            episode=episode,
            replan=0,
            global_index=len(all_queries),
            attempt=episode,
            route=route,
            mode=mode,
            threshold=threshold,
        )
        all_queries.append(query)
        episodes.append(
            {
                "episode_index": episode,
                "success": episode < successes,
                "query_count": 1,
                "total_actual_video_steps": query["actual_video_steps"],
                "summary": _summary([query]),
                "queries": [query],
            }
        )
    runtime = _runtime(mode, static_route, threshold)
    protocol = {
        **deepcopy(contract["protocol_shared"]),
        "task_suite_name": suite,
        "task_id": task_id,
        "source_initial_state_count": 50,
        "max_environment_steps": 700 if suite == "libero_10" else 400,
        "environment_assets": deepcopy(_ENVIRONMENT_ASSETS),
    }
    protocol["protocol_sha256"] = _canonical_sha(protocol)
    return {
        "task_suite": suite,
        "task_id": task_id,
        "total_episodes": trials,
        "successes": successes,
        "success_episodes": list(range(successes)),
        "failure_episodes": list(range(successes, trials)),
        "attempted_episodes": trials,
        "invalid_episode_count": 0,
        "invalid_episodes": [],
        "routing": {"episodes": episodes, "invalid_attempts": [], "summary": _summary(all_queries)},
        "routing_runtime_identity": runtime,
        "simulator_runtime_identity": deepcopy(
            contract["simulator_runtime_identity"]
        ),
        "model_artifact_identity": _model_identity(),
        "evaluation_git_identity": deepcopy(GIT),
        "evaluation_protocol_identity": protocol,
    }


def test_valid_static_task_returns_json_safe_receipt():
    contract = _contract()
    receipt = validate_task_result(contract, _result(contract))

    assert receipt["condition_id"] == "wo"
    assert receipt["successes"] == 1
    assert receipt["route_counts"] == {"wo": 2, "w": 0}
    assert receipt["actual_total_video_nfe"] == 0
    assert receipt["actual_video_nfe_per_query"] == 0.0
    json.dumps(receipt, allow_nan=False)


def test_valid_gate_task_enforces_threshold_and_reports_actual_nfe():
    contract = _contract(mode="gate", inference_mode=None, threshold=0.5)
    receipt = validate_task_result(contract, _result(contract))

    assert receipt["route_counts"] == {"wo": 1, "w": 1}
    assert receipt["actual_total_video_nfe"] == 10
    assert receipt["actual_video_nfe_per_query"] == 5.0


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda value: value.update(total_episodes=3), "total_episodes"),
        (lambda value: value["success_episodes"].append(1), "successes disagrees"),
        (lambda value: value["routing"]["episodes"][1]["queries"][0].update(global_query_index=7), "globally contiguous"),
        (lambda value: value["routing"]["episodes"][0]["queries"][0].update(query_id="wrong"), "query_id"),
        (lambda value: value["routing"]["episodes"][0]["queries"][0].update(actual_video_steps=10), "wo routing query"),
        (lambda value: value["routing"]["summary"]["counts"].update(total=99), "counts disagrees"),
        (lambda value: value["evaluation_git_identity"].update(commit="a" * 40), "does not match"),
        (lambda value: value["model_artifact_identity"]["alignment_export"].update(sha256="a" * 64), "alignment_export_sha256 mismatch"),
        (lambda value: value["evaluation_protocol_identity"].update(source_initial_state_count=49), "source_initial_state_count does not match"),
        (lambda value: value["evaluation_protocol_identity"].update(protocol_sha256="a" * 64), "self-SHA256"),
    ],
)
def test_task_validation_rejects_tampered_results(mutator, match):
    contract = _contract()
    result = _result(contract)
    mutator(result)
    with pytest.raises(ValueError, match=match):
        validate_task_result(contract, result)


def test_gate_validation_rejects_wrong_decision_and_calibration_identity():
    contract = _contract(mode="gate", inference_mode=None, threshold=0.5)
    result = _result(contract)
    wrong_query = result["routing"]["episodes"][0]["queries"][0]
    wrong_query["selected_mode"] = "w"
    wrong_query["actual_video_steps"] = 10
    with pytest.raises(ValueError, match="threshold decision rule"):
        validate_task_result(contract, result)

    result = _result(contract)
    result["routing_runtime_identity"]["gate_calibration"]["manifest_file"][
        "semantic_sha256"
    ] = "a" * 64
    # Keep protocol binding synchronized: the independent expected identity
    # check must still catch the wrong calibration.
    protocol = result["evaluation_protocol_identity"]
    protocol["routing_runtime_identity_sha256"] = _canonical_sha(
        result["routing_runtime_identity"]
    )
    protocol.pop("protocol_sha256")
    protocol["protocol_sha256"] = _canonical_sha(protocol)
    with pytest.raises(ValueError, match="calibration manifest"):
        validate_task_result(contract, result)


def test_attempt_and_invalid_accounting_is_strict():
    contract = _contract()
    result = _result(contract)
    result["attempted_episodes"] = 3
    with pytest.raises(ValueError, match="attempted/invalid"):
        validate_task_result(contract, result)


def test_result_environment_assets_must_match_contract_even_if_self_consistent(tmp_path):
    contract = _contract()
    result = _result(contract)
    replacement = tmp_path / "replacement.bddl"
    replacement.write_bytes(b"different-valid-file\n")
    replacement_identity = {
        "path": str(replacement.resolve()),
        "sha256": hashlib.sha256(replacement.read_bytes()).hexdigest(),
        "size_bytes": replacement.stat().st_size,
    }
    protocol = result["evaluation_protocol_identity"]
    protocol["environment_assets"]["task_bddl"] = replacement_identity
    protocol.pop("protocol_sha256")
    protocol["protocol_sha256"] = _canonical_sha(protocol)

    with pytest.raises(ValueError, match="environment_assets do not match"):
        validate_task_result(contract, result)


def test_simulator_identity_records_required_runtime_components():
    identity = _SIMULATOR_RUNTIME_IDENTITY
    assert set(identity["packages"]) == {
        "torch", "numpy", "robosuite", "mujoco", "libero"
    }
    assert identity["python"]["prefix"]
    assert identity["python"]["base_prefix"]
    assert identity["libero_source_tree"]["coverage"] == (
        "source_and_physics_configuration_files"
    )
    assert identity["runtime_environment"]["renderer_policy"].startswith(
        "egl_with_child_"
    )


def test_simulator_identity_schema_rejects_rehashed_missing_nested_field():
    contract = _contract()
    identity = contract["simulator_runtime_identity"]
    del identity["python"]["prefix"]
    identity.pop("identity_sha256")
    identity["identity_sha256"] = _canonical_sha(identity)

    with pytest.raises(ValueError, match="python schema mismatch"):
        validate_condition_contract(contract)


def test_task_validation_physically_rejects_simulator_source_drift():
    contract = _contract()
    result = _result(contract)
    runtime_source = (
        Path(contract["simulator_runtime_identity"]["libero_source_tree"]["root"])
        / "runtime.py"
    )
    original = runtime_source.read_bytes()
    try:
        runtime_source.write_bytes(b"VALUE = 2\n")
        with pytest.raises(ValueError, match="drifted"):
            validate_task_result(contract, result)
    finally:
        runtime_source.write_bytes(original)


def test_task_validation_rejects_self_consistent_wrong_simulator_identity():
    contract = _contract()
    result = _result(contract)
    wrong = result["simulator_runtime_identity"]
    wrong["python"]["version"] = "0.0.0"
    wrong.pop("identity_sha256")
    wrong["identity_sha256"] = _canonical_sha(wrong)
    with pytest.raises(ValueError, match="does not match"):
        validate_task_result(contract, result)


def test_contract_environment_assets_are_physically_verified():
    contract = _contract()
    task_bddl = contract["expected_tasks"][0]["environment_assets"]["task_bddl"]
    with open(task_bddl["path"], "ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(ValueError, match="byte identity mismatch"):
        validate_condition_contract(contract)


def test_contract_rejects_duplicate_tasks_and_static_gate_identity():
    contract = _contract()
    contract["expected_tasks"].append(deepcopy(contract["expected_tasks"][0]))
    with pytest.raises(ValueError, match="duplicate task"):
        validate_condition_contract(contract)

    contract = _contract()
    contract["expected_identities"]["gate_checkpoint_sha256"] = SHA["gate"]
    with pytest.raises(ValueError, match="null Gate"):
        validate_condition_contract(contract)


def test_result_file_receipt_binds_exact_bytes(tmp_path):
    contract = _contract()
    result = _result(contract)
    path = tmp_path / "result.json"
    raw = json.dumps(result, indent=2).encode()
    path.write_bytes(raw)

    receipt = validate_task_result_file(contract, path)

    assert receipt["result_file"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert receipt["result_file"]["size_bytes"] == len(raw)
    assert receipt["condition_contract_sha256"] == condition_contract_sha256(contract)


def test_aggregate_validates_full_40_task_coverage_and_pools(tmp_path):
    tasks = [
        {"task_suite_name": suite, "task_id": task_id}
        for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10")
        for task_id in range(10)
    ]
    contract = _contract(trials=1, tasks=tasks)
    paths = []
    for index, task in enumerate(tasks):
        result = _result(
            contract,
            suite=task["task_suite_name"],
            task_id=task["task_id"],
            successes=index % 2,
        )
        path = tmp_path / f"result_{index:02d}.json"
        path.write_text(json.dumps(result), encoding="utf-8")
        paths.append(path)

    aggregate = aggregate_condition_result_files(contract, list(reversed(paths)))

    assert aggregate["task_count"] == 40
    assert aggregate["episode_count"] == 40
    assert aggregate["successes"] == 20
    assert aggregate["success_rate"] == pytest.approx(0.5)
    assert aggregate["query_count"] == 40
    assert aggregate["route_counts"] == {"wo": 40, "w": 0}
    assert len(aggregate["result_file_ledger"]) == 40
    assert aggregate["result_file_ledger"][0]["task_suite_name"] == "libero_10"
    json.dumps(aggregate, allow_nan=False)


def test_aggregate_rejects_missing_and_duplicate_task_files(tmp_path):
    tasks = [
        {"task_suite_name": "libero_spatial", "task_id": 0},
        {"task_suite_name": "libero_spatial", "task_id": 1},
    ]
    contract = _contract(trials=1, tasks=tasks)
    path0 = tmp_path / "task0.json"
    path0.write_text(json.dumps(_result(contract, task_id=0)), encoding="utf-8")
    with pytest.raises(ValueError, match="coverage mismatch"):
        aggregate_condition_result_files(contract, [path0])
    with pytest.raises(ValueError, match="duplicate file"):
        aggregate_condition_result_files(contract, [path0, path0])

    duplicate = tmp_path / "duplicate_task0.json"
    duplicate.write_text(path0.read_text(), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate task result"):
        aggregate_condition_result_files(contract, [path0, duplicate])
