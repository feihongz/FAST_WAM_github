from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path
import signal
import sys
import threading

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_libero_phase_a.py"
LAUNCHER = REPO_ROOT / "scripts" / "jihe" / "eval_libero_phase_a_8xh100.sh"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_libero_phase_a", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


phase_a = _load_module()
SIMULATOR_SHA_ONLY = {"identity_sha256": "f" * 64}


class _LoadedCalibration:
    def __init__(self, target: float):
        assets = phase_a.FrozenAssets()
        thresholds = {
            0.10: 0.6516609191894531,
            0.25: 0.5982248783111572,
            0.50: 0.5151455998420715,
            0.75: 0.47937312722206116,
            0.90: 0.43893179297447205,
        }
        self.threshold = thresholds[target]
        self.manifest = {
            "gate_checkpoint": {
                "sha256": "a" * 64,
                "adapter_checkpoint_sha256": assets.adapter_sha256,
                "base_checkpoint_sha256": assets.base_sha256,
                "data_manifest_sha256": assets.data_manifest_sha256,
            }
        }
        self.receipt = {
            "manifest_file": {"semantic_sha256": "b" * 64},
            "source_identities": {
                "normalization_stats_sha256": assets.normalization_stats_sha256,
            },
            "selected_point": {
                "expected_video_steps_per_query": target * 10,
            }
        }


def _conditions(monkeypatch):
    monkeypatch.setattr(
        phase_a,
        "load_gate_calibration_selection",
        lambda _path, *, expected_complete_sha256, target_with_rate, configured_video_steps: (
            _LoadedCalibration(target_with_rate)
        ),
    )
    return phase_a.resolve_conditions(phase_a.FrozenAssets())


def _fake_task_sources():
    return phase_a.TaskSourceLedger(
        task_map_path="/frozen/libero_suite_task_map.py",
        task_map_sha256="c" * 64,
        task_map_size_bytes=1,
        tasks=tuple(
            phase_a.TaskSource(
                suite=task.suite,
                task_id=task.task_id,
                task_name=f"{task.suite}_task_{task.task_id}",
                source_initial_state_count=50,
                task_bddl_path=f"/frozen/{task.key}.bddl",
                task_bddl_sha256="d" * 64,
                task_bddl_size_bytes=1,
                initial_states_path=f"/frozen/{task.key}.pruned_init",
                initial_states_sha256="e" * 64,
                initial_states_size_bytes=1,
            )
            for task in phase_a.all_tasks()
        ),
    )


def _orchestrator(tmp_path):
    return phase_a.Orchestrator(
        run_root=tmp_path,
        python_bin="/venv/bin/python",
        gpu_devices=tuple(str(index) for index in range(8)),
        assets=phase_a.FrozenAssets(),
        conditions=(),
        run_identity=phase_a.RunIdentity("a" * 40, False, ()),
        task_sources=_fake_task_sources(),
        simulator_runtime_identity=SIMULATOR_SHA_ONLY,
        retry_delay_s=0.0,
    )


def test_conditions_are_compute_ordered_and_thresholds_come_from_loader(monkeypatch):
    conditions = _conditions(monkeypatch)
    assert [condition.name for condition in conditions] == [
        "static_wo",
        "gate_r010",
        "gate_r025",
        "gate_r050",
        "gate_r075",
        "gate_r090",
        "static_w",
    ]
    assert [condition.expected_nfe_per_query for condition in conditions] == [
        0.0,
        1.0,
        2.5,
        5.0,
        7.5,
        9.0,
        10.0,
    ]
    assert conditions[1].resolved_threshold == pytest.approx(0.6516609191894531)


def test_task_source_ledger_freezes_official_assets_and_initial_state_counts(
    monkeypatch, tmp_path
):
    libero_root = tmp_path / "LIBERO"
    benchmark_root = libero_root / "libero" / "libero"
    task_map_path = benchmark_root / "benchmark" / "libero_suite_task_map.py"
    task_map_path.parent.mkdir(parents=True)
    (benchmark_root / "__init__.py").write_text("# package\n", encoding="utf-8")
    task_map = {
        suite: [f"{suite}_task_{task_id}" for task_id in range(10)]
        for suite in phase_a.TASK_SUITES
    }
    task_map_path.write_text(f"libero_task_map = {task_map!r}\n", encoding="utf-8")
    for suite, names in task_map.items():
        for task_id, name in enumerate(names):
            bddl = benchmark_root / "bddl_files" / suite / f"{name}.bddl"
            init = benchmark_root / "init_files" / suite / f"{name}.pruned_init"
            bddl.parent.mkdir(parents=True, exist_ok=True)
            init.parent.mkdir(parents=True, exist_ok=True)
            bddl.write_text(f"bddl-{suite}-{task_id}\n", encoding="utf-8")
            init.write_bytes(f"init-{suite}-{task_id}".encode("utf-8"))
    monkeypatch.setattr(
        phase_a,
        "_trusted_initial_state_count",
        lambda path: 50 + int(path.stem.rsplit("_", 1)[-1].split(".", 1)[0]),
    )
    assets = replace(phase_a.FrozenAssets(), libero_root=str(libero_root))

    ledger = phase_a.build_task_source_ledger(assets)

    assert len(ledger.tasks) == 40
    assert ledger.tasks[0].suite == "libero_10"
    assert ledger.tasks[0].source_initial_state_count == 50
    assert ledger.tasks[-1].source_initial_state_count == 59
    manifest = ledger.manifest()
    assert manifest["ledger_sha256"] == phase_a._canonical_sha256(
        {key: value for key, value in manifest.items() if key != "ledger_sha256"}
    )
    monkeypatch.setenv("FASTWAM_LIBERO_ROOT", str(libero_root))
    simulator_identity = phase_a.capture_libero_simulator_runtime_identity(libero_root)
    contract = phase_a.condition_contract(
        condition=phase_a.Condition(
            "static_wo", "static", "wo", None, None, 0.0, None, None
        ),
        assets=assets,
        run_identity=phase_a.RunIdentity("a" * 40, False, ()),
        task_sources=ledger,
        simulator_runtime_identity=simulator_identity,
    )
    phase_a._formal_validate_contract(contract)

    Path(ledger.tasks[0].task_bddl_path).write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="byte identity mismatch"):
        phase_a._formal_validate_contract(contract)


def test_gate_worker_command_uses_complete_artifact_without_direct_gate_fields(
    monkeypatch, tmp_path
):
    condition = _conditions(monkeypatch)[3]
    assets = phase_a.FrozenAssets()
    command = phase_a.build_evaluator_command(
        python_bin="/venv/bin/python",
        assets=assets,
        condition=condition,
        task=phase_a.Task("libero_goal", 4),
        attempt_dir=tmp_path / "attempt_01",
        simulator_runtime_identity=SIMULATOR_SHA_ONLY,
    )
    joined = " ".join(command)
    assert command[0] == "/venv/bin/python"
    assert "EVALUATION.routing_mode=gate" in command
    assert (
        f"EVALUATION.gate_calibration_complete={assets.calibration_complete}" in command
    )
    assert (
        "EVALUATION.gate_calibration_complete_sha256="
        f"{assets.calibration_complete_sha256}" in command
    )
    assert "EVALUATION.gate_calibration_target_with_rate=0.5" in command
    assert "EVALUATION.gate_threshold=" not in joined
    assert "EVALUATION.gate_checkpoint=" not in joined
    assert "EVALUATION.use_manifest_text_cache=true" in command
    assert "EVALUATION.num_inference_steps=10" in command
    assert "EVALUATION.replan_steps=32" in command
    assert "EVALUATION.action_horizon=32" in command
    assert "EVALUATION.save_videos=false" in command
    assert "EVALUATION.visualize_future_video=false" in command
    assert "gpu_id=0" in command
    assert "EVALUATION.device=cuda:0" in command
    assert "seed=42" in command


def test_static_endpoints_share_the_locked_runtime_contract(monkeypatch, tmp_path):
    conditions = _conditions(monkeypatch)
    for condition, mode in ((conditions[0], "wo"), (conditions[-1], "w")):
        command = phase_a.build_evaluator_command(
            python_bin="python",
            assets=phase_a.FrozenAssets(),
            condition=condition,
            task=phase_a.Task("libero_10", 0),
            attempt_dir=tmp_path / condition.name,
            simulator_runtime_identity=SIMULATOR_SHA_ONLY,
        )
        assert "EVALUATION.routing_mode=static" in command
        assert f"EVALUATION.inference_mode={mode}" in command
        assert "EVALUATION.use_manifest_text_cache=true" in command
        assert not any("gate_calibration" in value for value in command)


def test_single_gpu_child_environment_drops_distributed_parent_state():
    parent = {
        "KEEP_ME": "yes",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "WORLD_SIZE": "8",
        "LOCAL_WORLD_SIZE": "8",
        "RANK": "5",
        "LOCAL_RANK": "5",
        "MASTER_ADDR": "10.0.0.1",
        "MASTER_PORT": "29500",
        "TORCHELASTIC_RUN_ID": "jihe-parent",
        "TORCHELASTIC_RESTART_COUNT": "2",
        "ACCELERATE_USE_DEEPSPEED": "true",
        "ACCELERATE_MIXED_PRECISION": "fp16",
        "PMI_RANK": "5",
        "PMIX_RANK": "5",
        "OMPI_COMM_WORLD_RANK": "5",
        "OMPI_COMM_WORLD_SIZE": "8",
        "MV2_COMM_WORLD_RANK": "5",
        "I_MPI_PIN": "1",
        "MPI_LOCALRANKID": "5",
        "SLURM_PROCID": "5",
        "SLURM_NTASKS": "8",
    }

    child = phase_a.single_gpu_child_environment(
        parent, gpu_device="5"
    )

    assert child == {
        "KEEP_ME": "yes",
        "CUDA_VISIBLE_DEVICES": "5",
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "MUJOCO_EGL_DEVICE_ID": "5",
    }
    assert "WORLD_SIZE" not in child
    assert "LOCAL_RANK" not in child
    assert not any(key.startswith("TORCHELASTIC_") for key in child)
    assert not any(key.startswith(("PMI_", "PMIX_", "OMPI_")) for key in child)


def test_parse_gpu_devices_rejects_non_numeric_tokens():
    with pytest.raises(ValueError, match="exactly 8 distinct GPU"):
        phase_a._parse_gpu_devices(
            "GPU-0,GPU-1,GPU-2,GPU-3,GPU-4,GPU-5,GPU-6,GPU-7"
        )


def test_single_gpu_child_environment_rejects_a_gpu_list():
    with pytest.raises(ValueError, match="exactly one GPU"):
        phase_a.single_gpu_child_environment({}, gpu_device="0,1")


def _valid_result(module, *, task, condition, assets, identity):
    w_queries = 0 if condition.inference_mode == "wo" else 6
    wo_queries = 0 if condition.inference_mode == "w" else 4
    if condition.routing_mode == "gate":
        w_queries, wo_queries = 3, 7
    calibration = None
    if condition.routing_mode == "gate":
        calibration = {
            "complete_file": {"sha256": assets.calibration_complete_sha256},
            "target_with_rate": condition.target_with_rate,
        }
    return {
        "task_suite": task.suite,
        "task_id": task.task_id,
        "total_episodes": module.NUM_TRIALS,
        "successes": 20,
        "success_episodes": list(range(20)),
        "failure_episodes": list(range(20, module.NUM_TRIALS)),
        "invalid_episode_count": 0,
        "evaluation_git_identity": {
            "commit": identity.commit,
            "tracked_dirty": False,
            "untracked_source_files": [],
        },
        "model_artifact_identity": {
            "alignment_export": {"sha256": assets.adapter_sha256},
            "base_checkpoint": {"sha256": assets.base_sha256},
            "data_manifest_sha256": assets.data_manifest_sha256,
        },
        "routing_runtime_identity": {
            "routing_mode": condition.routing_mode,
            "inference_mode": condition.inference_mode,
            "configured_video_steps": module.VIDEO_STEPS,
            "use_manifest_text_cache": True,
            "timing": {
                "enabled": True,
                "warmup_queries_per_task": module.TIMING_WARMUP_QUERIES,
                "save_query_metrics": True,
            },
            "gate_threshold": condition.resolved_threshold,
            "gate_calibration": calibration,
        },
        "routing": {
            "summary": {
                "counts": {
                    "total": w_queries + wo_queries,
                    "wo": wo_queries,
                    "w": w_queries,
                },
                "effective_video_steps": {
                    "total": module.VIDEO_STEPS * w_queries,
                    "mean": module.VIDEO_STEPS * w_queries / (w_queries + wo_queries),
                },
            }
        },
    }


def test_strict_validation_then_atomic_promotion(monkeypatch, tmp_path):
    condition = _conditions(monkeypatch)[2]
    task = phase_a.Task("libero_spatial", 3)
    assets = phase_a.FrozenAssets()
    identity = phase_a.RunIdentity("a" * 40, False, ())
    attempt = tmp_path / "attempts" / "attempt_01"
    source = attempt / task.suite / "gpu0_task3_results.json"
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps(
            _valid_result(
                phase_a,
                task=task,
                condition=condition,
                assets=assets,
                identity=identity,
            )
        ),
        encoding="utf-8",
    )

    def fake_formal_validator(_contract, result_path):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        counts = payload["routing"]["summary"]["counts"]
        return {
            "task_suite_name": task.suite,
            "task_id": task.task_id,
            "result_file": {"sha256": phase_a._sha256_file(result_path)},
            "successes": payload["successes"],
            "num_episodes": payload["total_episodes"],
            "query_count": counts["total"],
            "route_counts": {"wo": counts["wo"], "w": counts["w"]},
            "actual_total_video_nfe": payload["routing"]["summary"][
                "effective_video_steps"
            ]["total"],
        }

    monkeypatch.setattr(phase_a, "_formal_validate_task_file", fake_formal_validator)

    receipt = phase_a.validate_task_result(
        source,
        task=task,
        condition=condition,
        assets=assets,
        run_identity=identity,
        task_sources=_fake_task_sources(),
        simulator_runtime_identity=SIMULATOR_SHA_ONLY,
    )
    condition_dir = tmp_path / "conditions" / condition.name
    promoted = phase_a._promote_result(
        source=source,
        condition_dir=condition_dir,
        task=task,
        receipt=receipt,
    )

    assert promoted.is_file()
    assert not source.exists()
    assert (
        condition_dir / "validation_receipts" / f"{task.key}.json"
    ).is_file()
    with pytest.raises(FileExistsError, match="overwrite"):
        phase_a._promote_result(
            source=promoted,
            condition_dir=condition_dir,
            task=task,
            receipt=receipt,
        )


def test_validator_rejects_wrong_compute_accounting(monkeypatch, tmp_path):
    condition = _conditions(monkeypatch)[-1]
    task = phase_a.Task("libero_object", 9)
    assets = phase_a.FrozenAssets()
    identity = phase_a.RunIdentity("b" * 40, False, ())
    payload = _valid_result(
        phase_a,
        task=task,
        condition=condition,
        assets=assets,
        identity=identity,
    )
    payload["routing"]["summary"]["effective_video_steps"]["total"] = 1
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        phase_a,
        "_formal_validate_task_file",
        lambda _contract, _path: (_ for _ in ()).throw(
            ValueError("actual video NFE disagrees with raw queries")
        ),
    )
    with pytest.raises(ValueError, match="video NFE"):
        phase_a.validate_task_result(
            path,
            task=task,
            condition=condition,
            assets=assets,
            run_identity=identity,
            task_sources=_fake_task_sources(),
            simulator_runtime_identity=SIMULATOR_SHA_ONLY,
        )


def test_fresh_run_refuses_existing_directory_and_resume_checks_spec(tmp_path):
    assets = phase_a.FrozenAssets()
    run_root = tmp_path / "run"
    spec = {"schema_version": 1, "value": "locked"}
    phase_a.prepare_run_root(
        run_root=run_root,
        resume=False,
        spec=spec,
        assets=assets,
    )
    with pytest.raises(FileExistsError):
        phase_a.prepare_run_root(
            run_root=run_root,
            resume=False,
            spec=spec,
            assets=assets,
        )
    phase_a.prepare_run_root(
        run_root=run_root,
        resume=True,
        spec=spec,
        assets=assets,
    )
    with pytest.raises(ValueError, match="spec"):
        phase_a.prepare_run_root(
            run_root=run_root,
            resume=True,
            spec={"schema_version": 1, "value": "changed"},
            assets=assets,
        )


def test_resume_rejects_complete_root_without_touching_marker(tmp_path):
    assets = phase_a.FrozenAssets()
    run_root = tmp_path / "run"
    spec = {"schema_version": 1, "value": "locked"}
    phase_a.prepare_run_root(
        run_root=run_root,
        resume=False,
        spec=spec,
        assets=assets,
    )
    complete = run_root / "COMPLETE.json"
    complete.write_bytes(b"{\"complete\":true}\n")
    before = (complete.read_bytes(), complete.stat().st_mtime_ns)

    with pytest.raises(ValueError, match="already complete"):
        phase_a.prepare_run_root(
            run_root=run_root,
            resume=True,
            spec=spec,
            assets=assets,
        )

    assert (complete.read_bytes(), complete.stat().st_mtime_ns) == before


def test_resume_archives_stale_failed_marker_after_contract_checks(tmp_path):
    assets = phase_a.FrozenAssets()
    run_root = tmp_path / "run"
    spec = {"schema_version": 1, "value": "locked"}
    phase_a.prepare_run_root(
        run_root=run_root,
        resume=False,
        spec=spec,
        assets=assets,
    )
    failed = run_root / "FAILED.json"
    failed_payload = b"{\"error\":\"old\"}\n"
    failed.write_bytes(failed_payload)

    phase_a.prepare_run_root(
        run_root=run_root,
        resume=True,
        spec=spec,
        assets=assets,
    )

    assert not failed.exists()
    archived = list((run_root / "history" / "failures").glob("FAILED.*.json"))
    assert len(archived) == 1
    assert archived[0].read_bytes() == failed_payload


def test_atomic_text_preserves_old_target_and_cleans_temp_on_replace_error(
    monkeypatch, tmp_path
):
    target = tmp_path / "config.yaml"
    target.write_text("old\n", encoding="utf-8")
    real_replace = phase_a.os.replace

    def fail_target_replace(source, destination):
        if Path(destination) == target:
            raise OSError("injected replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(phase_a.os, "replace", fail_target_replace)
    with pytest.raises(OSError, match="injected"):
        phase_a._atomic_text(target, "new\n")

    assert target.read_text(encoding="utf-8") == "old\n"
    assert list(tmp_path.glob(".config.yaml.tmp.*")) == []


def test_experiment_spec_round_trips_through_json(monkeypatch):
    conditions = _conditions(monkeypatch)
    spec = phase_a._experiment_spec(
        assets=phase_a.FrozenAssets(),
        conditions=conditions,
        run_identity=phase_a.RunIdentity("c" * 40, False, ("one.py",)),
        task_sources=_fake_task_sources(),
        simulator_runtime_identity=SIMULATOR_SHA_ONLY,
        python_bin="/venv/bin/python",
    )
    assert json.loads(json.dumps(spec)) == spec
    assert spec["python_bin"] == "/venv/bin/python"


def test_attempt_numbers_are_never_reused(tmp_path):
    assert phase_a._next_attempt_number(tmp_path) == 1
    (tmp_path / "attempt_01").mkdir(parents=True)
    (tmp_path / "attempt_03").mkdir()
    assert phase_a._next_attempt_number(tmp_path) == 4


def test_attempt_recovery_counts_only_terminal_failures(tmp_path):
    condition = phase_a.Condition(
        "static_wo", "static", "wo", None, None, 0.0, None, None
    )
    task = phase_a.Task("libero_spatial", 0)
    task_root = tmp_path / task.key
    for attempt, status in ((1, "aborted"), (3, "rejected")):
        attempt_dir = task_root / f"attempt_{attempt:02d}"
        attempt_dir.mkdir(parents=True)
        phase_a._atomic_json(
            attempt_dir / "attempt.json",
            {
                "schema_version": 1,
                "kind": "libero_phase_a_task_attempt",
                "condition": condition.name,
                "task": phase_a.asdict(task),
                "attempt": attempt,
                "status": status,
            },
        )
    (task_root / "attempt_02").mkdir()

    assert phase_a._recover_attempt_history(
        task_root,
        condition=condition,
        task=task,
    ) == 1
    assert phase_a._next_attempt_number(task_root) == 4
    recovered = json.loads(
        (task_root / "attempt_02" / "attempt.json").read_text(encoding="utf-8")
    )
    assert recovered["status"] == "abandoned"


def test_launch_failures_are_bounded_independently_of_attempt_ids(
    monkeypatch, tmp_path
):
    orchestrator = _orchestrator(tmp_path)
    condition = phase_a.Condition(
        "static_wo", "static", "wo", None, None, 0.0, None, None
    )
    task = phase_a.Task("libero_spatial", 0)
    launches = 0

    def fail_launch(*_args, **_kwargs):
        nonlocal launches
        launches += 1
        raise OSError("injected launch failure")

    monkeypatch.setattr(phase_a.subprocess, "Popen", fail_launch)
    assert not orchestrator._run_task(
        condition=condition,
        condition_dir=tmp_path / "conditions" / condition.name,
        task=task,
        gpu_slot=0,
        gpu_device="0",
    )
    assert launches == phase_a.MAX_ATTEMPTS
    attempt_root = tmp_path / "attempts" / condition.name / task.key
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(attempt_root.glob("attempt_*/attempt.json"))
    ]
    assert [record["status"] for record in records] == [
        "launch_failed"
    ] * phase_a.MAX_ATTEMPTS
    assert not orchestrator._run_task(
        condition=condition,
        condition_dir=tmp_path / "conditions" / condition.name,
        task=task,
        gpu_slot=0,
        gpu_device="0",
    )
    assert launches == phase_a.MAX_ATTEMPTS


def test_signal_cannot_miss_child_between_spawn_and_active_registration(
    monkeypatch, tmp_path
):
    orchestrator = _orchestrator(tmp_path)
    entered_popen = threading.Event()
    release_popen = threading.Event()
    killed: list[tuple[int, int]] = []

    class FakeProcess:
        pid = 4242

        @staticmethod
        def poll():
            return None

    process = FakeProcess()

    def blocking_popen(*_args, **_kwargs):
        entered_popen.set()
        assert release_popen.wait(timeout=2)
        return process

    monkeypatch.setattr(phase_a.subprocess, "Popen", blocking_popen)
    monkeypatch.setattr(
        phase_a.os,
        "killpg",
        lambda pid, signum: killed.append((pid, signum)),
    )
    launched: list[object] = []
    launcher = threading.Thread(
        target=lambda: launched.append(
            orchestrator._spawn_active(
                ["/venv/bin/python", "evaluator.py"],
                environment={},
                log_stream=object(),
            )
        )
    )
    launcher.start()
    assert entered_popen.wait(timeout=2)
    stopper = threading.Thread(
        target=orchestrator.request_stop,
        args=(signal.SIGTERM,),
    )
    stopper.start()
    assert orchestrator.stop_event.wait(timeout=2)
    release_popen.set()
    launcher.join(timeout=2)
    stopper.join(timeout=2)

    assert not launcher.is_alive()
    assert not stopper.is_alive()
    assert launched == [process]
    assert killed == [(process.pid, signal.SIGTERM)]
    assert orchestrator.active == {process.pid: process}
    orchestrator._unregister_active(process)
    assert orchestrator.active == {}


def test_jihe_launcher_exposes_only_fresh_or_explicit_resume_interface():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "scripts/run_libero_phase_a.py" in text
    assert "--resume /absolute/run/root" in text
    assert "FASTWAM_DRY_RUN=1" in text
    assert "expected exactly 8 visible H100 GPUs" in text
    assert "NPROC_PER_NODE" not in text
    assert "export MUJOCO_GL=egl" in text
    assert "export PYOPENGL_PLATFORM=egl" in text
    assert "unset MUJOCO_EGL_DEVICE_ID" in text
