from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from experiments.libero.init_state_compat import load_libero_task_init_states
from experiments.robotwin.eval_robotwin_single import (
    _append_override,
    _select_cuda_visible_device,
)
from experiments.robotwin.fastwam_policy import deploy_policy


REPO_ROOT = Path(__file__).resolve().parents[1]
ALIGNED_MODEL_TARGET = "fastwam.runtime.create_fastwam_unified_aligned"


def test_libero_legacy_numpy_init_states_load_on_new_pytorch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_states_root = tmp_path / "init_states"
    init_states_path = init_states_root / "suite" / "states.pt"
    init_states_path.parent.mkdir(parents=True)
    expected = [np.arange(7, dtype=np.float32)]
    torch.save(expected, init_states_path)

    task = SimpleNamespace(
        problem_folder="suite",
        init_states_file="states.pt",
    )
    task_suite = SimpleNamespace(get_task=lambda task_id: task)
    loaded = load_libero_task_init_states(
        task_suite,
        0,
        init_states_root=init_states_root,
    )

    np.testing.assert_array_equal(loaded[0], expected[0])


@pytest.mark.parametrize(
    ("config_name", "task_name"),
    [
        ("sim_libero", "libero_stage3_alignment_2cam224_1e-4"),
        ("sim_robotwin", "robotwin_stage3_alignment_3cam384_1e-4"),
    ],
)
def test_stage3_eval_configs_accept_declared_artifact_overrides(
    config_name: str,
    task_name: str,
) -> None:
    adapter_path = "/tmp/formal-stage3/adapter.pt"
    adapter_sha256 = "a" * 64
    base_sha256 = "b" * 64
    data_manifest_sha256 = "c" * 64
    training_contract_sha256 = "d" * 64

    # None of these overrides uses Hydra's `+` syntax. Successful composition
    # therefore also guards that every Stage 3 endpoint field is declared in
    # the base simulation config rather than being appended ad hoc.
    with initialize_config_dir(
        config_dir=str(REPO_ROOT / "configs"),
        version_base="1.3",
    ):
        cfg = compose(
            config_name=config_name,
            overrides=[
                f"task={task_name}",
                f"EVALUATION.stage3_adapter_path={adapter_path}",
                f"EVALUATION.stage3_adapter_sha256={adapter_sha256}",
                f"EVALUATION.stage3_base_sha256={base_sha256}",
                (
                    "EVALUATION.stage3_data_manifest_sha256="
                    f"{data_manifest_sha256}"
                ),
                (
                    "EVALUATION.stage3_training_contract_sha256="
                    f"{training_contract_sha256}"
                ),
                "EVALUATION.stage3_global_step=200",
            ],
        )

    assert cfg.model._target_ == ALIGNED_MODEL_TARGET
    assert cfg.model.load_text_encoder is True
    assert cfg.EVALUATION.stage3_adapter_path == adapter_path
    assert cfg.EVALUATION.stage3_adapter_sha256 == adapter_sha256
    assert cfg.EVALUATION.stage3_base_sha256 == base_sha256
    assert cfg.EVALUATION.stage3_data_manifest_sha256 == data_manifest_sha256
    assert (
        cfg.EVALUATION.stage3_training_contract_sha256
        == training_contract_sha256
    )
    assert cfg.EVALUATION.stage3_global_step == 200


def test_robotwin_append_override_preserves_path_and_sha_as_strings() -> None:
    adapter_path = Path("/tmp/formal stage3/adapter export.pt")
    # An all-numeric value catches accidental forwarding as an integer.
    adapter_sha256 = "0" * 64
    overrides: list[str] = []

    _append_override(overrides, "stage3_adapter_path", adapter_path)
    _append_override(overrides, "stage3_adapter_sha256", adapter_sha256)

    assert overrides[0::2] == [
        "--stage3_adapter_path",
        "--stage3_adapter_sha256",
    ]
    decoded_path = ast.literal_eval(overrides[1])
    decoded_sha256 = ast.literal_eval(overrides[3])
    assert isinstance(decoded_path, str)
    assert decoded_path == str(adapter_path)
    assert isinstance(decoded_sha256, str)
    assert decoded_sha256 == adapter_sha256


@pytest.mark.parametrize(
    ("inherited", "gpu_id", "expected"),
    [
        (None, 2, "2"),
        ("3", 0, "3"),
        (
            "GPU-01234567-89ab-cdef-0123-456789abcdef",
            0,
            "GPU-01234567-89ab-cdef-0123-456789abcdef",
        ),
        ("4,7", 1, "7"),
    ],
)
def test_robotwin_gpu_selection_stays_within_inherited_allocation(
    monkeypatch: pytest.MonkeyPatch,
    inherited: str | None,
    gpu_id: int,
    expected: str,
) -> None:
    if inherited is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", inherited)

    assert _select_cuda_visible_device(gpu_id) == expected


def test_robotwin_gpu_selection_rejects_out_of_range_logical_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-only")

    with pytest.raises(ValueError, match="outside the inherited"):
        _select_cuda_visible_device(1)


def _compose_robotwin(task_name: str):
    with initialize_config_dir(
        config_dir=str(REPO_ROOT / "configs"),
        version_base="1.3",
    ):
        return compose(
            config_name="sim_robotwin",
            overrides=[f"task={task_name}"],
        )


def test_robotwin_aligned_endpoint_rejects_missing_adapter(tmp_path) -> None:
    cfg = _compose_robotwin("robotwin_stage3_alignment_3cam384_1e-4")

    with pytest.raises(ValueError, match="requires stage3_adapter_path"):
        deploy_policy._prepare_stage3_eval_artifacts(
            {},
            cfg,
            checkpoint_path=str(tmp_path / "base.pt"),
            dataset_stats_path=tmp_path / "dataset_stats.json",
        )


def test_robotwin_legacy_endpoint_rejects_orphan_stage3_identity(
    tmp_path,
) -> None:
    cfg = _compose_robotwin("robotwin_unified_shared_3cam_384_1e-4")

    with pytest.raises(ValueError, match="identity fields require"):
        deploy_policy._prepare_stage3_eval_artifacts(
            {"stage3_global_step": 200},
            cfg,
            checkpoint_path=str(tmp_path / "base.pt"),
            dataset_stats_path=tmp_path / "dataset_stats.json",
        )


def test_robotwin_policy_global_step_falls_back_past_official_none(
    tmp_path,
    monkeypatch,
) -> None:
    cfg = _compose_robotwin("robotwin_stage3_alignment_3cam384_1e-4")
    cfg.EVALUATION.stage3_adapter_path = str(tmp_path / "adapter.pt")
    cfg.EVALUATION.stage3_adapter_sha256 = "a" * 64
    cfg.EVALUATION.stage3_global_step = 200
    captured = {}

    def fake_inspect(**kwargs):
        captured.update(kwargs)
        return {"sentinel": True}

    monkeypatch.setattr(
        deploy_policy,
        "inspect_aligned_model_artifacts",
        fake_inspect,
    )
    identity = deploy_policy._prepare_stage3_eval_artifacts(
        {"stage3_global_step": None},
        cfg,
        checkpoint_path=str(tmp_path / "base.pt"),
        dataset_stats_path=tmp_path / "dataset_stats.json",
    )

    assert identity == {"sentinel": True}
    assert captured["expected_global_step"] == 200


def test_robotwin_stage3_endpoint_identity_receipt_is_atomic_and_complete(
    tmp_path,
) -> None:
    output_dir = tmp_path / "nested" / "endpoint"
    model_identity = {
        "alignment_export": {"sha256": "a" * 64},
        "base_checkpoint": {"sha256": "b" * 64},
    }
    evaluation_runtime = {
        "task_name": "pick_dual_bottles",
        "inference_mode": "w",
        "num_inference_steps": 2,
    }

    receipt_path = deploy_policy._write_stage3_endpoint_identity(
        str(output_dir),
        model_artifact_identity=model_identity,
        evaluation_runtime=evaluation_runtime,
    )

    assert receipt_path == (
        output_dir.resolve() / "stage3_endpoint_model_identity.json"
    )
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "kind": "stage3_endpoint_model_identity",
        "model_artifact_identity": model_identity,
        "evaluation_runtime": evaluation_runtime,
    }
    assert list(output_dir.glob(".*.tmp")) == []
