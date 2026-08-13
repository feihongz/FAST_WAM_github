from __future__ import annotations

from pathlib import Path
import subprocess

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from experiments.libero.gate import collect_demo_utility as single
from experiments.libero.gate import collect_demo_utility_target_v2_pilot500 as collector_module
from experiments.libero.gate.collect_demo_utility_target_v2_pilot500 import (
    _assert_formal_git_state,
    _prepare_output,
    _rehash_bound_components,
    _validate_resume_state,
    _validate_config,
)
from experiments.libero.gate.collect_demo_utility_multiseed import collect_replicate_grid


class CountingDataset:
    def __init__(self):
        self.calls = []

    def __getitem__(self, source_index):
        self.calls.append(source_index)
        return {"source_index": source_index}


def test_remainder_grid_reuses_seed42_and_infers_only_43_to_46():
    dataset = CountingDataset()
    selected = [
        {"source_index": 11, "selection_order": 0, "utility": 0.1,
         "source_metadata": {"requested_sample_idx": 11}},
        {"source_index": 22, "selection_order": 1, "utility": -0.1,
         "source_metadata": {"requested_sample_idx": 22}},
    ]
    calls = []
    written = []

    def infer(model, sample, pilot, base_seed):
        calls.append((sample["source_index"], base_seed))
        return {"utility": base_seed}

    summary = collect_replicate_grid(
        dataset=dataset,
        model=object(),
        selected_records=selected,
        base_seeds=(42, 43, 44, 45, 46),
        reuse_base_seed=42,
        existing_keys=set(),
        infer_record=infer,
        finalize_record=lambda utility, pilot, index, seed, sample: {
            "source_index": sample["source_index"],
            "replicate_index": index,
            "base_seed": seed,
            "utility": utility["utility"],
        },
        write_record=written.append,
    )
    assert dataset.calls == [11, 22]
    assert calls == [
        (11, 43), (11, 44), (11, 45), (11, 46),
        (22, 43), (22, 44), (22, 45), (22, 46),
    ]
    assert summary == {"new": 10, "reused": 2, "inferred": 8, "errors": 0}


def _config():
    dataset_root = "/datasets"
    values = {
        "eval_num_inference_steps": 10,
        "ckpt": "/tmp/checkpoint.pt",
        "data": {"train": {
            "num_frames": 33,
            "action_video_freq_ratio": 4,
            "dataset_dirs": [
                f"{dataset_root}/libero_spatial_no_noops_lerobot",
                f"{dataset_root}/libero_object_no_noops_lerobot",
                f"{dataset_root}/libero_goal_no_noops_lerobot",
                f"{dataset_root}/libero_10_no_noops_lerobot",
            ],
            "text_embedding_cache_dir": "/cache/libero",
        }},
        "model": {},
        "COLLECTOR": {
            "pilot_dir": "/pilot", "phase25_dir": "/phase25",
            "existing_target_v2_dir": "/target100", "output_dir": "/out",
            "remainder_target_v2_dir": "/remainder", "combined_target_v2_dir": "/combined",
            "expected_pilot_manifest_sha256": "1" * 64,
            "expected_pilot_records_sha256": "2" * 64,
            "expected_phase25_manifest_sha256": "3" * 64,
            "expected_phase25_records_sha256": "4" * 64,
            "expected_phase25_selection_plan_sha256": "5" * 64,
            "expected_existing_target_v2_manifest_sha256": "6" * 64,
            "expected_existing_target_v2_targets_sha256": "7" * 64,
            "num_states": 400, "expected_pilot_count": 500,
            "expected_phase25_count": 100, "expected_pilot_base_seed": 42,
            "replicate_base_seeds": [42,43,44,45,46],
            "reuse_base_seed": 42, "require_clean_tracked_diff": True,
            "continue_on_error": False, "max_errors": 1,
            "num_inference_steps": 10, "full_prefix_steps": 10,
            "num_video_frames": 9, "rand_device": "cpu", "sigma_shift": None,
            "tiled": False, "force_custom_prefix": True,
        },
    }
    return OmegaConf.create(values)


def test_formal_config_is_frozen_to_400_states_and_fail_fast():
    cfg = _config()
    assert _validate_config(cfg) == (42, 43, 44, 45, 46)
    cfg.COLLECTOR.num_states = 399
    with pytest.raises(ValueError, match="frozen at 400"):
        _validate_config(cfg)
    cfg = _config()
    cfg.COLLECTOR.continue_on_error = True
    with pytest.raises(ValueError, match="first error"):
        _validate_config(cfg)
    cfg = _config()
    cfg.COLLECTOR.require_clean_tracked_diff = False
    with pytest.raises(ValueError, match="clean tracked diff"):
        _validate_config(cfg)


def test_formal_config_rejects_missing_relative_or_reordered_data_paths():
    cfg = _config()
    cfg.data.train.dataset_dirs = None
    with pytest.raises(ValueError, match="explicitly contain four"):
        _validate_config(cfg)
    cfg = _config()
    cfg.data.train.dataset_dirs[0] = "relative/libero_spatial_no_noops_lerobot"
    with pytest.raises(ValueError, match="absolute paths"):
        _validate_config(cfg)
    cfg = _config()
    cfg.data.train.dataset_dirs = list(reversed(cfg.data.train.dataset_dirs))
    with pytest.raises(ValueError, match="suite order"):
        _validate_config(cfg)
    cfg = _config()
    cfg.data.train.text_embedding_cache_dir = None
    with pytest.raises(ValueError, match="text_embedding_cache_dir.*explicitly"):
        _validate_config(cfg)


def test_documented_formal_data_overrides_compose_and_validate():
    root = Path(__file__).resolve().parents[1]
    dataset_dirs = [
        "/root/feihong/FastWAM/datasets/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot",
        "/root/feihong/FastWAM/datasets/libero_mujoco3.3.2/libero_object_no_noops_lerobot",
        "/root/feihong/FastWAM/datasets/libero_mujoco3.3.2/libero_goal_no_noops_lerobot",
        "/root/feihong/FastWAM/datasets/libero_mujoco3.3.2/libero_10_no_noops_lerobot",
    ]
    cache = "/root/feihong/FAST_WAM_github/data/text_embeds_cache/libero"
    doc = (root / "docs/GATE_UTILITY_TARGET_V2_PILOT500_EXPANSION.md").read_text(
        encoding="utf-8"
    )
    documented_dataset_override = f"data.train.dataset_dirs=[{','.join(dataset_dirs)}]"
    assert documented_dataset_override in doc
    assert f"data.train.text_embedding_cache_dir={cache}" in doc

    with initialize_config_dir(
        config_dir=str((root / "configs").resolve()), version_base="1.3"
    ):
        cfg = compose(
            config_name="collect_libero_demo_utility_target_v2_pilot500.yaml",
            overrides=[
                f"data.train.dataset_dirs=[{','.join(dataset_dirs)}]",
                f"data.train.text_embedding_cache_dir={cache}",
            ],
        )
    assert list(cfg.data.train.dataset_dirs) == dataset_dirs
    assert str(cfg.data.train.text_embedding_cache_dir) == cache
    assert all(Path(value).is_absolute() for value in cfg.data.train.dataset_dirs)


def _git(root, *args):
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    )


def _committed_repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "pilot500@example.invalid")
    _git(root, "config", "user.name", "Pilot500 Test")
    source = root / "scientific.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    _git(root, "add", "scientific.py")
    _git(root, "commit", "-m", "base")
    return root, source


def test_formal_git_guard_accepts_head_identical_science_and_unrelated_untracked(tmp_path):
    root, _ = _committed_repo(tmp_path)
    (root / "notes.txt").write_text("untracked notes\n", encoding="utf-8")
    result = _assert_formal_git_state(
        project_root=root, scientific_source_files=("scientific.py",)
    )
    assert set(result) == {"scientific.py"}


@pytest.mark.parametrize("staged", [False, True])
def test_formal_git_guard_rejects_unstaged_and_staged_tracked_changes(tmp_path, staged):
    root, source = _committed_repo(tmp_path)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    if staged:
        _git(root, "add", "scientific.py")
    with pytest.raises(RuntimeError, match="tracked-clean"):
        _assert_formal_git_state(
            project_root=root, scientific_source_files=("scientific.py",)
        )


def test_formal_git_guard_rejects_untracked_scientific_source(tmp_path):
    root, _ = _committed_repo(tmp_path)
    (root / "new_science.py").write_text("VALUE = 3\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not tracked"):
        _assert_formal_git_state(
            project_root=root, scientific_source_files=("new_science.py",)
        )


def test_resume_state_accepts_partial_and_rejects_sealed_pending_or_errors(tmp_path):
    _, records, errors, completion = _prepare_output(tmp_path / "run", resume=True)
    records.write_text('{"partial":true}\n', encoding="utf-8")
    _validate_resume_state(
        errors_path=errors, completion_path=completion, pending_keys={(1, 2)}
    )

    completion.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sealed expansion.*pending"):
        _validate_resume_state(
            errors_path=errors, completion_path=completion, pending_keys={(1, 2)}
        )

    completion.unlink()
    errors.write_text('{"error":"boom"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="contains errors"):
        _validate_resume_state(
            errors_path=errors, completion_path=completion, pending_keys=set()
        )


def _bound_component_fixture(tmp_path, monkeypatch):
    files = {}
    for name in (
        "pilot_manifest", "pilot_records", "phase25_manifest", "phase25_records",
        "target_manifest", "target_targets", "checkpoint", "stats", "vae",
    ):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(f"{name}-bytes".encode())
        files[name] = path
    dataset = tmp_path / "dataset"
    cache = tmp_path / "cache"
    dataset.mkdir()
    cache.mkdir()
    (dataset / "data.bin").write_bytes(b"dataset")
    (cache / "embedding.bin").write_bytes(b"cache")
    dataset_artifact = single._directory_tree_provenance(dataset, label="dataset")
    dataset_artifact.update({"dataset_index": 0, "dataset_name": "dataset"})
    cache_artifact = single._directory_tree_provenance(cache, label="cache")
    file_hash = {name: single._sha256_file(path) for name, path in files.items()}
    scientific = {"collector.py": "a" * 64}
    monkeypatch.setattr(
        collector_module, "_assert_formal_git_state", lambda: dict(scientific)
    )
    compatibility = {
        "pilot_manifest_sha256": file_hash["pilot_manifest"],
        "pilot_records_sha256": file_hash["pilot_records"],
        "phase25_manifest_sha256": file_hash["phase25_manifest"],
        "phase25_records_sha256": file_hash["phase25_records"],
        "existing_target_v2_manifest_sha256": file_hash["target_manifest"],
        "existing_target_v2_targets_sha256": file_hash["target_targets"],
        "checkpoint_sha256": file_hash["checkpoint"],
        "dataset_stats_sha256": file_hash["stats"],
        "vae_sha256": file_hash["vae"],
    }
    manifest = {
        "compatibility": compatibility,
        "pilot": {
            "manifest_path": str(files["pilot_manifest"]),
            "records_path": str(files["pilot_records"]),
        },
        "excluded_phase25": {
            "manifest_path": str(files["phase25_manifest"]),
            "records_path": str(files["phase25_records"]),
        },
        "existing_target_v2": {
            "manifest_path": str(files["target_manifest"]),
            "targets_path": str(files["target_targets"]),
        },
        "artifacts": {
            "checkpoint": {"path": str(files["checkpoint"])},
            "dataset_stats": {"path": str(files["stats"])},
            "vae": {"path": str(files["vae"])},
        },
        "scientific_source_files": scientific,
    }
    return manifest, dataset_artifact, cache_artifact, files, dataset, cache


@pytest.mark.parametrize(
    "mutated",
    [
        "pilot_manifest", "pilot_records", "phase25_manifest", "phase25_records",
        "target_manifest", "target_targets", "checkpoint", "stats", "vae",
        "dataset", "cache", "scientific",
    ],
)
def test_preseal_rehash_rejects_each_mutated_scientific_input(
    tmp_path, monkeypatch, mutated
):
    manifest, dataset_artifact, cache_artifact, files, dataset, cache = (
        _bound_component_fixture(tmp_path, monkeypatch)
    )
    if mutated in files:
        files[mutated].write_bytes(b"mutated")
    elif mutated == "dataset":
        (dataset / "data.bin").write_bytes(b"mutated")
    elif mutated == "cache":
        (cache / "embedding.bin").write_bytes(b"mutated")
    else:
        monkeypatch.setattr(
            collector_module, "_assert_formal_git_state",
            lambda: {"collector.py": "b" * 64},
        )
    with pytest.raises(RuntimeError, match="changed"):
        _rehash_bound_components(
            manifest=manifest,
            dataset_source_artifacts=[dataset_artifact],
            context_cache_artifact=cache_artifact,
        )


def test_preseal_rehash_returns_actual_bound_snapshot(tmp_path, monkeypatch):
    manifest, dataset_artifact, cache_artifact, _, _, _ = _bound_component_fixture(
        tmp_path, monkeypatch
    )
    snapshot = _rehash_bound_components(
        manifest=manifest,
        dataset_source_artifacts=[dataset_artifact],
        context_cache_artifact=cache_artifact,
    )
    assert snapshot["pilot"]["manifest_sha256"] == manifest["compatibility"][
        "pilot_manifest_sha256"
    ]
    assert snapshot["artifacts"]["dataset_sources"][0]["file_count"] == 1
