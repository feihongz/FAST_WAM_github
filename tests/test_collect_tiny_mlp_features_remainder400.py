from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file

from experiments.libero.gate import collect_tiny_mlp_features as base
from experiments.libero.gate import collect_tiny_mlp_features_remainder400 as r400
from experiments.libero.gate import offline_tiny_mlp as v1
from experiments.libero.gate import offline_tiny_mlp_remainder400 as followup


ROOT = Path(__file__).parents[1]
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _committed_repo(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "remainder400@example.invalid")
    _git(root, "config", "user.name", "Remainder400 Test")
    source = root / "scientific.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    _git(root, "add", "scientific.py")
    _git(root, "commit", "-m", "base")
    return root, source


def _target_row(index: int, *, component: str = "remainder400") -> dict:
    component_order = index if component == "remainder400" else index
    source_index = (1000 if component == "remainder400" else 0) + index
    sample_id = f"libero_spatial_no_noops_lerobot/episode_{source_index:06d}/frame_000000"
    return {
        "selection_order": component_order,
        "sample_id": sample_id,
        "source_index": source_index,
        "suite": "libero_spatial",
        "dataset_id": "libero_spatial_no_noops_lerobot",
        "dataset_name": "libero_spatial_no_noops_lerobot",
        "episode_index": source_index,
        "frame_index": 0,
        "task_index": source_index % 10,
        "task": f"task {source_index % 10}",
        "target_id": f"{sample_id}/utility_target_v2",
        "target_sha256": base.sha256_json({"target": source_index}),
        "source_pilot_record_sha256": base.sha256_json({"pilot": source_index}),
        "input_hashes": {
            "combined": base.sha256_json({"input": source_index}),
        },
        "current_proprio": [0.0] * 8,
    }


def _fake_anchor_inputs(monkeypatch: pytest.MonkeyPatch) -> dict:
    # This test exercises the consumer-side Pilot-500/component/completion
    # projection. Target-V2's full row-schema validation is already covered by
    # its own suite and by load_sealed_followup_targets in production.
    monkeypatch.setattr(r400.target_v2, "validate_target_manifest", lambda _: None)
    monkeypatch.setattr(
        r400.target_v2, "_validate_targets_against_manifest", lambda *_: None
    )

    remainder = [_target_row(index) for index in range(400)]
    remainder_records_sha = r400._canonical_jsonl_sha256(remainder)
    remainder_compatibility = {"kind": "libero_demo_utility_target_v2"}
    remainder_manifest = {
        "schema_version": 1,
        "kind": "libero_demo_utility_target_v2",
        "compatibility": remainder_compatibility,
        "compatibility_fingerprint": base.sha256_json(remainder_compatibility),
        "source": {"manifest_sha256": SHA_A, "records_sha256": SHA_B},
        "targets": {"canonical_records_sha256": remainder_records_sha},
    }
    remainder_manifest_sha = SHA_C

    existing = [_target_row(index, component="existing100") for index in range(100)]
    combined = [*existing, *remainder]
    states = [
        r400._component_projection(
            row,
            order,
            "existing100" if order < 100 else "remainder400",
        )
        for order, row in enumerate(combined)
    ]
    selection_sha = base.sha256_json(states)
    combined_records_sha = r400._canonical_jsonl_sha256(combined)
    binding = {
        "count": 400,
        "manifest_sha256": remainder_manifest_sha,
        "targets_sha256": remainder_records_sha,
        "manifest_fingerprint": remainder_manifest["compatibility_fingerprint"],
        "source_manifest_sha256": SHA_A,
        "source_records_sha256": SHA_B,
    }
    combined_compatibility = {
        "schema_version": 1,
        "kind": r400.COMBINED_KIND,
        "target_base_seeds": list(r400.TARGET_BASE_SEEDS),
        "num_states": 500,
        "remainder400": binding,
        "combined_selection_sha256": selection_sha,
        "combined_targets_sha256": combined_records_sha,
    }
    combined_manifest = {
        "schema_version": 1,
        "kind": r400.COMBINED_KIND,
        "compatibility": combined_compatibility,
        "compatibility_fingerprint": base.sha256_json(combined_compatibility),
        "components": {"remainder400": binding},
        "selection": {
            "algorithm": "immutable-pilot500-order-existing100-plus-exact-remainder400-v1",
            "num_states": 500,
            "ordered_states": states,
            "ordered_states_sha256": selection_sha,
        },
        "targets": {
            "count": 500,
            "canonical_records_sha256": combined_records_sha,
        },
        "policy": {
            "independent_validation_seeds_excluded": [47, 48, 49, 50],
        },
    }
    combined_manifest_sha = SHA_D
    completion = {
        "schema_version": 1,
        "kind": r400.COMBINED_COMPLETION_KIND,
        "manifest_sha256": combined_manifest_sha,
        "targets_sha256": combined_records_sha,
        "target_count": 500,
        "manifest_fingerprint": combined_manifest["compatibility_fingerprint"],
    }
    completion["completion_sha256"] = base.sha256_json(completion)
    return {
        "remainder_manifest": remainder_manifest,
        "remainder_targets": remainder,
        "remainder_manifest_sha256": remainder_manifest_sha,
        "remainder_records_sha256": remainder_records_sha,
        "combined_manifest": combined_manifest,
        "combined_targets": combined,
        "combined_manifest_sha256": combined_manifest_sha,
        "combined_records_sha256": combined_records_sha,
        "combined_completion": completion,
        "combined_completion_file_sha256": SHA_A,
    }


def _projection_kwargs(collector) -> dict:
    return {
        "latent_channels": int(collector.visual.latent_channels),
        "pooled_height": int(collector.visual.pooled_height),
        "pooled_width": int(collector.visual.pooled_width),
        "visual_dim": int(collector.visual.projection_dim),
        "visual_seed": int(collector.visual.projection_seed),
        "context_dim": int(collector.instruction.context_dim),
        "instruction_dim": int(collector.instruction.projection_dim),
        "mean_seed": int(collector.instruction.mean_projection_seed),
        "rms_seed": int(collector.instruction.rms_projection_seed),
    }


def _minimal_feature_manifest(extractor: dict) -> dict:
    compatibility = {
        "schema_version": 1,
        "kind": base.BUNDLE_KIND,
        "feature_record_schema_version": 1,
        "feature_dimensions": dict(base.EXPECTED_DIMS),
        "extractor": extractor,
        "extractor_fingerprint": extractor["extractor_fingerprint"],
        "scientific_source_files": {"fake_source.py": SHA_A},
        "num_states": 400,
        "global_join_contract": {
            "keys": ["sample_id", "source_index"],
            "source_index_semantics": "global requested_sample_idx/source_sample_idx",
            "dataset_local_source_metadata.source_index_allowed_as_join": False,
        },
    }
    return {
        "schema_version": 1,
        "kind": base.BUNDLE_KIND,
        "compatibility": compatibility,
        "compatibility_fingerprint": base.sha256_json(compatibility),
        "outputs": {"features": {"shapes": {}}},
    }


def test_exact137_extractor_and_projection_hashes_match_target100_defaults():
    target100 = OmegaConf.load(ROOT / "configs/collect_libero_gate_features.yaml")
    remainder400 = OmegaConf.load(
        ROOT / "configs/collect_libero_gate_features_remainder400.yaml"
    )
    old = target100.FEATURE_COLLECTOR
    new = remainder400.FEATURE_COLLECTOR
    old_projections = base.build_projection_matrices(**_projection_kwargs(old))
    new_projections = base.build_projection_matrices(**_projection_kwargs(new))
    assert base.EXPECTED_DIMS == {"full": 137, "visual": 64, "instruction": 65, "proprio": 8}
    for key in old_projections:
        assert torch.equal(old_projections[key], new_projections[key])
        assert base.tensor_content_sha256(old_projections[key]) == base.tensor_content_sha256(
            new_projections[key]
        )
    old_extractor = base._extractor_contract(
        OmegaConf.to_container(old, resolve=False), old_projections
    )
    new_extractor = base._extractor_contract(
        OmegaConf.to_container(new, resolve=False), new_projections
    )
    assert old_extractor == new_extractor
    assert old_extractor["extractor_fingerprint"] == new_extractor[
        "extractor_fingerprint"
    ]


def _formal_numeric_cfg():
    loaded = OmegaConf.load(
        ROOT / "configs/collect_libero_gate_features_remainder400.yaml"
    )
    return OmegaConf.create(
        {
            "mixed_precision": "bf16",
            "FEATURE_COLLECTOR": OmegaConf.to_container(
                loaded.FEATURE_COLLECTOR, resolve=False
            ),
        }
    )


def test_frozen_numerical_contract_matches_all_formal_hashes():
    projections, extractor = r400._validate_exact_numerical_contract(
        _formal_numeric_cfg()
    )
    assert extractor["extractor_fingerprint"] == r400.FROZEN_EXTRACTOR_FINGERPRINT
    assert {
        key: base.tensor_content_sha256(value) for key, value in projections.items()
    } == r400.FROZEN_PROJECTION_SHA256


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("mixed_precision", "fp16"),
        ("FEATURE_COLLECTOR.visual.latent_channels", 47),
        ("FEATURE_COLLECTOR.visual.pooled_height", 3),
        ("FEATURE_COLLECTOR.visual.pooled_width", 5),
        ("FEATURE_COLLECTOR.visual.projection_dim", 63),
        ("FEATURE_COLLECTOR.visual.projection_seed", 20260814),
        ("FEATURE_COLLECTOR.instruction.context_dim", 4095),
        ("FEATURE_COLLECTOR.instruction.projection_dim", 31),
        ("FEATURE_COLLECTOR.instruction.mean_projection_seed", 20260815),
        ("FEATURE_COLLECTOR.instruction.rms_projection_seed", 20260816),
        ("FEATURE_COLLECTOR.proprio_dim", 7),
    ],
)
def test_every_frozen_numerical_override_is_rejected(path: str, value):
    cfg = _formal_numeric_cfg()
    OmegaConf.update(cfg, path, value)
    with pytest.raises((ValueError, AssertionError), match="frozen|bf16"):
        r400._validate_exact_numerical_contract(cfg)


def test_formal_switch_false_rejected_before_git_inspection(monkeypatch):
    cfg = _formal_numeric_cfg()
    cfg.FEATURE_COLLECTOR.require_clean_tracked_diff = False
    monkeypatch.setattr(
        r400,
        "_formal_git_snapshot",
        lambda: pytest.fail("Git inspection must not run for a disabled formal guard"),
    )
    with pytest.raises(ValueError, match="require_clean_tracked_diff=true"):
        r400._formal_entry_preflight(cfg)


def test_formal_git_guard_accepts_unrelated_untracked(tmp_path: Path):
    root, _ = _committed_repo(tmp_path)
    (root / "notes.txt").write_text("untracked notes\n", encoding="utf-8")
    result = r400._assert_formal_git_state(
        project_root=root, scientific_source_files=("scientific.py",)
    )
    assert set(result) == {"scientific.py"}


@pytest.mark.parametrize("staged", [False, True])
def test_formal_git_guard_rejects_unstaged_and_staged_changes(
    tmp_path: Path, staged: bool
):
    root, source = _committed_repo(tmp_path)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    if staged:
        _git(root, "add", "scientific.py")
    with pytest.raises(RuntimeError, match="tracked-clean"):
        r400._assert_formal_git_state(
            project_root=root, scientific_source_files=("scientific.py",)
        )


def test_formal_git_guard_rejects_untracked_scientific_source(tmp_path: Path):
    root, _ = _committed_repo(tmp_path)
    (root / "new_science.py").write_text("VALUE = 3\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not tracked"):
        r400._assert_formal_git_state(
            project_root=root, scientific_source_files=("new_science.py",)
        )


def test_formal_git_guard_rejects_head_mismatch_hidden_from_status(tmp_path: Path):
    root, source = _committed_repo(tmp_path)
    _git(root, "update-index", "--assume-unchanged", "scientific.py")
    source.write_text("VALUE = 9\n", encoding="utf-8")
    assert not _git(
        root, "status", "--porcelain=v1", "--untracked-files=no"
    ).stdout.strip()
    with pytest.raises(RuntimeError, match="byte-identical to HEAD"):
        r400._assert_formal_git_state(
            project_root=root, scientific_source_files=("scientific.py",)
        )


def test_combined_completion_seals_exact_remainder_rows(monkeypatch: pytest.MonkeyPatch):
    payload = _fake_anchor_inputs(monkeypatch)
    anchor = r400.validate_combined_remainder_anchor(**payload)
    assert anchor["remainder_state_count"] == 400
    assert anchor["combined_state_count"] == 500
    assert anchor["remainder_records_sha256"] == payload["remainder_records_sha256"]
    assert anchor["combined_completion_sha256"] == payload["combined_completion"][
        "completion_sha256"
    ]

    changed = copy.deepcopy(payload)
    changed["combined_targets"][173]["target_id"] += "/substituted"
    with pytest.raises(ValueError, match="canonical JSONL|ordered selection"):
        r400.validate_combined_remainder_anchor(**changed)


def test_combined_anchor_rejects_component_and_completion_tampering(
    monkeypatch: pytest.MonkeyPatch,
):
    payload = _fake_anchor_inputs(monkeypatch)
    changed = copy.deepcopy(payload)
    changed_binding = dict(
        changed["combined_manifest"]["components"]["remainder400"]
    )
    changed_binding["targets_sha256"] = SHA_D
    changed["combined_manifest"]["components"]["remainder400"] = changed_binding
    with pytest.raises(ValueError, match="binding"):
        r400.validate_combined_remainder_anchor(**changed)

    changed = copy.deepcopy(payload)
    changed["combined_completion"]["target_count"] = 499
    with pytest.raises(ValueError, match="payload digest"):
        r400.validate_combined_remainder_anchor(**changed)


def test_augmented_manifest_changes_bundle_not_numerical_extractor():
    cfg = OmegaConf.load(ROOT / "configs/collect_libero_gate_features_remainder400.yaml")
    projections = base.build_projection_matrices(**_projection_kwargs(cfg.FEATURE_COLLECTOR))
    extractor = base._extractor_contract(
        OmegaConf.to_container(cfg.FEATURE_COLLECTOR, resolve=False), projections
    )
    original = _minimal_feature_manifest(extractor)
    anchor = {
        "kind": "libero_gate_remainder400_combined_anchor",
        "remainder_manifest_sha256": SHA_A,
        "remainder_records_sha256": SHA_B,
        "combined_manifest_sha256": SHA_C,
        "combined_records_sha256": SHA_D,
        "combined_completion_file_sha256": SHA_A,
        "combined_completion_sha256": SHA_B,
        "combined_compatibility_fingerprint": SHA_C,
        "remainder_state_count": 400,
        "combined_state_count": 500,
    }
    augmented = r400.augment_remainder400_manifest(
        original,
        pilot500_anchor=anchor,
        followup_protocol={"path": str(ROOT / r400.FOLLOWUP_PROTOCOL_RELATIVE), "sha256": SHA_D},
        formal_git={
            "head": "1" * 40,
            "require_clean_tracked_diff": True,
            "scientific_source_files": {"fake_source.py": SHA_A},
        },
    )
    assert augmented["compatibility_fingerprint"] != original[
        "compatibility_fingerprint"
    ]
    assert augmented["compatibility"]["extractor_fingerprint"] == extractor[
        "extractor_fingerprint"
    ]
    assert augmented["compatibility"]["extractor"] == extractor
    assert augmented["compatibility"]["feature_dimensions"] == base.EXPECTED_DIMS
    assert augmented["compatibility"]["data_boundary"][
        "independent_validation_records_allowed"
    ] is False
    assert augmented["compatibility"]["formal_git_head"] == "1" * 40
    assert augmented["formal_git"]["scientific_source_files"] == {
        "fake_source.py": SHA_A
    }


def test_formal_400_bundle_is_accepted_by_v1_training_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _, extractor = r400._validate_exact_numerical_contract(_formal_numeric_cfg())
    manifest = _minimal_feature_manifest(extractor)
    manifest_path = tmp_path / base.MANIFEST_FILENAME
    base._atomic_write_bytes(manifest_path, base._serialize_json(manifest))

    targets = [_target_row(index) for index in range(400)]
    matrices = {
        key: torch.zeros(400, width, dtype=torch.float32)
        for key, width in base.EXPECTED_DIMS.items()
    }
    rows = [
        base.build_feature_record(
            target,
            {key: matrices[key][order] for key in base.TENSOR_KEYS},
            extractor_fingerprint=r400.FROZEN_EXTRACTOR_FINGERPRINT,
        )
        for order, target in enumerate(targets)
    ]
    index_path = tmp_path / base.INDEX_FILENAME
    feature_path = tmp_path / base.FEATURES_FILENAME
    base._atomic_write_bytes(index_path, base._serialize_jsonl(rows))
    base._atomic_save_safetensors(feature_path, matrices, metadata={"kind": base.BUNDLE_KIND})
    completion = base.completion_payload(
        manifest_path=manifest_path,
        index_path=index_path,
        features_path=feature_path,
        matrices=matrices,
        manifest_fingerprint=manifest["compatibility_fingerprint"],
        num_states=400,
    )
    base._atomic_write_bytes(
        tmp_path / base.COMPLETION_FILENAME, base._serialize_json(completion)
    )

    def fake_target_loader(*_, **kwargs):
        assert kwargs["expected_num_states"] == 400
        return {"kind": "fake_target_v2"}, targets

    monkeypatch.setattr(v1, "load_target_bundle", fake_target_loader)
    loaded = v1.load_training_inputs(
        target_dir=tmp_path / "target",
        target_manifest_sha256=SHA_A,
        target_targets_sha256=SHA_B,
        feature_dir=tmp_path,
        feature_completion_sha256=completion["completion_sha256"],
        expected_num_states=400,
    )
    assert len(loaded.targets) == len(loaded.features.index) == 400
    accepted = followup._extractor_contract(loaded.features)
    assert v1.canonical_json(accepted) == v1.canonical_json(extractor)


def test_unsealed_recovery_cleans_only_crash_orphans(tmp_path: Path):
    output = tmp_path / "output"
    progress = output / ".rows"
    progress.mkdir(parents=True)
    targets = [{"selection_order": 0}, {"selection_order": 1}]
    json0, tensor0 = base._progress_paths(progress, 0)
    json1, tensor1 = base._progress_paths(progress, 1)
    tensor0.write_bytes(b"crashed-before-json")
    json1.write_text("{}\n", encoding="utf-8")
    tensor1.write_bytes(b"complete-pair")
    (progress / ".000000.json.crash").write_bytes(b"temporary")
    (output / base.INDEX_FILENAME).write_bytes(b"public-index-orphan")
    (output / base.FEATURES_FILENAME).write_bytes(b"public-features-orphan")
    (output / r400.PENDING_COMPLETION_FILENAME).write_bytes(b"private-pending-seal")
    (output / f".{base.FEATURES_FILENAME}.crash").write_bytes(b"temporary")

    recovered = r400._recover_unsealed_output(
        output_dir=output,
        progress_dir=progress,
        targets=targets,
    )
    assert recovered["sealed"] is False
    assert not json0.exists() and not tensor0.exists()
    assert json1.exists() and tensor1.exists()
    assert not (output / base.INDEX_FILENAME).exists()
    assert not (output / base.FEATURES_FILENAME).exists()
    assert not (output / r400.PENDING_COMPLETION_FILENAME).exists()
    assert r400.PENDING_COMPLETION_FILENAME in recovered["removed"]
    assert not any(path.name.startswith(".") for path in progress.iterdir())


def test_snapshot_has_one_canonical_vae_binding():
    artifact = {"path": "/artifact", "sha256": SHA_A}
    tree = {
        "path": "/tree",
        "sha256": SHA_B,
        "file_count": 1,
        "total_size_bytes": 2,
    }
    snapshot = r400._content_snapshot(
        checkpoint=artifact,
        stats=artifact,
        vae=artifact,
        dataset_sources=[tree],
        context_cache=tree,
        phase25_manifest_sha256=SHA_C,
        formal_git={
            "head": "1" * 40,
            "require_clean_tracked_diff": True,
            "scientific_source_files": {"science.py": SHA_D},
        },
        pilot500_anchor={"kind": "anchor"},
        followup_protocol_sha256=SHA_D,
    )
    assert snapshot["vae_sha256"] == SHA_A
    assert "vae" not in snapshot
    assert list(snapshot).count("vae_sha256") == 1


class _FakeDataset:
    def __getitem__(self, index: int) -> int:
        return int(index)


class _FakeVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def encode(self, images, *, device, tiled):
        assert len(images) == 1 and images[0].shape[1] == 1 and tiled is False
        value = images[0].float().mean().cpu()
        latent = torch.arange(48 * 4 * 8, dtype=torch.float32).reshape(1, 48, 1, 4, 8)
        return latent + value


def test_cpu_fake_collection_is_atomic_resumable_and_rechecks_all_live_states(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkpoint = tmp_path / "checkpoint.pt"
    stats = tmp_path / "stats.json"
    phase = tmp_path / "phase25.json"
    checkpoint.write_bytes(b"checkpoint")
    stats.write_text("{}\n")
    phase.write_text("{}\n")
    data_dirs = []
    for index in range(4):
        path = tmp_path / f"dataset_{index}"
        path.mkdir()
        data_dirs.append(str(path))
    cache = tmp_path / "context_cache"
    cache.mkdir()
    remainder_dir = tmp_path / "remainder"
    remainder_dir.mkdir()
    output = tmp_path / "features"

    targets = [_target_row(index) for index in range(400)]
    target_compatibility = {"kind": "fake_target_v2", "num_states": 400}
    target_manifest = {
        "compatibility": target_compatibility,
        "compatibility_fingerprint": base.sha256_json(target_compatibility),
        "source": {"checkpoint_sha256": SHA_A, "dataset_stats_sha256": SHA_A, "vae_sha256": SHA_A},
    }
    anchor = {
        "kind": "libero_gate_remainder400_combined_anchor",
        "remainder_manifest_sha256": SHA_A,
        "remainder_records_sha256": SHA_B,
        "remainder_compatibility_fingerprint": target_manifest[
            "compatibility_fingerprint"
        ],
        "combined_manifest_sha256": SHA_C,
        "combined_records_sha256": SHA_D,
        "combined_completion_file_sha256": SHA_A,
        "combined_completion_sha256": SHA_B,
        "combined_compatibility_fingerprint": SHA_C,
        "remainder_state_count": 400,
        "combined_state_count": 500,
    }
    phase_manifest = {
        "compatibility": {},
        "compatibility_fingerprint": SHA_D,
    }

    cfg = OmegaConf.create(
        {
            "ckpt": str(checkpoint),
            "mixed_precision": "bf16",
            # Mirrors the inherited train config's time-varying `${now}` run
            # directory. It is operational only and may change on resume.
            "output_dir": "./runs/train/smoke",
            "model": {},
            "data": {
                "train": {
                    "dataset_dirs": data_dirs,
                    "text_embedding_cache_dir": str(cache),
                }
            },
            "FEATURE_COLLECTOR": {
                "remainder_target_v2_dir": str(remainder_dir),
                "expected_remainder_manifest_sha256": SHA_A,
                "expected_remainder_records_sha256": SHA_B,
                "combined_target_v2_dir": str(tmp_path / "combined"),
                "expected_combined_manifest_sha256": SHA_C,
                "expected_combined_records_sha256": SHA_D,
                "expected_combined_completion_file_sha256": SHA_A,
                "expected_followup_protocol_sha256": SHA_D,
                "phase25_manifest_path": str(phase),
                "expected_phase25_manifest_sha256": SHA_C,
                "output_dir": str(output),
                "dataset_stats_path": str(stats),
                "device": "cpu",
                "resume": True,
                "require_clean_tracked_diff": True,
                "expected_num_states": 400,
                "max_new_rows": 13,
                "visual": {
                    "latent_channels": 48,
                    "pooled_height": 2,
                    "pooled_width": 4,
                    "projection_dim": 64,
                    "projection_seed": 20260815,
                },
                "instruction": {
                    "context_dim": 6,
                    "projection_dim": 32,
                    "mean_projection_seed": 20260816,
                    "rms_projection_seed": 20260817,
                },
                "proprio_dim": 8,
            },
        }
    )

    monkeypatch.setattr(
        r400,
        "load_sealed_followup_targets",
        lambda _: (remainder_dir, target_manifest, targets, anchor),
    )
    monkeypatch.setattr(base, "_load_phase25_manifest", lambda *_, **__: phase_manifest)
    monkeypatch.setattr(
        r400,
        "_stable_file_provenance",
        lambda path, **_: {"path": str(path), "sha256": SHA_A, "size_bytes": 1},
    )
    monkeypatch.setattr(
        r400,
        "_resolve_vae_artifact",
        lambda _: {"path": str(tmp_path / "vae.pt"), "sha256": SHA_A, "size_bytes": 1},
    )

    def fake_tree(path, **_):
        return {
            "path": str(Path(path).resolve()),
            "sha256": SHA_A,
            "file_count": 1,
            "total_size_bytes": 1,
        }

    monkeypatch.setattr(r400, "_directory_tree_provenance", fake_tree)
    monkeypatch.setattr(r400, "_scientific_data_config", lambda _: {})
    monkeypatch.setattr(r400, "_assert_artifacts_match_followup", lambda **_: None)
    monkeypatch.setattr(
        r400,
        "_verify_followup_protocol",
        lambda _: {"path": str(ROOT / r400.FOLLOWUP_PROTOCOL_RELATIVE), "sha256": SHA_D},
    )
    formal_git = {
        "head": "1" * 40,
        "require_clean_tracked_diff": True,
        "scientific_source_files": {"fake_source.py": SHA_A},
    }
    monkeypatch.setattr(r400, "_formal_git_snapshot", lambda: copy.deepcopy(formal_git))
    fake_projections = base.build_projection_matrices(
        latent_channels=48,
        pooled_height=2,
        pooled_width=4,
        visual_dim=64,
        visual_seed=20260815,
        context_dim=6,
        instruction_dim=32,
        mean_seed=20260816,
        rms_seed=20260817,
    )
    fake_extractor = base._extractor_contract(
        {
            "visual": OmegaConf.to_container(
                cfg.FEATURE_COLLECTOR.visual, resolve=True
            ),
            "instruction": OmegaConf.to_container(
                cfg.FEATURE_COLLECTOR.instruction, resolve=True
            ),
            "proprio_dim": 8,
        },
        fake_projections,
    )
    monkeypatch.setattr(
        r400,
        "_validate_exact_numerical_contract",
        lambda _: (fake_projections, fake_extractor),
    )
    monkeypatch.setattr(
        r400,
        "_instantiate_dataset_and_contract",
        lambda *_, **__: (_FakeDataset(), [], {}),
    )
    monkeypatch.setattr(
        r400,
        "_load_frozen_vae",
        lambda **_: (_FakeVAE(), torch.float32),
    )
    live_checks = {"count": 0}

    def fake_live_state(*, sample, target, ranges, task_tables):
        assert int(sample) == int(target["source_index"])
        live_checks["count"] += 1
        return SimpleNamespace(
            input_image=torch.full((3, 4, 8), float(sample % 7), dtype=torch.float32),
            context=torch.tensor(
                [[1, 2, 3, 4, 5, 6], [0, 0, 0, 0, 0, 0]], dtype=torch.float32
            ),
            proprio=torch.arange(8, dtype=torch.float32),
        )

    monkeypatch.setattr(base, "_validate_live_state", fake_live_state)
    rehashes = {"count": 0, "fail_on": 3}

    def fake_rehash(*_, **__):
        rehashes["count"] += 1
        if rehashes["count"] == rehashes["fail_on"]:
            assert (output / r400.PENDING_COMPLETION_FILENAME).is_file()
            assert not (output / base.COMPLETION_FILENAME).exists()
            raise ValueError("simulated post-seal input mutation")

    monkeypatch.setattr(r400, "_rehash_all_inputs", fake_rehash)

    partial = r400.collect(cfg)
    assert partial == {
        "num_states": 400,
        "existing": 0,
        "new": 13,
        "progress_rows": 13,
        "complete": False,
        "output_dir": str(output.resolve()),
    }
    assert len(list((output / ".rows").glob("*.json"))) == 13
    assert len(list((output / ".rows").glob("*.safetensors"))) == 13
    assert not (output / base.INDEX_FILENAME).exists()
    assert not (output / base.FEATURES_FILENAME).exists()
    assert not (output / base.COMPLETION_FILENAME).exists()
    assert live_checks["count"] == 400

    # Simulate process death after the atomic progress tensor rename but before
    # its JSON pair. Resume must delete this orphan and recompute row 13.
    _, orphan_tensor = base._progress_paths(output / ".rows", 13)
    orphan_tensor.write_bytes(b"crashed-progress-tensor")
    cfg.FEATURE_COLLECTOR.max_new_rows = None
    cfg.output_dir = "./runs/train/resume"
    with pytest.raises(ValueError, match="post-seal input mutation"):
        r400.collect(cfg)
    assert live_checks["count"] == 800
    assert rehashes["count"] == 3
    assert len(list((output / ".rows").glob("*.json"))) == 400
    assert len(list((output / ".rows").glob("*.safetensors"))) == 400
    assert (output / base.INDEX_FILENAME).exists()
    assert (output / base.FEATURES_FILENAME).exists()
    # The public seal never exists before the required post-build rehash.
    assert not (output / base.COMPLETION_FILENAME).exists()
    assert (output / r400.PENDING_COMPLETION_FILENAME).exists()

    # Resume removes the kill-window pending seal and public pair, then
    # deterministically rebuilds from all 400 complete progress pairs.
    rehashes["fail_on"] = -1
    completed = r400.collect(cfg)
    assert completed["complete"] is True
    assert completed["num_states"] == completed["progress_rows"] == 400
    assert completed["new"] == 0
    assert live_checks["count"] == 1200
    assert rehashes["count"] == 5
    assert not (output / r400.PENDING_COMPLETION_FILENAME).exists()
    completion = base.validate_completion(output)
    assert completion["num_states"] == 400
    tensors = load_file(str(output / base.FEATURES_FILENAME), device="cpu")
    assert {key: tuple(value.shape) for key, value in tensors.items()} == {
        "full": (400, 137),
        "visual": (400, 64),
        "instruction": (400, 65),
        "proprio": (400, 8),
    }
    index_rows = [
        json.loads(line)
        for line in (output / base.INDEX_FILENAME).read_text().splitlines()
    ]
    assert len(index_rows) == 400
    assert not ({"utility", "e0", "efull", "valid_length"} & set(index_rows[0]))

    # A seal flips recovery to strict validation: public tampering is rejected
    # and never auto-deleted or regenerated.
    (output / base.INDEX_FILENAME).write_text("{\"tampered\":true}\n")
    with pytest.raises(ValueError, match="feature_index_sha256"):
        r400.collect(cfg)
    assert live_checks["count"] == 1200
    assert (output / base.COMPLETION_FILENAME).exists()


def test_config_exposes_no_original100_or_validation_data_surface():
    cfg = OmegaConf.load(ROOT / "configs/collect_libero_gate_features_remainder400.yaml")
    keys = {str(key).lower() for key in cfg.FEATURE_COLLECTOR.keys()}
    assert int(cfg.FEATURE_COLLECTOR.expected_num_states) == 400
    assert cfg.FEATURE_COLLECTOR.require_clean_tracked_diff is True
    assert "remainder_target_v2_dir" in keys
    assert "combined_target_v2_dir" in keys
    assert "expected_combined_completion_file_sha256" in keys
    assert "expected_followup_protocol_sha256" in keys
    assert not any("validation" in key or "original100" in key for key in keys)
    source = (
        ROOT / "experiments/libero/gate/collect_tiny_mlp_features_remainder400.py"
    ).read_text(encoding="utf-8")
    assert "collect_demo_utility_target_v2_validation" not in source
    assert "load_validation" not in source
