from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
from pathlib import Path

import pytest

import fastwam.gating.label_job as label_job_module
import fastwam.gating.source_guard as source_guard_module
from fastwam.alignment.data_identity import canonical_data_manifest_sha256
from fastwam.gating.artifacts import (
    build_label_artifact_context,
    build_label_contract,
    build_label_row_from_context,
)
from fastwam.gating.contracts import build_episode_split
from fastwam.gating.label_job import (
    enumerate_label_samples,
    plan_label_chunks,
    run_label_job,
)
from fastwam.gating.source_guard import (
    SourceDataDriftError,
    capture_selected_source_snapshot,
    make_source_stat_guard,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entry(path: Path, *, anchor: Path, role: str) -> dict:
    return {
        "role": role,
        "relative_path": path.relative_to(anchor).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _manifest(tmp_path: Path, *, selected_path: Path | None = None) -> tuple[dict, Path]:
    dataset_root = tmp_path / "dataset"
    cache_root = tmp_path / "cache"
    stats_root = tmp_path / "stats"
    dataset_root.mkdir(parents=True)
    cache_root.mkdir()
    stats_root.mkdir()
    selected = selected_path or (dataset_root / "selected.bin")
    if selected_path is None:
        selected.write_bytes(b"AAAA")
    cache = cache_root / "prompt.pt"
    stats = stats_root / "stats.json"
    cache.write_bytes(b"embedding")
    stats.write_bytes(b"stats")
    manifest = {
        "schema_version": 1,
        "kind": "stage3_libero_data_manifest",
        "sampling": {},
        "num_frames": 2,
        "dataset_roots": [
            {
                "dataset_index": 0,
                "root": str(dataset_root.resolve()),
                "selected_episodes": [0, 1],
                "num_frames": 2,
                "episode_boundaries": [
                    {"episode_index": 0, "from": 0, "to": 1, "length": 1},
                    {"episode_index": 1, "from": 1, "to": 2, "length": 1},
                ],
                "video_keys": [],
                "files": [_entry(selected, anchor=dataset_root, role="parquet")],
            }
        ],
        "text_embedding_cache": {
            "root": str(cache_root.resolve()),
            "files": [_entry(cache, anchor=cache_root, role="text_embedding")],
        },
        "normalization_stats": {
            "root": str(stats_root.resolve()),
            "files": [_entry(stats, anchor=stats_root, role="normalization_stats")],
        },
        "decoder": {},
    }
    manifest["manifest_sha256"] = canonical_data_manifest_sha256(manifest)
    return manifest, selected


def _context(manifest: dict):
    split = build_episode_split(
        manifest,
        validation_fraction=0.5,
        split_seed=7,
    )
    contract = build_label_contract(
        data_manifest=manifest,
        episode_split=split,
        base_checkpoint_sha256="a" * 64,
        adapter_checkpoint_sha256="b" * 64,
        normalization_stats_sha256="c" * 64,
        data_config_sha256="d" * 64,
        vae_sha256="e" * 64,
        label_runtime_config_sha256="f" * 64,
        git_identity={
            "commit": "1" * 40,
            "tracked_dirty": False,
            "untracked_source_files": [],
        },
        base_seed=42,
        num_seed_pairs=2,
        relative_margin=0.05,
        num_shards=1,
        chunk_size=8,
    )
    return build_label_artifact_context(
        contract=contract,
        data_manifest=manifest,
        episode_split=split,
    )


def test_snapshot_is_immutable_and_supports_stat_and_content_rechecks(tmp_path):
    manifest, _ = _manifest(tmp_path)
    snapshot = capture_selected_source_snapshot(manifest)

    assert snapshot.data_manifest_sha256 == manifest["manifest_sha256"]
    assert len(snapshot.files) == 3
    snapshot.check_stats()
    snapshot.check_content()
    with pytest.raises(FrozenInstanceError):
        snapshot.files[0].st_ino = 0


def test_full_recheck_detects_same_size_drift_present_at_capture(tmp_path):
    manifest, selected = _manifest(tmp_path)
    # Merge has no prior dataset validation: capture is cheap, then one full
    # recheck rejects same-size content drift against the manifest SHA.
    selected.write_bytes(b"BBBB")
    snapshot = capture_selected_source_snapshot(manifest)

    snapshot.check_stats()
    with pytest.raises(SourceDataDriftError, match="content SHA256 drifted"):
        snapshot.check_content()


def test_full_recheck_resolves_manifest_path_again_after_hashing(
    tmp_path,
    monkeypatch,
):
    manifest, selected_link = _manifest(tmp_path)
    anchor = Path(manifest["dataset_roots"][0]["root"])
    target_a = anchor / "first.bin"
    target_b = anchor / "second.bin"
    target_a.write_bytes(b"AAAA")
    target_b.write_bytes(b"BBBB")
    selected_link.unlink()
    selected_link.symlink_to(target_a.name)
    entry = manifest["dataset_roots"][0]["files"][0]
    entry.update(
        size_bytes=target_a.stat().st_size,
        sha256=_sha256(target_a),
    )
    manifest["manifest_sha256"] = canonical_data_manifest_sha256(manifest)
    snapshot = capture_selected_source_snapshot(manifest)
    selected_key = snapshot.files[0].key
    original_resolve = source_guard_module._resolve_current
    calls = 0

    def switch_after_hash(source):
        nonlocal calls
        calls += 1
        if calls == 2:
            selected_link.unlink()
            selected_link.symlink_to(target_b.name)
        return original_resolve(source)

    monkeypatch.setattr(
        source_guard_module,
        "_resolve_current",
        switch_after_hash,
    )
    with pytest.raises(SourceDataDriftError, match="resolved path drifted"):
        snapshot.check_content(keys=(selected_key,))
    assert calls == 2


def test_full_recheck_ends_with_a_complete_stat_sweep(tmp_path, monkeypatch):
    manifest, selected = _manifest(tmp_path)
    snapshot = capture_selected_source_snapshot(manifest)
    later_key = snapshot.files[1].key
    original_resolve = source_guard_module._resolve_current
    mutated = False

    def mutate_earlier_file_while_hashing_later(source):
        nonlocal mutated
        if source.key == later_key and not mutated:
            selected.write_bytes(b"BBBBB")
            mutated = True
        return original_resolve(source)

    monkeypatch.setattr(
        source_guard_module,
        "_resolve_current",
        mutate_earlier_file_while_hashing_later,
    )
    with pytest.raises(SourceDataDriftError, match="filesystem identity drifted"):
        snapshot.check_content()
    assert mutated


def test_label_job_drift_before_publish_leaves_no_chunk(tmp_path, monkeypatch):
    manifest, selected = _manifest(tmp_path / "source")
    context = _context(manifest)
    planned = enumerate_label_samples(context)
    # Formal generate brackets its existing full dataset validation with this
    # capture and a cheap stat recheck. check_content stands in for that one
    # full validation in this tiny source-only test.
    snapshot = capture_selected_source_snapshot(manifest)
    snapshot.check_content()
    snapshot.check_stats()
    guard = make_source_stat_guard(snapshot)

    class MutatingDataset:
        def __len__(self) -> int:
            return len(planned)

        def __getitem__(self, index: int) -> dict:
            if index == planned[0].global_sample_index:
                selected.write_bytes(b"BBBBB")
            return {"sample_identity": dict(planned[index].identity)}

    def fake_row(planned_sample, sample, **kwargs):
        del sample, kwargs
        return build_label_row_from_context(
            context=context,
            identity=planned_sample.identity,
            e0=1.0,
            e10=0.5,
            relative_gain=0.5,
            label=True,
            sample_weight=1.0,
            num_video_frames=5,
        )

    monkeypatch.setattr(label_job_module, "_row_for_sample", fake_row)
    plan = plan_label_chunks(
        context=context,
        output_dir=tmp_path / "chunks",
        chunk_size=8,
    )[0]
    with pytest.raises(SourceDataDriftError, match="filesystem identity drifted"):
        run_label_job(
            object(),
            MutatingDataset(),
            context=context,
            output_dir=tmp_path / "chunks",
            chunk_size=8,
            source_guard=guard,
        )
    assert not plan.path.exists()


def test_label_job_checks_guard_before_resuming_existing_chunk(
    tmp_path,
    monkeypatch,
):
    manifest, _ = _manifest(tmp_path / "source")
    context = _context(manifest)
    plans = plan_label_chunks(
        context=context,
        output_dir=tmp_path / "chunks",
        chunk_size=8,
    )
    plans[0].path.parent.mkdir(parents=True)
    plans[0].path.write_bytes(b"existing")
    calls = 0

    def reject_drift() -> None:
        nonlocal calls
        calls += 1
        raise SourceDataDriftError("synthetic drift")

    monkeypatch.setattr(
        label_job_module,
        "_load_existing_chunk",
        lambda *args, **kwargs: pytest.fail("guard must run before resume load"),
    )
    with pytest.raises(SourceDataDriftError, match="synthetic drift"):
        run_label_job(
            object(),
            [None, None],
            context=context,
            output_dir=tmp_path / "chunks",
            chunk_size=8,
            source_guard=reject_drift,
        )
    assert calls == 1
