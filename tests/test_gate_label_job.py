from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import fastwam.gating.artifacts as gate_artifacts
import fastwam.gating.label_job as gate_label_job
from fastwam.alignment.checkpointing import canonical_json_sha256
from fastwam.alignment.data_identity import canonical_data_manifest_sha256
from fastwam.gating.artifacts import (
    COHORT_CHUNK_PLAN_ALGORITHM,
    build_label_artifact_context,
    build_label_contract,
    load_complete_label_chunk_from_context,
    shard_for_sample_id,
)
from fastwam.gating.contracts import build_episode_split
from fastwam.gating.inference import PairedActionRollouts
from fastwam.gating.label_job import (
    LabelJobDependencies,
    enumerate_label_samples,
    iter_label_chunks,
    iter_label_samples,
    plan_label_chunks,
    run_label_job,
)
from fastwam.gating.selection import (
    build_selection_artifacts,
    selected_rows_for_coverage,
)


def _data_manifest() -> dict:
    manifest = {
        "schema_version": 1,
        "kind": "stage3_libero_data_manifest",
        "sampling": {},
        "num_frames": 6,
        "dataset_roots": [
            {
                "dataset_index": 0,
                "root": "/data/a",
                "selected_episodes": [2, 5],
                "num_frames": 6,
                "episode_boundaries": [
                    {
                        "episode_index": 2,
                        "from": 0,
                        "to": 2,
                        "length": 2,
                    },
                    {
                        "episode_index": 5,
                        "from": 2,
                        "to": 6,
                        "length": 4,
                    },
                ],
                "video_keys": [],
                "files": [],
            }
        ],
        "text_embedding_cache": {},
        "normalization_stats": {},
        "decoder": {},
    }
    manifest["manifest_sha256"] = canonical_data_manifest_sha256(manifest)
    return manifest


def _context(*, num_shards: int = 2, chunk_size: int = 2):
    manifest = _data_manifest()
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
        vae_sha256="f" * 64,
        label_runtime_config_sha256="1" * 64,
        git_identity={
            "commit": "e" * 40,
            "tracked_dirty": False,
            "untracked_source_files": [],
        },
        base_seed=42,
        num_seed_pairs=2,
        relative_margin=0.05,
        num_shards=num_shards,
        chunk_size=chunk_size,
    )
    return build_label_artifact_context(
        contract=contract,
        data_manifest=manifest,
        episode_split=split,
    )


def _selection_context(
    *,
    num_shards: int = 2,
    chunk_size: int = 2,
    cohort_plan: bool = True,
):
    manifest = _data_manifest()
    selection = build_selection_artifacts(
        manifest,
        episode_strata=[
            {
                "dataset_index": 0,
                "episode_index": 2,
                "stratum_id": "task",
            },
            {
                "dataset_index": 0,
                "episode_index": 5,
                "stratum_id": "task",
            },
        ],
        validation_fraction=0.5,
        split_seed=7,
        selection_seed=11,
        max_temporal_bins=2,
        train_targets=(1, 2),
        validation_target=1,
        coverage_names=("pilot", "formal"),
    )
    contract_kwargs = {}
    if cohort_plan:
        contract_kwargs["chunk_plan_algorithm"] = COHORT_CHUNK_PLAN_ALGORITHM
    contract = build_label_contract(
        data_manifest=manifest,
        episode_split=selection.episode_split,
        base_checkpoint_sha256="a" * 64,
        adapter_checkpoint_sha256="b" * 64,
        normalization_stats_sha256="c" * 64,
        data_config_sha256="d" * 64,
        vae_sha256="f" * 64,
        label_runtime_config_sha256="1" * 64,
        git_identity={
            "commit": "e" * 40,
            "tracked_dirty": False,
            "untracked_source_files": [],
        },
        base_seed=42,
        num_seed_pairs=2,
        relative_margin=0.05,
        num_shards=num_shards,
        chunk_size=chunk_size,
        **contract_kwargs,
    )
    context = build_label_artifact_context(
        contract=contract,
        data_manifest=manifest,
        episode_split=selection.episode_split,
    )
    return context, selection


def _sample(identity) -> dict:
    return {
        "video": torch.zeros(3, 5, 16, 16),
        "action": torch.zeros(4, 3),
        "proprio": torch.zeros(4, 8),
        "context": torch.zeros(6, 16),
        "context_mask": torch.ones(6, dtype=torch.bool),
        "gate_context_mask": torch.ones(6, dtype=torch.bool),
        "action_is_pad": torch.zeros(4, dtype=torch.bool),
        "action_dim_is_pad": torch.zeros(3, dtype=torch.bool),
        "sample_identity": dict(identity),
    }


class RecordingDataset:
    def __init__(self, identities) -> None:
        self.samples = [_sample(sample.identity) for sample in identities]
        self.requests: list[int] = []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        self.requests.append(index)
        return self.samples[index]


class FakeRolloutRunner:
    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.calls: list[dict] = []
        self.fail_on_call = fail_on_call

    def __call__(
        self,
        model,
        sample,
        *,
        seeds,
        num_inference_steps,
        sigma_shift,
        rand_device,
        tiled,
    ) -> PairedActionRollouts:
        del model
        self.calls.append(
            {
                "identity": dict(sample["sample_identity"]),
                "seeds": tuple(seeds),
                "num_inference_steps": num_inference_steps,
                "sigma_shift": sigma_shift,
                "rand_device": rand_device,
                "tiled": tiled,
            }
        )
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise RuntimeError("synthetic rollout failure")
        action = sample["action"]
        shape = (len(seeds), 1, *action.shape)
        return PairedActionRollouts(
            action_wo=torch.ones(shape),
            action_w=torch.zeros(shape),
            seeds=tuple(seeds),
            action_horizon=action.shape[0],
            num_video_frames=sample["video"].shape[1],
            num_inference_steps=num_inference_steps,
        )


def _dependencies(runner) -> LabelJobDependencies:
    return LabelJobDependencies(run_rollouts=runner)


def test_plan_is_metadata_only_deterministic_and_fixed_by_shard(tmp_path):
    context = _context(num_shards=3)

    samples = enumerate_label_samples(context)
    plans = plan_label_chunks(
        context=context,
        output_dir=tmp_path,
        chunk_size=2,
    )

    assert [sample.global_sample_index for sample in samples] == list(range(6))
    assert plans == plan_label_chunks(
        context=context,
        output_dir=tmp_path,
        chunk_size=2,
    )
    with pytest.raises(ValueError, match="immutable label contract"):
        plan_label_chunks(
            context=context,
            output_dir=tmp_path,
            chunk_size=3,
        )
    flattened = [sample for plan in plans for sample in plan.samples]
    assert {sample.sample_id for sample in flattened} == {
        sample.sample_id for sample in samples
    }
    for plan in plans:
        assert 1 <= len(plan.samples) <= 2
        assert plan.planned_sample_ids == tuple(sorted(plan.planned_sample_ids))
        assert plan.path == (
            Path(tmp_path).resolve()
            / f"shard-{plan.shard_index:05d}"
            / f"chunk-{plan.chunk_index:08d}.json"
        )
        assert all(
            shard_for_sample_id(
                sample.sample_id,
                num_shards=context.contract["num_shards"],
            )
            == plan.shard_index
            for sample in plan.samples
        )
    with pytest.raises(TypeError):
        samples[0].identity["global_sample_index"] = 99
    assert list(tmp_path.iterdir()) == []


def test_streamed_label_enumeration_matches_compatibility_materialization():
    context = _context(num_shards=3)
    streamed = tuple(iter_label_samples(context))
    assert streamed == enumerate_label_samples(context)
    assert [sample.global_sample_index for sample in streamed] == list(range(6))


def test_streamed_chunk_plans_match_materialized_api_for_selected_shards(
    tmp_path,
):
    context = _context(num_shards=3)
    selected = [2, 0]
    materialized = plan_label_chunks(
        context=context,
        output_dir=tmp_path,
        chunk_size=2,
        shard_indices=selected,
    )
    streamed = tuple(
        iter_label_chunks(
            context=context,
            output_dir=tmp_path,
            chunk_size=2,
            shard_indices=selected,
        )
    )

    assert streamed == materialized
    assert [plan.shard_index for plan in streamed] == sorted(
        plan.shard_index for plan in streamed
    )
    assert all(len(plan.samples) <= 2 for plan in streamed)


def test_million_sample_plan_source_is_lazy_on_iterator_construction(
    tmp_path,
    monkeypatch,
):
    context = _context(num_shards=1)
    sample = enumerate_label_samples(context)[0]
    entered = False

    def million_synthetic_samples(_context):
        nonlocal entered
        entered = True
        for _ in range(1_000_000):
            yield sample

    monkeypatch.setattr(
        gate_label_job, "iter_label_samples", million_synthetic_samples
    )
    plans = iter_label_chunks(
        context=context,
        output_dir=tmp_path,
        chunk_size=2,
    )

    assert entered is False
    plans.close()
    assert entered is False


def test_streaming_plan_detects_duplicate_sample_ids_globally(
    tmp_path,
    monkeypatch,
):
    context = _context(num_shards=1)
    sample = enumerate_label_samples(context)[0]

    def duplicate_samples(_context):
        yield sample
        yield sample

    monkeypatch.setattr(gate_label_job, "iter_label_samples", duplicate_samples)
    with pytest.raises(RuntimeError, match="duplicate sample IDs"):
        tuple(
            iter_label_chunks(
                context=context,
                output_dir=tmp_path,
                chunk_size=2,
            )
        )


def test_subset_selection_arguments_and_chunk_plan_algorithm_are_fail_closed(
    tmp_path,
):
    context, selection = _selection_context()

    with pytest.raises(ValueError, match="must be provided together"):
        tuple(iter_label_samples(context, selection_artifacts=selection))
    with pytest.raises(ValueError, match="must be provided together"):
        tuple(iter_label_samples(context, coverage_tier="pilot"))
    with pytest.raises(ValueError, match="unsupported label chunk plan algorithm"):
        iter_label_chunks(
            context=context,
            output_dir=tmp_path,
            chunk_size=2,
        )

    legacy_plan_context, same_selection = _selection_context(cohort_plan=False)
    with pytest.raises(ValueError, match="unsupported label chunk plan algorithm"):
        tuple(
            iter_label_samples(
                legacy_plan_context,
                selection_artifacts=same_selection,
                coverage_tier="pilot",
            )
        )
    with pytest.raises(ValueError, match="unsupported label chunk plan algorithm"):
        iter_label_chunks(
            context=legacy_plan_context,
            output_dir=tmp_path,
            chunk_size=2,
            selection_artifacts=same_selection,
            coverage_tier="pilot",
        )


def test_subset_plan_exact_coverage_and_expansion_preserves_old_cohorts(tmp_path):
    context, selection = _selection_context(num_shards=3)
    pilot_rows = selected_rows_for_coverage(selection, tier="pilot")
    formal_rows = selected_rows_for_coverage(selection, tier="formal")
    pilot_samples = enumerate_label_samples(
        context,
        selection_artifacts=selection,
        coverage_tier="pilot",
    )
    formal_samples = enumerate_label_samples(
        context,
        selection_artifacts=selection,
        coverage_tier="formal",
    )

    assert {sample.sample_id for sample in pilot_samples} == {
        row["sample_id"] for row in pilot_rows
    }
    assert {sample.sample_id for sample in formal_samples} == {
        row["sample_id"] for row in formal_rows
    }
    assert {
        (sample.sample_id, sample.cohort_index) for sample in formal_samples
    } == {(row["sample_id"], row["cohort_index"]) for row in formal_rows}

    pilot_plans = plan_label_chunks(
        context=context,
        output_dir=tmp_path,
        chunk_size=2,
        selection_artifacts=selection,
        coverage_tier="pilot",
    )
    formal_plans = plan_label_chunks(
        context=context,
        output_dir=tmp_path,
        chunk_size=2,
        selection_artifacts=selection,
        coverage_tier="formal",
    )
    root = Path(tmp_path).resolve()
    for plan in formal_plans:
        assert plan.cohort_index is not None
        assert plan.path == (
            root
            / f"cohort-{plan.cohort_index:05d}"
            / f"shard-{plan.shard_index:05d}"
            / f"chunk-{plan.chunk_index:08d}.json"
        )
        assert all(
            sample.cohort_index == plan.cohort_index for sample in plan.samples
        )
    assert [
        (plan.cohort_index, plan.shard_index, plan.chunk_index)
        for plan in formal_plans
    ] == sorted(
        (plan.cohort_index, plan.shard_index, plan.chunk_index)
        for plan in formal_plans
    )

    formal_by_path = {plan.path: plan for plan in formal_plans}
    assert all(formal_by_path[plan.path] == plan for plan in pilot_plans)
    assert {sample.sample_id for plan in formal_plans for sample in plan.samples} == {
        row["sample_id"] for row in formal_rows
    }
    assert list(tmp_path.iterdir()) == []


def test_subset_streaming_plan_rejects_duplicate_ids_across_cohorts(
    tmp_path,
    monkeypatch,
):
    context, selection = _selection_context(num_shards=1)
    sample = enumerate_label_samples(
        context,
        selection_artifacts=selection,
        coverage_tier="pilot",
    )[0]
    duplicate = gate_label_job.PlannedLabelSample(
        sample_id=sample.sample_id,
        identity=sample.identity,
        shard_index=sample.shard_index,
        cohort_index=sample.cohort_index + 1,
    )

    def duplicate_samples(_context, **_kwargs):
        yield sample
        yield duplicate

    monkeypatch.setattr(gate_label_job, "iter_label_samples", duplicate_samples)
    with pytest.raises(RuntimeError, match="duplicate sample IDs"):
        tuple(
            iter_label_chunks(
                context=context,
                output_dir=tmp_path,
                chunk_size=2,
                selection_artifacts=selection,
                coverage_tier="pilot",
            )
        )


def test_subset_run_expands_only_new_cohort_and_reuses_old_chunk_bytes(tmp_path):
    context, selection = _selection_context(num_shards=1)
    full_identities = enumerate_label_samples(
        _context(num_shards=1, chunk_size=2)
    )
    pilot_plans = plan_label_chunks(
        context=context,
        output_dir=tmp_path,
        chunk_size=2,
        selection_artifacts=selection,
        coverage_tier="pilot",
    )
    pilot_dataset = RecordingDataset(full_identities)
    first = run_label_job(
        object(),
        pilot_dataset,
        context=context,
        output_dir=tmp_path,
        chunk_size=2,
        selection_artifacts=selection,
        coverage_tier="pilot",
        dependencies=_dependencies(FakeRolloutRunner()),
    )

    assert len(pilot_dataset) == context.data_manifest["num_frames"] == 6
    assert pilot_dataset.requests == [
        sample.global_sample_index
        for plan in pilot_plans
        for sample in plan.samples
    ]
    assert first.written_chunk_count == len(pilot_plans)
    assert first.inferred_sample_count == sum(
        len(plan.samples) for plan in pilot_plans
    )
    before = {path: path.read_bytes() for path in first.chunk_paths}
    for plan in pilot_plans:
        payload = load_complete_label_chunk_from_context(
            plan.path,
            context=context,
            planned_sample_ids=plan.planned_sample_ids,
            selection_sha256=selection.descriptor["selection_sha256"],
            cohort_index=plan.cohort_index,
        )
        assert payload["schema_version"] == 2
        assert payload["cohort_index"] == plan.cohort_index

    formal_plans = plan_label_chunks(
        context=context,
        output_dir=tmp_path,
        chunk_size=2,
        selection_artifacts=selection,
        coverage_tier="formal",
    )
    new_plans = [plan for plan in formal_plans if plan.path not in before]
    formal_dataset = RecordingDataset(full_identities)
    second = run_label_job(
        object(),
        formal_dataset,
        context=context,
        output_dir=tmp_path,
        chunk_size=2,
        selection_artifacts=selection,
        coverage_tier="formal",
        dependencies=_dependencies(FakeRolloutRunner()),
    )

    assert second.resumed_chunk_count == len(pilot_plans)
    assert second.written_chunk_count == len(new_plans)
    assert second.inferred_sample_count == sum(
        len(plan.samples) for plan in new_plans
    )
    assert formal_dataset.requests == [
        sample.global_sample_index
        for plan in new_plans
        for sample in plan.samples
    ]
    assert {path: path.read_bytes() for path in before} == before


def test_run_uses_streaming_planner_not_materializing_api(
    tmp_path,
    monkeypatch,
):
    context = _context(num_shards=2)
    identities = enumerate_label_samples(context)

    def forbidden_materialization(**_kwargs):
        raise AssertionError("formal run must not materialize all chunk plans")

    monkeypatch.setattr(
        gate_label_job, "plan_label_chunks", forbidden_materialization
    )
    result = run_label_job(
        object(),
        RecordingDataset(identities),
        context=context,
        output_dir=tmp_path,
        chunk_size=2,
        dependencies=_dependencies(FakeRolloutRunner()),
    )

    assert result.planned_sample_count == 6
    assert result.inferred_sample_count == 6
    assert result.planned_chunk_count == len(result.chunk_paths)


def test_run_writes_complete_chunks_then_resumes_without_dataset_reads(tmp_path):
    context = _context(num_shards=2)
    identities = enumerate_label_samples(context)
    dataset = RecordingDataset(identities)
    runner = FakeRolloutRunner()
    plans = plan_label_chunks(
        context=context,
        output_dir=tmp_path,
        chunk_size=2,
    )

    first = run_label_job(
        object(),
        dataset,
        context=context,
        output_dir=tmp_path,
        chunk_size=2,
        dependencies=_dependencies(runner),
    )

    assert first.written_chunk_count == first.planned_chunk_count == len(plans)
    assert first.resumed_chunk_count == 0
    assert first.inferred_sample_count == first.planned_sample_count == 6
    assert dataset.requests == [
        sample.global_sample_index for plan in plans for sample in plan.samples
    ]
    before = {path: path.read_bytes() for path in first.chunk_paths}
    for plan in plans:
        chunk = load_complete_label_chunk_from_context(
            plan.path,
            context=context,
            planned_sample_ids=plan.planned_sample_ids,
        )
        assert chunk["shard_index"] == plan.shard_index
        assert chunk["chunk_index"] == plan.chunk_index
        assert chunk["row_count"] == len(plan.samples)
    assert not list(tmp_path.rglob("*.tmp"))

    resumed_dataset = RecordingDataset(identities)

    def forbidden_rollout(*args, **kwargs):
        del args, kwargs
        raise AssertionError("a complete chunk must skip inference")

    second = run_label_job(
        object(),
        resumed_dataset,
        context=context,
        output_dir=tmp_path,
        chunk_size=2,
        dependencies=_dependencies(forbidden_rollout),
    )

    assert second.written_chunk_count == 0
    assert second.resumed_chunk_count == second.planned_chunk_count == len(plans)
    assert second.inferred_sample_count == 0
    assert resumed_dataset.requests == []
    assert {path: path.read_bytes() for path in second.chunk_paths} == before


def test_run_uses_scoped_guard_twice_per_chunk_and_keeps_legacy_api(tmp_path):
    context = _context(num_shards=2)
    identities = enumerate_label_samples(context)
    plans = plan_label_chunks(
        context=context,
        output_dir=tmp_path / "scoped",
        chunk_size=2,
    )

    class RecordingScopedGuard:
        def __init__(self) -> None:
            self.calls = []
            self.legacy_calls = 0

        def __call__(self) -> None:
            self.legacy_calls += 1

        def check_sample_identities(self, samples) -> None:
            self.calls.append(
                tuple(
                    (sample["dataset_index"], sample["episode_index"])
                    for sample in samples
                )
            )

    scoped_guard = RecordingScopedGuard()
    run_label_job(
        object(),
        RecordingDataset(identities),
        context=context,
        output_dir=tmp_path / "scoped",
        chunk_size=2,
        dependencies=_dependencies(FakeRolloutRunner()),
        source_guard=scoped_guard,
    )
    expected = []
    for plan in plans:
        chunk_identities = tuple(
            (
                sample.identity["dataset_index"],
                sample.identity["episode_index"],
            )
            for sample in plan.samples
        )
        expected.extend((chunk_identities, chunk_identities))
    assert scoped_guard.calls == expected
    assert scoped_guard.legacy_calls == 0

    legacy_calls = 0

    def legacy_guard() -> None:
        nonlocal legacy_calls
        legacy_calls += 1

    legacy_result = run_label_job(
        object(),
        RecordingDataset(identities),
        context=context,
        output_dir=tmp_path / "legacy",
        chunk_size=2,
        dependencies=_dependencies(FakeRolloutRunner()),
        source_guard=legacy_guard,
    )
    assert legacy_calls == 2 * legacy_result.planned_chunk_count


def test_dataset_identity_mismatch_fails_before_rollout_or_publish(tmp_path):
    context = _context(num_shards=2, chunk_size=3)
    identities = enumerate_label_samples(context)
    all_plans = plan_label_chunks(
        context=context,
        output_dir=tmp_path,
        chunk_size=3,
    )
    first_plan = all_plans[0]
    first_sample = first_plan.samples[0]
    dataset = RecordingDataset(identities)
    other = identities[(first_sample.global_sample_index + 1) % len(identities)]
    dataset.samples[first_sample.global_sample_index]["sample_identity"] = dict(
        other.identity
    )
    runner = FakeRolloutRunner()

    with pytest.raises(ValueError, match="dataset sample_identity disagrees"):
        run_label_job(
            object(),
            dataset,
            context=context,
            output_dir=tmp_path,
            chunk_size=3,
            shard_indices=[first_plan.shard_index],
            dependencies=_dependencies(runner),
        )

    assert runner.calls == []
    assert not first_plan.path.exists()


def test_corrupt_existing_chunk_is_never_silently_overwritten(tmp_path):
    context = _context(num_shards=2)
    identities = enumerate_label_samples(context)
    plan = plan_label_chunks(
        context=context,
        output_dir=tmp_path,
        chunk_size=2,
    )[0]
    plan.path.parent.mkdir(parents=True)
    corrupt = b'{"interrupted":'
    plan.path.write_bytes(corrupt)
    dataset = RecordingDataset(identities)

    def forbidden_rollout(*args, **kwargs):
        del args, kwargs
        raise AssertionError("corrupt resume must fail before inference")

    with pytest.raises(ValueError, match="unreadable or incomplete"):
        run_label_job(
            object(),
            dataset,
            context=context,
            output_dir=tmp_path,
            chunk_size=2,
            shard_indices=[plan.shard_index],
            dependencies=_dependencies(forbidden_rollout),
        )

    assert plan.path.read_bytes() == corrupt
    assert dataset.requests == []


def test_failed_inference_publishes_no_partial_chunk_and_retry_is_atomic(tmp_path):
    context = _context(num_shards=1, chunk_size=99)
    identities = enumerate_label_samples(context)
    plan = plan_label_chunks(
        context=context,
        output_dir=tmp_path,
        chunk_size=99,
    )[0]
    failing_dataset = RecordingDataset(identities)

    with pytest.raises(RuntimeError, match="synthetic rollout failure"):
        run_label_job(
            object(),
            failing_dataset,
            context=context,
            output_dir=tmp_path,
            chunk_size=99,
            dependencies=_dependencies(FakeRolloutRunner(fail_on_call=2)),
        )

    assert not plan.path.exists()
    assert not list(tmp_path.rglob("*.tmp"))

    retry = run_label_job(
        object(),
        RecordingDataset(identities),
        context=context,
        output_dir=tmp_path,
        chunk_size=99,
        dependencies=_dependencies(FakeRolloutRunner()),
    )
    assert retry.written_chunk_count == 1
    assert retry.resumed_chunk_count == 0
    assert load_complete_label_chunk_from_context(
        plan.path,
        context=context,
        planned_sample_ids=plan.planned_sample_ids,
    )["row_count"] == 6
    assert not list(tmp_path.rglob("*.tmp"))


def test_failed_inference_immediately_removes_streaming_plan_index(
    tmp_path,
    monkeypatch,
):
    context = _context(num_shards=1, chunk_size=99)
    identities = enumerate_label_samples(context)
    original_temporary_directory = gate_label_job.tempfile.TemporaryDirectory
    plan_index_directories: list[Path] = []

    def tracking_temporary_directory(*args, **kwargs):
        directory = original_temporary_directory(*args, **kwargs)
        plan_index_directories.append(Path(directory.name))
        return directory

    monkeypatch.setattr(
        gate_label_job.tempfile,
        "TemporaryDirectory",
        tracking_temporary_directory,
    )
    with pytest.raises(RuntimeError, match="synthetic rollout failure"):
        run_label_job(
            object(),
            RecordingDataset(identities),
            context=context,
            output_dir=tmp_path,
            chunk_size=99,
            dependencies=_dependencies(FakeRolloutRunner(fail_on_call=1)),
        )

    assert plan_index_directories
    assert all(not path.exists() for path in plan_index_directories)


def test_publish_race_with_identical_complete_winner_resumes(tmp_path, monkeypatch):
    context = _context(num_shards=1, chunk_size=99)
    identities = enumerate_label_samples(context)
    dataset = RecordingDataset(identities)
    original_link = gate_artifacts.os.link
    link_calls = 0

    def identical_winner_first(source, destination):
        nonlocal link_calls
        link_calls += 1
        # Simulate another worker atomically publishing this exact complete
        # candidate immediately before our own create-if-absent operation.
        original_link(source, destination)
        return original_link(source, destination)

    monkeypatch.setattr(gate_artifacts.os, "link", identical_winner_first)
    result = run_label_job(
        object(),
        dataset,
        context=context,
        output_dir=tmp_path,
        chunk_size=99,
        dependencies=_dependencies(FakeRolloutRunner()),
    )

    assert link_calls == 1
    assert result.written_chunk_count == 0
    assert result.resumed_chunk_count == result.planned_chunk_count == 1
    assert result.inferred_sample_count == 6
    assert load_complete_label_chunk_from_context(
        result.chunk_paths[0],
        context=context,
        planned_sample_ids=plan_label_chunks(
            context=context,
            output_dir=tmp_path,
            chunk_size=99,
        )[0].planned_sample_ids,
    )["row_count"] == 6
    assert not list(tmp_path.rglob("*.tmp"))


@pytest.mark.parametrize(
    ("winner_kind", "message"),
    [
        ("different", "coordinates"),
        ("corrupt", "unreadable or incomplete"),
    ],
)
def test_publish_race_never_overwrites_different_or_corrupt_winner(
    tmp_path,
    monkeypatch,
    winner_kind,
    message,
):
    context = _context(num_shards=1, chunk_size=99)
    identities = enumerate_label_samples(context)
    dataset = RecordingDataset(identities)
    original_link = gate_artifacts.os.link
    winner_bytes: dict[str, bytes] = {}

    def competing_winner_first(source, destination):
        candidate = Path(source)
        destination = Path(destination)
        if winner_kind == "different":
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            payload["chunk_index"] += 1
            unhashed = dict(payload)
            unhashed.pop("chunk_sha256")
            payload["chunk_sha256"] = canonical_json_sha256(unhashed)
            content = (
                json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True)
                + "\n"
            ).encode("utf-8")
        else:
            content = b'{"interrupted":'
        competitor = destination.with_name(".competing-worker.tmp")
        competitor.write_bytes(content)
        original_link(competitor, destination)
        competitor.unlink()
        winner_bytes["value"] = content
        return original_link(candidate, destination)

    monkeypatch.setattr(gate_artifacts.os, "link", competing_winner_first)
    with pytest.raises(ValueError, match=message):
        run_label_job(
            object(),
            dataset,
            context=context,
            output_dir=tmp_path,
            chunk_size=99,
            dependencies=_dependencies(FakeRolloutRunner()),
        )

    plan = plan_label_chunks(
        context=context,
        output_dir=tmp_path,
        chunk_size=99,
    )[0]
    assert plan.path.read_bytes() == winner_bytes["value"]
    assert dataset.requests == [
        sample.global_sample_index for sample in plan.samples
    ]
    assert not list(tmp_path.rglob("*.tmp"))
