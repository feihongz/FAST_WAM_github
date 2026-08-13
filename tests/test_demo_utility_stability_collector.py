from __future__ import annotations

import copy

import pytest

from experiments.libero.gate.collect_demo_utility_multiseed import (
    AUDIT_KIND,
    _validate_manifest_integrity,
    collect_replicate_grid,
)
from experiments.libero.gate.collect_demo_utility import _sha256_json


class CountingDataset:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def __getitem__(self, source_index: int):
        self.calls.append(source_index)
        return {"source_index": source_index}


def _selected(source_index: int, order: int) -> dict:
    return {
        "source_index": source_index,
        "selection_order": order,
        "selection_bin": "NZ",
        "sample_id": f"suite/episode_{source_index:06d}/frame_000000",
        "seed": source_index + 1,
        "utility": float(source_index),
        "source_metadata": {
            "requested_sample_idx": source_index,
            "source_sample_idx": source_index,
        },
    }


def test_grid_decodes_each_state_once_reuses_pilot_and_infers_other_seeds():
    dataset = CountingDataset()
    selected = [_selected(11, 0), _selected(22, 1)]
    inference_calls: list[tuple[int, int]] = []
    written: list[dict] = []

    def infer(model, sample, pilot, base_seed):
        assert model == "frozen-model"
        inference_calls.append((sample["source_index"], base_seed))
        return {"seed": base_seed, "utility": base_seed / 100.0}

    def finalize(utility, pilot, replicate_index, base_seed, sample):
        return {
            **utility,
            "source_index": sample["source_index"],
            "replicate_index": replicate_index,
            "replicate_base_seed": base_seed,
            "pilot_object_reused": utility is not pilot,
        }

    summary = collect_replicate_grid(
        dataset=dataset,
        model="frozen-model",
        selected_records=selected,
        base_seeds=(42, 43, 44, 45, 46),
        reuse_base_seed=42,
        existing_keys=set(),
        infer_record=infer,
        finalize_record=finalize,
        write_record=written.append,
    )

    assert dataset.calls == [11, 22]
    assert len(inference_calls) == 8
    assert all(seed != 42 for _, seed in inference_calls)
    assert len(written) == 10
    assert [row["replicate_base_seed"] for row in written] == [42, 43, 44, 45, 46] * 2
    assert [row["utility"] for row in written if row["replicate_base_seed"] == 42] == [
        11.0,
        22.0,
    ]
    assert summary == {"new": 10, "reused": 2, "inferred": 8, "errors": 0}


def test_grid_resume_uses_composite_keys_and_skips_fully_complete_states():
    dataset = CountingDataset()
    selected = [_selected(11, 0), _selected(22, 1)]
    existing = {(11, 0), (11, 2)} | {(22, index) for index in range(5)}
    inference_calls: list[int] = []
    written: list[dict] = []

    def infer(model, sample, pilot, base_seed):
        inference_calls.append(base_seed)
        return {"seed": base_seed}

    summary = collect_replicate_grid(
        dataset=dataset,
        model=object(),
        selected_records=selected,
        base_seeds=(42, 43, 44, 45, 46),
        reuse_base_seed=42,
        existing_keys=existing,
        infer_record=infer,
        finalize_record=lambda utility, pilot, index, seed, sample: {
            **utility,
            "source_index": sample["source_index"],
            "replicate_index": index,
        },
        write_record=written.append,
    )

    assert dataset.calls == [11]
    assert inference_calls == [43, 45, 46]
    assert [(row["source_index"], row["replicate_index"]) for row in written] == [
        (11, 1),
        (11, 3),
        (11, 4),
    ]
    assert summary == {"new": 3, "reused": 0, "inferred": 3, "errors": 0}


def test_grid_with_no_pending_cells_never_decodes_or_needs_model():
    dataset = CountingDataset()
    selected = [_selected(11, 0)]
    summary = collect_replicate_grid(
        dataset=dataset,
        model=None,
        selected_records=selected,
        base_seeds=(42, 43, 44, 45, 46),
        reuse_base_seed=42,
        existing_keys={(11, index) for index in range(5)},
        infer_record=lambda *args: pytest.fail("inference must not run"),
        finalize_record=lambda *args: pytest.fail("finalization must not run"),
        write_record=lambda *args: pytest.fail("writing must not run"),
    )
    assert dataset.calls == []
    assert summary == {"new": 0, "reused": 0, "inferred": 0, "errors": 0}


def _manifest() -> dict:
    states = [
        {
            "selection_order": 0,
            "source_index": 11,
            "sample_id": "suite/episode_000011/frame_000000",
        }
    ]
    state_sha = _sha256_json(states)
    compatibility = {
        "schema_version": 1,
        "kind": AUDIT_KIND,
        "pilot_manifest_fingerprint": "1" * 64,
        "pilot_manifest_sha256": "2" * 64,
        "pilot_records_sha256": "3" * 64,
        "selection_plan_sha256": state_sha,
        "replicate_base_seeds": [42, 43, 44, 45, 46],
    }
    return {
        "kind": AUDIT_KIND,
        "compatibility": compatibility,
        "compatibility_fingerprint": _sha256_json(compatibility),
        "pilot": {
            "manifest_fingerprint": "1" * 64,
            "manifest_sha256": "2" * 64,
            "records_sha256": "3" * 64,
        },
        "selection": {
            "num_states": 1,
            "ordered_states": states,
            "ordered_states_sha256": state_sha,
        },
        "replicates": {
            "base_seeds": [42, 43, 44, 45, 46],
            "expected_record_count": 5,
        },
    }


def test_stability_manifest_integrity_rejects_selection_and_pilot_tampering():
    manifest = _manifest()
    _validate_manifest_integrity(manifest)

    tampered_selection = copy.deepcopy(manifest)
    tampered_selection["selection"]["ordered_states"][0]["source_index"] = 99
    with pytest.raises(ValueError, match="selection digest"):
        _validate_manifest_integrity(tampered_selection)

    tampered_pilot = copy.deepcopy(manifest)
    tampered_pilot["pilot"]["records_sha256"] = "4" * 64
    with pytest.raises(ValueError, match="Pilot records_sha256"):
        _validate_manifest_integrity(tampered_pilot)
