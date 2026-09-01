from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts import train_video_gate as train_cli


def _subset_request() -> dict:
    return {
        "label_selection": {
            "directory": "/contracts/libero-selection",
            "expected_sha256": "a" * 64,
        },
        "label_coverage": {
            "tier": "formal",
            "expected_sha256": "b" * 64,
        },
    }


def test_resolved_config_accepts_only_legacy_or_complete_subset_schema():
    legacy = {key: None for key in train_cli._ROOT_KEYS}
    assert set(train_cli._resolved_config(legacy)) == train_cli._ROOT_KEYS

    incomplete = dict(legacy, label_selection=_subset_request()["label_selection"])
    with pytest.raises(ValueError, match="must be provided together"):
        train_cli._resolved_config(incomplete)

    subset = dict(legacy, **_subset_request())
    assert set(train_cli._resolved_config(subset)) == train_cli._SUBSET_ROOT_KEYS


def test_load_optional_selection_binds_coverage_and_sorted_sample_ids(monkeypatch):
    split = {"assignment_sha256": "c" * 64}
    coverage = {
        "coverage_sha256": "b" * 64,
        "sample_count": 2,
        "active_cohort_indices": [0, 4],
    }
    artifacts = SimpleNamespace(
        descriptor={"selection_sha256": "a" * 64},
        episode_split=split,
        coverages={"formal": coverage},
    )
    calls = []
    monkeypatch.setattr(
        train_cli,
        "load_selection_artifacts",
        lambda directory, *, data_manifest: (
            calls.append((directory, data_manifest)) or artifacts
        ),
    )
    monkeypatch.setattr(
        train_cli,
        "selected_rows_for_coverage",
        lambda value, *, tier: (
            {"sample_id": "f" * 64},
            {"sample_id": "0" * 64},
        ),
    )

    selection, loaded_coverage, sample_ids = train_cli._load_optional_selection(
        _subset_request(),
        data_manifest={"manifest_sha256": "d" * 64},
        episode_split=split,
    )

    assert selection is artifacts
    assert loaded_coverage is coverage
    assert sample_ids == ("0" * 64, "f" * 64)
    assert calls == [
        (
            "/contracts/libero-selection",
            {"manifest_sha256": "d" * 64},
        )
    ]


def test_load_optional_selection_fails_closed_on_binding_drift(monkeypatch):
    split = {"assignment_sha256": "c" * 64}
    request = _subset_request()
    artifacts = SimpleNamespace(
        descriptor={"selection_sha256": "a" * 64},
        episode_split=split,
        coverages={
            "formal": {
                "coverage_sha256": "e" * 64,
                "sample_count": 1,
                "active_cohort_indices": [0],
            }
        },
    )
    monkeypatch.setattr(
        train_cli,
        "load_selection_artifacts",
        lambda *_args, **_kwargs: artifacts,
    )

    with pytest.raises(ValueError, match="coverage SHA256 mismatch"):
        train_cli._load_optional_selection(
            request,
            data_manifest={"manifest_sha256": "d" * 64},
            episode_split=split,
        )

    artifacts.episode_split = {"assignment_sha256": "9" * 64}
    artifacts.coverages["formal"]["coverage_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="episode split mismatch"):
        train_cli._load_optional_selection(
            request,
            data_manifest={"manifest_sha256": "d" * 64},
            episode_split=split,
        )
