from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from fastwam.alignment.data_identity import canonical_data_manifest_sha256
from fastwam.gating.artifacts import build_label_contract, build_label_row
from fastwam.gating.contracts import build_episode_split
from fastwam.gating.dataset import Stage2GateDataset


class _StrictRobotVideoDataset:
    gate_input_schema_version = 1

    def __init__(self, samples: list[dict]) -> None:
        self.samples = samples
        self.requests: list[int] = []
        self.strict_data_mode = True
        self.lerobot_dataset = SimpleNamespace(strict_data_mode=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        self.requests.append(index)
        return self.samples[index]


class _PostQueryTrap(dict):
    _FORBIDDEN = frozenset({"video", "action", "future"})

    def __getitem__(self, key):
        if key in self._FORBIDDEN:
            raise AssertionError(f"forbidden post-query field accessed: {key}")
        return super().__getitem__(key)


def _manifest() -> dict:
    manifest = {
        "schema_version": 1,
        "kind": "stage3_libero_data_manifest",
        "sampling": {},
        "num_frames": 4,
        "dataset_roots": [
            {
                "dataset_index": 0,
                "root": "/data/a",
                "selected_episodes": [2, 5],
                "num_frames": 4,
                "episode_boundaries": [
                    {"episode_index": 2, "from": 0, "to": 2, "length": 2},
                    {"episode_index": 5, "from": 2, "to": 4, "length": 2},
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


def _identities() -> list[dict[str, int]]:
    return [
        {
            "global_sample_index": global_index,
            "dataset_index": 0,
            "episode_index": 2 if global_index < 2 else 5,
            "frame_index": global_index % 2,
            "dataset_frame_index": global_index,
        }
        for global_index in range(4)
    ]


def _sample(identity: dict[str, int]) -> dict:
    offset = 1000.0 * identity["global_sample_index"]
    input_image = torch.arange(3 * 2 * 2, dtype=torch.float32).reshape(
        3, 2, 2
    )
    return {
        "input_image": input_image + offset,
        "context": torch.full((3, 4), offset, dtype=torch.float32),
        "context_mask": torch.tensor([True, False, True]),
        "proprio": torch.arange(5, dtype=torch.float32) + offset,
        "sample_identity": dict(identity),
    }


def _job() -> tuple[_StrictRobotVideoDataset, dict, dict, list[dict]]:
    manifest = _manifest()
    episode_split = build_episode_split(
        manifest,
        validation_fraction=0.5,
        split_seed=9,
    )
    contract = build_label_contract(
        data_manifest=manifest,
        episode_split=episode_split,
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
        num_shards=2,
        chunk_size=2,
    )
    identities = _identities()
    rows = []
    for index, identity in enumerate(identities):
        e10 = 0.8 if index % 2 == 0 else 1.0
        rows.append(
            build_label_row(
                contract=contract,
                data_manifest=manifest,
                episode_split=episode_split,
                identity=identity,
                e0=1.0,
                e10=e10,
                relative_gain=1.0 - e10,
                label=e10 < 0.95,
                sample_weight=0.5 + index,
                num_video_frames=5,
            )
        )
    samples = [_sample(identity) for identity in identities]
    return _StrictRobotVideoDataset(samples), manifest, episode_split, rows


def _dataset(
    base: _StrictRobotVideoDataset,
    manifest: dict,
    episode_split: dict,
    rows: list[dict],
    *,
    split: str,
) -> Stage2GateDataset:
    return Stage2GateDataset(
        base,
        label_rows=rows,
        data_manifest=manifest,
        episode_split=episode_split,
        split=split,
    )


@pytest.mark.parametrize("selected_split", ["train", "validation"])
def test_gate_dataset_is_lazy_fixed_order_and_current_only(selected_split):
    base, manifest, episode_split, rows = _job()
    rows.reverse()  # Input artifact order must not affect training order.
    expected_rows = sorted(
        (row for row in rows if row["split"] == selected_split),
        key=lambda row: row["global_sample_index"],
    )

    dataset = _dataset(
        base,
        manifest,
        episode_split,
        rows,
        split=selected_split,
    )

    assert base.requests == []  # Construction is metadata-only.
    assert len(dataset) == len(expected_rows)
    assert dataset.labels == tuple(row["label"] for row in expected_rows)
    assert dataset.sample_ids == tuple(row["sample_id"] for row in expected_rows)
    with pytest.raises(AttributeError):
        dataset.labels = ()
    with pytest.raises(AttributeError):
        dataset.sample_ids = ()
    for index, row in enumerate(expected_rows):
        item = dataset[index]
        source = base.samples[row["global_sample_index"]]
        assert set(item) == {
            "input_image",
            "context",
            "context_mask",
            "proprio",
            "label",
            "sample_weight",
            "sample_id",
        }
        assert item["input_image"] is source["input_image"]
        assert item["context"] is source["context"]
        assert item["context_mask"] is source["context_mask"]
        assert item["proprio"] is source["proprio"]
        assert item["label"] is row["label"]
        assert item["sample_weight"] == pytest.approx(row["sample_weight"])
        assert item["sample_id"] == row["sample_id"]
        assert {"video", "action", "future", "e0", "e10", "E0", "E10"}.isdisjoint(
            item
        )

    assert base.requests == [row["global_sample_index"] for row in expected_rows]


def test_train_and_validation_splits_are_disjoint_and_cover_manifest():
    train_base, manifest, episode_split, rows = _job()
    validation_base = _StrictRobotVideoDataset(deepcopy(train_base.samples))
    train = _dataset(
        train_base, manifest, episode_split, list(reversed(rows)), split="train"
    )
    validation = _dataset(
        validation_base,
        manifest,
        episode_split,
        list(reversed(rows)),
        split="validation",
    )

    for index in range(len(train)):
        train[index]
    for index in range(len(validation)):
        validation[index]

    assert set(train_base.requests).isdisjoint(validation_base.requests)
    assert sorted(train_base.requests + validation_base.requests) == list(range(4))
    assert train_base.requests == sorted(train_base.requests)
    assert validation_base.requests == sorted(validation_base.requests)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows[:-1], "coverage must be complete"),
        (
            lambda rows: rows[:-1] + [deepcopy(rows[0])],
            "duplicate global_sample_index",
        ),
        (
            lambda rows: [
                dict(rows[0], dataset_frame_index=1),
                *rows[1:],
            ],
            "dataset frame index disagrees",
        ),
        (
            lambda rows: [dict(rows[0], sample_id="0" * 64), *rows[1:]],
            "sample_id disagrees",
        ),
        (
            lambda rows: [
                dict(
                    rows[0],
                    split=(
                        "validation" if rows[0]["split"] == "train" else "train"
                    ),
                ),
                *rows[1:],
            ],
            "split disagrees",
        ),
    ],
)
def test_constructor_rejects_missing_duplicate_or_drifted_join(mutate, message):
    base, manifest, episode_split, rows = _job()

    with pytest.raises(ValueError, match=message):
        _dataset(
            base,
            manifest,
            episode_split,
            mutate(deepcopy(rows)),
            split="train",
        )

    assert base.requests == []


def test_getitem_rejects_lazy_robot_sample_identity_drift():
    base, manifest, episode_split, rows = _job()
    dataset = _dataset(base, manifest, episode_split, rows, split="train")
    first_train = min(
        row["global_sample_index"] for row in rows if row["split"] == "train"
    )
    wrong_index = first_train + 1 if first_train % 2 == 0 else first_train - 1
    base.samples[first_train]["sample_identity"] = dict(_identities()[wrong_index])

    with pytest.raises(ValueError, match="sample_identity drifted"):
        dataset[0]


def test_constructor_requires_strict_robot_video_dataset():
    base, manifest, episode_split, rows = _job()
    base.strict_data_mode = False

    with pytest.raises(ValueError, match="strict_data_mode"):
        _dataset(base, manifest, episode_split, rows, split="train")


def test_gate_getitem_never_accesses_post_query_fields():
    base, manifest, episode_split, rows = _job()
    base.samples = [_PostQueryTrap(sample) for sample in base.samples]
    dataset = _dataset(base, manifest, episode_split, rows, split="train")

    item = dataset[0]

    assert set(item).isdisjoint(_PostQueryTrap._FORBIDDEN)



def test_sparse_gate_dataset_is_exact_lazy_and_uses_global_indices(monkeypatch):
    base, manifest, episode_split, rows = _job()
    selected = [rows[0], rows[2]]
    expected_ids = tuple(sorted(row["sample_id"] for row in selected))

    def forbidden_manifest_expansion(_manifest):
        raise AssertionError("sparse dataset must not expand every manifest frame")

    monkeypatch.setattr(
        "fastwam.gating.dataset._manifest_identities",
        forbidden_manifest_expansion,
    )
    dataset = Stage2GateDataset(
        base,
        label_rows=list(reversed(selected)),
        data_manifest=manifest,
        episode_split=episode_split,
        split=selected[0]["split"],
        expected_sample_ids=expected_ids,
    )

    expected_rows = sorted(
        (row for row in selected if row["split"] == selected[0]["split"]),
        key=lambda row: row["global_sample_index"],
    )
    assert base.requests == []
    assert dataset.sample_ids == tuple(row["sample_id"] for row in expected_rows)
    for index in range(len(dataset)):
        dataset[index]
    assert base.requests == [row["global_sample_index"] for row in expected_rows]


@pytest.mark.parametrize("mode", ["missing", "extra", "replacement"])
def test_sparse_gate_dataset_rejects_coverage_drift(mode):
    base, manifest, episode_split, rows = _job()
    selected = [rows[0], rows[2]]
    expected = sorted(row["sample_id"] for row in selected)
    if mode == "missing":
        artifact_rows = selected[:1]
    elif mode == "extra":
        artifact_rows = selected + [rows[3]]
    else:
        artifact_rows = [rows[0], rows[3]]
    with pytest.raises(ValueError, match="exactly match expected_sample_ids"):
        Stage2GateDataset(
            base,
            label_rows=artifact_rows,
            data_manifest=manifest,
            episode_split=episode_split,
            split=selected[0]["split"],
            expected_sample_ids=expected,
        )
    assert base.requests == []


@pytest.mark.parametrize(
    "expected_factory,error",
    [
        (lambda ids: [], "non-empty"),
        (lambda ids: list(reversed(ids)), "sorted and unique"),
        (lambda ids: [ids[0], ids[0]], "sorted and unique"),
        (lambda ids: [ids[0].upper(), ids[1]], "lowercase hex"),
    ],
)
def test_sparse_gate_dataset_rejects_invalid_expected_ids(
    expected_factory, error
):
    base, manifest, episode_split, rows = _job()
    selected = [rows[0], rows[2]]
    ids = sorted(row["sample_id"] for row in selected)
    with pytest.raises(ValueError, match=error):
        Stage2GateDataset(
            base,
            label_rows=selected,
            data_manifest=manifest,
            episode_split=episode_split,
            split=selected[0]["split"],
            expected_sample_ids=expected_factory(ids),
        )
    assert base.requests == []


def test_sparse_gate_dataset_rejects_duplicate_rows():
    base, manifest, episode_split, rows = _job()
    selected = [rows[0], rows[2]]
    expected = sorted(row["sample_id"] for row in selected)
    with pytest.raises(ValueError, match="duplicate global_sample_index"):
        Stage2GateDataset(
            base,
            label_rows=[selected[0], selected[0], selected[1]],
            data_manifest=manifest,
            episode_split=episode_split,
            split=selected[0]["split"],
            expected_sample_ids=expected,
        )
