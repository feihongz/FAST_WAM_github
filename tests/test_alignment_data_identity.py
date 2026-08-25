from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from fastwam.alignment.data_identity import (
    DATA_MANIFEST_KIND,
    LEROBOT_META_PATHS,
    build_robot_video_dataset_manifest,
    canonical_data_manifest_sha256,
    validate_robot_video_dataset_manifest,
)


class _FakeMeta:
    def __init__(
        self,
        *,
        episodes: dict[int, dict],
        video_keys: list[str],
        tasks: dict[int, str],
    ) -> None:
        self.episodes = episodes
        self.video_keys = video_keys
        self.tasks = tasks

    @staticmethod
    def get_data_file_path(episode: int) -> Path:
        return Path(f"data/chunk-000/episode_{episode:06d}.parquet")

    @staticmethod
    def get_video_file_path(episode: int, video_key: str) -> Path:
        return Path(
            f"videos/chunk-000/{video_key}/episode_{episode:06d}.mp4"
        )


class _FakeRobotVideoDataset:
    def __init__(
        self,
        *,
        parts: list[SimpleNamespace],
        cache_root: Path,
    ) -> None:
        roots = [str(part.root) for part in parts]
        base = SimpleNamespace(
            dataset_dirs=roots,
            multi_dataset=SimpleNamespace(_datasets=parts),
            obs_size=33,
            global_sample_stride=1,
        )
        self.lerobot_dataset = base
        self.num_frames = 33
        self.action_video_freq_ratio = 4
        self.video_sample_indices = list(range(0, 33, 4))
        self.context_len = 128
        self.text_embedding_cache_dir = str(cache_root)
        self.override_instruction = None
        self.strict_data_mode = True
        self._length = sum(part.num_frames for part in parts)

    def __len__(self) -> int:
        return self._length


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _prompt_cache_name(task: str, context_len: int = 128) -> str:
    prompt = (
        "A video recorded from a robot's point of view executing the following "
        f"instruction: {task}"
    )
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return f"{digest}.t5_len{context_len}.wan22ti2v5b.pt"


def _part(
    root: Path,
    *,
    selected_episodes: list[int],
    episode_rows: dict[int, dict],
) -> SimpleNamespace:
    for relative in LEROBOT_META_PATHS:
        _write(root / relative, f"{root.name}:{relative}\n".encode())
    for episode in selected_episodes:
        _write(
            root / _FakeMeta.get_data_file_path(episode),
            f"parquet:{root.name}:{episode}".encode(),
        )
        for video_key in ("image", "wrist_image"):
            _write(
                root / _FakeMeta.get_video_file_path(episode, video_key),
                f"mp4:{root.name}:{episode}:{video_key}".encode(),
            )

    starts: list[int] = []
    ends: list[int] = []
    cursor = 0
    for episode in selected_episodes:
        starts.append(cursor)
        cursor += episode_rows[episode]["length"]
        ends.append(cursor)
    return SimpleNamespace(
        root=root,
        episodes=selected_episodes,
        meta=_FakeMeta(
            episodes=episode_rows,
            video_keys=["image", "wrist_image"],
            tasks={
                index: row["prompt_task"]
                for index, row in episode_rows.items()
            },
        ),
        hf_dataset={"task_index": selected_episodes},
        episode_data_index={"from": starts, "to": ends},
        num_frames=cursor,
        video_backend="torchcodec",
        allow_video_backend_fallback=False,
    )


@pytest.fixture
def fake_data(tmp_path):
    # Root names intentionally disagree with manifest order to prove the caller's
    # ordered dataset list is preserved instead of path-sorted.
    root_z = tmp_path / "z_dataset"
    root_a = tmp_path / "a_dataset"
    rows_z = {
        0: {
            "length": 3,
            "tasks": ["coarse label", "shared task", "quality label"],
            "prompt_task": "shared task",
        },
        1: {
            "length": 2,
            "tasks": ["coarse label", "z task", "quality label"],
            "prompt_task": "z task",
        },
    }
    rows_a = {
        4: {
            "length": 4,
            "tasks": ["coarse label", "a task", "quality label"],
            "prompt_task": "a task",
        }
    }
    part_z = _part(
        root_z,
        selected_episodes=[1, 0],
        episode_rows=rows_z,
    )
    part_a = _part(
        root_a,
        selected_episodes=[4],
        episode_rows=rows_a,
    )

    cache_root = tmp_path / "text_cache"
    for task in ("z task", "shared task", "a task"):
        _write(cache_root / _prompt_cache_name(task), f"embed:{task}".encode())
    stats = tmp_path / "assets" / "dataset_stats.json"
    _write(stats, b'{"action": "stats"}\n')
    dataset = _FakeRobotVideoDataset(
        parts=[part_z, part_a],
        cache_root=cache_root,
    )
    return dataset, stats


def _resign(manifest: dict) -> None:
    manifest["manifest_sha256"] = canonical_data_manifest_sha256(manifest)


def test_manifest_binds_order_boundaries_and_exact_selected_files(fake_data):
    dataset, stats = fake_data
    manifest = build_robot_video_dataset_manifest(
        dataset,
        normalization_stats_path=stats,
    )

    assert manifest["kind"] == DATA_MANIFEST_KIND
    assert [Path(row["root"]).name for row in manifest["dataset_roots"]] == [
        "z_dataset",
        "a_dataset",
    ]
    assert manifest["dataset_roots"][0]["selected_episodes"] == [1, 0]
    assert manifest["dataset_roots"][0]["episode_boundaries"] == [
        {"episode_index": 1, "from": 0, "to": 2, "length": 2},
        {"episode_index": 0, "from": 2, "to": 5, "length": 3},
    ]
    assert manifest["num_frames"] == 9
    assert manifest["sampling"]["num_frames"] == 33

    files = manifest["dataset_roots"][0]["files"]
    assert [entry["relative_path"] for entry in files[:4]] == list(
        LEROBOT_META_PATHS
    )
    assert [entry["role"] for entry in files[4:]] == [
        "parquet",
        "video",
        "video",
        "parquet",
        "video",
        "video",
    ]
    assert len(manifest["text_embedding_cache"]["files"]) == 3
    assert manifest["decoder"]["datasets"] == [
        {"dataset_index": 0, "backend": "torchcodec", "allow_fallback": False},
        {"dataset_index": 1, "backend": "torchcodec", "allow_fallback": False},
    ]
    for root in manifest["dataset_roots"]:
        for entry in root["files"]:
            assert not Path(entry["relative_path"]).is_absolute()
            assert entry["size_bytes"] > 0
            assert len(entry["sha256"]) == 64

    assert validate_robot_video_dataset_manifest(
        dataset,
        manifest,
        normalization_stats_path=stats,
    ) == manifest


def test_manifest_is_canonical_and_repeatable(fake_data):
    dataset, stats = fake_data
    first = build_robot_video_dataset_manifest(
        dataset,
        normalization_stats_path=stats,
    )
    second = build_robot_video_dataset_manifest(
        dataset,
        normalization_stats_path=stats,
    )

    assert first == second
    reordered = dict(reversed(list(first.items())))
    assert canonical_data_manifest_sha256(reordered) == first["manifest_sha256"]


def test_full_validation_detects_same_size_content_tamper(fake_data):
    dataset, stats = fake_data
    manifest = build_robot_video_dataset_manifest(
        dataset,
        normalization_stats_path=stats,
    )
    video = (
        Path(manifest["dataset_roots"][0]["root"])
        / manifest["dataset_roots"][0]["files"][5]["relative_path"]
    )
    original = video.read_bytes()
    video.write_bytes(b"X" * len(original))

    with pytest.raises(ValueError, match="drifted"):
        validate_robot_video_dataset_manifest(
            dataset,
            manifest,
            normalization_stats_path=stats,
        )
    # The explicitly non-formal mode checks path and size but trusts hashes.
    validate_robot_video_dataset_manifest(
        dataset,
        manifest,
        normalization_stats_path=stats,
        full_content_verify=False,
    )


def test_validation_detects_manifest_path_and_hash_tamper(fake_data):
    dataset, stats = fake_data
    manifest = build_robot_video_dataset_manifest(
        dataset,
        normalization_stats_path=stats,
    )

    wrong_path = copy.deepcopy(manifest)
    wrong_path["dataset_roots"][0]["files"][0]["relative_path"] = (
        "meta/not-info.json"
    )
    _resign(wrong_path)
    with pytest.raises(ValueError, match="drifted"):
        validate_robot_video_dataset_manifest(
            dataset,
            wrong_path,
            normalization_stats_path=stats,
        )

    wrong_hash = copy.deepcopy(manifest)
    wrong_hash["dataset_roots"][0]["files"][0]["sha256"] = "0" * 64
    _resign(wrong_hash)
    with pytest.raises(ValueError, match="drifted"):
        validate_robot_video_dataset_manifest(
            dataset,
            wrong_hash,
            normalization_stats_path=stats,
        )


def test_validation_detects_file_size_drift_without_rehashing(fake_data):
    dataset, stats = fake_data
    manifest = build_robot_video_dataset_manifest(
        dataset,
        normalization_stats_path=stats,
    )
    parquet = (
        Path(manifest["dataset_roots"][1]["root"])
        / manifest["dataset_roots"][1]["files"][4]["relative_path"]
    )
    parquet.write_bytes(parquet.read_bytes() + b"changed-size")

    with pytest.raises(ValueError, match="drifted"):
        validate_robot_video_dataset_manifest(
            dataset,
            manifest,
            normalization_stats_path=stats,
            full_content_verify=False,
        )
