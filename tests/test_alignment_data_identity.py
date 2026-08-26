from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

import fastwam.alignment.data_identity as data_identity_module
import fastwam.alignment.text_cache_index as text_cache_index_module
from fastwam.alignment.data_identity import (
    DATA_MANIFEST_KIND,
    DATA_MANIFEST_V2_KIND,
    DATA_MANIFEST_V2_SCHEMA_VERSION,
    DEFAULT_DATA_MANIFEST_HASH_WORKERS,
    DEFAULT_PROMPT_TEMPLATE,
    LEROBOT_META_PATHS,
    TEXT_CACHE_FILENAME_SUFFIX_TEMPLATE,
    build_robot_video_dataset_manifest,
    canonical_data_manifest_sha256,
    require_supported_data_manifest_header,
    resolve_text_cache_index_descriptor_path,
    selected_text_cache_prompts,
    validate_robot_video_dataset_manifest,
)
from fastwam.alignment.text_cache_index import build_text_cache_index


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


def _build_v2_cache_index(tmp_path: Path, dataset, *, prompts=None) -> Path:
    selected = tuple(prompts or selected_text_cache_prompts(dataset))
    descriptor_path = tmp_path / "identity" / "text.index.json"
    build_text_cache_index(
        cache_root=dataset.text_embedding_cache_dir,
        prompts=selected,
        context_len=dataset.context_len,
        prompt_template=DEFAULT_PROMPT_TEMPLATE,
        filename_suffix=TEXT_CACHE_FILENAME_SUFFIX_TEMPLATE.format(
            context_len=dataset.context_len
        ),
        index_path=tmp_path / "identity" / "text.index",
        descriptor_path=descriptor_path,
    )
    return descriptor_path


def test_v2_manifest_uses_external_index_without_touching_cache_payloads(
    fake_data,
    tmp_path,
    monkeypatch,
):
    dataset, stats = fake_data
    descriptor_path = _build_v2_cache_index(tmp_path, dataset)
    cache_root = Path(dataset.text_embedding_cache_dir).resolve()
    original_selected_file = data_identity_module._selected_file
    original_stable_reader = text_cache_index_module._read_stable_regular_file

    def reject_cache_payload_selection(**kwargs):
        if Path(kwargs["anchor"]).resolve() == cache_root:
            pytest.fail("v2 manifest must not stat or hash cache payloads")
        return original_selected_file(**kwargs)

    def reject_cache_payload_open(anchor, relative_path):
        if Path(anchor).resolve() == cache_root:
            pytest.fail("v2 manifest must not open cache payloads")
        return original_stable_reader(anchor, relative_path)

    monkeypatch.setattr(
        data_identity_module,
        "_selected_file",
        reject_cache_payload_selection,
    )
    monkeypatch.setattr(
        text_cache_index_module,
        "_read_stable_regular_file",
        reject_cache_payload_open,
    )

    manifest = build_robot_video_dataset_manifest(
        dataset,
        normalization_stats_path=stats,
        schema_version=DATA_MANIFEST_V2_SCHEMA_VERSION,
        text_cache_index_descriptor_path=descriptor_path,
    )
    assert manifest["kind"] == DATA_MANIFEST_V2_KIND
    assert "files" not in manifest["text_embedding_cache"]
    integrity = manifest["text_embedding_cache"]["integrity"]
    assert [entry["role"] for entry in integrity["files"]] == [
        "text_cache_index_descriptor",
        "text_cache_index",
    ]
    assert require_supported_data_manifest_header(manifest) == 2
    assert resolve_text_cache_index_descriptor_path(manifest) == (
        descriptor_path.resolve()
    )
    assert validate_robot_video_dataset_manifest(
        dataset,
        manifest,
        normalization_stats_path=stats,
    ) == manifest


def test_v2_parallel_hashing_is_deterministic_and_reused_by_full_validation(
    fake_data,
    tmp_path,
    monkeypatch,
):
    dataset, stats = fake_data
    descriptor_path = _build_v2_cache_index(tmp_path, dataset)
    serial = build_robot_video_dataset_manifest(
        dataset,
        normalization_stats_path=stats,
        schema_version=DATA_MANIFEST_V2_SCHEMA_VERSION,
        text_cache_index_descriptor_path=descriptor_path,
        hash_workers=1,
    )

    original_hash = data_identity_module._sha256_file
    lock = threading.Lock()
    active = 0
    maximum_active = 0
    hash_calls = 0
    hashed_paths: list[Path] = []

    def observed_hash(path, **kwargs):
        nonlocal active, maximum_active, hash_calls
        with lock:
            active += 1
            hash_calls += 1
            hashed_paths.append(Path(path))
            maximum_active = max(maximum_active, active)
        try:
            time.sleep(0.005)
            return original_hash(path, **kwargs)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(data_identity_module, "_sha256_file", observed_hash)
    parallel = build_robot_video_dataset_manifest(
        dataset,
        normalization_stats_path=stats,
        schema_version=DATA_MANIFEST_V2_SCHEMA_VERSION,
        text_cache_index_descriptor_path=descriptor_path,
        hash_workers=4,
    )
    assert parallel == serial
    assert parallel["manifest_sha256"] == serial["manifest_sha256"]
    assert 1 < maximum_active <= 4

    calls_after_build = hash_calls
    maximum_active = 0
    assert validate_robot_video_dataset_manifest(
        dataset,
        serial,
        normalization_stats_path=stats,
        hash_workers=3,
    ) == serial
    assert hash_calls > calls_after_build
    assert 1 < maximum_active <= 3

    calls_after_full_validation = hash_calls
    paths_after_full_validation = len(hashed_paths)
    validate_robot_video_dataset_manifest(
        dataset,
        serial,
        normalization_stats_path=stats,
        full_content_verify=False,
        hash_workers=3,
    )
    # Fast topology validation still authenticates the compact descriptor file,
    # but must not hash the selected parquet/video/static inventory.
    assert hash_calls == calls_after_full_validation + 1
    assert hashed_paths[paths_after_full_validation:] == [
        descriptor_path.resolve()
    ]


def test_manifest_hash_plan_is_bounded_and_cleans_up_after_failure(
    tmp_path,
    monkeypatch,
):
    assert DEFAULT_DATA_MANIFEST_HASH_WORKERS == 32
    workers = 2
    plan = data_identity_module._SelectedFileHashPlan(workers)
    for index in range(20):
        path = tmp_path / f"{index:03d}.bin"
        _write(path, f"payload-{index}".encode())
        info = path.stat()
        plan.add(
            entry={"sha256": None},
            resolved_path=path,
            expected_stat_identity=data_identity_module._stat_identity(info),
        )

    original_hash = data_identity_module._sha256_file
    lock = threading.Lock()
    active = 0
    started = 0

    def failing_hash(path, **kwargs):
        nonlocal active, started
        with lock:
            active += 1
            started += 1
        try:
            if path.name == "000.bin":
                raise OSError("synthetic manifest hash failure")
            time.sleep(0.02)
            return original_hash(path, **kwargs)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(data_identity_module, "_sha256_file", failing_hash)
    with pytest.raises(OSError, match="synthetic manifest hash failure"):
        plan.execute()

    assert active == 0
    assert started <= workers * 2
    assert plan._work == []
    assert not any(
        thread.name.startswith("fastwam-data-manifest-hash")
        for thread in threading.enumerate()
    )


def test_v2_manifest_requires_exact_selected_prompt_coverage(fake_data, tmp_path):
    dataset, stats = fake_data
    prompts = selected_text_cache_prompts(dataset)
    descriptor_path = _build_v2_cache_index(
        tmp_path,
        dataset,
        prompts=prompts[:-1],
    )
    with pytest.raises(ValueError, match="prompt count"):
        build_robot_video_dataset_manifest(
            dataset,
            normalization_stats_path=stats,
            schema_version=DATA_MANIFEST_V2_SCHEMA_VERSION,
            text_cache_index_descriptor_path=descriptor_path,
        )


def test_manifest_header_dispatch_rejects_bool_and_mixed_pairs(fake_data, tmp_path):
    dataset, stats = fake_data
    descriptor_path = _build_v2_cache_index(tmp_path, dataset)
    manifest = build_robot_video_dataset_manifest(
        dataset,
        normalization_stats_path=stats,
        schema_version=DATA_MANIFEST_V2_SCHEMA_VERSION,
        text_cache_index_descriptor_path=descriptor_path,
    )

    with pytest.raises(ValueError, match="must be an integer"):
        require_supported_data_manifest_header(
            {"schema_version": True, "kind": DATA_MANIFEST_KIND}
        )
    mixed = copy.deepcopy(manifest)
    mixed["kind"] = DATA_MANIFEST_KIND
    _resign(mixed)
    with pytest.raises(ValueError, match="schema/kind pair"):
        validate_robot_video_dataset_manifest(
            dataset,
            mixed,
            normalization_stats_path=stats,
        )


def test_v1_default_and_explicit_schema_are_identical(fake_data):
    dataset, stats = fake_data
    implicit = build_robot_video_dataset_manifest(
        dataset,
        normalization_stats_path=stats,
    )
    explicit = build_robot_video_dataset_manifest(
        dataset,
        normalization_stats_path=stats,
        schema_version=1,
    )
    assert explicit == implicit


class _UniqueTaskIndexDataset:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def unique(self, column):
        self.calls.append(column)
        return self.result

    def __getitem__(self, _column):
        raise AssertionError("frame-level task_index fallback must not be read")


def test_selected_prompts_use_hf_unique_without_reading_frame_column(fake_data):
    dataset, _ = fake_data
    fallback_prompts = selected_text_cache_prompts(dataset)
    first_part = dataset.lerobot_dataset.multi_dataset._datasets[0]
    unique_dataset = _UniqueTaskIndexDataset([1, 0, 1])
    first_part.hf_dataset = unique_dataset

    assert selected_text_cache_prompts(dataset) == fallback_prompts
    assert unique_dataset.calls == ["task_index"]


@pytest.mark.parametrize(
    ("result", "error_type", "message"),
    [
        (None, TypeError, "must be iterable"),
        ([], ValueError, "contains no frame task indices"),
        ([True], ValueError, "must be an integer"),
        ([999], ValueError, "absent from metadata"),
    ],
)
def test_selected_prompts_reject_invalid_hf_unique_results(
    fake_data,
    result,
    error_type,
    message,
):
    dataset, _ = fake_data
    first_part = dataset.lerobot_dataset.multi_dataset._datasets[0]
    first_part.hf_dataset = _UniqueTaskIndexDataset(result)

    with pytest.raises(error_type, match=message):
        selected_text_cache_prompts(dataset)


def test_selected_prompts_reject_noncallable_hf_unique(fake_data):
    dataset, _ = fake_data
    first_part = dataset.lerobot_dataset.multi_dataset._datasets[0]

    class NonCallableUnique:
        unique = [1, 0]

        def __getitem__(self, _column):
            raise AssertionError("invalid unique must not trigger fallback")

    first_part.hf_dataset = NonCallableUnique()
    with pytest.raises(TypeError, match="unique must be callable"):
        selected_text_cache_prompts(dataset)


def test_public_v2_descriptor_resolver_validates_manifest_binding(
    fake_data,
    tmp_path,
):
    dataset, stats = fake_data
    descriptor_path = _build_v2_cache_index(tmp_path, dataset)
    manifest = build_robot_video_dataset_manifest(
        dataset,
        normalization_stats_path=stats,
        schema_version=DATA_MANIFEST_V2_SCHEMA_VERSION,
        text_cache_index_descriptor_path=descriptor_path,
    )

    wrong_descriptor = copy.deepcopy(manifest)
    wrong_descriptor["text_embedding_cache"]["integrity"][
        "descriptor_sha256"
    ] = "0" * 64
    _resign(wrong_descriptor)
    with pytest.raises(ValueError, match="does not bind"):
        resolve_text_cache_index_descriptor_path(wrong_descriptor)

    wrong_contract = copy.deepcopy(manifest)
    wrong_contract["text_embedding_cache"]["context_len"] += 1
    _resign(wrong_contract)
    with pytest.raises(ValueError, match="contract differs"):
        resolve_text_cache_index_descriptor_path(wrong_contract)

    wrong_index = copy.deepcopy(manifest)
    index_entry = next(
        entry
        for entry in wrong_index["text_embedding_cache"]["integrity"]["files"]
        if entry["role"] == "text_cache_index"
    )
    index_entry["sha256"] = "0" * 64
    _resign(wrong_index)
    with pytest.raises(ValueError, match="index entry differs"):
        resolve_text_cache_index_descriptor_path(wrong_index)
