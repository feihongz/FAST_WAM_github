from __future__ import annotations

from collections import OrderedDict
import pickle
from pathlib import Path

import pytest
import torch

import fastwam.alignment.text_cache_index as index_module
from fastwam.alignment.text_cache_index import (
    build_text_cache_index,
    prompt_sha256,
)
from fastwam.datasets.lerobot import current_robot_video_dataset as current_module
from fastwam.datasets.lerobot import robot_video_dataset as robot_module
from fastwam.datasets.lerobot.current_robot_video_dataset import (
    CurrentRobotVideoDataset,
)
from fastwam.datasets.lerobot.robot_video_dataset import (
    DEFAULT_PROMPT,
    RobotVideoDataset,
)


CONTEXT_LEN = 3
FILENAME_SUFFIX = ".t5_len3.wan22ti2v5b.pt"


def _payload_path(cache_root: Path, prompt: str, suffix: str) -> Path:
    return cache_root / f"{prompt_sha256(prompt)}{suffix}"


def _build_index(
    tmp_path: Path,
    prompts: list[str],
    *,
    cache_root: Path | None = None,
    context_len: int = CONTEXT_LEN,
    prompt_template: str = DEFAULT_PROMPT,
    filename_suffix: str = FILENAME_SUFFIX,
) -> tuple[Path, Path]:
    root = cache_root if cache_root is not None else tmp_path / "cache"
    root.mkdir(parents=True, exist_ok=True)
    for value, prompt in enumerate(dict.fromkeys(prompts), start=1):
        torch.save(
            {
                "context": torch.full(
                    (context_len, 4), value, dtype=torch.bfloat16
                ),
                "mask": torch.tensor(
                    [index % 2 == 0 for index in range(context_len)],
                    dtype=torch.bool,
                ),
            },
            _payload_path(root, prompt, filename_suffix),
        )
    descriptor_path = tmp_path / "identity" / "cache.index.json"
    build_text_cache_index(
        cache_root=root,
        prompts=prompts,
        context_len=context_len,
        prompt_template=prompt_template,
        filename_suffix=filename_suffix,
        index_path=tmp_path / "identity" / "cache.index",
        descriptor_path=descriptor_path,
    )
    return root, descriptor_path


def _bare_dataset(
    cache_root: Path,
    *,
    context_len: int = CONTEXT_LEN,
    cache_limit: int = 2,
) -> RobotVideoDataset:
    dataset = RobotVideoDataset.__new__(RobotVideoDataset)
    dataset.text_embedding_cache_dir = str(cache_root)
    dataset.context_len = context_len
    dataset.text_context_cache_max_entries = cache_limit
    dataset._text_context_cache = OrderedDict()
    dataset._text_cache_index_descriptor_path = None
    dataset._text_cache_index_expected_identity = None
    dataset._text_cache_index = None
    dataset._text_cache_index_pid = None
    return dataset


def _replace_receipt_with_same_contract(
    tmp_path: Path,
    *,
    cache_root: Path,
    descriptor_path: Path,
    prompts: list[str],
) -> None:
    for value, prompt in enumerate(dict.fromkeys(prompts), start=101):
        torch.save(
            {
                "context": torch.full(
                    (CONTEXT_LEN, 4), value, dtype=torch.bfloat16
                ),
                "mask": torch.tensor([True, False, True]),
            },
            _payload_path(cache_root, prompt, FILENAME_SUFFIX),
        )
    replacement_root = tmp_path / "replacement"
    replacement_descriptor = replacement_root / "cache.index.json"
    replacement_index = replacement_root / "cache.index"
    build_text_cache_index(
        cache_root=cache_root,
        prompts=prompts,
        context_len=CONTEXT_LEN,
        prompt_template=DEFAULT_PROMPT,
        filename_suffix=FILENAME_SUFFIX,
        index_path=replacement_index,
        descriptor_path=replacement_descriptor,
    )
    descriptor_path.with_name("cache.index").write_bytes(
        replacement_index.read_bytes()
    )
    descriptor_path.write_bytes(replacement_descriptor.read_bytes())


def test_full_loader_binding_is_lazy_clears_lru_and_keeps_clone_semantics(
    tmp_path,
):
    prompts = ["prompt-0", "prompt-1", "prompt-2"]
    cache_root, descriptor_path = _build_index(tmp_path, prompts)
    dataset = _bare_dataset(cache_root)
    dataset._text_context_cache["unverified"] = (
        torch.zeros(CONTEXT_LEN, 4),
        torch.zeros(CONTEXT_LEN, dtype=torch.bool),
    )

    dataset.bind_text_cache_index(descriptor_path)

    assert dataset._text_cache_index_descriptor_path == str(
        descriptor_path.resolve()
    )
    assert dataset._text_cache_index is None
    assert dataset._text_cache_index_pid is None
    assert not dataset._text_context_cache

    context, mask = dataset._get_cached_text_context("prompt-0")
    assert dataset._text_cache_index is not None
    assert dataset._text_cache_index_pid == robot_module.os.getpid()
    context.fill_(-1)
    mask.fill_(False)
    cached_context, cached_mask = dataset._get_cached_text_context("prompt-0")
    assert torch.all(cached_context == 1)
    assert cached_mask.tolist() == [True, False, True]

    dataset._get_cached_text_context("prompt-1")
    dataset._get_cached_text_context("prompt-0")
    dataset._get_cached_text_context("prompt-2")
    assert tuple(dataset._text_context_cache) == ("prompt-0", "prompt-2")


@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        ("cache_root", "cache_root"),
        ("context_len", "context_len"),
        ("prompt_template", "prompt_template"),
        ("filename_suffix", "filename_suffix"),
    ],
)
def test_binding_rejects_every_loader_contract_mismatch(
    tmp_path, mismatch, message
):
    build_kwargs = {}
    if mismatch == "prompt_template":
        build_kwargs["prompt_template"] = "Different template: {task}"
    elif mismatch == "filename_suffix":
        build_kwargs["filename_suffix"] = ".different.pt"
    cache_root, descriptor_path = _build_index(
        tmp_path, ["prompt"], **build_kwargs
    )
    dataset = _bare_dataset(cache_root)
    if mismatch == "cache_root":
        other_root = tmp_path / "other-cache"
        other_root.mkdir()
        dataset.text_embedding_cache_dir = str(other_root)
    elif mismatch == "context_len":
        dataset.context_len += 1

    with pytest.raises(ValueError, match=message):
        dataset.bind_text_cache_index(descriptor_path)
    assert dataset._text_cache_index_descriptor_path is None
    assert dataset._text_cache_index is None


def test_binding_rejects_index_tamper_before_persisting_path(tmp_path):
    cache_root, descriptor_path = _build_index(tmp_path, ["prompt"])
    index_path = descriptor_path.with_name("cache.index")
    changed = bytearray(index_path.read_bytes())
    changed[-1] ^= 1
    index_path.write_bytes(changed)
    dataset = _bare_dataset(cache_root)

    with pytest.raises(ValueError, match="index SHA256"):
        dataset.bind_text_cache_index(descriptor_path)
    assert dataset._text_cache_index_descriptor_path is None


def test_bound_loader_rejects_payload_tamper_before_torch_load(
    tmp_path, monkeypatch
):
    prompt = "prompt"
    cache_root, descriptor_path = _build_index(tmp_path, [prompt])
    dataset = _bare_dataset(cache_root)
    dataset.bind_text_cache_index(descriptor_path)
    payload_path = _payload_path(cache_root, prompt, FILENAME_SUFFIX)
    changed = bytearray(payload_path.read_bytes())
    changed[-1] ^= 1
    payload_path.write_bytes(changed)
    calls = 0

    def forbidden_torch_load(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("unverified bytes reached torch.load")

    monkeypatch.setattr(index_module.torch, "load", forbidden_torch_load)
    with pytest.raises(ValueError, match="payload SHA256"):
        dataset._get_cached_text_context(prompt)
    assert calls == 0


def test_bound_loader_never_falls_back_for_an_unindexed_prompt(tmp_path):
    cache_root, descriptor_path = _build_index(tmp_path, ["indexed"])
    dataset = _bare_dataset(cache_root)
    dataset.bind_text_cache_index(descriptor_path)

    with pytest.raises(KeyError, match="absent"):
        dataset._get_cached_text_context("not-indexed")


def test_full_loader_pickle_and_pid_change_reopen_verified_index(
    tmp_path, monkeypatch
):
    cache_root, descriptor_path = _build_index(tmp_path, ["prompt"])
    dataset = _bare_dataset(cache_root)
    dataset.bind_text_cache_index(descriptor_path)
    dataset._get_cached_text_context("prompt")
    original_index = dataset._text_cache_index

    restored = pickle.loads(pickle.dumps(dataset))
    assert restored._text_cache_index_descriptor_path == str(
        descriptor_path.resolve()
    )
    assert restored._text_cache_index is None
    assert restored._text_cache_index_pid is None
    assert not restored._text_context_cache
    restored._get_cached_text_context("prompt")
    opened_index = restored._text_cache_index
    assert opened_index is not None

    child_pid = robot_module.os.getpid() + 1000
    monkeypatch.setattr(robot_module.os, "getpid", lambda: child_pid)
    restored._get_cached_text_context("prompt")
    assert opened_index._mapping is None
    assert restored._text_cache_index is not opened_index
    assert restored._text_cache_index_pid == child_pid
    assert original_index._mapping is not None


def test_pickle_reopen_rejects_self_consistent_same_path_replacement(tmp_path):
    prompts = ["prompt-0", "prompt-1"]
    cache_root, descriptor_path = _build_index(tmp_path, prompts)
    dataset = _bare_dataset(cache_root)
    dataset.bind_text_cache_index(descriptor_path)
    pinned_identity = dataset._text_cache_index_expected_identity
    restored = pickle.loads(pickle.dumps(dataset))

    _replace_receipt_with_same_contract(
        tmp_path,
        cache_root=cache_root,
        descriptor_path=descriptor_path,
        prompts=prompts,
    )

    assert restored._text_cache_index_expected_identity == pinned_identity
    with pytest.raises(ValueError, match="immutable identity mismatch"):
        restored._get_cached_text_context("prompt-0")


def test_pid_reopen_rejects_self_consistent_same_path_replacement(
    tmp_path, monkeypatch
):
    prompts = ["prompt-0", "prompt-1"]
    cache_root, descriptor_path = _build_index(tmp_path, prompts)
    dataset = _bare_dataset(cache_root)
    dataset.bind_text_cache_index(descriptor_path)
    dataset._get_cached_text_context("prompt-0")

    _replace_receipt_with_same_contract(
        tmp_path,
        cache_root=cache_root,
        descriptor_path=descriptor_path,
        prompts=prompts,
    )
    monkeypatch.setattr(robot_module.os, "getpid", lambda: 987654321)

    with pytest.raises(ValueError, match="immutable identity mismatch"):
        dataset._get_cached_text_context("prompt-0")


class _Processor:
    def __init__(self) -> None:
        self.num_obs_steps = 2


class _SourceBase:
    def __init__(self) -> None:
        self.dataset_dirs = ["/data/robotwin"]
        self.shape_meta = {"images": [], "state": [], "action": []}
        self.val_set_proportion = 0.0
        self.is_training_set = True
        self.seed = 42
        self.global_sample_stride = 1
        self.video_backend = "torchcodec"
        self.strict_data_mode = True
        self.processor = _Processor()

    def __len__(self) -> int:
        return 1


class _CurrentBase:
    def __init__(self, **kwargs) -> None:
        self.strict_data_mode = kwargs["strict_data_mode"]
        self.processor = None

    def _set_return_images(self, flag: bool) -> None:
        self.return_images = flag

    def set_processor(self, processor):
        self.processor = processor
        return self

    def __len__(self) -> int:
        return 1


def _source_for_current(cache_root: Path) -> RobotVideoDataset:
    source = _bare_dataset(cache_root)
    source.lerobot_dataset = _SourceBase()
    source.strict_data_mode = True
    source.skip_padding_as_possible = False
    source.concat_multi_camera = "robotwin"
    source.override_instruction = None
    source.resize_transform = object()
    source.crop_transform = object()
    source.normalize_transform = object()
    return source


def test_current_loader_propagates_path_not_mmap_and_is_pickle_safe(
    tmp_path, monkeypatch
):
    prompts = ["prompt-0", "prompt-1"]
    cache_root, descriptor_path = _build_index(tmp_path, prompts)
    source = _source_for_current(cache_root)
    source.bind_text_cache_index(descriptor_path)
    source._get_cached_text_context("prompt-0")
    source_index = source._text_cache_index
    monkeypatch.setattr(current_module, "BaseLerobotDataset", _CurrentBase)

    current = CurrentRobotVideoDataset(source)
    assert current._text_cache_index_descriptor_path == str(
        descriptor_path.resolve()
    )
    assert (
        current._text_cache_index_expected_identity
        == source._text_cache_index_expected_identity
    )
    assert current._text_cache_index is None
    current._get_cached_text_context("prompt-0")
    assert current._text_cache_index is not None
    assert current._text_cache_index is not source_index

    restored = pickle.loads(pickle.dumps(current))
    assert restored._text_cache_index_descriptor_path == str(
        descriptor_path.resolve()
    )
    assert (
        restored._text_cache_index_expected_identity
        == source._text_cache_index_expected_identity
    )
    assert restored._text_cache_index is None
    assert restored._text_cache_index_pid is None
    assert not restored._text_context_cache
    restored._get_cached_text_context("prompt-0")
    assert restored._text_cache_index is not None

    payload_path = _payload_path(cache_root, "prompt-1", FILENAME_SUFFIX)
    changed = bytearray(payload_path.read_bytes())
    changed[-1] ^= 1
    payload_path.write_bytes(changed)
    with pytest.raises(ValueError, match="payload SHA256"):
        restored._get_cached_text_context("prompt-1")
