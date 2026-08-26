"""Strict current-frame-only input source for Stage 2 Gate training."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from copy import copy, deepcopy
import hashlib
import os
from typing import Any

import torch
import torchvision.transforms.functional as transforms_F

from fastwam.alignment.text_cache_index import TextCacheIndexIdentity

from .base_lerobot_dataset import BaseLerobotDataset
from .robot_video_dataset import (
    DEFAULT_PROMPT,
    RobotVideoDataset,
    _bind_dataset_text_cache_index,
    _close_dataset_text_cache_index,
    _dataset_text_cache_getstate,
    _expected_text_cache_filename_suffix,
    _get_dataset_text_cache_index,
)


GATE_INPUT_SCHEMA_VERSION = 1
_GATE_INPUT_KEYS = frozenset(
    {"input_image", "context", "context_mask", "proprio", "sample_identity"}
)


class CurrentRobotVideoDataset(torch.utils.data.Dataset):
    """Load only the current image/state/text needed by the small Video Gate.

    This object owns a separate one-step LeRobot query plan and a separate
    processor object. Read-only transforms and normalization parameters are
    shared with the full video dataset, but num_obs_steps is never mutated on
    the label-generation processor.
    """

    gate_input_schema_version = GATE_INPUT_SCHEMA_VERSION

    def __init__(self, source: RobotVideoDataset) -> None:
        if not isinstance(source, RobotVideoDataset):
            raise TypeError("source must be a RobotVideoDataset")
        if not source.strict_data_mode:
            raise ValueError("current-only Gate inputs require strict_data_mode")
        if source.skip_padding_as_possible:
            raise ValueError(
                "current-only Gate inputs forbid padding-driven sample replacement"
            )

        source_base = source.lerobot_dataset
        if not bool(getattr(source_base, "strict_data_mode", False)):
            raise ValueError("underlying LeRobot source must enable strict_data_mode")
        source_processor = getattr(source_base, "processor", None)
        if source_processor is None:
            raise ValueError("current-only Gate inputs require a configured processor")
        if not hasattr(source_processor, "num_obs_steps"):
            raise TypeError("Gate input processor must expose num_obs_steps")

        current_processor = copy(source_processor)
        current_processor.num_obs_steps = 1
        if current_processor is source_processor:
            raise RuntimeError("current-only processor clone unexpectedly aliased source")

        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=list(source_base.dataset_dirs),
            shape_meta=deepcopy(source_base.shape_meta),
            obs_size=1,
            action_size=0,
            val_set_proportion=source_base.val_set_proportion,
            is_training_set=source_base.is_training_set,
            seed=source_base.seed,
            global_sample_stride=source_base.global_sample_stride,
            video_backend=source_base.video_backend,
            strict_data_mode=True,
            load_actions=False,
        )
        self.lerobot_dataset._set_return_images(True)
        self.lerobot_dataset.set_processor(current_processor)
        if len(self.lerobot_dataset) != len(source):
            raise ValueError(
                "current-only source length differs from the full video source"
            )

        self.strict_data_mode = True
        self.processor = current_processor
        self.concat_multi_camera = source.concat_multi_camera
        self.override_instruction = source.override_instruction
        self.text_embedding_cache_dir = source.text_embedding_cache_dir
        self.context_len = source.context_len
        cache_limit = getattr(source, "text_context_cache_max_entries", None)
        if isinstance(cache_limit, bool) or not isinstance(cache_limit, int):
            raise TypeError(
                "source.text_context_cache_max_entries must be an integer"
            )
        if cache_limit <= 0:
            raise ValueError(
                "source.text_context_cache_max_entries must be positive"
            )
        self.text_context_cache_max_entries = cache_limit
        self._text_context_cache: OrderedDict[
            str, tuple[torch.Tensor, torch.Tensor]
        ] = OrderedDict()
        self._text_cache_index_descriptor_path: str | None = None
        self._text_cache_index_expected_identity: (
            TextCacheIndexIdentity | None
        ) = None
        self._text_cache_index = None
        self._text_cache_index_pid: int | None = None
        self.resize_transform = source.resize_transform
        self.crop_transform = source.crop_transform
        self.normalize_transform = source.normalize_transform
        source_descriptor = getattr(
            source, "_text_cache_index_descriptor_path", None
        )
        source_identity = getattr(
            source, "_text_cache_index_expected_identity", None
        )
        if source_descriptor is not None:
            if not isinstance(source_identity, TextCacheIndexIdentity):
                raise RuntimeError(
                    "source text cache binding has no immutable identity"
                )
            self.bind_text_cache_index(source_descriptor, source_identity)
        elif source_identity is not None:
            raise RuntimeError(
                "source has a text cache identity without a descriptor"
            )

    def bind_text_cache_index(
        self,
        descriptor_path,
        expected_identity: TextCacheIndexIdentity | None = None,
    ) -> None:
        """Bind the same fail-closed v2 cache receipt as the full source."""

        _bind_dataset_text_cache_index(
            self,
            descriptor_path,
            expected_identity,
        )

    def __getstate__(self) -> dict:
        return _dataset_text_cache_getstate(self)

    def __del__(self) -> None:
        try:
            _close_dataset_text_cache_index(self)
        except Exception:
            pass

    def __len__(self) -> int:
        return len(self.lerobot_dataset)

    def _prepare_input_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        video = pixel_values
        if video.ndim == 5:
            num_cameras, time_steps, channels, height, width = video.shape
        elif video.ndim == 4:
            time_steps, channels, height, width = video.shape
            num_cameras = 1
            video = video.view(1, time_steps, channels, height, width)
        else:
            raise ValueError(
                "current-only pixel_values must have shape [T,C,H,W] or "
                f"[N,T,C,H,W], got {tuple(video.shape)}"
            )
        if time_steps != 1:
            raise ValueError(
                f"current-only source decoded {time_steps} frames instead of one"
            )

        if self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError(
                    "concat_multi_camera='robotwin' requires exactly 3 cameras"
                )
            cam_top = transforms_F.resize(
                video[0],
                size=[256, 320],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            cam_left = transforms_F.resize(
                video[1],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            cam_right = transforms_F.resize(
                video[2],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            video = torch.cat(
                [cam_top, torch.cat([cam_left, cam_right], dim=-1)],
                dim=-2,
            )
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat(
                    [video[index] for index in range(num_cameras)], dim=-1
                )
            elif self.concat_multi_camera == "vertical":
                video = torch.cat(
                    [video[index] for index in range(num_cameras)], dim=-2
                )
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical, robotwin."
                )
        else:
            video = video.squeeze(0)

        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)
        if video.ndim != 4 or video.shape[0] != 1 or video.shape[1] != 3:
            raise ValueError(
                "current-only transformed image must have shape [1,3,H,W]"
            )
        return video[0]

    def _get_cached_text_context(
        self, prompt: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_index = _get_dataset_text_cache_index(self)
        cached = self._text_context_cache.get(prompt)
        if cached is not None:
            self._text_context_cache.move_to_end(prompt)
            return cached[0].clone(), cached[1].clone()

        if cache_index is not None:
            payload = cache_index.load_verified_payload(prompt, map_location="cpu")
            cache_path = cache_index.descriptor_path
        else:
            if self.text_embedding_cache_dir is None:
                raise ValueError("text_embedding_cache_dir is not set")
            hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            suffix = _expected_text_cache_filename_suffix(self.context_len)
            cache_path = os.path.join(
                self.text_embedding_cache_dir,
                f"{hashed}{suffix}",
            )
            if not os.path.isfile(cache_path):
                raise FileNotFoundError(
                    f"Missing text embedding cache: {cache_path}. "
                    "Run scripts/precompute_text_embeds.py first."
                )
            payload = torch.load(cache_path, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise TypeError(f"text embedding cache is not a mapping: {cache_path}")
        context = payload.get("context")
        raw_mask = payload.get("mask")
        if not isinstance(context, torch.Tensor) or context.ndim != 2:
            raise ValueError(
                f"cached context must be [L,D] in {cache_path}"
            )
        if not isinstance(raw_mask, torch.Tensor) or raw_mask.ndim != 1:
            raise ValueError(
                f"cached context mask must be [L] in {cache_path}"
            )
        context_mask = raw_mask.bool()
        if (
            context.shape[0] != self.context_len
            or context_mask.shape[0] != self.context_len
        ):
            raise ValueError(
                f"cached context length must equal {self.context_len}: {cache_path}"
            )
        cached_context = context.detach().clone()
        cached_mask = context_mask.detach().clone()
        self._text_context_cache[prompt] = (
            cached_context,
            cached_mask,
        )
        while (
            len(self._text_context_cache)
            > self.text_context_cache_max_entries
        ):
            self._text_context_cache.popitem(last=False)
        return cached_context.clone(), cached_mask.clone()

    def __getitem__(self, index: int) -> dict[str, Any]:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("current-only Gate input index must be an integer")
        if index < 0 or index >= len(self):
            raise IndexError(
                f"current-only Gate input index {index} is out of range"
            )

        sample = self.lerobot_dataset[index]
        if not isinstance(sample, Mapping):
            raise TypeError("current-only LeRobot sample must be a mapping")
        if any(
            key == "action" or str(key).startswith("action.")
            for key in sample
        ):
            raise RuntimeError("current-only LeRobot sample exposed action data")

        input_image = self._prepare_input_image(sample["pixel_values"])
        proprio = sample["proprio"]
        if (
            not isinstance(proprio, torch.Tensor)
            or proprio.ndim != 2
            or proprio.shape[0] != 1
        ):
            raise ValueError("current-only proprio must have shape [1,D]")

        task = sample["instruction"]
        if self.override_instruction is not None:
            task = self.override_instruction
        prompt = DEFAULT_PROMPT.format(task=task)
        context, context_mask = self._get_cached_text_context(prompt)
        context[~context_mask] = 0.0

        identity = sample.get("sample_identity")
        if not isinstance(identity, Mapping):
            raise ValueError("current-only sample has no sample_identity mapping")
        result = {
            "input_image": input_image,
            "context": context,
            "context_mask": context_mask,
            "proprio": proprio[0],
            "sample_identity": dict(identity),
        }
        if set(result) != _GATE_INPUT_KEYS:
            raise RuntimeError("current-only Gate input schema drifted")
        return result


__all__ = [
    "CurrentRobotVideoDataset",
    "GATE_INPUT_SCHEMA_VERSION",
]
