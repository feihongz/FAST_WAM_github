"""Strict current-observation source for Stage 2 label generation."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from copy import copy, deepcopy
from typing import Any

import torch

from fastwam.alignment.text_cache_index import TextCacheIndexIdentity

from .base_lerobot_dataset import BaseLerobotDataset
from .current_robot_video_dataset import CurrentRobotVideoDataset
from .robot_video_dataset import (
    DEFAULT_PROMPT,
    RobotVideoDataset,
    _bind_dataset_text_cache_index,
    _close_dataset_text_cache_index,
    _dataset_text_cache_getstate,
)


LABEL_INPUT_SCHEMA_VERSION = 1
_LABEL_SAMPLE_KEYS = frozenset(
    {
        "video",
        "action",
        "proprio",
        "prompt",
        "context",
        "context_mask",
        "gate_context_mask",
        "image_is_pad",
        "action_is_pad",
        "action_dim_is_pad",
        "proprio_is_pad",
        "sample_identity",
    }
)


class LabelRobotVideoDataset(torch.utils.data.Dataset):
    """Load t0 image/state plus the real future action horizon.

    The Stage 2 rollout interface still receives ``video`` and ``proprio`` with
    their historical shapes.  Their time dimensions are zero-copy expanded
    views of t0; no future observation is queried or decoded.  The normalized
    action target and its padding masks remain the exact full horizon.
    """

    label_input_schema_version = LABEL_INPUT_SCHEMA_VERSION

    def __init__(self, source: RobotVideoDataset) -> None:
        if not isinstance(source, RobotVideoDataset):
            raise TypeError("source must be a RobotVideoDataset")
        if not source.strict_data_mode:
            raise ValueError("label-only inputs require strict_data_mode")
        if source.skip_padding_as_possible:
            raise ValueError(
                "label-only inputs forbid padding-driven sample replacement"
            )

        source_base = source.lerobot_dataset
        if not bool(getattr(source_base, "strict_data_mode", False)):
            raise ValueError("underlying LeRobot source must enable strict_data_mode")
        source_processor = getattr(source_base, "processor", None)
        if source_processor is None:
            raise ValueError("label-only inputs require a configured processor")
        if not hasattr(source_processor, "num_obs_steps"):
            raise TypeError("label-only processor must expose num_obs_steps")

        action_horizon = getattr(source_base, "action_size", None)
        if (
            isinstance(action_horizon, bool)
            or not isinstance(action_horizon, int)
            or action_horizon < 1
        ):
            raise ValueError("source action horizon must be a positive integer")
        if action_horizon != source.num_frames - 1:
            raise ValueError("source action horizon differs from num_frames - 1")
        num_video_frames = len(source.video_sample_indices)
        if num_video_frames < 2 or num_video_frames % 4 != 1:
            raise ValueError(
                "source video frame count must be at least two and satisfy T % 4 == 1"
            )
        if action_horizon % (num_video_frames - 1) != 0:
            raise ValueError(
                "source action horizon must be divisible by video transitions"
            )

        label_processor = copy(source_processor)
        label_processor.num_obs_steps = 1
        if label_processor is source_processor:
            raise RuntimeError("label-only processor clone unexpectedly aliased source")

        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=list(source_base.dataset_dirs),
            shape_meta=deepcopy(source_base.shape_meta),
            obs_size=1,
            action_size=action_horizon,
            val_set_proportion=source_base.val_set_proportion,
            is_training_set=source_base.is_training_set,
            seed=source_base.seed,
            global_sample_stride=source_base.global_sample_stride,
            video_backend=source_base.video_backend,
            strict_data_mode=True,
            load_actions=True,
            allow_independent_action_horizon=True,
        )
        self.lerobot_dataset._set_return_images(True)
        self.lerobot_dataset.set_processor(label_processor)
        if len(self.lerobot_dataset) != len(source):
            raise ValueError("label-only source length differs from full video source")

        self.strict_data_mode = True
        self.skip_padding_as_possible = False
        self.processor = label_processor
        self.num_frames = source.num_frames
        self.action_horizon = action_horizon
        self.num_video_frames = num_video_frames
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
        elif source_identity is not None or getattr(
            source, "_text_cache_index", None
        ) is not None:
            raise RuntimeError(
                "source has a text cache identity without a verified descriptor"
            )

    def bind_text_cache_index(
        self,
        descriptor_path,
        expected_identity: TextCacheIndexIdentity | None = None,
    ) -> None:
        """Bind and independently verify the full source's v2 cache receipt."""

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

    def _get_cached_text_context(
        self, prompt: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Keep v1 and v2 payload validation exactly aligned with the full view.
        return RobotVideoDataset._get_cached_text_context(self, prompt)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("label-only input index must be an integer")
        if index < 0 or index >= len(self):
            raise IndexError(f"label-only input index {index} is out of range")

        sample = self.lerobot_dataset[index]
        if not isinstance(sample, Mapping):
            raise TypeError("label-only LeRobot sample must be a mapping")

        input_image = CurrentRobotVideoDataset._prepare_input_image(
            self, sample["pixel_values"]
        )
        action = sample["action"]
        proprio = sample["proprio"]
        if (
            not isinstance(action, torch.Tensor)
            or action.ndim != 2
            or action.shape[0] != self.action_horizon
        ):
            raise ValueError("label-only action has an unexpected horizon")
        if (
            not isinstance(proprio, torch.Tensor)
            or proprio.ndim != 2
            or proprio.shape[0] != 1
        ):
            raise ValueError("label-only proprio must have shape [1,D]")

        image_is_pad = sample["image_is_pad"]
        proprio_is_pad = sample["proprio_is_pad"]
        if (
            not isinstance(image_is_pad, torch.Tensor)
            or image_is_pad.ndim != 1
            or image_is_pad.shape[0] != 1
        ):
            raise ValueError("label-only image padding mask must have length one")
        if (
            not isinstance(proprio_is_pad, torch.Tensor)
            or proprio_is_pad.ndim != 1
            or proprio_is_pad.shape[0] != 1
        ):
            raise ValueError("label-only proprio padding mask must have length one")

        task = sample["instruction"]
        if self.override_instruction is not None:
            task = self.override_instruction
        prompt = DEFAULT_PROMPT.format(task=task)
        context, context_mask = self._get_cached_text_context(prompt)
        gate_context_mask = context_mask.clone()
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)

        identity = sample.get("sample_identity")
        if not isinstance(identity, Mapping):
            raise ValueError("label-only sample has no sample_identity mapping")
        result = {
            "video": input_image.unsqueeze(1).expand(
                -1, self.num_video_frames, -1, -1
            ),
            "action": action,
            "proprio": proprio.expand(self.action_horizon, -1),
            "prompt": prompt,
            "context": context,
            "context_mask": context_mask,
            "gate_context_mask": gate_context_mask,
            "image_is_pad": image_is_pad.expand(self.num_video_frames),
            "action_is_pad": sample["action_is_pad"],
            "action_dim_is_pad": sample["action_dim_is_pad"],
            "proprio_is_pad": proprio_is_pad.expand(self.num_frames),
            "sample_identity": dict(identity),
        }
        if set(result) != _LABEL_SAMPLE_KEYS:
            raise RuntimeError("label-only sample schema drifted")
        return result


__all__ = [
    "LABEL_INPUT_SCHEMA_VERSION",
    "LabelRobotVideoDataset",
]
