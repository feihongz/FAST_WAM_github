#!/usr/bin/env python3
"""Strict TorchCodec smoke for representative RoboTwin 2.0 AV1 videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from fastwam.datasets.lerobot.lerobot.datasets.video_utils import (
    decode_video_frames_torchcodec,
)


def _read_info(dataset_root: Path) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    try:
        payload = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read RoboTwin metadata: {info_path}") from error
    if not isinstance(payload, dict):
        raise TypeError("RoboTwin info.json must contain a mapping")
    return payload


def _video_keys(info: dict[str, Any]) -> tuple[str, ...]:
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("RoboTwin metadata has no feature mapping")
    keys = tuple(
        key
        for key, spec in features.items()
        if isinstance(key, str)
        and isinstance(spec, dict)
        and spec.get("dtype") == "video"
    )
    if len(keys) != 3 or len(set(keys)) != len(keys):
        raise ValueError("RoboTwin strict smoke requires exactly three video keys")
    return keys


def _episode_indices(info: dict[str, Any], raw: str | None) -> tuple[int, ...]:
    total = info.get("total_episodes")
    if isinstance(total, bool) or not isinstance(total, int) or total < 1:
        raise ValueError("RoboTwin total_episodes must be a positive integer")
    if raw is None:
        values = (0, (total - 1) // 2, total - 1)
    else:
        try:
            values = tuple(int(value.strip()) for value in raw.split(","))
        except ValueError as error:
            raise ValueError("--episodes must be comma-separated integers") from error
    if not values or len(values) != len(set(values)):
        raise ValueError("RoboTwin smoke episode indices must be non-empty and unique")
    if any(value < 0 or value >= total for value in values):
        raise ValueError("RoboTwin smoke episode index is out of range")
    return values


def _video_path(
    dataset_root: Path,
    info: dict[str, Any],
    *,
    episode_index: int,
    video_key: str,
) -> Path:
    template = info.get("video_path")
    chunk_size = info.get("chunks_size")
    if not isinstance(template, str) or not template:
        raise ValueError("RoboTwin metadata has no video_path template")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("RoboTwin chunks_size must be a positive integer")
    relative = template.format(
        episode_chunk=episode_index // chunk_size,
        episode_index=episode_index,
        video_key=video_key,
    )
    candidate = dataset_root / relative
    resolved_root = dataset_root.resolve(strict=True)
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"RoboTwin video path escapes dataset root: {candidate}") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"RoboTwin video is not a regular file: {resolved}")
    return resolved


def run_smoke(
    dataset_root: str | Path,
    *,
    episodes: str | None = None,
    expected_codec: str = "av1",
) -> list[dict[str, Any]]:
    root = Path(dataset_root).expanduser().resolve(strict=True)
    info = _read_info(root)
    keys = _video_keys(info)
    indices = _episode_indices(info, episodes)

    from torchcodec.decoders import VideoDecoder

    results: list[dict[str, Any]] = []
    for episode_index in indices:
        for video_key in keys:
            path = _video_path(
                root,
                info,
                episode_index=episode_index,
                video_key=video_key,
            )
            decoder = VideoDecoder(path, device="cpu", seek_mode="approximate")
            metadata = decoder.metadata
            if metadata.codec != expected_codec:
                raise ValueError(
                    f"unexpected codec for {path}: {metadata.codec!r}"
                )
            frame_count = len(decoder)
            fps = float(metadata.average_fps)
            if frame_count < 1 or fps <= 0.0:
                raise ValueError(f"invalid video metadata for {path}")
            frame_indices = (0, frame_count // 2, frame_count - 1)
            timestamps = [index / fps for index in frame_indices]
            frames = decode_video_frames_torchcodec(
                path,
                timestamps,
                tolerance_s=(0.51 / fps),
                device="cpu",
            )
            if tuple(frames.shape) != (3, 3, 480, 640):
                raise ValueError(
                    f"unexpected decoded shape for {path}: {tuple(frames.shape)}"
                )
            if frames.dtype != torch.float32 or not torch.isfinite(frames).all():
                raise ValueError(f"invalid decoded tensor for {path}")
            minimum = float(frames.min().item())
            maximum = float(frames.max().item())
            if minimum < 0.0 or maximum > 1.0:
                raise ValueError(f"decoded values are outside [0,1] for {path}")
            results.append(
                {
                    "episode_index": episode_index,
                    "video_key": video_key,
                    "path": str(path),
                    "codec": metadata.codec,
                    "frame_count": frame_count,
                    "sampled_frame_indices": frame_indices,
                    "shape": list(frames.shape),
                    "dtype": str(frames.dtype),
                    "value_range": [minimum, maximum],
                }
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default="./data/robotwin2.0/robotwin2.0",
        help="RoboTwin 2.0 LeRobot dataset root",
    )
    parser.add_argument(
        "--episodes",
        default=None,
        help="optional comma-separated episode indices; defaults to first/middle/last",
    )
    parser.add_argument("--expected-codec", default="av1")
    args = parser.parse_args()
    results = run_smoke(
        args.dataset_root,
        episodes=args.episodes,
        expected_codec=args.expected_codec,
    )
    print(json.dumps({"status": "ok", "videos": results}, indent=2))


if __name__ == "__main__":
    main()
