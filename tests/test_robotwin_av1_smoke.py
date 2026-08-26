from pathlib import Path

import pytest

from scripts.smoke_robotwin_av1 import (
    _episode_indices,
    _video_keys,
    _video_path,
)


def _info() -> dict:
    return {
        "total_episodes": 10,
        "chunks_size": 4,
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
        "features": {
            "observation.state": {"dtype": "float32"},
            "observation.images.cam_high": {"dtype": "video"},
            "observation.images.cam_left_wrist": {"dtype": "video"},
            "observation.images.cam_right_wrist": {"dtype": "video"},
        },
    }


def test_robotwin_av1_smoke_defaults_cover_first_middle_last():
    assert _episode_indices(_info(), None) == (0, 4, 9)
    assert _video_keys(_info()) == (
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    )


@pytest.mark.parametrize("raw", ["", "0,0", "-1", "10", "x"])
def test_robotwin_av1_smoke_rejects_invalid_episode_selection(raw):
    with pytest.raises(ValueError):
        _episode_indices(_info(), raw)


def test_robotwin_av1_smoke_resolves_metadata_template_under_root(tmp_path: Path):
    info = _info()
    expected = (
        tmp_path
        / "videos/chunk-001/observation.images.cam_high/episode_000005.mp4"
    )
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"not decoded by this path-only test")

    actual = _video_path(
        tmp_path,
        info,
        episode_index=5,
        video_key="observation.images.cam_high",
    )
    assert actual == expected.resolve()


def test_robotwin_av1_smoke_rejects_path_escape(tmp_path: Path):
    info = _info()
    info["video_path"] = "../outside.mp4"
    outside = tmp_path.parent / "outside.mp4"
    outside.write_bytes(b"outside")
    with pytest.raises(ValueError, match="escapes dataset root"):
        _video_path(
            tmp_path,
            info,
            episode_index=0,
            video_key="observation.images.cam_high",
        )
