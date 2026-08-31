from __future__ import annotations

from copy import deepcopy
import sys
from types import ModuleType, SimpleNamespace

import pytest

from fastwam.gating import runtime_identity


FFMPEG_RUNTIME = {
    "executable_version": "ffmpeg version 4.4.2 deterministic-test",
    "torchcodec_runtime": {
        "ffmpeg_version": "4.4.2",
        "libraries": {
            "libavcodec": [58, 134, 100],
            "libavfilter": [7, 110, 100],
            "libavformat": [58, 76, 100],
            "libavutil": [56, 70, 100],
        },
    },
}


def _fake_torch_runtime():
    def resolve_device(value):
        index = int(str(value).split(":", 1)[1])
        return SimpleNamespace(type="cuda", index=index)

    return SimpleNamespace(
        __version__="2.7.1+cu128",
        device=resolve_device,
        version=SimpleNamespace(cuda="12.8"),
        backends=SimpleNamespace(
            cudnn=SimpleNamespace(
                version=lambda: 91002,
                benchmark=False,
                deterministic=True,
                allow_tf32=True,
            ),
            cuda=SimpleNamespace(
                matmul=SimpleNamespace(allow_tf32=False),
            ),
        ),
        cuda=SimpleNamespace(
            get_device_capability=lambda _device: (9, 0),
            get_device_name=lambda _device: "NVIDIA H100 80GB HBM3",
        ),
        are_deterministic_algorithms_enabled=lambda: True,
        is_deterministic_algorithms_warn_only_enabled=lambda: False,
    )


def _collect(device: str, **overrides):
    options = {
        "package_version_resolver": lambda name: f"{name}-version",
        "torch_runtime": _fake_torch_runtime(),
        "ffmpeg_runtime_resolver": lambda: deepcopy(FFMPEG_RUNTIME),
        "nvidia_driver_version_resolver": lambda: "580.173.02",
    }
    options.update(overrides)
    return runtime_identity.collect_numerical_runtime_environment(
        device,
        **options,
    )


def test_runtime_identity_is_rank_invariant_across_same_gpu_model():
    rank_zero = _collect("cuda:0")
    rank_three = _collect("cuda:3")

    assert rank_zero == rank_three
    assert rank_zero["ffmpeg"] == FFMPEG_RUNTIME
    assert rank_zero["device"]["nvidia_driver_version"] == "580.173.02"
    assert "index" not in rank_zero["device"]
    assert "uuid" not in rank_zero["device"]


def test_runtime_identity_fails_closed_on_incomplete_libav_report():
    incomplete = deepcopy(FFMPEG_RUNTIME)
    del incomplete["torchcodec_runtime"]["libraries"]["libavutil"]

    with pytest.raises(RuntimeError, match="required libav"):
        _collect(
            "cuda:0",
            ffmpeg_runtime_resolver=lambda: incomplete,
        )


def test_runtime_identity_fails_closed_on_unparseable_driver():
    with pytest.raises(RuntimeError, match="NVIDIA driver version"):
        _collect(
            "cuda:0",
            nvidia_driver_version_resolver=lambda: "unknown",
        )


def test_ffmpeg_collector_binds_torchcodec_loaded_library_versions(monkeypatch):
    fake_core = ModuleType("torchcodec._core")
    fake_core.get_ffmpeg_library_versions = lambda: {
        "libavformat": [58, 76, 100],
        "libavutil": [56, 70, 100],
        "libavfilter": [7, 110, 100],
        "libavcodec": [58, 134, 100],
        "ffmpeg_version": "4.4.2",
    }
    fake_package = ModuleType("torchcodec")
    fake_package.__path__ = []
    fake_package._core = fake_core
    monkeypatch.setitem(sys.modules, "torchcodec", fake_package)
    monkeypatch.setitem(sys.modules, "torchcodec._core", fake_core)

    def fake_run(command, **kwargs):
        assert command == ["ffmpeg", "-version"]
        assert kwargs["check"] is True
        return SimpleNamespace(
            stdout=(
                "ffmpeg version 4.4.2 deterministic-test\n"
                "libavutil 56.70.100\n"
            )
        )

    monkeypatch.setattr(runtime_identity.subprocess, "run", fake_run)

    assert runtime_identity._collect_ffmpeg_runtime_identity() == FFMPEG_RUNTIME


def test_driver_collector_requires_one_node_global_version(monkeypatch):
    monkeypatch.setattr(
        runtime_identity.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout="580.173.02\n580.173.02\n580.173.02\n580.173.02\n"
        ),
    )
    assert runtime_identity._collect_nvidia_driver_version() == "580.173.02"

    monkeypatch.setattr(
        runtime_identity.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout="580.173.02\n580.159.03\n"
        ),
    )
    with pytest.raises(RuntimeError, match="one identical version"):
        runtime_identity._collect_nvidia_driver_version()
