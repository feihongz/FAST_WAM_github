"""Lightweight numerical runtime identity shared by Stage 2 jobs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version as distribution_version
import re
import subprocess
from typing import Any

import torch


_NUMERICAL_PACKAGES = (
    "torchcodec",
    "torchvision",
    "datasets",
    "pyarrow",
    "av",
    "numpy",
    "accelerate",
    "lerobot",
)


_REQUIRED_LIBAV_LIBRARIES = frozenset(
    {"libavcodec", "libavformat", "libavutil"}
)
_NVIDIA_DRIVER_VERSION_PATTERN = re.compile(r"^[0-9]+(?:\.[0-9]+)+$")


def _package_version(name: str) -> str | None:
    try:
        return distribution_version(name)
    except PackageNotFoundError:
        return None


def _optional_backend_bool(owner: Any, name: str) -> bool | None:
    value = getattr(owner, name, None)
    return None if value is None else bool(value)


def _normalize_library_version(name: str, value: Any) -> list[int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != 3
        or any(
            isinstance(component, bool) or not isinstance(component, int)
            for component in value
        )
        or any(component < 0 for component in value)
    ):
        raise RuntimeError(
            f"TorchCodec reported an invalid {name} version: {value!r}"
        )
    return [int(component) for component in value]


def _normalize_ffmpeg_runtime_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("FFmpeg runtime identity must be a mapping")
    if set(value) != {"executable_version", "torchcodec_runtime"}:
        raise RuntimeError("FFmpeg runtime identity fields are invalid")

    executable_version = value["executable_version"]
    if (
        not isinstance(executable_version, str)
        or executable_version != executable_version.strip()
        or not executable_version.startswith("ffmpeg version ")
        or "\n" in executable_version
        or "\r" in executable_version
    ):
        raise RuntimeError("FFmpeg executable version line is invalid")

    torchcodec_runtime = value["torchcodec_runtime"]
    if not isinstance(torchcodec_runtime, Mapping):
        raise RuntimeError("TorchCodec FFmpeg runtime identity must be a mapping")
    if set(torchcodec_runtime) != {"ffmpeg_version", "libraries"}:
        raise RuntimeError("TorchCodec FFmpeg runtime fields are invalid")

    ffmpeg_version = torchcodec_runtime["ffmpeg_version"]
    if (
        not isinstance(ffmpeg_version, str)
        or not ffmpeg_version
        or ffmpeg_version != ffmpeg_version.strip()
        or "\n" in ffmpeg_version
        or "\r" in ffmpeg_version
    ):
        raise RuntimeError("TorchCodec FFmpeg version is invalid")

    libraries_value = torchcodec_runtime["libraries"]
    if not isinstance(libraries_value, Mapping):
        raise RuntimeError("TorchCodec libav versions must be a mapping")
    library_names = set(libraries_value)
    if not _REQUIRED_LIBAV_LIBRARIES.issubset(library_names):
        missing = sorted(_REQUIRED_LIBAV_LIBRARIES - library_names)
        raise RuntimeError(
            f"TorchCodec did not report required libav libraries: {missing}"
        )
    if any(
        not isinstance(name, str) or not name.startswith("libav")
        for name in library_names
    ):
        raise RuntimeError("TorchCodec reported an invalid libav library name")

    libraries = {
        name: _normalize_library_version(name, libraries_value[name])
        for name in sorted(library_names)
    }
    return {
        "executable_version": executable_version,
        "torchcodec_runtime": {
            "ffmpeg_version": ffmpeg_version,
            "libraries": libraries,
        },
    }


def _collect_ffmpeg_runtime_identity() -> dict[str, Any]:
    """Collect the CLI and exact libav runtime loaded by TorchCodec."""

    try:
        completed = subprocess.run(
            ["ffmpeg", "-version"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError) as error:
        raise RuntimeError(
            "failed to identify the FFmpeg executable used by Stage 2"
        ) from error
    lines = completed.stdout.splitlines()
    if not lines:
        raise RuntimeError("ffmpeg -version returned no version line")

    try:
        from torchcodec._core import get_ffmpeg_library_versions

        reported = get_ffmpeg_library_versions()
    except Exception as error:
        raise RuntimeError(
            "failed to identify the libav runtime loaded by TorchCodec"
        ) from error
    if not isinstance(reported, Mapping):
        raise RuntimeError("TorchCodec FFmpeg version report must be a mapping")

    return _normalize_ffmpeg_runtime_identity(
        {
            "executable_version": lines[0],
            "torchcodec_runtime": {
                "ffmpeg_version": reported.get("ffmpeg_version"),
                "libraries": {
                    name: library_version
                    for name, library_version in reported.items()
                    if isinstance(name, str) and name.startswith("libav")
                },
            },
        }
    )


def _normalize_nvidia_driver_version(value: Any) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or _NVIDIA_DRIVER_VERSION_PATTERN.fullmatch(value) is None
    ):
        raise RuntimeError(f"NVIDIA driver version is invalid: {value!r}")
    return value


def _collect_nvidia_driver_version() -> str:
    """Return one node-global driver release without GPU/rank identifiers."""

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError) as error:
        raise RuntimeError("failed to identify the NVIDIA driver runtime") from error

    versions = {
        _normalize_nvidia_driver_version(line.strip())
        for line in completed.stdout.splitlines()
        if line.strip()
    }
    if len(versions) != 1:
        raise RuntimeError(
            "NVIDIA driver query must report one identical version across "
            f"all visible GPUs, got {sorted(versions)}"
        )
    return next(iter(versions))


def collect_numerical_runtime_environment(
    device: Any,
    *,
    package_version_resolver: Any = None,
    torch_runtime: Any = None,
    ffmpeg_runtime_resolver: Any = None,
    nvidia_driver_version_resolver: Any = None,
) -> dict[str, Any]:
    """Collect software, accelerator, and backend state that can affect numerics."""

    resolver = package_version_resolver or _package_version
    runtime = torch if torch_runtime is None else torch_runtime
    ffmpeg_resolver = (
        ffmpeg_runtime_resolver or _collect_ffmpeg_runtime_identity
    )
    driver_resolver = (
        nvidia_driver_version_resolver or _collect_nvidia_driver_version
    )
    resolved_device = runtime.device(str(device))
    versions = {"torch": str(runtime.__version__)}
    versions.update({name: resolver(name) for name in _NUMERICAL_PACKAGES})

    deterministic = getattr(
        runtime,
        "are_deterministic_algorithms_enabled",
        lambda: False,
    )
    warn_only = getattr(
        runtime,
        "is_deterministic_algorithms_warn_only_enabled",
        lambda: False,
    )
    cudnn_backend = runtime.backends.cudnn
    cuda_backend = getattr(runtime.backends, "cuda", None)
    matmul_backend = (
        None if cuda_backend is None else getattr(cuda_backend, "matmul", None)
    )
    backend = {
        "deterministic_algorithms": bool(deterministic()),
        "deterministic_warn_only": bool(warn_only()),
        "cudnn_benchmark": _optional_backend_bool(cudnn_backend, "benchmark"),
        "cudnn_deterministic": _optional_backend_bool(
            cudnn_backend, "deterministic"
        ),
        "cudnn_allow_tf32": _optional_backend_bool(cudnn_backend, "allow_tf32"),
        "cuda_matmul_allow_tf32": _optional_backend_bool(
            matmul_backend, "allow_tf32"
        ),
    }

    device_identity: dict[str, Any] = {"type": str(resolved_device.type)}
    if resolved_device.type == "cuda":
        capability = runtime.cuda.get_device_capability(resolved_device)
        if (
            not isinstance(capability, Sequence)
            or len(capability) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in capability
            )
        ):
            raise RuntimeError("CUDA device capability is invalid")
        device_identity.update(
            {
                "cuda_version": (
                    None
                    if runtime.version.cuda is None
                    else str(runtime.version.cuda)
                ),
                "cudnn_version": runtime.backends.cudnn.version(),
                "capability": [int(value) for value in capability],
                "nvidia_driver_version": _normalize_nvidia_driver_version(
                    driver_resolver()
                ),
                "name": str(runtime.cuda.get_device_name(resolved_device)),
            }
        )
    payload = {
        "versions": versions,
        "device": device_identity,
        "ffmpeg": _normalize_ffmpeg_runtime_identity(ffmpeg_resolver()),
        "backend": backend,
    }
    if not isinstance(payload, Mapping):
        raise RuntimeError("numerical runtime identity construction failed")
    return payload


__all__ = ["collect_numerical_runtime_environment"]
