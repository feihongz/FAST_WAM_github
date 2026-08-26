"""Lightweight numerical runtime identity shared by Stage 2 jobs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version as distribution_version
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


def _package_version(name: str) -> str | None:
    try:
        return distribution_version(name)
    except PackageNotFoundError:
        return None


def _optional_backend_bool(owner: Any, name: str) -> bool | None:
    value = getattr(owner, name, None)
    return None if value is None else bool(value)


def collect_numerical_runtime_environment(
    device: Any,
    *,
    package_version_resolver: Any = None,
    torch_runtime: Any = None,
) -> dict[str, Any]:
    """Collect software, accelerator, and backend state that can affect numerics."""

    resolver = package_version_resolver or _package_version
    runtime = torch if torch_runtime is None else torch_runtime
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
                "name": str(runtime.cuda.get_device_name(resolved_device)),
            }
        )
    payload = {
        "versions": versions,
        "device": device_identity,
        "backend": backend,
    }
    if not isinstance(payload, Mapping):
        raise RuntimeError("numerical runtime identity construction failed")
    return payload


__all__ = ["collect_numerical_runtime_environment"]
