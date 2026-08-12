"""Pure helpers for collecting paired LIBERO demonstration utility labels.

This module deliberately owns no dataset or model construction.  It receives one
already-processed ``RobotVideoDataset`` sample and calls a frozen-compatible
FastWAM inference interface twice.  Only the prefix length changes between the
two calls; in particular, future demonstration frames are never passed to the
model.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import torch


# Keep this mapping explicit: accepting arbitrary ``libero_*`` names would make
# provenance typos silently create a new suite in the collected dataset.
LIBERO_DATASET_TO_SUITE: dict[str, str] = {
    "libero_spatial_no_noops_lerobot": "libero_spatial",
    "libero_object_no_noops_lerobot": "libero_object",
    "libero_goal_no_noops_lerobot": "libero_goal",
    "libero_10_no_noops_lerobot": "libero_10",
}

INPUT_HASH_SCHEMA_VERSION = 1
INPUT_HASH_COMPONENTS = (
    "input_image",
    "proprio",
    "context",
    "context_mask",
    "valid_target_action",
    "action_is_pad",
)


def _scalar(value: Any, *, field: str) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"metadata {field!r} must be scalar, got shape {tuple(value.shape)}")
        return value.detach().cpu().item()
    # NumPy scalars and similar objects expose ``item``.  Do not call it on
    # strings or containers.
    if not isinstance(value, (str, bytes, list, tuple, dict)) and hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _required_int(metadata: Mapping[str, Any], field: str) -> int:
    if field not in metadata:
        raise ValueError(f"metadata is missing required field {field!r}")
    value = _scalar(metadata[field], field=field)
    if isinstance(value, bool):
        raise ValueError(f"metadata {field!r} must be an integer, not bool")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"metadata {field!r} must be an integer, got {value!r}") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"metadata {field!r} must be an integer, got {value!r}")
    if result < 0:
        raise ValueError(f"metadata {field!r} must be non-negative, got {result}")
    return result


def _canonical_dataset_name(value: Any) -> str:
    value = _scalar(value, field="dataset_name")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"dataset_name/dataset_id must be a non-empty string, got {value!r}")
    dataset_name = Path(value.rstrip("/")).name
    if dataset_name not in LIBERO_DATASET_TO_SUITE:
        raise ValueError(
            f"unknown LIBERO dataset {dataset_name!r}; expected one of "
            f"{sorted(LIBERO_DATASET_TO_SUITE)}"
        )
    return dataset_name


@dataclass(frozen=True)
class SampleIdentity:
    """Stable, human-auditable identity for one source demonstration frame."""

    dataset_id: str
    dataset_name: str
    suite: str
    episode_index: int
    frame_index: int
    task_index: int
    task: str

    def __post_init__(self) -> None:
        canonical = _canonical_dataset_name(self.dataset_name)
        if self.dataset_name != canonical:
            raise ValueError(
                f"dataset_name must be canonical basename {canonical!r}, got {self.dataset_name!r}"
            )
        if self.dataset_id != canonical:
            raise ValueError(
                "dataset_id must use the canonical dataset basename so IDs and seeds do not "
                f"depend on a machine-specific root: expected {canonical!r}, got {self.dataset_id!r}"
            )
        expected_suite = LIBERO_DATASET_TO_SUITE[canonical]
        if self.suite != expected_suite:
            raise ValueError(
                f"suite mismatch for {canonical!r}: expected {expected_suite!r}, got {self.suite!r}"
            )
        for field in ("episode_index", "frame_index", "task_index"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer, got {value!r}")
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValueError("task must be a non-empty string")

    @property
    def sample_id(self) -> str:
        return (
            f"{self.dataset_id}/episode_{self.episode_index:06d}/"
            f"frame_{self.frame_index:06d}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "suite": self.suite,
            "episode_index": self.episode_index,
            "episode_id": self.episode_index,
            "frame_index": self.frame_index,
            "task_index": self.task_index,
            "task_id": self.task_index,
            "task_id_source": "lerobot_task_index",
            "task": self.task,
        }


def parse_sample_identity(
    metadata: Mapping[str, Any],
    *,
    task_by_index: Optional[Mapping[int, str]] = None,
) -> SampleIdentity:
    """Parse and cross-check source metadata without machine-specific paths.

    ``dataset_name``, ``dataset_id`` and ``repo_id`` are accepted as source
    aliases, but any simultaneously present aliases must resolve to the same
    known LIBERO dataset.  ``task_by_index`` provides the strongest available
    task-string/index validation when the caller has the LeRobot task table.
    """

    if not isinstance(metadata, Mapping):
        raise TypeError(f"metadata must be a mapping, got {type(metadata).__name__}")

    dataset_fields = {
        key: metadata[key]
        for key in ("dataset_name", "dataset_id", "repo_id")
        if key in metadata and metadata[key] is not None
    }
    dataset_values = list(dataset_fields.values())
    if not dataset_values:
        raise ValueError("metadata must contain dataset_name, dataset_id, or repo_id")
    canonical_names = [_canonical_dataset_name(value) for value in dataset_values]
    if len(set(canonical_names)) != 1:
        raise ValueError(f"conflicting dataset identifiers in metadata: {canonical_names}")
    dataset_name = canonical_names[0]
    suite = LIBERO_DATASET_TO_SUITE[dataset_name]

    if "suite" in metadata:
        reported_suite = _scalar(metadata["suite"], field="suite")
        if reported_suite != suite:
            raise ValueError(
                f"metadata suite mismatch for {dataset_name!r}: "
                f"expected {suite!r}, got {reported_suite!r}"
            )

    episode_index = _required_int(metadata, "episode_index")
    frame_index = _required_int(metadata, "frame_index")
    task_index = _required_int(metadata, "task_index")

    task_values: list[str] = []
    for key in ("task", "task_string", "source_task"):
        if key not in metadata or metadata[key] is None:
            continue
        value = _scalar(metadata[key], field=key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"metadata {key!r} must be a non-empty string, got {value!r}")
        task_values.append(value)
    if not task_values:
        raise ValueError("metadata must contain task, task_string, or source_task")
    if len(set(task_values)) != 1:
        raise ValueError(f"conflicting task strings in metadata: {task_values}")
    task = task_values[0]

    if task_by_index is not None:
        if task_index not in task_by_index:
            raise ValueError(f"task_index {task_index} is absent from the source task table")
        expected_task = task_by_index[task_index]
        if not isinstance(expected_task, str) or task != expected_task:
            raise ValueError(
                f"task string/index mismatch: index {task_index} maps to "
                f"{expected_task!r}, metadata contains {task!r}"
            )

    identity = SampleIdentity(
        dataset_id=dataset_name,
        dataset_name=dataset_name,
        suite=suite,
        episode_index=episode_index,
        frame_index=frame_index,
        task_index=task_index,
        task=task,
    )
    if "sample_id" in metadata and metadata["sample_id"] != identity.sample_id:
        raise ValueError(
            f"metadata sample_id mismatch: expected {identity.sample_id!r}, "
            f"got {metadata['sample_id']!r}"
        )
    return identity


def stable_sample_seed(
    base_seed: int,
    identity_or_dataset_id: SampleIdentity | str,
    episode_index: Optional[int] = None,
    frame_index: Optional[int] = None,
) -> int:
    """Derive a scheduling-independent inference seed using SHA256.

    The first 64 digest bits are masked to the non-negative signed-int64 range,
    which is portable across ``torch.Generator`` backends.
    """

    if isinstance(base_seed, bool):
        raise ValueError("base_seed must be an integer, not bool")
    base_seed = int(base_seed)
    if isinstance(identity_or_dataset_id, SampleIdentity):
        if episode_index is not None or frame_index is not None:
            raise ValueError("episode_index/frame_index must be omitted when passing SampleIdentity")
        dataset_id = identity_or_dataset_id.dataset_id
        episode_index = identity_or_dataset_id.episode_index
        frame_index = identity_or_dataset_id.frame_index
    else:
        dataset_id = _canonical_dataset_name(identity_or_dataset_id)
        if episode_index is None or frame_index is None:
            raise ValueError("episode_index and frame_index are required with a dataset_id")
        if isinstance(episode_index, bool) or isinstance(frame_index, bool):
            raise ValueError("episode_index/frame_index must be integers, not bool")
        episode_index = int(episode_index)
        frame_index = int(frame_index)
        if episode_index < 0 or frame_index < 0:
            raise ValueError("episode_index/frame_index must be non-negative")

    payload = f"{base_seed}\0{dataset_id}\0{episode_index}\0{frame_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big") & ((1 << 63) - 1)


@dataclass(frozen=True)
class CurrentState:
    """Only the model inputs available at the current demonstration frame."""

    input_image: torch.Tensor
    proprio: torch.Tensor
    context: torch.Tensor
    context_mask: torch.Tensor
    target_action: torch.Tensor
    action_is_pad: torch.Tensor
    source_video_frames: int


def tensor_content_sha256(tensor: torch.Tensor) -> str:
    """Hash a tensor's exact dtype, shape, and contiguous CPU bytes."""

    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    value = tensor.detach().to(device="cpu").contiguous()
    byte_view = value.reshape(-1).view(torch.uint8)
    header = json.dumps(
        {
            "schema_version": INPUT_HASH_SCHEMA_VERSION,
            "dtype": str(value.dtype),
            "shape": [int(dim) for dim in value.shape],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(byte_view.numpy().tobytes(order="C"))
    return digest.hexdigest()


def current_state_input_hashes(state: CurrentState) -> dict[str, str]:
    """Fingerprint every tensor that can affect paired inference or utility.

    The video hash deliberately covers only ``input_image`` (the cloned current
    frame), never a future demonstration frame.
    """

    if not isinstance(state, CurrentState):
        raise TypeError(f"state must be CurrentState, got {type(state).__name__}")
    valid = ~state.action_is_pad.to(dtype=torch.bool)
    components = {
        "input_image": state.input_image,
        "proprio": state.proprio,
        "context": state.context,
        "context_mask": state.context_mask,
        "valid_target_action": state.target_action[valid],
        "action_is_pad": state.action_is_pad,
    }
    hashes = {
        name: tensor_content_sha256(components[name]) for name in INPUT_HASH_COMPONENTS
    }
    combined_payload = json.dumps(
        hashes,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    hashes["combined"] = hashlib.sha256(combined_payload).hexdigest()
    return hashes


def _tensor(sample: Mapping[str, Any], key: str) -> torch.Tensor:
    if key not in sample:
        raise ValueError(f"sample is missing required field {key!r}")
    value = sample[key]
    if not isinstance(value, torch.Tensor):
        try:
            value = torch.as_tensor(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"sample field {key!r} must be tensor-like") from exc
    return value


def extract_current_state(sample: Mapping[str, Any]) -> CurrentState:
    """Extract t=0 inputs and GT action; never return a future video frame."""

    if not isinstance(sample, Mapping):
        raise TypeError(f"sample must be a mapping, got {type(sample).__name__}")

    video = _tensor(sample, "video")
    if video.ndim != 4 or video.shape[0] != 3:
        raise ValueError(f"video must have shape [3,T,H,W], got {tuple(video.shape)}")
    if video.shape[1] < 1 or video.shape[2] < 1 or video.shape[3] < 1:
        raise ValueError(f"video dimensions must be non-empty, got {tuple(video.shape)}")

    proprio = _tensor(sample, "proprio")
    if proprio.ndim != 2 or proprio.shape[0] < 1 or proprio.shape[1] < 1:
        raise ValueError(f"proprio must have shape [T,D] with T,D>0, got {tuple(proprio.shape)}")

    context = _tensor(sample, "context")
    context_mask = _tensor(sample, "context_mask")
    if context.ndim != 2 or context.shape[0] < 1 or context.shape[1] < 1:
        raise ValueError(f"context must have shape [L,D] with L,D>0, got {tuple(context.shape)}")
    if context_mask.ndim != 1 or context_mask.shape[0] != context.shape[0]:
        raise ValueError(
            "context_mask must have shape [L] matching context; got "
            f"{tuple(context_mask.shape)} and {tuple(context.shape)}"
        )

    action = _tensor(sample, "action")
    action_is_pad = _tensor(sample, "action_is_pad")
    if action.ndim != 2 or action.shape[0] < 1 or action.shape[1] < 1:
        raise ValueError(f"action must have shape [T,A] with T,A>0, got {tuple(action.shape)}")
    if action_is_pad.ndim != 1 or action_is_pad.shape[0] != action.shape[0]:
        raise ValueError(
            "action_is_pad must have shape [T] matching action; got "
            f"{tuple(action_is_pad.shape)} and {tuple(action.shape)}"
        )

    # Clone each t=0 slice so subsequent dataset-buffer reuse or deliberate
    # changes to future frames cannot affect the paired inputs.
    return CurrentState(
        input_image=video[:, 0].detach().clone(),
        proprio=proprio[0].detach().clone(),
        context=context.detach().clone(),
        context_mask=context_mask.detach().to(dtype=torch.bool).clone(),
        target_action=action.detach().clone(),
        action_is_pad=action_is_pad.detach().to(dtype=torch.bool).clone(),
        source_video_frames=int(video.shape[1]),
    )


def valid_chunk_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    action_is_pad: torch.Tensor,
) -> tuple[float, int]:
    """Return normalized-space per-element MSE over valid action timesteps."""

    if not isinstance(prediction, torch.Tensor):
        prediction = torch.as_tensor(prediction)
    if not isinstance(target, torch.Tensor):
        target = torch.as_tensor(target)
    if not isinstance(action_is_pad, torch.Tensor):
        action_is_pad = torch.as_tensor(action_is_pad)
    if prediction.ndim != 2 or target.ndim != 2 or prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have identical [T,A] shapes, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if action_is_pad.ndim != 1 or action_is_pad.shape[0] != target.shape[0]:
        raise ValueError(
            f"action_is_pad must be [T]={target.shape[0]}, got {tuple(action_is_pad.shape)}"
        )
    valid = ~action_is_pad.to(device=prediction.device, dtype=torch.bool)
    valid_length = int(valid.sum().item())
    if valid_length == 0:
        raise ValueError("cannot compute utility for an all-padding action chunk")

    prediction_f64 = prediction.to(dtype=torch.float64)
    target_f64 = target.to(device=prediction.device, dtype=torch.float64)
    valid_prediction = prediction_f64[valid]
    valid_target = target_f64[valid]
    if not torch.isfinite(valid_prediction).all() or not torch.isfinite(valid_target).all():
        raise ValueError("valid prediction/target elements must all be finite")
    return float(torch.mean((valid_prediction - valid_target) ** 2).item()), valid_length


def _validate_route(
    prediction: Any,
    *,
    expected_prefix_steps: int,
    num_inference_steps: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not isinstance(prediction, Mapping):
        raise TypeError(f"model inference must return a mapping, got {type(prediction).__name__}")
    missing = {
        key
        for key in ("action", "video_prefix_steps", "num_inference_steps", "force_custom_prefix")
        if key not in prediction
    }
    if missing:
        raise AssertionError(f"model result is missing route metadata: {sorted(missing)}")
    actual_prefix = int(_scalar(prediction["video_prefix_steps"], field="video_prefix_steps"))
    actual_steps = int(_scalar(prediction["num_inference_steps"], field="num_inference_steps"))
    raw_custom_prefix = _scalar(prediction["force_custom_prefix"], field="force_custom_prefix")
    if not isinstance(raw_custom_prefix, bool):
        raise AssertionError(
            "model route metadata force_custom_prefix must be bool, "
            f"got {raw_custom_prefix!r}"
        )
    custom_prefix = raw_custom_prefix
    if actual_prefix != expected_prefix_steps:
        raise AssertionError(
            f"prefix route mismatch: requested {expected_prefix_steps}, model reported {actual_prefix}"
        )
    if actual_steps != num_inference_steps:
        raise AssertionError(
            f"inference-step mismatch: requested {num_inference_steps}, model reported {actual_steps}"
        )
    if not custom_prefix:
        raise AssertionError("paired utility collection requires model-reported force_custom_prefix=true")

    action = prediction["action"]
    if not isinstance(action, torch.Tensor):
        action = torch.as_tensor(action)
    if action.ndim != 2:
        raise ValueError(f"model action must have shape [T,A], got {tuple(action.shape)}")
    if not torch.isfinite(action).all():
        raise ValueError("model action prediction contains non-finite values")
    route = {
        "inference_mode": "prefix",
        "video_prefix_steps": actual_prefix,
        "num_inference_steps": actual_steps,
        "force_custom_prefix": custom_prefix,
    }
    return action.detach(), route


def _model_cuda_device(model: Any) -> Optional[torch.device]:
    """Return the model CUDA device used for accurate wall-clock timing."""

    raw_device = getattr(model, "device", None)
    if raw_device is None and hasattr(model, "parameters"):
        try:
            raw_device = next(model.parameters()).device
        except (StopIteration, TypeError):
            raw_device = None
    if raw_device is None:
        return None
    try:
        device = torch.device(raw_device)
    except (TypeError, RuntimeError):
        return None
    if device.type != "cuda":
        return None
    if not torch.cuda.is_available():
        raise RuntimeError("model is on CUDA but torch.cuda.is_available() is false")
    return device


def _synchronize_cuda(device: Optional[torch.device]) -> None:
    if device is not None:
        torch.cuda.synchronize(device)


@dataclass(frozen=True)
class DemoUtilityRecord:
    """One JSON-safe paired N=0/N=full utility label."""

    identity: SampleIdentity
    seed: int
    num_inference_steps: int
    n0: int
    nfull: int
    e0: float
    efull: float
    utility: float
    valid_length: int
    target_action_shape: tuple[int, int]
    pred_n0_shape: tuple[int, int]
    pred_nfull_shape: tuple[int, int]
    input_hashes: Mapping[str, str]
    n0_latency_ms: float
    nfull_latency_ms: float
    total_latency_ms: float
    n0_route: Mapping[str, Any]
    nfull_route: Mapping[str, Any]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported record schema_version {self.schema_version}")
        if self.n0 != 0 or self.nfull != self.num_inference_steps or self.nfull <= 0:
            raise ValueError(
                "record endpoints must be N=0 and N=full=num_inference_steps; got "
                f"n0={self.n0}, nfull={self.nfull}, steps={self.num_inference_steps}"
            )
        if self.valid_length <= 0 or self.valid_length > self.target_action_shape[0]:
            raise ValueError(f"invalid valid_length {self.valid_length}")
        if not (
            self.target_action_shape == self.pred_n0_shape == self.pred_nfull_shape
            and len(self.target_action_shape) == 2
        ):
            raise ValueError("target and paired prediction shapes must be identical [T,A]")
        for field in ("e0", "efull", "utility", "n0_latency_ms", "nfull_latency_ms", "total_latency_ms"):
            value = float(getattr(self, field))
            if not math.isfinite(value):
                raise ValueError(f"record field {field} must be finite, got {value}")
        if self.e0 < 0 or self.efull < 0:
            raise ValueError("MSE values must be non-negative")
        if self.n0_latency_ms < 0 or self.nfull_latency_ms < 0 or self.total_latency_ms < 0:
            raise ValueError("latencies must be non-negative")
        if not math.isclose(self.utility, self.e0 - self.efull, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("utility must equal e0 - efull")
        expected_hash_keys = set(INPUT_HASH_COMPONENTS) | {"combined"}
        if set(self.input_hashes) != expected_hash_keys:
            raise ValueError(
                "input_hashes keys must exactly match "
                f"{sorted(expected_hash_keys)}, got {sorted(self.input_hashes)}"
            )
        for name, value in self.input_hashes.items():
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"input_hashes[{name!r}] is not a lowercase SHA-256")
        component_hashes = {
            name: self.input_hashes[name] for name in INPUT_HASH_COMPONENTS
        }
        expected_combined = hashlib.sha256(
            json.dumps(
                component_hashes,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if self.input_hashes["combined"] != expected_combined:
            raise ValueError("input_hashes combined digest does not match its components")

    @property
    def sample_id(self) -> str:
        return self.identity.sample_id

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            **self.identity.to_dict(),
            "seed": int(self.seed),
            "num_inference_steps": int(self.num_inference_steps),
            "n0": int(self.n0),
            "nfull": int(self.nfull),
            "e0": float(self.e0),
            "efull": float(self.efull),
            "utility": float(self.utility),
            "valid_length": int(self.valid_length),
            "target_action_shape": list(self.target_action_shape),
            "pred_n0_shape": list(self.pred_n0_shape),
            "pred_nfull_shape": list(self.pred_nfull_shape),
            "input_hashes": dict(self.input_hashes),
            "n0_latency_ms": float(self.n0_latency_ms),
            "nfull_latency_ms": float(self.nfull_latency_ms),
            "total_latency_ms": float(self.total_latency_ms),
            "n0_route": dict(self.n0_route),
            "nfull_route": dict(self.nfull_route),
        }
        return result


def _resolve_metadata(sample: Mapping[str, Any], metadata: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if metadata is not None:
        return metadata
    for key in ("metadata", "source_metadata"):
        candidate = sample.get(key)
        if isinstance(candidate, Mapping):
            return candidate
    return sample


def collect_paired_utility(
    model: Any,
    sample: Mapping[str, Any],
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    sample_id: Optional[str] = None,
    base_seed: int = 42,
    seed: Optional[int] = None,
    num_inference_steps: int = 10,
    full_prefix_steps: Optional[int] = None,
    num_video_frames: Optional[int] = None,
    rand_device: str = "cpu",
    sigma_shift: Optional[float] = None,
    tiled: bool = False,
    force_custom_prefix: bool = True,
    extra_inference_kwargs: Optional[Mapping[str, Any]] = None,
    task_by_index: Optional[Mapping[int, str]] = None,
) -> DemoUtilityRecord:
    """Collect a paired N=0/N=full label from exactly one current state.

    Both calls receive the same image/proprio/context objects, seed, random
    device and inference options.  The sole changed inference kwarg is
    ``video_prefix_steps``.
    """

    if not hasattr(model, "infer_action_mode"):
        raise TypeError("model must provide infer_action_mode")
    num_inference_steps = int(num_inference_steps)
    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")
    if full_prefix_steps is None:
        full_prefix_steps = num_inference_steps
    full_prefix_steps = int(full_prefix_steps)
    if full_prefix_steps != num_inference_steps:
        raise ValueError(
            "paired endpoint collection requires full_prefix_steps == num_inference_steps, "
            f"got {full_prefix_steps} and {num_inference_steps}"
        )
    if not force_custom_prefix:
        raise ValueError("paired utility collection requires force_custom_prefix=True")
    if not isinstance(rand_device, str) or not rand_device:
        raise ValueError("rand_device must be a non-empty string")

    identity = parse_sample_identity(
        _resolve_metadata(sample, metadata),
        task_by_index=task_by_index,
    )
    if sample_id is not None and sample_id != identity.sample_id:
        raise ValueError(
            f"sample_id mismatch: parsed {identity.sample_id!r}, caller supplied {sample_id!r}"
        )
    expected_seed = stable_sample_seed(base_seed, identity)
    if seed is None:
        seed = expected_seed
    elif isinstance(seed, bool):
        raise ValueError("seed must be an integer, not bool")
    else:
        seed = int(seed)
        if seed != expected_seed:
            raise ValueError(
                "explicit seed must equal stable_sample_seed(base_seed, identity): "
                f"expected {expected_seed}, got {seed}"
            )

    state = extract_current_state(sample)
    input_hashes = current_state_input_hashes(state)
    if num_video_frames is None:
        num_video_frames = state.source_video_frames
    num_video_frames = int(num_video_frames)
    if num_video_frames <= 0:
        raise ValueError("num_video_frames must be positive")

    common_kwargs: dict[str, Any] = {
        "prompt": None,
        "input_image": state.input_image,
        "action_horizon": int(state.target_action.shape[0]),
        "num_video_frames": num_video_frames,
        "proprio": state.proprio,
        "context": state.context,
        "context_mask": state.context_mask,
        "num_inference_steps": num_inference_steps,
        "sigma_shift": sigma_shift,
        "seed": seed,
        "rand_device": rand_device,
        "tiled": bool(tiled),
        "force_custom_prefix": True,
    }
    if extra_inference_kwargs:
        reserved = set(common_kwargs) | {"inference_mode", "video_prefix_steps"}
        conflicts = reserved.intersection(extra_inference_kwargs)
        if conflicts:
            raise ValueError(f"extra_inference_kwargs cannot override paired fields: {sorted(conflicts)}")
        common_kwargs.update(extra_inference_kwargs)

    timing_device = _model_cuda_device(model)
    _synchronize_cuda(timing_device)
    pair_start = time.perf_counter()
    with torch.inference_mode():
        _synchronize_cuda(timing_device)
        n0_start = time.perf_counter()
        raw_n0 = model.infer_action_mode(
            **common_kwargs,
            inference_mode="prefix",
            video_prefix_steps=0,
        )
        _synchronize_cuda(timing_device)
        n0_latency_ms = (time.perf_counter() - n0_start) * 1000.0

        _synchronize_cuda(timing_device)
        nfull_start = time.perf_counter()
        raw_nfull = model.infer_action_mode(
            **common_kwargs,
            inference_mode="prefix",
            video_prefix_steps=full_prefix_steps,
        )
        _synchronize_cuda(timing_device)
        nfull_latency_ms = (time.perf_counter() - nfull_start) * 1000.0
    _synchronize_cuda(timing_device)
    total_latency_ms = (time.perf_counter() - pair_start) * 1000.0

    pred_n0, n0_route = _validate_route(
        raw_n0,
        expected_prefix_steps=0,
        num_inference_steps=num_inference_steps,
    )
    pred_nfull, nfull_route = _validate_route(
        raw_nfull,
        expected_prefix_steps=full_prefix_steps,
        num_inference_steps=num_inference_steps,
    )
    e0, valid_length = valid_chunk_mse(pred_n0, state.target_action, state.action_is_pad)
    efull, full_valid_length = valid_chunk_mse(
        pred_nfull,
        state.target_action,
        state.action_is_pad,
    )
    if full_valid_length != valid_length:
        raise AssertionError("paired endpoint metrics used different valid lengths")

    def _shape2(tensor: torch.Tensor, name: str) -> tuple[int, int]:
        if tensor.ndim != 2:
            raise ValueError(f"{name} must be [T,A], got shape {tuple(tensor.shape)}")
        return int(tensor.shape[0]), int(tensor.shape[1])

    target_shape = _shape2(state.target_action, "target action")
    n0_shape = _shape2(pred_n0, "N=0 prediction")
    nfull_shape = _shape2(pred_nfull, "N=full prediction")
    if not (target_shape == n0_shape == nfull_shape):
        raise ValueError(
            f"paired action shape mismatch: target={target_shape}, N=0={n0_shape}, "
            f"N=full={nfull_shape}"
        )

    return DemoUtilityRecord(
        identity=identity,
        seed=seed,
        num_inference_steps=num_inference_steps,
        n0=0,
        nfull=full_prefix_steps,
        e0=e0,
        efull=efull,
        utility=e0 - efull,
        valid_length=valid_length,
        target_action_shape=target_shape,
        pred_n0_shape=n0_shape,
        pred_nfull_shape=nfull_shape,
        input_hashes=input_hashes,
        n0_latency_ms=n0_latency_ms,
        nfull_latency_ms=nfull_latency_ms,
        total_latency_ms=total_latency_ms,
        n0_route=n0_route,
        nfull_route=nfull_route,
    )


__all__ = [
    "CurrentState",
    "DemoUtilityRecord",
    "LIBERO_DATASET_TO_SUITE",
    "SampleIdentity",
    "collect_paired_utility",
    "current_state_input_hashes",
    "extract_current_state",
    "parse_sample_identity",
    "stable_sample_seed",
    "tensor_content_sha256",
    "valid_chunk_mse",
]
