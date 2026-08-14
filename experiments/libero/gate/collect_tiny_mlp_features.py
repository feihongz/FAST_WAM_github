"""Collect the sealed current-state feature cache for Gate Phase 3.

This job is intentionally narrower than a policy inference job.  It rehydrates
the exact Target-V2 100-state plan, verifies every live input tensor (including
the label-only action tensors), and then passes *only* the current image,
instruction context, and current proprio to the frozen feature extractor.

The four public files are immutable and fail closed:

``manifest.json``
    Binds Target V2, Phase-2.5, artifacts, data/cache bytes, source, config, and
    the exact random projection matrices before feature extraction starts.
``feature_index.jsonl``
    Ordered identity/row hashes.  Dataset-local ``source_metadata.source_index``
    is deliberately absent; the only source index is the global dataset index.
``features.safetensors``
    Float32 tensors ``visual``, ``instruction``, ``proprio``, and ``full``.
``completion.json``
    Published last and binds the exact bytes and tensor contents of the other
    three files.

Private per-row progress files make interrupted GPU extraction resumable.  A
resume still rehydrates all states and rechecks live input hashes; cached rows
are never trusted merely because their filenames exist.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import math
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import hydra
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file, save_file
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.libero.gate import demo_utility_target_v2 as target_v2
from experiments.libero.gate.collect_demo_utility import (
    _assert_source_matches_plan,
    _dataset_instantiation_path_overrides,
    _directory_tree_provenance,
    _json_safe_metadata,
    _normalize_ranges,
    _resolve_existing_file,
    _resolve_vae_artifact,
    _scientific_data_config,
    _sha256_file,
    _stable_file_provenance,
    resolve_dataset_stats_path,
)
from experiments.libero.gate.demo_utility import (
    CurrentState,
    current_state_input_hashes,
    extract_current_state,
    parse_sample_identity,
)
from fastwam.utils.config_resolvers import register_default_resolvers


register_default_resolvers()

LOGGER = logging.getLogger(__name__)

BUNDLE_KIND = "libero_gate_current_state_feature_cache"
COMPLETION_KIND = "libero_gate_current_state_feature_cache_completion"
BUNDLE_SCHEMA_VERSION = 1
FEATURE_RECORD_SCHEMA_VERSION = 1
COMPLETION_SCHEMA_VERSION = 1
TENSOR_HASH_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
INDEX_FILENAME = "feature_index.jsonl"
FEATURES_FILENAME = "features.safetensors"
COMPLETION_FILENAME = "completion.json"
TENSOR_KEYS = ("full", "visual", "instruction", "proprio")
EXPECTED_DIMS = {"full": 137, "visual": 64, "instruction": 65, "proprio": 8}
SHA256_HEX_LENGTH = 64

SCIENTIFIC_SOURCE_FILES = (
    "configs/collect_libero_gate_features.yaml",
    "docs/GATE_OFFLINE_TINY_MLP_FEASIBILITY.md",
    "experiments/libero/gate/collect_demo_utility.py",
    "experiments/libero/gate/collect_tiny_mlp_features.py",
    "experiments/libero/gate/demo_utility.py",
    "experiments/libero/gate/demo_utility_target_v2.py",
    "src/fastwam/datasets/lerobot/base_lerobot_dataset.py",
    "src/fastwam/datasets/lerobot/robot_video_dataset.py",
    "src/fastwam/models/wan22/fastwam.py",
    "src/fastwam/models/wan22/fastwam_unified_shared.py",
    "src/fastwam/models/wan22/helpers/io.py",
    "src/fastwam/models/wan22/helpers/loader.py",
    "src/fastwam/models/wan22/helpers/state_dict_converters.py",
    "src/fastwam/models/wan22/wan_video_vae.py",
    "src/fastwam/runtime.py",
    "src/fastwam/utils/config_resolvers.py",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def require_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256, got {value!r}")
    return value


def tensor_content_sha256(tensor: torch.Tensor) -> str:
    """Hash dtype + canonical shape + contiguous CPU C-order bytes.

    This exact algorithm is part of the cache/trainer contract.  It matches the
    input-tensor hashing convention but has its own schema constant so it cannot
    silently inherit a future input-hash change.
    """

    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    value = tensor.detach().to(device="cpu").contiguous()
    header = canonical_json(
        {
            "schema_version": TENSOR_HASH_SCHEMA_VERSION,
            "dtype": str(value.dtype),
            "shape": [int(dim) for dim in value.shape],
        }
    ).encode("utf-8")
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    return sha256_bytes(header + b"\0" + raw)


def rademacher_projection(input_dim: int, output_dim: int, seed: int) -> torch.Tensor:
    """Return the protocol-fixed CPU float32 Rademacher projection matrix."""

    for name, value in (("input_dim", input_dim), ("output_dim", output_dim)):
        if isinstance(value, bool) or int(value) <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if isinstance(seed, bool) or int(seed) < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    signs = torch.randint(
        0,
        2,
        (int(input_dim), int(output_dim)),
        generator=generator,
        dtype=torch.int64,
        device="cpu",
    )
    return ((signs.to(torch.float32) * 2.0 - 1.0) / math.sqrt(input_dim)).contiguous()


def projection_metadata(matrix: torch.Tensor, *, seed: int) -> dict[str, Any]:
    return {
        "algorithm": "cpu_torch_generator_rademacher_v1",
        "seed": int(seed),
        "shape": [int(value) for value in matrix.shape],
        "dtype": str(matrix.dtype),
        "layout": "row_major",
        "values": "{-1,+1}/sqrt(input_dim)",
        "matrix_content_sha256": tensor_content_sha256(matrix),
    }


def build_projection_matrices(
    *,
    latent_channels: int = 48,
    pooled_height: int = 2,
    pooled_width: int = 4,
    visual_dim: int = 64,
    visual_seed: int = 20260815,
    context_dim: int = 4096,
    instruction_dim: int = 32,
    mean_seed: int = 20260816,
    rms_seed: int = 20260817,
) -> dict[str, torch.Tensor]:
    visual_input_dim = int(latent_channels) * (
        int(pooled_height) * int(pooled_width) + 1
    )
    return {
        "visual": rademacher_projection(visual_input_dim, visual_dim, visual_seed),
        "instruction_mean": rademacher_projection(
            context_dim, instruction_dim, mean_seed
        ),
        "instruction_rms": rademacher_projection(
            context_dim, instruction_dim, rms_seed
        ),
    }


def pool_visual_latent(
    latent: torch.Tensor,
    projection: torch.Tensor,
    *,
    latent_channels: int = 48,
    pooled_height: int = 2,
    pooled_width: int = 4,
) -> torch.Tensor:
    """Pool one current-frame VAE latent, then apply the frozen projection."""

    value = torch.as_tensor(latent).detach().to(device="cpu", dtype=torch.float32)
    if value.ndim == 5:
        if value.shape[0] != 1:
            raise ValueError(f"VAE latent batch must be 1, got {tuple(value.shape)}")
        value = value[0]
    expected_prefix = (int(latent_channels), 1)
    if value.ndim != 4 or tuple(value.shape[:2]) != expected_prefix:
        raise ValueError(
            "VAE latent must have [C,1,H,W] with exact protocol channels; "
            f"expected prefix={expected_prefix}, got {tuple(value.shape)}"
        )
    if value.shape[2] < 1 or value.shape[3] < 1 or not torch.isfinite(value).all():
        raise ValueError("VAE latent must be finite with non-empty spatial dimensions")
    spatial = value[:, 0]
    pooled = F.adaptive_avg_pool2d(
        spatial, (int(pooled_height), int(pooled_width))
    ).reshape(-1)
    population_std = value.std(dim=(1, 2, 3), correction=0)
    unprojected = torch.cat((pooled, population_std), dim=0).contiguous()
    matrix = torch.as_tensor(projection).detach().to(device="cpu", dtype=torch.float32)
    if matrix.ndim != 2 or matrix.shape[0] != unprojected.numel():
        raise ValueError(
            f"visual projection shape {tuple(matrix.shape)} is incompatible with "
            f"pooled latent dim {unprojected.numel()}"
        )
    result = torch.matmul(unprojected, matrix).to(torch.float32).contiguous()
    if not torch.isfinite(result).all():
        raise ValueError("projected visual feature contains non-finite values")
    return result


def pool_instruction_context(
    context: torch.Tensor,
    mean_projection: torch.Tensor,
    rms_projection: torch.Tensor,
    *,
    context_dim: int = 4096,
) -> torch.Tensor:
    """Pool nonzero cached context rows without consulting the rewritten mask."""

    value = torch.as_tensor(context).detach().to(device="cpu", dtype=torch.float32)
    if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] != int(context_dim):
        raise ValueError(
            f"context must have [L,{context_dim}] with L>0, got {tuple(value.shape)}"
        )
    if not torch.isfinite(value).all():
        raise ValueError("context contains non-finite values")
    active = torch.any(value != 0, dim=1)
    active_count = int(active.sum().item())
    if active_count == 0:
        raise ValueError("context has no active (nonzero) cached token rows")
    active_rows = value[active]
    mean = active_rows.mean(dim=0)
    rms = torch.sqrt(torch.mean(active_rows.square(), dim=0))
    mean_matrix = torch.as_tensor(mean_projection).detach().to(
        device="cpu", dtype=torch.float32
    )
    rms_matrix = torch.as_tensor(rms_projection).detach().to(
        device="cpu", dtype=torch.float32
    )
    if mean_matrix.ndim != 2 or rms_matrix.ndim != 2:
        raise ValueError("instruction projections must be matrices")
    if mean_matrix.shape[0] != context_dim or rms_matrix.shape[0] != context_dim:
        raise ValueError("instruction projection input dimensions do not match context")
    if mean_matrix.shape[1] != rms_matrix.shape[1]:
        raise ValueError("instruction mean/RMS projection dimensions differ")
    fraction = torch.tensor(
        [active_count / int(value.shape[0])], dtype=torch.float32, device="cpu"
    )
    result = torch.cat((mean @ mean_matrix, rms @ rms_matrix, fraction)).contiguous()
    if not torch.isfinite(result).all():
        raise ValueError("instruction feature contains non-finite values")
    return result


def extract_allowed_features(
    *,
    input_image: torch.Tensor,
    context: torch.Tensor,
    proprio: torch.Tensor,
    encode_current_image: Callable[[torch.Tensor], torch.Tensor],
    projections: Mapping[str, torch.Tensor],
    latent_channels: int = 48,
    pooled_height: int = 2,
    pooled_width: int = 4,
    context_dim: int = 4096,
    proprio_dim: int = 8,
) -> dict[str, torch.Tensor]:
    """Extract only observation/instruction/proprio features.

    The signature has no future video, action, pad mask, label, utility, or
    identity argument.  This makes the leakage boundary directly testable.
    """

    image = torch.as_tensor(input_image).detach()
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"input_image must have [3,H,W], got {tuple(image.shape)}")
    latent = encode_current_image(image)
    visual = pool_visual_latent(
        latent,
        projections["visual"],
        latent_channels=latent_channels,
        pooled_height=pooled_height,
        pooled_width=pooled_width,
    )
    instruction = pool_instruction_context(
        context,
        projections["instruction_mean"],
        projections["instruction_rms"],
        context_dim=context_dim,
    )
    proprio_feature = torch.as_tensor(proprio).detach().to(
        device="cpu", dtype=torch.float32
    ).contiguous()
    if proprio_feature.ndim != 1 or proprio_feature.numel() != int(proprio_dim):
        raise ValueError(
            f"current proprio must have [{proprio_dim}], got {tuple(proprio_feature.shape)}"
        )
    if not torch.isfinite(proprio_feature).all():
        raise ValueError("current proprio contains non-finite values")
    full = torch.cat((visual, instruction, proprio_feature), dim=0).contiguous()
    features = {
        "visual": visual,
        "instruction": instruction,
        "proprio": proprio_feature,
        "full": full,
    }
    expected = {
        "visual": int(projections["visual"].shape[1]),
        "instruction": int(projections["instruction_mean"].shape[1]) * 2 + 1,
        "proprio": int(proprio_dim),
    }
    expected["full"] = expected["visual"] + expected["instruction"] + expected["proprio"]
    for key, tensor in features.items():
        if tensor.dtype != torch.float32 or tensor.ndim != 1 or tensor.numel() != expected[key]:
            raise AssertionError(
                f"invalid {key} feature contract: dtype={tensor.dtype}, shape={tuple(tensor.shape)}"
            )
    return features


def prepare_vae_input(
    input_image: torch.Tensor, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Match UniShare's current-image VAE input path exactly.

    UniShare casts ``input_image`` to its model dtype before inserting the
    singleton time axis.  The feature-only loader owns just the VAE, so its
    parameter dtype is the authoritative equivalent.  Keeping this conversion
    explicit prevents a float32 image / bf16 convolution mismatch and makes the
    parity contract directly testable.
    """

    image = torch.as_tensor(input_image)
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"input_image must have [3,H,W], got {tuple(image.shape)}")
    if not torch.is_floating_point(image) or not torch.isfinite(image).all():
        raise ValueError("input_image must be finite floating point")
    return image.to(device=device, dtype=dtype).unsqueeze(1)


def feature_id(
    *,
    sample_id: str,
    source_index: int,
    target_sha256: str,
    input_combined_sha256: str,
    extractor_fingerprint: str,
) -> str:
    return sha256_json(
        {
            "kind": BUNDLE_KIND,
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "sample_id": str(sample_id),
            "source_index": int(source_index),
            "target_sha256": require_sha256(target_sha256, field="target_sha256"),
            "input_combined_sha256": require_sha256(
                input_combined_sha256, field="input_combined_sha256"
            ),
            "extractor_fingerprint": require_sha256(
                extractor_fingerprint, field="extractor_fingerprint"
            ),
        }
    )


def feature_record_sha256(record: Mapping[str, Any]) -> str:
    return sha256_json(
        {key: value for key, value in record.items() if key != "feature_record_sha256"}
    )


def build_feature_record(
    target: Mapping[str, Any],
    features: Mapping[str, torch.Tensor],
    *,
    extractor_fingerprint: str,
) -> dict[str, Any]:
    hashes = {key: tensor_content_sha256(features[key]) for key in TENSOR_KEYS}
    record: dict[str, Any] = {
        "feature_record_schema_version": FEATURE_RECORD_SCHEMA_VERSION,
        "kind": BUNDLE_KIND,
        "selection_order": int(target["selection_order"]),
        "sample_id": str(target["sample_id"]),
        "source_index": int(target["source_index"]),
        "suite": str(target["suite"]),
        "task_index": int(target["task_index"]),
        "task": str(target["task"]),
        "target_id": str(target["target_id"]),
        "target_sha256": str(target["target_sha256"]),
        "input_combined_sha256": str(target["input_hashes"]["combined"]),
        "feature_id": feature_id(
            sample_id=str(target["sample_id"]),
            source_index=int(target["source_index"]),
            target_sha256=str(target["target_sha256"]),
            input_combined_sha256=str(target["input_hashes"]["combined"]),
            extractor_fingerprint=extractor_fingerprint,
        ),
        "feature_hashes": hashes,
    }
    record["feature_record_sha256"] = feature_record_sha256(record)
    return record


def _feature_dimensions_from_manifest(manifest: Mapping[str, Any]) -> dict[str, int]:
    dimensions = manifest["compatibility"]["feature_dimensions"]
    return {key: int(dimensions[key]) for key in TENSOR_KEYS}


def validate_feature_record(
    record: Mapping[str, Any],
    features: Mapping[str, torch.Tensor],
    target: Mapping[str, Any],
    *,
    extractor_fingerprint: str,
    expected_dimensions: Mapping[str, int] = EXPECTED_DIMS,
) -> None:
    required = {
        "feature_record_schema_version",
        "kind",
        "selection_order",
        "sample_id",
        "source_index",
        "suite",
        "task_index",
        "task",
        "target_id",
        "target_sha256",
        "input_combined_sha256",
        "feature_id",
        "feature_hashes",
        "feature_record_sha256",
    }
    if set(record) != required:
        raise ValueError(
            f"feature index fields differ from schema: missing={sorted(required-set(record))}, "
            f"extra={sorted(set(record)-required)}"
        )
    if int(record["feature_record_schema_version"]) != FEATURE_RECORD_SCHEMA_VERSION:
        raise ValueError("unsupported feature record schema")
    if record["kind"] != BUNDLE_KIND:
        raise ValueError("feature record kind is invalid")
    bindings = (
        ("selection_order", int(target["selection_order"])),
        ("sample_id", str(target["sample_id"])),
        ("source_index", int(target["source_index"])),
        ("suite", str(target["suite"])),
        ("task_index", int(target["task_index"])),
        ("task", str(target["task"])),
        ("target_id", str(target["target_id"])),
        ("target_sha256", str(target["target_sha256"])),
        ("input_combined_sha256", str(target["input_hashes"]["combined"])),
    )
    for field, expected in bindings:
        if record[field] != expected:
            raise ValueError(f"feature record {field} differs from Target V2")
    expected_id = feature_id(
        sample_id=str(target["sample_id"]),
        source_index=int(target["source_index"]),
        target_sha256=str(target["target_sha256"]),
        input_combined_sha256=str(target["input_hashes"]["combined"]),
        extractor_fingerprint=extractor_fingerprint,
    )
    if record["feature_id"] != expected_id:
        raise ValueError("feature_id is not bound to target/input/extractor")
    if record["feature_record_sha256"] != feature_record_sha256(record):
        raise ValueError("feature_record_sha256 is invalid")
    hashes = record["feature_hashes"]
    if not isinstance(hashes, Mapping) or set(hashes) != set(TENSOR_KEYS):
        raise ValueError("feature_hashes must contain exactly the four public tensor keys")
    normalized: dict[str, torch.Tensor] = {}
    for key in TENSOR_KEYS:
        tensor = torch.as_tensor(features[key]).detach().to(device="cpu").contiguous()
        if (
            tensor.dtype != torch.float32
            or tensor.ndim != 1
            or tensor.numel() != int(expected_dimensions[key])
        ):
            raise ValueError(
                f"{key} row must be float32 [{expected_dimensions[key]}], "
                f"got {tensor.dtype} {tuple(tensor.shape)}"
            )
        if hashes[key] != tensor_content_sha256(tensor):
            raise ValueError(f"{key} row content hash is invalid")
        normalized[key] = tensor
    if not torch.equal(
        normalized["full"],
        torch.cat(
            (normalized["visual"], normalized["instruction"], normalized["proprio"])
        ),
    ):
        raise ValueError("full feature is not exact visual+instruction+proprio concatenation")


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _serialize_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _serialize_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(f"{canonical_json(row)}\n" for row in rows).encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_save_safetensors(
    path: Path, tensors: Mapping[str, torch.Tensor], *, metadata: Mapping[str, str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(raw_temp)
    try:
        save_file(
            {key: value.detach().to(device="cpu").contiguous() for key, value in tensors.items()},
            str(temporary),
            metadata=dict(metadata),
        )
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _scientific_source_provenance() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in SCIENTIFIC_SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"scientific source is missing: {path}")
        result[relative] = _sha256_file(path)
    return result


def _load_phase25_manifest(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    expected = require_sha256(expected_sha256, field="expected_phase25_manifest_sha256")
    if _sha256_file(path) != expected:
        raise ValueError("Phase-2.5 manifest SHA-256 differs from the configured trust anchor")
    manifest = _load_json(path, label="Phase-2.5 manifest")
    compatibility = manifest.get("compatibility")
    if not isinstance(compatibility, Mapping):
        raise ValueError("Phase-2.5 manifest has no compatibility mapping")
    fingerprint = require_sha256(
        manifest.get("compatibility_fingerprint"),
        field="Phase-2.5 compatibility_fingerprint",
    )
    if fingerprint != sha256_json(compatibility):
        raise ValueError("Phase-2.5 compatibility fingerprint is invalid")
    return manifest


def _assert_artifacts_match_phase25(
    *,
    phase25: Mapping[str, Any],
    target_manifest: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    stats: Mapping[str, Any],
    vae: Mapping[str, Any],
    dataset_sources: Sequence[Mapping[str, Any]],
    context_cache: Mapping[str, Any],
    model_config_sha256: str,
    data_config_sha256: str,
) -> None:
    compatibility = phase25["compatibility"]
    source = target_manifest["source"]
    phase_manifest_sha = _sha256_file(Path(str(phase25["_manifest_path"])))
    if source["manifest_sha256"] != phase_manifest_sha:
        raise ValueError("Target V2 is not bound to the configured Phase-2.5 manifest")
    for label, artifact, field in (
        ("checkpoint", checkpoint, "checkpoint_sha256"),
        ("dataset stats", stats, "dataset_stats_sha256"),
        ("VAE", vae, "vae_sha256"),
    ):
        actual = str(artifact["sha256"])
        if actual != compatibility[field] or actual != source[field]:
            raise ValueError(f"live {label} bytes differ from Target V2 / Phase-2.5")
    expected_sources = compatibility.get("dataset_source_content")
    actual_sources = [
        {
            "dataset_name": str(item["dataset_name"]),
            "sha256": str(item["sha256"]),
            "file_count": int(item["file_count"]),
            "total_size_bytes": int(item["total_size_bytes"]),
        }
        for item in dataset_sources
    ]
    if actual_sources != expected_sources:
        raise ValueError("live dataset source content differs from Phase-2.5")
    if context_cache["sha256"] != compatibility.get("context_cache_sha256"):
        raise ValueError("live text context cache differs from Phase-2.5")
    if model_config_sha256 != compatibility.get("model_config_sha256"):
        raise ValueError("resolved model config differs from Phase-2.5")
    if data_config_sha256 != compatibility.get("data_config_sha256"):
        raise ValueError("resolved scientific data config differs from Phase-2.5")


def _extractor_contract(
    collector: Mapping[str, Any], projections: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    visual = collector["visual"]
    instruction = collector["instruction"]
    contract = {
        "schema_version": 1,
        "current_image_rule": "extract_current_state(video[:,0])",
        "visual": {
            "vae_loader": "_load_registered_model(content_hashed_path,wan_video_vae)",
            "vae_latent_order": "channel,time,height,width",
            "required_time": 1,
            "latent_channels": int(visual["latent_channels"]),
            "adaptive_pool": [int(visual["pooled_height"]), int(visual["pooled_width"])],
            "flatten_order": "channel_major_row_major",
            "std": "population_over_time_height_width_correction_0",
            "projection": projection_metadata(
                projections["visual"], seed=int(visual["projection_seed"])
            ),
        },
        "instruction": {
            "context_dim": int(instruction["context_dim"]),
            "active_token_rule": "row_any_element_nonzero",
            "pool": ["active_row_mean", "active_row_rms", "active_fraction"],
            "mean_projection": projection_metadata(
                projections["instruction_mean"],
                seed=int(instruction["mean_projection_seed"]),
            ),
            "rms_projection": projection_metadata(
                projections["instruction_rms"],
                seed=int(instruction["rms_projection_seed"]),
            ),
        },
        "proprio": {"rule": "normalized_proprio[0]", "dim": int(collector["proprio_dim"])},
        "feature_order": ["visual", "instruction", "proprio"],
        "feature_dimensions": dict(EXPECTED_DIMS),
        "forbidden_feature_fields": [
            "future_video",
            "action",
            "action_is_pad",
            "valid_length",
            "e0",
            "efull",
            "utility",
            "uncertainty",
            "selection_bin",
            "identity_or_route_fields",
        ],
        "tensor_hash_algorithm": "sha256(canonical_json(dtype,shape,schema=1)+NUL+C_bytes)",
    }
    contract["extractor_fingerprint"] = sha256_json(contract)
    return contract


def _build_manifest(
    *,
    cfg: DictConfig,
    target_manifest: Mapping[str, Any],
    targets: Sequence[Mapping[str, Any]],
    target_dir: Path,
    target_manifest_sha256: str,
    target_records_sha256: str,
    phase25: Mapping[str, Any],
    phase25_manifest_sha256: str,
    checkpoint: Mapping[str, Any],
    stats: Mapping[str, Any],
    vae: Mapping[str, Any],
    dataset_sources: Sequence[Mapping[str, Any]],
    context_cache: Mapping[str, Any],
    extractor: Mapping[str, Any],
    scientific_sources: Mapping[str, str],
) -> dict[str, Any]:
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    scientific_cfg = json.loads(canonical_json(resolved_cfg))
    scientific_collector = scientific_cfg["FEATURE_COLLECTOR"]
    # These fields affect scheduling/location only, never feature bytes.  In
    # particular, a one-row GPU smoke must resume into the same immutable cache.
    scientific_collector.pop("resume", None)
    scientific_collector.pop("max_new_rows", None)
    scientific_collector.pop("output_dir", None)
    # The inherited training config also resolves a top-level run directory
    # from `${now:...}`. It is unused by the feature collector and therefore
    # must not make a one-row smoke incompatible with its later resume.
    scientific_cfg.pop("output_dir", None)
    target_projection = [
        {
            "selection_order": int(row["selection_order"]),
            "sample_id": str(row["sample_id"]),
            "source_index": int(row["source_index"]),
            "target_id": str(row["target_id"]),
            "target_sha256": str(row["target_sha256"]),
            "input_combined_sha256": str(row["input_hashes"]["combined"]),
        }
        for row in targets
    ]
    compatibility: dict[str, Any] = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "kind": BUNDLE_KIND,
        "feature_record_schema_version": FEATURE_RECORD_SCHEMA_VERSION,
        "target_manifest_sha256": target_manifest_sha256,
        "target_records_sha256": target_records_sha256,
        "target_compatibility_fingerprint": target_manifest["compatibility_fingerprint"],
        "target_selection_sha256": sha256_json(target_projection),
        "ordered_target_ids_sha256": sha256_json([row["target_id"] for row in targets]),
        "ordered_target_sha256_sha256": sha256_json(
            [row["target_sha256"] for row in targets]
        ),
        "phase25_manifest_sha256": phase25_manifest_sha256,
        "phase25_compatibility_fingerprint": phase25["compatibility_fingerprint"],
        "checkpoint_sha256": checkpoint["sha256"],
        "dataset_stats_sha256": stats["sha256"],
        "vae_sha256": vae["sha256"],
        "dataset_source_content": [
            {
                "dataset_name": item["dataset_name"],
                "sha256": item["sha256"],
                "file_count": item["file_count"],
                "total_size_bytes": item["total_size_bytes"],
            }
            for item in dataset_sources
        ],
        "context_cache_sha256": context_cache["sha256"],
        "scientific_resolved_config_sha256": sha256_json(scientific_cfg),
        "scientific_source_files": dict(scientific_sources),
        "extractor": dict(extractor),
        "extractor_fingerprint": extractor["extractor_fingerprint"],
        "feature_dimensions": dict(EXPECTED_DIMS),
        "num_states": len(targets),
        "output_filenames": {
            "manifest": MANIFEST_FILENAME,
            "feature_index": INDEX_FILENAME,
            "features": FEATURES_FILENAME,
            "completion": COMPLETION_FILENAME,
        },
        "global_join_contract": {
            "keys": ["sample_id", "source_index"],
            "source_index_semantics": "global requested_sample_idx/source_sample_idx",
            "dataset_local_source_metadata.source_index_allowed_as_join": False,
        },
    }
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "kind": BUNDLE_KIND,
        "created_at_utc": _utc_now(),
        "compatibility": compatibility,
        "compatibility_fingerprint": sha256_json(compatibility),
        "target_v2": {
            "path": str(target_dir),
            "manifest_sha256": target_manifest_sha256,
            "records_sha256": target_records_sha256,
            "compatibility_fingerprint": target_manifest["compatibility_fingerprint"],
            "ordered_targets": target_projection,
        },
        "phase25": {
            "manifest_path": str(phase25["_manifest_path"]),
            "manifest_sha256": phase25_manifest_sha256,
            "compatibility_fingerprint": phase25["compatibility_fingerprint"],
        },
        "artifacts": {
            "checkpoint": dict(checkpoint),
            "dataset_stats": dict(stats),
            "vae": dict(vae),
            "dataset_sources": [dict(item) for item in dataset_sources],
            "text_embedding_cache": dict(context_cache),
        },
        "extractor": dict(extractor),
        "resolved_config": resolved_cfg,
        "outputs": {
            "feature_index": {"filename": INDEX_FILENAME},
            "features": {
                "filename": FEATURES_FILENAME,
                "keys": list(TENSOR_KEYS),
                "dtype": "torch.float32",
                "shapes": {key: [len(targets), EXPECTED_DIMS[key]] for key in TENSOR_KEYS},
            },
            "completion": {"filename": COMPLETION_FILENAME},
        },
    }


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("kind") != BUNDLE_KIND or int(manifest.get("schema_version", -1)) != 1:
        raise ValueError("unsupported feature-cache manifest")
    compatibility = manifest.get("compatibility")
    if not isinstance(compatibility, Mapping):
        raise ValueError("feature-cache compatibility must be a mapping")
    fingerprint = require_sha256(
        manifest.get("compatibility_fingerprint"), field="compatibility_fingerprint"
    )
    if fingerprint != sha256_json(compatibility):
        raise ValueError("feature-cache compatibility fingerprint is invalid")
    if compatibility.get("kind") != BUNDLE_KIND:
        raise ValueError("feature-cache compatibility kind is invalid")
    if compatibility.get("feature_dimensions") != EXPECTED_DIMS:
        raise ValueError("feature dimensions differ from frozen 137-d protocol")
    extractor = compatibility.get("extractor")
    if not isinstance(extractor, Mapping):
        raise ValueError("manifest has no extractor contract")
    if extractor.get("extractor_fingerprint") != sha256_json(
        {key: value for key, value in extractor.items() if key != "extractor_fingerprint"}
    ):
        raise ValueError("extractor fingerprint is invalid")
    if compatibility.get("extractor_fingerprint") != extractor["extractor_fingerprint"]:
        raise ValueError("compatibility is not bound to extractor")
    join = compatibility.get("global_join_contract")
    if not isinstance(join, Mapping) or join.get(
        "dataset_local_source_metadata.source_index_allowed_as_join"
    ) is not False:
        raise ValueError("manifest does not explicitly reject dataset-local joins")


def ensure_immutable_manifest(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    validate_manifest(payload)
    if path.exists():
        existing = _load_json(path, label="feature-cache manifest")
        validate_manifest(existing)
        if existing["compatibility_fingerprint"] != payload["compatibility_fingerprint"]:
            raise ValueError("resume manifest differs from the frozen feature-cache contract")
        return existing
    _atomic_write_bytes(path, _serialize_json(payload))
    stored = _load_json(path, label="feature-cache manifest")
    validate_manifest(stored)
    return stored


def _validate_live_state(
    *,
    sample: Mapping[str, Any],
    target: Mapping[str, Any],
    ranges: Sequence[Mapping[str, Any]],
    task_tables: Mapping[int, Mapping[int, str]],
) -> CurrentState:
    metadata_raw = sample.get("metadata")
    if not isinstance(metadata_raw, Mapping):
        raise ValueError("strict dataset sample has no mapping-valued metadata")
    metadata = _json_safe_metadata(metadata_raw)
    source_index = int(target["source_index"])
    _assert_source_matches_plan(metadata, source_index, ranges)
    # The global join is explicitly the requested/source sample index.  The
    # underlying per-dataset metadata source_index is neither read nor compared.
    if int(metadata["requested_sample_idx"]) != source_index or int(
        metadata["source_sample_idx"]
    ) != source_index:
        raise ValueError("dataset did not return the requested global source index")
    dataset_index = int(metadata["dataset_index"])
    identity = parse_sample_identity(metadata, task_by_index=task_tables[dataset_index])
    expected_identity = identity.to_dict()
    for field in (
        "sample_id",
        "dataset_id",
        "dataset_name",
        "suite",
        "episode_index",
        "frame_index",
        "task_index",
        "task",
    ):
        if target[field] != expected_identity[field]:
            raise ValueError(f"live dataset identity differs from Target V2 for {field}")
    state = extract_current_state(sample)
    actual_hashes = current_state_input_hashes(state)
    if actual_hashes != target["input_hashes"]:
        raise ValueError(
            f"live current-state input hashes differ from Target V2 for {target['sample_id']}"
        )
    expected_proprio = torch.tensor(target["current_proprio"], dtype=torch.float32)
    actual_proprio = state.proprio.detach().to(device="cpu", dtype=torch.float32)
    if not torch.equal(actual_proprio, expected_proprio):
        raise ValueError("live current proprio differs from Target V2")
    return state


def _progress_paths(progress_dir: Path, order: int) -> tuple[Path, Path]:
    stem = f"{int(order):06d}"
    return progress_dir / f"{stem}.json", progress_dir / f"{stem}.safetensors"


def _write_progress_row(
    progress_dir: Path,
    record: Mapping[str, Any],
    features: Mapping[str, torch.Tensor],
    *,
    extractor_fingerprint: str,
) -> None:
    order = int(record["selection_order"])
    json_path, tensor_path = _progress_paths(progress_dir, order)
    if json_path.exists() or tensor_path.exists():
        raise FileExistsError(f"partial/duplicate progress row for selection_order={order}")
    _atomic_save_safetensors(
        tensor_path,
        features,
        metadata={
            "kind": BUNDLE_KIND,
            "feature_id": str(record["feature_id"]),
            "extractor_fingerprint": extractor_fingerprint,
        },
    )
    _atomic_write_bytes(json_path, _serialize_json(record))


def _load_progress_row(
    progress_dir: Path,
    target: Mapping[str, Any],
    *,
    extractor_fingerprint: str,
    expected_dimensions: Mapping[str, int],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]] | None:
    json_path, tensor_path = _progress_paths(progress_dir, int(target["selection_order"]))
    if json_path.exists() != tensor_path.exists():
        raise ValueError(f"orphaned progress file for order={target['selection_order']}")
    if not json_path.exists():
        return None
    record = _load_json(json_path, label="feature progress record")
    tensors = load_file(str(tensor_path), device="cpu")
    if set(tensors) != set(TENSOR_KEYS):
        raise ValueError("progress safetensors contains unexpected/missing keys")
    validate_feature_record(
        record,
        tensors,
        target,
        extractor_fingerprint=extractor_fingerprint,
        expected_dimensions=expected_dimensions,
    )
    return record, tensors


def _build_final_from_progress(
    progress_dir: Path,
    targets: Sequence[Mapping[str, Any]],
    *,
    extractor_fingerprint: str,
    expected_dimensions: Mapping[str, int],
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    rows: list[dict[str, Any]] = []
    by_key: dict[str, list[torch.Tensor]] = {key: [] for key in TENSOR_KEYS}
    for target in targets:
        loaded = _load_progress_row(
            progress_dir,
            target,
            extractor_fingerprint=extractor_fingerprint,
            expected_dimensions=expected_dimensions,
        )
        if loaded is None:
            raise ValueError(f"missing completed progress row {target['selection_order']}")
        record, tensors = loaded
        rows.append(record)
        for key in TENSOR_KEYS:
            by_key[key].append(tensors[key])
    matrices = {key: torch.stack(by_key[key]).contiguous() for key in TENSOR_KEYS}
    return rows, matrices


def _validate_final_files(
    *,
    index_path: Path,
    features_path: Path,
    rows: Sequence[Mapping[str, Any]],
    matrices: Mapping[str, torch.Tensor],
) -> None:
    expected_index = _serialize_jsonl(rows)
    if not index_path.is_file() or index_path.read_bytes() != expected_index:
        raise ValueError("existing feature_index.jsonl differs from verified progress rows")
    if not features_path.is_file():
        raise FileNotFoundError(f"features.safetensors is missing: {features_path}")
    stored = load_file(str(features_path), device="cpu")
    if set(stored) != set(TENSOR_KEYS):
        raise ValueError("final features.safetensors has unexpected/missing keys")
    for key in TENSOR_KEYS:
        if not torch.equal(stored[key], matrices[key]):
            raise ValueError(f"final tensor {key} differs from verified progress rows")


def _publish_final_files(
    *,
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    matrices: Mapping[str, torch.Tensor],
    manifest_fingerprint: str,
) -> tuple[Path, Path]:
    index_path = output_dir / INDEX_FILENAME
    features_path = output_dir / FEATURES_FILENAME
    if index_path.exists() or features_path.exists():
        if not (index_path.exists() and features_path.exists()):
            raise ValueError("orphaned final feature output without completion")
        _validate_final_files(
            index_path=index_path,
            features_path=features_path,
            rows=rows,
            matrices=matrices,
        )
        return index_path, features_path
    _atomic_write_bytes(index_path, _serialize_jsonl(rows))
    _atomic_save_safetensors(
        features_path,
        matrices,
        metadata={
            "kind": BUNDLE_KIND,
            "manifest_compatibility_fingerprint": manifest_fingerprint,
        },
    )
    _validate_final_files(
        index_path=index_path, features_path=features_path, rows=rows, matrices=matrices
    )
    return index_path, features_path


def completion_payload(
    *,
    manifest_path: Path,
    index_path: Path,
    features_path: Path,
    matrices: Mapping[str, torch.Tensor],
    manifest_fingerprint: str,
    num_states: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "kind": COMPLETION_KIND,
        "completed_at_utc": _utc_now(),
        "manifest_sha256": _sha256_file(manifest_path),
        "manifest_compatibility_fingerprint": manifest_fingerprint,
        "feature_index_sha256": _sha256_file(index_path),
        "features_sha256": _sha256_file(features_path),
        "num_states": int(num_states),
        "files": {
            "manifest": MANIFEST_FILENAME,
            "feature_index": INDEX_FILENAME,
            "features": FEATURES_FILENAME,
        },
        "tensors": {
            key: {
                "shape": [int(dim) for dim in matrices[key].shape],
                "dtype": str(matrices[key].dtype),
                "content_sha256": tensor_content_sha256(matrices[key]),
            }
            for key in TENSOR_KEYS
        },
    }
    payload["completion_sha256"] = sha256_json(payload)
    return payload


def validate_completion(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / MANIFEST_FILENAME
    index_path = output_dir / INDEX_FILENAME
    features_path = output_dir / FEATURES_FILENAME
    completion_path = output_dir / COMPLETION_FILENAME
    completion = _load_json(completion_path, label="feature-cache completion")
    if completion.get("kind") != COMPLETION_KIND or int(
        completion.get("schema_version", -1)
    ) != COMPLETION_SCHEMA_VERSION:
        raise ValueError("unsupported feature-cache completion")
    expected_digest = require_sha256(
        completion.get("completion_sha256"), field="completion_sha256"
    )
    unhashed = {key: value for key, value in completion.items() if key != "completion_sha256"}
    if expected_digest != sha256_json(unhashed):
        raise ValueError("completion_sha256 is invalid")
    manifest = _load_json(manifest_path, label="feature-cache manifest")
    validate_manifest(manifest)
    expected_files = {
        "manifest_sha256": _sha256_file(manifest_path),
        "feature_index_sha256": _sha256_file(index_path),
        "features_sha256": _sha256_file(features_path),
    }
    for field, actual in expected_files.items():
        if completion.get(field) != actual:
            raise ValueError(f"completion {field} does not match live file bytes")
    if completion.get("manifest_compatibility_fingerprint") != manifest.get(
        "compatibility_fingerprint"
    ):
        raise ValueError("completion is not bound to the manifest fingerprint")
    tensors = load_file(str(features_path), device="cpu")
    if set(tensors) != set(TENSOR_KEYS):
        raise ValueError("sealed features.safetensors keys are invalid")
    expected_count = int(manifest["compatibility"]["num_states"])
    if int(completion.get("num_states", -1)) != expected_count:
        raise ValueError("completion state count differs from manifest")
    tensor_seals = completion.get("tensors")
    if not isinstance(tensor_seals, Mapping) or set(tensor_seals) != set(TENSOR_KEYS):
        raise ValueError("completion tensor seals are invalid")
    dimensions = _feature_dimensions_from_manifest(manifest)
    for key in TENSOR_KEYS:
        expected = {
            "shape": [expected_count, dimensions[key]],
            "dtype": "torch.float32",
            "content_sha256": tensor_content_sha256(tensors[key]),
        }
        if tensor_seals[key] != expected:
            raise ValueError(f"completion tensor seal differs for {key}")
    return completion


def _instantiate_dataset_and_contract(
    cfg: DictConfig,
    *,
    stats_path: Path,
    dataset_sources: Sequence[Mapping[str, Any]],
    context_cache: Mapping[str, Any],
) -> tuple[Any, list[dict[str, Any]], dict[int, dict[int, str]]]:
    paths = _dataset_instantiation_path_overrides(dataset_sources, context_cache)
    dataset = instantiate(
        cfg.data.train,
        **paths,
        is_training_set=False,
        pretrained_norm_stats=str(stats_path),
        strict_getitem=True,
        return_metadata=True,
        skip_padding_as_possible=False,
    )
    if not hasattr(dataset, "dataset_index_ranges") or not hasattr(
        dataset, "dataset_task_table"
    ):
        raise TypeError("strict feature dataset lacks auditable range/task-table APIs")
    ranges = _normalize_ranges(dataset.dataset_index_ranges())
    if sum(int(item["population"]) for item in ranges) != len(dataset):
        raise ValueError("dataset ranges do not cover the instantiated dataset")
    if [item["dataset_name"] for item in ranges] != [
        item["dataset_name"] for item in dataset_sources
    ]:
        raise ValueError("dataset order differs from Phase-2.5 artifact order")
    task_tables: dict[int, dict[int, str]] = {}
    for item in ranges:
        index = int(item["dataset_index"])
        raw = dataset.dataset_task_table(index)
        if not isinstance(raw, Mapping) or not raw:
            raise ValueError(f"dataset task table {index} is empty")
        task_tables[index] = {int(key): str(value) for key, value in raw.items()}
    return dataset, ranges, task_tables


def collect(cfg: DictConfig) -> dict[str, Any]:
    collector = cfg.FEATURE_COLLECTOR
    expected_num_states = int(collector.expected_num_states)
    if expected_num_states != 100:
        raise ValueError("Phase-3 V1 feature cache is hard-locked to exact Target-V2 100")
    raw_max_new_rows = collector.get("max_new_rows")
    max_new_rows = None if raw_max_new_rows is None else int(raw_max_new_rows)
    if max_new_rows is not None and max_new_rows <= 0:
        raise ValueError("FEATURE_COLLECTOR.max_new_rows must be positive or null")
    checkpoint_path = _resolve_existing_file(cfg.get("ckpt"), label="ckpt")
    stats_path = resolve_dataset_stats_path(checkpoint_path, collector.get("dataset_stats_path"))
    target_dir = Path(os.path.expandvars(os.path.expanduser(str(collector.target_v2_dir)))).resolve()
    target_manifest, targets = target_v2.load_target_bundle(
        target_dir,
        expected_manifest_sha256=str(collector.expected_target_manifest_sha256),
        expected_targets_sha256=str(collector.expected_target_records_sha256),
        expected_num_states=expected_num_states,
    )
    phase_path = _resolve_existing_file(
        collector.phase25_manifest_path, label="Phase-2.5 manifest"
    )
    phase25 = _load_phase25_manifest(
        phase_path, expected_sha256=str(collector.expected_phase25_manifest_sha256)
    )
    phase25["_manifest_path"] = str(phase_path)

    output_dir = Path(
        os.path.expandvars(os.path.expanduser(str(collector.output_dir)))
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".collector.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_stream:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another feature collector is using {output_dir}") from exc

        if not bool(collector.resume) and any(output_dir.iterdir()):
            allowed = {lock_path.name}
            existing = [path for path in output_dir.iterdir() if path.name not in allowed]
            if existing:
                raise FileExistsError("resume=false requires a fresh output directory")

        LOGGER.info("Hashing Target-bound checkpoint/stats/VAE/dataset/context artifacts")
        checkpoint = _stable_file_provenance(checkpoint_path, label="UniShare checkpoint")
        stats = _stable_file_provenance(stats_path, label="dataset stats")
        vae = _resolve_vae_artifact(cfg)
        dataset_sources: list[dict[str, Any]] = []
        for index, raw_path in enumerate(cfg.data.train.dataset_dirs):
            artifact = _directory_tree_provenance(
                raw_path, label=f"LIBERO source dataset {index}"
            )
            artifact["dataset_index"] = index
            artifact["dataset_name"] = Path(str(artifact["path"])).name
            dataset_sources.append(artifact)
        context_cache = _directory_tree_provenance(
            cfg.data.train.text_embedding_cache_dir, label="LIBERO text embedding cache"
        )
        model_config = OmegaConf.to_container(cfg.model, resolve=True)
        data_config = _scientific_data_config(OmegaConf.to_container(cfg.data, resolve=True))
        _assert_artifacts_match_phase25(
            phase25=phase25,
            target_manifest=target_manifest,
            checkpoint=checkpoint,
            stats=stats,
            vae=vae,
            dataset_sources=dataset_sources,
            context_cache=context_cache,
            model_config_sha256=sha256_json(model_config),
            data_config_sha256=sha256_json(data_config),
        )

        visual_cfg = collector.visual
        instruction_cfg = collector.instruction
        projections = build_projection_matrices(
            latent_channels=int(visual_cfg.latent_channels),
            pooled_height=int(visual_cfg.pooled_height),
            pooled_width=int(visual_cfg.pooled_width),
            visual_dim=int(visual_cfg.projection_dim),
            visual_seed=int(visual_cfg.projection_seed),
            context_dim=int(instruction_cfg.context_dim),
            instruction_dim=int(instruction_cfg.projection_dim),
            mean_seed=int(instruction_cfg.mean_projection_seed),
            rms_seed=int(instruction_cfg.rms_projection_seed),
        )
        extractor = _extractor_contract(
            OmegaConf.to_container(collector, resolve=True), projections
        )
        if extractor["feature_dimensions"] != EXPECTED_DIMS:
            raise ValueError("configured features differ from frozen 137-d protocol")
        scientific_sources = _scientific_source_provenance()
        manifest_payload = _build_manifest(
            cfg=cfg,
            target_manifest=target_manifest,
            targets=targets,
            target_dir=target_dir,
            target_manifest_sha256=_sha256_file(target_dir / target_v2.TARGET_MANIFEST_FILENAME),
            target_records_sha256=_sha256_file(target_dir / target_v2.TARGETS_FILENAME),
            phase25=phase25,
            phase25_manifest_sha256=_sha256_file(phase_path),
            checkpoint=checkpoint,
            stats=stats,
            vae=vae,
            dataset_sources=dataset_sources,
            context_cache=context_cache,
            extractor=extractor,
            scientific_sources=scientific_sources,
        )
        manifest_path = output_dir / MANIFEST_FILENAME
        manifest = ensure_immutable_manifest(manifest_path, manifest_payload)
        fingerprint = str(manifest["compatibility_fingerprint"])
        extractor_fingerprint = str(manifest["compatibility"]["extractor_fingerprint"])
        dimensions = _feature_dimensions_from_manifest(manifest)
        progress_dir = output_dir / ".rows"
        progress_dir.mkdir(exist_ok=True)

        dataset, ranges, task_tables = _instantiate_dataset_and_contract(
            cfg,
            stats_path=stats_path,
            dataset_sources=dataset_sources,
            context_cache=context_cache,
        )
        pending = [
            target
            for target in targets
            if _load_progress_row(
                progress_dir,
                target,
                extractor_fingerprint=extractor_fingerprint,
                expected_dimensions=dimensions,
            )
            is None
        ]
        LOGGER.info(
            "Feature plan: total=%d existing=%d pending=%d",
            len(targets),
            len(targets) - len(pending),
            len(pending),
        )

        vae_model = None
        device = torch.device(str(collector.device))
        if pending:
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("FEATURE_COLLECTOR.device requests unavailable CUDA")
            if device.type == "cuda" and device.index is not None:
                torch.cuda.set_device(device)
            # Load the exact content-hashed VAE directly. Constructing a random 5B
            # DiT merely to reach model.vae would waste memory and cannot affect
            # this protocol-fixed current-image representation.
            from fastwam.models.wan22.helpers.loader import _load_registered_model

            vae_model = _load_registered_model(
                str(vae["path"]),
                "wan_video_vae",
                torch_dtype={"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
                    str(cfg.get("mixed_precision", "bf16")).lower()
                ],
                device=str(device),
            )
            vae_model.requires_grad_(False)
            vae_model.eval()
            if any(parameter.requires_grad for parameter in vae_model.parameters()):
                raise AssertionError("feature VAE is not frozen")
            try:
                vae_dtype = next(vae_model.parameters()).dtype
            except StopIteration as exc:
                raise ValueError("loaded feature VAE has no parameters") from exc
            if vae_dtype not in {torch.float16, torch.bfloat16, torch.float32}:
                raise ValueError(f"unsupported feature VAE dtype {vae_dtype}")

        new_count = 0
        for target in tqdm(targets, desc="current-state gate features"):
            if max_new_rows is not None and new_count >= max_new_rows:
                break
            sample = dataset[int(target["source_index"])]
            state = _validate_live_state(
                sample=sample,
                target=target,
                ranges=ranges,
                task_tables=task_tables,
            )
            existing = _load_progress_row(
                progress_dir,
                target,
                extractor_fingerprint=extractor_fingerprint,
                expected_dimensions=dimensions,
            )
            if existing is not None:
                continue
            if vae_model is None:
                raise AssertionError("pending feature row has no frozen VAE model")

            def encode_current_image(image: torch.Tensor) -> torch.Tensor:
                value = prepare_vae_input(image, device=device, dtype=vae_dtype)
                with torch.inference_mode():
                    return vae_model.encode([value], device=device, tiled=False)

            with torch.inference_mode():
                features = extract_allowed_features(
                    input_image=state.input_image,
                    context=state.context,
                    proprio=state.proprio,
                    encode_current_image=encode_current_image,
                    projections=projections,
                    latent_channels=int(visual_cfg.latent_channels),
                    pooled_height=int(visual_cfg.pooled_height),
                    pooled_width=int(visual_cfg.pooled_width),
                    context_dim=int(instruction_cfg.context_dim),
                    proprio_dim=int(collector.proprio_dim),
                )
            record = build_feature_record(
                target, features, extractor_fingerprint=extractor_fingerprint
            )
            validate_feature_record(
                record,
                features,
                target,
                extractor_fingerprint=extractor_fingerprint,
                expected_dimensions=dimensions,
            )
            _write_progress_row(
                progress_dir,
                record,
                features,
                extractor_fingerprint=extractor_fingerprint,
            )
            new_count += 1

        completed_progress = sum(
            _load_progress_row(
                progress_dir,
                target,
                extractor_fingerprint=extractor_fingerprint,
                expected_dimensions=dimensions,
            )
            is not None
            for target in targets
        )
        if completed_progress != len(targets):
            return {
                "num_states": len(targets),
                "existing": len(targets) - len(pending),
                "new": new_count,
                "progress_rows": completed_progress,
                "complete": False,
                "output_dir": str(output_dir),
            }

        rows, matrices = _build_final_from_progress(
            progress_dir,
            targets,
            extractor_fingerprint=extractor_fingerprint,
            expected_dimensions=dimensions,
        )
        index_path, features_path = _publish_final_files(
            output_dir=output_dir,
            rows=rows,
            matrices=matrices,
            manifest_fingerprint=fingerprint,
        )
        completion_path = output_dir / COMPLETION_FILENAME
        if completion_path.exists():
            validate_completion(output_dir)
        else:
            completion = completion_payload(
                manifest_path=manifest_path,
                index_path=index_path,
                features_path=features_path,
                matrices=matrices,
                manifest_fingerprint=fingerprint,
                num_states=len(targets),
            )
            _atomic_write_bytes(completion_path, _serialize_json(completion))
            validate_completion(output_dir)
        return {
            "num_states": len(targets),
            "existing": len(targets) - new_count,
            "new": new_count,
            "progress_rows": len(targets),
            "complete": True,
            "output_dir": str(output_dir),
            "completion_sha256": _load_json(
                completion_path, label="feature-cache completion"
            )["completion_sha256"],
        }


@hydra.main(
    version_base="1.3",
    config_path="../../../configs",
    config_name="collect_libero_gate_features.yaml",
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    summary = collect(cfg)
    LOGGER.info("Gate current-state feature cache complete: %s", summary)


if __name__ == "__main__":
    main()
