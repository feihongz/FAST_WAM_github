import json
import inspect
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import torch
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

# try:
#     import rootutils

#     rootutils.setup_root(__file__, indicator=".python-version", pythonpath=True)
# except ModuleNotFoundError:
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def _prefer_fastwam_libero_package() -> None:
    fastwam_libero_root = os.environ.get("FASTWAM_LIBERO_ROOT")
    if fastwam_libero_root:
        sys.path = [p for p in sys.path if p != "/root/code/feihong/LIBERO/libero"]
        if fastwam_libero_root not in sys.path:
            sys.path.insert(0, fastwam_libero_root)


_prefer_fastwam_libero_package()

from experiments.libero.libero_utils import (
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    invert_gripper_action,
    quat2axisangle,
    save_prediction_video,
    save_rollout_video,
)
from experiments.libero.init_state_compat import load_libero_task_init_states
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.pytorch_utils import set_global_seed
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT


def _ensure_libero_compat_alias() -> None:
    import libero as libero_pkg

    sys.modules.setdefault("libero.libero", libero_pkg)


try:
    from libero.libero import benchmark
except ModuleNotFoundError as exc:
    if exc.name != "libero.libero":
        raise
    _ensure_libero_compat_alias()
    from libero import benchmark

from experiments.libero.action_ensembler import ActionEnsembler

from fastwam.alignment.eval_loading import (
    inspect_aligned_model_artifacts,
    load_prepared_aligned_model,
    verify_aligned_runtime_asset,
)
from fastwam.alignment.checkpointing import read_git_identity
from fastwam.gating.eval_runtime import (
    BoundPromptContext,
    ManifestBoundPromptContextProvider,
    load_gate_for_evaluation,
)
from fastwam.gating.routing import (
    BinaryVideoRouter,
    summarize_routing_telemetry,
)

for _resolver_name, _resolver in (
    ("eval", eval),
    ("max", lambda x: max(x)),
    ("split", lambda s, idx: s.split("/")[int(idx)]),
):
    if not OmegaConf.has_resolver(_resolver_name):
        OmegaConf.register_new_resolver(_resolver_name, _resolver)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_eval_device(cfg: DictConfig) -> str:
    eval_device = cfg.EVALUATION.get("device")
    if eval_device is not None:
        return str(eval_device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dataset_stats_path(cfg: DictConfig) -> Path:
    explicit = cfg.EVALUATION.get("dataset_stats_path")
    candidates: list[Path] = []

    if explicit is not None:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(str(explicit)))))

    ckpt = Path(os.path.expanduser(os.path.expandvars(str(cfg.ckpt))))
    for parent in list(ckpt.parents)[:4]:
        candidates.append(parent / "dataset_stats.json")

    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    msg = (
        "Failed to locate dataset_stats.json. Tried explicit "
        "EVALUATION.dataset_stats_path and checkpoint parent directories. "
        "Please pass EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )
    raise FileNotFoundError(msg)


def _optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "null"}:
        return None
    return text


def _stage3_expected_value(
    cfg: DictConfig,
    *,
    evaluation_key: str,
    fallback_key: str,
) -> Optional[str]:
    explicit = _optional_text(cfg.EVALUATION.get(evaluation_key))
    if explicit is not None:
        return explicit
    return _optional_text(OmegaConf.select(cfg, fallback_key))


def _prepare_stage3_eval_artifacts(
    cfg: DictConfig,
    *,
    dataset_stats_path: Path,
) -> Optional[dict[str, Any]]:
    aligned_target = "fastwam.runtime.create_fastwam_unified_aligned"
    model_target = _optional_text(OmegaConf.select(cfg, "model._target_"))
    adapter_path = _optional_text(cfg.EVALUATION.get("stage3_adapter_path"))
    adapter_sha256 = _optional_text(
        cfg.EVALUATION.get("stage3_adapter_sha256")
    )
    if adapter_path is None:
        if model_target == aligned_target:
            raise ValueError(
                "Aligned LIBERO endpoint eval requires "
                "EVALUATION.stage3_adapter_path and "
                "EVALUATION.stage3_adapter_sha256"
            )
        companion_fields = {
            name: cfg.EVALUATION.get(name)
            for name in (
                "stage3_adapter_sha256",
                "stage3_base_sha256",
                "stage3_data_manifest_sha256",
                "stage3_training_contract_sha256",
                "stage3_global_step",
            )
            if _optional_text(cfg.EVALUATION.get(name)) is not None
        }
        if companion_fields:
            raise ValueError(
                "Stage 3 identity fields require "
                "EVALUATION.stage3_adapter_path: "
                f"{sorted(companion_fields)}"
            )
        return None
    if adapter_sha256 is None:
        raise ValueError(
            "EVALUATION.stage3_adapter_sha256 is required when loading "
            "a Stage 3 Adapter"
        )

    if model_target != aligned_target:
        raise ValueError(
            "Stage 3 endpoint eval requires "
            "task=libero_stage3_alignment_2cam224_1e-4"
        )
    base_sha256 = _stage3_expected_value(
        cfg,
        evaluation_key="stage3_base_sha256",
        fallback_key="base.expected_sha256",
    )
    data_manifest_sha256 = _stage3_expected_value(
        cfg,
        evaluation_key="stage3_data_manifest_sha256",
        fallback_key="data_manifest.expected_sha256",
    )
    if base_sha256 is None or data_manifest_sha256 is None:
        raise ValueError(
            "Stage 3 endpoint eval requires locked base and data-manifest SHA256 "
            "values from the selected Stage 3 task or explicit EVALUATION overrides"
        )

    vae_path = _optional_text(OmegaConf.select(cfg, "assets.vae.path"))
    vae_sha256 = _optional_text(
        OmegaConf.select(cfg, "assets.vae.expected_sha256")
    )
    stats_sha256 = _optional_text(
        OmegaConf.select(cfg, "assets.normalization_stats.expected_sha256")
    )
    if vae_path is None or vae_sha256 is None or stats_sha256 is None:
        raise ValueError(
            "Stage 3 endpoint eval requires locked VAE and normalization-stats "
            "asset identities from the selected Stage 3 task"
        )

    expected_global_step = cfg.EVALUATION.get("stage3_global_step")
    if _optional_text(expected_global_step) is None:
        expected_global_step = None
    else:
        expected_global_step = int(expected_global_step)
    return inspect_aligned_model_artifacts(
        base_checkpoint_path=str(cfg.ckpt),
        expected_base_checkpoint_sha256=base_sha256,
        alignment_export_path=adapter_path,
        expected_alignment_export_sha256=adapter_sha256,
        expected_data_manifest_sha256=data_manifest_sha256,
        expected_training_contract_sha256=_optional_text(
            cfg.EVALUATION.get("stage3_training_contract_sha256")
        ),
        expected_global_step=expected_global_step,
        asset_paths={
            "vae": vae_path,
            "normalization_stats": dataset_stats_path,
        },
        expected_asset_sha256={
            "vae": vae_sha256,
            "normalization_stats": stats_sha256,
        },
    )


def _required_evaluation_text(cfg: DictConfig, key: str) -> str:
    value = _optional_text(cfg.EVALUATION.get(key))
    if value is None:
        raise ValueError(f"EVALUATION.{key} is required")
    return value


def _configured_inference_steps(cfg: DictConfig) -> int:
    value = cfg.EVALUATION.get("num_inference_steps", None)
    steps = int(cfg.get("eval_num_inference_steps", 20) if value is None else value)
    if steps <= 0:
        raise ValueError(
            f"EVALUATION.num_inference_steps must be positive, got {steps}"
        )
    return steps


def _prepare_video_router(
    cfg: DictConfig,
    *,
    stage3_artifact_identity: Optional[dict[str, Any]],
    model_device: str,
) -> tuple[
    BinaryVideoRouter,
    Optional[ManifestBoundPromptContextProvider],
    dict[str, Any],
]:
    """Build one query router and bind every learned-Gate input artifact."""

    routing_mode = str(cfg.EVALUATION.get("routing_mode", "static")).strip().lower()
    configured_video_steps = _configured_inference_steps(cfg)
    use_manifest_text_cache = bool(
        cfg.EVALUATION.get("use_manifest_text_cache", False)
    )
    prompt_context_provider: Optional[ManifestBoundPromptContextProvider] = None
    gate_identity: Optional[dict[str, Any]] = None

    if routing_mode == "static":
        inference_mode = str(
            cfg.EVALUATION.get("inference_mode", "wo")
        ).strip().lower()
        router = BinaryVideoRouter(
            routing_mode="static",
            configured_video_steps=configured_video_steps,
            inference_mode=inference_mode,
        )
    elif routing_mode == "gate":
        if stage3_artifact_identity is None:
            raise ValueError("Gate routing requires a strictly verified Stage 3 endpoint")
        # The Stage 2 labels and Gate checkpoint are defined for the complete
        # N=10 with-video branch.  Silently changing N would change the target.
        if configured_video_steps != 10:
            raise ValueError(
                "Gate routing is bound to N=10; set "
                "EVALUATION.num_inference_steps=10"
            )
        untracked = OmegaConf.to_container(
            cfg.EVALUATION.get("gate_expected_git_untracked_source_files", []),
            resolve=True,
        )
        if not isinstance(untracked, list) or any(
            not isinstance(path, str) for path in untracked
        ):
            raise ValueError(
                "EVALUATION.gate_expected_git_untracked_source_files must be a list of strings"
            )
        loaded_gate = load_gate_for_evaluation(
            _required_evaluation_text(cfg, "gate_checkpoint"),
            expected_checkpoint_sha256=_required_evaluation_text(
                cfg, "gate_checkpoint_sha256"
            ),
            expected_label_manifest_sha256=_required_evaluation_text(
                cfg, "gate_expected_label_manifest_sha256"
            ),
            expected_adapter_checkpoint_sha256=stage3_artifact_identity[
                "alignment_export"
            ]["sha256"],
            expected_base_checkpoint_sha256=stage3_artifact_identity[
                "base_checkpoint"
            ]["sha256"],
            expected_data_manifest_sha256=stage3_artifact_identity[
                "data_manifest_sha256"
            ],
            expected_episode_split_assignment_sha256=_required_evaluation_text(
                cfg, "gate_expected_episode_split_assignment_sha256"
            ),
            expected_training_config_sha256=_required_evaluation_text(
                cfg, "gate_expected_training_config_sha256"
            ),
            expected_git_identity={
                "commit": _required_evaluation_text(
                    cfg, "gate_expected_git_commit"
                ),
                "tracked_dirty": bool(
                    cfg.EVALUATION.get("gate_expected_git_tracked_dirty", False)
                ),
                "untracked_source_files": untracked,
            },
            device=model_device,
        )
        gate_identity = loaded_gate.identity
        router = BinaryVideoRouter(
            routing_mode="gate",
            configured_video_steps=configured_video_steps,
            gate=loaded_gate.gate,
            threshold=float(cfg.EVALUATION.get("gate_threshold", 0.5)),
        )
        use_manifest_text_cache = True
    elif routing_mode == "random":
        router = BinaryVideoRouter(
            routing_mode="random",
            configured_video_steps=configured_video_steps,
            random_seed=int(cfg.EVALUATION.get("random_seed", 42)),
            random_video_probability=float(
                cfg.EVALUATION.get("random_video_probability", 0.5)
            ),
        )
    else:
        raise ValueError(
            "EVALUATION.routing_mode must be one of: static, gate, random"
        )

    if use_manifest_text_cache:
        if stage3_artifact_identity is None:
            raise ValueError(
                "Manifest text-cache inference requires a strictly verified Stage 3 endpoint"
            )
        manifest_path = _optional_text(OmegaConf.select(cfg, "data_manifest.path"))
        if manifest_path is None:
            raise ValueError("data_manifest.path is required for cached-context eval")
        prompt_context_provider = ManifestBoundPromptContextProvider(
            manifest_path,
            expected_manifest_sha256=stage3_artifact_identity[
                "data_manifest_sha256"
            ],
            expected_prompt_template=DEFAULT_PROMPT,
        )

    identity = {
        "schema_version": 1,
        "kind": "binary_video_routing_runtime",
        "routing_mode": routing_mode,
        "inference_mode": router.inference_mode,
        "configured_video_steps": configured_video_steps,
        "gate_threshold": router.threshold,
        "gate_decision_rule": (
            "sigmoid(logit) >= gate_threshold -> w"
            if routing_mode == "gate"
            else None
        ),
        "random_video_probability": router.random_video_probability,
        "random_seed": (
            int(cfg.EVALUATION.get("random_seed", 42))
            if routing_mode == "random"
            else None
        ),
        "use_manifest_text_cache": use_manifest_text_cache,
        "prompt_template": DEFAULT_PROMPT,
        "gate_input_preprocessing": (
            "processor.val_transforms_per_camera_then_concat_then_2x_minus_1"
            if routing_mode == "gate"
            else None
        ),
        "timing": {
            "enabled": bool(cfg.EVALUATION.get("timing_enabled", True)),
            "warmup_queries_per_task": int(
                cfg.EVALUATION.get("timing_warmup_queries", 0)
            ),
            "save_query_metrics": bool(
                cfg.EVALUATION.get("save_query_metrics", True)
            ),
        },
        "gate_checkpoint": gate_identity,
    }
    return router, prompt_context_provider, identity


def _load_model_checkpoint(
    model: torch.nn.Module,
    ckpt: str,
    *,
    stage3_artifact_identity: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    if stage3_artifact_identity is not None:
        identity = load_prepared_aligned_model(
            model,
            stage3_artifact_identity,
        )
        logging.info(
            "Loaded strict Stage 3 endpoint model: %s",
            json.dumps(identity, sort_keys=True),
        )
        return identity

    model.load_checkpoint(ckpt)
    logging.info("Loaded checkpoint via model.load_checkpoint: %s", ckpt)
    return None


def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil_image = Image.fromarray(image)
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    cropped = resized.crop((left, top, left + width, top + height))
    return np.asarray(cropped, dtype=np.uint8)


def _normalize_proprio(
    proprio: np.ndarray,
    processor: FastWAMProcessor,
) -> torch.Tensor:
    state_meta = processor.shape_meta["state"]
    if len(state_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged state key in shape_meta['state']."
        )
    state_key = state_meta[0]["key"]

    state_batch = {"state": {state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}}
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    return state_batch["state"][state_key]


def _obs_to_model_input(
    obs: dict,
    cfg: DictConfig,
    processor: FastWAMProcessor,
    width: int,
    height: int,
    device: str,
    dtype: torch.dtype,
):
    imgs = get_libero_image(obs)
    image_meta = processor.shape_meta["images"]
    if len(image_meta) < int(processor.num_output_cameras):
        raise ValueError(
            f"shape_meta.images has {len(image_meta)} entries, "
            f"but num_output_cameras={processor.num_output_cameras}."
        )

    def _meta_to_hw(meta: dict, camera_idx: int) -> tuple[int, int]:
        shape = meta["shape"]
        if len(shape) != 3:
            raise ValueError(f"shape_meta.images[{camera_idx}].shape must be [C,H,W], got {shape}")
        return int(shape[1]), int(shape[2])

    concatenation = cfg.data.train.get("concat_multi_camera", "horizontal")
    num_cameras = processor.num_output_cameras
    if num_cameras == 1:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        rgb = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
    elif num_cameras == 2:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        wrist_h, wrist_w = _meta_to_hw(image_meta[1], camera_idx=1)
        primary = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
        wrist = _center_crop_resize(imgs["wrist_image"], width=wrist_w, height=wrist_h)
        if concatenation == "horizontal":
            rgb = np.concatenate([primary, wrist], axis=1)
        elif concatenation == "vertical":
            rgb = np.concatenate([primary, wrist], axis=0)
        else:
            raise ValueError(f"Invalid concat_multi_camera: {concatenation}")
    else:
        raise ValueError(f"LIBERO eval currently supports num_output_cameras in [1, 2], got {num_cameras}.")

    actual_h, actual_w = int(rgb.shape[0]), int(rgb.shape[1])
    expected_h, expected_w = int(height), int(width)
    image_shapes = [meta["shape"] for meta in image_meta]
    assert actual_h == expected_h and actual_w == expected_w, (
        "Input image size mismatch after per-camera resize + concat: "
        f"got (H,W)=({actual_h},{actual_w}), expected (H,W)=({expected_h},{expected_w}) "
        f"from data.train.video_size={[expected_h, expected_w]}; "
        f"shape_meta.images={image_shapes}, concat_multi_camera={concatenation}."
    )

    x = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    x = x * (2.0 / 255.0) - 1.0

    proprio = _normalize_proprio(_extract_sim_state(obs), processor)

    return x, proprio, imgs


def _obs_to_gate_input(
    imgs: dict[str, np.ndarray],
    *,
    cfg: DictConfig,
    processor: FastWAMProcessor,
    device: str,
) -> torch.Tensor:
    """Apply the exact processor image transforms used by Gate training."""

    transforms = processor.val_transforms
    if transforms is None:
        raise ValueError("Gate eval requires processor.val_transforms")
    camera_tensors: list[torch.Tensor] = []
    for camera_idx, meta in enumerate(processor.shape_meta["images"]):
        if camera_idx >= int(processor.num_output_cameras):
            break
        key = str(meta["key"])
        if key not in imgs:
            raise KeyError(f"LIBERO observation is missing Gate camera {key!r}")
        array = np.ascontiguousarray(imgs[key])
        if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
            raise ValueError(
                f"Gate camera {key!r} must be uint8 HWC RGB, got "
                f"shape={array.shape}, dtype={array.dtype}"
            )
        image = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        current_transforms = (
            transforms[key]
            if isinstance(transforms, (dict, DictConfig))
            else transforms
        )
        for transform in current_transforms:
            image = transform(image)
        expected_shape = (1, *tuple(int(x) for x in meta["shape"]))
        if tuple(image.shape) != expected_shape:
            raise ValueError(
                f"Gate camera {key!r} preprocessing shape mismatch: "
                f"expected={expected_shape}, actual={tuple(image.shape)}"
            )
        camera_tensors.append(image)

    num_cameras = int(processor.num_output_cameras)
    if len(camera_tensors) != num_cameras:
        raise ValueError(
            f"Gate expected {num_cameras} cameras, prepared {len(camera_tensors)}"
        )
    concatenation = str(cfg.data.train.get("concat_multi_camera", "horizontal"))
    if num_cameras == 1:
        image = camera_tensors[0]
    elif concatenation == "horizontal":
        image = torch.cat(camera_tensors, dim=-1)
    elif concatenation == "vertical":
        image = torch.cat(camera_tensors, dim=-2)
    else:
        raise ValueError(
            "LIBERO Gate preprocessing supports horizontal/vertical camera concat, "
            f"got {concatenation!r}"
        )

    expected_h, expected_w = (int(x) for x in cfg.data.train.video_size)
    if tuple(image.shape) != (1, 3, expected_h, expected_w):
        raise ValueError(
            "Gate concatenated image shape mismatch: "
            f"expected={(1, 3, expected_h, expected_w)}, actual={tuple(image.shape)}"
        )
    if not image.is_floating_point():
        raise ValueError("Gate processor image transforms must produce floating point")
    if not bool(torch.isfinite(image).all().item()):
        raise ValueError("Gate processor image contains non-finite values")
    if float(image.min().item()) < 0.0 or float(image.max().item()) > 1.0:
        raise ValueError("Gate processor image must be in [0,1] before normalization")
    return (image.to(device=device, dtype=torch.float32) * 2.0) - 1.0


def _extract_sim_state(obs: dict) -> np.ndarray:
    """Build simulator state from current observation.

    This is used as proprio input for model inference.
    """
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)
    return state


def _denormalize_action(action: torch.Tensor, processor: FastWAMProcessor) -> np.ndarray:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3:
        raise ValueError(f"Expected action tensor [B, T, D], got {tuple(action.shape)}")

    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged action key in shape_meta['action']."
        )

    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    action = action.to(dtype=torch.float32, device="cpu")
    denorm = normalizer.backward(action)
    return denorm.numpy()


def _get_num_video_frames(cfg: DictConfig) -> int:
    return (int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1


def _validate_visualize_future_video_cfg(cfg: DictConfig) -> None:
    if not bool(cfg.EVALUATION.get("visualize_future_video", False)):
        return

    routing_mode = str(cfg.EVALUATION.get("routing_mode", "static")).lower()
    if routing_mode != "static":
        raise ValueError(
            "Gate/random routing requires EVALUATION.visualize_future_video=false"
        )

    action_conditioned = cfg.model.video_dit_config.get("action_conditioned", None)
    if action_conditioned is not False:
        raise ValueError(
            "EVALUATION.visualize_future_video=true requires "
            "model.video_dit_config.action_conditioned=false."
        )


def _select_predicted_future_frames(pred_video: list[Image.Image], cfg: DictConfig) -> list[Image.Image]:
    if len(pred_video) == 0:
        raise ValueError("`infer_joint` returned an empty predicted video.")

    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = replan_steps // action_video_freq_ratio
    keep_frames = 1 + num_future_frames
    return list(pred_video[:keep_frames])


def _get_future_frame_capture_steps(cfg: DictConfig) -> list[int]:
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = replan_steps // action_video_freq_ratio
    return [step_idx * action_video_freq_ratio for step_idx in range(num_future_frames + 1)]


def _frame_to_rgb_array(frame: Any) -> np.ndarray:
    if isinstance(frame, dict):
        images = []
        for value in frame.values():
            value_array = np.array(value) if isinstance(value, Image.Image) else np.array(value, copy=True)
            images.append(value_array)
        return np.concatenate(images, axis=1)
    if isinstance(frame, Image.Image):
        return np.array(frame.convert("RGB"))
    return np.array(frame, copy=True)


def _compute_clip_mean_psnr(
    gt_frames: list[Any],
    pred_frames: list[Any],
    eps: float = 1e-8,
) -> Optional[float]:
    if len(gt_frames) == 0 or len(pred_frames) == 0:
        return None
    assert len(gt_frames) == len(pred_frames), (
        "GT/pred frame count mismatch for PSNR: "
        f"len(gt_frames)={len(gt_frames)} len(pred_frames)={len(pred_frames)}. "
        "This indicates temporal misalignment in future-video capture."
    )
    num_frames = len(gt_frames)

    frame_psnr_values = []
    for gt_frame, pred_frame in zip(gt_frames[:num_frames], pred_frames[:num_frames]):
        gt_image = _frame_to_rgb_array(gt_frame)
        pred_image = _frame_to_rgb_array(pred_frame)
        target_h, target_w = pred_image.shape[:2]
        if gt_image.shape[:2] != (target_h, target_w):
            gt_image = np.array(
                Image.fromarray(gt_image).resize((target_w, target_h), resample=Image.BILINEAR)
            )

        gt_f32 = gt_image.astype(np.float32)
        pred_f32 = pred_image.astype(np.float32)
        mse = float(np.mean((pred_f32 - gt_f32) ** 2))
        psnr = 10.0 * np.log10((255.0 * 255.0) / max(mse, eps))
        frame_psnr_values.append(float(psnr))

    if len(frame_psnr_values) == 0:
        return None
    return float(np.mean(frame_psnr_values))


def _is_black_frame(
    frame: Any,
    *,
    mean_threshold: float,
    std_threshold: float,
) -> bool:
    image = _frame_to_rgb_array(frame)
    if image.size == 0:
        return True
    if image.ndim == 2:
        image = image[..., None]
    image_f32 = image.astype(np.float32)
    return (
        float(image_f32.mean()) <= float(mean_threshold)
        and float(image_f32.std()) <= float(std_threshold)
    )


def _invalid_episode_reason(replay_images: list, cfg: DictConfig) -> Optional[str]:
    if len(replay_images) == 0:
        return "no_replay_images"
    if not bool(cfg.EVALUATION.get("black_screen_filter", False)):
        return None

    black_count = sum(
        1
        for frame in replay_images
        if _is_black_frame(
            frame,
            mean_threshold=float(cfg.EVALUATION.get("black_screen_mean_threshold", 5.0)),
            std_threshold=float(cfg.EVALUATION.get("black_screen_std_threshold", 2.0)),
        )
    )
    fraction = black_count / max(len(replay_images), 1)
    min_fraction = float(cfg.EVALUATION.get("black_screen_min_frame_fraction", 0.8))
    if fraction >= min_fraction:
        return f"black_screen_frames={black_count}/{len(replay_images)}"
    return None


def _synchronize_for_timing(device: str, *, enabled: bool) -> None:
    if not enabled:
        return
    target = torch.device(device)
    if target.type == "cuda":
        torch.cuda.synchronize(target)


def _predict_action_chunk(
    obs: dict,
    task_description: str,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    router: BinaryVideoRouter,
    prompt_context: Optional[BoundPromptContext],
) -> tuple[
    np.ndarray,
    dict,
    Optional[list[Image.Image]],
    dict[str, Any],
]:
    num_inference_steps = _configured_inference_steps(cfg)
    prompt_template = DEFAULT_PROMPT
    prompt = prompt_template.format(task=task_description)
    timing_enabled = bool(cfg.EVALUATION.get("timing_enabled", True))

    total_started_at = time.perf_counter()
    image, proprio, imgs = _obs_to_model_input(
        obs,
        cfg=cfg,
        processor=processor,
        width=input_w,
        height=input_h,
        device=model_device,
        dtype=model.torch_dtype,
    )
    if router.routing_mode == "gate":
        if prompt_context is None:
            raise ValueError("Gate routing requires a manifest-bound prompt context")
        gate_image = _obs_to_gate_input(
            imgs,
            cfg=cfg,
            processor=processor,
            device=model_device,
        )
        route_decision = router.route(
            input_image=gate_image,
            context=prompt_context.context,
            context_mask=prompt_context.gate_context_mask,
            proprio=proprio.to(device=model_device, dtype=torch.float32),
        )
    else:
        route_decision = router.route()
    inference_mode = str(route_decision["selected_mode"])
    preprocessing_finished_at = time.perf_counter()

    infer_kwargs = {
        "prompt": prompt if prompt_context is None else None,
        "input_image": image,
        "action_horizon": action_horizon,
        "negative_prompt": str(cfg.EVALUATION.get("negative_prompt", "")),
        "text_cfg_scale": float(cfg.EVALUATION.get("text_cfg_scale", 1.0)),
        "num_inference_steps": num_inference_steps,
        "proprio": proprio,
        "sigma_shift": (
            None
            if cfg.EVALUATION.get("sigma_shift") is None
            else float(cfg.EVALUATION.get("sigma_shift"))
        ),
        "seed": None if cfg.get("seed") is None else int(cfg.seed),
        "rand_device": str(cfg.EVALUATION.get("rand_device", "cpu")),
        "tiled": bool(cfg.EVALUATION.get("tiled", False)),
    }
    if prompt_context is not None:
        infer_kwargs["context"] = prompt_context.context
        infer_kwargs["context_mask"] = prompt_context.model_context_mask
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    predicted_future_frames = None
    if visualize_future_video or inference_mode == "w":
        infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)
    elif (
        not hasattr(model, "infer_action_mode")
        and "num_video_frames" in inspect.signature(model.infer_action).parameters
    ):
        infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)

    _synchronize_for_timing(model_device, enabled=timing_enabled)
    policy_started_at = time.perf_counter()
    with torch.no_grad():
        if visualize_future_video:
            if hasattr(model, "infer_joint_mode"):
                pred = model.infer_joint_mode(
                    **infer_kwargs,
                    inference_mode=inference_mode,
                )
            else:
                if inference_mode != "wo":
                    raise ValueError(
                        f"Model {type(model).__name__} does not support "
                        f"joint inference_mode={inference_mode}"
                    )
                pred = model.infer_joint(**infer_kwargs)
            predicted_future_frames = _select_predicted_future_frames(pred["video"], cfg)
        elif hasattr(model, "infer_action_mode"):
            pred = model.infer_action_mode(**infer_kwargs, inference_mode=inference_mode)
        else:
            if inference_mode != "wo":
                raise ValueError(
                    f"Model {type(model).__name__} does not support inference_mode={inference_mode}"
                )
            pred = model.infer_action(**infer_kwargs)
    _synchronize_for_timing(model_device, enabled=timing_enabled)
    policy_finished_at = time.perf_counter()
    action = pred["action"]  # [T, D]

    action = _denormalize_action(action, processor)[0]  # [T, D]

    # The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    action[..., -1] = action[..., -1] * 2 - 1
    action = invert_gripper_action(action)
    if bool(cfg.EVALUATION.get("binarize_gripper", False)):
        action[..., -1] = np.sign(action[..., -1])
    total_finished_at = time.perf_counter()
    query_metrics = dict(route_decision)
    query_metrics.update(
        {
            "preprocess_plus_gate_latency_s": float(
                preprocessing_finished_at - total_started_at
            ),
            "policy_latency_s": float(policy_finished_at - policy_started_at),
            "total_latency_s": float(total_finished_at - total_started_at),
            "timing_synchronized": timing_enabled,
        }
    )
    return action, imgs, predicted_future_frames, query_metrics


def _get_max_steps(task_suite_name: str) -> int:
    suite_steps = {
        "libero_spatial": 400,
        "libero_object": 400,
        "libero_goal": 400,
        "libero_10": 700,
        "libero_90": 700,
    }
    if task_suite_name not in suite_steps:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    return suite_steps[task_suite_name]


def _query_latency_summary(
    query_metrics: list[dict[str, Any]],
    key: str,
) -> dict[str, Any]:
    values = [
        float(query[key])
        for query in query_metrics
        if bool(query.get("timing_included", True)) and query.get(key) is not None
    ]
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None}
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def _summarize_routing_queries(
    query_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    decisions = [dict(query) for query in query_metrics]
    policy_latencies = [float(query["policy_latency_s"]) for query in query_metrics]
    summary = summarize_routing_telemetry(
        decisions,
        policy_latencies_s=policy_latencies,
    )
    timed = [query for query in query_metrics if bool(query.get("timing_included", True))]
    if timed:
        timed_summary = summarize_routing_telemetry(
            timed,
            policy_latencies_s=[float(query["policy_latency_s"]) for query in timed],
        )
        summary["latency_s"] = timed_summary["latency_s"]
        for route in ("wo", "w"):
            summary["by_route"][route]["latency_s"] = timed_summary["by_route"][
                route
            ]["latency_s"]
    else:
        summary["latency_s"] = {
            "gate": {"mean": None, "p50": None, "p95": None},
            "policy": {"mean": None, "p50": None, "p95": None},
        }
        for route in ("wo", "w"):
            summary["by_route"][route]["latency_s"] = {
                "gate": {"mean": None, "p50": None, "p95": None},
                "policy": {"mean": None, "p50": None, "p95": None},
            }

    summary["latency_s"]["preprocess_plus_gate"] = _query_latency_summary(
        query_metrics,
        "preprocess_plus_gate_latency_s",
    )
    summary["latency_s"]["total"] = _query_latency_summary(
        query_metrics,
        "total_latency_s",
    )
    for route in ("wo", "w"):
        route_queries = [
            query for query in query_metrics if query["selected_mode"] == route
        ]
        summary["by_route"][route]["latency_s"][
            "preprocess_plus_gate"
        ] = _query_latency_summary(route_queries, "preprocess_plus_gate_latency_s")
        summary["by_route"][route]["latency_s"]["total"] = (
            _query_latency_summary(route_queries, "total_latency_s")
        )
    summary["timing"] = {
        "query_count": len(timed),
        "warmup_query_count": len(query_metrics) - len(timed),
        "cuda_synchronized": bool(
            all(bool(query.get("timing_synchronized", False)) for query in query_metrics)
        )
        if query_metrics
        else None,
    }
    return summary


def run_single_episode(
    env,
    initial_state,
    task_description: str,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    episode_idx: int,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    router: BinaryVideoRouter,
    prompt_context: Optional[BoundPromptContext],
    routing_trace: list[dict[str, Any]],
) -> tuple[bool, list, list[dict[str, Any]], Optional[float]]:
    max_steps = _get_max_steps(cfg.EVALUATION.task_suite_name)
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    num_steps_wait = int(cfg.EVALUATION.get("num_steps_wait", 5))
    use_action_ensembler = bool(cfg.EVALUATION.get("use_action_ensembler", False))
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    capture_steps = set(_get_future_frame_capture_steps(cfg)[1:])

    env.reset()
    obs = env.set_init_state(initial_state)
    if use_action_ensembler:
        ensembler = ActionEnsembler()
        ensembler.reset()

    replay_images = []
    predicted_future_video_clips: list[dict[str, Any]] = []
    episode_future_clip_psnr: list[float] = []
    pending_actions: list[list[float]] = []
    current_predicted_future_clip: Optional[dict[str, Any]] = None
    current_replan_step = 0
    current_replan_idx = -1

    t = 0
    done = False
    pbar = tqdm(total=max_steps + num_steps_wait, desc=f"Episode {episode_idx + 1}")
    while t < max_steps + num_steps_wait:
        pbar.update(1)
        if t < num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action())
            t += 1
            continue

        if len(pending_actions) == 0:
            current_replan_idx += 1
            action_chunk, imgs, predicted_future_frames, query_metrics = _predict_action_chunk(
                obs=obs,
                task_description=task_description,
                model=model,
                processor=processor,
                cfg=cfg,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
                router=router,
                prompt_context=prompt_context,
            )
            query_metrics.update(
                {
                    "episode_index": int(episode_idx),
                    "replan_index": int(current_replan_idx),
                    "environment_step": int(t),
                    "query_id": (
                        f"{cfg.EVALUATION.task_suite_name}/"
                        f"{int(cfg.EVALUATION.task_id)}/"
                        f"{episode_idx}/{current_replan_idx}"
                    ),
                }
            )
            routing_trace.append(query_metrics)
            if predicted_future_frames is not None:
                current_predicted_future_clip = {
                    "replan_idx": current_replan_idx,
                    "gt_frames": [imgs.copy()],
                    "pred_frames": predicted_future_frames,
                }
            else:
                current_predicted_future_clip = None
            current_replan_step = 0
            if use_action_ensembler:
                ensembler.add_actions(action_chunk, t)
                pending_actions = [ensembler.get_action(ts).tolist() for ts in range(t, t + replan_steps)]
            else:
                pending_actions = action_chunk[:replan_steps].tolist()
            replay_images.append(imgs.copy())
        else:
            imgs = get_libero_image(obs)
            replay_images.append(imgs.copy())

        obs, _, done, _ = env.step(pending_actions.pop(0))
        if visualize_future_video and current_predicted_future_clip is not None:
            current_replan_step += 1
            if current_replan_step in capture_steps:
                current_predicted_future_clip["gt_frames"].append(get_libero_image(obs))
            if done or len(pending_actions) == 0:
                expected_frame_count = 1 + sum(
                    1 for capture_step in capture_steps if capture_step <= current_replan_step
                )
                gt_len = len(current_predicted_future_clip["gt_frames"])
                pred_len = len(current_predicted_future_clip["pred_frames"])
                assert gt_len == expected_frame_count, (
                    "GT future frames do not match expected capture count: "
                    f"gt_len={gt_len} expected={expected_frame_count} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']} "
                    f"current_replan_step={current_replan_step} capture_steps={sorted(capture_steps)}."
                )
                assert pred_len >= expected_frame_count, (
                    "Predicted future frames shorter than expected capture count: "
                    f"pred_len={pred_len} expected={expected_frame_count} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']}."
                )
                if pred_len != expected_frame_count:
                    logging.info(
                        "Align predicted clip length to executed steps: "
                        "episode=%s replan=%s done=%s expected=%s pred_full=%s",
                        episode_idx,
                        current_predicted_future_clip["replan_idx"],
                        done,
                        expected_frame_count,
                        pred_len,
                    )
                current_predicted_future_clip["pred_frames"] = current_predicted_future_clip["pred_frames"][
                    :expected_frame_count
                ]
                assert len(current_predicted_future_clip["gt_frames"]) == len(
                    current_predicted_future_clip["pred_frames"]
                ), (
                    "GT/pred frame count mismatch after alignment: "
                    f"len(gt_frames)={len(current_predicted_future_clip['gt_frames'])} "
                    f"len(pred_frames)={len(current_predicted_future_clip['pred_frames'])} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']}."
                )
                clip_psnr = _compute_clip_mean_psnr(
                    current_predicted_future_clip["gt_frames"],
                    current_predicted_future_clip["pred_frames"],
                )
                if clip_psnr is not None:
                    episode_future_clip_psnr.append(clip_psnr)
                predicted_future_video_clips.append(current_predicted_future_clip)
                current_predicted_future_clip = None
        if done:
            break
        t += 1
    pbar.close()

    episode_mean_psnr = (
        float(np.mean(episode_future_clip_psnr)) if len(episode_future_clip_psnr) > 0 else None
    )
    return bool(done), replay_images, predicted_future_video_clips, episode_mean_psnr


def run_single_task(
    task,
    initial_states,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    video_dir: Path,
    predicted_video_dir: Path,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    router: BinaryVideoRouter,
    prompt_context_provider: Optional[ManifestBoundPromptContextProvider],
) -> dict:
    env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, cfg.get("seed"))
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    save_videos = bool(cfg.EVALUATION.get("save_videos", True))
    retry_invalid_episodes = bool(cfg.EVALUATION.get("retry_invalid_episodes", False))
    max_invalid_retries = int(cfg.EVALUATION.get("max_invalid_episode_retries", 20))
    save_query_metrics = bool(cfg.EVALUATION.get("save_query_metrics", True))
    timing_warmup_queries = int(cfg.EVALUATION.get("timing_warmup_queries", 0))
    if timing_warmup_queries < 0:
        raise ValueError("EVALUATION.timing_warmup_queries must be non-negative")
    prompt_context = None
    if prompt_context_provider is not None:
        prompt_context = prompt_context_provider.load(
            DEFAULT_PROMPT.format(task=task_description),
            device=model_device,
        )
    results = {
        "successes": 0,
        "failure_episodes": [],
        "success_episodes": [],
        "invalid_episodes": [],
        "invalid_episode_count": 0,
        "attempted_episodes": 0,
        "task_description": task_description,
        "routing": {
            "summary": None,
            "episodes": [],
            "invalid_attempts": [],
            "prompt_context": (
                None if prompt_context is None else prompt_context.identity
            ),
        },
    }
    if visualize_future_video:
        results["episode_future_video_psnr"] = []
        results["future_video_psnr_mean"] = None

    target_valid_trials = int(cfg.EVALUATION.num_trials)
    valid_trial_idx = 0
    invalid_retries = 0
    attempted_idx = 0
    valid_query_metrics: list[dict[str, Any]] = []

    while valid_trial_idx < target_valid_trials:
        attempt_routing_trace: list[dict[str, Any]] = []
        try:
            success, replay_images, predicted_future_video_clips, episode_mean_psnr = run_single_episode(
                env=env,
                initial_state=initial_states[valid_trial_idx],
                task_description=task_description,
                model=model,
                processor=processor,
                cfg=cfg,
                episode_idx=valid_trial_idx,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
                router=router,
                prompt_context=prompt_context,
                routing_trace=attempt_routing_trace,
            )
            invalid_reason = _invalid_episode_reason(replay_images, cfg)
        except Exception as exc:
            if not retry_invalid_episodes:
                raise
            logging.exception(
                "Invalid LIBERO episode due to exception: task=%s trial=%s attempt=%s",
                cfg.EVALUATION.task_id,
                valid_trial_idx,
                attempted_idx,
            )
            success = False
            replay_images = []
            predicted_future_video_clips = []
            episode_mean_psnr = None
            invalid_reason = f"exception:{type(exc).__name__}"

        results["attempted_episodes"] += 1
        attempted_idx += 1

        if invalid_reason is not None:
            if not retry_invalid_episodes:
                raise RuntimeError(
                    f"Invalid LIBERO episode detected but retry_invalid_episodes=false: {invalid_reason}"
                )
            invalid_retries += 1
            invalid_record = {
                "target_trial": valid_trial_idx,
                "attempt": attempted_idx - 1,
                "reason": invalid_reason,
                "routing_summary": _summarize_routing_queries(
                    attempt_routing_trace
                ),
            }
            if save_query_metrics:
                invalid_record["queries"] = attempt_routing_trace
            results["invalid_episodes"].append(invalid_record)
            results["routing"]["invalid_attempts"].append(invalid_record)
            results["invalid_episode_count"] = invalid_retries
            logging.warning(
                "Discarding invalid LIBERO episode and retrying: task=%s trial=%s attempt=%s reason=%s invalid_retries=%s/%s",
                cfg.EVALUATION.task_id,
                valid_trial_idx,
                attempted_idx - 1,
                invalid_reason,
                invalid_retries,
                max_invalid_retries,
            )
            if invalid_retries > max_invalid_retries:
                raise RuntimeError(
                    "Exceeded max_invalid_episode_retries="
                    f"{max_invalid_retries} for task={cfg.EVALUATION.task_suite_name}/{cfg.EVALUATION.task_id}."
                )
            continue

        episode_queries: list[dict[str, Any]] = []
        for query in attempt_routing_trace:
            global_query_index = len(valid_query_metrics)
            query["attempt_index"] = int(attempted_idx - 1)
            query["global_query_index"] = int(global_query_index)
            query["timing_included"] = bool(
                global_query_index >= timing_warmup_queries
            )
            valid_query_metrics.append(query)
            episode_queries.append(query)
        episode_routing = {
            "episode_index": int(valid_trial_idx),
            "success": bool(success),
            "query_count": len(episode_queries),
            "total_actual_video_steps": int(
                sum(int(query["actual_video_steps"]) for query in episode_queries)
            ),
            "summary": _summarize_routing_queries(episode_queries),
        }
        if save_query_metrics:
            episode_routing["queries"] = episode_queries
        results["routing"]["episodes"].append(episode_routing)

        if success:
            results["successes"] += 1
            results["success_episodes"].append(valid_trial_idx)
        else:
            results["failure_episodes"].append(valid_trial_idx)
        if visualize_future_video:
            results["episode_future_video_psnr"].append(episode_mean_psnr)

        if save_videos:
            save_rollout_video(
                video_dir,
                replay_images,
                f"task{cfg.EVALUATION.task_id}_trial{valid_trial_idx}",
                success=success,
                task_description=task_description,
            )
        if visualize_future_video and save_videos:
            if len(predicted_future_video_clips) == 0:
                logging.warning(
                    "No predicted future frames collected for task %s trial %s.",
                    cfg.EVALUATION.task_id,
                    valid_trial_idx,
                )
            else:
                all_gt_frames = []
                all_pred_frames = []
                for clip in predicted_future_video_clips:
                    all_gt_frames.extend(clip["gt_frames"])
                    all_pred_frames.extend(clip["pred_frames"])
                    save_prediction_video(
                        predicted_video_dir,
                        clip["gt_frames"],
                        clip["pred_frames"],
                        f"task{cfg.EVALUATION.task_id}_trial{valid_trial_idx}",
                        clip["replan_idx"],
                        success=success,
                        task_description=task_description,
                    )
                save_prediction_video(
                    predicted_video_dir,
                    all_gt_frames,
                    all_pred_frames,
                    f"task{cfg.EVALUATION.task_id}_trial{valid_trial_idx}",
                    "all",
                    success=success,
                    task_description=task_description,
                )
        valid_trial_idx += 1

    if visualize_future_video:
        valid_episode_psnr = [x for x in results["episode_future_video_psnr"] if x is not None]
        if len(valid_episode_psnr) > 0:
            results["future_video_psnr_mean"] = float(np.mean(valid_episode_psnr))
    results["routing"]["summary"] = _summarize_routing_queries(
        valid_query_metrics
    )
    return results


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def eval_single_process(cfg: DictConfig):
    start_time = time.time()
    partial_state = PartialState()
    partial_state.config = cfg
    evaluation_git_identity = read_git_identity(project_root).as_dict()

    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")
    _validate_visualize_future_video_cfg(cfg)

    env_num = int(cfg.EVALUATION.get("env_num", 1))
    if env_num != 1:
        raise ValueError(
            "Only env_num=1 is supported in eval_libero_single.py. "
            "Use run_libero_manager/run_libero_parallel_test.sh for multi-GPU task parallelism."
        )

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    stage3_artifact_identity = _prepare_stage3_eval_artifacts(
        cfg,
        dataset_stats_path=dataset_stats_path,
    )
    model_device = _resolve_eval_device(cfg)
    router, prompt_context_provider, routing_runtime_identity = (
        _prepare_video_router(
            cfg,
            stage3_artifact_identity=stage3_artifact_identity,
            model_device=model_device,
        )
    )
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    if stage3_artifact_identity is not None:
        verify_aligned_runtime_asset(
            stage3_artifact_identity,
            "normalization_stats",
            phase="while normalization stats were being loaded",
        )
    logging.info("Using dataset stats: %s", dataset_stats_path)

    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    instantiate_kwargs: dict[str, Any] = {
        "model_dtype": model_dtype,
        "device": model_device,
    }
    if stage3_artifact_identity is not None:
        instantiate_kwargs["alignment_config"] = stage3_artifact_identity[
            "alignment_export"
        ]["export_metadata"]["alignment_config"]
    model = instantiate(cfg.model, **instantiate_kwargs)
    model_artifact_identity = _load_model_checkpoint(
        model,
        str(cfg.ckpt),
        stage3_artifact_identity=stage3_artifact_identity,
    )
    model = model.to(model_device).eval()

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    if action_horizon_cfg is None:
        action_horizon = int(cfg.data.train.num_frames) - 1
    else:
        action_horizon = int(action_horizon_cfg)
    if action_horizon <= 0:
        raise ValueError(f"EVALUATION.action_horizon must be positive, got {action_horizon}")

    video_size = cfg.data.train.get("video_size", [224, 224])
    if len(video_size) != 2:
        raise ValueError(f"data.train.video_size must be [H, W], got {video_size}")
    input_h = int(video_size[0])
    input_w = int(video_size[1])
    concat_multi_camera = cfg.data.train.get("concat_multi_camera", None)
    shape_meta_images = [meta["shape"] for meta in processor.shape_meta["images"]]

    local_log_dir = Path(cfg.EVALUATION.output_dir)
    local_log_dir.mkdir(parents=True, exist_ok=True)
    video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "videos"
    if bool(cfg.EVALUATION.get("save_videos", True)):
        video_dir.mkdir(parents=True, exist_ok=True)
    predicted_video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "predicted_videos"
    if bool(cfg.EVALUATION.get("visualize_future_video", False)) and bool(cfg.EVALUATION.get("save_videos", True)):
        predicted_video_dir.mkdir(parents=True, exist_ok=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.EVALUATION.task_suite_name]()
    task = task_suite.get_task(cfg.EVALUATION.task_id)
    initial_states = load_libero_task_init_states(
        task_suite,
        int(cfg.EVALUATION.task_id),
        init_states_root=benchmark.get_libero_path("init_states"),
    )

    while len(initial_states) < int(cfg.EVALUATION.num_trials):
        initial_states.extend(initial_states[: (int(cfg.EVALUATION.num_trials) - len(initial_states))])

    results = {
        "task_suite": cfg.EVALUATION.task_suite_name,
        "task_id": cfg.EVALUATION.task_id,
        "task_description": None,
        "successes": 0,
        "total_episodes": int(cfg.EVALUATION.num_trials),
        "gpu_id": int(cfg.gpu_id),
        "success_episodes": [],
        "failure_episodes": [],
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": 0,
    }

    logging.info("Running LIBERO evaluation with env_num=1")
    try:
        task_results = run_single_task(
            task=task,
            initial_states=initial_states,
            model=model,
            processor=processor,
            cfg=cfg,
            video_dir=video_dir,
            predicted_video_dir=predicted_video_dir,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            model_device=model_device,
            router=router,
            prompt_context_provider=prompt_context_provider,
        )
    finally:
        if prompt_context_provider is not None:
            prompt_context_provider.close()
    results.update(task_results)
    results["evaluation_git_identity"] = evaluation_git_identity
    results["routing_runtime_identity"] = routing_runtime_identity
    if model_artifact_identity is not None:
        results["model_artifact_identity"] = model_artifact_identity

    results["duration"] = time.time() - start_time
    output_dir = Path(cfg.EVALUATION.output_dir) / cfg.EVALUATION.task_suite_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"gpu{cfg.gpu_id}_task{cfg.EVALUATION.task_id}_results.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)

    print(
        f"Task {cfg.EVALUATION.task_id} completed: "
        f"{results['successes']}/{cfg.EVALUATION.num_trials} successes"
    )
    if results.get("future_video_psnr_mean") is not None:
        print(f"Task {cfg.EVALUATION.task_id} future-video PSNR mean: {results['future_video_psnr_mean']:.4f}")
    print(f"Time taken: {results['duration']:.2f} seconds")
    return results


if __name__ == "__main__":
    eval_single_process()
