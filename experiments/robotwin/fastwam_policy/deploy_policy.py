import json
import logging
import os
import sys
import time
import inspect
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.alignment.eval_loading import (
    inspect_aligned_model_artifacts,
    load_prepared_aligned_model,
    verify_aligned_runtime_asset,
)

logger = logging.getLogger(__name__)


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _parse_optional_int(value: Any) -> Optional[int]:
    if _is_none_like(value):
        return None
    return int(value)


def _parse_optional_float(value: Any) -> Optional[float]:
    if _is_none_like(value):
        return None
    return float(value)


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


def _resolve_sim_cfg_name(sim_cfg_path: Optional[str], sim_cfg_name: Optional[str]) -> str:
    configs_root = (PROJECT_ROOT / "configs").resolve()
    if not _is_none_like(sim_cfg_path):
        cfg_path = Path(str(sim_cfg_path)).expanduser().resolve()
        try:
            relative = cfg_path.relative_to(configs_root)
        except ValueError as exc:
            raise ValueError(
                f"`sim_cfg_path` must be under {configs_root}, got: {cfg_path}"
            ) from exc
        return relative.as_posix()

    if _is_none_like(sim_cfg_name):
        return "sim_robotwin.yaml"
    return str(sim_cfg_name)


def _compose_sim_cfg(
    sim_cfg_path: Optional[str],
    sim_cfg_name: Optional[str],
    sim_task: Optional[str],
) -> DictConfig:
    config_name = _resolve_sim_cfg_name(sim_cfg_path=sim_cfg_path, sim_cfg_name=sim_cfg_name)
    configs_root = (PROJECT_ROOT / "configs").resolve()
    overrides = []
    if not _is_none_like(sim_task):
        overrides.append(f"task={str(sim_task)}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def _resolve_dataset_stats_path(dataset_stats_path: Optional[str]) -> Path:
    if _is_none_like(dataset_stats_path):
        raise FileNotFoundError(
            "`dataset_stats_path` is required. "
            "Please pass it from eval entrypoint overrides."
        )
    resolved = Path(str(dataset_stats_path)).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset stats path not found: {resolved}")
    return resolved


def _optional_text(value: Any) -> Optional[str]:
    if _is_none_like(value):
        return None
    return str(value).strip()


def _first_text(*values: Any) -> Optional[str]:
    for value in values:
        text = _optional_text(value)
        if text is not None:
            return text
    return None


def _write_stage3_endpoint_identity(
    output_dir: str,
    *,
    model_artifact_identity: dict[str, Any],
    evaluation_runtime: dict[str, Any],
) -> Path:
    destination_dir = Path(output_dir).expanduser().resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / "stage3_endpoint_model_identity.json"
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    payload = {
        "schema_version": 1,
        "kind": "stage3_endpoint_model_identity",
        "model_artifact_identity": model_artifact_identity,
        "evaluation_runtime": evaluation_runtime,
    }
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _prepare_stage3_eval_artifacts(
    usr_args: Dict[str, Any],
    cfg: DictConfig,
    *,
    checkpoint_path: str,
    dataset_stats_path: Path,
) -> Optional[dict[str, Any]]:
    aligned_target = "fastwam.runtime.create_fastwam_unified_aligned"
    model_target = _optional_text(OmegaConf.select(cfg, "model._target_"))
    adapter_path = _first_text(
        usr_args.get("stage3_adapter_path"),
        cfg.EVALUATION.get("stage3_adapter_path"),
    )
    adapter_sha256 = _first_text(
        usr_args.get("stage3_adapter_sha256"),
        cfg.EVALUATION.get("stage3_adapter_sha256"),
    )
    if adapter_path is None:
        if model_target == aligned_target:
            raise ValueError(
                "Aligned RoboTwin endpoint eval requires stage3_adapter_path "
                "and stage3_adapter_sha256"
            )
        companion_fields = {
            name: _first_text(
                usr_args.get(name),
                cfg.EVALUATION.get(name),
            )
            for name in (
                "stage3_adapter_sha256",
                "stage3_base_sha256",
                "stage3_data_manifest_sha256",
                "stage3_training_contract_sha256",
                "stage3_global_step",
            )
        }
        companion_fields = {
            name: value
            for name, value in companion_fields.items()
            if value is not None
        }
        if companion_fields:
            raise ValueError(
                "Stage 3 identity fields require stage3_adapter_path: "
                f"{sorted(companion_fields)}"
            )
        return None
    if adapter_sha256 is None:
        raise ValueError(
            "stage3_adapter_sha256 is required when loading a Stage 3 Adapter"
        )

    if model_target != aligned_target:
        raise ValueError(
            "Stage 3 endpoint eval requires "
            "sim_task=robotwin_stage3_alignment_3cam384_1e-4"
        )
    base_sha256 = _first_text(
        usr_args.get("stage3_base_sha256"),
        cfg.EVALUATION.get("stage3_base_sha256"),
        OmegaConf.select(cfg, "base.expected_sha256"),
    )
    data_manifest_sha256 = _first_text(
        usr_args.get("stage3_data_manifest_sha256"),
        cfg.EVALUATION.get("stage3_data_manifest_sha256"),
        OmegaConf.select(cfg, "data_manifest.expected_sha256"),
    )
    if base_sha256 is None or data_manifest_sha256 is None:
        raise ValueError(
            "Stage 3 endpoint eval requires locked base and data-manifest SHA256 "
            "values from the selected Stage 3 task or explicit overrides"
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

    return inspect_aligned_model_artifacts(
        base_checkpoint_path=checkpoint_path,
        expected_base_checkpoint_sha256=base_sha256,
        alignment_export_path=adapter_path,
        expected_alignment_export_sha256=adapter_sha256,
        expected_data_manifest_sha256=data_manifest_sha256,
        expected_training_contract_sha256=_first_text(
            usr_args.get("stage3_training_contract_sha256"),
            cfg.EVALUATION.get("stage3_training_contract_sha256"),
        ),
        expected_global_step=_parse_optional_int(
            usr_args.get("stage3_global_step")
            if not _is_none_like(usr_args.get("stage3_global_step"))
            else cfg.EVALUATION.get("stage3_global_step")
        ),
        asset_paths={
            "vae": vae_path,
            "normalization_stats": dataset_stats_path,
        },
        expected_asset_sha256={
            "vae": vae_sha256,
            "normalization_stats": stats_sha256,
        },
    )


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    resized = pil_image.resize(size_wh, resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


class WorldActionRobotWinPolicy:
    def __init__(
        self,
        model_cfg: DictConfig,
        processor_cfg: DictConfig,
        checkpoint_path: str,
        dataset_stats_path: Path,
        device: str,
        model_dtype: torch.dtype,
        action_horizon: int,
        replan_steps: int,
        num_inference_steps: int,
        sigma_shift: Optional[float],
        seed: Optional[int],
        text_cfg_scale: float,
        negative_prompt: str,
        rand_device: str,
        tiled: bool,
        timing_enabled: bool,
        num_video_frames: int,
        inference_mode: str = "wo",
        stage3_artifact_identity: Optional[dict[str, Any]] = None,
    ) -> None:
        model_cfg_copy = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
        model_cfg_copy.load_text_encoder = True

        instantiate_kwargs: dict[str, Any] = {
            "model_dtype": model_dtype,
            "device": device,
        }
        if stage3_artifact_identity is not None:
            instantiate_kwargs["alignment_config"] = stage3_artifact_identity[
                "alignment_export"
            ]["export_metadata"]["alignment_config"]
        self.model = instantiate(model_cfg_copy, **instantiate_kwargs)
        if stage3_artifact_identity is None:
            self.model.load_checkpoint(checkpoint_path)
            self.model_artifact_identity = None
        else:
            self.model_artifact_identity = load_prepared_aligned_model(
                self.model,
                stage3_artifact_identity,
            )
            logger.info(
                "Loaded strict Stage 3 endpoint model: %s",
                json.dumps(self.model_artifact_identity, sort_keys=True),
            )
        self.model = self.model.to(device).eval()

        self.processor: FastWAMProcessor = instantiate(processor_cfg).eval()
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
        self.processor.set_normalizer_from_stats(dataset_stats)
        if stage3_artifact_identity is not None:
            verify_aligned_runtime_asset(
                stage3_artifact_identity,
                "normalization_stats",
                phase="while normalization stats were being loaded",
            )

        self.action_horizon = int(action_horizon)
        self.replan_steps = int(max(1, min(replan_steps, action_horizon)))
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)
        self.timing_enabled = bool(timing_enabled)
        self._num_video_frames = int(num_video_frames)
        self.inference_mode = str(inference_mode).lower()

        self.pending_actions: deque[np.ndarray] = deque()
        self.episode_count = 0
        self.step_count = 0
        self._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}

        logger.info(
            "Initialized WorldActionRobotWinPolicy | ckpt=%s | stats=%s | horizon=%d | replan=%d | inference_mode=%s",
            checkpoint_path,
            dataset_stats_path,
            self.action_horizon,
            self.replan_steps,
            self.inference_mode,
        )

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one merged state key in shape_meta['state'].")
        state_key = state_meta[0]["key"]

        state_batch = {"state": {state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action tensor [B,T,D], got {tuple(action.shape)}")

        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected exactly one merged action key in shape_meta['action'].")

        action_key = action_meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        denorm = normalizer.backward(action.to(dtype=torch.float32, device="cpu"))
        return denorm.numpy()

    def _build_robotwin_image_tensor(self, observation: Dict[str, Any]) -> torch.Tensor:
        obs_data = observation["observation"]
        head = _resize_rgb(obs_data["head_camera"]["rgb"], (320, 256))
        left = _resize_rgb(obs_data["left_camera"]["rgb"], (160, 128))
        right = _resize_rgb(obs_data["right_camera"]["rgb"], (160, 128))
        bottom = np.concatenate([left, right], axis=1)
        image = np.concatenate([head, bottom], axis=0)  # [384, 320, 3]

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(
            device=self.model.device,
            dtype=self.model.torch_dtype,
        )
        image_tensor = image_tensor * (2.0 / 255.0) - 1.0
        return image_tensor

    def _infer_action_chunk(self, observation: Dict[str, Any], instruction: str) -> np.ndarray:
        image_tensor = self._build_robotwin_image_tensor(observation)
        state_vector = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        proprio = self._normalize_state(state_vector)

        prompt = DEFAULT_PROMPT.format(task=instruction)
        infer_kwargs = {
            "prompt": prompt,
            "input_image": image_tensor,
            "action_horizon": self.action_horizon,
            "proprio": proprio,
            "negative_prompt": self.negative_prompt,
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": self.seed,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
        }
        if self.inference_mode == "w" or "num_video_frames" in inspect.signature(self.model.infer_action).parameters:
            infer_kwargs["num_video_frames"] = int(self._num_video_frames)
        infer_t0 = time.perf_counter() if self.timing_enabled else 0.0
        with torch.no_grad():
            if hasattr(self.model, "infer_action_mode"):
                pred = self.model.infer_action_mode(
                    **infer_kwargs,
                    inference_mode=self.inference_mode,
                )
            else:
                if self.inference_mode != "wo":
                    raise ValueError(
                        f"Model {type(self.model).__name__} does not support inference_mode={self.inference_mode}"
                    )
                pred = self.model.infer_action(**infer_kwargs)
        if self.timing_enabled:
            self._timing_rollout["infer_s"] += time.perf_counter() - infer_t0

        action_tensor = pred["action"]  # [T, D]
        action_chunk = self._denormalize_action(action_tensor)[0]  # [T, D]
        return action_chunk

    def _fill_action_queue(self, observation: Dict[str, Any], instruction: str) -> None:
        action_chunk = self._infer_action_chunk(observation=observation, instruction=instruction)
        n_exec = min(self.replan_steps, action_chunk.shape[0])
        for i in range(n_exec):
            self.pending_actions.append(np.asarray(action_chunk[i], dtype=np.float32))

    def should_request_observation(self) -> bool:
        return not self.pending_actions

    def step(self, task_env, observation: Optional[Dict[str, Any]]) -> None:
        if not self.pending_actions:
            if observation is None:
                raise ValueError(
                    "Observation is required when action queue is empty "
                    "(replan step for fastwam)."
                )
            instruction = task_env.get_instruction()
            self._fill_action_queue(observation=observation, instruction=instruction)

        if not self.pending_actions:
            logger.warning("No action generated; skip current eval step.")
            return

        action = self.pending_actions.popleft()
        sim_t0 = time.perf_counter() if self.timing_enabled else 0.0
        task_env.take_action(action, action_type="qpos")
        if self.timing_enabled:
            self._timing_rollout["sim_s"] += time.perf_counter() - sim_t0
        self.step_count += 1

    def reset_timing_rollout(self) -> None:
        self._timing_rollout["infer_s"] = 0.0
        self._timing_rollout["sim_s"] = 0.0

    def get_timing_rollout(self) -> Dict[str, float]:
        return {
            "infer_s": float(self._timing_rollout["infer_s"]),
            "sim_s": float(self._timing_rollout["sim_s"]),
        }

    def reset(self) -> None:
        self.pending_actions.clear()
        self.episode_count += 1
        self.step_count = 0
        self.reset_timing_rollout()


def encode_obs(observation: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return observation


def get_model(usr_args: Dict[str, Any]):
    sim_cfg_path = usr_args.get("sim_cfg_path")
    sim_cfg_name = usr_args.get("sim_cfg_name")
    sim_task = usr_args.get("sim_task")
    cfg = _compose_sim_cfg(
        sim_cfg_path=sim_cfg_path,
        sim_cfg_name=sim_cfg_name,
        sim_task=sim_task,
    )

    checkpoint_path = usr_args.get("ckpt_setting")
    if _is_none_like(checkpoint_path):
        raise ValueError("`ckpt_setting` is required and must be a valid checkpoint path.")

    device = str(usr_args.get("device") or cfg.EVALUATION.get("device") or "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; fallback device to cpu.")
        device = "cpu"

    mixed_precision = str(usr_args.get("mixed_precision") or cfg.get("mixed_precision", "bf16"))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    dataset_stats_path = _resolve_dataset_stats_path(
        dataset_stats_path=usr_args.get("dataset_stats_path"),
    )
    stage3_artifact_identity = _prepare_stage3_eval_artifacts(
        usr_args,
        cfg,
        checkpoint_path=str(checkpoint_path),
        dataset_stats_path=dataset_stats_path,
    )

    action_horizon = _parse_optional_int(usr_args.get("action_horizon"))
    if action_horizon is None:
        eval_horizon = _parse_optional_int(cfg.EVALUATION.get("action_horizon"))
        action_horizon = eval_horizon if eval_horizon is not None else int(cfg.data.train.num_frames) - 1
    if action_horizon <= 0:
        raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")

    replan_steps = _parse_optional_int(usr_args.get("replan_steps"))
    if replan_steps is None:
        replan_steps = int(cfg.EVALUATION.get("replan_steps", 8))

    num_inference_steps = _parse_optional_int(usr_args.get("num_inference_steps"))
    if num_inference_steps is None:
        num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps", cfg.eval_num_inference_steps))

    sigma_shift = _parse_optional_float(usr_args.get("sigma_shift"))
    if sigma_shift is None:
        sigma_shift = _parse_optional_float(cfg.EVALUATION.get("sigma_shift"))

    seed = _parse_optional_int(usr_args.get("seed"))
    text_cfg_scale = float(usr_args.get("text_cfg_scale", cfg.EVALUATION.get("text_cfg_scale", 1.0)))
    negative_prompt = str(usr_args.get("negative_prompt", cfg.EVALUATION.get("negative_prompt", "")))
    rand_device = str(usr_args.get("rand_device", cfg.EVALUATION.get("rand_device", "cpu")))
    tiled = _parse_bool(usr_args.get("tiled", cfg.EVALUATION.get("tiled", False)))
    inference_mode = str(usr_args.get("inference_mode", cfg.EVALUATION.get("inference_mode", "wo"))).lower()
    timing_enabled = _parse_bool(
        usr_args.get("timing_enabled", cfg.EVALUATION.get("timing_enabled", False))
    )

    policy = WorldActionRobotWinPolicy(
        model_cfg=cfg.model,
        processor_cfg=cfg.data.train.processor,
        checkpoint_path=str(checkpoint_path),
        dataset_stats_path=dataset_stats_path,
        device=device,
        model_dtype=model_dtype,
        action_horizon=action_horizon,
        replan_steps=replan_steps,
        num_inference_steps=num_inference_steps,
        sigma_shift=sigma_shift,
        seed=seed,
        text_cfg_scale=text_cfg_scale,
        negative_prompt=negative_prompt,
        rand_device=rand_device,
        tiled=tiled,
        timing_enabled=timing_enabled,
        num_video_frames=(int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1,
        inference_mode=inference_mode,
        stage3_artifact_identity=stage3_artifact_identity,
    )
    if policy.model_artifact_identity is not None:
        eval_output_dir = _first_text(usr_args.get("eval_output_dir"))
        if eval_output_dir is None:
            raise ValueError(
                "Stage 3 RoboTwin endpoint eval requires eval_output_dir "
                "for the model identity receipt"
            )
        receipt_path = _write_stage3_endpoint_identity(
            eval_output_dir,
            model_artifact_identity=policy.model_artifact_identity,
            evaluation_runtime={
                "task_name": _first_text(usr_args.get("task_name")),
                "task_config": _first_text(usr_args.get("task_config")),
                "instruction_type": _first_text(
                    usr_args.get("instruction_type")
                ),
                "inference_mode": inference_mode,
                "action_horizon": action_horizon,
                "replan_steps": replan_steps,
                "num_inference_steps": num_inference_steps,
                "sigma_shift": sigma_shift,
                "seed": seed,
            },
        )
        logger.info(
            "Wrote Stage 3 endpoint model identity receipt: %s",
            receipt_path,
        )
    return policy


def eval(TASK_ENV, model, observation: Optional[Dict[str, Any]]):
    obs = encode_obs(observation)
    model.step(TASK_ENV, obs)


def reset_model(model):
    model.reset()
