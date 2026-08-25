"""Accelerate/ZeRO runtime that owns only the Stage 3 Adapter."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from math import ceil
from pathlib import Path
import shutil
import time
from typing import Any, Mapping

from accelerate import Accelerator
from accelerate.utils import broadcast_object_list, gather_object
from omegaconf import OmegaConf
import torch
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)
from torch.utils.data import DataLoader

from fastwam.models.wan22.video_action_alignment import (
    save_alignment_checkpoint,
)
from fastwam.utils.logging_config import get_logger
from fastwam.utils.pytorch_utils import set_global_seed
from fastwam.utils.samplers import ResumableEpochSampler

from .checkpointing import (
    BaseCheckpointIdentity,
    GitIdentity,
    TRAINING_STATE_KIND,
    TRAINING_STATE_SCHEMA_VERSION,
    canonical_json_sha256,
    hash_state_tree,
    sha256_file,
    validate_training_state,
    write_json_atomic,
    write_text_atomic,
)
from .losses import Stage3LossOutput, stage3_alignment_loss
from .rollout import (
    STAGE3_SOLVER_STEPS,
    complete_stage3_velocity_panel,
    compute_stage3_frozen_panel,
    prepare_stage3_batch,
)
from .trainer import AlignmentVelocityModule


logger = get_logger(__name__)


def _require_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must contain exactly 64 lowercase hex chars")
    return value


def _plain_config(config: Any) -> dict[str, Any]:
    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=True)
    if not isinstance(config, dict):
        raise TypeError(f"Stage 3 config must be dict-like, got {type(config)}")
    return config


def _zero_stage(accelerator: Accelerator) -> int:
    plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    if plugin is None:
        return 0
    value = plugin.deepspeed_config.get("zero_optimization", {}).get("stage", 0)
    return int(value)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _scheduler_last_epoch(scheduler: Any) -> int:
    current = scheduler
    while hasattr(current, "scheduler"):
        current = current.scheduler
    value = getattr(current, "last_epoch", None)
    if value is None:
        raise RuntimeError("Stage 3 scheduler has no last_epoch state")
    return int(value)


class Stage3AlignmentTrainer:
    """Train and checkpoint only Adapter while the 5B base stays local/frozen."""

    def __init__(
        self,
        *,
        accelerator: Accelerator,
        model: torch.nn.Module,
        train_dataset,
        config: Mapping[str, Any],
        base_identity: BaseCheckpointIdentity,
        git_identity: GitIdentity,
        asset_identities: Mapping[str, Mapping[str, Any]] | None = None,
        data_identity: Mapping[str, Any] | None = None,
    ):
        self.accelerator = accelerator
        self.model = model
        self.train_dataset = train_dataset
        self.config = _plain_config(config)
        self.base_identity = base_identity
        self.git_identity = git_identity
        self.asset_identities = {
            str(name): dict(identity)
            for name, identity in (asset_identities or {}).items()
        }
        self.data_identity = dict(data_identity or {})
        self.data_manifest_sha256 = _require_sha256(
            self.data_identity.get("sha256"),
            field="data_identity.sha256",
        )
        self.training_cfg = dict(self.config["training"])
        self.stage3_cfg = dict(self.config["stage3"])
        self.checkpoint_cfg = dict(self.config["checkpoint"])
        self.output_dir = Path(self.config["output_dir"]).expanduser().resolve()

        self.batch_size = int(self.training_cfg["batch_size"])
        self.num_workers = int(self.training_cfg["num_workers"])
        self.drop_last = bool(self.training_cfg.get("drop_last", True))
        self.gradient_accumulation_steps = int(
            self.training_cfg["gradient_accumulation_steps"]
        )
        self.learning_rate = float(self.training_cfg["learning_rate"])
        self.weight_decay = float(self.training_cfg["weight_decay"])
        self.max_grad_norm = float(self.training_cfg["max_grad_norm"])
        self.num_epochs = int(self.training_cfg["num_epochs"])
        self.seed = int(self.training_cfg["seed"])
        self.log_every = int(self.config["runtime"]["log_every"])
        self.save_every = int(self.checkpoint_cfg["save_every"])
        self.keep_last = int(self.checkpoint_cfg["keep_last"])
        self.save_final = bool(self.checkpoint_cfg["save_final"])
        self.strict_resume = bool(self.checkpoint_cfg.get("strict_resume", True))
        self.num_solver_steps = int(self.stage3_cfg["num_solver_steps"])
        self.sigma_shift = self.stage3_cfg.get("sigma_shift")
        if self.num_solver_steps != STAGE3_SOLVER_STEPS:
            raise ValueError(
                f"Stage 3 requires exactly {STAGE3_SOLVER_STEPS} solver steps"
            )
        if self.batch_size <= 0 or self.num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers non-negative")
        if self.num_workers != 0:
            raise ValueError(
                "formal Stage 3 v1 requires training.num_workers=0 so worker "
                "RNG is covered by strict resume"
            )
        if not self.drop_last:
            raise ValueError(
                "formal Stage 3 v1 requires training.drop_last=true so "
                "distributed tail batches replay exactly after resume"
            )
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.gradient_accumulation_steps != int(
            self.accelerator.gradient_accumulation_steps
        ):
            raise ValueError(
                "Accelerator/config gradient accumulation mismatch: "
                f"{self.accelerator.gradient_accumulation_steps} vs "
                f"{self.gradient_accumulation_steps}"
            )
        if bool(self.accelerator.step_scheduler_with_optimizer):
            raise ValueError(
                "Stage 3 Accelerator must use "
                "step_scheduler_with_optimizer=False"
            )
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer configuration")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.keep_last < 1:
            raise ValueError("checkpoint.keep_last must be at least one")
        if not self.strict_resume:
            raise ValueError("formal Stage 3 supports strict resume only")
        if _zero_stage(self.accelerator) > 2:
            raise ValueError("Stage 3 supports DeepSpeed ZeRO stages 0, 1, or 2 only")
        if str(self.stage3_cfg.get("k_sampling", "uniform")) != "uniform":
            raise ValueError("Stage 3 currently supports uniform k sampling only")
        if self.num_epochs <= 0:
            raise ValueError("training.num_epochs must be positive")

        deepspeed_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        if deepspeed_plugin is not None:
            clipping = deepspeed_plugin.deepspeed_config.get("gradient_clipping")
            if clipping is None or str(clipping).lower() == "auto":
                raise ValueError(
                    "DeepSpeed Stage 3 config must set numeric "
                    "gradient_clipping"
                )
            if float(clipping) != self.max_grad_norm:
                raise ValueError(
                    "DeepSpeed gradient_clipping must equal "
                    "training.max_grad_norm"
                )

        def initialize_local_components() -> None:
            trainable_names = self.model.configure_alignment_training()
            if not trainable_names or not all(
                name.startswith("alignment_adapter.")
                for name in trainable_names
            ):
                raise RuntimeError(
                    "Stage 3 trainable set must be alignment_adapter.*"
                )
            # Keep FP32 optimizer master weights; autocast still runs Adapter
            # math in the requested BF16/FP16 compute dtype.
            self.model.alignment_adapter.to(dtype=torch.float32)
            self.model.zero_grad(set_to_none=True)
            self.adapter_module = AlignmentVelocityModule(
                self.model.alignment_adapter
            )
            adapter_params = tuple(self.adapter_module.parameters())
            if not adapter_params:
                raise RuntimeError("Stage 3 Adapter has no parameters")

            betas = tuple(float(value) for value in self.training_cfg["betas"])
            if len(betas) != 2:
                raise ValueError("training.betas must contain two values")
            self.optimizer = torch.optim.AdamW(
                adapter_params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=betas,
            )
            if {
                id(parameter)
                for group in self.optimizer.param_groups
                for parameter in group["params"]
            } != {id(parameter) for parameter in adapter_params}:
                raise RuntimeError(
                    "optimizer parameters must equal Adapter parameters"
                )

            worker_init_fn = set_global_seed(
                self.seed,
                get_worker_init_fn=True,
            )
            self.train_sampler = ResumableEpochSampler(
                dataset=self.train_dataset,
                seed=self.seed,
                batch_size=self.batch_size,
                num_processes=self.accelerator.num_processes,
            )
            self.loader_generator = torch.Generator(device="cpu").manual_seed(
                self.seed + int(self.accelerator.process_index)
            )
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                sampler=self.train_sampler,
                num_workers=self.num_workers,
                pin_memory=self.accelerator.device.type == "cuda",
                worker_init_fn=worker_init_fn,
                drop_last=self.drop_last,
                generator=self.loader_generator,
            )
            global_batch_size = self.batch_size * int(
                self.accelerator.num_processes
            )
            self.micro_batches_per_epoch = (
                len(self.train_dataset) // global_batch_size
            )

        self._run_all_rank_phase(
            initialize_local_components,
            phase="trainer local preflight",
        )
        self._assert_dataset_length_consistent()
        if self.micro_batches_per_epoch <= 0:
            raise ValueError(
                "drop_last would yield no batches; reduce batch size or "
                "world size"
            )
        if (
            self.micro_batches_per_epoch
            % self.gradient_accumulation_steps
            != 0
        ):
            raise ValueError(
                "formal Stage 3 requires complete gradient-accumulation "
                "groups in every epoch"
            )

        def initialize_scheduler() -> None:
            self.max_steps = self._resolve_max_steps()
            self.scheduler = self._build_scheduler()

        self._run_all_rank_phase(
            initialize_scheduler,
            phase="scheduler preflight",
        )

        (
            self.adapter_module,
            self.optimizer,
            self.train_loader,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.adapter_module,
            self.optimizer,
            self.train_loader,
            self.scheduler,
        )

        def initialize_prepared_state() -> None:
            self._validate_prepared_contract()
            if len(self.accelerator._models) != 1:
                raise RuntimeError(
                    "Stage 3 Accelerator must own exactly one Adapter-only "
                    "module"
                )
            self.optimizer.zero_grad(set_to_none=True)

            self.global_step = 0
            self.epoch = 0
            self.batch_in_epoch = 0
            self.exports_dir = self.output_dir / "checkpoints" / "exports"
            self.states_dir = self.output_dir / "checkpoints" / "states"
            self.contract_payload = self._build_contract_payload()
            self.training_contract_sha256 = canonical_json_sha256(
                self.contract_payload
            )
            self.runtime_versions = {
                "torch": torch.__version__,
                "accelerate": _package_version("accelerate"),
                "deepspeed": _package_version("deepspeed"),
                "torchcodec": _package_version("torchcodec"),
                "torchvision": _package_version("torchvision"),
                "av": _package_version("av"),
                "datasets": _package_version("datasets"),
                "pyarrow": _package_version("pyarrow"),
            }
            self.dataloader_contract = {
                "split_batches": bool(self.accelerator.split_batches),
                "even_batches": bool(self.accelerator.even_batches),
                "use_stateful_dataloader": bool(
                    getattr(
                        self.train_loader,
                        "use_stateful_dataloader",
                        False,
                    )
                ),
            }
            if self.dataloader_contract["split_batches"]:
                raise RuntimeError(
                    "Stage 3 does not support split_batches=true"
                )
            self.resume_contract = {
                "base_checkpoint_sha256": self.base_identity.sha256,
                "training_contract_sha256": self.training_contract_sha256,
                "git_commit": self.git_identity.commit,
                "world_size": int(self.accelerator.num_processes),
                "distributed_type": str(self.accelerator.distributed_type),
                "zero_stage": _zero_stage(self.accelerator),
                "mixed_precision": str(self.accelerator.mixed_precision),
                "device_type": self.accelerator.device.type,
                "gradient_accumulation_steps": (
                    self.gradient_accumulation_steps
                ),
                "batch_size_per_rank": self.batch_size,
                "dataset_length": len(self.train_dataset),
                "micro_batches_per_epoch": self.micro_batches_per_epoch,
                "drop_last": self.drop_last,
                "deepspeed_config_sha256": self.deepspeed_config_sha256,
                "asset_sha256": {
                    name: identity["sha256"]
                    for name, identity in sorted(
                        self.asset_identities.items()
                    )
                },
                "data_manifest_sha256": self.data_manifest_sha256,
                "git_tracked_dirty": self.git_identity.tracked_dirty,
                "git_untracked_source_files": list(
                    self.git_identity.untracked_source_files
                ),
                "versions": self.runtime_versions,
                "dataloader_contract": self.dataloader_contract,
            }

        self._run_all_rank_phase(
            initialize_prepared_state,
            phase="prepared trainer state validation",
        )

        def create_checkpoint_directories() -> None:
            self.exports_dir.mkdir(parents=True, exist_ok=True)
            self.states_dir.mkdir(parents=True, exist_ok=True)

        self._run_main_phase(
            create_checkpoint_directories,
            phase="directory setup",
        )
        resume = self.checkpoint_cfg.get("resume")
        if resume:
            self.resume_strict(resume)

    def _resolve_max_steps(self) -> int:
        configured = self.training_cfg.get("max_steps")
        if configured is not None:
            configured = int(configured)
            if configured <= 0:
                raise ValueError("training.max_steps must be positive or null")
            return configured
        optimizer_steps = max(
            ceil(
                self.micro_batches_per_epoch
                / self.gradient_accumulation_steps
            ),
            1,
        )
        return max(optimizer_steps * self.num_epochs, 1)

    def _assert_dataset_length_consistent(self) -> None:
        local = torch.tensor(
            [len(self.train_dataset)],
            device=self.accelerator.device,
            dtype=torch.int64,
        )
        gathered = self.accelerator.gather(local).reshape(-1)
        if not torch.all(gathered == gathered[0]):
            raise RuntimeError(
                "Stage 3 dataset length differs across ranks: "
                f"{gathered.cpu().tolist()}"
            )

    def _validate_prepared_contract(self) -> None:
        effective_accumulation = int(
            self.accelerator.gradient_accumulation_steps
        )
        if effective_accumulation != self.gradient_accumulation_steps:
            raise RuntimeError(
                "Accelerate changed gradient accumulation during prepare: "
                f"configured={self.gradient_accumulation_steps}, "
                f"effective={effective_accumulation}"
            )

        plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        if plugin is None:
            self.effective_deepspeed_config = None
            self.deepspeed_config_sha256 = None
            return

        config = plugin.deepspeed_config
        offload_optimizer = config.get("zero_optimization", {}).get(
            "offload_optimizer"
        )
        if offload_optimizer is not None:
            if not isinstance(offload_optimizer, Mapping):
                raise RuntimeError(
                    "DeepSpeed offload_optimizer must be a mapping"
                )
            if str(offload_optimizer.get("device", "none")).lower() != "none":
                raise RuntimeError(
                    "formal Stage 3 does not support optimizer offload"
                )
        self.effective_deepspeed_config = config
        self.deepspeed_config_sha256 = canonical_json_sha256(config)
        engine_accumulation = getattr(
            self.adapter_module,
            "gradient_accumulation_steps",
            None,
        )
        if callable(engine_accumulation):
            engine_accumulation = engine_accumulation()
        if int(engine_accumulation) != self.gradient_accumulation_steps:
            raise RuntimeError(
                "DeepSpeed engine gradient accumulation does not match Stage 3"
            )
        engine_micro_batch = getattr(
            self.adapter_module,
            "train_micro_batch_size_per_gpu",
            None,
        )
        if callable(engine_micro_batch):
            engine_micro_batch = engine_micro_batch()
        if int(engine_micro_batch) != self.batch_size:
            raise RuntimeError(
                "DeepSpeed engine micro batch does not match training.batch_size"
            )
        expected_global_batch = (
            self.batch_size
            * self.gradient_accumulation_steps
            * int(self.accelerator.num_processes)
        )
        if int(config.get("train_batch_size", -1)) != expected_global_batch:
            raise RuntimeError(
                "DeepSpeed effective train_batch_size does not match Stage 3"
            )
        if float(config.get("gradient_clipping", -1.0)) != self.max_grad_norm:
            raise RuntimeError(
                "DeepSpeed effective gradient clipping does not match Stage 3"
            )

    def _build_scheduler(self):
        warmup_ratio = float(self.training_cfg["warmup_ratio"])
        if not 0 <= warmup_ratio < 1:
            raise ValueError("training.warmup_ratio must be in [0,1)")
        warmup_steps = min(
            int(self.max_steps * warmup_ratio),
            max(self.max_steps - 1, 0),
        )
        remaining = max(self.max_steps - warmup_steps, 1)
        scheduler_type = str(self.training_cfg["lr_scheduler_type"]).lower()
        if scheduler_type == "cosine":
            main = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == "constant":
            main = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining)
        else:
            raise ValueError("lr_scheduler_type must be cosine or constant")
        if warmup_steps == 0:
            return main
        warmup = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup, main],
            milestones=[warmup_steps],
        )

    def _build_contract_payload(self) -> dict[str, Any]:
        return {
            "base_checkpoint_sha256": self.base_identity.sha256,
            "asset_identities": self.asset_identities,
            "data_identity": self.data_identity,
            "model": self.config["model"],
            "data": self.config["data"],
            "stage3": self.stage3_cfg,
            "training": {
                **self.training_cfg,
                "max_steps": self.max_steps,
            },
            "alignment_config": self.model.alignment_adapter.config(),
            "dataset_length": len(self.train_dataset),
            "micro_batches_per_epoch": self.micro_batches_per_epoch,
            "dataset_config_sha256": canonical_json_sha256(self.config["data"]),
            "effective_deepspeed_config": self.effective_deepspeed_config,
        }

    def _set_modes(self) -> None:
        self.model.eval()
        self.adapter_module.train()

    def build_loss(
        self,
        sample: Mapping[str, torch.Tensor],
        *,
        k: int | None = None,
    ) -> Stage3LossOutput:
        self._set_modes()
        if k is None:
            k = int(torch.randint(self.num_solver_steps, (1,)).item())
        with self.accelerator.autocast():
            prepared = prepare_stage3_batch(
                self.model,
                sample,
                k=k,
                num_solver_steps=self.num_solver_steps,
                sigma_shift=self.sigma_shift,
            )
            frozen = compute_stage3_frozen_panel(self.model, prepared)
            v_self = self.adapter_module(
                frozen.self_base_velocity,
                action_tokens=frozen.self_action_tokens,
                video_tokens=frozen.self_video_tokens,
                video_meta=frozen.self_video_meta,
            )
            panel = complete_stage3_velocity_panel(frozen, v_self)
            return stage3_alignment_loss(
                panel.v0,
                panel.v_gt,
                panel.v_self,
                panel.v_target,
                panel.action_is_pad,
                action_weight=panel.action_weight,
                helpful_relative_margin=float(
                    self.stage3_cfg["helpful_relative_margin"]
                ),
                lambda_action=float(self.stage3_cfg["lambda_action"]),
                lambda_align=float(self.stage3_cfg["lambda_align"]),
                lambda_safe=float(self.stage3_cfg["lambda_safe"]),
            )

    def _assert_no_base_gradients(self) -> None:
        leaked = [
            name
            for name, parameter in self.model.named_parameters()
            if not name.startswith("alignment_adapter.") and parameter.grad is not None
        ]
        if leaked:
            raise RuntimeError(
                "non-Adapter gradients detected: " + ", ".join(sorted(leaked))
            )

    def _adapter_export_state(self) -> dict[str, torch.Tensor]:
        wrapped_state = self.accelerator.get_state_dict(self.adapter_module)
        prefix = "adapter."
        if not wrapped_state or not all(name.startswith(prefix) for name in wrapped_state):
            raise RuntimeError("wrapped Adapter state_dict has unexpected keys")
        state = {
            name[len(prefix) :]: value
            for name, value in wrapped_state.items()
        }
        expected = set(
            self.accelerator.unwrap_model(self.adapter_module).adapter.state_dict()
        )
        if set(state) != expected:
            raise RuntimeError("wrapped Adapter state_dict is incomplete")
        return state

    def _manifest(self) -> dict[str, Any]:
        scheduler_last_epoch = _scheduler_last_epoch(self.scheduler)
        if scheduler_last_epoch != self.global_step:
            raise RuntimeError(
                "scheduler last_epoch must equal successful global_step"
            )
        epoch, batch_in_epoch = self._checkpoint_cursor()
        return {
            "schema_version": TRAINING_STATE_SCHEMA_VERSION,
            "kind": TRAINING_STATE_KIND,
            "complete": True,
            "global_step": int(self.global_step),
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
            "micro_step_in_accumulation": 0,
            "scheduler_last_epoch": scheduler_last_epoch,
            "base_checkpoint": self.base_identity.path,
            "base_checkpoint_sha256": self.base_identity.sha256,
            "base_checkpoint_size_bytes": self.base_identity.size_bytes,
            "alignment_config": self.model.alignment_adapter.config(),
            "training_contract": self.contract_payload,
            **self.resume_contract,
            "files": {},
        }

    def _checkpoint_cursor(self) -> tuple[int, int]:
        if not 0 <= self.batch_in_epoch <= self.micro_batches_per_epoch:
            raise RuntimeError(
                "batch_in_epoch is outside the current epoch: "
                f"{self.batch_in_epoch} not in "
                f"[0, {self.micro_batches_per_epoch}]"
            )
        if self.batch_in_epoch == self.micro_batches_per_epoch:
            return int(self.epoch + 1), 0
        return int(self.epoch), int(self.batch_in_epoch)

    def _run_main_phase(self, operation, *, phase: str) -> None:
        status: list[dict[str, Any] | None] = [None]
        if self.accelerator.is_main_process:
            try:
                operation()
                status[0] = {"ok": True}
            except Exception as error:
                status[0] = {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
        broadcast_object_list(status, from_process=0)
        result = status[0]
        if not isinstance(result, dict) or result.get("ok") is not True:
            if isinstance(result, dict):
                error_type = result.get("error_type", "UnknownError")
                message = result.get("message", "missing error message")
            else:
                error_type = type(result).__name__
                message = "invalid main-process status payload"
            raise RuntimeError(
                f"Stage 3 checkpoint {phase} failed on main process: "
                f"{error_type}: {message}"
            )

    def _run_all_rank_phase(self, operation, *, phase: str):
        value = None
        try:
            value = operation()
            local = {"ok": True}
        except Exception as error:
            local = {
                "ok": False,
                "error_type": type(error).__name__,
                "message": str(error),
            }
        gathered = gather_object([local])
        if (
            not isinstance(gathered, list)
            or len(gathered) != self.accelerator.num_processes
        ):
            raise RuntimeError(
                f"Stage 3 {phase} status gather returned invalid data"
            )
        failures = [
            item
            for item in gathered
            if not isinstance(item, dict) or item.get("ok") is not True
        ]
        if failures:
            failure = failures[0]
            if not isinstance(failure, dict):
                failure = {
                    "error_type": "InvalidStatus",
                    "message": repr(failure),
                }
            raise RuntimeError(
                f"Stage 3 {phase} failed on at least one rank: "
                f"{failure.get('error_type')}: {failure.get('message')}"
            )
        return value

    def _require_all_rank_identical(
        self,
        value: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        """Reject rank-local identities that are valid but not identical."""

        gathered = gather_object([value])
        if (
            not isinstance(gathered, list)
            or len(gathered) != self.accelerator.num_processes
            or any(not isinstance(item, dict) for item in gathered)
        ):
            raise RuntimeError(
                f"Stage 3 {phase} identity gather returned invalid data"
            )
        reference = gathered[0]
        if any(item != reference for item in gathered[1:]):
            fingerprints = [
                canonical_json_sha256(item)
                for item in gathered
            ]
            raise RuntimeError(
                f"Stage 3 {phase} identity differs across ranks: "
                f"{fingerprints}"
            )
        return reference

    def save_checkpoint(self) -> Path:
        if not self.accelerator.sync_gradients:
            raise RuntimeError("checkpoint must be saved on an optimizer boundary")
        tag = f"step_{self.global_step:06d}"
        target = self.states_dir / tag
        temporary = self.states_dir / f".incomplete-{tag}"

        def prepare_directory() -> None:
            if target.exists():
                raise FileExistsError(f"checkpoint already exists: {target}")
            if temporary.exists():
                shutil.rmtree(temporary)
            (temporary / "accelerator").mkdir(parents=True)

        self._run_main_phase(prepare_directory, phase="setup")

        self.accelerator.save_state(
            output_dir=str(temporary / "accelerator"),
            safe_serialization=True,
            exclude_frozen_parameters=True,
        )
        adapter_state = self._adapter_export_state()
        self.accelerator.wait_for_everyone()

        def finalize_checkpoint() -> None:
            export_in_state = temporary / "adapter_export.pt"
            adapter = self.accelerator.unwrap_model(self.adapter_module).adapter
            save_alignment_checkpoint(
                export_in_state,
                adapter,
                base_checkpoint=self.base_identity.path,
                data_manifest_sha256=self.data_manifest_sha256,
                base_checkpoint_sha256=self.base_identity.sha256,
                global_step=self.global_step,
                adapter_state_dict=adapter_state,
                git_commit=self.git_identity.commit,
                training_contract_sha256=self.training_contract_sha256,
                asset_identities=self.asset_identities,
            )
            manifest = self._manifest()
            manifest["files"] = hash_state_tree(temporary)
            manifest_path = write_json_atomic(
                temporary / "manifest.json",
                manifest,
            )
            write_json_atomic(
                temporary / "COMPLETE",
                {
                    "schema_version": TRAINING_STATE_SCHEMA_VERSION,
                    "kind": TRAINING_STATE_KIND,
                    "manifest_sha256": sha256_file(manifest_path),
                },
            )
            temporary.replace(target)

            export_path = self.exports_dir / f"{tag}.pt"
            temporary_export = self.exports_dir / f".{tag}.pt.tmp"
            shutil.copy2(target / "adapter_export.pt", temporary_export)
            temporary_export.replace(export_path)
            write_text_atomic(self.states_dir / "LATEST", f"{tag}\n")
            write_text_atomic(self.exports_dir / "LATEST", f"{tag}.pt\n")
            self._prune_checkpoints()

        self._run_main_phase(finalize_checkpoint, phase="finalization")
        return target

    def _prune_checkpoints(self) -> None:
        states = sorted(
            path
            for path in self.states_dir.glob("step_*")
            if path.is_dir() and (path / "COMPLETE").is_file()
        )
        for state in states[:-self.keep_last]:
            export = self.exports_dir / f"{state.name}.pt"
            shutil.rmtree(state)
            if export.exists():
                export.unlink()

    def resume_strict(self, state_dir: str | Path) -> None:
        def validate_local_state():
            state_path = Path(state_dir)
            if state_path.is_file() and state_path.name == "LATEST":
                state_path = state_path.parent / state_path.read_text(
                    encoding="utf-8"
                ).strip()
            manifest = validate_training_state(
                state_path,
                expected_contract=self.resume_contract,
            )
            checkpoint_identity = {
                "manifest_sha256": sha256_file(state_path / "manifest.json"),
                "global_step": int(manifest["global_step"]),
                "epoch": int(manifest["epoch"]),
                "batch_in_epoch": int(manifest["batch_in_epoch"]),
                "scheduler_last_epoch": int(
                    manifest["scheduler_last_epoch"]
                ),
            }
            return state_path, manifest, checkpoint_identity

        state_path, manifest, checkpoint_identity = self._run_all_rank_phase(
            validate_local_state,
            phase="resume validation",
        )
        self._require_all_rank_identical(
            checkpoint_identity,
            phase="resume checkpoint",
        )
        self._run_all_rank_phase(
            lambda: self.accelerator.load_state(
                str(state_path / "accelerator")
            ),
            phase="resume load",
        )

        def restore_cursor_and_validate() -> None:
            self.global_step = int(manifest["global_step"])
            self.epoch = int(manifest["epoch"])
            self.batch_in_epoch = int(manifest["batch_in_epoch"])
            self.train_sampler.set_epoch_offset(self.epoch)
            self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
            if _scheduler_last_epoch(self.scheduler) != int(
                manifest["scheduler_last_epoch"]
            ):
                raise RuntimeError(
                    "Stage 3 scheduler state did not restore exactly"
                )

        self._run_all_rank_phase(
            restore_cursor_and_validate,
            phase="resume state validation",
        )
        logger.info(
            "Strictly resumed Stage 3 at step=%d epoch=%d batch=%d",
            self.global_step,
            self.epoch,
            self.batch_in_epoch,
        )

    def train(self) -> None:
        data_iter = iter(self.train_loader)
        start_time = time.perf_counter()
        while self.global_step < self.max_steps:
            try:
                sample = next(data_iter)
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                data_iter = iter(self.train_loader)
                continue

            step_succeeded = False
            grad_norm = None
            with self.accelerator.accumulate(self.adapter_module):
                out = self.build_loss(sample)
                self.accelerator.backward(out.loss)
                if self.accelerator.sync_gradients:
                    self._assert_no_base_gradients()
                    grad_norm = self.accelerator.clip_grad_norm_(
                        self.adapter_module.parameters(),
                        self.max_grad_norm,
                    )
                self.optimizer.step()
                step_succeeded = (
                    self.accelerator.sync_gradients
                    and not self.accelerator.optimizer_step_was_skipped
                )
                if step_succeeded:
                    self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            if not step_succeeded:
                continue
            self.global_step += 1
            if self.log_every > 0 and self.global_step % self.log_every == 0:
                loss_value = float(
                    self.accelerator.gather(
                        out.loss.detach().float().reshape(1)
                    ).mean().item()
                )
                if self.accelerator.is_main_process:
                    elapsed = max(time.perf_counter() - start_time, 1e-6)
                    logger.info(
                        "[stage3] step=%d/%d loss=%.6f grad_norm=%.6f "
                        "lr=%.3e elapsed=%.1fs",
                        self.global_step,
                        self.max_steps,
                        loss_value,
                        float(grad_norm) if grad_norm is not None else float("nan"),
                        float(self.optimizer.param_groups[0]["lr"]),
                        elapsed,
                    )
            if self.save_every > 0 and self.global_step % self.save_every == 0:
                self.save_checkpoint()

        if self.save_final:
            tag = self.states_dir / f"step_{self.global_step:06d}"
            if not tag.exists():
                self.save_checkpoint()
