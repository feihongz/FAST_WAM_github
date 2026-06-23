from typing import Any, Optional

import torch
import torch.nn as nn

from .action_dit import ActionDiT
from .fastwam import FastWAM
from .fastwam_joint import FastWAMJoint
from .fastwam_unified_shared import FastWAMUnifiedShared
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT


class FastWAMUnifiedTwoAction(FastWAMUnifiedShared):
    """Unified variant with one shared video DiT and two ActionDiTs."""

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        loss_alpha_wo: float = 0.5,
        loss_alpha_w: float = 0.5,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for FastWAMUnifiedTwoAction.")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for FastWAMUnifiedTwoAction.")
        if bool(video_dit_config.get("action_conditioned", False)):
            raise ValueError(
                "FastWAMUnifiedTwoAction requires `video_dit_config['action_conditioned']=false`."
            )

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )
        video_expert = components.dit
        action_expert_wo = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        action_expert_w = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        for action_expert in (action_expert_wo, action_expert_w):
            if int(action_expert.num_heads) != int(video_expert.num_heads):
                raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
            if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
                raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
            if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
                raise ValueError("ActionDiT `num_layers` must match video expert.")

        mot_wo = MoT(
            mixtures={"video": video_expert, "action": action_expert_wo},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )
        mot_w = MoT(
            mixtures={"video": video_expert, "action": action_expert_w},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert_wo,
            mot=mot_wo,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
        )
        model.set_action_mix_weights(loss_alpha_wo, loss_alpha_w)
        model.action_expert_wo = action_expert_wo
        model.action_expert_w = action_expert_w
        model.mot_wo = mot_wo
        model.mot_w = mot_w
        model.action_expert = action_expert_wo
        model.mot = mot_wo
        model.dit = nn.ModuleDict({"mot_wo": mot_wo, "mot_w": mot_w})
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_wo_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
            "action_dit_w_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model

    def training_loss(self, sample, tiled: bool = False):
        if not hasattr(self, "loss_alpha_wo") or not hasattr(self, "loss_alpha_w"):
            self.set_action_mix_weights()

        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)
        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )
        action_pre_wo = self.action_expert_wo.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_pre_w = self.action_expert_w.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        video_seq_len = video_pre["tokens"].shape[1]
        video_tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
        mask_wo = self._build_wo_video_mask(
            video_seq_len=video_seq_len,
            action_seq_len=action_pre_wo["tokens"].shape[1],
            video_tokens_per_frame=video_tokens_per_frame,
            device=video_pre["tokens"].device,
        )
        mask_w = self._build_w_video_mask(
            video_seq_len=video_seq_len,
            action_seq_len=action_pre_w["tokens"].shape[1],
            video_tokens_per_frame=video_tokens_per_frame,
            device=video_pre["tokens"].device,
        )

        tokens_out_wo = self._mot_forward(
            mot=self.mot_wo,
            video_pre=video_pre,
            action_pre=action_pre_wo,
            attention_mask=mask_wo,
        )
        tokens_out_w = self._mot_forward(
            mot=self.mot_w,
            video_pre=video_pre,
            action_pre=action_pre_w,
            attention_mask=mask_w,
        )

        pred_video = self.video_expert.post_dit(tokens_out_wo["video"], video_pre)
        pred_action_wo = self.action_expert_wo.post_dit(tokens_out_wo["action"], action_pre_wo)
        pred_action_w = self.action_expert_w.post_dit(tokens_out_w["action"], action_pre_w)

        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device,
            dtype=loss_video_per_sample.dtype,
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        loss_action_wo = self._action_loss(
            pred_action=pred_action_wo,
            target_action=target_action,
            action_is_pad=action_is_pad,
            timestep_action=timestep_action,
        )
        loss_action_w = self._action_loss(
            pred_action=pred_action_w,
            target_action=target_action,
            action_is_pad=action_is_pad,
            timestep_action=timestep_action,
        )
        loss_action_mix = self.loss_alpha_wo * loss_action_wo + self.loss_alpha_w * loss_action_w
        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action_mix
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action_mix.detach().item()),
            "loss_action_wo": float(loss_action_wo.detach().item()),
            "loss_action_w": float(loss_action_w.detach().item()),
            "loss_alpha_wo": float(self.loss_alpha_wo),
            "loss_alpha_w": float(self.loss_alpha_w),
        }
        return loss_total, loss_dict

    @torch.no_grad()
    def infer_action_without_video(self, *args, **kwargs):
        return FastWAM.infer_action(self, *args, **kwargs)

    @torch.no_grad()
    def infer_action_with_video(self, *args, **kwargs):
        previous_mode = getattr(self, "_unified_inference_mode", "wo")
        previous_action_expert = self.action_expert
        previous_mot = self.mot
        self._unified_inference_mode = "w"
        self.action_expert = self.action_expert_w
        self.mot = self.mot_w
        try:
            return FastWAMJoint.infer_action(self, *args, **kwargs)
        finally:
            self.action_expert = previous_action_expert
            self.mot = previous_mot
            self._unified_inference_mode = previous_mode

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "dit": self.dit.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        if "dit" in payload:
            self.dit.load_state_dict(payload["dit"], strict=False)
        elif "mot_wo" in payload and "mot_w" in payload:
            self.mot_wo.load_state_dict(payload["mot_wo"], strict=False)
            self.mot_w.load_state_dict(payload["mot_w"], strict=False)
        elif "mot" in payload:
            self.mot_wo.load_state_dict(payload["mot"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing `dit`, `mot_wo`/`mot_w`, or `mot` keys: {path}")

        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload
