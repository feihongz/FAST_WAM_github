from typing import Any, Optional

import torch
import torch.nn.functional as F

from .fastwam import FastWAM
from .fastwam_joint import FastWAMJoint


class FastWAMUnifiedShared(FastWAM):
    """Unified variant with one shared video DiT and one shared ActionDiT."""

    @classmethod
    def from_wan22_pretrained(
        cls,
        loss_alpha_wo: float = 0.5,
        loss_alpha_w: float = 0.5,
        **kwargs,
    ):
        model = super().from_wan22_pretrained(**kwargs)
        model.set_action_mix_weights(loss_alpha_wo, loss_alpha_w)
        return model

    def set_action_mix_weights(self, alpha_wo: float = 0.5, alpha_w: float = 0.5):
        alpha_wo = float(alpha_wo)
        alpha_w = float(alpha_w)
        denom = alpha_wo + alpha_w
        if denom <= 0:
            raise ValueError("alpha_wo + alpha_w must be positive.")
        self.loss_alpha_wo = alpha_wo / denom
        self.loss_alpha_w = alpha_w / denom

    @torch.no_grad()
    def _build_wo_video_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        return FastWAM._build_mot_attention_mask(
            self,
            video_seq_len=video_seq_len,
            action_seq_len=action_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )

    @torch.no_grad()
    def _build_w_video_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        mask[video_seq_len:, video_seq_len:] = True
        mask[video_seq_len:, :video_seq_len] = True
        return mask

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        if str(getattr(self, "_unified_inference_mode", "wo")) == "w":
            return self._build_w_video_mask(
                video_seq_len=video_seq_len,
                action_seq_len=action_seq_len,
                video_tokens_per_frame=video_tokens_per_frame,
                device=device,
            )
        return self._build_wo_video_mask(
            video_seq_len=video_seq_len,
            action_seq_len=action_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )

    def _action_loss(
        self,
        pred_action: torch.Tensor,
        target_action: torch.Tensor,
        action_is_pad: torch.Tensor | None,
        timestep_action: torch.Tensor,
    ) -> torch.Tensor:
        action_loss_token = F.mse_loss(
            pred_action.float(),
            target_action.float(),
            reduction="none",
        ).mean(dim=2)
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device,
            dtype=action_loss_per_sample.dtype,
        )
        return (action_loss_per_sample * action_weight).mean()

    def _mot_forward(
        self,
        mot,
        video_pre: dict[str, Any],
        action_pre: dict[str, Any],
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

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
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        video_seq_len = video_pre["tokens"].shape[1]
        action_seq_len = action_pre["tokens"].shape[1]
        video_tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
        mask_wo = self._build_wo_video_mask(
            video_seq_len=video_seq_len,
            action_seq_len=action_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=video_pre["tokens"].device,
        )
        mask_w = self._build_w_video_mask(
            video_seq_len=video_seq_len,
            action_seq_len=action_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=video_pre["tokens"].device,
        )

        tokens_out_wo = self._mot_forward(
            mot=self.mot,
            video_pre=video_pre,
            action_pre=action_pre,
            attention_mask=mask_wo,
        )
        tokens_out_w = self._mot_forward(
            mot=self.mot,
            video_pre=video_pre,
            action_pre=action_pre,
            attention_mask=mask_w,
        )

        pred_video = self.video_expert.post_dit(tokens_out_wo["video"], video_pre)
        pred_action_wo = self.action_expert.post_dit(tokens_out_wo["action"], action_pre)
        pred_action_w = self.action_expert.post_dit(tokens_out_w["action"], action_pre)

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
    def _build_action_only_video_cache(
        self,
        first_frame_latents: torch.Tensor,
        latents_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> tuple[list[dict[str, torch.Tensor]], torch.Tensor, int]:
        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_wo_video_mask(
            video_seq_len=video_seq_len,
            action_seq_len=int(latents_action.shape[1]),
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )
        return video_kv_cache, attention_mask, video_seq_len

    @torch.no_grad()
    def _infer_action_without_video_custom(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action_video_prefix(video_prefix_steps=0)` requires "
                "`video_attention_mask_mode='first_frame_causal'`."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )

        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        previous_mode = getattr(self, "_unified_inference_mode", "wo")
        self._unified_inference_mode = "wo"
        try:
            video_kv_cache, attention_mask, video_seq_len = self._build_action_only_video_cache(
                first_frame_latents=first_frame_latents,
                latents_action=latents_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
            infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=self.device,
                dtype=latents_action.dtype,
                shift_override=sigma_shift,
            )
            for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
                timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
                pred_action = self._predict_action_noise_with_cache(
                    latents_action=latents_action,
                    timestep_action=timestep_action,
                    context=context,
                    context_mask=context_mask,
                    video_kv_cache=video_kv_cache,
                    attention_mask=attention_mask,
                    video_seq_len=video_seq_len,
                )
                latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
        finally:
            self._unified_inference_mode = previous_mode

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
            "video_prefix_steps": 0,
            "num_inference_steps": int(num_inference_steps),
            "force_custom_prefix": True,
        }

    @torch.no_grad()
    def infer_action_video_prefix(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: int,
        video_prefix_steps: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        force_custom_prefix: bool = False,
    ) -> dict[str, Any]:
        del negative_prompt, text_cfg_scale
        self.eval()

        video_prefix_steps = int(video_prefix_steps)
        num_inference_steps = int(num_inference_steps)
        if video_prefix_steps < 0:
            raise ValueError(f"`video_prefix_steps` must be non-negative, got {video_prefix_steps}")
        if video_prefix_steps > num_inference_steps:
            raise ValueError(
                "`video_prefix_steps` cannot exceed `num_inference_steps`: "
                f"{video_prefix_steps} > {num_inference_steps}"
            )
        force_custom_prefix = True
        if video_prefix_steps == 0:
            return self._infer_action_without_video_custom(
                prompt=prompt,
                input_image=input_image,
                action_horizon=action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
            )
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action_video_prefix` requires `video_attention_mask_mode='first_frame_causal'` "
                "for its action-only suffix."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}")

        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )

        previous_mode = getattr(self, "_unified_inference_mode", "wo")
        video_kv_cache = None
        wo_attention_mask = None
        wo_video_seq_len = 0
        try:
            for step_idx, (step_t_video, step_delta_video, step_t_action, step_delta_action) in enumerate(
                zip(infer_timesteps_video, infer_deltas_video, infer_timesteps_action, infer_deltas_action)
            ):
                timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
                if step_idx < video_prefix_steps:
                    self._unified_inference_mode = "w"
                    timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
                    pred_video, pred_action = self._predict_joint_noise(
                        latents_video=latents_video,
                        latents_action=latents_action,
                        timestep_video=timestep_video,
                        timestep_action=timestep_action,
                        context=context,
                        context_mask=context_mask,
                        fuse_vae_embedding_in_latents=fuse_flag,
                        gt_action=None,
                    )
                    latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
                    latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
                    latents_video[:, :, 0:1] = first_frame_latents.clone()
                    continue

                self._unified_inference_mode = "wo"
                if video_kv_cache is None or wo_attention_mask is None:
                    video_kv_cache, wo_attention_mask, wo_video_seq_len = self._build_action_only_video_cache(
                        first_frame_latents=first_frame_latents,
                        latents_action=latents_action,
                        context=context,
                        context_mask=context_mask,
                        fuse_vae_embedding_in_latents=fuse_flag,
                    )
                pred_action = self._predict_action_noise_with_cache(
                    latents_action=latents_action,
                    timestep_action=timestep_action,
                    context=context,
                    context_mask=context_mask,
                    video_kv_cache=video_kv_cache,
                    attention_mask=wo_attention_mask,
                    video_seq_len=wo_video_seq_len,
                )
                latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
        finally:
            self._unified_inference_mode = previous_mode

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
            "video_prefix_steps": video_prefix_steps,
            "num_inference_steps": num_inference_steps,
            "force_custom_prefix": force_custom_prefix,
        }

    @torch.no_grad()
    def infer_action_without_video(self, *args, **kwargs):
        return FastWAM.infer_action(self, *args, **kwargs)

    @torch.no_grad()
    def infer_action_with_video(self, *args, **kwargs):
        previous_mode = getattr(self, "_unified_inference_mode", "wo")
        self._unified_inference_mode = "w"
        try:
            return FastWAMJoint.infer_action(self, *args, **kwargs)
        finally:
            self._unified_inference_mode = previous_mode

    @torch.no_grad()
    def infer_action_mode(self, *args, inference_mode: str = "wo", **kwargs):
        mode = str(inference_mode).lower()
        if mode == "wo":
            return self.infer_action_without_video(*args, **kwargs)
        if mode == "w":
            return self.infer_action_with_video(*args, **kwargs)
        if mode == "prefix":
            return self.infer_action_video_prefix(*args, **kwargs)
        raise ValueError(f"Unknown inference_mode: {inference_mode}")
