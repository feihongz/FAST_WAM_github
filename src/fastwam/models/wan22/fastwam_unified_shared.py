from typing import Any

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
        raise ValueError(f"Unknown inference_mode: {inference_mode}")
