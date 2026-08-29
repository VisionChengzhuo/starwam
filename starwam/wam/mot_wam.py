"""MoT WAM implementation.

MoT WAM uses separate video and action experts, with per-layer joint attention
between their Q/K/V streams. The implementation keeps Fast-WAM-aligned details:
first-frame clean pinning, configurable first-frame/full-video action attention,
velocity target ``noise - sample``, weighted flow matching, and action inference.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

from starwam.action_model import build_action_expert, load_action_dit_init
from starwam.training.flow import add_flow_noise, build_inference_schedule, video_latent_pad_mask
from starwam.wam.base import WAMModel
from starwam.backbone.base import BaseBackbone
from starwam.config import FrameworkConfig
from starwam.modules.mot import MoT
from starwam.modules.scheduler import FlowMatchScheduler
from starwam.training.loss import flow_matching_loss
from starwam.training.metrics import action_monitor_metrics
from starwam.training.timing import TrainingTimingMixin


class MoTWAM(TrainingTimingMixin, WAMModel):
    """Fast-WAM/Motus-style multi-stream WAM."""

    taxonomy_model_family = "mot_wam"

    def __init__(
        self,
        backbone: BaseBackbone,
        config: FrameworkConfig,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.config = config
        self._device = device
        self._dtype = dtype
        self.backbone = backbone

        self.action_expert = build_action_expert(backbone.info, config)
        load_action_dit_init(
            self.action_expert,
            getattr(config, "action_expert_init_from", None),
            head_init=getattr(config, "action_expert_head_init", "random"),
        )

        self.mot = MoT(
            experts={"video": backbone.get_dit(), "action": self.action_expert},
            checkpoint_mixed_attn=config.mot_checkpoint_mixed_attn,
        )

        self.video_scheduler = FlowMatchScheduler(
            num_train_timesteps=config.video_scheduler.num_train_timesteps,
            shift=config.video_scheduler.train_shift,
        )
        self.action_scheduler = FlowMatchScheduler(
            num_train_timesteps=config.action_scheduler.num_train_timesteps,
            shift=config.action_scheduler.train_shift,
        )
        self.loss_lambda_video = config.loss_lambda_video
        self.loss_lambda_action = config.loss_lambda_action
        self.action_video_conditioning = getattr(config, "action_video_conditioning", "first_frame")
        if self.action_video_conditioning not in {"first_frame", "full_video"}:
            raise ValueError(
                "framework.action_video_conditioning must be 'first_frame' or 'full_video', "
                f"got {self.action_video_conditioning!r}"
            )
        self.proprio_dim = None
        self.proprio_encoder = None
        if getattr(config, "proprio_dim", None):
            self.proprio_dim = int(config.proprio_dim)
            if self.proprio_dim > 0:
                self.proprio_encoder = nn.Linear(self.proprio_dim, backbone.info.text_dim).to(device=device, dtype=dtype)
            else:
                self.proprio_dim = None

        self.backbone.get_dit().to(device=device, dtype=dtype)
        self.action_expert.to(device=device, dtype=dtype)
        self.mot.to(device=device, dtype=dtype)

    def training_step(self, sample: dict[str, Any]) -> tuple[Tensor, dict[str, float]]:
        video = sample["video"]
        action = sample["action"]
        context = sample["context"]
        context_mask = sample.get("context_mask")
        action_is_pad = sample.get("action_is_pad")
        image_is_pad = sample.get("image_is_pad")
        device = video.device
        context, context_mask = self._append_proprio_from_sample(sample, context, context_mask)

        with self.training_timer("vae_encode", device=device):
            with torch.no_grad():
                video_latents = self.backbone.encode_video(video)

        noisy_video, target_video, t_video = add_flow_noise(
            self.video_scheduler,
            video_latents,
            pin_first_latent_step=True,
        )
        noisy_action, target_action, t_action = add_flow_noise(self.action_scheduler, action)

        with self.training_timer("dit_forward", device=device):
            video_dit = self.backbone.get_dit()
            video_state = video_dit.pre_dit(noisy_video, t_video, context, context_mask)
            action_state = self.action_expert.pre_dit(noisy_action, t_action, context, context_mask)

            attention_mask = self._build_mot_attention_mask(
                video_seq_len=video_state["tokens"].shape[1],
                action_seq_len=action_state["tokens"].shape[1],
                video_tokens_per_frame=self._compute_video_tokens_per_frame(video_state),
                device=device,
                action_video_conditioning=self.action_video_conditioning,
            )
            output_tokens = self.mot({"video": video_state, "action": action_state}, attention_mask)

            pred_video = video_dit.post_dit(output_tokens["video"], video_state["meta"], video_state["t"])
            pred_action = self.action_expert.post_dit(output_tokens["action"])

        include_initial_video_step = not (video_latents.shape[2] > 1)
        if video_latents.shape[2] > 1:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]
        video_is_pad = video_latent_pad_mask(
            image_is_pad,
            latent_steps=pred_video.shape[2],
            include_initial_video_step=include_initial_video_step,
        )

        loss_video = flow_matching_loss(
            pred_video,
            target_video,
            t_video,
            self.video_scheduler,
            is_pad_mask=video_is_pad,
        )
        loss_action = flow_matching_loss(
            pred_action,
            target_action,
            t_action,
            self.action_scheduler,
            is_pad_mask=action_is_pad,
        )
        loss = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action

        loss_dict = {
            "loss_video": loss_video.item(),
            "loss_action": loss_action.item(),
            "loss_total": loss.item(),
        }
        loss_dict.update(action_monitor_metrics(
            pred_action.detach(),
            target_action.detach(),
            action.detach(),
            is_pad=action_is_pad,
            gripper_dim=getattr(self.config, "action_gripper_dim", -1),
        ))
        return loss, loss_dict

    @torch.no_grad()
    def infer_action(
        self,
        input_image: Tensor,
        context: Tensor,
        context_mask: Tensor,
        action_horizon: int,
        num_inference_steps: int = 10,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Tensor:
        """Denoise actions from images and text.

        input_image: [B, C, H, W] -> action: [B, K, A].
        """
        proprio = kwargs.get("proprio")
        num_video_frames = kwargs.get("num_video_frames")
        device = input_image.device
        dtype = input_image.dtype
        batch_size = input_image.shape[0]
        context, context_mask = self._append_proprio_to_context(context, context_mask, proprio)
        generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

        if self.action_video_conditioning == "full_video":
            if num_video_frames is None:
                raise ValueError("num_video_frames is required for full-video action inference")
            video_single = input_image.unsqueeze(2)
            first_frame_latents = self.backbone.encode_video(video_single)
            _, channels, _, height_latent, width_latent = first_frame_latents.shape
            vae = self.backbone.get_vae()
            latent_steps = max(1, (int(num_video_frames) - 1) // vae.temporal_compress + 1)
            video_latents = torch.randn(
                batch_size,
                channels,
                latent_steps,
                height_latent,
                width_latent,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            video_latents[:, :, 0:1] = first_frame_latents
            action_latents = torch.randn(
                batch_size,
                action_horizon,
                self.config.action_dim,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            video_timesteps, video_deltas = build_inference_schedule(
                self.config.video_scheduler,
                num_inference_steps,
                device,
                dtype,
            )
            action_timesteps, action_deltas = build_inference_schedule(
                self.config.action_scheduler,
                num_inference_steps,
                device,
                dtype,
            )
            video_dit = self.backbone.get_dit()
            for i in range(num_inference_steps):
                video_state = video_dit.pre_dit(
                    video_latents, video_timesteps[i].expand(batch_size), context, context_mask
                )
                action_state = self.action_expert.pre_dit(
                    action_latents, action_timesteps[i].expand(batch_size), context, context_mask
                )
                attention_mask = self._build_mot_attention_mask(
                    video_seq_len=video_state["tokens"].shape[1],
                    action_seq_len=action_state["tokens"].shape[1],
                    video_tokens_per_frame=self._compute_video_tokens_per_frame(video_state),
                    device=device,
                    action_video_conditioning="full_video",
                )
                output_tokens = self.mot({"video": video_state, "action": action_state}, attention_mask)
                video_velocity = video_dit.post_dit(output_tokens["video"], video_state["meta"], video_state["t"])
                action_velocity = self.action_expert.post_dit(output_tokens["action"])
                video_latents = FlowMatchScheduler.step(video_velocity, video_deltas[i], video_latents)
                action_latents = FlowMatchScheduler.step(action_velocity, action_deltas[i], action_latents)
                video_latents[:, :, 0:1] = first_frame_latents
            return action_latents

        video_single = input_image.unsqueeze(2)
        video_latents = self.backbone.encode_video(video_single)
        video_dit = self.backbone.get_dit()
        t_zero = torch.zeros(batch_size, device=device, dtype=torch.long)
        video_state = video_dit.pre_dit(video_latents, t_zero, context, context_mask)
        video_kv_cache = self.mot.prefill_video_cache(video_state)

        action_latents = torch.randn(
            batch_size,
            action_horizon,
            self.config.action_dim,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        timesteps, deltas = build_inference_schedule(
            self.config.action_scheduler,
            num_inference_steps,
            device,
            dtype,
        )
        for i in range(num_inference_steps):
            t_action = timesteps[i].expand(batch_size)
            action_state = self.action_expert.pre_dit(action_latents, t_action, context, context_mask)
            action_out = self.mot.forward_action_with_video_cache(action_state, video_kv_cache)
            velocity = self.action_expert.post_dit(action_out)
            action_latents = FlowMatchScheduler.step(velocity, deltas[i], action_latents)
        return action_latents

    @torch.no_grad()
    def infer_joint(
        self,
        input_image: Tensor,
        context: Tensor,
        context_mask: Tensor,
        num_video_frames: int,
        action_horizon: int,
        num_inference_steps: int = 20,
        **kwargs,
    ) -> dict[str, Any]:
        device = input_image.device
        dtype = input_image.dtype
        seed = kwargs.get("seed")
        context, context_mask = self._append_proprio_to_context(context, context_mask, kwargs.get("proprio"))
        generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

        video_single = input_image.unsqueeze(2)
        first_frame_latents = self.backbone.encode_video(video_single)
        _, channels, _, height_latent, width_latent = first_frame_latents.shape
        vae = self.backbone.get_vae()
        latent_steps = max(1, (int(num_video_frames) - 1) // vae.temporal_compress + 1)

        video_latents = torch.randn(
            1,
            channels,
            latent_steps,
            height_latent,
            width_latent,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        video_latents[:, :, 0:1] = first_frame_latents
        action_latents = torch.randn(
            1,
            action_horizon,
            self.config.action_dim,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        video_timesteps, video_deltas = build_inference_schedule(
            self.config.video_scheduler,
            num_inference_steps,
            device,
            dtype,
        )
        action_timesteps, action_deltas = build_inference_schedule(
            self.config.action_scheduler,
            num_inference_steps,
            device,
            dtype,
        )
        video_dit = self.backbone.get_dit()

        for i in range(num_inference_steps):
            t_video = video_timesteps[i].reshape(1)
            t_action = action_timesteps[i].reshape(1)
            video_state = video_dit.pre_dit(video_latents, t_video, context, context_mask)
            action_state = self.action_expert.pre_dit(action_latents, t_action, context, context_mask)
            attention_mask = self._build_mot_attention_mask(
                video_seq_len=video_state["tokens"].shape[1],
                action_seq_len=action_state["tokens"].shape[1],
                video_tokens_per_frame=self._compute_video_tokens_per_frame(video_state),
                device=device,
                action_video_conditioning=self.action_video_conditioning,
            )
            output_tokens = self.mot({"video": video_state, "action": action_state}, attention_mask)
            video_velocity = video_dit.post_dit(output_tokens["video"], video_state["meta"], video_state["t"])
            action_velocity = self.action_expert.post_dit(output_tokens["action"])
            video_latents = FlowMatchScheduler.step(video_velocity, video_deltas[i], video_latents)
            action_latents = FlowMatchScheduler.step(action_velocity, action_deltas[i], action_latents)
            video_latents[:, :, 0:1] = first_frame_latents

        decoded_video = self.backbone.decode_latents(video_latents)
        return {"video": decoded_video, "action": action_latents}

    def _append_proprio_from_sample(
        self,
        sample: dict[str, Any],
        context: Tensor,
        context_mask: Optional[Tensor],
    ) -> tuple[Tensor, Optional[Tensor]]:
        if self.proprio_encoder is None:
            return context, context_mask
        proprio = sample.get("proprio")
        if proprio is None:
            raise ValueError("sample['proprio'] is required when framework.proprio_dim is enabled")
        if proprio.ndim != 3:
            raise ValueError(f"sample['proprio'] must be [B, T, D], got {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[-1] != self.proprio_dim:
            raise ValueError(
                f"sample['proprio'] last dim must be {self.proprio_dim}, got {proprio.shape[-1]}"
            )
        return self._append_proprio_to_context(context, context_mask, proprio[:, 0, :])

    def _append_proprio_to_context(
        self,
        context: Tensor,
        context_mask: Optional[Tensor],
        proprio: Optional[Tensor],
    ) -> tuple[Tensor, Optional[Tensor]]:
        if self.proprio_encoder is None:
            return context, context_mask
        if proprio is None:
            raise ValueError("proprio is required when framework.proprio_dim is enabled")
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        if proprio.ndim != 2:
            raise ValueError(f"proprio must be [B, D] or [D], got {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[-1] != self.proprio_dim:
            raise ValueError(f"proprio last dim must be {self.proprio_dim}, got {proprio.shape[-1]}")
        if context_mask is None:
            context_mask = torch.ones(context.shape[:2], dtype=torch.bool, device=context.device)
        encoder_param = next(self.proprio_encoder.parameters())
        proprio_token = self.proprio_encoder(
            proprio.to(device=encoder_param.device, dtype=encoder_param.dtype).unsqueeze(1)
        ).to(device=context.device, dtype=context.dtype)
        proprio_mask = torch.ones((context.shape[0], 1), dtype=torch.bool, device=context.device)
        return torch.cat([context, proprio_token], dim=1), torch.cat([context_mask, proprio_mask], dim=1)

    def _compute_video_tokens_per_frame(self, video_state: dict) -> int:
        meta = video_state.get("meta", {})
        height = int(meta.get("H", 0))
        width = int(meta.get("W", 0))
        _, patch_h, patch_w = self.backbone.info.patch_size
        return max(1, height // patch_h) * max(1, width // patch_w)

    @staticmethod
    def _build_mot_attention_mask(
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
        action_video_conditioning: str = "first_frame",
    ) -> Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)
        first_frame = min(max(video_tokens_per_frame, 1), video_seq_len)

        video_block = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
        video_block[:first_frame, first_frame:] = False
        mask[:video_seq_len, :video_seq_len] = video_block

        mask[video_seq_len:, video_seq_len:] = True
        if action_video_conditioning == "first_frame":
            mask[video_seq_len:, :first_frame] = True
        elif action_video_conditioning == "full_video":
            mask[video_seq_len:, :video_seq_len] = True
        else:
            raise ValueError(
                "action_video_conditioning must be 'first_frame' or 'full_video', "
                f"got {action_video_conditioning!r}"
            )
        return mask
