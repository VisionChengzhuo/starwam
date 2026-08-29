"""StarWAM trainer: unified training loop with HuggingFace Accelerate + DeepSpeed."""

import json
import os
import re
import shutil
import time
import logging
from collections import defaultdict
from contextlib import nullcontext
from math import ceil
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

from starwam.config import TrainingConfig
from starwam.training.metrics import action_mse
from starwam.wam.base import WAMModel

logger = logging.getLogger(__name__)


def _checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if match:
        return int(match.group(1))
    meta_path = path / "trainer_state.json"
    if meta_path.is_file():
        with open(meta_path, "r", encoding="utf-8") as f:
            return int(json.load(f)["global_step"])
    raise ValueError(f"Cannot infer global step from checkpoint path: {path}")


class StarWAMTrainer:
    """Unified trainer for StarWAM models.

    Features:
    - HuggingFace Accelerate for distributed training (DeepSpeed ZeRO-2)
    - Freezes VAE + text encoder, trains only MoT (video + action experts)
    - Gradient accumulation, mixed precision, gradient clipping
    - Periodic eval, checkpointing, logging
    """

    def __init__(
        self,
        model: WAMModel,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        config: Optional[TrainingConfig] = None,
    ):
        self.config = config or TrainingConfig()
        if (
            self.config.train_action_mse_every is not None
            and self.config.train_action_mse_every <= 0
        ):
            raise ValueError(
                "training.train_action_mse_every must be positive or None, "
                f"got {self.config.train_action_mse_every!r}"
            )
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Setup accelerator
        try:
            from accelerate import Accelerator
            self.accelerator = Accelerator(
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                mixed_precision=self.config.mixed_precision,
                step_scheduler_with_optimizer=False,
            )
        except ImportError:
            # Fallback: no accelerator (single GPU, no mixed precision management)
            self.accelerator = None
            logger.warning("accelerate not installed, running without distributed support")

        self._uses_deepspeed = False
        if self.accelerator is not None:
            zero_stage = "none"
            deepspeed_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
            if deepspeed_plugin is not None:
                self._uses_deepspeed = True
                deepspeed_config = deepspeed_plugin.deepspeed_config
                zero_stage = deepspeed_config.get("zero_optimization", {}).get("stage", "unknown")
                # Accelerator.clip_grad_norm_ only reads DeepSpeed's global
                # norm. DeepSpeed performs the actual clipping inside
                # optimizer.step(), so propagate the trainer threshold into
                # the engine config before accelerator.prepare() builds it.
                deepspeed_config["gradient_clipping"] = float(self.config.max_grad_norm)
            logger.info(
                "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d mixed_precision=%s "
                "grad_accum=%d max_grad_norm=%.4f",
                self.accelerator.distributed_type,
                zero_stage,
                self.accelerator.num_processes,
                self.accelerator.mixed_precision,
                self.config.gradient_accumulation_steps,
                self.config.max_grad_norm,
            )

        self.global_step = 0
        self._wandb_run = None
        self._checkpoint_uploader = None
        self._resume_step: Optional[int] = None
        self._loss_metric_sums: dict[str, float] = defaultdict(float)
        self._loss_metric_counts: dict[str, int] = defaultdict(int)
        self._grad_norm_sum: Optional[torch.Tensor] = None
        self._grad_clipped_sum: Optional[torch.Tensor] = None
        self._grad_metric_count = 0
        self._load_model_only_if_requested()
        self._setup()
        self._resume_if_requested()
        self._apply_resume_lr_override_if_requested()
        self._init_wandb()
        self._init_component_timing()
        self._init_checkpoint_uploader()

    def _init_wandb(self):
        """Initialise a wandb run on the main process if enabled. Failures
        are non-fatal (logged as warnings)."""
        if not getattr(self.config, "wandb_enabled", False):
            return
        if not self._is_main_process():
            return
        try:
            import wandb  # type: ignore
        except ImportError:
            logger.warning("wandb_enabled=True but wandb not installed; skipping.")
            return
        try:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            id_path = output_dir / "wandb_id.txt"
            # A shared output directory does not imply that a fresh training
            # task should resume the previous W&B run. Reuse the persisted ID
            # only when the trainer actually restored a checkpoint.
            is_training_resume = self._resume_step is not None
            run_id = None
            if is_training_resume and id_path.is_file():
                run_id = id_path.read_text(encoding="utf-8").strip() or None
            if run_id is None:
                run_id = wandb.util.generate_id()
                id_path.write_text(f"{run_id}\n", encoding="utf-8")

            self._wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=getattr(self.config, "wandb_run_name", None),
                id=run_id,
                resume="allow" if is_training_resume else "never",
                dir=str(output_dir),
                reinit="finish_previous",
            )
            if self._wandb_run is not None:
                self._wandb_run.config.update({
                    "runtime/world_size": self.world_size,
                    "runtime/effective_global_batch_size": self.effective_global_batch_size,
                    "runtime/resolved_max_steps": self.max_steps,
                    "runtime/resolved_warmup_steps": self.warmup_steps,
                }, allow_val_change=True)
                self._wandb_run.define_metric("train/global_step")
                for namespace in ("train/*", "eval/*", "perf/*"):
                    self._wandb_run.define_metric(namespace, step_metric="train/global_step")
                logger.info(
                    "W&B initialized: id=%s url=%s",
                    self._wandb_run.id,
                    getattr(self._wandb_run, "url", None),
                )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"wandb.init failed ({e}); continuing without wandb.")
            self._wandb_run = None

    def _init_component_timing(self) -> None:
        """Enable non-blocking model timers when W&B logging is requested."""
        set_timing_enabled = getattr(self._unwrap(), "set_training_timing_enabled", None)
        if callable(set_timing_enabled):
            set_timing_enabled(self.config.wandb_enabled)

    def _wandb_log(self, payload: dict):
        if self._wandb_run is None:
            return
        try:
            # Loss, checkpoint, and eval may all log at the same
            # optimizer step. A custom step metric keeps every record instead
            # of dropping later calls after an explicit W&B step is committed.
            self._wandb_run.log({
                "train/global_step": self.global_step,
                **payload,
            })
        except Exception as e:  # noqa: BLE001
            logger.warning(f"wandb.log failed ({e})")

    def _init_checkpoint_uploader(self) -> None:
        upload_config = self.config.checkpoint_upload
        if not upload_config.enabled:
            if self._is_main_process():
                logger.info("Remote checkpoint upload disabled; checkpoints will remain local only")
            return
        if not self._is_main_process():
            return
        from starwam.tools.checkpoint_tos import build_checkpoint_uploader

        uploader = build_checkpoint_uploader(
            self.config,
            world_size=self.world_size,
        )
        assert uploader is not None
        self._checkpoint_uploader = uploader
        logger.info(
            "TOS checkpoint upload enabled: asynchronous=%s local_checkpoints_retained=true",
            upload_config.asynchronous,
        )

    def _accumulate_loss_metrics(self, loss_dict: dict) -> None:
        """Accumulate micro-batch metrics until the next logging step."""
        for key, raw_value in loss_dict.items():
            if isinstance(raw_value, torch.Tensor):
                if raw_value.numel() != 1:
                    continue
                value = float(raw_value.detach().float().item())
            elif isinstance(raw_value, (int, float)):
                value = float(raw_value)
            else:
                continue
            self._loss_metric_sums[key] += value
            self._loss_metric_counts[key] += 1

    def _capture_grad_norm(self, grad_norm: object) -> None:
        if grad_norm is None:
            return
        if isinstance(grad_norm, torch.Tensor):
            value = grad_norm.detach().float().reshape(())
        elif isinstance(grad_norm, (int, float)):
            value = torch.tensor(float(grad_norm), device=self._model_compute_param().device)
        else:
            return
        max_norm = max(float(self.config.max_grad_norm), 0.0)
        clipped = (value > max_norm).float()
        self._grad_norm_sum = value if self._grad_norm_sum is None else self._grad_norm_sum + value
        self._grad_clipped_sum = clipped if self._grad_clipped_sum is None else self._grad_clipped_sum + clipped
        self._grad_metric_count += 1

    def _distributed_mean_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        if not metrics or not (dist.is_available() and dist.is_initialized()):
            return metrics
        keys = sorted(metrics)
        backend = str(dist.get_backend()).lower()
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if backend == "nccl"
            else torch.device("cpu")
        )
        values = torch.tensor([metrics[key] for key in keys], dtype=torch.float64, device=device)
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= float(dist.get_world_size())
        return {key: float(value) for key, value in zip(keys, values.cpu().tolist())}

    def _flush_timing_window(
        self,
        *,
        timing_sums: dict[str, float],
        timing_counts: dict[str, int],
        completed_steps: int,
        elapsed: float,
    ) -> dict[str, float]:
        """Build rank-averaged W&B performance metrics for one log window."""
        completed_steps = max(int(completed_steps), 1)
        elapsed = max(float(elapsed), 1e-9)
        steps_per_second = completed_steps / elapsed
        local_metrics = {
            "perf/steps_per_sec": steps_per_second,
            "perf/samples_per_sec": steps_per_second * float(self.effective_global_batch_size),
            "perf/step_time_sec": elapsed / completed_steps,
            "perf/eta_hours": (
                max(self.max_steps - self.global_step, 0) / max(steps_per_second, 1e-9) / 3600.0
            ),
        }
        timer_names = {
            "dataloader": "perf/dataloader_time_sec",
            "forward": "perf/forward_time_sec",
            "backward": "perf/backward_time_sec",
            "optimizer": "perf/optimizer_time_sec",
            "vae_encode": "perf/vae_encode_time_sec",
            "dit_forward": "perf/dit_forward_time_sec",
        }
        for source_name, output_name in timer_names.items():
            if source_name not in timing_sums:
                continue
            count = timing_counts.get(source_name, completed_steps)
            local_metrics[output_name] = timing_sums[source_name] / max(int(count), 1)
        return self._distributed_mean_metrics(local_metrics)

    def _flush_train_metric_window(self) -> tuple[dict[str, float], dict[str, float]]:
        """Build one rank-averaged W&B payload and reset the logging window."""
        local_losses = {
            key: total / max(self._loss_metric_counts[key], 1)
            for key, total in self._loss_metric_sums.items()
        }
        averaged_losses = self._distributed_mean_metrics(local_losses)
        self._loss_metric_sums.clear()
        self._loss_metric_counts.clear()

        loss_names = {
            "loss_total": "train/loss",
            "loss_video": "train/video_loss",
            "loss_action": "train/action_loss",
            "loss_action_eef": "train/action_eef_loss",
            "loss_action_gripper": "train/action_gripper_loss",
        }
        payload: dict[str, float] = {
            wandb_name: averaged_losses[source_name]
            for source_name, wandb_name in loss_names.items()
            if source_name in averaged_losses
        }

        if self._grad_metric_count > 0 and self._grad_norm_sum is not None:
            count = float(self._grad_metric_count)
            grad_metrics = {
                "train/grad_norm": float((self._grad_norm_sum / count).item()),
                "train/grad_clipped_fraction": float((self._grad_clipped_sum / count).item()),
            }
            payload.update(self._distributed_mean_metrics(grad_metrics))
        self._grad_norm_sum = None
        self._grad_clipped_sum = None
        self._grad_metric_count = 0

        param_groups = list(self.optimizer.param_groups)
        if param_groups:
            payload["train/learning_rate"] = float(param_groups[0].get("lr", 0.0))
        if len(param_groups) > 1:
            for index, group in enumerate(param_groups):
                payload[f"train/learning_rate_group_{index}"] = float(group.get("lr", 0.0))
        scaler = getattr(self.accelerator, "scaler", None) if self.accelerator is not None else None
        if scaler is not None:
            try:
                payload["train/grad_scale"] = float(scaler.get_scale())
            except Exception:  # noqa: BLE001
                pass
        return averaged_losses, payload

    @staticmethod
    def _format_action_monitor(loss_dict: dict) -> str:
        if "loss_action_gripper" not in loss_dict:
            return ""
        return (
            f" | action_eef={loss_dict['loss_action_eef']:.4f}"
            f" | gripper={loss_dict['loss_action_gripper']:.4f}"
            f" | grip_open={loss_dict['action_target_gripper_open_rate']:.2f}"
        )

    def _setup(self):
        """Setup optimizer, scheduler, dataloader, freeze strategy."""
        # Freeze strategy: only train the MoT (contains both video + action experts)
        self._apply_freeze_strategy()

        # Optimizer
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95),
        )

        # Dataloader
        dataloader_kwargs = {}
        if self.config.num_workers > 0:
            dataloader_kwargs.update({
                "timeout": 120,
                "persistent_workers": True,
                "prefetch_factor": 1,
            })

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
            **dataloader_kwargs,
        )

        # LR scheduler. When max_steps is not explicit, match FastWAM's
        # optimizer-step horizon: epochs over the dataset at global batch size,
        # then divide by gradient accumulation.
        world_size = self.accelerator.num_processes if self.accelerator is not None else 1
        if self.config.max_steps is not None:
            max_steps = max(int(self.config.max_steps), 1)
        else:
            global_batch_size = max(int(self.config.batch_size) * max(int(world_size), 1), 1)
            micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
            max_steps = max(
                ceil(micro_steps_per_epoch / max(int(self.config.gradient_accumulation_steps), 1))
                * int(self.config.num_epochs),
                1,
            )
        if self.config.warmup_steps is not None:
            warmup_steps = int(self.config.warmup_steps)
        else:
            warmup_steps = int(max_steps * self.config.warmup_ratio)

        sched_type = str(self.config.lr_scheduler_type).strip().lower()
        if sched_type in {"cosine", "cosine_with_min_lr"}:
            min_lr = self.config.learning_rate * 0.01 if sched_type == "cosine" else float(self.config.min_lr)
            warmup_steps = min(max(int(warmup_steps), 0), max_steps - 1)
            remaining_steps = max(max_steps - warmup_steps, 1)
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=min_lr,
            )
            if warmup_steps > 0:
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=1.0 / max(warmup_steps, 1),
                    end_factor=1.0,
                    total_iters=warmup_steps,
                )
                self.lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_scheduler, main_scheduler],
                    milestones=[warmup_steps],
                )
            else:
                self.lr_scheduler = main_scheduler
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)

        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.world_size = world_size
        self.effective_global_batch_size = (
            int(self.config.batch_size)
            * max(int(world_size), 1)
            * max(int(self.config.gradient_accumulation_steps), 1)
        )
        logger.info(
            "Training schedule: dataset_size=%d world_size=%d per_device_batch=%d grad_accum=%d "
            "effective_global_batch=%d max_steps=%d warmup_steps=%d scheduler=%s lr=%.2e",
            len(self.train_dataset),
            world_size,
            self.config.batch_size,
            self.config.gradient_accumulation_steps,
            self.effective_global_batch_size,
            self.max_steps,
            self.warmup_steps,
            self.config.lr_scheduler_type,
            self.config.learning_rate,
        )

        # Prepare with accelerator
        if self.accelerator:
            self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = (
                self.accelerator.prepare(
                    self.model, self.optimizer, self.train_dataloader, self.lr_scheduler
                )
            )

    def _apply_freeze_strategy(self):
        """Apply the configured training strategy.

        Supported strategies:
        - ``full``: train the MoT (or action_expert) end-to-end. Everything
          else (VAE, text encoder, optional shared backbone) is frozen.
        - ``lora``: freeze the entire base model and inject LoRA adapters
          (via ``peft``) into all DiT linear layers matching
          ``config.lora_target_modules``. Action-specific heads are kept
          trainable as full-rank parameters.
        - ``staged``: phase 1 -- train only the small action heads
          (action_encoder/decoder/embedder, state_encoder, action_proj_out)
          for ``config.staged_warmup_steps`` steps; phase 2 -- unfreeze
          MoT / action_expert / backbone DiT and continue full fine-tuning.
          The phase-2 transition is triggered from the training loop via
          :meth:`_maybe_unfreeze_staged`.
        """
        strategy = getattr(self.config, "strategy", "full")
        self._staged_unfrozen = False
        if strategy == "lora":
            self._apply_lora_strategy()
            return
        if strategy == "staged":
            self._apply_staged_phase1()
            return

        # Default: freeze everything, then unfreeze MoT / action_expert.
        self.model.requires_grad_(False)
        if hasattr(self.model, "mot"):
            self.model.mot.requires_grad_(True)
        elif hasattr(self.model, "shared_dit"):
            self.model.shared_dit.requires_grad_(True)
        elif hasattr(self.model, "action_expert"):
            self.model.action_expert.requires_grad_(True)
            model_config = getattr(self.model, "config", None)
            if bool(getattr(model_config, "feature_condition_train_backbone", False)):
                dit = self.model.backbone.get_dit()
                dit.requires_grad_(True)
        else:
            # Shared-DiT WAM variants train the world DiT directly; VAE/text encoder stay frozen.
            if hasattr(self.model, "backbone"):
                dit = self.model.backbone.get_dit()
                dit.requires_grad_(True)
        # Always keep small action heads trainable.
        containers = [self.model]
        if hasattr(self.model, "shared_dit"):
            containers.append(self.model.shared_dit)
        for container in containers:
            for name in (
                "action_encoder", "action_decoder",
                "action_embedder", "action_proj_out",
                "state_encoder", "proprio_encoder",
                "feature_projector", "feature_timestep_embedding",
            ):
                mod = getattr(container, name, None)
                if mod is not None:
                    mod.requires_grad_(True)

        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            f"Strategy=full | params: {total:,} total, {trainable:,} trainable "
            f"({100*trainable/total:.1f}%)"
        )

    def _apply_lora_strategy(self):
        """Inject LoRA adapters into the DiT (and action_expert if present)."""
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as e:
            raise RuntimeError(
                "training.strategy='lora' requires the `peft` package. "
                "Install with: pip install peft"
            ) from e

        self.model.requires_grad_(False)
        target_modules = list(self.config.lora_target_modules)

        if hasattr(self.model, "mot"):
            wrap_target = self.model.mot
            wrap_attr = "mot"
        elif hasattr(self.model, "shared_dit"):
            wrap_target = self.model.shared_dit
            wrap_attr = "shared_dit"
        elif hasattr(self.model, "backbone"):
            wrap_target = self.model.backbone.get_dit()
            wrap_attr = "_lora_dit"
        else:
            raise RuntimeError(
                "Cannot apply LoRA: model has no `.mot`, `.shared_dit`, or `.backbone`."
            )

        lora_cfg = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        peft_module = get_peft_model(wrap_target, lora_cfg)
        if wrap_attr == "mot":
            self.model.mot = peft_module
        elif wrap_attr == "shared_dit":
            self.model.shared_dit = peft_module
        else:
            # peft wraps the DiT in-place; track it on the WAM model so its
            # adapter params are visible to the optimizer.
            self.model._lora_dit = peft_module

        # Keep small action heads trainable as full params.
        containers = [self.model]
        if hasattr(self.model, "shared_dit"):
            containers.append(self.model.shared_dit)
        for container in containers:
            for name in (
                "action_encoder", "action_decoder",
                "action_embedder", "action_proj_out",
                "state_encoder", "proprio_encoder",
                "feature_projector", "feature_timestep_embedding",
            ):
                mod = getattr(container, name, None)
                if mod is not None:
                    mod.requires_grad_(True)

        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            f"Strategy=lora (r={self.config.lora_r}, alpha={self.config.lora_alpha}) "
            f"| params: {total:,} total, {trainable:,} trainable "
            f"({100*trainable/total:.2f}%)"
        )

    def _apply_staged_phase1(self):
        """Phase 1 of the staged strategy: only action heads are trainable."""
        self.model.requires_grad_(False)
        unfrozen = []
        containers = [("model", self.model)]
        if hasattr(self.model, "shared_dit"):
            containers.append(("shared_dit", self.model.shared_dit))
        for prefix, container in containers:
            for name in (
                "action_encoder", "action_decoder",
                "action_embedder", "action_proj_out",
                "state_encoder", "proprio_encoder",
                "feature_projector", "feature_timestep_embedding",
            ):
                mod = getattr(container, name, None)
                if mod is not None:
                    mod.requires_grad_(True)
                    unfrozen.append(f"{prefix}.{name}")

        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        warmup = getattr(self.config, "staged_warmup_steps", 1000)
        logger.info(
            f"Strategy=staged | phase=1/2 (action heads only) | warmup_steps={warmup} "
            f"| unfrozen={unfrozen} | params: {total:,} total, {trainable:,} trainable "
            f"({100*trainable/max(total,1):.2f}%)"
        )

    def _maybe_unfreeze_staged(self):
        """If running under ``staged`` and warmup is complete, unfreeze the
        backbone / MoT / action_expert in-place. Idempotent and safe to call
        every step. Existing optimizer param groups continue holding action-
        head params; newly-unfrozen weights are added as a fresh param group
        so their grads start being applied immediately.
        """
        if getattr(self, "_staged_unfrozen", True):
            return
        if getattr(self.config, "strategy", "full") != "staged":
            return
        warmup = getattr(self.config, "staged_warmup_steps", 1000)
        if self.global_step < warmup:
            return

        added_params = []
        if hasattr(self.model, "mot"):
            for p in self.model.mot.parameters():
                if not p.requires_grad:
                    p.requires_grad_(True)
                    added_params.append(p)
        elif hasattr(self.model, "shared_dit"):
            for p in self.model.shared_dit.parameters():
                if not p.requires_grad:
                    p.requires_grad_(True)
                    added_params.append(p)
        elif hasattr(self.model, "action_expert"):
            for p in self.model.action_expert.parameters():
                if not p.requires_grad:
                    p.requires_grad_(True)
                    added_params.append(p)
        elif hasattr(self.model, "backbone"):
            dit = self.model.backbone.get_dit()
            for p in dit.parameters():
                if not p.requires_grad:
                    p.requires_grad_(True)
                    added_params.append(p)

        if added_params:
            self.optimizer.add_param_group({
                "params": added_params,
                "lr": self.optimizer.param_groups[0]["lr"],
                "weight_decay": self.config.weight_decay,
            })

        self._staged_unfrozen = True
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            f"Strategy=staged | phase=2/2 unfrozen at step={self.global_step} "
            f"| params: {total:,} total, {trainable:,} trainable "
            f"({100*trainable/max(total,1):.2f}%)"
        )

    def _unwrap(self):
        """Return the underlying model, unwrapping DDP/Accelerate wrappers."""
        if self.accelerator is not None:
            return self.accelerator.unwrap_model(self.model)
        return self.model

    def _model_training_timer(self, name: str, device: torch.device):
        timer = getattr(self._unwrap(), "training_timer", None)
        if not callable(timer):
            return nullcontext()
        return timer(name, device=device)

    def _finalize_and_collect_model_timings(
        self,
        timing_sums: dict[str, float],
        timing_counts: dict[str, int],
        *,
        synchronize: bool = False,
    ) -> None:
        """Collect timing groups, synchronizing only at a logging boundary."""
        model = self._unwrap()
        finalize = getattr(model, "finalize_training_timing_step", None)
        if callable(finalize):
            finalize()
        pop_completed = getattr(model, "pop_completed_training_timings", None)
        if not callable(pop_completed):
            return
        if synchronize and self.config.wandb_enabled:
            device = self._model_compute_param().device
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(device)
        for sample in pop_completed():
            for name, value in sample.items():
                timing_sums[name] += float(value)
                timing_counts[name] += 1

    def _model_compute_param(self):
        for param in self.model.parameters():
            if param.requires_grad:
                return param
        return next(self.model.parameters())

    def _load_model_only_if_requested(self):
        resume = getattr(self.config, "resume", None)
        if not resume or not bool(getattr(self.config, "resume_model_only", False)):
            return
        resume_dir = Path(resume)
        if not resume_dir.is_dir():
            raise FileNotFoundError(f"training.resume must point to a checkpoint directory: {resume}")
        if self.accelerator is None:
            model_path = resume_dir / "model.pt"
            if not model_path.is_file():
                raise FileNotFoundError(f"single-GPU checkpoint missing model.pt: {model_path}")
            payload = torch.load(model_path, map_location="cpu", weights_only=False)
            state = payload.get("model_state_dict", payload)
        else:
            model_path = resume_dir / "pytorch_model" / "mp_rank_00_model_states.pt"
            if not model_path.is_file():
                raise FileNotFoundError(f"DeepSpeed checkpoint missing model state: {model_path}")
            payload = torch.load(model_path, map_location="cpu", weights_only=False)
            state = payload.get("module")
            if not isinstance(state, dict):
                raise KeyError(f"No module state dict found in {model_path}")
        result = self.model.load_state_dict(state, strict=False)
        if result.missing_keys:
            logger.warning("Missing model-only resume keys, first 20: %s", result.missing_keys[:20])
        if result.unexpected_keys:
            logger.warning("Unexpected model-only resume keys, first 20: %s", result.unexpected_keys[:20])
        if self._is_main_process():
            logger.info(
                "Loaded model weights from %s; restarting optimizer/scheduler with lr=%.2e",
                resume_dir,
                self.config.learning_rate,
            )

    def _resume_if_requested(self):
        resume = getattr(self.config, "resume", None)
        if not resume or bool(getattr(self.config, "resume_model_only", False)):
            return
        resume_dir = Path(resume)
        if not resume_dir.is_dir():
            raise FileNotFoundError(f"training.resume must point to a checkpoint directory: {resume}")
        if self.accelerator is None:
            model_path = resume_dir / "model.pt"
            if not model_path.is_file():
                raise FileNotFoundError(f"single-GPU checkpoint missing model.pt: {model_path}")
            payload = torch.load(model_path, map_location="cpu")
            self.model.load_state_dict(payload["model_state_dict"])
            self.global_step = int(payload.get("step", _checkpoint_step(resume_dir)))
        else:
            self.accelerator.load_state(str(resume_dir))
            self.global_step = int(_checkpoint_step(resume_dir))
        self._resume_step = self.global_step
        if self._is_main_process():
            logger.info(f"Resumed training state from {resume_dir} at step={self.global_step}")

    def _apply_resume_lr_override_if_requested(self):
        resume_lr = getattr(self.config, "resume_lr", None)
        resume_scheduler = getattr(self.config, "resume_lr_scheduler_type", None)
        if resume_lr is None and not resume_scheduler:
            return
        if not getattr(self.config, "resume", None):
            raise ValueError("training.resume_lr/resume_lr_scheduler_type require training.resume")
        if resume_lr is not None:
            lr_value = float(resume_lr)
            for group in self.optimizer.param_groups:
                group["lr"] = lr_value
                group["initial_lr"] = lr_value
        sched_type = str(resume_scheduler or "").strip().lower()
        if sched_type:
            if sched_type != "constant":
                raise ValueError("training.resume_lr_scheduler_type currently supports only 'constant'")
            self.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)
        if self.accelerator is not None:
            self.lr_scheduler = self.accelerator.prepare(self.lr_scheduler)
        if self._is_main_process():
            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                "Applied resume LR override: lr=%.2e scheduler=%s",
                current_lr,
                sched_type or "restored",
            )

    def train(self):
        """Main training loop."""
        logger.info(f"Starting training: max_steps={self.max_steps}, batch_size={self.config.batch_size}")
        logger.info(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        logger.info(f"Effective global batch size: {self.effective_global_batch_size}")

        self.model.train()
        # Use a natural ``yield from`` infinite iterator so that accelerate's
        # DataLoaderShard can complete each epoch normally (including its
        # end-of-epoch broadcast to all ranks).  Catching StopIteration and
        # calling iter() again bypasses that broadcast, causing a NCCL hang
        # at every epoch boundary.
        def _inf_loader(dl):
            while True:
                yield from dl

        data_iter = _inf_loader(self.train_dataloader)
        model_param = self._model_compute_param()
        compute_device = model_param.device
        compute_dtype = model_param.dtype
        timing_window_started = time.perf_counter()
        timing_window_step = self.global_step
        step_timings: dict[str, float] = defaultdict(float)
        step_timing_counts: dict[str, int] = defaultdict(int)
        micro_step_in_accum = 0

        stop_step = self._target_stop_step()
        while self.global_step < stop_step:
            timer_started = time.perf_counter()
            batch = next(data_iter)
            step_timings["dataloader"] += time.perf_counter() - timer_started

            # Move to device if no accelerator
            if not self.accelerator:
                new_batch = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        v = v.to(compute_device)
                        if v.dtype.is_floating_point and v.dtype != compute_dtype:
                            v = v.to(compute_dtype)
                    new_batch[k] = v
                batch = new_batch

            # Forward + backward
            if self.accelerator:
                stepped = False
                with self.accelerator.accumulate(self.model):
                    with self._model_training_timer("forward", compute_device):
                        with self.accelerator.autocast():
                            loss, loss_dict = self._unwrap().training_step(batch)
                    self._accumulate_loss_metrics(loss_dict)

                    with self._model_training_timer("backward", compute_device):
                        self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        if not self._uses_deepspeed:
                            grad_norm = self.accelerator.clip_grad_norm_(
                                self.model.parameters(), self.config.max_grad_norm
                            )
                            self._capture_grad_norm(grad_norm)

                        with self._model_training_timer("optimizer", compute_device):
                            self.optimizer.step()
                            self.lr_scheduler.step()
                            self.optimizer.zero_grad(set_to_none=True)
                        if self._uses_deepspeed:
                            # DeepSpeed computes/clips the current gradients
                            # during optimizer.step() and only then updates its
                            # cached pre-clip global norm.
                            grad_norm = self.accelerator.clip_grad_norm_(
                                self.model.parameters(), self.config.max_grad_norm
                            )
                            self._capture_grad_norm(grad_norm)
                        self.global_step += 1
                        self._maybe_unfreeze_staged()
                        stepped = True

                # Logging / save / eval must happen OUTSIDE the
                # ``accelerator.accumulate`` context so that DDP gradient-sync
                # state and the eval-time forward passes do not interleave —
                # otherwise subsequent training steps deadlock on a stale
                # collective op.
                if not stepped:
                    continue
                self._maybe_log_train_action_mse(batch)
            else:
                # Simple single-GPU path
                with self._model_training_timer("forward", compute_device):
                    loss, loss_dict = self._unwrap().training_step(batch)
                self._accumulate_loss_metrics(loss_dict)

                with self._model_training_timer("backward", compute_device):
                    (loss / self.config.gradient_accumulation_steps).backward()
                micro_step_in_accum += 1
                if micro_step_in_accum < self.config.gradient_accumulation_steps:
                    continue

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self._capture_grad_norm(grad_norm)

                with self._model_training_timer("optimizer", compute_device):
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                micro_step_in_accum = 0
                self.global_step += 1
                self._maybe_unfreeze_staged()
                self._maybe_log_train_action_mse(batch)

            should_log = self.global_step % self.config.log_every == 0
            self._finalize_and_collect_model_timings(
                step_timings,
                step_timing_counts,
                synchronize=should_log,
            )

            if should_log:
                now = time.perf_counter()
                perf_payload = self._flush_timing_window(
                    timing_sums=step_timings,
                    timing_counts=step_timing_counts,
                    completed_steps=self.global_step - timing_window_step,
                    elapsed=now - timing_window_started,
                )
                timing_window_started = now
                timing_window_step = self.global_step
                step_timings.clear()
                step_timing_counts.clear()
                averaged_losses, train_payload = self._flush_train_metric_window()
                train_payload.update(perf_payload)
                if self._is_main_process():
                    lr = train_payload.get("train/learning_rate", 0.0)
                    logger.info(
                        f"step={self.global_step} | loss={averaged_losses['loss_total']:.4f} "
                        f"| video={averaged_losses['loss_video']:.4f} "
                        f"| action={averaged_losses['loss_action']:.4f}"
                        f"{self._format_action_monitor(averaged_losses)} "
                        f"| grad_norm={train_payload.get('train/grad_norm', float('nan')):.4f} "
                        f"| lr={lr:.2e} "
                        f"| steps/s={perf_payload['perf/steps_per_sec']:.3f} "
                        f"| samples/s={perf_payload['perf/samples_per_sec']:.1f} "
                        f"| ETA={perf_payload['perf/eta_hours']:.2f}h"
                    )
                    self._wandb_log(train_payload)

            if self.global_step % self.config.save_every == 0:
                self._save_checkpoint()

            if (
                self.val_dataset is not None
                and self.global_step % self.config.eval_every == 0
            ):
                self._run_eval()

        if self.global_step >= self.max_steps:
            logger.info(f"Training complete. Total steps: {self.global_step}")
        else:
            logger.info(
                f"Debug stop reached at step={self.global_step} "
                f"(scheduler max_steps={self.max_steps})"
            )

    @torch.no_grad()
    def _maybe_log_train_action_mse(self, batch: dict) -> None:
        """Denoise actions for the current training batch and log their MSE."""
        interval = self.config.train_action_mse_every
        if interval is None or self.global_step % interval != 0:
            return

        was_training = self.model.training
        self.model.eval()
        try:
            video = batch["video"]
            target_action = batch["action"]
            context = batch["context"]
            context_mask = batch.get("context_mask")
            action_is_pad = batch.get("action_is_pad")
            proprio = batch.get("proprio")

            model = self._unwrap()
            action_steps = int(
                self.config.eval_action_num_inference_steps
                or self.config.eval_num_inference_steps
            )
            inference_proprio = proprio
            if inference_proprio is not None and inference_proprio.ndim == 3:
                inference_proprio = inference_proprio[:, 0, :]

            inference_kwargs = {
                "proprio": inference_proprio,
                "num_video_frames": video.shape[2],
            }
            num_inference_steps = action_steps
            if hasattr(model, "shared_dit"):
                num_inference_steps = self.config.eval_num_inference_steps
                inference_kwargs["action_num_inference_steps"] = action_steps

            autocast_context = (
                self.accelerator.autocast if self.accelerator is not None else nullcontext
            )
            with autocast_context():
                pred_action = model.infer_action(
                    video[:, :, 0],
                    context,
                    context_mask,
                    action_horizon=target_action.shape[1],
                    num_inference_steps=num_inference_steps,
                    seed=self.config.seed,
                    **inference_kwargs,
                )
            mse = action_mse(pred_action, target_action, is_pad=action_is_pad)["mse"]

            # Weight each rank by its number of valid action scalars before reducing.
            if action_is_pad is None:
                local_count = pred_action.new_tensor(pred_action.numel(), dtype=torch.float32)
            else:
                local_count = (~action_is_pad).sum().to(torch.float32) * pred_action.shape[-1]
            mse_sum_and_count = torch.stack((mse.float() * local_count, local_count))
            if self.accelerator is not None:
                mse_sum_and_count = self.accelerator.reduce(mse_sum_and_count, reduction="sum")
            train_action_mse = float(
                (mse_sum_and_count[0] / mse_sum_and_count[1].clamp_min(1.0)).item()
            )
        finally:
            if was_training:
                self.model.train()

        if self._is_main_process():
            logger.info(
                "[train sample] step=%d | action_mse=%.6f | inference_steps=%d",
                self.global_step,
                train_action_mse,
                action_steps,
            )
            self._wandb_log({"train/action_mse": train_action_mse})

    def _target_stop_step(self) -> int:
        debug_stop = getattr(self.config, "debug_stop_after_steps", None)
        if debug_stop is None:
            return self.max_steps
        if debug_stop <= 0:
            raise ValueError("training.debug_stop_after_steps must be positive when set")
        return min(self.max_steps, (self._resume_step or self.global_step) + int(debug_stop))

    def _is_main_process(self) -> bool:
        """Return True if this is the rank-0 process (or no accelerator)."""
        if self.accelerator is None:
            return True
        return getattr(self.accelerator, "is_main_process", True)

    def _save_checkpoint(self):
        """Save model checkpoint."""
        save_dir = os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}")
        os.makedirs(save_dir, exist_ok=True)

        if self.accelerator:
            # `save_state` internally synchronises across ranks; only rank-0
            # writes the meta files but all ranks must call into it.
            self.accelerator.save_state(save_dir)
            self.accelerator.wait_for_everyone()
        else:
            torch.save(
                {"model_state_dict": self.model.state_dict(), "step": self.global_step},
                os.path.join(save_dir, "model.pt"),
            )
        if self._is_main_process():
            with open(os.path.join(save_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
                json.dump({"global_step": self.global_step}, f, indent=2)
            logger.info(f"Saved checkpoint to {save_dir}")
            if self._checkpoint_uploader is not None:
                self._checkpoint_uploader.submit(Path(save_dir), self.global_step)
            self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        """Keep only the last ``save_total_limit`` checkpoints (by step)."""
        limit = self.config.save_total_limit
        if not limit or limit <= 0:
            return
        out_dir = self.config.output_dir
        if not os.path.isdir(out_dir):
            return
        ckpts = []
        for name in os.listdir(out_dir):
            match = re.fullmatch(r"checkpoint-(\d+)", name)
            if match:
                ckpts.append((int(match.group(1)), os.path.join(out_dir, name)))
        ckpts.sort(key=lambda item: item[0])
        to_delete = ckpts[: max(0, len(ckpts) - limit)]
        if self._checkpoint_uploader is not None:
            # Never race a background upload. A remote-enabled checkpoint
            # becomes eligible for local retention cleanup only after the
            # verified-upload marker has been written atomically.
            from starwam.tools.checkpoint_tos.backend import TOS_UPLOAD_MARKER

            to_delete = [
                item
                for item in to_delete
                if os.path.isfile(os.path.join(item[1], TOS_UPLOAD_MARKER))
            ]
        for _, path in to_delete:
            shutil.rmtree(path)
            logger.info("Removed old checkpoint: %s", path)

    def close(self) -> None:
        """Flush checkpoint uploads and close the W&B run."""
        if self._checkpoint_uploader is not None:
            uploader = self._checkpoint_uploader
            self._checkpoint_uploader = None
            logger.info("Waiting for queued TOS checkpoint uploads to finish")
            uploader.close()
        if self._wandb_run is not None:
            run = self._wandb_run
            self._wandb_run = None
            try:
                run.finish()
            except Exception as exc:  # noqa: BLE001
                logger.warning("wandb.finish failed: %s", exc)

    def _run_eval(self):
        """Evaluate the model on a small slice of val_dataset.

        ALL ranks run eval simultaneously (no barriers, no rank-0-only path).
        This is critical: any barrier asymmetry between rank 0 and other ranks
        desynchronises the NCCL sequence counter, causing the next epoch-start
        ``synchronize_rng_states`` broadcast (NumelIn=5056) to hang.
        Only rank 0 logs the results.
        """
        from starwam.training.metrics import action_dim_mse, action_mse, video_psnr

        if self.val_dataset is None:
            return

        was_training = self.model.training
        self.model.eval()
        try:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                drop_last=False,
            )
            max_samples = max(1, getattr(self.config, "eval_max_samples", 4))
            compute_video = getattr(self.config, "eval_compute_video_psnr", False)
            mse_list: list[float] = []
            mse_per_dim: list[torch.Tensor] = []
            pred_gripper_values: list[torch.Tensor] = []
            target_gripper_values: list[torch.Tensor] = []
            psnr_list: list[float] = []
            n_seen = 0
            model_param = self._model_compute_param()
            device = model_param.device
            model_dtype = model_param.dtype

            for vb in val_loader:
                if n_seen >= max_samples:
                    break
                batch = {}
                for k, v in vb.items():
                    if isinstance(v, torch.Tensor):
                        v = v.to(device)
                        if v.dtype.is_floating_point and v.dtype != model_dtype:
                            v = v.to(model_dtype)
                    batch[k] = v
                video = batch["video"]
                action_gt = batch["action"]
                ctx = batch["context"]
                cmask = batch.get("context_mask")
                is_pad = batch.get("action_is_pad")
                proprio = batch.get("proprio")
                T_a = action_gt.shape[1]
                first_frame = video[:, :, 0]
                eval_extra_kwargs = {
                    "proprio": None if proprio is None else proprio[:, 0, :],
                }
                if hasattr(self._unwrap(), "shared_dit"):
                    eval_extra_kwargs["action_num_inference_steps"] = int(
                        self.config.eval_action_num_inference_steps or self.config.eval_num_inference_steps
                    )
                ac_ctx = self.accelerator.autocast() if self.accelerator is not None else nullcontext()
                with torch.no_grad(), ac_ctx:
                    pred_a = self._unwrap().infer_action(
                        first_frame, ctx, cmask,
                        action_horizon=T_a,
                        num_inference_steps=self.config.eval_num_inference_steps,
                        seed=self.config.seed,
                        num_video_frames=video.shape[2],
                        **eval_extra_kwargs,
                    )
                target_a = action_gt[:, :T_a]
                am = action_mse(pred_a, target_a, is_pad=is_pad)
                mse_list.append(float(am["mse"].item()))
                mse_per_dim.append(action_dim_mse(pred_a, target_a, is_pad=is_pad).detach().cpu())
                if is_pad is None:
                    pred_gripper_values.append(pred_a[..., -1].detach().float().cpu().reshape(-1))
                    target_gripper_values.append(target_a[..., -1].detach().float().cpu().reshape(-1))
                else:
                    keep = (~is_pad).detach().cpu()
                    pred_gripper_values.append(pred_a[..., -1].detach().float().cpu()[keep])
                    target_gripper_values.append(target_a[..., -1].detach().float().cpu()[keep])
                if compute_video:
                    ac_ctx2 = self.accelerator.autocast() if self.accelerator is not None else nullcontext()
                    with torch.no_grad(), ac_ctx2:
                        out = self._unwrap().infer_joint(
                            first_frame, ctx, cmask,
                            num_video_frames=video.shape[2],
                            action_horizon=T_a,
                            num_inference_steps=self.config.eval_num_inference_steps,
                            seed=self.config.seed,
                            **eval_extra_kwargs,
                        )
                    pv = out["video"]
                    psnr_list.append(float(
                        video_psnr(pv, video[:, :, : pv.shape[2]]).item()
                    ))
                n_seen += 1

            # Only rank 0 logs — no distributed ops needed
            if self._is_main_process():
                if not mse_list:
                    logger.warning("[eval] no samples produced metrics")
                else:
                    avg_mse = sum(mse_list) / len(mse_list)
                    log_msg = f"[eval] step={self.global_step} | action_mse={avg_mse:.4f} (n={len(mse_list)})"
                    wandb_payload = {"eval/action_mse": avg_mse}
                    if psnr_list:
                        avg_psnr = sum(psnr_list) / len(psnr_list)
                        log_msg += f" | video_psnr={avg_psnr:.2f}dB"
                        wandb_payload["eval/video_psnr"] = avg_psnr
                    if mse_per_dim:
                        dim_mse = torch.stack(mse_per_dim, dim=0).mean(dim=0)
                        eef_mse = dim_mse[:-1].mean().item()
                        gripper_mse = dim_mse[-1].item()
                        log_msg += f" | action_eef_mse={eef_mse:.4f} | gripper_mse={gripper_mse:.4f}"
                        wandb_payload["eval/action_eef_mse"] = float(eef_mse)
                        wandb_payload["eval/action_gripper_mse"] = float(gripper_mse)
                    if pred_gripper_values:
                        pred_gripper = torch.cat(pred_gripper_values)
                        target_gripper = torch.cat(target_gripper_values)
                        wandb_payload["eval/pred_gripper_open_rate"] = float((pred_gripper > 0).float().mean().item())
                        wandb_payload["eval/target_gripper_open_rate"] = float((target_gripper > 0).float().mean().item())
                    logger.info(log_msg)
                    self._wandb_log(wandb_payload)
        except Exception as e:  # noqa: BLE001
            if self._is_main_process():
                logger.warning(f"[eval] skipped due to error: {e}")
        finally:
            if was_training:
                self.model.train()
