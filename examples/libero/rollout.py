"""LIBERO environment rollout for StarWAM policies."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from starwam.config import load_config  # noqa: E402
from starwam.data.lerobot import (  # noqa: E402
    DEFAULT_TEXT_CACHE_ENCODER_ID,
    DEFAULT_TEXT_PROMPT,
    format_text_prompt,
    iter_task_records,
    load_text_cache,
    save_text_cache,
    load_lerobot_stats,
    text_cache_path,
)
from starwam.utils.config_cli import apply_overrides  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("starwam.rollout_libero")

LIBERO_DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
LIBERO_ENV_RESOLUTION = 256


def _latest_checkpoint(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    candidates: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* directories found under {output_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def _strip_known_prefixes(state_dict: dict[str, Any]) -> dict[str, Any]:
    prefixes = ("module.", "model.", "_orig_mod.")
    out = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        out[new_key] = value
    return out


def _extract_checkpoint_state(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    for key in ("model_state_dict", "module", "state_dict"):
        state = payload.get(key)
        if isinstance(state, dict):
            return state
    return payload


def _checkpoint_meta(payload: Any, checkpoint_file: str | Path | None = None) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    if checkpoint_file is not None:
        meta["checkpoint_file"] = str(checkpoint_file)
    if isinstance(payload, dict):
        for key in ("global_steps", "global_step", "step", "ds_version"):
            if key in payload:
                meta[key] = payload[key]
    return meta


def _load_model_state(model: torch.nn.Module, state: dict[str, Any]) -> None:
    state = _strip_known_prefixes(state)
    result = model.load_state_dict(state, strict=False)
    logger.info(
        "Loaded checkpoint tensors: tensors=%d missing=%d unexpected=%d",
        len(state),
        len(result.missing_keys),
        len(result.unexpected_keys),
    )
    if result.missing_keys:
        logger.warning("Missing checkpoint keys, first 20: %s", result.missing_keys[:20])
    if result.unexpected_keys:
        logger.warning("Unexpected checkpoint keys, first 20: %s", result.unexpected_keys[:20])


def _load_checkpoint(model: torch.nn.Module, checkpoint: str | Path) -> dict[str, Any]:
    checkpoint = Path(checkpoint)
    if checkpoint.is_dir():
        for name in ("model.pt", "pytorch_model.bin", "model.bin", "pytorch_model/mp_rank_00_model_states.pt"):
            path = checkpoint / name
            if not path.is_file():
                continue
            payload = torch.load(path, map_location="cpu", weights_only=False)
            state = _extract_checkpoint_state(payload)
            _load_model_state(model, state)
            return _checkpoint_meta(payload, path)

        safetensors_files = sorted(checkpoint.glob("*.safetensors"))
        if safetensors_files:
            from safetensors.torch import load_file

            state = {}
            for path in safetensors_files:
                state.update(load_file(str(path), device="cpu"))
            _load_model_state(model, state)
            return {"checkpoint_files": [str(path) for path in safetensors_files]}

        raise FileNotFoundError(
            f"Unsupported checkpoint directory {checkpoint}. Expected model.pt, pytorch_model.bin, or *.safetensors."
        )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = _extract_checkpoint_state(payload)
    _load_model_state(model, state)
    return _checkpoint_meta(payload, checkpoint)


def _patch_torch_load_for_libero_init_states() -> None:
    original_load = torch.load
    if getattr(original_load, "_starwam_libero_compat", False):
        return

    def load_with_legacy_default(*args: Any, **kwargs: Any):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    load_with_legacy_default._starwam_libero_compat = True  # type: ignore[attr-defined]
    torch.load = load_with_legacy_default  # type: ignore[assignment]


def _add_libero_to_path(libero_home: str | None) -> None:
    if not libero_home:
        return
    path = Path(libero_home).expanduser().resolve()
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))
    default_config_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "starwam" / "libero_config"
    config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", str(default_config_dir)))
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    config_file = config_dir / "config.yaml"
    benchmark_root = path / "libero" / "libero"
    if not config_file.exists() and benchmark_root.is_dir():
        import yaml

        with open(config_file, "w", encoding="utf-8") as f:
            yaml.safe_dump({
                "benchmark_root": str(benchmark_root),
                "bddl_files": str(benchmark_root / "bddl_files"),
                "init_states": str(benchmark_root / "init_files"),
                "datasets": str(path / "libero" / "datasets"),
                "assets": str(benchmark_root / "assets"),
            }, f)


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = math.sqrt(max(0.0, 1.0 - float(quat[3]) * float(quat[3])))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * (2.0 * math.acos(float(quat[3]))) / den).astype(np.float32)


def _extract_proprio(obs: dict[str, Any], proprio_dim: int | None) -> torch.Tensor | None:
    if not proprio_dim or proprio_dim <= 0:
        return None
    state = np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            _quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float32)),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        ],
        axis=0,
    )
    if state.shape[0] < proprio_dim:
        raise ValueError(f"LIBERO proprio dim {state.shape[0]} is smaller than configured proprio_dim={proprio_dim}")
    return torch.as_tensor(state[:proprio_dim], dtype=torch.float32).view(1, proprio_dim)


def _resize_rgb(image: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    height, width = size_hw
    return np.asarray(Image.fromarray(image).resize((width, height), resample=Image.BILINEAR), dtype=np.uint8)


def _obs_to_images(obs: dict[str, Any], config: Any) -> dict[str, np.ndarray]:
    primary = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    size_hw = tuple(int(x) for x in config.data.video_size)
    primary = _resize_rgb(primary, size_hw)
    wrist = _resize_rgb(wrist, size_hw)

    video_keys = list(getattr(config.data, "video_keys", []) or [getattr(config.data, "video_key", "observation.images.image")])
    if len(video_keys) >= 2:
        if getattr(config.data, "concat_multi_camera", "horizontal") == "vertical":
            image = np.concatenate([primary, wrist], axis=0)
        else:
            image = np.concatenate([primary, wrist], axis=1)
    else:
        image = primary
    return {"image": primary, "wrist_image": wrist, "concat": image}


def _obs_to_image(obs: dict[str, Any], config: Any) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
    images = _obs_to_images(obs, config)
    image = images["concat"]
    tensor = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor * (2.0 / 255.0) - 1.0
    return tensor, images


def _load_action_stats(config: Any) -> dict[str, torch.Tensor] | None:
    if not getattr(config.data, "normalize_actions", False):
        return None
    stats_path = getattr(config.data, "action_stats_path", None)
    if not stats_path:
        raise ValueError("data.normalize_actions=true requires data.action_stats_path for rollout denormalization")
    stats = load_lerobot_stats(stats_path)
    if "action" not in stats:
        raise KeyError(f"No action stats found in {stats_path}")
    return stats["action"]


def _load_state_stats(config: Any) -> dict[str, torch.Tensor] | None:
    if not getattr(config.data, "normalize_states", False):
        return None
    stats_path = getattr(config.data, "state_stats_path", None) or getattr(config.data, "action_stats_path", None)
    if not stats_path:
        raise ValueError("data.normalize_states=true requires data.state_stats_path or data.action_stats_path")
    stats = load_lerobot_stats(stats_path)
    if "state" not in stats:
        raise KeyError(f"No state stats found in {stats_path}")
    return stats["state"]


def _stat_tensor(stats: dict[str, torch.Tensor], key: str, dim: int, dtype: torch.dtype) -> torch.Tensor:
    value = stats[key].to(dtype)
    if value.numel() < dim:
        raise ValueError(f"state stats {key} dim {value.numel()} is smaller than proprio dim {dim}")
    return value[:dim]


def _normalize_state(proprio: torch.Tensor, config: Any, stats: dict[str, torch.Tensor] | None) -> torch.Tensor:
    if stats is None:
        return proprio
    mode = getattr(config.data, "state_norm_mode", "minmax")
    dim = int(proprio.shape[-1])
    if mode == "zscore":
        mean = _stat_tensor(stats, "mean", dim, proprio.dtype)
        std = _stat_tensor(stats, "std", dim, proprio.dtype).clamp_min(1e-6)
        return ((proprio - mean) / std).clamp(-5.0, 5.0)
    if mode != "minmax":
        raise ValueError(f"Unsupported state_norm_mode={mode!r}")
    state_min = _stat_tensor(stats, "min", dim, proprio.dtype)
    state_max = _stat_tensor(stats, "max", dim, proprio.dtype)
    normalized = 2.0 * (proprio - state_min) / (state_max - state_min).clamp_min(1e-6) - 1.0
    return normalized.clamp(-5.0, 5.0)


def _denormalize_action(action: torch.Tensor, config: Any, stats: dict[str, torch.Tensor] | None) -> np.ndarray:
    action = action.detach().float().cpu()
    if stats is None:
        denorm = action
    elif config.data.action_norm_mode == "zscore":
        denorm = action * stats["std"].clamp_min(1e-6) + stats["mean"]
    elif config.data.action_norm_mode == "minmax":
        action_min = stats["min"]
        action_max = stats["max"]
        denorm = (action.clamp(-1.0, 1.0) + 1.0) * 0.5 * (action_max - action_min).clamp_min(1e-6) + action_min
    else:
        raise ValueError(f"Unsupported action_norm_mode={config.data.action_norm_mode!r}")

    out = denorm.numpy()
    if out.ndim == 3:
        out = out[0]
    if out.shape[-1] != 7:
        raise ValueError(f"LIBERO rollout expects 7-D actions, got shape={out.shape}")

    gripper_open = out[..., -1] > 0.5
    out[..., -1] = np.where(gripper_open, -1.0, 1.0)
    return out.astype(np.float32)


def _build_task_cache_index(config: Any) -> dict[str, Path]:
    index: dict[str, Path] = {}
    cache_dir = getattr(config.data, "text_embedding_cache_dir", None)
    if not cache_dir:
        return index
    roots = list(config.data.dataset_dirs) if config.data.dataset_dirs else ([config.data.root] if config.data.root else [])
    prompt_template = getattr(config.data, "text_prompt_template", None) or DEFAULT_TEXT_PROMPT
    encoder_id = getattr(config.data, "text_cache_encoder_id", None) or DEFAULT_TEXT_CACHE_ENCODER_ID
    context_len = int(getattr(config.data, "text_len", 128))
    for root in roots:
        for record in iter_task_records(root):
            task = str(record["task"])
            cache = text_cache_path(cache_dir, task, context_len, prompt_template, encoder_id)
            if cache.is_file():
                index[task] = cache
    return index


def _load_context(task_description: str, config: Any, task_cache: dict[str, Path], model: torch.nn.Module, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    text_len = int(getattr(config.data, "text_len", 128))
    prompt_template = getattr(config.data, "text_prompt_template", None) or DEFAULT_TEXT_PROMPT
    encoder_id = getattr(config.data, "text_cache_encoder_id", None) or DEFAULT_TEXT_CACHE_ENCODER_ID
    cache_dir = getattr(config.data, "text_embedding_cache_dir", None)
    cache = task_cache.get(task_description)
    if cache is None and cache_dir:
        cache = text_cache_path(cache_dir, task_description, text_len, prompt_template, encoder_id)
    text_dim = int(getattr(getattr(getattr(model, "backbone", None), "info", None), "text_dim", 4096))
    if cache is not None and cache.is_file():
        context, mask = load_text_cache(cache, text_len, text_dim)
        return context.unsqueeze(0).to(device=device, dtype=dtype), mask.unsqueeze(0).to(device=device)

    if not cache_dir:
        raise KeyError("data.text_embedding_cache_dir must be set for rollout text conditioning")
    from starwam.backbone.wan22 import Wan22TextEncoder

    model_dir = Path(config.backbone.pretrained_model_id)
    encoder = Wan22TextEncoder(
        ckpt_path=str(model_dir / "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=str(model_dir / "google" / "umt5-xxl"),
        text_len=text_len,
        device=str(device),
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    prompt = format_text_prompt(task_description, prompt_template)
    with torch.no_grad():
        context, mask = encoder.encode([prompt])
    cache = text_cache_path(cache_dir, task_description, text_len, prompt_template, encoder_id)
    save_text_cache(cache, context[0], mask[0], prompt, task_description)
    task_cache[task_description] = cache
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return context.to(device=device, dtype=dtype), mask.to(device=device)


def _max_steps(task_suite_name: str) -> int:
    return {
        "libero_spatial": 400,
        "libero_object": 400,
        "libero_goal": 400,
        "libero_10": 700,
        "libero_90": 700,
    }.get(task_suite_name, 400)


def _safe_task_name(task_description: str) -> str:
    return "_".join(task_description.lower().replace(".", " ").split())[:80]


def _save_video(path: Path, frames: list[np.ndarray], fps: int = 10) -> None:
    if not frames:
        return
    import imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps)


def _sampled_video_frame_count(config: Any) -> int:
    action_grid_frames = int(config.data.num_frames)
    action_freq_ratio = max(1, int(getattr(config.data, "action_freq_ratio", 1)))
    return len(range(0, action_grid_frames, action_freq_ratio))


def _uses_decoupled_action_steps(config: Any) -> bool:
    return str(getattr(config.framework, "type", "")) == "shared_dit"


def _predict_action_chunk(
    model: torch.nn.Module,
    obs: dict[str, Any],
    task_description: str,
    config: Any,
    task_cache: dict[str, Path],
    action_stats: dict[str, torch.Tensor] | None,
    state_stats: dict[str, torch.Tensor] | None,
    device: torch.device,
    dtype: torch.dtype,
    num_inference_steps: int,
    action_num_inference_steps: int,
    seed: int | None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    image, images = _obs_to_image(obs, config)
    context, context_mask = _load_context(task_description, config, task_cache, model, device, dtype)
    proprio = _extract_proprio(obs, getattr(config.framework, "proprio_dim", None))
    if proprio is not None:
        proprio = _normalize_state(proprio, config, state_stats).to(device=device, dtype=dtype)

    infer_kwargs: dict[str, Any] = {
        "proprio": proprio,
        "num_video_frames": _sampled_video_frame_count(config),
    }
    if _uses_decoupled_action_steps(config):
        infer_kwargs["action_num_inference_steps"] = action_num_inference_steps

    pred = model.infer_action(
        input_image=image.to(device=device, dtype=dtype),
        context=context,
        context_mask=context_mask,
        action_horizon=int(config.framework.chunk_size),
        num_inference_steps=num_inference_steps,
        seed=seed,
        **infer_kwargs,
    )
    return _denormalize_action(pred, config, action_stats), images


def _rollout_episode(
    env: Any,
    initial_state: Any,
    task_description: str,
    model: torch.nn.Module,
    config: Any,
    task_cache: dict[str, Path],
    action_stats: dict[str, torch.Tensor] | None,
    state_stats: dict[str, torch.Tensor] | None,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
    episode_idx: int,
) -> tuple[bool, list[np.ndarray]]:
    env.reset()
    obs = env.set_init_state(initial_state)
    pending_actions: list[list[float]] = []
    frames: list[np.ndarray] = []
    done = False
    max_steps = int(args.max_steps or _max_steps(args.task_suite_name))

    for t in range(max_steps + args.num_steps_wait):
        if t < args.num_steps_wait:
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            continue

        frames.append(_obs_to_images(obs, config)["concat"])

        if not pending_actions:
            action_chunk, _ = _predict_action_chunk(
                model=model,
                obs=obs,
                task_description=task_description,
                config=config,
                task_cache=task_cache,
                action_stats=action_stats,
                state_stats=state_stats,
                device=device,
                dtype=dtype,
                num_inference_steps=args.num_inference_steps,
                action_num_inference_steps=args.action_num_inference_steps,
                seed=args.seed if args.fixed_seed else (None if args.seed is None else args.seed + episode_idx),
            )
            pending_actions = action_chunk[: args.replan_steps].tolist()

        obs, _, done, _ = env.step(pending_actions.pop(0))
        if done:
            break
    return bool(done), frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Run StarWAM policy rollout in LIBERO environments")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--libero-home", default=os.environ.get("LIBERO_HOME"))
    parser.add_argument("--task-suite-name", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=None, help="Run one task id; default runs all tasks in suite")
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--action-num-inference-steps", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixed-seed", action="store_true", help="Use the same diffusion seed for every episode")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    _add_libero_to_path(args.libero_home)
    _patch_torch_load_for_libero_init_states()
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to import LIBERO. Set LIBERO_HOME to the LIBERO source root and make sure its env deps are installed."
        ) from exc

    config = load_config(args.config)
    if args.override:
        config = apply_overrides(config, args.override)

    checkpoint = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint(config.training.output_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config.training.output_dir) / "libero_rollout" / checkpoint.name / args.task_suite_name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    mp = (config.training.mixed_precision or "no").lower()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(mp, torch.float32)
    if device.type == "cpu":
        dtype = torch.float32

    if args.num_inference_steps is None:
        args.num_inference_steps = int(getattr(config.inference, "num_inference_steps", 8))
    if args.action_num_inference_steps is None:
        args.action_num_inference_steps = int(getattr(config.inference, "action_num_inference_steps", args.num_inference_steps))
    if not _uses_decoupled_action_steps(config):
        args.action_num_inference_steps = args.num_inference_steps

    logger.info("Config: %s", args.config)
    logger.info("Checkpoint: %s", checkpoint)
    logger.info("Output: %s", output_dir)
    logger.info("Task suite: %s", args.task_suite_name)
    logger.info(
        "Inference steps: num=%d action=%d decoupled_action_steps=%s sampled_video_frames=%d",
        args.num_inference_steps,
        args.action_num_inference_steps,
        _uses_decoupled_action_steps(config),
        _sampled_video_frame_count(config),
    )

    from starwam import build_framework

    model = build_framework(config, device=str(device), dtype=dtype).to(device)
    meta = _load_checkpoint(model, checkpoint)
    model.eval()
    logger.info("Loaded checkpoint metadata: %s", meta)

    task_cache = _build_task_cache_index(config)
    logger.info("Loaded %d task text embeddings from recipe dataset dirs", len(task_cache))
    action_stats = _load_action_stats(config)
    state_stats = _load_state_stats(config)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task_ids = [args.task_id] if args.task_id is not None else list(range(task_suite.n_tasks))

    all_results: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "task_suite_name": args.task_suite_name,
        "num_trials": args.num_trials,
        "task_results": {},
    }
    total_success = 0
    total_trials = 0

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        task_description = task.language
        initial_states = task_suite.get_task_init_states(task_id)
        task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=str(task_bddl_file),
            camera_heights=LIBERO_ENV_RESOLUTION,
            camera_widths=LIBERO_ENV_RESOLUTION,
        )
        env.seed(args.seed)

        task_success = 0
        task_records = []
        logger.info("Task %s/%s: %s", task_id, task_suite.n_tasks - 1, task_description)
        try:
            for trial_idx in range(args.num_trials):
                success, frames = _rollout_episode(
                    env=env,
                    initial_state=initial_states[trial_idx],
                    task_description=task_description,
                    model=model,
                    config=config,
                    task_cache=task_cache,
                    action_stats=action_stats,
                    state_stats=state_stats,
                    device=device,
                    dtype=dtype,
                    args=args,
                    episode_idx=trial_idx,
                )
                task_success += int(success)
                total_success += int(success)
                total_trials += 1
                record = {"trial": trial_idx, "success": bool(success)}
                task_records.append(record)
                logger.info("Task %d trial %d success=%s", task_id, trial_idx, success)
                if args.save_video:
                    suffix = "success" if success else "failure"
                    video_path = output_dir / "videos" / f"task{task_id:02d}_trial{trial_idx:02d}_{suffix}_{_safe_task_name(task_description)}.mp4"
                    _save_video(video_path, frames)
        finally:
            env.close()

        task_rate = task_success / max(args.num_trials, 1)
        all_results["task_results"][str(task_id)] = {
            "task_description": task_description,
            "successes": task_success,
            "trials": args.num_trials,
            "success_rate": task_rate,
            "episodes": task_records,
        }
        logger.info("Task %d success_rate=%.4f (%d/%d)", task_id, task_rate, task_success, args.num_trials)

    all_results["total_successes"] = total_success
    all_results["total_trials"] = total_trials
    all_results["success_rate"] = total_success / max(total_trials, 1)
    result_path = output_dir / "results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Total success_rate=%.4f (%d/%d)", all_results["success_rate"], total_success, total_trials)
    logger.info("Saved rollout results to %s", result_path)


if __name__ == "__main__":
    main()
