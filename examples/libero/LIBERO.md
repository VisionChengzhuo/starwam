# LIBERO Examples

This document describes the LIBERO workflow for StarWAM: environment setup, recipe placeholders, preprocessing, training, and rollout/evaluation.

The generic StarWAM package documentation is in the repository root [README.md](../../README.md).

## 1. Environment

Use one Conda environment for training and LIBERO rollout.

```bash
conda create -n starwam-libero python=3.11 -y
conda activate starwam-libero

# Install the PyTorch wheel matching your CUDA setup.
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

pip install -r examples/libero/requirements.txt
pip install flash-attn --no-build-isolation
pip install -e .
```

Install LIBERO from source in the same environment:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO && pip install -e . --no-deps
```

Notes:

- `--no-deps` avoids changing versions pinned by `examples/libero/requirements.txt`.
- Install a CUDA/PyTorch-compatible `flash-attn` version if the default one fails.
- On headless servers, set `export MUJOCO_GL=egl` if rendering fails.
- Launch scripts default to `CONDA_ENV=starwam-libero`.

## 2. Recipes

Current LIBERO recipes are under:

```text
examples/libero/configs/recipes/
```

| Recipe | Model family | Backbone | Notes |
| --- | --- | --- | --- |
| `starwam_libero_mot_wan22_5b.yaml` | `mot_wam` | Wan2.2-TI2V-5B | Fast-WAM-aligned MoT recipe. Requires a preprocessed ActionDiT init payload. |
| `starwam_libero_mot_cosmos_predict2.yaml` | `mot_wam` | Cosmos-Predict2-2B-Video2World | MoT recipe with Cosmos-Predict2 backbone. Requires a preprocessed Cosmos ActionDiT init payload. |
| `starwam_libero_shared_dit_wan22_5b.yaml` | `shared_dit_wam` | Wan2.2-TI2V-5B | Shared-DiT/register-token recipe with decoupled video/action inference steps. |
| `starwam_libero_feature_conditioned_wan22_5b.yaml` | `feature_conditioned_action_model` | Wan2.2-TI2V-5B | Feature-conditioned action model: a single Wan DiT forward extracts observation tokens and a randomly initialized ActionDiT predicts actions. No preprocessed ActionDiT init required. |

### Wan2.2-TI2V-5B LIBERO results

Reported success rates (50 trials/task, 10 tasks/suite) for the three
Wan2.2-TI2V-5B recipes. Values are copied from each recipe header.

| Suite | MoT (`mot_wam`) | Shared-DiT (`shared_dit_wam`) | Feature-Conditioned (`feature_conditioned_action_model`) |
| --- | --- | --- | --- |
| libero_spatial | 97.8% | 98.8% | 90.8% |
| libero_object | 98.8% | 100.0% | 95.8% |
| libero_goal | 97.2% | 97.4% | 94.0% |
| libero_10 | 94.2% | 96.4% | 81.2% |
| Overall (micro) | 97.0% | 98.2% | 90.5% |
| Eval checkpoint | checkpoint-20000 | checkpoint-50000 | checkpoint-100000 |

### Download data and backbones

```bash
pip install -U "huggingface_hub[cli]"

# LIBERO LeRobot datasets
hf download IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot --repo-type dataset --local-dir /path/to/libero_spatial_lerobot
hf download IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot  --repo-type dataset --local-dir /path/to/libero_object_lerobot
hf download IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot    --repo-type dataset --local-dir /path/to/libero_goal_lerobot
hf download IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot      --repo-type dataset --local-dir /path/to/libero_10_lerobot

# Backbones
hf download Wan-AI/Wan2.2-TI2V-5B --local-dir /path/to/Wan2.2-TI2V-5B
hf download nvidia/Cosmos-Predict2-2B-Video2World --local-dir /path/to/Cosmos-Predict2-2B-Video2World
```

Set `data.dataset_dirs` to the four LIBERO dirs above, and set `backbone.pretrained_model_id` to the selected backbone dir.

## 3. Paths you must set

The release recipes use placeholder paths. Before running training or rollout, replace them in the YAML file or pass values through `--override`.

| Field | Required for | What to set |
| --- | --- | --- |
| `backbone.pretrained_model_id` | all recipes | Local Wan2.2 or Cosmos-Predict2 checkpoint directory. Download/prepare this yourself. |
| `framework.action_expert_init_from` | Wan2.2 MoT and Cosmos-Predict2 MoT | Output of Section 5.1 (`preprocess_action_dit_init`). Not needed for Shared-DiT or the feature-conditioned recipe (both leave it `null`). |
| `training.output_dir` | all recipes | Run output directory. Checkpoints, logs, stats, and caches are written here. |
| `data.dataset_dirs` | all real LIBERO runs | LeRobot-format LIBERO dataset dirs. Set in YAML or pass a quoted list through `--override`. |
| `data.text_embedding_cache_dir` | all real LIBERO runs | Text embedding cache dir. Training creates missing caches; Wan users may also precompute via Section 5.2. |
| `data.action_stats_path` | normalized-action recipes | Action stats JSON. Created from `data.dataset_dirs` if missing. |
| `data.state_stats_path` | normalized-state recipes | State/proprio stats JSON. Can share the same file as `data.action_stats_path`. |

The `--override` parser supports scalar `key=value` overrides and quoted Python/JSON-style lists.

Example overrides:

```bash
--override \
  'data.dataset_dirs=["/path/to/libero_spatial_lerobot","/path/to/libero_object_lerobot","/path/to/libero_goal_lerobot","/path/to/libero_10_lerobot"]' \
  backbone.pretrained_model_id=/path/to/Wan2.2-TI2V-5B \
  framework.action_expert_init_from=/path/to/preprocessed/starwam_action_dit_init_wan22.pt \
  training.output_dir=/path/to/output/starwam_libero_mot_wan22_5b \
  data.text_embedding_cache_dir=/path/to/output/starwam_libero_mot_wan22_5b/text_embedding_cache \
  data.action_stats_path=/path/to/output/starwam_libero_mot_wan22_5b/action_stats.json \
  data.state_stats_path=/path/to/output/starwam_libero_mot_wan22_5b/action_stats.json \
  training.wandb_enabled=false
```

## 4. Data format

StarWAM accepts both the LeRobot v2/v2.1 per-episode layout and the LeRobot v3
sharded layout. `LeRobotDataset` detects the layout from `meta/info.json`; the
recipe and `data.dataset_dirs` interface are identical for both versions. Recipes
use fields such as:

- RGB video observations, optionally from multiple cameras;
- low-dimensional actions;
- optional robot proprio/state;
- task language descriptions;
- precomputed T5 text embeddings.

Typical LIBERO settings:

```yaml
data:
  dataset_type: lerobot
  dataset_dirs:
    - /path/to/libero_spatial_lerobot
    - /path/to/libero_object_lerobot
    - /path/to/libero_goal_lerobot
    - /path/to/libero_10_lerobot
  video_keys:
    - observation.images.image
    - observation.images.wrist_image
  concat_multi_camera: horizontal
  action_key: action
  state_key: observation.state
  num_frames: 33
  action_freq_ratio: 4
  normalize_actions: true
  action_norm_mode: minmax
  normalize_states: true
  state_norm_mode: minmax
```

For LeRobot v3, each dataset root should contain the standard organization:

```text
meta/info.json
meta/tasks.parquet
meta/episodes/chunk-*/file-*.parquet
data/chunk-*/file-*.parquet
videos/<video_key>/chunk-*/file-*.mp4
```

Episode row ranges, shard paths, and video timestamp offsets are read from the
v3 episode metadata. No v3-to-v2 conversion is required.

For code-only smoke tests, use `data.dataset_type: synthetic`. This uses dummy samples and is not for real training/evaluation.

## 5. Preprocessing

### 5.1 ActionDiT initialization for MoT WAM

For Wan2.2 and Cosmos-Predict2 MoT, `framework.action_expert_init_from` is not a downloaded checkpoint. It is generated from the selected video DiT weights once before training. Both recipes still use the generic token-action `ActionDiT`; Cosmos-Predict2 uses a best-effort structural mapping from Cosmos transformer weights rather than a separate `cosmos_action_dit` implementation.

Wan2.2:

```bash
python -m starwam.tools.preprocess_action_dit_init \
  --config examples/libero/configs/recipes/starwam_libero_mot_wan22_5b.yaml \
  --source-backbone wan22 \
  --pretrained-model-id /path/to/Wan2.2-TI2V-5B \
  --output /path/to/preprocessed/starwam_action_dit_init_wan22.pt \
  --device cuda:0 \
  --dtype bfloat16
```

Cosmos-Predict2:

```bash
python -m starwam.tools.preprocess_action_dit_init \
  --config examples/libero/configs/recipes/starwam_libero_mot_cosmos_predict2.yaml \
  --source-backbone cosmos_predict2 \
  --pretrained-model-id /path/to/Cosmos-Predict2-2B-Video2World \
  --output /path/to/preprocessed/starwam_action_dit_init_cosmos_predict2_notimeproj.pt \
  --device cpu \
  --dtype bfloat16
```

Set the generated path in YAML or pass `--override framework.action_expert_init_from=/path/to/preprocessed/<payload>.pt`. Not needed for Shared-DiT or the feature-conditioned recipe (both leave it `null`).

### 5.2 Text embedding cache

Text embeddings are cache files generated from LIBERO task language. Training creates missing caches automatically. For Wan2.2 recipes, you can also precompute them explicitly:

```bash
python -m starwam.tools.precompute_text_cache \
  --config examples/libero/configs/recipes/starwam_libero_mot_wan22_5b.yaml \
  --pretrained-model-id /path/to/Wan2.2-TI2V-5B \
  --output-dir /path/to/output/starwam_libero_mot_wan22_5b/text_embedding_cache \
  --device cuda:0 \
  --dtype bf16 \
  --override 'data.dataset_dirs=["/path/to/libero_spatial_lerobot","/path/to/libero_object_lerobot","/path/to/libero_goal_lerobot","/path/to/libero_10_lerobot"]'
```

Use separate cache dirs for Wan and Cosmos. Their `data.text_cache_encoder_id` values are already set in the recipes.

### 5.3 Action/state normalization stats

If `data.normalize_actions=true` or `data.normalize_states=true`, the training dataset builder loads stats from the configured JSON path. If the file does not exist, it computes the stats from `data.dataset_dirs` and writes the JSON file.

Recommended setup:

```yaml
data:
  action_stats_path: /path/to/output/<run_name>/action_stats.json
  state_stats_path: /path/to/output/<run_name>/action_stats.json
```

Using the same JSON for action and state stats is supported; the file stores separate `action` and `state` entries.

## 6. Training

### 6.1 Wan2.2 MoT WAM

Run the ActionDiT preprocessing in Section 5.1, then launch training:

```bash
accelerate launch \
  --config_file configs/accelerate/deepspeed_zero2.yaml \
  -m starwam.training.train \
  --config examples/libero/configs/recipes/starwam_libero_mot_wan22_5b.yaml \
  --override \
    'data.dataset_dirs=["/path/to/libero_spatial_lerobot","/path/to/libero_object_lerobot","/path/to/libero_goal_lerobot","/path/to/libero_10_lerobot"]' \
    backbone.pretrained_model_id=/path/to/Wan2.2-TI2V-5B \
    framework.action_expert_init_from=/path/to/preprocessed/starwam_action_dit_init_wan22.pt \
    training.output_dir=/path/to/output/starwam_libero_mot_wan22_5b \
    data.text_embedding_cache_dir=/path/to/output/starwam_libero_mot_wan22_5b/text_embedding_cache \
    data.action_stats_path=/path/to/output/starwam_libero_mot_wan22_5b/action_stats.json \
    data.state_stats_path=/path/to/output/starwam_libero_mot_wan22_5b/action_stats.json \
    training.wandb_enabled=false
```

### 6.2 Cosmos-Predict2 MoT WAM

Launch training:

```bash
accelerate launch \
  --config_file configs/accelerate/deepspeed_zero2.yaml \
  -m starwam.training.train \
  --config examples/libero/configs/recipes/starwam_libero_mot_cosmos_predict2.yaml \
  --override \
    'data.dataset_dirs=["/path/to/libero_spatial_lerobot","/path/to/libero_object_lerobot","/path/to/libero_goal_lerobot","/path/to/libero_10_lerobot"]' \
    backbone.pretrained_model_id=/path/to/Cosmos-Predict2-2B-Video2World \
    framework.action_expert_init_from=/path/to/preprocessed/starwam_action_dit_init_cosmos_predict2_notimeproj.pt \
    training.output_dir=/path/to/output/starwam_libero_mot_cosmos_predict2 \
    data.text_embedding_cache_dir=/path/to/output/starwam_libero_mot_cosmos_predict2/text_embedding_cache \
    data.action_stats_path=/path/to/output/starwam_libero_mot_cosmos_predict2/action_stats.json \
    data.state_stats_path=/path/to/output/starwam_libero_mot_cosmos_predict2/action_stats.json \
    training.wandb_enabled=false
```

### 6.3 Wan2.2 Shared-DiT WAM

Launch training:

```bash
accelerate launch \
  --config_file configs/accelerate/deepspeed_zero2.yaml \
  -m starwam.training.train \
  --config examples/libero/configs/recipes/starwam_libero_shared_dit_wan22_5b.yaml \
  --override \
    'data.dataset_dirs=["/path/to/libero_spatial_lerobot","/path/to/libero_object_lerobot","/path/to/libero_goal_lerobot","/path/to/libero_10_lerobot"]' \
    backbone.pretrained_model_id=/path/to/Wan2.2-TI2V-5B \
    training.output_dir=/path/to/output/starwam_libero_shared_dit_wan22_5b \
    data.text_embedding_cache_dir=/path/to/output/starwam_libero_shared_dit_wan22_5b/text_embedding_cache \
    data.action_stats_path=/path/to/output/starwam_libero_shared_dit_wan22_5b/action_stats.json \
    data.state_stats_path=/path/to/output/starwam_libero_shared_dit_wan22_5b/action_stats.json \
    training.wandb_enabled=false
```

### 6.4 Wan2.2 Feature-Conditioned action model

The feature-conditioned recipe does not need the Section 5.1 ActionDiT init; the
action expert is randomly initialized (`framework.action_expert_init_from: null`).
Launch training:

```bash
accelerate launch \
  --config_file configs/accelerate/deepspeed_zero2.yaml \
  -m starwam.training.train \
  --config examples/libero/configs/recipes/starwam_libero_feature_conditioned_wan22_5b.yaml \
  --override \
    'data.dataset_dirs=["/path/to/libero_spatial_lerobot","/path/to/libero_object_lerobot","/path/to/libero_goal_lerobot","/path/to/libero_10_lerobot"]' \
    backbone.pretrained_model_id=/path/to/Wan2.2-TI2V-5B \
    training.output_dir=/path/to/output/starwam_libero_feature_conditioned_wan22_5b \
    data.text_embedding_cache_dir=/path/to/output/starwam_libero_feature_conditioned_wan22_5b/text_embedding_cache \
    data.action_stats_path=/path/to/output/starwam_libero_feature_conditioned_wan22_5b/action_stats.json \
    data.state_stats_path=/path/to/output/starwam_libero_feature_conditioned_wan22_5b/action_stats.json \
    training.wandb_enabled=false
```

### 6.5 Launch scripts

Convenience scripts are provided in:

```text
examples/libero/scripts/
```

Current scripts:

```text
launch_starwam_libero_mot_wan22_5b_8gpu.sh
launch_starwam_libero_shared_dit_wan22_5b_8gpu.sh
launch_starwam_libero_feature_conditioned_wan22_5b_8gpu.sh
launch_starwam_libero_mot_rollout.sh
```

Before running them, edit recipe paths or pass overrides through environment variables:

```bash
cd /path/to/starWAM

export CONDA_ENV=starwam-libero
export REPO_DIR=/path/to/starWAM
export TRAIN_OVERRIDES='data.dataset_dirs=["/path/to/libero_spatial_lerobot","/path/to/libero_object_lerobot","/path/to/libero_goal_lerobot","/path/to/libero_10_lerobot"] backbone.pretrained_model_id=/path/to/Wan2.2-TI2V-5B framework.action_expert_init_from=/path/to/preprocessed/starwam_action_dit_init_wan22.pt training.output_dir=/path/to/output/starwam_libero_mot_wan22_5b data.text_embedding_cache_dir=/path/to/output/starwam_libero_mot_wan22_5b/text_embedding_cache data.action_stats_path=/path/to/output/starwam_libero_mot_wan22_5b/action_stats.json data.state_stats_path=/path/to/output/starwam_libero_mot_wan22_5b/action_stats.json training.wandb_enabled=false'

bash examples/libero/scripts/launch_starwam_libero_mot_wan22_5b_8gpu.sh
```

## 7. Checkpoints

Training writes checkpoints under:

```text
${training.output_dir}/checkpoint-<step>/
```

The Wan2.2 MoT recipe retains the latest three local checkpoints. When remote
upload is enabled, an old checkpoint is never cleaned up until its verified
TOS marker exists.

TOS replication is trainer-level infrastructure and is not tied to LIBERO,
MoT, or Wan2.2. See [TOS checkpoint persistence](../../docs/CHECKPOINT_TOS.md)
for installation, configuration, and verification.

The rollout script can load:

- a checkpoint directory containing `model.pt`, `pytorch_model.bin`, or safetensors files;
- a direct checkpoint file;
- Accelerate/DeepSpeed checkpoint layouts handled by the loader.

If `--checkpoint` is omitted, rollout searches for the latest `checkpoint-*` under `training.output_dir`.

## 8. Rollout / Evaluation

### 8.1 Wan2.2 MoT rollout

```bash
python examples/libero/rollout.py \
  --config examples/libero/configs/recipes/starwam_libero_mot_wan22_5b.yaml \
  --checkpoint /path/to/output/starwam_libero_mot_wan22_5b/checkpoint-20000 \
  --task-suite-name libero_spatial \
  --num-trials 50 \
  --num-inference-steps 8 \
  --replan-steps 10 \
  --device cuda:0 \
  --libero-home /path/to/LIBERO \
  --override \
    backbone.pretrained_model_id=/path/to/Wan2.2-TI2V-5B \
    framework.action_expert_init_from=/path/to/preprocessed/starwam_action_dit_init_wan22.pt \
    training.output_dir=/path/to/output/starwam_libero_mot_wan22_5b \
    data.text_embedding_cache_dir=/path/to/output/starwam_libero_mot_wan22_5b/text_embedding_cache \
    data.action_stats_path=/path/to/output/starwam_libero_mot_wan22_5b/action_stats.json \
    data.state_stats_path=/path/to/output/starwam_libero_mot_wan22_5b/action_stats.json
```

### 8.2 Cosmos-Predict2 MoT rollout

```bash
python examples/libero/rollout.py \
  --config examples/libero/configs/recipes/starwam_libero_mot_cosmos_predict2.yaml \
  --checkpoint /path/to/output/starwam_libero_mot_cosmos_predict2/checkpoint-20000 \
  --task-suite-name libero_spatial \
  --num-trials 50 \
  --num-inference-steps 8 \
  --replan-steps 10 \
  --device cuda:0 \
  --libero-home /path/to/LIBERO \
  --override \
    backbone.pretrained_model_id=/path/to/Cosmos-Predict2-2B-Video2World \
    framework.action_expert_init_from=/path/to/preprocessed/starwam_action_dit_init_cosmos_predict2_notimeproj.pt \
    training.output_dir=/path/to/output/starwam_libero_mot_cosmos_predict2 \
    data.text_embedding_cache_dir=/path/to/output/starwam_libero_mot_cosmos_predict2/text_embedding_cache \
    data.action_stats_path=/path/to/output/starwam_libero_mot_cosmos_predict2/action_stats.json \
    data.state_stats_path=/path/to/output/starwam_libero_mot_cosmos_predict2/action_stats.json
```

### 8.3 Wan2.2 Shared-DiT rollout

Shared-DiT supports decoupled video/action denoising step counts. Pass both values explicitly:

```bash
python examples/libero/rollout.py \
  --config examples/libero/configs/recipes/starwam_libero_shared_dit_wan22_5b.yaml \
  --checkpoint /path/to/output/starwam_libero_shared_dit_wan22_5b/checkpoint-50000 \
  --task-suite-name libero_spatial \
  --num-trials 50 \
  --num-inference-steps 16 \
  --action-num-inference-steps 16 \
  --replan-steps 10 \
  --device cuda:0 \
  --libero-home /path/to/LIBERO \
  --override \
    backbone.pretrained_model_id=/path/to/Wan2.2-TI2V-5B \
    training.output_dir=/path/to/output/starwam_libero_shared_dit_wan22_5b \
    data.text_embedding_cache_dir=/path/to/output/starwam_libero_shared_dit_wan22_5b/text_embedding_cache \
    data.action_stats_path=/path/to/output/starwam_libero_shared_dit_wan22_5b/action_stats.json \
    data.state_stats_path=/path/to/output/starwam_libero_shared_dit_wan22_5b/action_stats.json
```

### 8.4 Wan2.2 Feature-Conditioned rollout

Feature-conditioned uses a single action denoising schedule (no decoupled steps),
so pass only `--num-inference-steps`.

```bash
python examples/libero/rollout.py \
  --config examples/libero/configs/recipes/starwam_libero_feature_conditioned_wan22_5b.yaml \
  --checkpoint /path/to/output/starwam_libero_feature_conditioned_wan22_5b/checkpoint-100000 \
  --task-suite-name libero_spatial \
  --num-trials 50 \
  --num-steps-wait 30 \
  --num-inference-steps 10 \
  --replan-steps 10 \
  --device cuda:0 \
  --libero-home /path/to/LIBERO \
  --override \
    backbone.pretrained_model_id=/path/to/Wan2.2-TI2V-5B \
    training.output_dir=/path/to/output/starwam_libero_feature_conditioned_wan22_5b \
    data.text_embedding_cache_dir=/path/to/output/starwam_libero_feature_conditioned_wan22_5b/text_embedding_cache \
    data.action_stats_path=/path/to/output/starwam_libero_feature_conditioned_wan22_5b/action_stats.json \
    data.state_stats_path=/path/to/output/starwam_libero_feature_conditioned_wan22_5b/action_stats.json
```

### 8.5 Rollout launcher and outputs

Use `examples/libero/scripts/launch_starwam_libero_mot_rollout.sh` to evaluate one recipe across multiple LIBERO suites. Set `CHECKPOINT` to either a checkpoint directory or a direct checkpoint file. Direct DeepSpeed `.pt` files are supported; the loader reads `model_state_dict`, `module`, or `state_dict` payloads.

```bash
cd /path/to/starWAM

export CONDA_ENV=starwam-libero
export REPO_DIR=/path/to/starWAM
export RECIPE=examples/libero/configs/recipes/starwam_libero_mot_cosmos_predict2.yaml
export CHECKPOINT=/path/to/output/starwam_libero_mot_cosmos_predict2/checkpoint-20000/pytorch_model/mp_rank_00_model_states.pt
export LIBERO_HOME=/path/to/LIBERO
export RUN_ALL_SUITES=1
export PARALLEL_SUITES=1
export NUM_TRIALS=50
export NUM_STEPS_WAIT=30
export REPLAN_STEPS=10
export NUM_INFERENCE_STEPS=8
export SUITE_GPUS="0 1 2 3"
export ROLLOUT_OVERRIDES='data.dataset_dirs=["/path/to/libero_spatial_lerobot","/path/to/libero_object_lerobot","/path/to/libero_goal_lerobot","/path/to/libero_10_lerobot"] backbone.pretrained_model_id=/path/to/Cosmos-Predict2-2B-Video2World framework.action_expert_init_from=/path/to/preprocessed/starwam_action_dit_init_cosmos_predict2_notimeproj.pt training.output_dir=/path/to/output/starwam_libero_mot_cosmos_predict2 data.text_embedding_cache_dir=/path/to/output/starwam_libero_mot_cosmos_predict2/text_embedding_cache data.action_stats_path=/path/to/output/starwam_libero_mot_cosmos_predict2/action_stats.json data.state_stats_path=/path/to/output/starwam_libero_mot_cosmos_predict2/action_stats.json'

bash examples/libero/scripts/launch_starwam_libero_mot_rollout.sh
```

With `RUN_ALL_SUITES=1` the launcher runs all four suites and each writes its
own `results.json`. Aggregate them into an overall success rate with
`examples/libero/summarize_results.py` (Section 8.6).

Useful rollout options:

```bash
--libero-home /path/to/LIBERO
--output-dir /path/to/rollout_outputs
--save-video
--fixed-seed
```

With `--save-video --output-dir /path/to/rollout_outputs`, videos are written to `/path/to/rollout_outputs/videos/`. Without `--output-dir`, videos are written under `${training.output_dir}/libero_rollout/<checkpoint-name>/<task-suite-name>/videos/`. Rollout videos save one frame per executed environment step after the initial wait period.

The rollout script loads the recipe/checkpoint, applies overrides, builds the model, and repeatedly calls `model.infer_action(...)` in LIBERO.

### 8.6 Results files and cross-suite summary

Each `rollout.py` run writes one `results.json` per suite:

```text
<training.output_dir>/libero_rollout/<checkpoint-name>/<task-suite-name>/results.json
```

Each `results.json` contains per-task success rates and the suite-level
micro-average (`total_successes`, `total_trials`, `success_rate`). It does
**not** aggregate across suites — each suite is a separate run/file.

To get the overall (cross-suite) success rate, aggregate the four
per-suite files with `examples/libero/summarize_results.py`:

```bash
# Aggregate every suite under one checkpoint directory:
python examples/libero/summarize_results.py \
  --rollout-dir /path/to/output/<recipe>/libero_rollout/<checkpoint-name>

# Or point at explicit results.json files:
python examples/libero/summarize_results.py \
  --results \
    /path/to/.../libero_spatial/results.json \
    /path/to/.../libero_object/results.json \
    /path/to/.../libero_goal/results.json \
    /path/to/.../libero_10/results.json \
  --output /path/to/summary.json
```

It prints a per-suite table plus the micro-average across all suites, and
writes `summary.json` (defaults to `<rollout-dir>/summary.json` when
`--rollout-dir` is used):

```text
Suite            Success  Trials  Success rate
---------------  -------  ------  ------------
libero_spatial   471      500     94.2%
libero_object    500      500     100.0%
libero_goal      484      500     96.8%
libero_10        481      500     96.2%
---------------  -------  ------  ------------
Overall (micro)  1936     2000    96.8%
```

The `Overall (micro)` value is `sum(total_successes) / sum(total_trials)`
across suites, matching the reported numbers in Section 2.

### 8.7 Rollout from released ModelScope checkpoints

Pretrained Wan2.2-TI2V-5B checkpoints are released on ModelScope at
[`panshaohua/starwam`](https://www.modelscope.cn/models/panshaohua/starwam).
Download them first:

```bash
pip install modelscope
modelscope download --model panshaohua/starwam --local_dir /path/to/starwam_ckpts
```

After download, the LIBERO checkpoints are laid out as:

```text
/path/to/starwam_ckpts/starwam-libero/
  action_stats.json                          # shared action stats (both models)
  mot/starwam_wan225b_mot.pt                  # MoT WAM
  sharedit/starwam_wan225b_shareddit.pt       # Shared-DiT WAM
```

Point `--checkpoint` directly at the `.pt` file (these files use custom names,
so pass the file path, not the directory). You still need the Wan2.2 backbone
locally for `backbone.pretrained_model_id`.

Shared-DiT:

```bash
CKPT_ROOT=/path/to/starwam_ckpts/starwam-libero

python examples/libero/rollout.py \
  --config examples/libero/configs/recipes/starwam_libero_shared_dit_wan22_5b.yaml \
  --checkpoint "$CKPT_ROOT/sharedit/starwam_wan225b_shareddit.pt" \
  --task-suite-name libero_spatial \
  --num-trials 50 \
  --num-inference-steps 16 \
  --action-num-inference-steps 16 \
  --replan-steps 10 \
  --device cuda:0 \
  --libero-home /path/to/LIBERO \
  --override \
    backbone.pretrained_model_id=/path/to/Wan2.2-TI2V-5B \
    training.output_dir=/path/to/output/starwam_libero_shared_dit_wan22_5b \
    data.text_embedding_cache_dir=/path/to/output/starwam_libero_shared_dit_wan22_5b/text_embedding_cache \
    data.action_stats_path="$CKPT_ROOT/action_stats.json" \
    data.state_stats_path="$CKPT_ROOT/action_stats.json"
```

MoT uses a single denoising schedule — pass only
`--num-inference-steps`:

```bash
CKPT_ROOT=/path/to/starwam_ckpts/starwam-libero

python examples/libero/rollout.py \
  --config examples/libero/configs/recipes/starwam_libero_mot_wan22_5b.yaml \
  --checkpoint "$CKPT_ROOT/mot/starwam_wan225b_mot.pt" \
  --task-suite-name libero_spatial \
  --num-trials 50 \
  --num-inference-steps 8 \
  --replan-steps 10 \
  --device cuda:0 \
  --libero-home /path/to/LIBERO \
  --override \
    backbone.pretrained_model_id=/path/to/Wan2.2-TI2V-5B \
    training.output_dir=/path/to/output/starwam_libero_mot_wan22_5b \
    data.text_embedding_cache_dir=/path/to/output/starwam_libero_mot_wan22_5b/text_embedding_cache \
    data.action_stats_path="$CKPT_ROOT/action_stats.json" \
    data.state_stats_path="$CKPT_ROOT/action_stats.json"
```

## 9. Decoupled action steps

MoT WAM uses a single denoising schedule for action rollout. For MoT recipes, `examples/libero/rollout.py` overrides `action_num_inference_steps` to match `num_inference_steps`, so `inference.action_num_inference_steps` is kept only for schema consistency.

Shared-DiT uses decoupled step counts, so rollout passes `--action-num-inference-steps` separately.

## 10. Troubleshooting

- Placeholder paths: replace all `/path/to/...` values before running.
- Missing ActionDiT init: run Section 5.1 and set `framework.action_expert_init_from`.
- LIBERO import error: install LIBERO or pass `--libero-home /path/to/LIBERO`.
- No checkpoint found: pass `--checkpoint` or check `training.output_dir`.
- Wrong action scale: check action/state stats paths and normalization settings.
- Multi-camera mismatch: check `video_keys`, `concat_multi_camera`, and `video_size`.
