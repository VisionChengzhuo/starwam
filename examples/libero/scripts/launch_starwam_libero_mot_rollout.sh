#!/bin/bash
# LIBERO environment rollout for StarWAM MoT checkpoints.
set -euo pipefail

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
CONDA_SH=${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-starwam-libero}
PY=${PY:-python}

RECIPE=${RECIPE:-examples/libero/configs/recipes/starwam_libero_mot_wan22_5b.yaml}
CHECKPOINT=${CHECKPOINT:-}
OUTPUT_DIR=${OUTPUT_DIR:-}
LIBERO_HOME=${LIBERO_HOME:-}
TASK_SUITE_NAME=${TASK_SUITE_NAME:-libero_spatial}
TASK_ID=${TASK_ID:-}
NUM_TRIALS=${NUM_TRIALS:-50}
NUM_STEPS_WAIT=${NUM_STEPS_WAIT:-30}
REPLAN_STEPS=${REPLAN_STEPS:-10}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-}
ACTION_NUM_INFERENCE_STEPS=${ACTION_NUM_INFERENCE_STEPS:-}
MAX_STEPS=${MAX_STEPS:-}
DEVICE=${DEVICE:-cuda:0}
SEED=${SEED:-42}
FIXED_SEED=${FIXED_SEED:-0}
SAVE_VIDEO=${SAVE_VIDEO:-0}
ROLLOUT_OVERRIDES=${ROLLOUT_OVERRIDES:-}
RUN_ALL_SUITES=${RUN_ALL_SUITES:-1}
PARALLEL_SUITES=${PARALLEL_SUITES:-1}
SUITE_NAMES=${SUITE_NAMES:-"libero_10 libero_goal libero_spatial libero_object"}
SUITE_GPUS=${SUITE_GPUS:-"0 1 2 3"}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-$MUJOCO_GL}
if [ "$MUJOCO_GL" = "osmesa" ] && [ -f /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ]; then
  case ":${LD_PRELOAD:-}:" in
    *:/usr/lib/x86_64-linux-gnu/libstdc++.so.6:*) ;;
    *) export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}" ;;
  esac
fi
if [ -n "$LIBERO_HOME" ]; then
  export LIBERO_HOME
  export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-$HOME/.cache/starwam/libero_config}
  mkdir -p "$LIBERO_CONFIG_PATH"
  cat > "$LIBERO_CONFIG_PATH/config.yaml" <<EOF_LIBERO
benchmark_root: $LIBERO_HOME/libero/libero
bddl_files: $LIBERO_HOME/libero/libero/bddl_files
init_states: $LIBERO_HOME/libero/libero/init_files
datasets: $LIBERO_HOME/libero/datasets
assets: $LIBERO_HOME/libero/libero/assets
EOF_LIBERO
  export PYTHONPATH="$LIBERO_HOME:${PYTHONPATH:-}"
fi

cd "$REPO_DIR"

if [ -f "$CONDA_SH" ]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
fi

RECIPE_OUTPUT_DIR=$($PY - "$RECIPE" $ROLLOUT_OVERRIDES <<'PY'
import sys
from starwam.config import load_config
from starwam.utils.config_cli import apply_overrides

recipe = sys.argv[1]
overrides = sys.argv[2:]
cfg = load_config(recipe)
cfg = apply_overrides(cfg, overrides)
print(cfg.training.output_dir)
PY
)

LOG_DIR="$RECIPE_OUTPUT_DIR/libero_rollout/logs"
mkdir -p "$LOG_DIR"

if [ "$RUN_ALL_SUITES" = "1" ] || [ "$RUN_ALL_SUITES" = "true" ]; then
  read -r -a SUITES <<< "$SUITE_NAMES"
  read -r -a GPUS <<< "$SUITE_GPUS"
  if [ "$PARALLEL_SUITES" = "1" ] || [ "$PARALLEL_SUITES" = "true" ]; then
    for idx in "${!SUITES[@]}"; do
      suite="${SUITES[$idx]}"
      gpu="${GPUS[$idx]:-${GPUS[0]:-0}}"
      (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export DEVICE="cuda:0"
        export TASK_SUITE_NAME="$suite"
        export RUN_ALL_SUITES=0
        bash "$0"
      ) &
    done
    wait
  else
    for suite in "${SUITES[@]}"; do
      TASK_SUITE_NAME="$suite" RUN_ALL_SUITES=0 bash "$0"
    done
  fi
  exit 0
fi

LOG_TAG=${LOG_TAG:-${TASK_SUITE_NAME}${TASK_ID:+_task${TASK_ID}}_$(basename "${OUTPUT_DIR:-default}")}
LOG_FILE="$LOG_DIR/rollout_${LOG_TAG}_$(date +%Y%m%d_%H%M%S).log"

ARGS=(
  --config "$RECIPE"
  --task-suite-name "$TASK_SUITE_NAME"
  --num-trials "$NUM_TRIALS"
  --num-steps-wait "$NUM_STEPS_WAIT"
  --replan-steps "$REPLAN_STEPS"
  --device "$DEVICE"
  --seed "$SEED"
)

if [ -n "$NUM_INFERENCE_STEPS" ]; then
  ARGS+=(--num-inference-steps "$NUM_INFERENCE_STEPS")
fi
if [ -n "$ACTION_NUM_INFERENCE_STEPS" ]; then
  ARGS+=(--action-num-inference-steps "$ACTION_NUM_INFERENCE_STEPS")
fi
if [ -n "$CHECKPOINT" ]; then
  ARGS+=(--checkpoint "$CHECKPOINT")
fi
if [ -n "$OUTPUT_DIR" ]; then
  ARGS+=(--output-dir "$OUTPUT_DIR")
fi
if [ -n "$LIBERO_HOME" ]; then
  ARGS+=(--libero-home "$LIBERO_HOME")
fi
if [ -n "$TASK_ID" ]; then
  ARGS+=(--task-id "$TASK_ID")
fi
if [ -n "$MAX_STEPS" ]; then
  ARGS+=(--max-steps "$MAX_STEPS")
fi
if [ "$FIXED_SEED" = "1" ] || [ "$FIXED_SEED" = "true" ]; then
  ARGS+=(--fixed-seed)
fi
if [ "$SAVE_VIDEO" = "1" ] || [ "$SAVE_VIDEO" = "true" ]; then
  ARGS+=(--save-video)
fi
if [ -n "$ROLLOUT_OVERRIDES" ]; then
  # shellcheck disable=SC2206
  EXTRA_OVERRIDES=($ROLLOUT_OVERRIDES)
  ARGS+=(--override "${EXTRA_OVERRIDES[@]}")
fi

cat <<EOF
[launch] mode: StarWAM LIBERO MoT rollout
[launch] recipe: $RECIPE
[launch] checkpoint: ${CHECKPOINT:-<latest under recipe output_dir>}
[launch] output_dir: ${OUTPUT_DIR:-<recipe output_dir>/libero_rollout/<checkpoint>/<task_suite>}
[launch] libero_home: ${LIBERO_HOME:-<python env import>}
[launch] task_suite_name: $TASK_SUITE_NAME
[launch] task_id: ${TASK_ID:-<all>}
[launch] num_trials: $NUM_TRIALS
[launch] replan_steps: $REPLAN_STEPS
[launch] num_inference_steps: ${NUM_INFERENCE_STEPS:-<recipe inference.num_inference_steps>}
[launch] action_num_inference_steps: ${ACTION_NUM_INFERENCE_STEPS:-<recipe inference.action_num_inference_steps>}
[launch] device: $DEVICE
[launch] seed: $SEED
[launch] fixed_seed: $FIXED_SEED
[launch] save_video: $SAVE_VIDEO
[launch] overrides: ${ROLLOUT_OVERRIDES:-<none>}
[launch] log_file: $LOG_FILE
EOF

exec "$PY" examples/libero/rollout.py "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"
