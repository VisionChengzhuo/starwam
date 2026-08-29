#!/bin/bash
# Generic StarWAM LIBERO launcher for local and Volcano multi-node GPU jobs.
set -euo pipefail

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
CONDA_SH=${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-starwam-libero}
PY=${PY:-python}

RECIPE=${RECIPE:-examples/libero/configs/recipes/starwam_libero_mot_wan22_5b.yaml}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-configs/accelerate/deepspeed_zero2.yaml}
TRAIN_OVERRIDES=${TRAIN_OVERRIDES:-}
DYNAMO_BACKEND=${DYNAMO_BACKEND:-}

# Explicit values support manual launches; Volcano supplies the MLP_* fallbacks
# when the same command is started on every worker.
NUM_MACHINES=${NUM_MACHINES:-${MLP_WORKER_NUM:-1}}
GPUS_PER_NODE=${GPUS_PER_NODE:-${MLP_WORKER_GPU:-8}}
MACHINE_RANK=${MACHINE_RANK:-${MLP_ROLE_INDEX:-0}}
MAIN_PROCESS_IP=${MAIN_PROCESS_IP:-${MLP_WORKER_0_HOST:-127.0.0.1}}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-${MLP_WORKER_0_PORT:-29617}}
NUM_PROCESSES=${NUM_PROCESSES:-$((NUM_MACHINES * GPUS_PER_NODE))}
DEEPSPEED_MULTINODE_LAUNCHER=${DEEPSPEED_MULTINODE_LAUNCHER:-standard}

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-1048576}
TOS_HOST=tos-cn-beijing.ivolces.com
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}$TOS_HOST"
export no_proxy="${no_proxy:+$no_proxy,}$TOS_HOST"

cd "$REPO_DIR"

if [ -f "$CONDA_SH" ]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
fi

OVERRIDE_ARGS=()
if [ -n "$TRAIN_OVERRIDES" ]; then
  # shellcheck disable=SC2206
  OVERRIDE_ARGS=($TRAIN_OVERRIDES)
fi

OUTPUT_DIR=$(
  "$PY" - "$RECIPE" "${OVERRIDE_ARGS[@]}" <<'PY'
import sys

from examples.libero.presets import validate_preset
from starwam.config import load_config
from starwam.utils.config_cli import apply_overrides

config = apply_overrides(load_config(sys.argv[1]), sys.argv[2:])
validate_preset(config)
print(config.training.output_dir)
PY
)

LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE=${LOG_FILE:-"$LOG_DIR/train.node${MACHINE_RANK}_$(date +%Y%m%d_%H%M%S).log"}

DISTRIBUTED_ARGS=(
  --num_processes "$NUM_PROCESSES"
  --num_machines "$NUM_MACHINES"
  --machine_rank "$MACHINE_RANK"
  --main_process_port "$MAIN_PROCESS_PORT"
)
if (( NUM_MACHINES > 1 )); then
  DISTRIBUTED_ARGS+=(
    --same_network
    --main_process_ip "$MAIN_PROCESS_IP"
    --deepspeed_multinode_launcher "$DEEPSPEED_MULTINODE_LAUNCHER"
  )
fi
if [ -n "$DYNAMO_BACKEND" ]; then
  DISTRIBUTED_ARGS+=(--dynamo_backend "$DYNAMO_BACKEND")
fi

CMD=(
  "$PY" -m accelerate.commands.launch
  --config_file "$ACCELERATE_CONFIG"
  "${DISTRIBUTED_ARGS[@]}"
  --module starwam.training.train
  --config "$RECIPE"
)
if (( ${#OVERRIDE_ARGS[@]} > 0 )); then
  CMD+=(--override "${OVERRIDE_ARGS[@]}")
fi

cat <<EOF
[launch] mode: StarWAM LIBERO distributed
[launch] topology: ${NUM_MACHINES}x${GPUS_PER_NODE} (world_size=$NUM_PROCESSES)
[launch] machine_rank: $MACHINE_RANK
[launch] main_process: $MAIN_PROCESS_IP:$MAIN_PROCESS_PORT
[launch] cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-<scheduler default>}
[launch] recipe: $RECIPE
[launch] accelerate_config: $ACCELERATE_CONFIG
[launch] dynamo_backend: ${DYNAMO_BACKEND:-<disabled>}
[launch] output_dir: $OUTPUT_DIR
[launch] overrides: ${TRAIN_OVERRIDES:-<none>}
[launch] log_file: $LOG_FILE
EOF

printf '[launch] command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [ "${DRY_RUN:-0}" = 1 ]; then
  exit 0
fi

exec "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
