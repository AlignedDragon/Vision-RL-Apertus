#!/bin/bash
#SBATCH --job-name=eval_vstar_verl
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=460000
#SBATCH --time=03:00:00
#SBATCH --environment=verl_env
#SBATCH --partition=normal
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
ulimit -n 65535
source /users/msayfiddinov/capscratch/miniconda/etc/profile.d/conda.sh
conda activate verl

PROJECT="/users/$USER/capscratch/verl-apertus"
VERL="/users/$USER/capscratch/verl"
EMU3_SRC="/users/$USER/capscratch/Emu3.5/src"
export PYTHONPATH="${PROJECT}:${VERL}:${EMU3_SRC}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Same stability env as cof_rl.slurm (OTel off avoids Ray AgentLoopWorker segfault).
export OTEL_SDK_DISABLED=true
export OTEL_TRACES_EXPORTER=none
export OTEL_METRICS_EXPORTER=none
export OTEL_LOGS_EXPORTER=none
export GRPC_ENABLE_FORK_SUPPORT=0
export RAY_experimental_enable_open_telemetry=0
export RAY_enable_open_telemetry=0
export NCCL_DEBUG=WARN
export HF_HOME="/capstor/scratch/cscs/$USER/hf_home"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline

MODEL="${MODEL:?set MODEL=<hf ckpt dir>}"
LABEL="${LABEL:?set LABEL=<base|sft|rl>}"
VAL="${VAL:-$PROJECT/data_prep/vstar_bench/val.parquet}"

echo "=== eval_vstar_verl $(date) ==="
echo "MODEL=$MODEL  LABEL=$LABEL  VAL=$VAL"
nvidia-smi --list-gpus

cd "$PROJECT"
# val_only: build val dataloader from V*Bench parquet, run one sglang agent-loop
# validation pass (greedy by default) and score via cof_rl_reward. train_files is
# pointed at VAL too so we don't depend on cof_rl/train.parquet for an eval-only run.
python3 -m verl.trainer.main_ppo \
    --config-path="$PROJECT/configs" \
    --config-name=cof_rl_grpo \
    actor_rollout_ref.model.path="$MODEL" \
    data.train_files="$VAL" \
    data.val_files="$VAL" \
    trainer.val_only=true \
    trainer.val_before_train=true \
    trainer.logger='[console]' \
    trainer.project_name=cof_rl_apertus_eval \
    trainer.experiment_name="vstar_eval_${LABEL}" \
    "$@"

echo "=== done ($(date)) ==="
