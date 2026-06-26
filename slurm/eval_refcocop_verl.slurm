#!/bin/bash
#SBATCH --job-name=eval_refcocop
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=460000
#SBATCH --time=06:00:00
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

# Same stability env as the RL/eval jobs (OTel off avoids Ray AgentLoopWorker segfault).
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

LABEL="${LABEL:?set LABEL=<base|sft|rl>}"
VAL="${VAL:-$PROJECT/data_prep/refcocop/val.parquet}"

# MODEL may be an explicit HF dir, or a sentinel that auto-discovers the latest
# checkpoint so this job can be chained with --dependency before the checkpoint exists.
BASE_MODEL_PATH="/capstor/store/cscs/swissai/infra01/MLLM/ablations/apertus-8b-img-SFT-32nodes-gbs512-mbs1-steps8030-img-text-seqlen8192-s2onlytxtloss/HF"
SFT_CKPT_DIR="${SFT_CKPT_DIR:-${PROJECT}/checkpoints/apertus8b-vcot-sft}"
RL_CKPT_DIR="${RL_CKPT_DIR:-${PROJECT}/checkpoints/vcot_rl_apertus/apertus8b_grpo_vcot_rl}"

latest_step_subdir() {  # <root> <leaf>  -> echoes <root>/global_step_<max>/<leaf> if it has weights
    local root="$1" leaf="$2" best=-1 best_path="" ckpt step cand
    shopt -s nullglob
    for ckpt in "$root"/global_step_*; do
        step="${ckpt##*_}"
        cand="$ckpt/$leaf"
        if [[ "$step" =~ ^[0-9]+$ ]] && [ "$step" -gt "$best" ] && [ -f "$cand/config.json" ]; then
            best="$step"; best_path="$cand"
        fi
    done
    shopt -u nullglob
    [ -n "$best_path" ] && { echo "$best_path"; return 0; } || return 1
}

MODEL="${MODEL:?set MODEL=<hf ckpt dir|base|auto-sft|auto-rl>}"
case "$MODEL" in
    base)     MODEL="$BASE_MODEL_PATH" ;;
    auto-sft) MODEL="$(latest_step_subdir "$SFT_CKPT_DIR" huggingface)" || { echo "no SFT ckpt under $SFT_CKPT_DIR"; exit 1; } ;;
    auto-rl)  MODEL="$(latest_step_subdir "$RL_CKPT_DIR" actor_hf_merged)" || { echo "no merged RL ckpt under $RL_CKPT_DIR"; exit 1; } ;;
esac

echo "=== eval_refcocop_verl $(date) ==="
echo "MODEL=$MODEL  LABEL=$LABEL  VAL=$VAL"
nvidia-smi --list-gpus

cd "$PROJECT"
# val_only: build the val dataloader from the RefCOCO+ parquet, run one sglang
# agent-loop validation pass and score via refcocop_reward (Acc@0.5 IoU on the
# draw_bbox prediction). train_files points at VAL too so the eval-only run does
# not depend on vcot_rl/train.parquet.
python3 -m verl.trainer.main_ppo \
    --config-path="$PROJECT/configs" \
    --config-name=vcot_rl_grpo \
    actor_rollout_ref.model.path="$MODEL" \
    data.train_files="$VAL" \
    data.val_files="$VAL" \
    reward.custom_reward_function.path="$PROJECT/rewards/refcocop_reward.py" \
    reward.custom_reward_function.name=compute_score \
    trainer.val_only=true \
    trainer.val_before_train=true \
    trainer.logger='[console]' \
    trainer.project_name=vcot_refcocop_eval \
    trainer.experiment_name="refcocop_eval_${LABEL}" \
    "$@"

echo "=== done ($(date)) ==="
