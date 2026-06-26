#!/bin/bash
#SBATCH --job-name=merge-vcot-rl
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=460000
#SBATCH --time=02:00:00
#SBATCH --environment=verl_env
#SBATCH --partition=normal
#SBATCH --gpus-per-node=1
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
export HF_HOME="${SCRATCH}/hf_home"
export TRANSFORMERS_CACHE="${HF_HOME}"

RL_CHECKPOINT_DIR="${RL_CHECKPOINT_DIR:-${PROJECT}/checkpoints/vcot_rl_apertus/apertus8b_grpo_vcot_rl}"
# If STEP is unset, auto-pick the latest global_step_* under RL_CHECKPOINT_DIR.
if [ -z "${STEP:-}" ]; then
    STEP="$(ls -d "$RL_CHECKPOINT_DIR"/global_step_* 2>/dev/null | sed 's#.*global_step_##' | sort -n | tail -1)"
fi
STEP="${STEP:?no global_step_* found under $RL_CHECKPOINT_DIR; set STEP=<n>}"
LOCAL_DIR="${LOCAL_DIR:-${RL_CHECKPOINT_DIR}/global_step_${STEP}/actor}"
TARGET_DIR="${TARGET_DIR:-${RL_CHECKPOINT_DIR}/global_step_${STEP}/actor_hf_merged}"

echo "=== Job started at $(date) ==="
echo "Merging FSDP checkpoint:"
echo "  step:       $STEP"
echo "  local_dir:  $LOCAL_DIR"
echo "  target_dir: $TARGET_DIR"

test -f "$LOCAL_DIR/fsdp_config.json"
test -f "$LOCAL_DIR/huggingface/config.json"
test -f "$LOCAL_DIR/huggingface/tokenizer_config.json"

cd "$VERL"
python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$LOCAL_DIR" \
    --target_dir "$TARGET_DIR" \
    --trust-remote-code

echo ""
echo "Merged files:"
find "$TARGET_DIR" -maxdepth 1 -type f -printf "  %f\n" | sort
echo "=== Job ended at $(date) ==="
