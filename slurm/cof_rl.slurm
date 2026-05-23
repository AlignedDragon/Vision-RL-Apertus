#!/bin/bash
#SBATCH --job-name=rl
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=460000
#SBATCH --time=10:00:00
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

# Note: using $USER means the script is portable across cluster users.
# Adjust EMU3_SRC if your install lives elsewhere.

export PYTHONPATH="${PROJECT}:${VERL}:${EMU3_SRC}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME="${SCRATCH}/hf_home"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HYDRA_FULL_ERROR=1 # for error tracking
export WANDB_MODE=online

BASE_MODEL_PATH="/capstor/store/cscs/swissai/infra01/MLLM/ablations/apertus-8b-img-SFT-32nodes-gbs512-mbs1-steps8030-img-text-seqlen8192-s2onlytxtloss/HF"
SFT_CHECKPOINT_DIR="${SFT_CHECKPOINT_DIR:-${SCRATCH}/verl-apertus/checkpoints/apertus8b-cof-sft}"

has_hf_weights() {
    local path="$1"
    [ -f "$path/config.json" ] && { [ -f "$path/model.safetensors.index.json" ] || compgen -G "$path/*.safetensors" >/dev/null; }
}

find_latest_sft_hf_checkpoint() {
    local root="$1"
    local latest_file="$root/latest_checkpointed_iteration.txt"
    local step candidate ckpt best_step best_path

    if [ -f "$latest_file" ]; then
        step="$(tr -dc '0-9' < "$latest_file")"
        candidate="$root/global_step_${step}/huggingface"
        if [ -n "$step" ] && has_hf_weights "$candidate"; then
            echo "$candidate"
            return 0
        fi
    fi

    best_step=-1
    best_path=""
    shopt -s nullglob
    for ckpt in "$root"/global_step_*; do
        step="${ckpt##*_}"
        candidate="$ckpt/huggingface"
        if [[ "$step" =~ ^[0-9]+$ ]] && [ "$step" -gt "$best_step" ] && has_hf_weights "$candidate"; then
            best_step="$step"
            best_path="$candidate"
        fi
    done
    shopt -u nullglob

    if [ -n "$best_path" ]; then
        echo "$best_path"
        return 0
    fi
    return 1
}

if [ -z "${RL_MODEL_PATH:-}" ]; then
    if SFT_MODEL_PATH="$(find_latest_sft_hf_checkpoint "$SFT_CHECKPOINT_DIR")"; then
        RL_MODEL_PATH="$SFT_MODEL_PATH"
        echo "Using SFT HF checkpoint for RL: $RL_MODEL_PATH"
    else
        RL_MODEL_PATH="$BASE_MODEL_PATH"
        echo "No complete SFT HF checkpoint found under $SFT_CHECKPOINT_DIR; using base model: $RL_MODEL_PATH"
    fi
else
    echo "Using RL_MODEL_PATH override: $RL_MODEL_PATH"
fi


# --- Pre-flight checks ---
echo "=== Pre-flight Checks ==="
ERROR_COUNT=0
check_path() {
    local label="$1" path="$2"
    if [ -e "$path" ]; then
        echo "  OK: $label"
    else
        echo "  FAIL: $label ($path)"
        ((ERROR_COUNT++))
    fi
}

check_path "verl repo" "$VERL/verl/__init__.py"
check_path "verl-apertus tools" "$PROJECT/tools/image_zoom_in_emu_tool.py"
check_path "tool config yaml" "$PROJECT/configs/cof_rl_tool_config.yaml"
check_path "trainer yaml" "$PROJECT/configs/cof_rl_grpo.yaml"
check_path "reward fn" "$PROJECT/rewards/cof_rl_reward.py"
check_path "train parquet" "$PROJECT/data_prep/cof_rl/train.parquet"
check_path "val parquet" "$PROJECT/data_prep/cof_rl/val.parquet"
check_path "Emu3.5 vision_tokenizer" "$EMU3_SRC/vision_tokenizer/__init__.py"
check_path "RL model config" "$RL_MODEL_PATH/config.json"
check_path "RL model tokenizer" "$RL_MODEL_PATH/tokenizer_config.json"

if [ $ERROR_COUNT -gt 0 ]; then
    echo "ERROR: $ERROR_COUNT pre-flight checks failed."
    exit 1
fi

echo ""
echo "=== GPU Info ==="
nvidia-smi --list-gpus
echo ""

echo "== Job started at $(date)"
echo "=== Starting GRPO training ==="
# Run from PROJECT so `python -m` does not put /workspace/src/verl (the
# container's bundled, outdated verl) on sys.path[0] and shadow the editable
# install at /capstor/.../verl.
cd "$PROJECT"
python3 -m verl.trainer.main_ppo \
    --config-path="$PROJECT/configs" \
    --config-name=cof_rl_grpo \
    actor_rollout_ref.model.path="$RL_MODEL_PATH" \
    "$@"

echo "=== Job ended at $(date) ==="
echo "=== Job Complete ==="
