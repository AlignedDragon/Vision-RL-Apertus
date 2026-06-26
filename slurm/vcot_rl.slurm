#!/bin/bash
#SBATCH --job-name=vcot_rl
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=460000
#SBATCH --time=08:00:00
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

# Stability env (see verl-apertus-rl-stability): OTel off avoids a Ray
# AgentLoopWorker segfault; no expandable_segments (NCCL invalid usage at FSDP init).
export OTEL_SDK_DISABLED=true
export OTEL_TRACES_EXPORTER=none
export OTEL_METRICS_EXPORTER=none
export OTEL_LOGS_EXPORTER=none
export GRPC_ENABLE_FORK_SUPPORT=0
export RAY_experimental_enable_open_telemetry=0
export RAY_enable_open_telemetry=0
export NCCL_DEBUG=WARN
export HF_HOME="${SCRATCH}/hf_home"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HYDRA_FULL_ERROR=1
export WANDB_MODE=online

BASE_MODEL_PATH="/capstor/store/cscs/swissai/infra01/MLLM/ablations/apertus-8b-img-SFT-32nodes-gbs512-mbs1-steps8030-img-text-seqlen8192-s2onlytxtloss/HF"
SFT_CHECKPOINT_DIR="${SFT_CHECKPOINT_DIR:-${PROJECT}/checkpoints/apertus8b-vcot-sft}"
RL_CHECKPOINT_DIR="${RL_CHECKPOINT_DIR:-${PROJECT}/checkpoints/vcot_rl_apertus/apertus8b_grpo_vcot_rl}"
RL_RESUME_MODE="${RL_RESUME_MODE:-auto}"

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

# "SFT then RL": initialise GRPO from the VCoT SFT HF checkpoint. Falls back to the
# base model only if no SFT checkpoint exists (override with RL_MODEL_PATH).
if [ -z "${RL_MODEL_PATH:-}" ]; then
    if SFT_MODEL_PATH="$(find_latest_sft_hf_checkpoint "$SFT_CHECKPOINT_DIR")"; then
        RL_MODEL_PATH="$SFT_MODEL_PATH"
        echo "Using VCoT SFT HF checkpoint for RL: $RL_MODEL_PATH"
    else
        RL_MODEL_PATH="$BASE_MODEL_PATH"
        echo "No complete VCoT SFT HF checkpoint found under $SFT_CHECKPOINT_DIR; using base model: $RL_MODEL_PATH"
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
check_path "draw_bbox tool" "$PROJECT/tools/image_draw_bbox_tool.py"
check_path "tool config yaml" "$PROJECT/configs/vcot_rl_tool_config.yaml"
check_path "trainer yaml" "$PROJECT/configs/vcot_rl_grpo.yaml"
check_path "reward fn" "$PROJECT/rewards/vcot_rl_reward.py"
check_path "train parquet" "$PROJECT/data_prep/vcot_rl/train.parquet"
check_path "val parquet" "$PROJECT/data_prep/vcot_rl/val.parquet"
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
echo "RL checkpoint dir: $RL_CHECKPOINT_DIR"
echo "RL resume mode: $RL_RESUME_MODE"
if [ -f "$RL_CHECKPOINT_DIR/latest_checkpointed_iteration.txt" ]; then
    echo "Found RL checkpoint tracker: $(cat "$RL_CHECKPOINT_DIR/latest_checkpointed_iteration.txt")"
else
    echo "No RL checkpoint tracker found; verl will start from scratch if resume_mode=auto."
fi
# Run from PROJECT so `python -m` does not put the container's bundled, outdated
# verl on sys.path[0] and shadow the editable install.
cd "$PROJECT"
# save_freq=50 guarantees a checkpoint lands inside the 8h wall-clock window.
python3 -m verl.trainer.main_ppo \
    --config-path="$PROJECT/configs" \
    --config-name=vcot_rl_grpo \
    actor_rollout_ref.model.path="$RL_MODEL_PATH" \
    trainer.default_local_dir="$RL_CHECKPOINT_DIR" \
    trainer.resume_mode="$RL_RESUME_MODE" \
    trainer.save_freq=50 \
    "$@"

echo "=== Job ended at $(date) ==="
echo "=== Job Complete ==="
