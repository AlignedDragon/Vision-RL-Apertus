#!/bin/bash

set -euo pipefail
ulimit -n 65535


PROJECT="/users/$USER/capscratch/verl-apertus"
EMU3_SRC="/users/$USER/capscratch/Emu3.5/src"

# Note: using $USER means the script is portable across cluster users.
# Adjust EMU3_SRC if your install lives elsewhere.

export PYTHONPATH="${PROJECT}:${EMU3_SRC}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME="${SCRATCH}/hf_home"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HYDRA_FULL_ERRO=1 # for error tracking

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

check_path "verl-apertus tools" "$PROJECT/tools/image_zoom_in_emu_tool.py"
check_path "tool config yaml" "$PROJECT/configs/cof_rl_tool_config.yaml"
check_path "trainer yaml" "$PROJECT/configs/cof_rl_grpo.yaml"
check_path "reward fn" "$PROJECT/rewards/cof_rl_reward.py"
check_path "train parquet" "$PROJECT/datasets/cof_rl/train.parquet"
check_path "val parquet" "$PROJECT/datasets/cof_rl/val.parquet"
check_path "Emu3.5 vision_tokenizer" "$EMU3_SRC/vision_tokenizer/__init__.py"

if [ $ERROR_COUNT -gt 0 ]; then
    echo "ERROR: $ERROR_COUNT pre-flight checks failed."
    exit 1
fi

# Verify the reward fn imports and self-tests pass.
python3 "$PROJECT/rewards/cof_rl_reward.py"

echo ""
echo "=== GPU Info ==="
nvidia-smi --list-gpus
echo ""

echo "== Job started at $(date)"
echo "=== Starting GRPO training ==="
python3 -m verl.trainer.main_ppo \
    --config-path="$PROJECT/configs" \
    --config-name=cof_rl_grpo "$@"

echo "=== Job ended at $(date) ==="
echo "=== Job Complete ==="
