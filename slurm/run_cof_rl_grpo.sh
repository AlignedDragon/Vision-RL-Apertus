#!/bin/bash
#SBATCH --job-name=cof_rl_grpo
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=460000
#SBATCH --time=24:00:00
#SBATCH --environment=verl_env
#SBATCH --partition=normal
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
ulimit -n 65535

source /users/msayfiddinov/capscratch/miniconda/etc/profile.d/conda.sh
conda activate verl

PROJECT="/users/$USER/verl-apertus"
VERL="/users/$USER/capscratch/verl"
EMU3_SRC="/users/$USER/Emu3.5/src"

# Note: using $USER means the script is portable across cluster users.
# Adjust EMU3_SRC if your install lives elsewhere.

export PYTHONPATH="${PROJECT}:${VERL}:${EMU3_SRC}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME="${SCRATCH}/hf_home"
export TRANSFORMERS_CACHE="${HF_HOME}"

# Pre-import the custom agent loop so the @register decorator runs before
# main_ppo reads each row's agent_name.
export VERL_AGENT_LOOPS_EXTRA="agent_loops.cof_tool_agent_loop"

cd "$VERL"

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
check_path "verl-apertus agent loop" "$PROJECT/agent_loops/cof_tool_agent_loop.py"
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

# Verify the custom agent loop registers cleanly (catches drift early).
python3 -c "
import sys
sys.path.insert(0, '$PROJECT')
import agent_loops.cof_tool_agent_loop as _
print('cof_tool_agent loop imported OK')
"

# Verify the reward fn imports and self-tests pass.
python3 "$PROJECT/rewards/cof_rl_reward.py"

echo ""
echo "=== GPU Info ==="
nvidia-smi --list-gpus
echo ""

echo "=== Starting GRPO training ==="
python3 -m verl.trainer.main_ppo \
    --config-path="$PROJECT/configs" \
    --config-name=cof_rl_grpo "$@"

echo "=== Job Complete ==="
