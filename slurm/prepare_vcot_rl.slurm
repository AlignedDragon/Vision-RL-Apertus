#!/bin/bash
#SBATCH --job-name=prepare_vcot_rl
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=460000
#SBATCH --time=04:00:00
#SBATCH --environment=verl_env
#SBATCH --partition=normal
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Full RL split, sharded across 8 GPUs as an 8-way job array (one single-GPU task
# per shard — the proven 1-GPU prep pattern, replicated). VQ-encode runs ~3.9
# img/s on 1 GH200; ~24.4k rows/shard -> ~1.7h. We use ALL the single-word RL data
# (no cap). Each task writes its own atomic shard file; finalize with the dependent
# merge job (slurm/merge_vcot_rl_shards.slurm). The shards/ dir is cleaned ONCE by
# the submitter BEFORE this array starts (NOT here — array tasks must not race).

set -euo pipefail

echo "=== Array task ${SLURM_ARRAY_TASK_ID} started at $(date '+%Y-%m-%d %H:%M:%S') ==="

# --- Conda ---
source /users/msayfiddinov/capscratch/miniconda/etc/profile.d/conda.sh
conda activate verl

# --- Paths ---
BASE="/users/$USER/capscratch/verl-apertus"
cd "$BASE"

EMU3_SRC="/users/$USER/capscratch/Emu3.5/src"
export PYTHONPATH="${EMU3_SRC}:${BASE}:${PYTHONPATH:-}"

APERTUS_CHECKPOINT="/capstor/store/cscs/swissai/infra01/MLLM/ablations/apertus-8b-img-SFT-32nodes-gbs512-mbs1-steps8030-img-text-seqlen8192-s2onlytxtloss/HF"
VQ_MODEL_PATH="/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/Emu3.5-VisionTokenizer"

export HF_HOME="${SCRATCH}/hf_home"
export TRANSFORMERS_CACHE="${HF_HOME}"
export PYTHONUNBUFFERED=1

rm -rf ~/.cache/torch_extensions/ 2>/dev/null || true

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

check_path "Apertus checkpoint" "$APERTUS_CHECKPOINT/config.json"
check_path "Apertus tokenizer" "$APERTUS_CHECKPOINT/tokenizer_config.json"
check_path "VQ model" "$VQ_MODEL_PATH/model.ckpt"
check_path "Emu3.5 vision_tokenizer" "$EMU3_SRC/vision_tokenizer/__init__.py"
check_path "raw.jsonl from download step" "$BASE/data_prep/vcot/raw.jsonl"
check_path "images dir from download step" "$BASE/data_prep/vcot/images"
check_path "VCOT tool config" "$BASE/configs/vcot_rl_tool_config.yaml"

python3 -c "import einops" 2>/dev/null && echo "  OK: einops installed" || { echo "  FAIL: einops not installed"; ((ERROR_COUNT++)); }

if [ $ERROR_COUNT -gt 0 ]; then
    echo ""
    echo "ERROR: $ERROR_COUNT pre-flight checks failed."
    echo "Hint: run data_prep/vcot_download.py on the login node first."
    exit 1
fi

echo ""
echo "=== GPU Info ==="
nvidia-smi --list-gpus
echo ""

# --- Run this shard ---
NUM_SHARDS=8
VCOT_RL_LIMIT="${VCOT_RL_LIMIT:-0}"   # 0 = full RL split (default); >0 = debug subset
LIMIT_ARG=""
if [ "$VCOT_RL_LIMIT" != "0" ]; then
    LIMIT_ARG="--limit $VCOT_RL_LIMIT"
fi

echo "=== Shard ${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS} parse (limit='${VCOT_RL_LIMIT}') ==="
python3 data_prep/vcot_rl_parse.py --config configs/apertus.yaml \
    --num-shards "$NUM_SHARDS" --shard-index "$SLURM_ARRAY_TASK_ID" $LIMIT_ARG
echo ""

echo "=== Array task ${SLURM_ARRAY_TASK_ID} ended at $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "=== Shard Complete (merge separately with slurm/merge_vcot_rl_shards.slurm) ==="
