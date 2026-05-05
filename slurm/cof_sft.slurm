#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --account=infra01
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=460000
#SBATCH --time=10:00:00
#SBATCH --environment=verl
#SBATCH --partition=normal
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

echo "=== Job started at $(date '+%Y-%m-%d %H:%M:%S') ==="

# --- Conda ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate verl

# --- Paths ---
BASE="/users/$USER/capscratch/verl-apertus"
cd "$BASE"

EMU3_SRC="/users/$USER/capscratch/Emu3.5/src"
export PYTHONPATH="${EMU3_SRC}:${BASE}:${PYTHONPATH:-}"

APERTUS_CHECKPOINT="/capstor/store/cscs/swissai/infra01/MLLM/ablations/apertus-8b-img-SFT-32nodes-gbs512-mbs1-steps8030-img-text-seqlen8192-s2onlytxtloss/HF"

export HF_HOME="${SCRATCH}/hf_home"
export TRANSFORMERS_CACHE="${HF_HOME}"
export PYTHONUNBUFFERED=1

# Clean stale torch extension caches (known GH200 issue)
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
check_path "SFT train parquet" "$BASE/datasets/cof_sft/train.parquet"
check_path "SFT val parquet" "$BASE/datasets/cof_sft/val.parquet"
check_path "Custom dataset class" "$BASE/datasets/cof_sft_dataset.py"
check_path "Hydra config" "$BASE/configs/sft_apertus_cof.yaml"

python3 -c "import verl, transformers" 2>/dev/null \
    && echo "  OK: verl + transformers importable" \
    || { echo "  FAIL: verl or transformers not importable"; ((ERROR_COUNT++)); }

if [ $ERROR_COUNT -gt 0 ]; then
    echo ""
    echo "ERROR: $ERROR_COUNT pre-flight checks failed."
    exit 1
fi

echo ""
echo "=== GPU Info ==="
nvidia-smi --list-gpus
echo ""

# --- Rendezvous setup for multi-node torchrun ---
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
HEAD_ADDR=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address | awk '{print $1}')
RDZV_PORT=29500
RDZV_ID=${SLURM_JOB_ID}

echo "=== Distributed Setup ==="
echo "  nodes=$SLURM_NNODES  gpus_per_node=$SLURM_GPUS_ON_NODE"
echo "  head_node=$HEAD_NODE  head_addr=$HEAD_ADDR  port=$RDZV_PORT"
echo ""

# --- Launch SFT ---
echo "=== Starting SFT Training ==="
srun --nodes="$SLURM_NNODES" --ntasks-per-node=1 --gpus-per-node="$SLURM_GPUS_ON_NODE" \
    bash -c "torchrun \
        --nnodes=$SLURM_NNODES \
        --nproc-per-node=$SLURM_GPUS_ON_NODE \
        --rdzv-id=$RDZV_ID \
        --rdzv-backend=c10d \
        --rdzv-endpoint=$HEAD_ADDR:$RDZV_PORT \
        -m verl.trainer.sft_trainer \
        --config-path=$BASE/configs \
        --config-name=cof_sft"

echo ""
echo "=== Job ended at $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "=== Job Complete ==="
