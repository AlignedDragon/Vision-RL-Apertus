#!/bin/bash
#SBATCH --job-name=qwen_vqa_baseline
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=460000
#SBATCH --time=2:00:00
#SBATCH --environment=verl_env
#SBATCH --partition=normal
#SBATCH --gpus-per-node=4
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

set -euo pipefail

echo "=== Job started at $(date '+%Y-%m-%d %H:%M:%S') ==="

# --- Conda ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate verl

# --- Paths ---
BASE="/users/$USER/capscratch/verl-apertus"
cd "$BASE"

export PYTHONPATH="${BASE}:${PYTHONPATH:-}"
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

# Check datasets exist
for DATASET in textvqa hrbench vstar pope; do
    check_path "Dataset $DATASET" "$BASE/data_prep/$DATASET/metadata.jsonl"
done

# Check dependencies
python3 -c "from qwen_vl_utils import process_vision_info" 2>/dev/null && echo "  OK: qwen-vl-utils" || { echo "  FAIL: qwen-vl-utils not installed (pip install qwen-vl-utils)"; ((ERROR_COUNT++)); }

if [ $ERROR_COUNT -gt 0 ]; then
    echo ""
    echo "ERROR: $ERROR_COUNT pre-flight checks failed. Fix and retry."
    exit 1
fi

echo ""
echo "=== GPU Info ==="
nvidia-smi --list-gpus
echo ""

# --- Run inference on all datasets ---
for DATASET in textvqa hrbench vstar pope; do
    echo "=== Inference: $DATASET ==="
    python3 inference/run_qwen_baseline.py \
        --config configs/qwen.yaml \
        --dataset-dir "data_prep/$DATASET"
    echo ""
done

# --- Run evaluation ---
echo "=== Evaluation ==="

# TextVQA — VQA soft accuracy
python3 evaluation/compute_accuracy.py \
    --predictions results/textvqa/qwen_baseline/predictions.jsonl \
    --mode vqa
echo ""

# HR-Bench — MCQ exact match
python3 evaluation/compute_accuracy.py \
    --predictions results/hrbench/qwen_baseline/predictions.jsonl \
    --mode mcq
echo ""

# VStar — MCQ exact match
python3 evaluation/compute_accuracy.py \
    --predictions results/vstar/qwen_baseline/predictions.jsonl \
    --mode mcq
echo ""

# POPE — Binary yes/no + F1
python3 evaluation/compute_accuracy.py \
    --predictions results/pope/qwen_baseline/predictions.jsonl \
    --mode binary
echo ""

echo "=== Job ended at $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "=== Job Complete ==="

