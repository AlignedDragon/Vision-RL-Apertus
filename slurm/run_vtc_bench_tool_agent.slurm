#!/bin/bash
#SBATCH --job-name=apertus_vtc_bench_tool_agent
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=460000
#SBATCH --time=8:00:00
#SBATCH --environment=verl_env
#SBATCH --partition=normal
#SBATCH --gpus-per-node=4
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

set -euo pipefail

# --- Conda ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate verl

# --- Paths ---
BASE="/users/$USER/verl-apertus"
cd "$BASE"

EMU3_SRC="/users/$USER/Emu3.5/src"
VTC_BENCH="/capstor/scratch/cscs/$USER/VTC-Bench/eval"
export PYTHONPATH="${EMU3_SRC}:${VTC_BENCH}:${BASE}:${PYTHONPATH:-}"

APERTUS_CHECKPOINT="/capstor/store/cscs/swissai/infra01/MLLM/ablations/apertus-8b-img-SFT-32nodes-gbs512-mbs1-steps8030-img-text-seqlen8192-s2onlytxtloss/HF"
VQ_MODEL_PATH="/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/Emu3.5-VisionTokenizer"
export VQ_MODEL_PATH

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
check_path "VQ model" "$VQ_MODEL_PATH/model.ckpt"
check_path "Emu3.5 vision_tokenizer" "$EMU3_SRC/vision_tokenizer/__init__.py"
check_path "Dataset metadata" "$BASE/datasets/vtc_bench/metadata.jsonl"
check_path "VTC-Bench opencv_interface" "$VTC_BENCH/qwen_agent/tools/opencv_interface.py"

python3 -c "import einops; import cv2" 2>/dev/null && echo "  OK: deps installed" || { echo "  FAIL: missing deps (einops/opencv-python)"; ((ERROR_COUNT++)); }

if [ $ERROR_COUNT -gt 0 ]; then
    echo ""
    echo "ERROR: $ERROR_COUNT pre-flight checks failed. Fix and retry."
    exit 1
fi

echo ""
echo "=== GPU Info ==="
nvidia-smi --list-gpus
echo ""

# --- Run inference ---
echo "=== Starting VTC-Bench Tool-Agent Inference ==="
python3 inference/run_vtc_bench_tool_agent.py --config configs/apertus.yaml
echo ""

# --- Run evaluation ---
echo "=== Starting Evaluation ==="
python3 evaluation/compute_vtc_accuracy.py --predictions results/vtc_bench/tool_agent/predictions.jsonl
echo ""

echo "=== Job Complete ==="
