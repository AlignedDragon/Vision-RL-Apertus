#!/bin/bash
#SBATCH --job-name=prepare_vcot_test
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

# Render the held-out TEST split (split_indices.json test_indices, ~3k rows) into a
# single in-distribution eval parquet: data_prep/vcot_test/test.parquet. Uses the
# EXACT VCoT training prompt (vcot_rl_parse rendering), so the eval is consistent
# with training. ~3k images VQ-encode in ~15min on 1 GH200.

set -euo pipefail
echo "=== Job started at $(date '+%Y-%m-%d %H:%M:%S') ==="

source /users/msayfiddinov/capscratch/miniconda/etc/profile.d/conda.sh
conda activate verl

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

echo "=== Pre-flight Checks ==="
ERROR_COUNT=0
check_path() { if [ -e "$2" ]; then echo "  OK: $1"; else echo "  FAIL: $1 ($2)"; ((ERROR_COUNT++)); fi; }
check_path "Apertus checkpoint" "$APERTUS_CHECKPOINT/config.json"
check_path "VQ model" "$VQ_MODEL_PATH/model.ckpt"
check_path "Emu3.5 vision_tokenizer" "$EMU3_SRC/vision_tokenizer/__init__.py"
check_path "raw.jsonl" "$BASE/data_prep/vcot/raw.jsonl"
check_path "images dir" "$BASE/data_prep/vcot/images"
check_path "VCOT tool config" "$BASE/configs/vcot_rl_tool_config.yaml"
python3 -c "import einops" 2>/dev/null && echo "  OK: einops" || { echo "  FAIL: einops"; ((ERROR_COUNT++)); }
[ $ERROR_COUNT -gt 0 ] && { echo "ERROR: $ERROR_COUNT pre-flight checks failed."; exit 1; }

echo ""
nvidia-smi --list-gpus
echo "=== Rendering TEST split ==="
python3 data_prep/vcot_rl_parse.py --config configs/apertus.yaml --render-split test
echo ""
echo "=== Job ended at $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "=== Job Complete ==="
