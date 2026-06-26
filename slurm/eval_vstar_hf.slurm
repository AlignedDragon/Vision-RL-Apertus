#!/bin/bash
#SBATCH --job-name=eval_vstar_hf
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120000
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

BASE="/users/$USER/capscratch/verl-apertus"
cd "$BASE"
EMU3_SRC="/users/$USER/capscratch/Emu3.5/src"
export PYTHONPATH="${EMU3_SRC}:${BASE}:${PYTHONPATH:-}"
export HF_HOME="/capstor/scratch/cscs/$USER/hf_home"
export PYTHONUNBUFFERED=1
rm -rf ~/.cache/torch_extensions/ 2>/dev/null || true

MODEL="${MODEL:?set MODEL=<hf ckpt dir>}"
LABEL="${LABEL:?set LABEL=<base|sft|rl>}"
VAL="${VAL:-$BASE/data_prep/vstar_bench/val.parquet}"
NUM="${NUM:-191}"
TEMP="${TEMP:-1.0}"
SEED="${SEED:-0}"
OUT="${OUT:-$BASE/results/vstar/${LABEL}_hf_temp${TEMP}_seed${SEED}.jsonl}"
mkdir -p "$(dirname "$OUT")"

echo "=== eval_vstar_hf $(date) ==="
echo "MODEL=$MODEL"
echo "LABEL=$LABEL TEMP=$TEMP SEED=$SEED VAL=$VAL -> $OUT"
nvidia-smi --list-gpus

python3 inference/run_cof_hf.py \
    --model "$MODEL" \
    --val "$VAL" \
    --tool-config "$BASE/configs/cof_rl_tool_config.yaml" \
    --num-samples "$NUM" \
    --temperature "$TEMP" \
    --seed "$SEED" \
    --output "$OUT"

echo "=== done -> $OUT ($(date)) ==="
