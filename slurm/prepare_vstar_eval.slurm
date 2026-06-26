#!/bin/bash
#SBATCH --job-name=prepare_vstar
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120000
#SBATCH --time=01:00:00
#SBATCH --environment=verl_env
#SBATCH --partition=normal
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
echo "=== Job started at $(date '+%Y-%m-%d %H:%M:%S') ==="

source /users/msayfiddinov/capscratch/miniconda/etc/profile.d/conda.sh
conda activate verl

BASE="/users/$USER/capscratch/verl-apertus"
cd "$BASE"
EMU3_SRC="/users/$USER/capscratch/Emu3.5/src"
export PYTHONPATH="${EMU3_SRC}:${BASE}:${PYTHONPATH:-}"
export HF_HOME="${SCRATCH:-/capstor/scratch/cscs/$USER}/hf_home"
export TRANSFORMERS_CACHE="${HF_HOME}"
export PYTHONUNBUFFERED=1
rm -rf ~/.cache/torch_extensions/ 2>/dev/null || true

echo "=== GPU Info ===" && nvidia-smi --list-gpus && echo ""

echo "=== Building V*Bench eval parquet ==="
VSTAR_SNAP="/capstor/scratch/cscs/msayfiddinov/hf_home/hub/datasets--craigwu--vstar_bench/snapshots/d9ae62c903da0c98336e85c5ee89cd863b04b4da"
python3 data_prep/vstar_parse.py --snapshot "$VSTAR_SNAP"

echo "=== Job ended at $(date '+%Y-%m-%d %H:%M:%S') ==="
