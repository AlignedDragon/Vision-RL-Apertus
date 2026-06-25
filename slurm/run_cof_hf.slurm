#!/bin/bash
#SBATCH --job-name=cof_hf
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120000
#SBATCH --time=00:30:00
#SBATCH --environment=verl_env
#SBATCH --partition=normal
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
ulimit -n 65535

source /users/msayfiddinov/capscratch/miniconda/etc/profile.d/conda.sh
conda activate verl

PROJECT="/users/$USER/capscratch/verl-apertus"
EMU3_SRC="/users/$USER/capscratch/Emu3.5/src"
export PYTHONPATH="${PROJECT}:${EMU3_SRC}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME="${SCRATCH}/hf_home"

MODEL="${MODEL:-${PROJECT}/checkpoints/cof_rl_apertus/apertus8b_grpo_cof_rl/global_step_100/actor_hf_merged}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"

cd "$PROJECT"
echo "== Job start $(date) -- model=$MODEL, num_samples=$NUM_SAMPLES"
python3 inference/run_cof_hf.py \
    --model "$MODEL" \
    --val "$PROJECT/data_prep/cof_rl/val.parquet" \
    --tool-config "$PROJECT/configs/cof_rl_tool_config.yaml" \
    --num-samples "$NUM_SAMPLES" \
    "$@"
echo "== Job end $(date)"
