#!/bin/bash
#SBATCH --job-name=eval_vstar_direct
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
ulimit -n 65535
source /users/msayfiddinov/capscratch/miniconda/etc/profile.d/conda.sh
conda activate verl

BASE="/users/$USER/capscratch/verl-apertus"; cd "$BASE"
EMU3_SRC="/users/$USER/capscratch/Emu3.5/src"
export PYTHONPATH="${EMU3_SRC}:${BASE}:${PYTHONPATH:-}"
export HF_HOME="/capstor/scratch/cscs/$USER/hf_home"
export PYTHONUNBUFFERED=1
rm -rf ~/.cache/torch_extensions/ 2>/dev/null || true

BASE_M="/capstor/store/cscs/swissai/infra01/MLLM/ablations/apertus-8b-img-SFT-32nodes-gbs512-mbs1-steps8030-img-text-seqlen8192-s2onlytxtloss/HF"
SFT_M="$BASE/checkpoints/apertus8b-cof-sft/global_step_118/huggingface"
RL_M="$BASE/checkpoints/cof_rl_apertus/apertus8b_grpo_cof_rl/global_step_50/actor_hf_merged"

nvidia-smi --list-gpus
for pair in "base:$BASE_M" "sft:$SFT_M" "rl:$RL_M"; do
    lab="${pair%%:*}"; m="${pair#*:}"
    echo "===== DIRECT eval $lab ====="
    python3 inference/run_direct_mcq.py --model "$m" --label "$lab"
done
echo "=== done ==="
