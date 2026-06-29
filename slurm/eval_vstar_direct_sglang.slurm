#!/bin/bash
#SBATCH --job-name=eval_vstar_direct_sglang
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

# Direct (no-tool) V*Bench MCQ eval on the sglang offline Engine (sglang ONLY; the HF
# run_direct_mcq.py is intentionally not used). Loops base/sft/rl over one parquet. Base is
# scored from its free-text response (it doesn't emit display_answers).
#
# NOTE (env): the offline sglang.Engine needs psutil+xgrammar C-exts that are broken in the
# current verl env. `.pylibs` (isolated psutil) is prepended below; xgrammar may still block
# the Engine -- if so, get direct numbers via eval_vstar_verl.slurm (verl sglang server) and
# re-score the dump from the response. See memory verl-env-broken-cexts.
#
# Usage:
#   sbatch slurm/eval_vstar_direct_sglang.slurm                       # 256 set (default)
#   VAL=$BASE/data_prep/vstar_bench_2048/val.parquet SUFFIX=2048 sbatch slurm/eval_vstar_direct_sglang.slurm
# Override model paths with BASE_M / SFT_M / RL_M if auto-discovery is wrong.

set -euo pipefail
ulimit -n 65535
source /users/msayfiddinov/capscratch/miniconda/etc/profile.d/conda.sh
conda activate verl

BASE="/users/$USER/capscratch/verl-apertus"; cd "$BASE"
EMU3_SRC="/users/$USER/capscratch/Emu3.5/src"
# .pylibs first: isolated psutil shadows the env's broken _psutil_linux C-ext (the offline
# sglang.Engine workers import psutil). Does NOT touch the shared conda env. Create with:
#   pip install --target=$BASE/.pylibs --no-deps psutil
export PYTHONPATH="${BASE}/.pylibs:${BASE}:${EMU3_SRC}:${PYTHONPATH:-}"
export HF_HOME="/capstor/scratch/cscs/$USER/hf_home"
export TRANSFORMERS_CACHE="${HF_HOME}"
export PYTHONUNBUFFERED=1
export OTEL_SDK_DISABLED=true
rm -rf ~/.cache/torch_extensions/ 2>/dev/null || true

VAL="${VAL:-$BASE/data_prep/vstar_bench/val.parquet}"
SUFFIX="${SUFFIX:-256}"

BASE_M="${BASE_M:-/capstor/store/cscs/swissai/infra01/MLLM/ablations/apertus-8b-img-SFT-32nodes-gbs512-mbs1-steps8030-img-text-seqlen8192-s2onlytxtloss/HF}"

find_latest_sft() {
    local root="$1"
    local latest="$root/latest_checkpointed_iteration.txt"
    local step best=-1 bestp="" c
    if [ -f "$latest" ]; then
        step="$(tr -dc '0-9' < "$latest")"
        c="$root/global_step_${step}/huggingface"
        [ -n "$step" ] && [ -f "$c/config.json" ] && { echo "$c"; return 0; }
    fi
    shopt -s nullglob
    for d in "$root"/global_step_*; do
        step="${d##*_}"; c="$d/huggingface"
        [[ "$step" =~ ^[0-9]+$ ]] && [ "$step" -gt "$best" ] && [ -f "$c/config.json" ] && { best="$step"; bestp="$c"; }
    done
    shopt -u nullglob
    [ -n "$bestp" ] && { echo "$bestp"; return 0; }
    return 1
}

SFT_M="${SFT_M:-$(find_latest_sft "$BASE/checkpoints/apertus8b-cof-sft-256" || find_latest_sft "$BASE/checkpoints/apertus8b-cof-sft" || true)}"
RL_M="${RL_M:-$BASE/checkpoints/cof_rl_apertus_256/apertus8b_grpo_cof_rl/global_step_50/actor_hf_merged}"

echo "=== direct sglang eval $(date) ===  VAL=$VAL SUFFIX=$SUFFIX"
echo "BASE_M=$BASE_M"; echo "SFT_M=$SFT_M"; echo "RL_M=$RL_M"
nvidia-smi --list-gpus

for pair in "base:$BASE_M" "sft:$SFT_M" "rl:$RL_M"; do
    lab="${pair%%:*}"; m="${pair#*:}"
    if [ -z "$m" ] || [ ! -f "$m/config.json" ]; then
        echo "===== SKIP $lab (no ckpt at '$m') ====="; continue
    fi
    echo "===== DIRECT-sglang eval ${lab}_${SUFFIX} ($m) ====="
    python3 inference/run_direct_mcq_sglang.py --model "$m" --label "${lab}_${SUFFIX}" --val "$VAL"
done
echo "=== done ($(date)) ==="
