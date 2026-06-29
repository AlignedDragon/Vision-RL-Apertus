#!/bin/bash
#SBATCH --job-name=eval_vcot_test
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=460000
#SBATCH --time=04:00:00
#SBATCH --environment=verl_env
#SBATCH --partition=normal
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# In-distribution VCoT eval: run a checkpoint over the held-out TEST split
# (data_prep/vcot_test/test.parquet) with the SAME prompt/task as training, and
# score answer-accuracy and grounding Acc@0.5 SEPARATELY via vcot_eval_reward.
# Mirrors eval_refcocop_verl: a val_only main_ppo pass with the reward overridden.
#   Usage: LABEL=base  MODEL=base       sbatch slurm/eval_vcot_test.slurm
#          LABEL=sft   MODEL=auto-sft   sbatch slurm/eval_vcot_test.slurm
#          LABEL=rl    MODEL=auto-rl    sbatch slurm/eval_vcot_test.slurm   # needs merged RL ckpt

set -euo pipefail
ulimit -n 65535
source /users/msayfiddinov/capscratch/miniconda/etc/profile.d/conda.sh
conda activate verl

PROJECT="/users/$USER/capscratch/verl-apertus"
VERL="/users/$USER/capscratch/verl"
EMU3_SRC="/users/$USER/capscratch/Emu3.5/src"
export PYTHONPATH="${PROJECT}:${VERL}:${EMU3_SRC}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OTEL_SDK_DISABLED=true
export OTEL_TRACES_EXPORTER=none
export OTEL_METRICS_EXPORTER=none
export OTEL_LOGS_EXPORTER=none
export GRPC_ENABLE_FORK_SUPPORT=0
export RAY_experimental_enable_open_telemetry=0
export RAY_enable_open_telemetry=0
export NCCL_DEBUG=WARN
export HF_HOME="${SCRATCH}/hf_home"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline

LABEL="${LABEL:?set LABEL=<base|sft|rl>}"
VAL="${VAL:-$PROJECT/data_prep/vcot_test/test.parquet}"

BASE_MODEL_PATH="/capstor/store/cscs/swissai/infra01/MLLM/ablations/apertus-8b-img-SFT-32nodes-gbs512-mbs1-steps8030-img-text-seqlen8192-s2onlytxtloss/HF"
SFT_CKPT_DIR="${SFT_CKPT_DIR:-${PROJECT}/checkpoints/apertus8b-vcot-sft}"
RL_CKPT_DIR="${RL_CKPT_DIR:-${PROJECT}/checkpoints/vcot_rl_apertus/apertus8b_grpo_vcot_rl}"

latest_step_subdir() {  # <root> <leaf> -> <root>/global_step_<max>/<leaf> if it has weights
    local root="$1" leaf="$2" best=-1 best_path="" ckpt step cand
    shopt -s nullglob
    for ckpt in "$root"/global_step_*; do
        step="${ckpt##*_}"; cand="$ckpt/$leaf"
        if [[ "$step" =~ ^[0-9]+$ ]] && [ "$step" -gt "$best" ] && [ -f "$cand/config.json" ]; then
            best="$step"; best_path="$cand"
        fi
    done
    shopt -u nullglob
    [ -n "$best_path" ] && { echo "$best_path"; return 0; } || return 1
}

MODEL="${MODEL:?set MODEL=<hf ckpt dir|base|auto-sft|auto-rl>}"
case "$MODEL" in
    base)     MODEL="$BASE_MODEL_PATH" ;;
    auto-sft) MODEL="$(latest_step_subdir "$SFT_CKPT_DIR" huggingface)" || { echo "no SFT ckpt under $SFT_CKPT_DIR"; exit 1; } ;;
    auto-rl)  MODEL="$(latest_step_subdir "$RL_CKPT_DIR" actor_hf_merged)" || { echo "no MERGED RL ckpt under $RL_CKPT_DIR (run merge_vcot_rl_checkpoint.slurm first)"; exit 1; } ;;
esac

echo "=== eval_vcot_test $(date) ==="
echo "MODEL=$MODEL  LABEL=$LABEL  VAL=$VAL"
[ -f "$VAL" ] || { echo "FAIL: test parquet missing: $VAL (run prepare_vcot_test.slurm)"; exit 1; }
nvidia-smi --list-gpus

cd "$PROJECT"
# val_only pass over the in-distribution TEST set; vcot_eval_reward returns a dict
# so verl logs val-core/vcot_rl/{score=Acc@0.5, answer_acc, grounding_acc, iou}.
# validation_data_dir captures per-sample (input/output/gts/score+extras) so the
# base model's free-text answers can be inspected / re-scored offline.
python3 -m verl.trainer.main_ppo \
    --config-path="$PROJECT/configs" \
    --config-name=vcot_rl_grpo \
    actor_rollout_ref.model.path="$MODEL" \
    data.train_files="$VAL" \
    data.val_files="$VAL" \
    reward.custom_reward_function.path="$PROJECT/rewards/vcot_eval_reward.py" \
    reward.custom_reward_function.name=compute_score \
    trainer.val_only=true \
    trainer.val_before_train=true \
    trainer.logger='[console]' \
    trainer.validation_data_dir="$PROJECT/slurm/eval_logs/vcot_test_${LABEL}" \
    trainer.log_val_generations=10 \
    trainer.project_name=vcot_test_eval \
    trainer.experiment_name="vcot_test_eval_${LABEL}" \
    "$@"

echo "=== done ($(date)) ==="
