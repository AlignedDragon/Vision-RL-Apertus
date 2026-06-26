#!/bin/bash
#SBATCH --job-name=prepare_refcocop_eval
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=460000
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

# Absolute path (not ${SCRATCH}, which may be unset in the container) so the
# RefCOCO+ snapshot cached on the login node is found on the compute node.
export HF_HOME="/capstor/scratch/cscs/${USER}/hf_home"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

rm -rf ~/.cache/torch_extensions/ 2>/dev/null || true

# Use the already-cached lmms-lab/RefCOCOplus snapshot (downloaded on login node).
# Glob loop (not `ls | head`) so an empty match doesn't trip `set -o pipefail`.
SNAP=""
for d in "${HF_HOME}"/hub/datasets--lmms-lab--RefCOCOplus/snapshots/*/; do
    [ -d "$d" ] && { SNAP="${d%/}"; break; }
done
echo "RefCOCO+ snapshot: ${SNAP:-<none, will fall back to cached hf_hub_download>}"

LIMIT="${REFCOCOP_LIMIT:-1000}"
echo "=== Building RefCOCO+ val eval set (limit=$LIMIT) ==="
python3 data_prep/refcocop_parse.py ${SNAP:+--snapshot "$SNAP"} --limit "$LIMIT"

echo ""
echo "=== Job ended at $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "=== Job Complete ==="
