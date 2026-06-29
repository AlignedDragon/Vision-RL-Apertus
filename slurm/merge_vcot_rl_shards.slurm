#!/bin/bash
#SBATCH --job-name=merge_vcot_rl
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200000
#SBATCH --time=01:00:00
#SBATCH --environment=verl_env
#SBATCH --partition=normal
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Stitch the 8 RL shard metadata files -> metadata.jsonl -> train/val parquet.
# No GPU/model load (the --merge branch returns before any model is touched), but
# we request 1 GPU so the verl_env container brings up identically to the shards.
# Submit with: --dependency=afterok:<prepare_vcot_rl array job id>.

set -euo pipefail

echo "=== Merge started at $(date '+%Y-%m-%d %H:%M:%S') ==="

source /users/msayfiddinov/capscratch/miniconda/etc/profile.d/conda.sh
conda activate verl

BASE="/users/$USER/capscratch/verl-apertus"
cd "$BASE"
EMU3_SRC="/users/$USER/capscratch/Emu3.5/src"
export PYTHONPATH="${EMU3_SRC}:${BASE}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

NUM_SHARDS=8
echo "=== Merging $NUM_SHARDS shards ==="
python3 data_prep/vcot_rl_parse.py --config configs/apertus.yaml --merge --num-shards "$NUM_SHARDS"

# Gate the RL job: load the written parquet EXACTLY as verl does (datasets.load_dataset
# + select + filter). Fails the job if the nested-chunked-array bug recurs, so the
# afterok-dependent vcot_rl never launches on an unloadable dataset.
echo "=== Validating parquet loads via HuggingFace datasets (verl path) ==="
python3 - <<'PYEOF'
import datasets
for f in ["data_prep/vcot_rl/train.parquet", "data_prep/vcot_rl/val.parquet"]:
    ds = datasets.load_dataset("parquet", data_files=f)["train"]
    n = len(ds)
    ds.select(range(min(100, n)))
    ds.select(range(min(5000, n))).filter(lambda _x: True, num_proc=1)
    print(f"VALIDATED {f}: {n} rows load+select+filter OK", flush=True)
PYEOF

echo "=== Merge ended at $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "=== Job Complete ==="
