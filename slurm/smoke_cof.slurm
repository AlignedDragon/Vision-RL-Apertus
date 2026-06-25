#!/bin/bash
#SBATCH --job-name=smoke_cof
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200000
#SBATCH --time=00:30:00
#SBATCH --environment=verl_env
#SBATCH --partition=normal
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

echo "=== Smoke started at $(date '+%Y-%m-%d %H:%M:%S') ==="

source /users/msayfiddinov/capscratch/miniconda/etc/profile.d/conda.sh
conda activate verl

PROJECT="/users/$USER/capscratch/verl-apertus"
VERL="/users/$USER/capscratch/verl"
EMU3_SRC="/users/$USER/capscratch/Emu3.5/src"
cd "$PROJECT"
export PYTHONPATH="${PROJECT}:${VERL}:${EMU3_SRC}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME="${SCRATCH}/hf_home"
export TRANSFORMERS_CACHE="${HF_HOME}"
rm -rf ~/.cache/torch_extensions/ 2>/dev/null || true

echo "=== GPU Info ==="
nvidia-smi --list-gpus
echo ""

# --- 1) Smoke parse of the merged smoke raw (CoF-RL + DeepEyes subset) ---
echo "=== Smoke parse (raw_smoke.jsonl) ==="
python3 data_prep/cof_rl_parse.py \
    --config configs/apertus.yaml \
    --input data_prep/cof_rl/raw_smoke.jsonl \
    --output data_prep/cof_rl_smoke/metadata.jsonl
echo ""

# --- 2) Inspect smoke parquet: schema + image_path points to ORIGINAL file ---
echo "=== Parquet sanity ==="
python3 - <<'PY'
import pyarrow.parquet as pq
from PIL import Image
t = pq.read_table("data_prep/cof_rl_smoke/train.parquet")
print("train rows:", t.num_rows, "cols:", t.column_names)
row = {k: t.column(k)[0].as_py() for k in t.column_names}
ti = row["extra_info"]["tools_kwargs"]["image_zoom_in_tool"]["create_kwargs"]["image_path"]
im = Image.open(ti)
print("sample image_path:", ti)
print("  exists, size:", im.size, "(NOT necessarily mult-of-16 => original)")
# data_source histogram
import collections
ds = collections.Counter(t.column("data_source").to_pylist()
                         + pq.read_table("data_prep/cof_rl_smoke/val.parquet").column("data_source").to_pylist())
print("data_source histogram:", dict(ds))
PY
echo ""

# --- 3) Zoom tool unit test (crop from original + bbox rescale) ---
echo "=== Tool unit test ==="
python3 tests/smoke_zoom_tool.py
echo ""

echo "=== Smoke ended at $(date '+%Y-%m-%d %H:%M:%S') ==="
