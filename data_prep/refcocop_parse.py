"""Build a RefCOCO+ (val) referring-expression-grounding eval set in the VCoT-RL
parquet schema.

Produces data_prep/refcocop/val.parquet, consumable by verl main_ppo
trainer.val_only via configs/vcot_rl_grpo.yaml (sglang agent-loop eval), scored by
rewards/refcocop_reward.compute_score (Acc@0.5 IoU on the draw_bbox prediction).

RefCOCO+ is referring-expression comprehension: an image + a referring expression
-> localise the referred region. We render the same Apertus tool-agent prompt the
VCoT RL data uses (system + draw_bbox_tool/display_answers schemas + a user block
of IBQ image tokens, the expression framed as a locate instruction, and the
draw-bbox instruction), so base/SFT/RL are all evaluated in-distribution. The gold
box ([x,y,w,h], original pixels) is converted to [x1,y1,x2,y2] and scaled into the
smart_resize'd (perceived) pixel space via scale_bbox -- matching what the model
emits and what the reward's IoU compares against -- and carried in extra_info["bbox"].

Annotation source: HuggingFace `lmms-lab/RefCOCOplus` (val split, ~3805 referred
objects). Each row carries the COCO image bytes, the gold `bbox` ([x,y,w,h]) and an
`answer` list of referring expressions. We take one expression per object (the
first) and optionally sample `--limit` of them (eval rollouts are slow). The IBQ
tokens are computed from the embedded image; `image_path` points at the matching
COCO train2014 file on store (the draw_bbox tool is a no-op stub at rollout, so the
path is only forward-compat metadata).

Run on a GPU node inside the verl_env container (needs the Emu3.5 VQ model):
    python data_prep/refcocop_parse.py --limit 1000
"""
import argparse
import glob
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_prep.vcot_rl_parse import (
    load_tool_schemas,
    render_apertus_prompt,
    _build_parquet_record,
)
from data_prep.vcot_sft_parse import scale_bbox

DEFAULT_COCO_IMAGES = "/capstor/store/cscs/swissai/infra01/datasets/coco/train2014"

# RefCOCO+ REC: the ANSWER is the bounding box, so the reward scores ONLY the bbox
# (Acc@0.5). But keep the full draw_bbox + display_answers instruction the models were
# SFT/RL-trained on: dropping the display_answers tail makes the prompt
# out-of-distribution and SFT format-collapses (stops emitting the box). The
# display_answers output is simply ignored by the bbox-only reward.
INSTRUCTION = (
    "Draw a bounding box around the region described by calling the draw_bbox_tool, "
    "then call the display_answers tool exactly once with a single word naming it."
)


def build_user_text(expr: str, image_token_str: str) -> str:
    return (
        f"{image_token_str} Locate the region described by: \"{expr.strip()}\"\n\n"
        f"{INSTRUCTION}"
    )


def resolve_val_shards(snapshot: str | None) -> list[str]:
    if snapshot:
        shards = sorted(glob.glob(str(Path(snapshot) / "data/val-*.parquet")))
        if shards:
            return shards
    from huggingface_hub import hf_hub_download
    out = []
    for shard in ["data/val-00000-of-00002.parquet", "data/val-00001-of-00002.parquet"]:
        out.append(hf_hub_download("lmms-lab/RefCOCOplus", shard, repo_type="dataset"))
    return out


def coco_store_path(file_name: str, coco_images: Path) -> str:
    """Map a RefCOCO+ `file_name` (COCO_train2014_xxx_<refidx>.jpg) to the COCO file."""
    base = re.sub(r"_\d+\.jpg$", ".jpg", file_name)
    return str(coco_images / base)


def main():
    ap = argparse.ArgumentParser(description="Render RefCOCO+ val grounding prompts in VCoT format")
    ap.add_argument("--snapshot", default=None,
                    help="lmms-lab/RefCOCOplus snapshot dir (default: resolve/download from HF cache)")
    ap.add_argument("--coco-images", default=DEFAULT_COCO_IMAGES)
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "data_prep/refcocop"))
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs/apertus.yaml"))
    ap.add_argument("--tool_config", default=str(PROJECT_ROOT / "configs/vcot_rl_tool_config.yaml"))
    ap.add_argument("--limit", type=int, default=1000,
                    help="Random sample of this many objects (0 = all ~3805). Eval is slow; 1000 gives ~1.5%% SE.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    coco_images = Path(args.coco_images)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    shards = resolve_val_shards(args.snapshot)
    print(f"val shards: {shards}", flush=True)

    rows = []
    for s in shards:
        rows.extend(pq.read_table(s).to_pylist())
    print(f"Loaded {len(rows)} RefCOCO+ val objects", flush=True)

    # One expression per referred object (the first non-empty), then sample.
    examples = []
    for r in rows:
        answers = r.get("answer") or []
        expr = next((str(a).strip() for a in answers if str(a).strip()), "")
        if not expr or not r.get("bbox") or len(r["bbox"]) != 4:
            continue
        examples.append({
            "expr": expr,
            "bbox_xywh": r["bbox"],
            "file_name": r["file_name"],
            "image_bytes": r["image"]["bytes"],
            "question_id": r.get("question_id"),
        })
    print(f"Usable objects (one expr each): {len(examples)}", flush=True)

    rng = np.random.default_rng(args.seed)
    if args.limit and 0 < args.limit < len(examples):
        idx = sorted(rng.permutation(len(examples))[: args.limit].tolist())
        examples = [examples[i] for i in idx]
        print(f"Sampled {len(examples)} objects (seed={args.seed})", flush=True)

    tool_schemas = load_tool_schemas(args.tool_config)
    print(f"Tool schemas: {', '.join(s['name'] for s in tool_schemas)}", flush=True)

    from transformers import AutoTokenizer
    from inference.vision import encode_image, load_vq_model, smart_resize

    print(f"Loading Apertus tokenizer from {cfg['model']['checkpoint']} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg["model"]["checkpoint"], trust_remote_code=True)
    print(f"Loading IBQ vision tokenizer from {cfg['model']['vq_model']} ...", flush=True)
    vq = load_vq_model(cfg["model"]["vq_model"], device="cuda:0")
    print("Models loaded", flush=True)

    records, skipped = [], 0
    meta_path = out_dir / "metadata.jsonl"
    with open(meta_path, "w", encoding="utf-8") as mf:
        for i, ex in enumerate(examples):
            try:
                image = Image.open(io.BytesIO(ex["image_bytes"])).convert("RGB")
                resized = smart_resize(image)
                x, y, w, h = ex["bbox_xywh"]
                gold = scale_bbox([x, y, x + w, y + h], image.size, resized.size)
                token_str = encode_image(resized, vq)
            except Exception as e:
                skipped += 1
                print(f"  SKIP {i}: encode failed: {e}", flush=True)
                continue

            user_content = build_user_text(ex["expr"], token_str)
            prompt_str = render_apertus_prompt(tok, user_content, tool_schemas)
            img_path = coco_store_path(ex["file_name"], coco_images)

            meta = {
                "question_id": ex.get("question_id", i),
                "prompt": prompt_str,
                "image_path": img_path,
                "reward_model": {"style": "rule", "ground_truth": ex["expr"]},
                "data_source": "refcocop",
                "ability": "refcocop_val",
                "agent_name": "vcot_tool_agent",
                "extra_info": {
                    "index": str(i),
                    "answer": ex["expr"],
                    "dataset": "refcocop",
                    "file_name": ex["file_name"],
                    "bbox": gold,
                    "image_wh": [resized.size[0], resized.size[1]],
                },
            }
            mf.write(json.dumps(meta, ensure_ascii=False) + "\n")
            records.append(_build_parquet_record(meta, "val"))
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(examples)}] skipped={skipped}", flush=True)

    print(f"Built {len(records)} records, skipped {skipped}", flush=True)
    pq.write_table(pa.Table.from_pylist(records), out_dir / "val.parquet")
    print(f"Wrote {out_dir / 'val.parquet'}", flush=True)


if __name__ == "__main__":
    main()
