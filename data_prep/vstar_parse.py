"""Build a V*Bench (craigwu/vstar_bench) eval set in the CoF-RL parquet schema.

Produces data_prep/vstar_bench/val.parquet, consumable UNCHANGED by BOTH:
  - inference/run_cof_hf.py            (HF multi-turn tool-agent eval)
  - verl main_ppo trainer.val_only    (sglang agent-loop eval, cof_rl_grpo.yaml)

V*Bench items are MCQ whose gold ('label') is a single letter (A/B/C/D). That
matches the model's display_answers single-token output and the exact-match
scoring used by both cof_rl_reward.compute_score and run_cof_hf.py. We reuse the
exact prompt-rendering + record-building helpers from cof_rl_parse so the schema
and Apertus chat-template framing are identical to the RL data.

Run on a GPU node inside the verl_env container (needs the Emu3.5 VQ model):
    python data_prep/vstar_parse.py
"""
import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.vision import encode_image, load_vq_model, smart_resize
from data_prep.cof_rl_parse import (
    load_tool_schemas,
    render_apertus_prompt,
    build_system_message,
    _build_parquet_record,
    APERTUS_INSTRUCTION,
)


def resolve_snapshot() -> str:
    """Local cached snapshot dir for craigwu/vstar_bench (no network)."""
    from huggingface_hub import snapshot_download
    return snapshot_download("craigwu/vstar_bench", repo_type="dataset", local_files_only=True)


def main():
    ap = argparse.ArgumentParser(description="Render V*Bench prompts in Apertus CoF format")
    ap.add_argument("--snapshot", default=None, help="vstar_bench snapshot dir (default: resolve from HF cache)")
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "data_prep/vstar_bench"))
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs/apertus.yaml"))
    ap.add_argument("--tool_config", default=str(PROJECT_ROOT / "configs/cof_rl_tool_config.yaml"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-patches", dest="max_patches", type=int, default=256,
                    help="input-image token cap (256 = CoF budget, default; 2048 = high-res baseline eval set)")
    args = ap.parse_args()

    snap = Path(args.snapshot or resolve_snapshot())
    questions = snap / "test_questions.jsonl"
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(questions) if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"Loaded {len(rows)} V*Bench questions from {questions}", flush=True)

    tool_schemas = load_tool_schemas(args.tool_config)
    print(f"Tool schemas: {', '.join(s['name'] for s in tool_schemas)}", flush=True)

    from transformers import AutoTokenizer
    print(f"Loading Apertus tokenizer from {cfg['model']['checkpoint']} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg["model"]["checkpoint"], trust_remote_code=True)
    print(f"Loading IBQ vision tokenizer from {cfg['model']['vq_model']} ...", flush=True)
    vq = load_vq_model(cfg["model"]["vq_model"], device="cuda:0")
    print("Models loaded", flush=True)

    records, skipped, cat_counts = [], 0, {}
    meta_path = out_dir / "metadata.jsonl"
    with open(meta_path, "w", encoding="utf-8") as mf:
        for i, row in enumerate(rows):
            img_path = snap / row["image"]
            if not img_path.exists():
                print(f"  SKIP {i}: missing image {img_path}", flush=True)
                skipped += 1
                continue
            try:
                image = Image.open(img_path).convert("RGB")
                # CoF image-encoding budget: 256 IBQ tokens (match CoF training); --max-patches
                # 2048 builds the high-res baseline eval set.
                token_str = encode_image(smart_resize(image, max_patches=args.max_patches), vq, max_patches=args.max_patches)
            except Exception as e:
                print(f"  SKIP {i}: IBQ encode failed: {e}", flush=True)
                skipped += 1
                continue

            # V*Bench 'text' already carries the question + lettered options +
            # "Answer with the option's letter ... directly." We prepend the IBQ
            # image tokens and append the display_answers instruction (CoF style).
            user_msg = {
                "role": "user",
                "content": f"{token_str}\n{row['text']}\n\n{APERTUS_INSTRUCTION}",
            }
            prompt_str = render_apertus_prompt(tok, build_system_message(), user_msg, tool_schemas)

            meta = {
                "question_id": row.get("question_id", i),
                "prompt": prompt_str,
                "image_path": str(img_path),
                "reward_model": {"ground_truth": row["label"], "style": "rule"},
                "data_source": "vstar_bench",
                "ability": row.get("category", ""),
                "agent_name": "cof_tool_agent",
                "extra_info": {},
            }
            mf.write(json.dumps(meta, ensure_ascii=False) + "\n")
            rec = _build_parquet_record(meta, "val")
            rec["extra_info"]["category"] = row.get("category", "")
            records.append(rec)
            cat_counts[row.get("category", "")] = cat_counts.get(row.get("category", ""), 0) + 1
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(rows)}] skipped={skipped}", flush=True)

    print(f"Built {len(records)} records, skipped {skipped}, categories={cat_counts}", flush=True)
    pq.write_table(pa.Table.from_pylist(records), out_dir / "val.parquet")
    print(f"Wrote {out_dir / 'val.parquet'}", flush=True)


if __name__ == "__main__":
    main()
