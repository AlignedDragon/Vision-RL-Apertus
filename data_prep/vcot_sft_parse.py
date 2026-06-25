"""Parse Visual-CoT raw.jsonl into Apertus-formatted SFT examples.

This mirrors the CoF SFT data-prep flow but uses Visual-CoT rows:
  - deterministic SFT:RL split of 1:10 over the filtered raw rows
  - one image per row, encoded as inline IBQ tokens
  - tools loaded from configs/vcot_rl_tool_config.yaml
  - assistant turn: thoughts, full answer, display_answers call, draw_bbox_tool call

Output records:
    {
      "text": str,
      "image_paths": list[str],
      "raw_index": int,
      "answer": str,
      "bbox": list[float]
    }

Usage (interactive on a GPU node):
    python data_prep/vcot_sft_parse.py --limit 5
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

APERTUS_SYSTEM = "You are a helpful assistant with access to tools."
APERTUS_INSTRUCTION = (
    "Call the display_answers tool exactly once at the end of your response, "
    "passing your final answer as the single element of the `answers` argument.\n"
    "Draw a bounding box around the region you used to determine your answer"
)

ENUM_RE = re.compile(r"(?:(?<=^)|(?<=[\s\n]))\d+\.\s*")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_tool_schemas(path: str) -> list[dict]:
    """Load unwrapped tool schemas from a verl tool config YAML."""
    with open(path) as f:
        config = yaml.safe_load(f)
    tools = config.get("tools") or []
    schemas: list[dict] = []
    for i, tool in enumerate(tools):
        if "tool_schema" not in tool:
            raise ValueError(f"Missing tool_schema for tool entry {i} in {path}")
        schema = tool["tool_schema"]
        if schema.get("type") == "function" and "function" in schema:
            schema = schema["function"]
        if not isinstance(schema, dict) or not schema.get("name"):
            raise ValueError(f"Invalid tool schema for tool entry {i} in {path}")
        schemas.append(schema)

    names = {schema["name"] for schema in schemas}
    required = {"draw_bbox_tool", "display_answers"}
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"Missing required tool schema(s) in {path}: {', '.join(missing)}")
    return schemas


def strip_thought_enumeration(text: str) -> str:
    text = ENUM_RE.sub("", text.strip())
    return re.sub(r"\s+", " ", text).strip()


def build_user_text(question: str, image_token_str: str) -> str:
    return f"{image_token_str} Question: {question.strip()}\n\n{APERTUS_INSTRUCTION}"


def build_messages(row: dict, image_token_str: str) -> list[dict]:
    bbox = row["bboxs"][0]
    display_args = json.dumps({"answers": [row["answer"]]}, ensure_ascii=False)
    bbox_args = json.dumps({"bbox_2d": bbox}, ensure_ascii=False)

    # Only GQA (cot_with_detailed_reasoning_steps) ships a `thought` reasoning
    # chain and a sentence-form `full_answer`; the other 11 subsets carry only
    # question/answer/bbox. Emit the thoughts block only when reasoning exists,
    # and fall back to the short `answer` for the response text when there is no
    # `full_answer`.
    blocks: list[dict] = []
    thought = row.get("thought")
    if isinstance(thought, str) and thought.strip():
        blocks.append({"type": "thoughts", "text": strip_thought_enumeration(thought)})
    full_answer = row.get("full_answer")
    response_text = (
        full_answer.strip()
        if isinstance(full_answer, str) and full_answer.strip()
        else row["answer"].strip()
    )
    blocks.append({"type": "response", "text": response_text})
    blocks.append({"type": "tool_calls", "calls": [
        {"name": "draw_bbox_tool", "arguments": bbox_args},
        {"name": "display_answers", "arguments": display_args},
    ]})
    return [
        {"role": "system", "content": APERTUS_SYSTEM},
        {"role": "user", "content": build_user_text(row["question"], image_token_str)},
        {"role": "assistant", "content": {"blocks": blocks}},
        {"role": "tool", "content": json.dumps("Bounding box drawn")},
        {"role": "tool", "content": json.dumps("Answer displayed")},
    ]


def load_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_raw_index"] = i
            rows.append(row)
    return rows


def get_or_create_split_indices(
    n_rows: int,
    path: Path,
    seed: int,
    sft_weight: int = 1,
    rl_weight: int = 10,
) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            split = json.load(f)
        if split.get("n_rows") != n_rows:
            raise ValueError(
                f"{path} was built for {split.get('n_rows')} rows, current raw has {n_rows}"
            )
        return split

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_rows).tolist()
    n_sft = (n_rows * sft_weight) // (sft_weight + rl_weight)
    split = {
        "seed": seed,
        "ratio": {"sft": sft_weight, "rl": rl_weight},
        "n_rows": n_rows,
        "sft_indices": sorted(perm[:n_sft]),
        "rl_indices": sorted(perm[n_sft:]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(split, f)
    return split


def resolve_image_path(row: dict, images_root: Path) -> Path:
    """Resolve a row to images_root/<dataset>/<basename>.

    Images were extracted into per-dataset folders, so resolution must key on
    (dataset, basename): bare basenames collide across subsets (COCO/OpenImages
    ids recur) and several subsets reuse the same physical image pool. The
    output folder name matches the row's `dataset` field (e.g. visual7w rows are
    labelled `v7w`).
    """
    dataset = row.get("dataset")
    if not dataset:
        raise KeyError(f"row missing 'dataset' field: keys={sorted(row)}")
    base = Path(row["image"]).name
    path = images_root / dataset / base
    if path.exists():
        return path
    raise FileNotFoundError(
        f"Could not find image dataset={dataset!r} base={base!r} under {images_root}"
    )


def main():
    parser = argparse.ArgumentParser(description="Render Visual-CoT SFT examples in Apertus format")
    parser.add_argument("--input", default=None, help="Default: data_prep/vcot/raw.jsonl")
    parser.add_argument("--output", default=None, help="Default: data_prep/vcot_sft/metadata.jsonl")
    parser.add_argument("--images-root", default=None, help="Default: data_prep/vcot/images")
    parser.add_argument("--split-indices", default=None, help="Default: data_prep/vcot/split_indices.json")
    parser.add_argument("--config", default="configs/apertus.yaml")
    parser.add_argument("--tool-config", default="configs/vcot_rl_tool_config.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N SFT rows (debug)")
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_dir = PROJECT_ROOT / "data_prep" / "vcot"
    out_dir = PROJECT_ROOT / "data_prep" / "vcot_sft"
    input_path = Path(args.input) if args.input else raw_dir / "raw.jsonl"
    output_path = Path(args.output) if args.output else out_dir / "metadata.jsonl"
    images_root = Path(args.images_root) if args.images_root else raw_dir / "images"
    split_path = Path(args.split_indices) if args.split_indices else raw_dir / "split_indices.json"

    rows = load_rows(input_path)
    split = get_or_create_split_indices(len(rows), split_path, seed=args.seed)
    sft_indices = set(split["sft_indices"])
    rows = [row for row in rows if row["_raw_index"] in sft_indices]
    if args.limit:
        rows = rows[: args.limit]
    print(f"Loaded {len(rows)} SFT rows from {input_path}")
    print(f"Split file: {split_path} (SFT={len(split['sft_indices'])}, RL={len(split['rl_indices'])})")

    tool_schemas = load_tool_schemas(args.tool_config)
    print(f"Loaded tool schemas from {args.tool_config}: "
          f"{', '.join(schema['name'] for schema in tool_schemas)}")

    config = load_config(args.config)
    from PIL import Image
    from transformers import AutoTokenizer
    from inference.vision import encode_image, load_vq_model

    print(f"Loading Apertus tokenizer from {config['model']['checkpoint']} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["checkpoint"], trust_remote_code=True
    )

    print(f"Loading IBQ vision tokenizer from {config['model']['vq_model']} ...")
    vq_model = load_vq_model(config["model"]["vq_model"], device="cuda:0")
    print("Models loaded")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    skipped = 0
    text_lens: list[int] = []
    start = time.time()

    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, row in enumerate(rows):
            try:
                src_path = resolve_image_path(row, images_root)
                img = Image.open(src_path).convert("RGB")
                image_token_str = encode_image(img, vq_model)
                messages = build_messages(row, image_token_str)
                text = tokenizer.apply_chat_template(
                    messages,
                    tools=tool_schemas,
                    enable_thinking=True,
                    add_generation_prompt=False,
                    tokenize=False,
                )
                if not text.rstrip().endswith("<|assistant_end|>"):
                    text = text + "<|assistant_end|>"

                out_f.write(json.dumps({
                    "text": text,
                    "image_paths": [str(src_path)],
                    "raw_index": row["_raw_index"],
                    "answer": row["answer"],
                    "bbox": row["bboxs"][0],
                }, ensure_ascii=False) + "\n")
                text_lens.append(len(text))
            except Exception as e:
                print(f"  SKIP row {row.get('_raw_index', i)}: {e}")
                skipped += 1
                continue

            if (i + 1) % 50 == 0 or i == len(rows) - 1:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                print(f"  [{i+1}/{len(rows)}] {rate:.2f} rows/s  skipped={skipped}")

    written = len(rows) - skipped
    print(f"\nWrote {written} records to {output_path}")
    if skipped:
        print(f"Skipped {skipped} rows")

    if text_lens:
        text_lens.sort()
        n = len(text_lens)
        print(f"\ntext char-length stats: min={text_lens[0]} "
              f"p50={text_lens[n // 2]} p95={text_lens[int(n * 0.95)]} "
              f"max={text_lens[-1]}")

    meta_rows: list[dict] = []
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                meta_rows.append(json.loads(line))

    if meta_rows:
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(meta_rows))
        n_val = max(1, int(round(len(meta_rows) * args.val_ratio))) if len(meta_rows) > 1 else 0
        val_set = set(perm[:n_val].tolist())
        train_recs = [m for i, m in enumerate(meta_rows) if i not in val_set]
        val_recs = [m for i, m in enumerate(meta_rows) if i in val_set]

        train_out = output_path.parent / "train.parquet"
        val_out = output_path.parent / "val.parquet"
        pq.write_table(pa.Table.from_pylist(train_recs), train_out)
        pq.write_table(pa.Table.from_pylist(val_recs), val_out)
        print(f"\nWrote {len(train_recs)} rows to {train_out}")
        print(f"Wrote {len(val_recs)} rows to {val_out}")


if __name__ == "__main__":
    main()
