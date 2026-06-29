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
import os
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

# Split policy (v3): three disjoint sets, deterministic from `seed`.
#   SFT  = DEFAULT_N_SFT rows sampled from GQA-with-thought (only GQA ships a
#          non-empty `thought`, the reasoning chain SFT needs).
#   TEST = DEFAULT_N_TEST rows sampled from the remaining (non-SFT) pool, reserved
#          as a clean held-out eval set — excluded from BOTH SFT and RL so the
#          in-distribution eval never overlaps training.
#   RL   = everything else (all subsets incl. leftover GQA, all single-word).
# SFT selection is identical to v2 (same seed/rng draw), so re-splitting keeps the
# SFT set stable; TEST is carved next, then RL = rest.
SPLIT_VERSION = 3
DEFAULT_N_SFT = 10000
DEFAULT_N_TEST = 3000


def is_sft_eligible(row: dict) -> bool:
    """SFT requires a reasoning chain: only GQA rows carry a non-empty `thought`."""
    if row.get("dataset") != "gqa":
        return False
    t = row.get("thought")
    return isinstance(t, str) and bool(t.strip())


def sft_eligible_indices(rows: list[dict]) -> list[int]:
    """Raw-index order is stable (== file line index), so this is deterministic."""
    return [row["_raw_index"] for row in rows if is_sft_eligible(row)]


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


def scale_bbox(bbox, src_wh, dst_wh) -> list[int]:
    """Scale an [x1, y1, x2, y2] box from src (w, h) pixels to dst (w, h) pixels.

    The model perceives the smart_resize'd image (its IBQ token grid encodes the
    resized pixels), so the gold bbox must be expressed in that same resized
    space for both SFT targets and the RL reward to point where the model sees.
    """
    sw, sh = src_wh
    dw, dh = dst_wh
    sx, sy = dw / sw, dh / sh
    x1, y1, x2, y2 = bbox
    return [round(x1 * sx), round(y1 * sy), round(x2 * sx), round(y2 * sy)]


def build_messages(row: dict, image_token_str: str, bbox) -> list[dict]:
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
    rows: list[dict],
    path: Path,
    seed: int = 42,
    n_sft: int = DEFAULT_N_SFT,
    n_test: int = DEFAULT_N_TEST,
) -> dict:
    """Deterministic SFT / TEST / RL split (v3).

    SFT  = `seed`-deterministic sample of `n_sft` SFT-eligible (GQA-with-thought)
           rows (identical draw to v2, so the SFT set is stable across versions).
    TEST = `n_test` rows sampled from the remaining non-SFT pool — a clean held-out
           eval set, excluded from both SFT and RL.
    RL   = every remaining raw index (non-SFT, non-TEST).
    All three parses call this with the full `rows`, so they agree regardless of
    run order. A file from an older format / mismatched config is regenerated.
    """
    n_rows = len(rows)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            split = json.load(f)
        if (
            split.get("version") == SPLIT_VERSION
            and split.get("n_rows") == n_rows
            and split.get("n_sft") == n_sft
            and split.get("n_test") == n_test
            and split.get("seed") == seed
        ):
            return split
        print(
            f"[split] Regenerating {path}: stale/old format "
            f"(version={split.get('version')!r}, n_rows={split.get('n_rows')!r}, "
            f"n_sft={split.get('n_sft')!r}, n_test={split.get('n_test')!r}, "
            f"seed={split.get('seed')!r})"
        )

    eligible = sft_eligible_indices(rows)  # GQA-with-thought, raw-index order
    if len(eligible) < n_sft:
        raise ValueError(
            f"Only {len(eligible)} SFT-eligible (gqa+thought) rows; need n_sft={n_sft}"
        )

    rng = np.random.default_rng(seed)
    sft_chosen = rng.choice(np.asarray(eligible, dtype=np.int64), size=n_sft, replace=False)
    sft_set = {int(i) for i in sft_chosen}

    non_sft = [i for i in range(n_rows) if i not in sft_set]
    if len(non_sft) < n_test:
        raise ValueError(f"Only {len(non_sft)} non-SFT rows; need n_test={n_test}")
    test_chosen = rng.choice(np.asarray(non_sft, dtype=np.int64), size=n_test, replace=False)
    test_set = {int(i) for i in test_chosen}
    rl_indices = [i for i in non_sft if i not in test_set]  # non-SFT minus TEST

    split = {
        "version": SPLIT_VERSION,
        "seed": seed,
        "n_rows": n_rows,
        "n_sft": n_sft,
        "n_test": n_test,
        "policy": "sft=sample(gqa_with_thought,n_sft); test=sample(non_sft,n_test); rl=non_sft-test",
        "sft_indices": sorted(sft_set),
        "test_indices": sorted(test_set),
        "rl_indices": rl_indices,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # Per-process tmp name + atomic rename: SFT prep and the 8 RL shard procs may
    # all regenerate concurrently; each writes its own tmp (identical content) and
    # atomically renames, so the final file is never torn. Last writer wins.
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(split, f)
    tmp.replace(path)
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
    parser.add_argument("--n_sft", type=int, default=DEFAULT_N_SFT,
                        help="Number of GQA-with-thought rows to sample for SFT")
    parser.add_argument("--n_test", type=int, default=DEFAULT_N_TEST,
                        help="Number of held-out TEST rows reserved (excluded from SFT and RL)")
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
    split = get_or_create_split_indices(rows, split_path, seed=args.seed, n_sft=args.n_sft, n_test=args.n_test)
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
    from inference.vision import encode_image, load_vq_model, smart_resize

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
                # Express the gold bbox in the resized (perceived) image space.
                resized = smart_resize(img)
                bbox = scale_bbox(row["bboxs"][0], img.size, resized.size)
                image_token_str = encode_image(resized, vq_model)
                messages = build_messages(row, image_token_str, bbox)
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
                    "bbox": bbox,
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
