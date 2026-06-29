"""Parse Visual-CoT raw.jsonl into Apertus-formatted RL metadata + parquet.

Mirrors the CoF-RL data-prep flow (`data_prep/cof_rl_parse.py`) but consumes the
Visual-CoT rows produced by `vcot_download.py` (all 12 subsets, filtered to a
single-word answer + single bbox).

The SFT and RL stages share one deterministic split (`data_prep/vcot/split_indices.json`,
seed 42, sft:rl = 1:10). `vcot_sft_parse.py` renders the `sft_indices`; this script
renders the disjoint `rl_indices`.

Per RL row, render an Apertus 3-block prompt:
  - system:    "You are a helpful assistant with access to tools."
  - developer: draw_bbox_tool + display_answers, loaded from
               configs/vcot_rl_tool_config.yaml and rendered via the chat template's tools=
  - user:      IBQ vision tokens for the image, the question, and an instruction to
               draw a bounding box around the evidence region and emit the final
               answer via the display_answers tool.

Output records (metadata.jsonl):
    {
      "question_id": int,
      "prompt": str,                                 # fully rendered Apertus prompt
      "image_path": str,                             # absolute
      "reward_model": {"style": "rule", "ground_truth": str},
      "data_source": "vcot_rl",
      "ability": str,                                # subset (gqa, vsr, ...)
      "agent_name": "vcot_tool_agent",
      "extra_info": {"index", "answer", "dataset", "bbox", "image_wh"}
    }

The reward is `rewards/vcot_rl_reward.py` (`compute_score`): a half/half score of
0.5 * answer_match + 0.5 * IoU(pred_bbox, gold_bbox). The answer half matches the
last display_answers call against `ground_truth`; the bbox half is the IoU of the
last draw_bbox_tool call's bbox_2d against the gold `bbox` carried here in
`extra_info` (both in the resized/perceived image space, i.e. smart_resize'd
pixels). The draw_bbox_tool itself is a no-op stub at rollout — only the emitted
bbox_2d coordinates are scored.

Usage (interactive on a GPU node):
    python data_prep/vcot_rl_parse.py --limit 5

Usage (SLURM): clone slurm/prepare_vcot_sft.slurm and swap the script.
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Share the constants/helpers and the SFT:RL split with the SFT parse so the two
# stages stay disjoint and resolve images the same way. Heavy deps (torch, PIL,
# transformers, vision) are imported inside main().
from data_prep.vcot_sft_parse import (
    APERTUS_SYSTEM,
    DEFAULT_N_SFT,
    DEFAULT_N_TEST,
    load_config,
    load_tool_schemas,
    load_rows,
    get_or_create_split_indices,
    resolve_image_path,
    scale_bbox,
)

# RL keeps only single-word ground truths (the filter already enforces this), so
# the instruction asks for a single word; the display_answers call carries it.
APERTUS_INSTRUCTION = (
    "Draw a bounding box around the region you use to determine your answer by "
    "calling the draw_bbox_tool. Then call the display_answers tool exactly once "
    "at the end of your response, passing your final answer as a single word in "
    "the `answers` argument."
)

USER_BLOCK = re.compile(r"<\|user_start\|>(.*?)<\|user_end\|>", re.DOTALL)


def build_user_text(question: str, image_token_str: str) -> str:
    """Apertus user block: IBQ tokens, the question, then the Apertus instruction."""
    return f"{image_token_str} Question: {question.strip()}\n\n{APERTUS_INSTRUCTION}"


def render_apertus_prompt(tokenizer, user_content: str, tool_schemas: list[dict]) -> str:
    """Render system + developer (tools) + user via the Apertus chat template."""
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": APERTUS_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        tools=tool_schemas,
        enable_thinking=True,
        add_generation_prompt=True,
        tokenize=False,
    )


def _extract_user_content(rendered_prompt: str) -> str:
    matches = USER_BLOCK.findall(rendered_prompt)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly 1 user block in rendered prompt, found {len(matches)}"
        )
    return matches[0]


def _build_parquet_record(meta: dict, split: str) -> dict:
    """Convert one metadata.jsonl row into the verl-RL parquet schema."""
    user_content = _extract_user_content(meta["prompt"])

    image_path = meta["image_path"]
    if not Path(image_path).is_absolute():
        raise ValueError(f"image_path is not absolute: {image_path!r}")

    extra = dict(meta["extra_info"])
    extra["split"] = split
    extra["need_tools_kwargs"] = True
    # draw_bbox_tool.create ignores kwargs today (no-op stub), but we pass the
    # original image path for forward compatibility with a UI/IoU backend.
    extra["tools_kwargs"] = {
        "draw_bbox_tool": {"create_kwargs": {"image_path": image_path}},
    }

    return {
        "data_source": meta["data_source"],
        "agent_name": meta["agent_name"],
        "prompt": [
            {"role": "system", "content": APERTUS_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        "ability": meta.get("ability", ""),
        "reward_model": meta["reward_model"],
        "extra_info": extra,
    }


def _split_and_write_parquet(
    metadata_path: Path, out_dir: Path, val_ratio: float, seed: int, max_val: int = 1000
) -> tuple[int, int]:
    """Read metadata.jsonl, deterministic shuffle+split, write train/val parquet."""
    meta_rows: list[dict] = []
    with open(metadata_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            meta_rows.append(json.loads(line))

    n = len(meta_rows)
    if n == 0:
        return 0, 0

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * val_ratio))) if n > 1 else 0
    n_val = min(n_val, max_val)  # cap val (0.05*~195k would be a wastefully large eval set)
    val_idx = set(perm[:n_val].tolist())

    train_records: list[dict] = []
    val_records: list[dict] = []
    for i, meta in enumerate(meta_rows):
        split = "val" if i in val_idx else "train"
        rec = _build_parquet_record(meta, split)
        (val_records if split == "val" else train_records).append(rec)

    out_dir.mkdir(parents=True, exist_ok=True)
    # Small row groups are REQUIRED at this scale. HuggingFace `datasets` reads a
    # parquet with batch_size = row_groups[0].num_rows; the prompts are huge (~8KB
    # each), so a single big row group makes the nested `prompt`/`extra_info`
    # arrays exceed pyarrow's per-chunk limit -> forced multi-chunk -> "Nested data
    # conversions not implemented for chunked array outputs" at load. Small groups
    # keep each read batch single-chunk. (Harmless for the small SFT/CoF parquets.)
    rg = 2048
    pq.write_table(pa.Table.from_pylist(train_records), out_dir / "train.parquet", row_group_size=rg)
    pq.write_table(pa.Table.from_pylist(val_records), out_dir / "val.parquet", row_group_size=rg)
    return len(train_records), len(val_records)


def shard_metadata_path(out_dir: Path, shard_index: int, num_shards: int) -> Path:
    return out_dir / "shards" / f"metadata.shard-{shard_index}-of-{num_shards}.jsonl"


def merge_shards(
    out_dir: Path, num_shards: int, val_ratio: float, seed: int, max_val: int
) -> tuple[int, int]:
    """Concatenate all shard metadata files -> metadata.jsonl -> train/val parquet.

    Hard-fails unless exactly `num_shards` finished shard files are present (a
    crashed shard leaves only a `.tmp`, so it cannot be mistaken for done).
    """
    shard_dir = out_dir / "shards"
    files = sorted(
        shard_dir.glob(f"metadata.shard-*-of-{num_shards}.jsonl"),
        key=lambda p: int(p.name.split("-")[1]),
    )
    if len(files) != num_shards:
        raise SystemExit(
            f"Expected {num_shards} shard files in {shard_dir}, found {len(files)}: "
            f"{[f.name for f in files]} (a shard crashed or is still .tmp)"
        )
    merged = out_dir / "metadata.jsonl"
    n_lines = 0
    with open(merged, "w", encoding="utf-8") as out:
        for fp in files:  # deterministic order: shard 0, 1, 2, ...
            with open(fp, encoding="utf-8") as fin:
                for line in fin:
                    out.write(line)
                    n_lines += 1
    print(f"[merge] concatenated {len(files)} shards -> {merged} ({n_lines} rows)")
    return _split_and_write_parquet(merged, out_dir, val_ratio, seed, max_val)


def _write_single_parquet(metadata_path: Path, out_dir: Path, name: str = "test.parquet") -> int:
    """Render all metadata rows into one parquet (no train/val split) — eval set."""
    meta_rows = []
    with open(metadata_path) as f:
        for line in f:
            line = line.strip()
            if line:
                meta_rows.append(json.loads(line))
    records = [_build_parquet_record(m, "test") for m in meta_rows]
    out_dir.mkdir(parents=True, exist_ok=True)
    # Small row groups: keep nested columns single-chunk for HF datasets (see the
    # note in _split_and_write_parquet).
    pq.write_table(pa.Table.from_pylist(records), out_dir / name, row_group_size=2048)
    return len(records)


def main():
    parser = argparse.ArgumentParser(description="Render Visual-CoT RL prompts in Apertus format")
    parser.add_argument("--input", default=None, help="Default: data_prep/vcot/raw.jsonl")
    parser.add_argument("--output", default=None, help="Default: data_prep/vcot_rl/metadata.jsonl")
    parser.add_argument("--images-root", default=None, help="Default: data_prep/vcot/images")
    parser.add_argument("--split-indices", default=None, help="Default: data_prep/vcot/split_indices.json")
    parser.add_argument("--config", default="configs/apertus.yaml")
    parser.add_argument("--tool_config", default="configs/vcot_rl_tool_config.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N RL rows (debug)")
    parser.add_argument("--n_sft", type=int, default=DEFAULT_N_SFT,
                        help="SFT holdout size (shared with vcot_sft_parse so the split agrees)")
    parser.add_argument("--n_test", type=int, default=DEFAULT_N_TEST,
                        help="TEST holdout size (shared with vcot_sft_parse so the split agrees)")
    parser.add_argument("--render-split", choices=["rl", "test"], default="rl",
                        help="rl: render rl_indices -> vcot_rl/{train,val}.parquet (shardable). "
                             "test: render the held-out test_indices -> vcot_test/test.parquet "
                             "(single file, in-distribution eval set).")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total shard processes (one per GPU). >1 => write a per-shard "
                             "metadata file and skip inline parquet (use --merge to finalize).")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="Which shard this process renders, in [0, num_shards).")
    parser.add_argument("--merge", action="store_true",
                        help="Merge mode: concatenate shard metadata -> metadata.jsonl -> "
                             "train/val parquet. No GPU/model load.")
    parser.add_argument("--max_val", type=int, default=1000,
                        help="Hard cap on validation rows.")
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Merge mode: no rows/models loaded; just stitch shards and write parquet.
    if args.merge:
        out_dir = PROJECT_ROOT / "data_prep" / "vcot_rl"
        n_train, n_val = merge_shards(
            out_dir, args.num_shards, args.val_ratio, args.seed, args.max_val
        )
        print(f"[merge] wrote {n_train} train / {n_val} val parquet rows")
        return

    render_test = args.render_split == "test"
    raw_dir = PROJECT_ROOT / "data_prep" / "vcot"
    out_dir = PROJECT_ROOT / "data_prep" / ("vcot_test" if render_test else "vcot_rl")
    input_path = Path(args.input) if args.input else raw_dir / "raw.jsonl"
    output_path = Path(args.output) if args.output else out_dir / "metadata.jsonl"
    images_root = Path(args.images_root) if args.images_root else raw_dir / "images"
    split_path = Path(args.split_indices) if args.split_indices else raw_dir / "split_indices.json"

    config = load_config(args.config)

    # Load the requested slice of the shared SFT/TEST/RL split. TEST = held-out
    # eval set (disjoint from SFT and RL); RL = all rows minus SFT minus TEST.
    all_rows = load_rows(input_path)
    split = get_or_create_split_indices(
        all_rows, split_path, seed=args.seed, n_sft=args.n_sft, n_test=args.n_test
    )
    sel_indices = set(split["test_indices" if render_test else "rl_indices"])
    rows = [row for row in all_rows if row["_raw_index"] in sel_indices]

    # Single-word answerability filter -> keep ALL such rows (no cap). Note the
    # ~83k leftover GQA rows here DO carry a `thought`/`full_answer` in raw.jsonl,
    # but RL deliberately reads ONLY question/answer/bboxs below: the reasoning
    # trace is never rendered into the prompt, so the model cannot read off the
    # answer ("no-cheat"). A post-render assertion in QA double-checks this.
    num_multi = 0
    kept = []
    for row in rows:
        ans = str(row.get("answer", "")).strip()
        if not ans or " " in ans:
            num_multi += 1
            continue
        kept.append(row)
    rows = kept
    if args.limit:
        rows = rows[: args.limit]

    # Strided shard slice (disjoint + covering); striding interleaves the big
    # contiguous gqa block so every shard gets a runtime-balanced dataset mix.
    # TEST is small -> always single-process.
    sharded = args.num_shards > 1 and not render_test
    if sharded:
        if not (0 <= args.shard_index < args.num_shards):
            raise SystemExit(
                f"--shard-index {args.shard_index} out of range for --num-shards {args.num_shards}"
            )
        rows = rows[args.shard_index :: args.num_shards]

    # Sharded runs write a per-shard metadata file (atomic rename on completion)
    # and skip inline parquet; `--merge` stitches them. Single-process is unchanged.
    if sharded:
        output_path = shard_metadata_path(out_dir, args.shard_index, args.num_shards)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_path = output_path.with_name(output_path.name + ".tmp")
    else:
        write_path = output_path

    print(f"Loaded {len(all_rows)} raw rows from {input_path}")
    print(f"Split file: {split_path} (SFT={len(split['sft_indices'])}, "
          f"TEST={len(split.get('test_indices', []))}, RL={len(split['rl_indices'])})")
    if sharded:
        print(f"Shard {args.shard_index}/{args.num_shards}: {len(rows)} rows (dropped {num_multi} multi-word total)")
    else:
        print(f"{'TEST' if render_test else 'RL'} rows after single-word filter: "
              f"{len(rows)} (dropped {num_multi} multi-word)")

    tool_schemas = load_tool_schemas(args.tool_config)
    print(f"Loaded tool schemas from {args.tool_config}: "
          f"{', '.join(schema['name'] for schema in tool_schemas)}")

    print(f"Loading Apertus tokenizer from {config['model']['checkpoint']} ...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["checkpoint"], trust_remote_code=True
    )

    print(f"Loading IBQ vision tokenizer from {config['model']['vq_model']} ...")
    from PIL import Image
    from inference.vision import encode_image, load_vq_model, smart_resize
    vq_model = load_vq_model(config["model"]["vq_model"], device="cuda:0")
    print("Models loaded")

    write_path.parent.mkdir(parents=True, exist_ok=True)
    skipped = 0
    data_source_counts: dict[str, int] = {}
    prompt_lens: list[int] = []
    start = time.time()

    with open(write_path, "w", encoding="utf-8") as out_f:
        for i, row in enumerate(rows):
            qid = row["_raw_index"]
            dataset = row.get("dataset") or ""

            try:
                src_path = resolve_image_path(row, images_root)
            except Exception as e:
                print(f"  SKIP row {i} (qid={qid}): {e}")
                skipped += 1
                continue

            try:
                image = Image.open(src_path).convert("RGB")
                # The model perceives the resized image, so build prompt tokens
                # from it and express the gold bbox in that same resized space
                # (matches SFT and what the reward's IoU compares against).
                resized = smart_resize(image)
                bbox = scale_bbox(row["bboxs"][0], image.size, resized.size)
                image_token_str = encode_image(resized, vq_model)
            except Exception as e:
                print(f"  SKIP row {i} (qid={qid}): IBQ encode failed: {e}")
                skipped += 1
                continue

            user_content = build_user_text(row["question"], image_token_str)
            prompt_str = render_apertus_prompt(tokenizer, user_content, tool_schemas)

            answer = str(row["answer"]).strip()
            record = {
                "question_id": qid,
                "prompt": prompt_str,
                "image_path": str(src_path),
                "reward_model": {"style": "rule", "ground_truth": answer},
                "data_source": "vcot_rl",
                "ability": dataset,
                "agent_name": "vcot_tool_agent",
                "extra_info": {
                    "index": str(qid),
                    "answer": answer,
                    "dataset": dataset,
                    "bbox": bbox,
                    "image_wh": [resized.size[0], resized.size[1]],
                },
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            data_source_counts[dataset] = data_source_counts.get(dataset, 0) + 1
            prompt_lens.append(len(prompt_str))

            if (i + 1) % 50 == 0 or i == len(rows) - 1:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                print(f"  [{i+1}/{len(rows)}] {rate:.2f} rows/s  skipped={skipped}")

    written = len(rows) - skipped

    if sharded:
        write_path.replace(output_path)  # atomic: mark this shard done
        print(f"\n[shard {args.shard_index}/{args.num_shards}] wrote {written} records "
              f"-> {output_path} (skipped {skipped})")
        return  # shards do NOT write parquet; run --merge to finalize

    print(f"\nWrote {written} records to {output_path}")
    if skipped:
        print(f"Skipped {skipped} rows")

    print("\nsubset (ability) histogram:")
    for k, v in sorted(data_source_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    if prompt_lens:
        prompt_lens.sort()
        n = len(prompt_lens)
        print(f"\nprompt char-length stats: min={prompt_lens[0]} "
              f"p50={prompt_lens[n // 2]} p95={prompt_lens[int(n * 0.95)]} "
              f"max={prompt_lens[-1]}")

    if render_test:
        n = _write_single_parquet(output_path, out_dir, "test.parquet")
        print(f"\nWrote {n} rows to {out_dir / 'test.parquet'} (in-distribution eval set)")
        return

    n_train, n_val = _split_and_write_parquet(
        output_path, output_path.parent, args.val_ratio, args.seed, args.max_val
    )
    print(f"\nWrote {n_train} rows to {output_path.parent / 'train.parquet'}")
    print(f"Wrote {n_val} rows to {output_path.parent / 'val.parquet'}")


if __name__ == "__main__":
    main()
