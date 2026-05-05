"""Parse CoF-SFT-Data raw.jsonl into Apertus-formatted SFT examples.

Per row, render a full Apertus conversation:
  - system:    "You are a helpful assistant."
  - developer: source tool (image_zoom_in_tool) + display_answers, rendered via tools=
  - user:      original question with first <image> replaced by IBQ tokens for image_paths[0],
               Qwen "Think in the mind first..." trailer stripped, Apertus instruction appended
  - assistant: thoughts + tool_calls (extracted from <think>...</think> + <tool_call>...</tool_call>)
  - tool:      IBQ tokens for image_paths[1] only (no surrounding text)
  - assistant: thoughts + display_answers tool call (final <answer>...</answer> wrapped)

Output records:
    {
      "text": str,                 # full rendered Apertus conversation
      "image_paths": [str, str]    # absolute paths: [user image, tool-response image]
    }

Usage (interactive on a GPU node):
    python data_prep/prepare_cof_sft_parse.py --limit 5

Usage (SLURM): clone slurm/prepare_cof_rl.slurm and swap the script + dataset dir.
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

# Re-export the constants/helpers we share with the RL parse. Heavy deps
# (torch, PIL, transformers, vision_tokenizer) are imported inside main()
# so the pure-python helpers below can be unit-tested without them.
from data_prep.prepare_cof_rl_parse import (
    APERTUS_SYSTEM,
    QWEN_TRAILER_SENTINEL,
    DISPLAY_ANSWERS_TOOL,
    extract_tool_def,
    load_config,
)

# SFT keeps multi-word ground-truths, so the instruction must accept any answer
# string (RL drops multi-word answers and instructs "single word"). The single
# element of the `answers` array carries the original answer verbatim.
APERTUS_INSTRUCTION = (
    "Call the display_answers tool exactly once at the end of your response, "
    "passing your final answer as the single element of the `answers` argument."
)

THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


def build_user_text(raw_user_text: str, image_token_str: str) -> str:
    head = raw_user_text.split(QWEN_TRAILER_SENTINEL, 1)[0].rstrip()
    head = head.replace("<image>", image_token_str, 1)
    return f"{head}\n\n{APERTUS_INSTRUCTION}"


def parse_intermediate_assistant(text: str) -> tuple[str, dict]:
    """Extract <think> + <tool_call> from an intermediate assistant turn."""
    think_m = THINK_RE.search(text)
    call_m = TOOL_CALL_RE.search(text)
    if not think_m or not call_m:
        raise ValueError("intermediate assistant missing <think> or <tool_call>")
    call_obj = json.loads(call_m.group(1).strip())
    args_str = json.dumps(call_obj["arguments"], ensure_ascii=False, separators=(", ", ": "))
    return think_m.group(1).strip(), {"name": call_obj["name"], "arguments": args_str}


def parse_final_assistant(text: str) -> tuple[str, dict]:
    """Extract <think> + <answer>; convert answer into a display_answers tool call."""
    think_m = THINK_RE.search(text)
    ans_m = ANSWER_RE.search(text)
    if not think_m or not ans_m:
        raise ValueError("final assistant missing <think> or <answer>")
    args_str = json.dumps({"answers": [ans_m.group(1).strip()]}, ensure_ascii=False)
    return think_m.group(1).strip(), {"name": "display_answers", "arguments": args_str}


def build_messages(src_msgs: list, image1_tokens: str, image2_tokens: str) -> list:
    """Convert a 5-message Qwen conversation into Apertus messages.

    Source order: system, user, assistant, user(=tool response), assistant.
    """
    raw_user = next(m["content"] for m in src_msgs if m["role"] == "user")
    asst_msgs = [m for m in src_msgs if m["role"] == "assistant"]
    if len(asst_msgs) != 2:
        raise ValueError(f"expected 2 assistant turns, got {len(asst_msgs)}")
    th1, call1 = parse_intermediate_assistant(asst_msgs[0]["content"])
    th2, call2 = parse_final_assistant(asst_msgs[1]["content"])
    return [
        {"role": "system", "content": APERTUS_SYSTEM},
        {"role": "user", "content": build_user_text(raw_user, image1_tokens)},
        {"role": "assistant", "content": {"blocks": [
            {"type": "thoughts", "text": th1},
            {"type": "tool_calls", "calls": [call1]},
        ]}},
        {"role": "tool", "content": image2_tokens},
        {"role": "assistant", "content": {"blocks": [
            {"type": "thoughts", "text": th2},
            {"type": "tool_calls", "calls": [call2]},
        ]}},
    ]


def main():
    parser = argparse.ArgumentParser(description="Render CoF-SFT trajectories in Apertus format")
    parser.add_argument("--input", default=None, help="Default: data_prep/cof_sft/raw.jsonl")
    parser.add_argument("--output", default=None, help="Default: data_prep/cof_sft/metadata.jsonl")
    parser.add_argument("--config", default="configs/apertus.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N rows (debug)")
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_dir = PROJECT_ROOT / "data_prep" / "cof_sft"
    input_path = Path(args.input) if args.input else dataset_dir / "raw.jsonl"
    output_path = Path(args.output) if args.output else dataset_dir / "metadata.jsonl"

    config = load_config(args.config)

    rows = []
    with open(input_path) as f:
        for line in f:
            rows.append(json.loads(line))
        if args.limit:
            rows = rows[: args.limit]
    print(f"Loaded {len(rows)} rows from {input_path}")

    src_system = next(m["content"] for m in rows[0]["messages"] if m["role"] == "system")
    tool_def = extract_tool_def(src_system)
    print(f"Extracted tool: {tool_def['name']}")
    for i in range(1, min(50, len(rows))):
        other_system = next(m["content"] for m in rows[i]["messages"] if m["role"] == "system")
        if extract_tool_def(other_system) != tool_def:
            raise RuntimeError(f"Tool definition drift at row {i}")
    print(f"Tool definition consistent across first {min(50, len(rows))} rows")

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
            paths = row.get("image_paths") or []
            if len(paths) > 2:
                print(f"  SKIP row {i}: expected 2 image_paths, got {len(paths)}")
                skipped += 1
                continue

            img1_path = dataset_dir/"images"/paths[0]
            img2_path = dataset_dir/"images"/paths[1]
            if not img1_path.exists() or not img2_path.exists():
                print(f"  SKIP row {i}: missing image(s) {img1_path} / {img2_path}")
                skipped += 1
                continue

            try:
                img1_tokens = encode_image(Image.open(img1_path), vq_model)
                img2_tokens = encode_image(Image.open(img2_path), vq_model)
            except Exception as e:
                print(f"  SKIP row {i}: IBQ encode failed: {e}")
                skipped += 1
                continue

            try:
                messages = build_messages(row["messages"], img1_tokens, img2_tokens)
            except Exception as e:
                print(f"  SKIP row {i}: message build failed: {e}")
                skipped += 1
                continue

            text = tokenizer.apply_chat_template(
                messages,
                tools=[tool_def, DISPLAY_ANSWERS_TOOL],
                enable_thinking=True,
                add_generation_prompt=False,
                tokenize=False,
            )

            out_f.write(json.dumps({
                "text": text,
                "image_paths": [str(img1_path), str(img2_path)],
            }, ensure_ascii=False) + "\n")
            text_lens.append(len(text))

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

    # Deterministic shuffle + split → train/val parquet (consumed by
    # CoFSFTDataset; metadata.jsonl above stays for human inspection).
    meta_rows: list[dict] = []
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
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
