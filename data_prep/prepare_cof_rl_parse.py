"""Parse CoF-RL-Data raw.jsonl into Apertus-formatted metadata.jsonl.

Per row, render an Apertus 3-block prompt:
  - system:    "You are a helpful assistant."
  - developer: tool capabilities (the source <tools> tool plus the
               `display_answers` tool used to emit the final answer),
               rendered as TypeScript types via the chat template's tools= parameter
  - user:      original question with <image> replaced by IBQ vision tokens,
               the Qwen "Think in the mind first..." trailer stripped, and an
               instruction to emit the final answer via the `display_answers` tool.

Output records:
    {
      "question_id": int,
      "prompt": str,                                 # fully rendered Apertus prompt
      "image_path": str,
      "reward_model": {"ground_truth": str, "style": str},
      "data_source": str,
      "ability": str,
      "agent_name": str,
      "extra_info": {...passthrough...}
    }

Usage (interactive on a GPU node):
    python data_prep/prepare_cof_rl_parse.py --limit 5

Usage (SLURM):
    sbatch slurm/prepare_cof_rl.slurm
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
import torch
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.vision import encode_image, load_vq_model

QWEN_TRAILER_SENTINEL = "Think in the mind first"
APERTUS_INSTRUCTION = (
    "Call the display_answers tool exactly once at the end of your response, "
    "passing your final answer as a single word in the `answers` argument."
)
APERTUS_SYSTEM = "You are a helpful assistant with access to tools."

DISPLAY_ANSWERS_TOOL = {
    "name": "display_answers",
    "description": "Display the answers to the user.",
    "parameters": {
        "type": "object",
        "properties": {
            "answers": {
                "type": "array",
                "items": {"type":"string"},
                "description": "The final answer.",
            },
        },
        "required": ["answers"],
    },
}


USER_BLOCK = re.compile(r"<\|user_start\|>(.*?)<\|user_end\|>", re.DOTALL)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


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

    qid = meta["question_id"]
    answer = meta["reward_model"]["ground_truth"]

    return {
        "data_source": "cof_rl",
        "agent_name": "cof_tool_agent",
        "prompt": [
            {"role": "system", "content": APERTUS_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        "ability": meta.get("ability", ""),
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "index": qid,
            "split": split,
            "answer": answer,
            "need_tools_kwargs": True,
            "tools_kwargs": {
                "image_zoom_in_tool": {
                    "create_kwargs": {"image_path": image_path},
                },
            },
        },
    }


def _split_and_write_parquet(
    metadata_path: Path, out_dir: Path, val_ratio: float, seed: int
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
    val_idx = set(perm[:n_val].tolist())

    train_records: list[dict] = []
    val_records: list[dict] = []
    for i, meta in enumerate(meta_rows):
        split = "val" if i in val_idx else "train"
        rec = _build_parquet_record(meta, split)
        (val_records if split == "val" else train_records).append(rec)

    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(train_records), out_dir / "train.parquet")
    pq.write_table(pa.Table.from_pylist(val_records), out_dir / "val.parquet")
    return len(train_records), len(val_records)


def extract_tool_def(system_text: str) -> dict:
    """Parse the <tools>{...}</tools> JSON in the source system prompt.

    Source format (Qwen-style):
        <tools>
        {"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}
        </tools>

    Apertus's chat_template.jinja accesses tool.name / tool.description / tool.parameters
    directly, so we unwrap and return the inner `function` dict.
    """
    content = next(
        (m.strip() for m in re.findall(r"<tools>(.*?)</tools>", system_text, re.DOTALL) if m.strip()),
        None,
    )
    if content is None:
        raise ValueError("No non-empty <tools>...</tools> block found in system content")
    obj = json.loads(content)
    if obj.get("type") != "function" or "function" not in obj:
        raise ValueError(f"Unexpected tool envelope: {obj.keys()}")
    return obj["function"]

def build_system_message() -> dict:
    """Apertus system block: a single short instruction (constant)."""
    return {"role": "system", "content": APERTUS_SYSTEM}


def build_user_message(raw_user_text: str, image_token_str: str) -> dict:
    """Apertus user block: strip the Qwen trailer, splice in IBQ tokens, append Apertus instruction."""
    head = raw_user_text.split(QWEN_TRAILER_SENTINEL, 1)[0].rstrip()
    head = head.replace("<image>", image_token_str, 1)
    return {"role": "user", "content": f"{head}\n\n{APERTUS_INSTRUCTION}"}


def render_apertus_prompt(tokenizer, system_msg: dict, user_msg: dict, tool_def: dict) -> str:
    """Render system + developer (tools) + user via the Apertus chat template."""
    return tokenizer.apply_chat_template(
        [system_msg, user_msg],
        tools=[tool_def, DISPLAY_ANSWERS_TOOL],
        enable_thinking=True,
        add_generation_prompt=True,
        tokenize=False,
    )


def get_system_text(prompt_messages: list) -> str:
    """Find the system message in the source prompt list."""
    for m in prompt_messages:
        if m.get("role") == "system":
            return m["content"]
    raise ValueError("No system message in source prompt")


def get_user_text(prompt_messages: list) -> str:
    for m in prompt_messages:
        if m.get("role") == "user":
            return m["content"]
    raise ValueError("No user message in source prompt")


def main():
    parser = argparse.ArgumentParser(description="Render CoF-RL prompts in Apertus format")
    parser.add_argument("--input", default=None, help="Default: data_prep/cof_rl/raw.jsonl")
    parser.add_argument("--output", default=None, help="Default: data_prep/cof_rl/metadata.jsonl")
    parser.add_argument("--config", default="configs/apertus.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N rows (debug)")
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_dir = PROJECT_ROOT / "data_prep" / "cof_rl"
    input_path = Path(args.input) if args.input else dataset_dir / "raw.jsonl"
    output_path = Path(args.output) if args.output else dataset_dir / "metadata.jsonl"

    config = load_config(args.config)

    # Load source rows
    rows = []
    num_multi = 0 # number of questions with multi word responses
    with open(input_path) as f:
        for line in f:
            line = json.loads(line)
            if " " in line["reward_model"]["ground_truth"]:
                num_multi += 1
                continue
            rows.append(line)
            
        if args.limit:
            rows = rows[: args.limit]

    print(f"Loaded {len(rows)} rows from {input_path}")
    print(f"Dropped {num_multi} questions for including multiple words")

    # Extract and validate tool definition (identical across rows)
    tool_def = extract_tool_def(get_system_text(rows[0]["prompt"]))
    print(f"Extracted tool: {tool_def['name']}")
    for i in range(1, min(50, len(rows))):
        other = extract_tool_def(get_system_text(rows[i]["prompt"]))
        if other != tool_def:
            raise RuntimeError(f"Tool definition drift at row {i}")
    print(f"Tool definition consistent across first {min(50, len(rows))} rows")

    # Load tokenizer + IBQ vision model
    print(f"Loading Apertus tokenizer from {config['model']['checkpoint']} ...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["checkpoint"], trust_remote_code=True
    )

    print(f"Loading IBQ vision tokenizer from {config['model']['vq_model']} ...")
    vq_model = load_vq_model(config["model"]["vq_model"], device="cuda:0")
    print("Models loaded")

    # Process rows
    output_path.parent.mkdir(parents=True, exist_ok=True)
    skipped = 0
    data_source_counts: dict[str, int] = {}
    prompt_lens: list[int] = []
    start = time.time()

    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, row in enumerate(rows):
            qid = row["extra_info"].get("index", i)

            if not row["image_paths"]:
                print(f"  SKIP row {i} (qid={qid}): no image path")
                skipped += 1
                continue

            image_path = dataset_dir / row["image_paths"][0]
            if not image_path.exists():
                print(f"  SKIP row {i} (qid={qid}): missing image {image_path}")
                skipped += 1
                continue

            try:
                image = Image.open(image_path)
                image_token_str = encode_image(image, vq_model)
            except Exception as e:
                print(f"  SKIP row {i} (qid={qid}): IBQ encode failed: {e}")
                skipped += 1
                continue

            user_text = get_user_text(row["prompt"])
            system_msg = build_system_message()
            user_msg = build_user_message(user_text, image_token_str)
            prompt_str = render_apertus_prompt(tokenizer, system_msg, user_msg, tool_def)

            record = {
                "question_id": qid,
                "prompt": prompt_str,
                "image_path": str(image_path),
                "reward_model": row["reward_model"],
                "data_source": row["data_source"],
                "ability": row["ability"],
                "agent_name": row["agent_name"],
                "extra_info": row["extra_info"],
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            data_source_counts[row["data_source"]] = data_source_counts.get(row["data_source"], 0) + 1
            prompt_lens.append(len(prompt_str))

            if (i + 1) % 50 == 0 or i == len(rows) - 1:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                print(f"  [{i+1}/{len(rows)}] {rate:.2f} rows/s  skipped={skipped}")

    written = len(rows) - skipped
    print(f"\nWrote {written} records to {output_path}")
    if skipped:
        print(f"Skipped {skipped} rows")

    print("\ndata_source histogram:")
    for k, v in sorted(data_source_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    if prompt_lens:
        prompt_lens.sort()
        n = len(prompt_lens)
        print(f"\nprompt char-length stats: min={prompt_lens[0]} "
              f"p50={prompt_lens[n // 2]} p95={prompt_lens[int(n * 0.95)]} "
              f"max={prompt_lens[-1]}")

    n_train, n_val = _split_and_write_parquet(
        output_path, output_path.parent, args.val_ratio, args.seed
    )
    print(f"\nWrote {n_train} rows to {output_path.parent / 'train.parquet'}")
    print(f"Wrote {n_val} rows to {output_path.parent / 'val.parquet'}")


if __name__ == "__main__":
    main()
