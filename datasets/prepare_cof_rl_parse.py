"""Parse CoF-RL-Data raw.jsonl into Apertus-formatted metadata.jsonl.

Per row:
  1. IBQ-encode the image into Apertus visual tokens.
  2. Strip the Qwen formatting trailer from the user content.
  3. Replace the literal `<image>` placeholder with the IBQ token string.
  4. Append a short Apertus-style instruction trailer.
  5. Render the full prompt via tokenizer.apply_chat_template with the
     extracted tool definition and enable_thinking=True.

Output records:
    {
      "question_id": int,
      "prompt": str,                                 # fully rendered Apertus prompt
      "reward_model": {"ground_truth": str, "style": str},
      "data_source": str,
      "ability": str,
      "agent_name": str,
      "extra_info": {...passthrough...}
    }

Usage (interactive on a GPU node):
    python datasets/prepare_cof_rl_parse.py --limit 5

Usage (SLURM):
    sbatch slurm/prepare_cof_rl.slurm
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.vision import encode_image, load_vq_model

QWEN_TRAILER_SENTINEL = "Think in the mind first"
APERTUS_INSTRUCTION = (
    "Think step by step and then decide whether to call tools one or more time OR provide finals answer. <|inner_prefix|>...<|inner_suffix|>. "
    "Format strictyly as: <|inner_prefix|>...<|inner_suffix|> <|tools_prefix|>[{...}]<|tools_suffix|> (if any tools needed) OR <|answer_prefix|>...<|answer_suffix|> (if no tools needed)"
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def extract_tool_def(system_text: str) -> dict:
    """Parse the <tools>{...}</tools> JSON in the source system prompt.

    Source format (Qwen-style):
        <tools>
        {"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}
        </tools>

    Apertus's chat_template.jinja accesses tool.name / tool.description / tool.parameters
    directly, so we unwrap and return the inner `function` dict.
    """
    match = re.search(r"<tools>\s*(.*?)\s*</tools>", system_text, re.DOTALL)
    if not match:
        raise ValueError("No <tools>...</tools> block found in system content")
    obj = json.loads(match.group(1))
    if obj.get("type") != "function" or "function" not in obj:
        raise ValueError(f"Unexpected tool envelope: {obj.keys()}")
    return obj["function"]

# ! here
def clean_user_content(text: str, image_token_str: str) -> str:
    """Strip the Qwen trailer, splice in the IBQ image tokens, append Apertus trailer."""
    head = text.split(QWEN_TRAILER_SENTINEL, 1)[0].rstrip()
    head = head.replace("<image>", image_token_str, 1)
    return f"{head}\n\n{APERTUS_INSTRUCTION}"


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
    parser.add_argument("--input", default=None, help="Default: datasets/cof_rl/raw.jsonl")
    parser.add_argument("--output", default=None, help="Default: datasets/cof_rl/metadata.jsonl")
    parser.add_argument("--config", default="configs/apertus.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N rows (debug)")
    args = parser.parse_args()

    dataset_dir = PROJECT_ROOT / "datasets" / "cof_rl"
    input_path = Path(args.input) if args.input else dataset_dir / "raw.jsonl"
    output_path = Path(args.output) if args.output else dataset_dir / "metadata.jsonl"

    config = load_config(args.config)

    # Load source rows
    rows = []
    with open(input_path) as f:
        for line in f:
            rows.append(json.loads(line))
        if args.limit:
            rows = rows[: args.limit]
    print(f"Loaded {len(rows)} rows from {input_path}")

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
            cleaned_user = clean_user_content(user_text, image_token_str)
            messages = [{"role": "user", "content": cleaned_user}]

            prompt_str = tokenizer.apply_chat_template(
                messages,
                tools=[tool_def],
                enable_thinking=True,
                add_generation_prompt=True,
                tokenize=False,
            )

            record = {
                "question_id": qid,
                "prompt": prompt_str,
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


if __name__ == "__main__":
    main()
