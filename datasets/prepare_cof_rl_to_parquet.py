"""Convert CoF-RL JSONL outputs into verl-ready parquet.

Reads:
  datasets/cof_rl/raw.jsonl       -- original Qwen-style messages + image_paths
  datasets/cof_rl/metadata.jsonl  -- Apertus-rendered prompts (only used to
                                    extract the IBQ token string per row, so
                                    we don't re-run GPU-bound IBQ encoding)

Writes:
  datasets/cof_rl/train.parquet
  datasets/cof_rl/val.parquet

Each row carries:
  - prompt:        list[{role, content}] -- system + user (with IBQ tokens
                   inline). The developer/tools block is rendered by verl
                   at rollout time from tool_config.yaml.
  - agent_name:    "cof_tool_agent"      -- routes to our custom loop.
  - data_source:   "cof_rl"              -- routes to our custom reward fn.
  - reward_model:  {style, ground_truth}
  - extra_info:
      need_tools_kwargs: True
      tools_kwargs.image_zoom_in_tool.create_kwargs.image_path: <abs path>
      index, split, answer

Usage:
    python datasets/prepare_cof_rl_to_parquet.py [--limit N] [--val_ratio 0.05]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.prepare_cof_rl_parse import (  # noqa: E402
    APERTUS_SYSTEM,
    build_user_message,
    get_user_text,
)

IBQ_BLOCK = re.compile(r"<\|img_start\|>.*?<\|img_end\|>", re.DOTALL)


def _extract_ibq_token_str(rendered_prompt: str) -> str:
    matches = IBQ_BLOCK.findall(rendered_prompt)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly 1 IBQ block in rendered prompt, found {len(matches)}"
        )
    return matches[0]


def _row_index(row: dict, fallback: int) -> int:
    extra = row.get("extra_info") or {}
    if "index" in extra:
        return int(extra["index"])
    return fallback


def _build_record(raw: dict, meta: dict, split: str) -> dict:
    user_text = get_user_text(raw["prompt"])
    image_token_str = _extract_ibq_token_str(meta["prompt"])
    user_msg = build_user_message(user_text, image_token_str)

    image_path = meta["image_path"]
    if not Path(image_path).is_absolute():
        raise ValueError(f"image_path is not absolute: {image_path!r}")

    qid = meta.get("question_id", _row_index(raw, 0))
    answer = meta["reward_model"]["ground_truth"]

    return {
        "data_source": "cof_rl",
        "agent_name": "cof_tool_agent",
        "prompt": [
            {"role": "system", "content": APERTUS_SYSTEM},
            user_msg,
        ],
        "ability": meta.get("ability", raw.get("ability", "")),
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


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=None, help="Default: datasets/cof_rl/raw.jsonl")
    parser.add_argument("--metadata", default=None, help="Default: datasets/cof_rl/metadata.jsonl")
    parser.add_argument("--out_dir", default=None, help="Default: datasets/cof_rl")
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Process only first N rows (debug)")
    args = parser.parse_args()

    dataset_dir = PROJECT_ROOT / "datasets" / "cof_rl"
    raw_path = Path(args.raw) if args.raw else dataset_dir / "raw.jsonl"
    meta_path = Path(args.metadata) if args.metadata else dataset_dir / "metadata.jsonl"
    out_dir = Path(args.out_dir) if args.out_dir else dataset_dir

    raw_rows = _read_jsonl(raw_path)
    meta_rows = _read_jsonl(meta_path)

    # Align by index. metadata.jsonl is a filtered subset of raw.jsonl
    # (skipped rows for missing images); index in meta points back to raw.
    raw_by_idx = {_row_index(r, i): r for i, r in enumerate(raw_rows)}

    aligned: list[tuple[dict, dict]] = []
    for m in meta_rows:
        qid = m.get("question_id")
        if qid is None or qid not in raw_by_idx:
            raise RuntimeError(f"metadata row question_id={qid!r} has no match in raw.jsonl")
        aligned.append((raw_by_idx[qid], m))

    if args.limit:
        aligned = aligned[: args.limit]

    print(f"Aligned {len(aligned)} (raw, metadata) pairs from {raw_path} + {meta_path}")

    # Deterministic shuffle then split.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(aligned))
    n_val = max(1, int(round(len(aligned) * args.val_ratio))) if len(aligned) > 1 else 0
    val_idx = set(perm[:n_val].tolist())

    train_records: list[dict] = []
    val_records: list[dict] = []
    for i, (raw, meta) in enumerate(aligned):
        split = "val" if i in val_idx else "train"
        rec = _build_record(raw, meta, split)
        # Validation: exactly one IBQ block survived in the user message.
        user_content = rec["prompt"][1]["content"]
        n_blocks = len(IBQ_BLOCK.findall(user_content))
        if n_blocks != 1:
            raise RuntimeError(f"row {i}: expected 1 IBQ block in user content, got {n_blocks}")
        ip = rec["extra_info"]["tools_kwargs"]["image_zoom_in_tool"]["create_kwargs"]["image_path"]
        if not Path(ip).exists():
            print(f"  WARN row {i}: image_path does not exist on this host: {ip}")
        (val_records if split == "val" else train_records).append(rec)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_out = out_dir / "train.parquet"
    val_out = out_dir / "val.parquet"

    pq.write_table(pa.Table.from_pylist(train_records), train_out)
    pq.write_table(pa.Table.from_pylist(val_records), val_out)

    print(f"Wrote {len(train_records)} rows to {train_out}")
    print(f"Wrote {len(val_records)} rows to {val_out}")


if __name__ == "__main__":
    main()
