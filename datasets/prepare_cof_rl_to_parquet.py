"""Convert CoF-RL metadata.jsonl into verl-ready parquet.

Current split:
Loaded 8163 metadata rows from /capstor/scratch/cscs/msayfiddinov/verl-apertus/datasets/cof_rl/metadata.jsonl
Wrote 7755 rows to /capstor/scratch/cscs/msayfiddinov/verl-apertus/datasets/cof_rl/train.parquet
Wrote 408 rows to /capstor/scratch/cscs/msayfiddinov/verl-apertus/datasets/cof_rl/val.parquet

Reads:
  datasets/cof_rl/metadata.jsonl  -- Apertus-rendered prompts produced by
                                     prepare_cof_rl_parse.py. Contains
                                     everything we need: rendered prompt
                                     (with IBQ tokens + Apertus instruction
                                     already spliced in), image_path,
                                     reward_model, ability, extra_info.

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
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent

APERTUS_SYSTEM = "You are a helpful assistant with access to tools."

USER_BLOCK = re.compile(r"<\|user_start\|>(.*?)<\|user_end\|>", re.DOTALL)


def _extract_user_content(rendered_prompt: str) -> str:
    matches = USER_BLOCK.findall(rendered_prompt)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly 1 user block in rendered prompt, found {len(matches)}"
        )
    return matches[0]


def _build_record(meta: dict, split: str) -> dict:
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
    parser.add_argument("--metadata", default=None, help="Default: datasets/cof_rl/metadata.jsonl")
    parser.add_argument("--out_dir", default=None, help="Default: datasets/cof_rl")
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Process only first N rows (debug)")
    args = parser.parse_args()

    dataset_dir = PROJECT_ROOT / "datasets" / "cof_rl"
    meta_path = Path(args.metadata) if args.metadata else dataset_dir / "metadata.jsonl"
    out_dir = Path(args.out_dir) if args.out_dir else dataset_dir

    meta_rows = _read_jsonl(meta_path)

    if args.limit:
        meta_rows = meta_rows[: args.limit]

    print(f"Loaded {len(meta_rows)} metadata rows from {meta_path}")

    # Deterministic shuffle then split.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(meta_rows))
    n_val = max(1, int(round(len(meta_rows) * args.val_ratio))) if len(meta_rows) > 1 else 0
    val_idx = set(perm[:n_val].tolist())

    train_records: list[dict] = []
    val_records: list[dict] = []
    for i, meta in enumerate(meta_rows):
        split = "val" if i in val_idx else "train"
        rec = _build_record(meta, split)
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
