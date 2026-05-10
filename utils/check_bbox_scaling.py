"""Verify bbox scaling in train.parquet against raw.jsonl.

For each sampled parquet row, opens the first resized image and the matching
original, then checks that every [x1,y1,x2,y2] tuple in parquet assistant
turns equals round(raw_bbox * scale).

Usage:
    python utils/check_bbox_scaling.py --n 10
    python utils/check_bbox_scaling.py --n all
"""

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
BBOX_RE = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]")
ASSIST_RE = re.compile(r"<\|assistant_start\|>(.*?)<\|assistant_end\|>", re.DOTALL)


def raw_bboxes(messages):
    out = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for m in BBOX_RE.finditer(msg["content"]):
            out.append(tuple(map(int, m.groups())))
    return out


def parquet_bboxes(text):
    return [
        tuple(map(int, m.groups()))
        for am in ASSIST_RE.finditer(text)
        for m in BBOX_RE.finditer(am.group(1))
    ]


def scale(b, osize, rsize):
    sx, sy = rsize[0] / osize[0], rsize[1] / osize[1]
    return (
        min(round(b[0] * sx), rsize[0]),
        min(round(b[1] * sy), rsize[1]),
        min(round(b[2] * sx), rsize[0]),
        min(round(b[3] * sy), rsize[1]),
    )


def check_row(row, raw_index):
    pq_img = str(row["image_paths"][0])
    p = Path(pq_img)
    raw_key = f"{p.parent.name}/{p.name}"
    orig = Path(*[("images_original" if s == "images" else s) for s in p.parts])
    raw = raw_index.get(raw_key)
    if raw is None or not orig.exists() or not p.exists():
        return f"  SKIP {raw_key}: raw={raw is not None} orig={orig.exists()} resized={p.exists()}\n"

    osize = Image.open(orig).size
    rsize = Image.open(p).size
    rb = raw_bboxes(raw["messages"])
    pb = parquet_bboxes(row["text"])
    eb = [scale(b, osize, rsize) for b in rb]

    from collections import Counter
    ec, pc = Counter(eb), Counter(pb)
    missing = list((ec - pc).elements())  # expected but not in parquet
    extra = list((pc - ec).elements())    # in parquet but not expected
    bad = len(missing) + len(extra)
    count_mm = len(rb) != len(pb)

    lines = [
        f"  {raw_key}  orig={osize} resized={rsize}  "
        f"scale=({rsize[0]/osize[0]:.4f},{rsize[1]/osize[1]:.4f})  "
        f"bboxes raw={len(rb)} pq={len(pb)}  unmatched={bad}"
    ]
    for r, e in zip(rb, eb):
        lines.append(f"    {list(r)} -> exp {list(e)}")
    for m in missing:
        lines.append(f"    MISSING (expected, not in pq): {list(m)}")
    for x in extra:
        lines.append(f"    EXTRA   (in pq, not expected): {list(x)}")
    if count_mm:
        lines.append(f"    COUNT MISMATCH (raw={len(rb)} pq={len(pb)})")
    return "\n".join(lines) + "\n", bad, count_mm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=str(ROOT / "data_prep/cof_sft/train.parquet"))
    ap.add_argument("--raw", default=str(ROOT / "data_prep/cof_sft/raw.jsonl"))
    ap.add_argument("--n", default="10", help="number of random rows, or 'all'")
    ap.add_argument("--output", default=str(ROOT / "utils/check_bbox_scaling.txt"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    if args.n.lower() == "all":
        idxs = list(range(len(df)))
    else:
        random.seed(args.seed)
        idxs = random.sample(range(len(df)), k=min(int(args.n), len(df)))

    raw_index = {}
    with open(args.raw) as f:
        for line in f:
            r = json.loads(line)
            if r.get("image_paths"):
                raw_index[r["image_paths"][0]] = r

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_bad_pairs = n_count_mm = 0
    with open(out, "w") as fo:
        fo.write(f"parquet: {args.parquet}\nraw: {args.raw}\nrows: {len(idxs)}/{len(df)}\n\n")
        for i, ri in enumerate(idxs, 1):
            res = check_row(df.iloc[ri], raw_index)
            if isinstance(res, str):
                fo.write(f"[{i}] {res}")
                continue
            text, bad, count_mm = res
            fo.write(f"[{i}] row {ri}\n{text}\n")
            n_bad_pairs += bad
            n_count_mm += count_mm
        summary = f"\nSUMMARY: {len(idxs)} rows, {n_bad_pairs} unmatched bboxes, {n_count_mm} count mismatches\n"
        fo.write(summary)
        print(summary.strip())
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
