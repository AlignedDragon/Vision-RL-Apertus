"""Print 3 random SFT training examples with trainable spans highlighted.

Trainable text (what the model computes loss on) is shown in GREEN;
masked text (prompt / tool outputs) is shown in DIM.
"""

import argparse
import random

import pandas as pd

from data_prep.cof_sft_dataset import compute_train_spans


GREEN = "\033[92m"
DIM = "\033[2m"
RESET = "\033[0m"


def render(text: str) -> str:
    spans = compute_train_spans(text)
    out = []
    cur = 0
    for lo, hi in spans:
        if cur < lo:
            out.append(DIM + text[cur:lo] + RESET)
        out.append(GREEN + text[lo:hi] + RESET)
        cur = hi
    if cur < len(text):
        out.append(DIM + text[cur:] + RESET)
    return "".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", default="data_prep/cof_sft/train.parquet")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--seed", type=int, default=9)
    args = p.parse_args()

    random.seed(args.seed)

    df = pd.read_parquet(args.parquet)
    idxs = random.sample(range(len(df)), k=min(args.n, len(df)))

    for i, idx in enumerate(idxs):
        row = df.iloc[idx]
        text = row["text"]
        imgs = row["image_paths"] if "image_paths" in row else []
        print(f"\n{'='*80}\nEXAMPLE {i+1}/{len(idxs)}  (row {idx})  len={len(text)} chars  images={list(imgs)}\n{'='*80}")
        print(render(text))


if __name__ == "__main__":
    main()
