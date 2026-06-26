"""Compress cof_rl metadata.jsonl for human inspection.

Each IBQ image-token block
    <|img_start|>H*W<|img_token_start|> <|visual token ...> ... <|img_end|>
is replaced with a compact  "HxW<IMAGE_TOKENS>"  filler, keeping everything else.

Usage:
    python utils/compress_metadata.py            # data_prep/cof_rl/metadata.jsonl -> metadata_vis.jsonl
    python utils/compress_metadata.py PATH        # any metadata.jsonl
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data_prep/cof_rl/metadata.jsonl"
DST = SRC.with_name("metadata_vis.jsonl")

IMG = re.compile(r"<\|img_start\|>(\d+)\*(\d+)<\|img_token_start\|>.*?<\|img_end\|>", re.DOTALL)


def strip_images(s: str) -> str:
    return IMG.sub(lambda m: f"{m.group(1)}*{m.group(2)}<IMAGE_TOKENS>", s)


n = 0
with open(SRC) as f, open(DST, "w") as o:
    for line in f:
        r = json.loads(line)
        for key in ("prompt", "text"):
            if isinstance(r.get(key), str):
                r[key] = strip_images(r[key])
        o.write(json.dumps(r, ensure_ascii=False) + "\n")
        n += 1
print(f"wrote {n} rows -> {DST}")
