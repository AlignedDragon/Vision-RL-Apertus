"""Download ChenShawn/DeepEyes-Datasets-47k and emit CoF-RL raw rows.

NOTE: DeepEyes is NOT part of training by default. It is opt-in only: running
this script merely produces raw_deepeyes.jsonl; training includes it solely when
prepare_cof_rl.slurm is run with INCLUDE_DEEPEYES=true.

DeepEyes ships verl-format parquet with images embedded as bytes. We extract
the ORIGINAL-resolution images to disk and map each row to the same raw schema
that ``cof_rl_parse.py`` consumes, so the existing parse/encode/filter pipeline
(single-token answer filter included) applies uniformly after merging.

Only the zoom rows (``env_name == "visual_toolbox_v2"``: V* + chart) are taken
by default, since those are the ones that exercise the image_zoom_in_tool. The
``data_thinklite_reasoning_acc`` split (math, no image tool) is opt-in via
``--include-thinklite``.

Uses huggingface_hub + pyarrow only (no `datasets`), so it runs on the login
node as well as inside the training container.

Usage:
    python data_prep/deepeyes_rl_download.py                 # zoom rows only
    python data_prep/deepeyes_rl_download.py --limit 100     # smoke subset
    python data_prep/deepeyes_rl_download.py --include-thinklite
"""

import argparse
import json
from pathlib import Path

REPO_ID = "ChenShawn/DeepEyes-Datasets-47k"
SCRIPT_DIR = Path(__file__).resolve().parent

# (parquet filename, short tag used in image paths / indices)
ZOOM_FILES = [
    ("data_0.1.2_visual_toolbox_v2.parquet", "vtb1"),
    ("data_v0.8_visual_toolbox_v2.parquet", "vtb2"),
]
THINKLITE_FILE = ("data_thinklite_reasoning_acc.parquet", "think")


def _image_ext(data: bytes) -> str:
    if data[:2] == b"\xff\xd8":
        return "jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return "png"


def process_file(
    filename: str,
    tag: str,
    out_dir: Path,
    raw_f,
    limit: int | None,
) -> tuple[int, int]:
    """Stream one parquet, extract images, append CoF-RL raw rows. Returns (written, skipped)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    print(f"[{tag}] downloading {filename} ...", flush=True)
    path = hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset")
    print(f"[{tag}] reading {path}", flush=True)

    img_root = out_dir / "images" / "deepeyes" / tag
    img_root.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(path)
    written = skipped = seen = 0
    for batch in pf.iter_batches(batch_size=256):
        cols = batch.to_pydict()
        n = len(cols["prompt"])
        for i in range(n):
            if limit is not None and seen >= limit:
                return written, skipped
            seen += 1

            reward_model = cols["reward_model"][i] or {}
            gt = reward_model.get("ground_truth")
            extra = cols["extra_info"][i] or {}
            question = extra.get("question")
            images = cols["images"][i] or []

            if not gt or not question or not images:
                skipped += 1
                continue

            img_bytes = images[0].get("bytes")
            if not img_bytes:
                skipped += 1
                continue

            src_index = str(extra.get("index", seen))
            uid = f"deepeyes-{tag}-{src_index}"
            ext = _image_ext(img_bytes)
            rel_path = f"images/deepeyes/{tag}/{src_index}.{ext}"
            (out_dir / rel_path).write_bytes(img_bytes)

            data_source = cols["data_source"][i] if "data_source" in cols else None
            ability = cols["ability"][i] if "ability" in cols else ""

            raw_f.write(json.dumps({
                # Clean user message: <image> + the raw question; cof_rl_parse
                # splices IBQ tokens for <image>, prepends the Apertus system +
                # tool schema, and appends the display_answers instruction.
                "prompt": [{"role": "user", "content": f"<image>\n{question}"}],
                "image_paths": [rel_path],
                "groundtruth_complete": None,
                # style=rule: we score with the CoF exact-match reward, not a judge.
                "reward_model": {"ground_truth": gt, "style": "rule"},
                "data_source": f"deepeyes:{data_source}" if data_source else "deepeyes",
                "agent_name": "cof_tool_agent",
                "ability": ability or "",
                "extra_info": {
                    "index": uid,
                    "question": question,
                    "answer": extra.get("answer", gt),
                    "split": extra.get("split", "train"),
                    "deepeyes_source_file": filename,
                },
            }, ensure_ascii=False) + "\n")
            written += 1

        if seen % 2048 == 0:
            print(f"[{tag}] seen={seen} written={written} skipped={skipped}", flush=True)

    return written, skipped


def main():
    parser = argparse.ArgumentParser(description="Download DeepEyes-47k as CoF-RL raw rows")
    parser.add_argument("--output-dir", default=None, help="Default: data_prep/cof_rl")
    parser.add_argument("--output", default=None, help="Default: <output-dir>/raw_deepeyes.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Max rows per parquet file (smoke)")
    parser.add_argument("--include-thinklite", action="store_true",
                        help="Also include the math reasoning split (no image tool)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "cof_rl"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = Path(args.output) if args.output else out_dir / "raw_deepeyes.jsonl"

    files = list(ZOOM_FILES)
    if args.include_thinklite:
        files.append(THINKLITE_FILE)

    total_w = total_s = 0
    with open(raw_path, "w", encoding="utf-8") as raw_f:
        for filename, tag in files:
            w, s = process_file(filename, tag, out_dir, raw_f, args.limit)
            print(f"[{tag}] done: written={w} skipped={s}", flush=True)
            total_w += w
            total_s += s

    print(f"\nWrote {total_w} rows ({total_s} skipped) to {raw_path}")
    print("Next: merge with cof_rl/raw.jsonl, then run data_prep/cof_rl_parse.py")


if __name__ == "__main__":
    main()
