"""Download xintongzhang/CoF-sft-Data and extract images.

No GPU needed — runs on the login node. Downloads the train split, fetches
images.zip, extracts it, and writes a raw.jsonl that the parse step consumes.

Usage:
    python datasets/prepare_cof_sft_download.py
    python datasets/prepare_cof_sft_download.py --output-dir /tmp/cof_sft
"""

import argparse
import json
import zipfile
from pathlib import Path

REPO_ID = "xintongzhang/CoF-SFT-Data-5.4k"
SCRIPT_DIR = Path(__file__).resolve().parent


def download_dataset(output_dir: Path, split: str = "train") -> int:
    """Download the HF dataset split and write raw.jsonl. Returns row count."""
    from datasets import load_dataset

    print(f"Loading {REPO_ID} split={split} ...")
    ds = load_dataset(REPO_ID, split=split)
    print(f"Loaded {len(ds)} rows")

    raw_path = output_dir / "raw.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for row in ds:
            # Normalize image path: source uses "./images/foo.jpg"; strip the leading "./"
            images = row.get("images") or []
            image_paths = []
            for entry in images:
                p = entry["image"] if isinstance(entry, dict) else entry
                if p.startswith("./"):
                    p = p[2:]
                image_paths.append(p)

            f.write(json.dumps({
                "messages": row["messages"],
                "image_paths": image_paths,
            }, ensure_ascii=False) + "\n")

    print(f"Wrote {raw_path}")
    return len(ds)


def download_and_extract_images(output_dir: Path):
    """Download images.zip from HF and extract into output_dir/images/."""
    from huggingface_hub import hf_hub_download

    images_dir = output_dir / "images"
    if images_dir.exists() and any(images_dir.iterdir()):
        count = sum(1 for _ in images_dir.rglob("*") if _.is_file())
        print(f"images/ already populated ({count} files), skipping extraction")
        return

    print("Downloading images.zip ...")
    zip_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="images.zip",
        repo_type="dataset",
    )
    print(f"Got {zip_path}")

    images_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting into {images_dir} ...")
    with zipfile.ZipFile(zip_path) as zf:
        # The zip contains entries like "images/foo.jpg"; extract directly into output_dir
        # so paths line up with what we wrote in raw.jsonl.
        zf.extractall(output_dir)

    count = sum(1 for _ in images_dir.rglob("*") if _.is_file())
    print(f"Extracted {count} files")


def main():
    parser = argparse.ArgumentParser(description="Download CoF-SFT-Data for Apertus prep")
    parser.add_argument("--output-dir", default=None, help="Default: datasets/cof_sft")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "cof_sft"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(output_dir, args.split)
    n_rows = download_dataset(output_dir, split=args.split)
    download_and_extract_images(output_dir)

    print(f"\nDone. {n_rows} rows ready at {output_dir}")
    print(f"Next: sbatch slurm/prepare_cof_sft.slurm")


if __name__ == "__main__":
    main()
