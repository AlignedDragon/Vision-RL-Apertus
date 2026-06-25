"""Download xintongzhang/CoF-sft-Data and extract images.

No GPU needed — runs on the login node. Downloads the train split, fetches
images.zip, extracts it, and writes a raw.jsonl that the parse step consumes.

Usage:
    python data_prep/cof_sft_download.py
    python data_prep/cof_sft_download.py --output-dir /tmp/cof_sft
"""

import argparse
import json
import zipfile
from pathlib import Path

REPO_ID = "xintongzhang/CoF-SFT-Data-5.4k"
SCRIPT_DIR = Path(__file__).resolve().parent


def download_dataset(output_dir: Path, split: str = "train") -> int:
    """Download cof_sft_data.json from HF and write raw.jsonl. Returns row count."""
    from huggingface_hub import hf_hub_download

    if split != "train":
        raise ValueError(f"This dataset only ships a 'train' split, got {split!r}")

    print(f"Downloading cof_sft_data.json from {REPO_ID} ...")
    json_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="cof_sft_data.json",
        repo_type="dataset",
    )
    with open(json_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    print(f"Loaded {len(rows)} rows from {json_path}")

    raw_path = output_dir / "raw.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for row in rows:
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
    return len(rows)


def download_and_extract_images(output_dir: Path):
    """Download images.zip from HF and extract into output_dir/images_original/.

    The pristine downloads live under images_original/. The parse step then
    reads from images_original/ and writes resized/cropped artifacts into
    images/ — so this script can be re-run safely without losing the
    originals to in-place rewrites.
    """
    from huggingface_hub import hf_hub_download

    originals_dir = output_dir / "images_original"
    if originals_dir.exists() and any(originals_dir.iterdir()):
        count = sum(1 for _ in originals_dir.rglob("*") if _.is_file())
        print(f"images_original/ already populated ({count} files), skipping extraction")
        return

    print("Downloading images.zip ...")
    zip_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="images.zip",
        repo_type="dataset",
    )
    print(f"Got {zip_path}")

    originals_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting into {originals_dir} ...")
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            # Strip leading "images/" so files land directly in images_original/.
            name = member.filename
            if name.startswith("images/"):
                name = name[len("images/"):]
            elif name.startswith("./images/"):
                name = name[len("./images/"):]
            if not name:
                continue
            target = originals_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())

    count = sum(1 for _ in originals_dir.rglob("*") if _.is_file())
    print(f"Extracted {count} files")


def main():
    parser = argparse.ArgumentParser(description="Download CoF-SFT-Data for Apertus prep")
    parser.add_argument("--output-dir", default=None, help="Default: data_prep/cof_sft")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "cof_sft"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading '{args.split}' split into {output_dir}")
    n_rows = download_dataset(output_dir, split=args.split)
    download_and_extract_images(output_dir)

    print(f"\nDone. {n_rows} rows ready at {output_dir}")
    print(f"Next: sbatch slurm/prepare_cof_sft.slurm")


if __name__ == "__main__":
    main()
