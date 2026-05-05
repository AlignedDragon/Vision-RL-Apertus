"""Prepare VTC-Bench dataset for Apertus evaluation.

Downloads images from HuggingFace and creates metadata.jsonl in the
standard verl-apertus format.

Usage:
    python data_prep/prepare_vtc_bench.py
    python data_prep/prepare_vtc_bench.py --tsv-path /path/to/VTC-Bench.tsv
"""

import argparse
import csv
import json
import os
import shutil
from pathlib import Path


def parse_tsv(tsv_path: str) -> list[dict]:
    """Parse VTC-Bench TSV file into records."""
    records = []
    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            records.append(row)
    print(f"Parsed {len(records)} rows from {tsv_path}")
    return records


def build_metadata(records: list[dict]) -> list[dict]:
    """Convert TSV records to metadata.jsonl format."""
    metadata = []
    for row in records:
        question = row["question"]
        answer = row["answer"]
        options = {}
        is_mc = False

        # Check if multiple choice (A/B/C/D columns non-empty)
        for letter in ("A", "B", "C", "D"):
            val = row.get(letter, "").strip()
            if val:
                options[letter] = val
                is_mc = True

        # For MC: append options to question and instruct single-letter answer
        if is_mc:
            opts_text = "\n".join(f"{k}. {v}" for k, v in options.items())
            question = f"{question}\n\nOptions:\n{opts_text}\n\nAnswer with a single letter."

        metadata.append({
            "question_id": row["id"],
            "question": question,
            "answers": [answer],
            "image_file": row["image"],  # relative path like images/attention_focusing/...
            "category": row["category"],
            "is_mc": is_mc,
        })

    mc_count = sum(1 for m in metadata if m["is_mc"])
    oe_count = len(metadata) - mc_count
    print(f"Built metadata: {len(metadata)} total, {mc_count} MC, {oe_count} open-ended")
    return metadata


def download_images(output_dir: Path, tsv_records: list[dict]):
    """Download VTC-Bench images from HuggingFace.

    The HF dataset (zzzhu/VTC-Bench) has columns: image (PIL), label (int index).
    We match images to TSV rows by order (label 0 = TSV row 0).
    """
    images_dir = output_dir / "images"
    if images_dir.exists() and any(images_dir.rglob("*.jpg")):
        count = sum(1 for _ in images_dir.rglob("*") if _.is_file())
        print(f"Images directory already exists with {count} files, skipping download")
        return

    print("Downloading VTC-Bench dataset from HuggingFace...")
    from datasets import load_dataset

    ds = load_dataset("zzzhu/VTC-Bench", split="train")
    print(f"Downloaded {len(ds)} samples")

    if len(ds) != len(tsv_records):
        print(f"WARNING: HF dataset has {len(ds)} samples but TSV has {len(tsv_records)} rows")

    for i, sample in enumerate(ds):
        if i >= len(tsv_records):
            break
        # Get image path from TSV (e.g. "images/attention_focusing/attention_focusing_1.jpg")
        rel_path = tsv_records[i]["image"]
        dest = output_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Save PIL image (convert RGBA→RGB for JPEG compatibility)
        pil_img = sample["image"]
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        pil_img.save(dest)

        if (i + 1) % 100 == 0:
            print(f"  Saved {i+1}/{len(ds)} images")

    print(f"Images saved to {images_dir}")


def verify_images(metadata: list[dict], dataset_dir: Path) -> int:
    """Verify all images referenced in metadata exist."""
    missing = 0
    for record in metadata:
        img_path = dataset_dir / record["image_file"]
        if not img_path.exists():
            if missing < 5:
                print(f"  MISSING: {img_path}")
            missing += 1
    if missing:
        print(f"WARNING: {missing}/{len(metadata)} images missing")
    else:
        print(f"All {len(metadata)} images verified")
    return missing


def main():
    parser = argparse.ArgumentParser(description="Prepare VTC-Bench for Apertus evaluation")
    parser.add_argument(
        "--tsv-path",
        default="/capstor/scratch/cscs/msayfiddinov/VTC-Bench/data/VTC-Bench.tsv",
        help="Path to VTC-Bench.tsv",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: data_prep/vtc_bench)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "data_prep" / "vtc_bench"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Parse TSV
    records = parse_tsv(args.tsv_path)

    # Step 2: Build metadata
    metadata = build_metadata(records)

    # Step 3: Download images
    download_images(output_dir, records)

    # Step 4: Write metadata.jsonl
    metadata_path = output_dir / "metadata.jsonl"
    with open(metadata_path, "w", encoding="utf-8") as f:
        for record in metadata:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(metadata)} records to {metadata_path}")

    # Step 5: Verify images
    verify_images(metadata, output_dir)

    print("\nDone! Dataset ready at:", output_dir)


if __name__ == "__main__":
    main()
