"""Download and prepare a 200-sample TextVQA validation subset.

Downloads annotations from Facebook's CDN and individual images from Flickr.
No GPU or ML libraries needed — runs on the login node.

Usage:
    conda activate verl
    python datasets/prepare_textvqa.py
"""

import io
import json
import os
import random
import sys
import urllib.request
from pathlib import Path

from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUM_SAMPLES = 200
OVERSAMPLE = 300  # download extra to handle broken Flickr URLs
SEED = 42
ANNOTATIONS_URL = (
    "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json"
)

# Output directory: datasets/textvqa/ in the repo root
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "textvqa"
IMAGES_DIR = OUTPUT_DIR / "images"
METADATA_FILE = OUTPUT_DIR / "metadata.jsonl"


def download_json(url: str) -> dict:
    """Download and parse a JSON file from a URL."""
    print(f"Downloading annotations from {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def download_image(url: str, timeout: int = 30) -> Image.Image:
    """Download an image from a URL and return as PIL Image."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return Image.open(io.BytesIO(resp.read())).convert("RGB")


def main():
    # Create output directories
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Download annotations
    data = download_json(ANNOTATIONS_URL)
    all_samples = data["data"]
    print(f"Total validation samples: {len(all_samples)}")

    # Over-sample to handle broken Flickr URLs, then keep first NUM_SAMPLES
    random.seed(SEED)
    candidates = random.sample(all_samples, min(OVERSAMPLE, len(all_samples)))
    print(f"Trying {len(candidates)} candidates to get {NUM_SAMPLES} working samples (seed={SEED})")

    # Download images and collect metadata
    metadata_records = []
    failed = 0
    attempted = 0

    for sample in candidates:
        if len(metadata_records) >= NUM_SAMPLES:
            break

        attempted += 1
        qid = sample["question_id"]
        image_file = f"images/{qid}.jpg"
        image_path = OUTPUT_DIR / image_file

        # Skip if already downloaded
        if image_path.exists():
            metadata_records.append(
                {
                    "question_id": qid,
                    "image_id": sample["image_id"],
                    "question": sample["question"],
                    "answers": sample["answers"],
                    "image_file": image_file,
                }
            )
            continue

        # Try flickr_300k_url first, then flickr_original_url
        urls_to_try = [sample["flickr_300k_url"]]
        if sample["flickr_original_url"] != sample["flickr_300k_url"]:
            urls_to_try.append(sample["flickr_original_url"])

        success = False
        for url in urls_to_try:
            try:
                img = download_image(url)
                img.save(image_path, "JPEG")
                success = True
                break
            except Exception:
                continue

        if success:
            metadata_records.append(
                {
                    "question_id": qid,
                    "image_id": sample["image_id"],
                    "question": sample["question"],
                    "answers": sample["answers"],
                    "image_file": image_file,
                }
            )
        else:
            failed += 1

        # Progress
        done = len(metadata_records)
        if done % 20 == 0 or done == NUM_SAMPLES:
            print(f"  [{done}/{NUM_SAMPLES}] downloaded ({failed} failed, {attempted} attempted)")

    # Write metadata
    with open(METADATA_FILE, "w") as f:
        for record in metadata_records:
            f.write(json.dumps(record) + "\n")

    # --- Verification ---
    print("\n=== Verification ===")
    num_images = len(list(IMAGES_DIR.glob("*.jpg")))
    num_metadata = sum(1 for _ in open(METADATA_FILE))
    print(f"Images downloaded: {num_images}")
    print(f"Metadata entries:  {num_metadata}")
    if failed:
        print(f"Failed downloads:  {failed} (skipped, replaced by next candidate)")

    # Show 3 example samples
    print("\n=== Example Samples ===")
    with open(METADATA_FILE) as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            record = json.loads(line)
            img = Image.open(OUTPUT_DIR / record["image_file"])
            print(f"  [{i+1}] Q: {record['question']}")
            print(f"      Answers: {record['answers'][:3]}...")
            print(f"      Image: {img.size[0]}x{img.size[1]}")

    if num_images == NUM_SAMPLES and num_metadata == NUM_SAMPLES:
        print(f"\nSUCCESS: {NUM_SAMPLES} samples ready at {OUTPUT_DIR}")
    else:
        print(f"\nWARNING: Expected {NUM_SAMPLES}, got {num_images} images / {num_metadata} metadata")
        sys.exit(1)


if __name__ == "__main__":
    main()
