"""Download and prepare 4 VQA datasets from HuggingFace.

Downloads TextVQA, HR-Bench, VStar-Bench, and POPE.
Each gets standardized to JSONL metadata + images/ directory.
No GPU needed — runs on the login node.

Usage:
    pip install datasets Pillow huggingface_hub
    python data_prep/prepare_all.py                  # all datasets
    python data_prep/prepare_all.py --only textvqa   # single dataset
"""

import base64
import io
import json
import random
import re
import sys
from pathlib import Path

from datasets import load_dataset
from PIL import Image

SEED = 42
NUM_SAMPLES = 200
SCRIPT_DIR = Path(__file__).resolve().parent


def save_dataset(name: str, records: list[dict], images: dict):
    """Save records as metadata.jsonl and images as JPEG.

    images: dict mapping question_id -> PIL.Image
    """
    out_dir = SCRIPT_DIR / name
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    for qid, img in images.items():
        img.convert("RGB").save(img_dir / f"{qid}.jpg", "JPEG")

    with open(out_dir / "metadata.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"  Saved {len(records)} samples to {out_dir}")


# ---------------------------------------------------------------------------
# TextVQA
# ---------------------------------------------------------------------------

def prepare_textvqa():
    """facebook/textvqa — validation split, 200 samples.

    Uses refs/convert/parquet revision since the dataset script is no longer
    supported by newer datasets library versions.
    """
    print("\n=== TextVQA ===")
    ds = load_dataset(
        "facebook/textvqa",
        split="validation",
        revision="refs/convert/parquet",
    )

    random.seed(SEED)
    indices = random.sample(range(len(ds)), NUM_SAMPLES)

    records, images = [], {}
    for idx in indices:
        row = ds[idx]
        qid = row["question_id"]
        records.append({
            "question_id": qid,
            "question": row["question"],
            "answers": row["answers"],
            "image_file": f"images/{qid}.jpg",
        })
        images[qid] = row["image"]

    save_dataset("textvqa", records, images)

def prepare_tool_vqa():
    """DietCoke4671/ToolVQA.
    """
    from huggingface_hub import hf_hub_download
    print("\n=== ToolVQA ===")
    ds = load_dataset(
        "json",
        data_files="hf://datasets/DietCoke4671/ToolVQA/train.jsonl",
        split="train",
    )

    random.seed(SEED)
    indices = random.sample(range(len(ds)), NUM_SAMPLES)

    records, images = [], {}
    for idx in indices:
        row = ds[idx]
        qid = idx
        records.append({
            "question_id": idx,
            "image_path": row["image_path"],
            "question": row["question"],
            "context": row["context"],
            "ori_question": row["ori_question"],
            "thought_rethink": row["thought_rethink"],
            "thought_question": row["question"],
            "answer": row["answer"],
            "type": row["type"],
        })

        image_path_in_repo = row["image_path"]
        local_path = hf_hub_download(
            repo_id="DietCoke4671/ToolVQA",
            filename=image_path_in_repo,
            repo_type="dataset",
        )
        img = Image.open(local_path)
        images[qid] = img

    save_dataset("toolvqa", records, images)
# ---------------------------------------------------------------------------
# HR-Bench
# ---------------------------------------------------------------------------

def prepare_hrbench():
    """DreamMr/HR-Bench — hrbench_4k split, 200 samples.

    Multiple choice. Image is base64-encoded JPEG string.
    """
    print("\n=== HR-Bench ===")
    ds = load_dataset("DreamMr/HR-Bench", split="hrbench_4k")

    random.seed(SEED)
    indices = random.sample(range(len(ds)), min(NUM_SAMPLES, len(ds)))

    records, images = [], {}
    for idx in indices:
        row = ds[idx]
        qid = row["index"]
        correct_letter = row["answer"]
        correct_text = row[correct_letter]
        options = {k: row[k] for k in ["A", "B", "C", "D"]}

        # Decode base64 JPEG
        img_bytes = base64.b64decode(row["image"])
        img = Image.open(io.BytesIO(img_bytes))

        records.append({
            "question_id": qid,
            "question": row["question"],
            "answers": [correct_text],
            "image_file": f"images/{qid}.jpg",
            "extra": {
                **options,
                "correct_letter": correct_letter,
                "category": row["category"],
            },
        })
        images[qid] = img

    save_dataset("hrbench", records, images)


# ---------------------------------------------------------------------------
# VStar-Bench
# ---------------------------------------------------------------------------

def prepare_vstar():
    """craigwu/vstar_bench — test split, all 191 samples.

    Multiple choice. Images are stored as files in the HF repo,
    the `image` column is just a relative path string.
    """
    print("\n=== VStar-Bench ===")
    from huggingface_hub import hf_hub_download

    ds = load_dataset("craigwu/vstar_bench", split="test")

    records, images = [], {}
    for i, row in enumerate(ds):
        qid = i
        correct_letter = row["label"]
        text = row["text"]

        # Parse options: format is "(A) option\n(B) option\n..."
        options = {}
        for letter in ["A", "B", "C", "D"]:
            match = re.search(rf"\({letter}\)\s*(.+?)(?:\n|$)", text)
            if match:
                options[letter] = match.group(1).strip()

        correct_text = options.get(correct_letter, correct_letter)
        # Question is the first line
        question_text = text.split("\n")[0].strip()

        # Download image from HF repo
        image_path_in_repo = row["image"]
        local_path = hf_hub_download(
            repo_id="craigwu/vstar_bench",
            filename=image_path_in_repo,
            repo_type="dataset",
        )
        img = Image.open(local_path)

        records.append({
            "question_id": qid,
            "question": question_text,
            "answers": [correct_text],
            "image_file": f"images/{qid}.jpg",
            "extra": {
                "options": options,
                "correct_letter": correct_letter,
                "category": row["category"],
                "original_text": text,
            },
        })
        images[qid] = img

        if (i + 1) % 50 == 0:
            print(f"  Downloaded {i+1}/{len(ds)} images")

    save_dataset("vstar", records, images)


# ---------------------------------------------------------------------------
# POPE
# ---------------------------------------------------------------------------

def prepare_pope():
    """lmms-lab/POPE — test split, 200 samples (balanced yes/no)."""
    print("\n=== POPE ===")
    ds = load_dataset("lmms-lab/POPE", split="test")

    yes_indices = [i for i in range(len(ds)) if ds[i]["answer"] == "yes"]
    no_indices = [i for i in range(len(ds)) if ds[i]["answer"] == "no"]

    random.seed(SEED)
    half = NUM_SAMPLES // 2
    selected = random.sample(yes_indices, half) + random.sample(no_indices, half)
    random.shuffle(selected)

    records, images = [], {}
    for idx in selected:
        row = ds[idx]
        qid = idx

        records.append({
            "question_id": qid,
            "question": row["question"],
            "answers": [row["answer"]],
            "image_file": f"images/{qid}.jpg",
            "extra": {
                "category": row["category"],
                "image_source": row["image_source"],
            },
        })
        images[qid] = row["image"]

    save_dataset("pope", records, images)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DATASETS = {
    "textvqa": prepare_textvqa,
    "hrbench": prepare_hrbench,
    "vstar": prepare_vstar,
    "pope": prepare_pope,
}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download and prepare VQA datasets")
    parser.add_argument("--only", choices=list(DATASETS.keys()), help="Prepare only one dataset")
    args = parser.parse_args()

    targets = [args.only] if args.only else list(DATASETS.keys())

    prepare_tool_vqa()
    # for name in targets:
    #     try:
    #         DATASETS[name]()
    #     except Exception as e:
    #         print(f"ERROR preparing {name}: {e}", file=sys.stderr)
    #         import traceback
    #         traceback.print_exc()
    #         continue

    print("\nDone.")


if __name__ == "__main__":
    main()
