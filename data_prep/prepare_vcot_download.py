"""Download deepcs233/Visual-CoT train metadata and write filtered raw.jsonl.

No GPU needed. This keeps the CoF data-prep style: download the train split,
filter rows whose answer is a single word and that have a single bbox, and
write a compact raw.jsonl for the later parse/SFT/RL steps.

Usage:
    python data_prep/prepare_vcot_download.py
    python data_prep/prepare_vcot_download.py --output-dir /tmp/vcot
"""

import argparse
import json
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

REPO_ID = "deepcs233/Visual-CoT"
SCRIPT_DIR = Path(__file__).resolve().parent


def is_single_word_answer(answer) -> bool:
    if not isinstance(answer, str):
        return False
    answer = answer.strip()
    if not answer:
        return False
    return len(answer.split()) == 1


def has_single_bbox(bboxs) -> bool:
    return isinstance(bboxs, list) and len(bboxs) == 1


def iter_train_rows():
    """Yield train rows from JSON/JSONL files in the HF dataset repo."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    train_files = [
        path
        for path in files
        if re.search(r"(^|/|_)train(_|\.|-)", path, flags=re.IGNORECASE)
        and path.lower().endswith((".jsonl", ".json"))
    ]
    detailed_train_files = [
        path for path in train_files if path.startswith("cot_with_detailed_reasoning_steps/")
    ]
    if detailed_train_files:
        train_files = detailed_train_files
    if not train_files:
        raise FileNotFoundError(f"No train JSON/JSONL files found in {REPO_ID}")

    for filename in sorted(train_files):
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
        )
        print(f"Loading {filename} from {local_path}")
        with open(local_path, "r", encoding="utf-8") as f:
            if filename.lower().endswith(".jsonl"):
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
            else:
                rows = json.load(f)
                if isinstance(rows, dict):
                    rows = rows.get("train") or rows.get("data") or rows.get("rows")
                if not isinstance(rows, list):
                    raise ValueError(f"Unsupported JSON structure in {filename}")
                yield from rows


def normalize_row(row: dict) -> dict:
    missing = [
        key
        for key in ("question", "image", "thought", "full_answer", "answer", "bboxs")
        if key not in row
    ]
    if missing:
        raise KeyError(f"Missing required fields {missing} in row with keys={sorted(row)}")
    return {
        "question": row["question"],
        "image": row["image"],
        "data_source": row.get("data_source") or row.get("dataset") or row.get("source"),
        "thought": row["thought"],
        "full_answer": row["full_answer"],
        "answer": row["answer"],
        "bboxs": row["bboxs"],
    }


def download_dataset(output_dir: Path, split: str = "train") -> tuple[int, int]:
    """Download train metadata and write filtered raw.jsonl.

    Returns (kept_rows, total_rows).
    """
    if split != "train":
        raise ValueError(f"Only the 'train' split is supported, got {split!r}")

    print(f"Loading {REPO_ID} split={split} ...")
    raw_path = output_dir / "raw.jsonl"
    kept = 0
    total = 0
    with open(raw_path, "w", encoding="utf-8") as f:
        for row in iter_train_rows():
            total += 1
            if not is_single_word_answer(row.get("answer")):
                continue
            if not has_single_bbox(row.get("bboxs")):
                continue
            f.write(json.dumps(normalize_row(row), ensure_ascii=False) + "\n")
            kept += 1

    print(f"Wrote {kept}/{total} rows to {raw_path}")
    return kept, total


def download_and_extract_images(output_dir: Path):
    """Download split image tar from HF and extract into images/."""
    from huggingface_hub import HfApi, hf_hub_download

    images_dir = output_dir / "images"
    if images_dir.exists() and any(images_dir.iterdir()):
        count = sum(1 for _ in images_dir.rglob("*") if _.is_file())
        print(f"images/ already populated ({count} files), skipping extraction")
        return

    api = HfApi()
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    parts = sorted(path for path in files if path.startswith("cot_images_tar_split/cot_images_"))
    if not parts:
        raise FileNotFoundError(f"No cot image tar parts found in {REPO_ID}")

    print(f"Downloading {len(parts)} image tar part(s) ...")
    part_paths = [
        hf_hub_download(repo_id=REPO_ID, filename=part, repo_type="dataset")
        for part in parts
    ]

    images_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting images into {images_dir} ...")
    if shutil.which("tar"):
        cmd = ["tar", "-xf", "-", "-C", str(images_dir)]
        with subprocess.Popen(cmd, stdin=subprocess.PIPE) as proc:
            assert proc.stdin is not None
            for part_path in part_paths:
                with open(part_path, "rb") as f:
                    shutil.copyfileobj(f, proc.stdin)
            proc.stdin.close()
            rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"tar extraction failed with exit code {rc}")
    else:
        class _ConcatReader:
            def __init__(self, paths):
                self.paths = iter(paths)
                self.current = None

            def read(self, size=-1):
                chunks = []
                remaining = size
                while size < 0 or remaining > 0:
                    if self.current is None:
                        try:
                            self.current = open(next(self.paths), "rb")
                        except StopIteration:
                            break
                    chunk = self.current.read(-1 if size < 0 else remaining)
                    if not chunk:
                        self.current.close()
                        self.current = None
                        continue
                    chunks.append(chunk)
                    if size > 0:
                        remaining -= len(chunk)
                return b"".join(chunks)

        with tarfile.open(fileobj=_ConcatReader(part_paths), mode="r|*") as tf:
            tf.extractall(images_dir)

    count = sum(1 for _ in images_dir.rglob("*") if _.is_file())
    print(f"Extracted {count} files")


def main():
    parser = argparse.ArgumentParser(description="Download Visual-CoT train metadata for Apertus prep")
    parser.add_argument("--output-dir", default=None, help="Default: data_prep/vcot")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "vcot"
    output_dir.mkdir(parents=True, exist_ok=True)

    kept, total = download_dataset(output_dir, split=args.split)
    download_and_extract_images(output_dir)

    print(f"\nDone. {kept}/{total} filtered rows ready at {output_dir}")


if __name__ == "__main__":
    main()
