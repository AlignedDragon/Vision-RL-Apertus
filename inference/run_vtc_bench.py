"""Apertus baseline inference on VTC-Bench.

Runs Apertus 8B on 680 VTC-Bench samples with no tools (single-turn VQA).
Uses 4-GPU data parallelism: VQ encoding on GPU 0, then 4 parallel
processes each load the model and process a quarter of samples.

Usage (via SLURM):
    sbatch slurm/run_vtc_bench_baseline.slurm

Usage (interactive, on a GPU node):
    python inference/run_vtc_bench.py --config configs/apertus.yaml
    python inference/run_vtc_bench.py --config configs/apertus.yaml --max-samples 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp
import yaml
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.vision import encode_image, load_vq_model


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_metadata(metadata_path: str) -> list[dict]:
    records = []
    with open(metadata_path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def build_prompt_manual(image_token_str: str, question: str) -> str:
    """Build Apertus prompt with VTC-Bench question.

    For MCQ questions, the question already contains options and
    "Answer with a single letter." instruction from data preparation.
    For open-ended, we add a concise answer instruction.
    """
    return (
        "<s>"
        "<|system_start|>"
        "You are Apertus, a helpful assistant created by the SwissAI initiative.\n"
        "Knowledge cutoff: 2024-04\n"
        "Current date: 2026-04-15"
        "<|system_end|>"
        "<|developer_start|>"
        "Deliberation: disabled\n"
        "Tool Capabilities: disabled"
        "<|developer_end|>"
        "<|user_start|>"
        f"{image_token_str}\n{question}\n\n"
        "Answer concisely."
        "<|user_end|>"
        "<|assistant_start|>"
    )


def encode_all_images(
    metadata: list[dict],
    dataset_dir: Path,
    vq_model_path: str,
    device: str = "cuda:0",
) -> dict[str, str]:
    """Encode all images to IBQ token strings."""
    print(f"Loading VQ model from {vq_model_path} ...")
    vq_model = load_vq_model(vq_model_path, device=device)
    print(f"VQ model loaded on {device}")

    image_tokens = {}
    start = time.time()
    for i, record in enumerate(metadata):
        image_path = dataset_dir / record["image_file"]
        image = Image.open(image_path)
        token_str = encode_image(image, vq_model)
        image_tokens[record["question_id"]] = token_str

        if (i + 1) % 50 == 0 or i == len(metadata) - 1:
            elapsed = time.time() - start
            print(f"  Encoded [{i+1}/{len(metadata)}] images ({elapsed:.1f}s)")

    del vq_model
    torch.cuda.empty_cache()
    print(f"VQ model freed. Encoded {len(image_tokens)} images total.")

    return image_tokens


def worker_inference(
    gpu_id: int,
    samples: list[dict],
    image_tokens: dict[str, str],
    config: dict,
    output_path: Path,
):
    """Worker process: load Apertus on one GPU and generate answers."""
    device = f"cuda:{gpu_id}"
    checkpoint = config["model"]["checkpoint"]

    print(f"[GPU {gpu_id}] Loading model from {checkpoint} ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"[GPU {gpu_id}] Model loaded. Processing {len(samples)} samples.")

    max_new_tokens = config.get("generation", {}).get("max_new_tokens", 256)
    # Use longer generation for VTC-Bench
    max_new_tokens = max(max_new_tokens, 512)

    # Stop tokens
    stop_ids = [tokenizer.eos_token_id]
    assistant_end_id = tokenizer.convert_tokens_to_ids("<|assistant_end|>")
    if isinstance(assistant_end_id, int) and assistant_end_id != tokenizer.unk_token_id:
        stop_ids.append(assistant_end_id)

    results = []
    start = time.time()

    for i, sample in enumerate(samples):
        qid = sample["question_id"]
        question = sample["question"]
        token_str = image_tokens[qid]

        prompt_text = build_prompt_manual(token_str, question)

        input_ids = tokenizer.encode(prompt_text, return_tensors="pt", add_special_tokens=False)
        input_ids = input_ids.to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=stop_ids,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_ids = output_ids[0, input_ids.shape[1]:]
        prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        results.append({
            "question_id": qid,
            "question": question,
            "prediction": prediction,
            "answers": sample["answers"],
            "category": sample["category"],
            "is_mc": sample["is_mc"],
            "prompt_tokens": int(input_ids.shape[1]),
            "generated_tokens": int(generated_ids.shape[0]),
        })

        if (i + 1) % 10 == 0 or i == len(samples) - 1:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            print(f"[GPU {gpu_id}] [{i+1}/{len(samples)}] {rate:.1f} samples/s")

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"[GPU {gpu_id}] Done. Wrote {len(results)} predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Apertus baseline inference on VTC-Bench")
    parser.add_argument("--config", default="configs/apertus.yaml", help="Config file path")
    parser.add_argument("--dataset-dir", default=None, help="Override dataset directory")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples")
    args = parser.parse_args()

    config = load_config(args.config)

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else PROJECT_ROOT / "data_prep" / "vtc_bench"
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "results" / "vtc_bench" / "baseline"
    metadata_path = dataset_dir / "metadata.jsonl"

    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(str(metadata_path))
    if args.max_samples:
        metadata = metadata[:args.max_samples]
    print(f"Loaded {len(metadata)} samples from {metadata_path}")

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("ERROR: No GPUs available. Run this on a compute node via SLURM.")
        sys.exit(1)
    print(f"Available GPUs: {num_gpus}")

    # Phase 1: Encode all images on GPU 0
    image_tokens = encode_all_images(
        metadata, dataset_dir, config["model"]["vq_model"], device="cuda:0"
    )

    # Phase 2: Parallel inference across all GPUs
    chunks = [[] for _ in range(num_gpus)]
    for i, sample in enumerate(metadata):
        chunks[i % num_gpus].append(sample)

    print(f"\nStarting inference on {num_gpus} GPUs ({[len(c) for c in chunks]} samples each)")

    mp.set_start_method("spawn", force=True)
    partial_paths = []
    processes = []

    for gpu_id in range(num_gpus):
        partial_path = output_dir / f"predictions_gpu{gpu_id}.jsonl"
        partial_paths.append(partial_path)

        p = mp.Process(
            target=worker_inference,
            args=(gpu_id, chunks[gpu_id], image_tokens, config, partial_path),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    for i, p in enumerate(processes):
        if p.exitcode != 0:
            print(f"ERROR: Worker GPU {i} failed with exit code {p.exitcode}")
            sys.exit(1)

    # Phase 3: Merge partial results
    final_path = output_dir / "predictions.jsonl"
    all_results = []
    for pp in partial_paths:
        with open(pp) as f:
            for line in f:
                all_results.append(json.loads(line))

    all_results.sort(key=lambda r: r["question_id"])

    with open(final_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    for pp in partial_paths:
        pp.unlink(missing_ok=True)

    print(f"\n=== Inference Complete ===")
    print(f"Total predictions: {len(all_results)}")
    print(f"Output: {final_path}")

    # Category breakdown
    from collections import Counter
    cat_counts = Counter(r["category"] for r in all_results)
    print(f"\nPer-category counts:")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count}")

    # Show example predictions
    print(f"\n=== Example Predictions ===")
    for r in all_results[:5]:
        print(f"  Q: {r['question'][:80]}...")
        print(f"  A: {r['prediction'][:100]}")
        print(f"  Expected: {r['answers']}")
        print()


if __name__ == "__main__":
    main()
