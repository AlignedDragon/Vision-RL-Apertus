"""Qwen 2.5 VL 7B baseline inference on VQA datasets.

Runs Qwen2.5-VL-7B-Instruct on standardized VQA datasets with no special
prompting. Uses 4-GPU data parallelism — each GPU loads its own model copy
and processes a quarter of samples.

Usage (via SLURM):
    sbatch slurm/run_qwen_baseline.slurm

Usage (interactive, on a GPU node):
    python inference/run_qwen_baseline.py --config configs/qwen.yaml --dataset-dir datasets/textvqa
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_metadata(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def worker_inference(
    gpu_id: int,
    samples: list[dict],
    dataset_dir: Path,
    config: dict,
    output_path: Path,
):
    """Worker process: load Qwen on one GPU and generate answers."""
    device = f"cuda:{gpu_id}"
    checkpoint = config["model"]["checkpoint"]
    max_new_tokens = config["generation"]["max_new_tokens"]

    print(f"[GPU {gpu_id}] Loading Qwen model from {checkpoint} ...")
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    processor = AutoProcessor.from_pretrained(checkpoint)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    print(f"[GPU {gpu_id}] Model loaded. Processing {len(samples)} samples.")

    results = []
    start = time.time()

    for i, sample in enumerate(samples):
        image = Image.open(dataset_dir / sample["image_file"]).convert("RGB")

        messages = [
            {
                "role": "system",
                "content": "Answer with a single word or short phrase.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": sample["question"]},
                ],
            },
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        prompt_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        generated_ids = output_ids[0, prompt_len:]
        prediction = processor.decode(generated_ids, skip_special_tokens=True).strip()

        results.append({
            "question_id": sample["question_id"],
            "question": sample["question"],
            "prediction": prediction,
            "answers": sample["answers"],
            "prompt_tokens": int(prompt_len),
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
    parser = argparse.ArgumentParser(description="Qwen 2.5 VL baseline inference")
    parser.add_argument("--config", default="configs/qwen.yaml")
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else PROJECT_ROOT / "datasets" / "textvqa"
    dataset_name = dataset_dir.name
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "results" / dataset_name / "qwen_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(str(dataset_dir / "metadata.jsonl"))
    print(f"Loaded {len(metadata)} samples from {dataset_dir}")

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("ERROR: No GPUs available.")
        sys.exit(1)
    print(f"Available GPUs: {num_gpus}")

    # Split samples across GPUs
    chunks = [[] for _ in range(num_gpus)]
    for i, sample in enumerate(metadata):
        chunks[i % num_gpus].append(sample)

    print(f"Starting inference on {num_gpus} GPUs ({[len(c) for c in chunks]} samples each)")

    mp.set_start_method("spawn", force=True)
    partial_paths = []
    processes = []

    for gpu_id in range(num_gpus):
        partial_path = output_dir / f"predictions_gpu{gpu_id}.jsonl"
        partial_paths.append(partial_path)

        p = mp.Process(
            target=worker_inference,
            args=(gpu_id, chunks[gpu_id], dataset_dir, config, partial_path),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    for i, p in enumerate(processes):
        if p.exitcode != 0:
            print(f"ERROR: Worker GPU {i} failed with exit code {p.exitcode}")
            sys.exit(1)

    # Merge partial results
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

    for r in all_results[:5]:
        print(f"\n  Q: {r['question']}")
        print(f"  A: {r['prediction']}")
        print(f"  Expected: {r['answers'][:3]}")


if __name__ == "__main__":
    main()
