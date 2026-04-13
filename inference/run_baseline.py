"""Apertus baseline inference on TextVQA.

Runs Apertus 8B on 200 TextVQA samples with no special prompting.
Uses 4-GPU data parallelism: VQ encoding on GPU 0, then 4 parallel
processes each load the model and process a quarter of samples.

Usage (via SLURM):
    sbatch slurm/run_baseline.slurm

Usage (interactive, on a GPU node):
    python inference/run_baseline.py --config configs/apertus.yaml
"""

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
    """Build Apertus prompt manually (fallback if apply_chat_template fails).

    Follows the checkpoint's chat_template.jinja structure exactly:
    - System message with default Apertus identity
    - Developer section with deliberation=disabled, tools=disabled
    - User message with image tokens + question
    - Generation prompt (assistant_start)
    """
    return (
        "<s>"
        "<|system_start|>"
        "You are Apertus, a helpful assistant created by the SwissAI initiative.\n"
        "Knowledge cutoff: 2024-04\n"
        "Current date: 2026-04-13"
        "<|system_end|>"
        "<|developer_start|>"
        "Deliberation: disabled\n"
        "Tool Capabilities: disabled"
        "<|developer_end|>"
        "<|user_start|>"
        f"{image_token_str}\n{question}"
        "<|user_end|>"
        "<|assistant_start|>"
    )


def try_apply_chat_template(tokenizer, question: str) -> str | None:
    """Try using the model's built-in chat template. Returns None on failure."""
    messages = [{"role": "user", "content": f"<|image|>\n{question}"}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return None


def encode_all_images(
    metadata: list[dict],
    dataset_dir: Path,
    vq_model_path: str,
    device: str = "cuda:0",
) -> dict[int, str]:
    """Encode all images to IBQ token strings using the VQ model.

    Returns dict mapping question_id -> image_token_string.
    """
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

    # Free VQ model memory
    del vq_model
    torch.cuda.empty_cache()
    print(f"VQ model freed. Encoded {len(image_tokens)} images total.")

    return image_tokens


def worker_inference(
    gpu_id: int,
    samples: list[dict],
    image_tokens: dict[int, str],
    config: dict,
    output_path: Path,
):
    """Worker process: load Apertus on one GPU and generate answers for a chunk of samples."""
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

    # Check if apply_chat_template works (test once)
    test_prompt = try_apply_chat_template(tokenizer, "test question")
    use_template = test_prompt is not None
    if use_template:
        print(f"[GPU {gpu_id}] Using model's built-in chat template")
    else:
        print(f"[GPU {gpu_id}] Chat template failed, using manual prompt construction")

    max_new_tokens = config["generation"]["max_new_tokens"]

    # Stop tokens: </s> (2) and <|assistant_end|> (68)
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

        # Build prompt
        if use_template:
            prompt_text = try_apply_chat_template(tokenizer, question)
            prompt_text = prompt_text.replace("<|image|>", token_str, 1)
        else:
            prompt_text = build_prompt_manual(token_str, question)

        # Tokenize (BOS already in prompt from template)
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt", add_special_tokens=False)
        input_ids = input_ids.to(device)

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy
                eos_token_id=stop_ids,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the generated tokens (skip prompt)
        generated_ids = output_ids[0, input_ids.shape[1]:]
        prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        results.append({
            "question_id": qid,
            "question": question,
            "prediction": prediction,
            "answers": sample["answers"],
            "prompt_tokens": int(input_ids.shape[1]),
            "generated_tokens": int(generated_ids.shape[0]),
        })

        if (i + 1) % 10 == 0 or i == len(samples) - 1:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            print(f"[GPU {gpu_id}] [{i+1}/{len(samples)}] {rate:.1f} samples/s")

    # Write partial results
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"[GPU {gpu_id}] Done. Wrote {len(results)} predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Apertus baseline inference on TextVQA")
    parser.add_argument("--config", default="configs/apertus.yaml", help="Config file path")
    parser.add_argument("--dataset-dir", default=None, help="Override dataset directory")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    args = parser.parse_args()

    config = load_config(args.config)

    # Resolve paths
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else PROJECT_ROOT / "datasets" / "textvqa"
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "results" / "textvqa" / "baseline"
    metadata_path = dataset_dir / "metadata.jsonl"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    metadata = load_metadata(str(metadata_path))
    print(f"Loaded {len(metadata)} samples from {metadata_path}")

    # Detect available GPUs
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
    # Split samples evenly across GPUs
    chunks = [[] for _ in range(num_gpus)]
    for i, sample in enumerate(metadata):
        chunks[i % num_gpus].append(sample)

    print(f"\nStarting inference on {num_gpus} GPUs ({[len(c) for c in chunks]} samples each)")

    # Use multiprocessing for true parallelism
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

    # Wait for all processes
    for p in processes:
        p.join()

    # Check for failures
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

    # Sort by question_id for deterministic output
    all_results.sort(key=lambda r: r["question_id"])

    with open(final_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    # Clean up partial files
    for pp in partial_paths:
        pp.unlink(missing_ok=True)

    print(f"\n=== Inference Complete ===")
    print(f"Total predictions: {len(all_results)}")
    print(f"Output: {final_path}")

    # Show a few example predictions
    print(f"\n=== Example Predictions ===")
    for r in all_results[:5]:
        print(f"  Q: {r['question']}")
        print(f"  A: {r['prediction']}")
        print(f"  Expected: {r['answers'][:3]}")
        print()


if __name__ == "__main__":
    main()
