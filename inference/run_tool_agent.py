"""Apertus tool-agent inference — multi-turn loop with CogCoM-style tools.

The model is prompted with few-shot examples showing how to call tools.
When it generates a TOOL: {...} line, we parse it, execute the tool,
append the result, and let the model continue.

Usage (via SLURM):
    sbatch slurm/run_tool_agent.slurm

Usage (interactive, on a GPU node):
    python inference/run_tool_agent.py --config configs/apertus.yaml --dataset-dir data_prep/textvqa
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.vision import encode_image, load_vq_model
from tools.engine import execute_tool, TOOL_DESCRIPTIONS

# ---------------------------------------------------------------------------
# Few-shot examples that teach the model how to use tools
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = """Here are examples of how to use tools:

Example 1:
User: What does the sign in the image say?
Assistant: I'll read the text on the sign.
TOOL: {"name": "OCR", "args": {}}
[Tool result: "Grand Central Terminal"]
The sign says "Grand Central Terminal".

Example 2:
User: How many dogs are in the park?
Assistant: Let me count the dogs.
TOOL: {"name": "Counting", "args": {"description": "dog"}}
[Tool result: 3]
There are 3 dogs in the park.

Example 3:
User: What small text is written at the bottom of the label?
Assistant: The text is small, let me zoom in on the bottom of the label first.
TOOL: {"name": "Grounding", "args": {"description": "label at the bottom"}}
[Tool result: [[50, 300, 200, 400]]]
TOOL: {"name": "CropZoomIn", "args": {"bbox": [50, 300, 200, 400], "ratio": 3.0}}
[Tool result: (zoomed image)]
Now let me read the text in the zoomed region.
TOOL: {"name": "OCR", "args": {}}
[Tool result: "Best before 2025-03-15"]
The small text at the bottom of the label says "Best before 2025-03-15"."""

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_MSG = (
    "You are Apertus, a helpful assistant created by the SwissAI initiative.\n"
    "Knowledge cutoff: 2024-04\n"
    "Current date: 2026-04-14"
)

TOOL_INSTRUCTION = (
    "You have access to tools that help you examine images more closely. "
    "Use them when the question requires reading text, locating objects, "
    "counting, or doing math on visual information.\n\n"
    f"{TOOL_DESCRIPTIONS}\n\n"
    f"{FEW_SHOT_EXAMPLES}\n\n"
    "When you have the final answer, state it concisely in a single short phrase. "
    "Do not wrap your answer in a full sentence."
)


def build_initial_prompt(image_token_str: str, question: str) -> str:
    """Build the first-turn prompt with tool instructions and image."""
    return (
        "<s>"
        "<|system_start|>"
        f"{SYSTEM_MSG}"
        "<|system_end|>"
        "<|developer_start|>"
        "Deliberation: enabled\n"
        "Tool Capabilities: enabled"
        "<|developer_end|>"
        "<|user_start|>"
        f"{TOOL_INSTRUCTION}\n\n"
        f"{image_token_str}\n"
        f"{question}"
        "<|user_end|>"
        "<|assistant_start|>"
    )


def append_tool_result(prompt: str, result_text: str) -> str:
    """Append a tool result and re-open assistant generation."""
    return (
        prompt +
        "<|assistant_end|>"
        "<|user_start|>"
        f"[Tool result: {result_text}]"
        "<|user_end|>"
        "<|assistant_start|>"
    )


# ---------------------------------------------------------------------------
# Tool call parsing
# ---------------------------------------------------------------------------

TOOL_PATTERN = re.compile(r'TOOL:\s*(\{.*\})', re.DOTALL)


def parse_tool_call(text: str) -> dict | None:
    """Extract the last TOOL: {...} call from generated text.

    Returns {"name": str, "args": dict} or None.
    """
    matches = TOOL_PATTERN.findall(text)
    if not matches:
        return None
    try:
        return json.loads(matches[-1])
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Main inference
# ---------------------------------------------------------------------------

def load_metadata(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def run_tool_agent(
    sample: dict,
    image_token_str: str,
    current_image: Image.Image,
    vq_model,
    tokenizer,
    model,
    device: str,
    max_turns: int = 5,
    max_new_tokens: int = 256,
) -> dict:
    """Run multi-turn tool-agent loop for one sample.

    Returns prediction dict with question_id, prediction, turns, tool_calls.
    """
    prompt = build_initial_prompt(image_token_str, sample["question"])
    tool_calls_log = []

    stop_ids = [tokenizer.eos_token_id]
    assistant_end_id = tokenizer.convert_tokens_to_ids("<|assistant_end|>")
    if isinstance(assistant_end_id, int) and assistant_end_id != tokenizer.unk_token_id:
        stop_ids.append(assistant_end_id)

    for turn in range(max_turns):
        input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
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
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Check for tool call
        tool_call = parse_tool_call(generated_text)
        if tool_call is None:
            # No tool call — this is the final answer
            prompt += generated_text
            break

        # Execute tool
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})

        try:
            result = execute_tool(tool_name, tool_args, current_image)
        except Exception as e:
            result = {"text": f"Error: {e}", "image": None}

        tool_calls_log.append({
            "turn": turn,
            "tool": tool_name,
            "args": tool_args,
            "result_text": result["text"],
            "produced_image": result["image"] is not None,
        })

        # Build result text for prompt
        if result["image"] is not None:
            # Re-encode the new image as tokens
            current_image = result["image"]
            new_image_tokens = encode_image(current_image, vq_model)
            result_text = f"(zoomed image)\n{new_image_tokens}"
        else:
            result_text = result["text"] or "(no result)"

        # Append the generated text (up to and including the tool call) + tool result
        prompt += generated_text
        prompt = append_tool_result(prompt, result_text)

    # Extract final answer: text after the last tool result, or the full response if no tools used
    final_text = generated_text
    # If there were tool calls, the final answer is the last generation
    prediction = final_text.strip()
    # Remove any remaining TOOL: lines from the prediction
    prediction = TOOL_PATTERN.sub("", prediction).strip()

    return {
        "question_id": sample["question_id"],
        "question": sample["question"],
        "prediction": prediction,
        "answers": sample["answers"],
        "num_turns": len(tool_calls_log) + 1,
        "tool_calls": tool_calls_log,
    }


def main():
    parser = argparse.ArgumentParser(description="Apertus tool-agent inference")
    parser.add_argument("--config", default="configs/apertus.yaml")
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples (for testing)")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else PROJECT_ROOT / "data_prep" / "textvqa"
    dataset_name = dataset_dir.name
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "results" / dataset_name / "tool_agent"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(str(dataset_dir / "metadata.jsonl"))
    if args.max_samples:
        metadata = metadata[:args.max_samples]
    print(f"Loaded {len(metadata)} samples from {dataset_dir}")

    # Load VQ model on GPU 0 for image encoding
    device = "cuda:0"
    vq_model = load_vq_model(config["model"]["vq_model"], device=device)
    print("VQ model loaded")

    # Pre-encode all images
    print("Encoding images...")
    image_tokens = {}
    original_images = {}
    for i, record in enumerate(metadata):
        img = Image.open(dataset_dir / record["image_file"])
        image_tokens[record["question_id"]] = encode_image(img, vq_model)
        original_images[record["question_id"]] = img.convert("RGB")
        if (i + 1) % 50 == 0:
            print(f"  Encoded {i+1}/{len(metadata)}")

    # Load Apertus model (single GPU for tool agent — needs sequential turns)
    print(f"Loading Apertus model on {device}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint = config["model"]["checkpoint"]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    model.eval()
    print("Model loaded")

    # Run tool-agent loop for each sample
    results = []
    start = time.time()

    for i, sample in enumerate(metadata):
        qid = sample["question_id"]
        result = run_tool_agent(
            sample=sample,
            image_token_str=image_tokens[qid],
            current_image=original_images[qid],
            vq_model=vq_model,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_turns=args.max_turns,
        )
        results.append(result)

        elapsed = time.time() - start
        print(f"[{i+1}/{len(metadata)}] {elapsed:.0f}s | turns={result['num_turns']} tools={len(result['tool_calls'])} | Q: {sample['question'][:60]}...")

    # Save results
    output_path = output_dir / "predictions.jsonl"
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\n=== Done ===")
    print(f"Predictions: {output_path}")
    print(f"Total time: {time.time() - start:.0f}s")

    # Show examples
    for r in results[:3]:
        print(f"\n  Q: {r['question']}")
        print(f"  A: {r['prediction']}")
        print(f"  Tools: {[tc['tool'] for tc in r['tool_calls']]}")
        print(f"  Expected: {r['answers'][:3]}")


if __name__ == "__main__":
    main()
