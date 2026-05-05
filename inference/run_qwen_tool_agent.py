"""Qwen 2.5 VL tool-agent inference — multi-turn loop with CogCoM-style tools.

Same tool engine and few-shot prompting as the Apertus tool agent, but
adapted for Qwen's processor-based image handling. Image tool results
(CropZoomIn, Line) are passed as new PIL images in the conversation —
no IBQ re-encoding needed.

Usage (via SLURM):
    sbatch slurm/run_qwen_tool_agent.slurm

Usage (interactive, on a GPU node):
    python inference/run_qwen_tool_agent.py --config configs/qwen.yaml --dataset-dir data_prep/textvqa
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

from tools.engine import execute_tool, TOOL_DESCRIPTIONS

# ---------------------------------------------------------------------------
# Few-shot examples and prompts (same as Apertus tool agent)
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

TOOL_INSTRUCTION = (
    "You have access to tools that help you examine images more closely. "
    "Use them when the question requires reading text, locating objects, "
    "counting, or doing math on visual information.\n\n"
    f"{TOOL_DESCRIPTIONS}\n\n"
    f"{FEW_SHOT_EXAMPLES}\n\n"
    "When you have the final answer, state it concisely in a single short phrase. "
    "Do not wrap your answer in a full sentence."
)

TOOL_PATTERN = re.compile(r'TOOL:\s*(\{.*\})', re.DOTALL)


def parse_tool_call(text: str) -> dict | None:
    matches = TOOL_PATTERN.findall(text)
    if not matches:
        return None
    try:
        return json.loads(matches[-1])
    except json.JSONDecodeError:
        return None


def load_metadata(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def run_tool_agent(
    sample: dict,
    image: Image.Image,
    processor,
    model,
    process_vision_info_fn,
    device: str,
    max_turns: int = 5,
    max_new_tokens: int = 256,
) -> dict:
    """Run multi-turn tool-agent loop for one sample."""
    current_image = image.copy()
    tool_calls_log = []

    # Build initial conversation
    messages = [
        {"role": "system", "content": TOOL_INSTRUCTION},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": current_image},
                {"type": "text", "text": sample["question"]},
            ],
        },
    ]

    for turn in range(max_turns):
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info_fn(messages)
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
        generated_text = processor.decode(generated_ids, skip_special_tokens=True).strip()

        tool_call = parse_tool_call(generated_text)
        if tool_call is None:
            break

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

        # Append assistant response + tool result to conversation
        messages.append({"role": "assistant", "content": generated_text})

        if result["image"] is not None:
            current_image = result["image"]
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "[Tool result: (zoomed image)]"},
                    {"type": "image", "image": current_image},
                ],
            })
        else:
            result_text = result["text"] or "(no result)"
            messages.append({
                "role": "user",
                "content": f"[Tool result: {result_text}]",
            })

    # Extract final answer
    prediction = generated_text
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
    parser = argparse.ArgumentParser(description="Qwen 2.5 VL tool-agent inference")
    parser.add_argument("--config", default="configs/qwen.yaml")
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else PROJECT_ROOT / "data_prep" / "textvqa"
    dataset_name = dataset_dir.name
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "results" / dataset_name / "qwen_tool_agent"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(str(dataset_dir / "metadata.jsonl"))
    if args.max_samples:
        metadata = metadata[:args.max_samples]
    print(f"Loaded {len(metadata)} samples from {dataset_dir}")

    device = "cuda:0"
    checkpoint = config["model"]["checkpoint"]

    print(f"Loading Qwen model from {checkpoint}...")
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    processor = AutoProcessor.from_pretrained(checkpoint)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    print("Model loaded")

    results = []
    start = time.time()

    for i, sample in enumerate(metadata):
        image = Image.open(dataset_dir / sample["image_file"]).convert("RGB")

        result = run_tool_agent(
            sample=sample,
            image=image,
            processor=processor,
            model=model,
            process_vision_info_fn=process_vision_info,
            device=device,
            max_turns=args.max_turns,
            max_new_tokens=config["generation"]["max_new_tokens"],
        )
        results.append(result)

        elapsed = time.time() - start
        print(f"[{i+1}/{len(metadata)}] {elapsed:.0f}s | turns={result['num_turns']} tools={len(result['tool_calls'])} | Q: {sample['question'][:60]}...")

    output_path = output_dir / "predictions.jsonl"
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\n=== Done ===")
    print(f"Predictions: {output_path}")
    print(f"Total time: {time.time() - start:.0f}s")

    for r in results[:3]:
        print(f"\n  Q: {r['question']}")
        print(f"  A: {r['prediction']}")
        print(f"  Tools: {[tc['tool'] for tc in r['tool_calls']]}")
        print(f"  Expected: {r['answers'][:3]}")


if __name__ == "__main__":
    main()
