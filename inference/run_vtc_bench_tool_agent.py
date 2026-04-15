"""Apertus tool-agent inference on VTC-Bench using native tool calling format.

Uses Apertus's native tool calling format (<|tools_prefix|>/<|tools_suffix|>)
with VTC-Bench's OpenCV tool implementations.

Usage (via SLURM):
    sbatch slurm/run_vtc_bench_tool_agent.slurm

Usage (interactive, on a GPU node):
    python inference/run_vtc_bench_tool_agent.py --config configs/apertus.yaml
    python inference/run_vtc_bench_tool_agent.py --config configs/apertus.yaml --max-samples 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Add VTC-Bench to path for opencv_interface
VTC_BENCH_ROOT = Path("/capstor/scratch/cscs/msayfiddinov/VTC-Bench/eval")
sys.path.insert(0, str(VTC_BENCH_ROOT))

from inference.vision import encode_image, load_vq_model

# Mock qwen_agent dependencies that opencv_interface imports but _apply_op doesn't use
import types
import logging

_mock_schema = types.ModuleType("qwen_agent.llm.schema")


class _ContentItem:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


_mock_schema.ContentItem = _ContentItem

_mock_utils = types.ModuleType("qwen_agent.utils.utils")
_mock_utils.load_image_from_base64 = lambda x: None
_mock_utils.logger = logging.getLogger("opencv_interface")

sys.modules["qwen_agent"] = types.ModuleType("qwen_agent")
sys.modules["qwen_agent.llm"] = types.ModuleType("qwen_agent.llm")
sys.modules["qwen_agent.llm.schema"] = _mock_schema
sys.modules["qwen_agent.utils"] = types.ModuleType("qwen_agent.utils")
sys.modules["qwen_agent.utils.utils"] = _mock_utils

# Import VTC-Bench's OpenCV tool execution directly from file
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "opencv_interface",
    str(VTC_BENCH_ROOT / "qwen_agent" / "tools" / "opencv_interface.py"),
)
_opencv_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_opencv_mod)
_apply_op = _opencv_mod._apply_op
_cv2_to_pil = _opencv_mod._cv2_to_pil


# ---------------------------------------------------------------------------
# Tool definitions (TypeScript-style for Apertus chat template)
# ---------------------------------------------------------------------------

# Pre-rendered tool capability string matching the chat_template.jinja format.
# Each tool is defined as: // description\ntype name = (_: {params}) => any;
TOOL_CAPABILITIES = """\
// Converts image to grayscale
type colorspace_gray = (_: {
image: string
}) => any;
// Converts image to HSV color space
type colorspace_hsv = (_: {
image: string
}) => any;
// Converts image to LAB color space
type colorspace_lab = (_: {
image: string
}) => any;
// Resizes the image to specified dimensions or by preset (half/double)
type resize = (_: {
image: string,
param: {
width: number,
height: number,
preset?: string
}
}) => any;
// Rotates the image by specified angle in degrees (clockwise)
type rotate = (_: {
image: string,
param: {
angle: number
}
}) => any;
// Shifts the image by pixels in a direction (left/right/up/down)
type translate = (_: {
image: string,
param: {
direction: string,
distance?: number
}
}) => any;
// Flips the image horizontally or vertically
type flip = (_: {
image: string,
param: {
direction: string
}
}) => any;
// Applies blur (average/gaussian/median/bilateral) to reduce noise
type blur = (_: {
image: string,
param: {
method: string,
ksize: number
}
}) => any;
// Applies thresholding to create binary image (binary/otsu/adaptive)
type threshold = (_: {
image: string,
param: {
mode: string,
invert?: boolean,
color_mode?: string
}
}) => any;
// Applies morphological operations (erode/dilate/open/close)
type morphology = (_: {
image: string,
param?: {
op?: string,
kernel_size?: number,
iterations?: number
}
}) => any;
// Computes gradient images using Sobel or Laplacian
type gradients = (_: {
image: string,
param: {
mode: string
}
}) => any;
// Detects edges using Canny edge detector
type canny = (_: {
image: string,
param?: {
preset?: string,
threshold_low?: number,
threshold_high?: number
}
}) => any;
// Applies image pyramid (pyr_up/pyr_down)
type pyramid = (_: {
image: string,
param: {
mode: string
}
}) => any;
// Finds contours and returns bounding boxes/areas
type contours = (_: {
image: string,
param?: {
mode?: string,
rank?: number,
max_contours?: number
}
}) => any;
// Draws detected contours on the image
type draw_contours = (_: {
image: string,
param?: {
mode?: string,
rank?: number
}
}) => any;
// Draws a line between two points
type draw_line = (_: {
image: string,
param: {
x1: number,
y1: number,
x2: number,
y2: number,
color?: number[],
thickness?: number
}
}) => any;
// Calculates contour areas
type contour_area = (_: {
image: string,
param?: {
mode?: string,
rank?: number
}
}) => any;
// Computes contour perimeters
type arc_length = (_: {
image: string,
param?: {
mode?: string,
rank?: number
}
}) => any;
// Approximates contours with fewer points
type approx_poly = (_: {
image: string,
param?: {
epsilon_ratio?: number,
rank?: number
}
}) => any;
// Enhances contrast via histogram equalization or CLAHE
type histogram = (_: {
image: string,
param?: {
mode?: string,
clip_limit?: number,
tile_grid_size?: number
}
}) => any;
// Computes DFT magnitude spectrum
type dft = (_: {
image: string
}) => any;
// Matches a template within the source image
type template_match = (_: {
image: string,
param: {
template: string
}
}) => any;
// Detects lines using Hough transform
type hough_lines = (_: {
image: string,
param?: {
min_line_length?: number,
max_line_gap?: number,
threshold?: number
}
}) => any;
// Detects circles using Hough transform
type hough_circles = (_: {
image: string,
param?: {
min_radius?: number,
max_radius?: number,
min_dist?: number
}
}) => any;
// Applies watershed segmentation
type watershed = (_: {
image: string,
param?: {
kernel_size?: number,
iterations?: number
}
}) => any;
// Foreground/background segmentation using GrabCut
type grabcut = (_: {
image: string,
param: {
rect: number[]
}
}) => any;
// Detects keypoints (harris/shi_tomasi/orb/sift)
type features = (_: {
image: string,
param?: {
method?: string,
max_corners?: number
}
}) => any;
// Applies denoising
type denoise = (_: {
image: string,
param?: {
method?: string,
h?: number
}
}) => any;
// Inpaints regions in the image
type inpaint = (_: {
image: string,
param?: {
method?: string,
radius?: number
}
}) => any;
// Creates mask for pixels within a color range
type inrange_color = (_: {
image: string,
param: {
lower: number[],
upper: number[],
colorspace?: string
}
}) => any;
// Crops a rectangular region
type crop = (_: {
image: string,
param: {
x: number,
y: number,
width: number,
height: number
}
}) => any;
// Zooms into a specific region
type zoom_in = (_: {
image: string,
param: {
x: number,
y: number,
width: number,
height: number,
target_width?: number,
target_height?: number
}
}) => any;
// Flood fill from a seed point
type floodfill = (_: {
image: string,
param: {
seed_x: number,
seed_y: number
}
}) => any;
// Finds connected components in binary image
type connected_components_with_stats = (_: {
image: string,
param?: {
connectivity?: number
}
}) => any;
// Scales image to 0-255 range
type convertscaleabs = (_: {
image: string,
param?: {
alpha?: number,
beta?: number
}
}) => any;
// Draws a circle at specified coordinates
type draw_circle = (_: {
image: string,
param: {
center_x: number,
center_y: number,
radius: number,
color?: number[],
thickness?: number
}
}) => any;"""


# ---------------------------------------------------------------------------
# Prompt construction (native format)
# ---------------------------------------------------------------------------

SYSTEM_MSG = (
    "You are Apertus, a helpful assistant created by the SwissAI initiative.\n"
    "Knowledge cutoff: 2024-04\n"
    "Current date: 2026-04-15"
)


def build_initial_prompt(image_token_str: str, question: str) -> str:
    """Build first-turn prompt with native tool definitions."""
    return (
        "<s>"
        "<|system_start|>"
        f"{SYSTEM_MSG}"
        "<|system_end|>"
        "<|developer_start|>"
        "Deliberation: disabled\n"
        "Tool Capabilities:\n"
        f"{TOOL_CAPABILITIES}"
        "<|developer_end|>"
        "<|user_start|>"
        f"{image_token_str}\n{question}\n\n"
        "Use the available image processing tools to analyze the image and answer the question. "
        "Answer concisely."
        "<|user_end|>"
        "<|assistant_start|>"
    )


def append_tool_result(prompt: str, result_text: str) -> str:
    """Append tool output in native format and re-open assistant generation.

    Native format: tool outputs come as [result] after the tool call,
    then assistant continues.
    """
    return prompt + f"[{result_text}]"


# ---------------------------------------------------------------------------
# Tool call parsing (native format)
# ---------------------------------------------------------------------------

NATIVE_TOOL_PATTERN = re.compile(
    r'<\|tools_prefix\|>\[(.*?)\]<\|tools_suffix\|>', re.DOTALL
)


def parse_native_tool_call(text: str) -> tuple[str, dict] | None:
    """Parse a native tool call from generated text.

    Expected format: <|tools_prefix|>[{"op_name": {args}}]<|tools_suffix|>

    Returns (op_name, params) or None.
    """
    match = NATIVE_TOOL_PATTERN.search(text)
    if not match:
        return None
    try:
        calls = json.loads("[" + match.group(1) + "]")
        if not calls:
            return None
        # Take the first call
        call = calls[0]
        if isinstance(call, dict) and len(call) == 1:
            op_name = list(call.keys())[0]
            params = call[op_name]
            if isinstance(params, dict):
                return op_name, params
    except (json.JSONDecodeError, IndexError, KeyError):
        pass
    return None


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def execute_opencv_tool(
    op_name: str, params: dict, pil_image: Image.Image
) -> tuple[Image.Image | None, str]:
    """Execute a VTC-Bench OpenCV tool on a PIL image.

    Returns (result_image, message_text).
    result_image is None if the operation only returns text/data.
    """
    cv_img = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    try:
        processed_cv, message = _apply_op(cv_img, op_name, params.get("param", {}))
        result_pil = _cv2_to_pil(processed_cv)
        return result_pil, message
    except Exception as e:
        return None, f"Error: {e}"


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
    max_new_tokens: int = 512,
) -> dict:
    """Run multi-turn tool-agent loop for one sample using native tool format."""
    prompt = build_initial_prompt(image_token_str, sample["question"])
    tool_calls_log = []
    final_text = ""

    stop_ids = [tokenizer.eos_token_id]
    # Stop on assistant_end (final answer) or tools_suffix (end of tool call)
    for token_name in ("<|assistant_end|>", "<|tools_suffix|>"):
        tid = tokenizer.convert_tokens_to_ids(token_name)
        if isinstance(tid, int) and tid != tokenizer.unk_token_id:
            stop_ids.append(tid)

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
        # Decode WITHOUT skipping special tokens so we can see tool call markers
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()

        # Check for native tool call
        tool_call = parse_native_tool_call(generated_text)
        if tool_call is None:
            # No tool call — this is the final answer
            # Decode again with skip_special_tokens for clean text
            final_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            prompt += generated_text
            break

        op_name, params = tool_call

        # Execute tool
        try:
            result_image, result_text = execute_opencv_tool(op_name, params, current_image)
        except Exception as e:
            result_image = None
            result_text = f"Error executing {op_name}: {e}"

        tool_calls_log.append({
            "turn": turn,
            "tool": op_name,
            "params": params,
            "result_text": result_text[:500],
            "produced_image": result_image is not None,
        })

        # Build result string for prompt
        if result_image is not None:
            current_image = result_image
            new_image_tokens = encode_image(current_image, vq_model)
            tool_result_str = f"{result_text}\n\n{new_image_tokens}"
        else:
            tool_result_str = result_text or "(no result)"

        # Append generated text (including tool call) + tool result
        prompt += generated_text
        prompt = append_tool_result(prompt, tool_result_str)
    else:
        # Reached max turns — use last generated text
        final_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Clean up prediction: remove any tool call remnants
    prediction = final_text
    prediction = NATIVE_TOOL_PATTERN.sub("", prediction).strip()

    return {
        "question_id": sample["question_id"],
        "question": sample["question"],
        "prediction": prediction,
        "answers": sample["answers"],
        "category": sample["category"],
        "is_mc": sample["is_mc"],
        "num_turns": len(tool_calls_log) + 1,
        "tool_calls": tool_calls_log,
    }


def main():
    parser = argparse.ArgumentParser(description="Apertus tool-agent on VTC-Bench")
    parser.add_argument("--config", default="configs/apertus.yaml")
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else PROJECT_ROOT / "datasets" / "vtc_bench"
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "results" / "vtc_bench" / "tool_agent"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(str(dataset_dir / "metadata.jsonl"))
    if args.max_samples:
        metadata = metadata[:args.max_samples]
    print(f"Loaded {len(metadata)} samples from {dataset_dir}")

    # Load VQ model on GPU 0
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

    # Load Apertus model
    print(f"Loading Apertus model on {device}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint = config["model"]["checkpoint"]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    model.eval()
    print("Model loaded")

    # Run tool-agent loop
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
        print(
            f"[{i+1}/{len(metadata)}] {elapsed:.0f}s | "
            f"turns={result['num_turns']} tools={len(result['tool_calls'])} | "
            f"Q: {sample['question'][:60]}..."
        )

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
        print(f"\n  Q: {r['question'][:80]}...")
        print(f"  A: {r['prediction'][:100]}")
        print(f"  Tools: {[tc['tool'] for tc in r['tool_calls']]}")
        print(f"  Expected: {r['answers']}")


if __name__ == "__main__":
    main()
