# Plan: Evaluate Apertus on VTC-Bench

## Context
VTC-Bench is a 680-problem benchmark for evaluating agentic multimodal models via compositional visual tool chaining (9 categories, 26 distinct tools in ground truth). We need to evaluate Apertus 8B on it. Implementation goes inside `verl-apertus` following existing patterns with minimal changes. Key decisions:
- **Reuse VTC-Bench's OpenCV tool implementations** (from `VTC-Bench/eval/qwen_agent/tools/opencv_interface.py`) — no reinventing the wheel
- **Use Apertus's native tool calling format** (`<|tools_prefix|>[{"name": args}]<|tools_suffix|>`) from `chat_template.jinja`
- **Raw output, no `<answer>` tags** — just prompt well for MCQ (single letter)
- **Images not yet downloaded** — need preparation step

## Files to Create/Modify

### 1. `datasets/prepare_vtc_bench.py` — Data preparation
- Parse `/capstor/scratch/cscs/msayfiddinov/VTC-Bench/data/VTC-Bench.tsv` (680 rows)
- Download images from HuggingFace `zzzhu/VTC-Bench` → `datasets/vtc_bench/images/`
- Generate `datasets/vtc_bench/metadata.jsonl`:
  ```json
  {"question_id": "attention_focusing_1", "question": "...", "answers": ["B"], "image_file": "images/attention_focusing/...", "category": "attention", "is_mc": true, "options": {"A": "6", "B": "5", ...}}
  ```
- For MC questions: append `"\nOptions:\nA. ...\nB. ...\nC. ...\nD. ...\nAnswer with a single letter."` to question
- For open-ended: keep question as-is, `answers: ["光陽機車"]`

### 2. `inference/run_vtc_bench.py` — Baseline inference (adapt `run_baseline.py`)
Minimal changes from `run_baseline.py`:
- Default paths: `datasets/vtc_bench` → `results/vtc_bench/baseline`
- Prompt: use `build_prompt_manual()` with added instruction for MCQ: `"For multiple choice questions, answer with a single letter (A, B, C, or D). For other questions, answer concisely."`
- **No `<answer>` tags** — use raw model output as prediction directly
- Include `category` and `is_mc` fields in output JSONL
- `max_new_tokens`: 512 (longer reasoning may be needed)
- Add `--max-samples` flag for testing

### 3. `inference/run_vtc_bench_tool_agent.py` — Tool-agent with native format + VTC-Bench OpenCV tools
This is the main new script. Uses Apertus's native tool calling format and VTC-Bench's OpenCV tool implementations.

**Architecture:**
- Import `_apply_op` and `_cv2_to_pil` from `VTC-Bench/eval/qwen_agent/tools/opencv_interface.py`
- Define OpenCV tools as JSON Schema dicts (name, description, parameters) matching VTC-Bench's tool definitions from `opencv_tools.py`
- Use `tokenizer.apply_chat_template(messages, tools=tool_defs, ...)` to format prompts
- **Multi-turn loop:**
  1. Generate text with tool definitions in the developer section
  2. Parse `<|tools_prefix|>[{"tool_name": args}]<|tools_suffix|>` from output
  3. Execute tool via `_apply_op(cv_img, op_name, params)` — takes OpenCV image, returns (processed_cv_img, message_str)
  4. Re-encode result image via IBQ → token string for next turn
  5. Append tool result to conversation and continue
  6. When no tool call detected, use the generated text as final answer
- Single-GPU sequential (same as existing `run_tool_agent.py`)
- Output: `results/vtc_bench/tool_agent/predictions.jsonl`

**Tool call parsing:**
```python
# Parse <|tools_prefix|>[{"tool_name": {args}}]<|tools_suffix|>
NATIVE_TOOL_PATTERN = re.compile(r'<\|tools_prefix\|>\[(.*?)\]<\|tools_suffix\|>', re.DOTALL)
```

**Tool execution bridge** (wrapping VTC-Bench's `_apply_op`):
```python
def execute_opencv_tool(op_name, params, pil_image):
    """Execute VTC-Bench OpenCV tool on a PIL image."""
    cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    processed_cv, message = _apply_op(cv_img, op_name, params)
    result_pil = _cv2_to_pil(processed_cv)
    return result_pil, message
```

**Tool definitions** — define the ~32 tools matching `opencv_tools.py` as a list of dicts with `name`, `description`, `parameters` (JSON Schema). These get rendered as TypeScript signatures by the chat template.

**Prompt construction** — use `apply_chat_template` with:
```python
messages = [
    {"role": "user", "content": f"{image_token_str}\n{question}"}
]
# tools = list of tool definition dicts
prompt = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
```

**After tool execution**, append results by extending the prompt with tool output tokens following the template format:
```
<|tools_prefix|>[{"op_name": {params}}]<|tools_suffix|>
[message_text\n\n{new_image_tokens}]
```

### 4. `evaluation/compute_vtc_accuracy.py` — VTC-Bench evaluation
- Load predictions JSONL
- Two evaluation modes based on `is_mc` field:
  - **MC**: Extract option letter (handle "B", "B.", "B) text", "The answer is B"), compare case-insensitively
  - **Open-ended**: Normalized exact match (import `normalize_answer` from `compute_accuracy.py`)
- Per-category accuracy (9 categories)
- Per-tier accuracy (Tier 1: attention/ocr/perceptual, Tier 2: measure/color/counting, Tier 3: chart/math/spatial)
- Overall accuracy, MC-only accuracy, open-ended-only accuracy
- Output JSON + printed summary

### 5. `configs/apertus.yaml` — Add VTC-Bench config
```yaml
vtc_bench:
  num_samples: 680
  seed: 42
  eval_mode: vtc
  tsv_path: /capstor/scratch/cscs/msayfiddinov/VTC-Bench/data/VTC-Bench.tsv
```

### 6. `slurm/run_vtc_bench_baseline.slurm` — Baseline SLURM (adapt `run_baseline.slurm`)
- Job name: `apertus_vtc_bench_baseline`, time: `4:00:00`
- Pre-flight: check `datasets/vtc_bench/metadata.jsonl`
- Run inference then evaluation

### 7. `slurm/run_vtc_bench_tool_agent.slurm` — Tool-agent SLURM
- Job name: `apertus_vtc_bench_tool_agent`, time: `8:00:00`
- Add VTC-Bench to PYTHONPATH for opencv_interface import
- Run inference then evaluation

## Key Files to Reference
- `inference/run_baseline.py` — template for baseline script
- `inference/run_tool_agent.py` — template for tool-agent (but switching to native format)
- `inference/vision.py` — `encode_image()`, `load_vq_model()` (unchanged)
- `evaluation/compute_accuracy.py` — `normalize_answer()` to reuse
- `HF/chat_template.jinja` — native tool calling format reference
- `VTC-Bench/eval/qwen_agent/tools/opencv_interface.py` — `_apply_op()`, `_cv2_to_pil()` to import
- `VTC-Bench/eval/qwen_agent/tools/opencv_tools.py` — tool definitions (name, description, parameters) to copy

## Implementation Order
1. `datasets/prepare_vtc_bench.py` — run first to get data
2. `configs/apertus.yaml` — add config entry
3. `inference/run_vtc_bench.py` — baseline (simplest, test pipeline)
4. `evaluation/compute_vtc_accuracy.py` — evaluate baseline
5. `slurm/run_vtc_bench_baseline.slurm`
6. `inference/run_vtc_bench_tool_agent.py` — tool agent (most complex)
7. `slurm/run_vtc_bench_tool_agent.slurm`

## Verification
1. Run `prepare_vtc_bench.py` — verify 680 entries in metadata.jsonl, images load
2. Test baseline: `python inference/run_vtc_bench.py --max-samples 5`
3. Test evaluation: `python evaluation/compute_vtc_accuracy.py --predictions results/vtc_bench/baseline/predictions.jsonl`
4. Test tool agent: `python inference/run_vtc_bench_tool_agent.py --max-samples 3`
5. Submit full SLURM jobs

## Notes
- VTC-Bench's `_apply_op()` works on OpenCV images (numpy arrays). We convert PIL↔OpenCV at the boundary, then re-encode result images via IBQ for Apertus.
- The `generate()` wrapper in VTC-Bench saves to files and returns ContentItem — we skip that and use `_apply_op()` directly.
- Some answers contain Chinese characters — evaluation preserves unicode.
- Apertus may not follow native tool format perfectly since SFT training data is unknown — fallback to raw text if no tool call detected.
