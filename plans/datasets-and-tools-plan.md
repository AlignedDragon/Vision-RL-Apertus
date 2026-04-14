# Plan: Datasets + CogCoM-Style Tools for Apertus

## Context

Apertus 8B baseline gets 4.5% on TextVQA (format mismatch, not blindness). We need to:
1. Download 4 HF datasets, standardize to existing JSONL+images format, 200 samples each
2. Add 6 CogCoM-style image manipulation tools
3. Build a multi-turn inference loop that prompts Apertus to use the tools via few-shot examples

Apertus was **never trained on tool calls** — we teach it via prompting only.

---

## Step 1: Dataset Download Script

**Create:** `datasets/prepare_all.py`

One script, downloads all 4 datasets using HF `datasets` library. Each produces `datasets/{name}/metadata.jsonl` + `images/` dir. Same schema as existing TextVQA:

```json
{"question_id": 123, "question": "...", "answers": ["..."], "image_file": "images/123.jpg", "extra": {}}
```

| Dataset | HF ID | Split to use | Standardization |
|---------|-------|-------------|-----------------|
| textvqa | `facebook/textvqa` | validation | Direct: 10 answers, subsample 200 |
| hrbench | `DreamMr/HR-Bench` | `hrbench_4k` | MCQ → answers=[correct option text], extra={A,B,C,D,label} |
| vstar | `craigwu/vstar_bench` | `test` | MCQ → answers=[correct option text], extra={options,label}. Only 191 samples, take all |
| pope | `lmms-lab/POPE` | `test` | Binary → answers=["yes"/"no"]. Subsample 200 balanced |

---

## Step 2: Evaluation Updates

**Modify:** `evaluation/compute_accuracy.py`

Add `--mode` flag:
- `vqa` (default): existing soft accuracy
- `mcq`: exact match after normalization
- `binary`: yes/no accuracy + F1

Minimal change — just add two elif branches to the scoring function.

---

## Step 3: CogCoM Tools

**Create:** `tools/` directory with one file per tool + an engine.

6 tools from CogCoM:

| Tool | File | Dependency | Implementation |
|------|------|-----------|----------------|
| OCR | `tools/ocr_tool.py` | easyocr | `easyocr.Reader(['en']).readtext(image)` |
| Grounding | `tools/grounding_tool.py` | groundingdino | GroundingDINO open-set detection (GPU) |
| CropZoomIn | `tools/crop_zoom_tool.py` | PIL (built-in) | Crop bbox + resize to original dims |
| Counting | `tools/counting_tool.py` | uses Grounding | Run grounding → count boxes |
| Calculate | `tools/calculate_tool.py` | none | `ast.literal_eval` safe math eval |
| Line | `tools/line_tool.py` | PIL (built-in) | `ImageDraw.line()` |

**Create:** `tools/engine.py` — Simple dict dispatch:
```python
def execute_tool(tool_name, args, image) -> {"text": str|None, "image": Image|None}
```

---

## Step 4: Multi-Turn Tool Agent

**Create:** `inference/run_tool_agent.py`

Extends baseline inference with a turn loop:
1. Build prompt with image + question + few-shot tool examples
2. Generate response
3. Parse for tool calls (regex on JSON patterns — since model isn't trained on special tokens, we prompt it to output JSON like `{"tool": "OCR", "args": {...}}`)
4. Execute tool, append result to conversation
5. Generate again (up to max_turns=5)
6. Extract final answer

The tool call format in the prompt is taught via few-shot examples — just JSON, nothing special.

---

## Step 5: Prompt with Few-Shot Examples

Included directly in `inference/run_tool_agent.py` (no separate module needed). The prompt teaches tool use by example:

```
You can use these tools to examine the image:
- OCR(bbox?) - read text from image region
- Grounding(description) - find objects, returns bounding boxes  
- CropZoomIn(bbox, ratio) - zoom into a region
- Counting(description) - count objects
- Calculate(expression) - do math
- Line(points) - draw lines on image

To use a tool, output: TOOL: {"name": "OCR", "args": {"bbox": [x1,y1,x2,y2]}}

Example:
Q: What does the sign say?
A: Let me read the text on the sign.
TOOL: {"name": "OCR", "args": {}}
[Tool result: "Welcome to Springfield"]
The sign says "Welcome to Springfield".
```

---

## Step 6: Config Update

**Modify:** `configs/apertus.yaml` — add new dataset paths and tool settings.

---

## Implementation Order

1. `datasets/prepare_all.py` (can run on login node, no GPU)
2. `evaluation/compute_accuracy.py` updates
3. `tools/` (OCR, Grounding, CropZoomIn, Counting, Calculate, Line, engine)
4. `inference/run_tool_agent.py` with embedded prompts
5. `configs/apertus.yaml` updates

## Dependencies

- `pip install datasets easyocr`
- GroundingDINO: needs separate install + model weights (~700MB). Will use `groundingdino` pip package.

## Verification

1. Run `python datasets/prepare_all.py` — check 200 samples per dataset, images load
2. Run baseline inference on each dataset — verify predictions + accuracy files
3. Test each tool independently on a sample image
4. Run tool agent on 5 TextVQA samples — verify multi-turn works, tools fire, answers produced
