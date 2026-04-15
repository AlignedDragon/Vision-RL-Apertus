# Plan: Qwen 2.5 VL 7B Evaluation on Same Datasets

## Context

We have 4 standardized VQA datasets (TextVQA, HR-Bench, VStar, POPE) in `datasets/` with JSONL+images format, plus an evaluation script with vqa/mcq/binary modes. We want to evaluate Qwen 2.5 VL 7B Instruct on the same datasets for comparison with Apertus 8B.

Key difference: Qwen uses continuous vision features (processor-based), not discrete IBQ tokens. No VQ model needed — much simpler image pipeline.

Branch: `qwen-eval`

---

## Step 1: Create branch

```
git checkout -b qwen-eval
```

## Step 2: Create `inference/run_qwen_baseline.py`

Mirrors `inference/run_baseline.py` structure but adapted for Qwen 2.5 VL:

- **Model**: `Qwen/Qwen2.5-VL-7B-Instruct` (HF hub, ~15GB bf16)
- **Dependencies**: `transformers`, `qwen-vl-utils`, `accelerate`
- **No VQ model** — images go through `AutoProcessor` + `process_vision_info()`
- **Chat template**: processor-based, not manual string construction
- **Multi-GPU**: Same 4-GPU data parallelism as Apertus baseline

**Prompt format:**
```python
messages = [
    {"role": "system", "content": "Answer with a single word or short phrase."},
    {"role": "user", "content": [
        {"type": "image", "image": pil_image},
        {"type": "text", "text": question},
    ]}
]
```

**Inference pattern per sample:**
```python
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)
output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
# Trim prompt tokens from output
generated = output_ids[0, inputs.input_ids.shape[1]:]
prediction = processor.decode(generated, skip_special_tokens=True).strip()
```

**Output**: Same JSONL format as Apertus — `predictions.jsonl` with `question_id, question, prediction, answers, prompt_tokens, generated_tokens`.

**Key simplifications vs Apertus baseline:**
- No Phase 1 (VQ encoding) — images processed inline during generation
- No manual prompt construction — processor handles chat template
- No Emu3.5 PYTHONPATH dependency

### Files to reuse:
- `evaluation/compute_accuracy.py` — unchanged, works on same prediction format
- Dataset loading: same `load_metadata()` + `Image.open()` pattern from `run_baseline.py`

## Step 3: Create `inference/run_qwen_tool_agent.py`

Mirrors `inference/run_tool_agent.py` but for Qwen:

- Same tool engine (`tools/engine.py`), same tool call format (`TOOL: {"name": ..., "args": ...}`)
- Same few-shot examples and multi-turn loop
- Difference: image re-encoding after CropZoomIn/Line tools passes new PIL image directly (no IBQ re-encode needed — just update the message with new image)
- Same parse logic (`TOOL_PATTERN` regex)

## Step 4: Config

**Modify:** `configs/qwen.yaml` (new file)

```yaml
model:
  checkpoint: Qwen/Qwen2.5-VL-7B-Instruct

generation:
  max_new_tokens: 256
  temperature: 0.0

dataset:
  textvqa:
    eval_mode: vqa
  hrbench:
    eval_mode: mcq
  vstar:
    eval_mode: mcq
  pope:
    eval_mode: binary

tool_agent:
  max_turns: 5
```

## Step 5: SLURM script

**Create:** `slurm/run_qwen_baseline.slurm`

Same as `slurm/run_baseline.slurm` but:
- Runs `inference/run_qwen_baseline.py`
- No Emu3.5 PYTHONPATH
- Needs `qwen-vl-utils` installed

---

## Implementation Order

1. `git checkout -b qwen-eval`
2. `configs/qwen.yaml`
3. `inference/run_qwen_baseline.py` (baseline, 4-GPU parallel)
4. `inference/run_qwen_tool_agent.py` (tool-use, single GPU)
5. `slurm/run_qwen_baseline.slurm`

## Dependencies

```bash
pip install qwen-vl-utils accelerate
```

## Verification

1. Run on 2-3 samples from TextVQA to confirm inference works
2. Run baseline on all 4 datasets, evaluate with `compute_accuracy.py`
3. Compare Qwen vs Apertus accuracy tables
