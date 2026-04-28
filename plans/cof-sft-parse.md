# Plan: `prepare_cof_sft_parse.py`

## Context
`prepare_cof_sft_download.py` already pulls the [CoF-SFT-Data-5.4k](https://huggingface.co/datasets/xintongzhang/CoF-SFT-Data-5.4k) raw dataset into [datasets/cof_sft/raw.jsonl](../datasets/cof_sft/raw.jsonl) with the original Qwen-style messages + `image_paths`. We need an SFT counterpart to [datasets/prepare_cof_rl_parse.py](../datasets/prepare_cof_rl_parse.py) that renders each row as a full Apertus-formatted training example: system + developer (tools) + user (with IBQ image tokens) + assistant (thoughts + tool_call) + **proper Apertus tool message** (image-only, no text) + assistant (thoughts + `display_answers` tool call).

Two key differences from the RL parse:
1. SFT data carries the full assistant trajectory, so we render the entire conversation (no `add_generation_prompt`).
2. Each row contains **2 images** — the user-given one (first user message) and the tool response one (second "user" message in the source, which in Apertus is a `role: tool` message containing only IBQ tokens, no surrounding text).

## Files

- **New:** [datasets/prepare_cof_sft_parse.py](../datasets/prepare_cof_sft_parse.py) — parse script (mirrors RL parse).
- **Reuse, no edits:**
  - [datasets/prepare_cof_rl_parse.py](../datasets/prepare_cof_rl_parse.py) — copy/share helpers (`extract_tool_def`, `QWEN_TRAILER_SENTINEL`, `APERTUS_INSTRUCTION`, `APERTUS_SYSTEM`, `DISPLAY_ANSWERS_TOOL`, `load_config`).
  - [inference/vision.py](../inference/vision.py) — `encode_image`, `load_vq_model`.
  - [configs/apertus.yaml](../configs/apertus.yaml) — same checkpoint + VQ paths.

## Source row shape (already produced by download step)

```json
{
  "messages": [
    {"role": "system",    "content": "<qwen system with <tools>...</tools>>"},
    {"role": "user",      "content": "<image> Question: ...Think in the mind first..."},
    {"role": "assistant", "content": "<think>...</think>\n<tool_call>{...}</tool_call>"},
    {"role": "user",      "content": "<image>\nThink in the mind first..."},
    {"role": "assistant", "content": "<think>...</think>\n<answer>C</answer>"}
  ],
  "image_paths": ["images/foo.jpg", "images/foo_zoom.jpg"]
}
```

The 2nd `user` message is semantically a tool response, not a user turn.

## Target Apertus messages (per row)

```python
[
  {"role": "system",    "content": APERTUS_SYSTEM},
  {"role": "user",      "content": f"{question_head}\n\n{APERTUS_INSTRUCTION}"  # <image> -> image1 IBQ tokens
                       },
  {"role": "assistant", "content": {"blocks": [
      {"type": "thoughts",   "text": <think_1>},
      {"type": "tool_calls", "calls": [{"name": <src_tool_name>, "arguments": <src_tool_args_json_str>}]}
  ]}},
  {"role": "tool",      "content": <image2 IBQ tokens>},   # ONLY image, no text
  {"role": "assistant", "content": {"blocks": [
      {"type": "thoughts",   "text": <think_2>},
      {"type": "tool_calls", "calls": [{"name": "display_answers",
                                        "arguments": json.dumps({"answer": <answer>})}]}
  ]}}
]
```

Rendered with the existing template via:

```python
tokenizer.apply_chat_template(
    messages,
    tools=[tool_def, DISPLAY_ANSWERS_TOOL],
    enable_thinking=True,
    add_generation_prompt=False,
    tokenize=False,
)
```

The template (apertus-format `chat_template.jinja` lines 232–290 for assistant blocks, 313–324 for tool messages) renders the tool message as `[<image2_tokens>]` after the assistant's `<|tools_prefix|>...<|tools_suffix|>` block and resumes inside the same assistant span — exactly the trajectory we want the model to imitate.

## Parsing helpers (new, kept terse)

- `THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)`
- `TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)`
- `ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)`

`parse_intermediate_assistant(text) -> (thoughts, ToolCall)`:
- one `<think>` + one `<tool_call>` JSON `{"name", "arguments"}`. Re-serialize `arguments` as a compact JSON string so it lands as valid JSON inside the rendered tool_calls block.

`parse_final_assistant(text) -> (thoughts, answer_str)`:
- one `<think>` + one `<answer>`. Build `display_answers` call with `arguments=json.dumps({"answer": answer})`.

## Output (one record per source row, JSONL)

```json
{
  "text": "<full rendered Apertus conversation string>",
  "image_paths": ["<abs path to image1>", "<abs path to image2>"]
}
```

This mirrors the RL parse's "rendered string + image path" shape; the trainer can compute SFT loss by masking on Apertus's special-token boundaries (`<|assistant_start|>` … `<|assistant_end|>`).

## Script outline

```python
# datasets/prepare_cof_sft_parse.py
import argparse, json, re, sys, time
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.vision import encode_image, load_vq_model
from datasets.prepare_cof_rl_parse import (
    APERTUS_SYSTEM, APERTUS_INSTRUCTION, QWEN_TRAILER_SENTINEL,
    DISPLAY_ANSWERS_TOOL, extract_tool_def, load_config,
)

THINK_RE     = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
ANSWER_RE    = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)

def build_user_text(raw_user_text, image_token_str):
    head = raw_user_text.split(QWEN_TRAILER_SENTINEL, 1)[0].rstrip()
    head = head.replace("<image>", image_token_str, 1)
    return f"{head}\n\n{APERTUS_INSTRUCTION}"

def parse_intermediate_assistant(text):
    think = THINK_RE.search(text).group(1).strip()
    call_obj = json.loads(TOOL_CALL_RE.search(text).group(1).strip())
    args_str = json.dumps(call_obj["arguments"], ensure_ascii=False, separators=(", ", ": "))
    return think, {"name": call_obj["name"], "arguments": args_str}

def parse_final_assistant(text):
    think = THINK_RE.search(text).group(1).strip()
    answer = ANSWER_RE.search(text).group(1).strip()
    args_str = json.dumps({"answer": answer}, ensure_ascii=False)
    return think, {"name": "display_answers", "arguments": args_str}

def build_messages(src_msgs, image1_tokens, image2_tokens):
    # src_msgs is the 5-message Qwen list; we expect [system, user, asst, user, asst].
    raw_user  = next(m["content"] for m in src_msgs if m["role"] == "user")
    asst1, asst2 = (m for m in src_msgs if m["role"] == "assistant")
    th1, call1 = parse_intermediate_assistant(asst1["content"])
    th2, call2 = parse_final_assistant(asst2["content"])
    return [
        {"role": "system", "content": APERTUS_SYSTEM},
        {"role": "user",   "content": build_user_text(raw_user, image1_tokens)},
        {"role": "assistant", "content": {"blocks": [
            {"type": "thoughts",   "text": th1},
            {"type": "tool_calls", "calls": [call1]},
        ]}},
        {"role": "tool", "content": image2_tokens},   # image only, no text
        {"role": "assistant", "content": {"blocks": [
            {"type": "thoughts",   "text": th2},
            {"type": "tool_calls", "calls": [call2]},
        ]}},
    ]

def main():
    # argparse: --input (default datasets/cof_sft/raw.jsonl),
    #          --output (default datasets/cof_sft/metadata.jsonl),
    #          --config configs/apertus.yaml, --limit
    # 1. load rows
    # 2. extract tool_def from rows[0]["messages"] system; sanity-check vs first 50
    # 3. load AutoTokenizer + IBQ vq_model on cuda:0
    # 4. for each row:
    #      - require len(image_paths) >= 2; else skip
    #      - encode image1 = image_paths[0], image2 = image_paths[1]
    #      - messages = build_messages(...)
    #      - text = tokenizer.apply_chat_template(
    #            messages, tools=[tool_def, DISPLAY_ANSWERS_TOOL],
    #            enable_thinking=True, add_generation_prompt=False, tokenize=False)
    #      - write {"text": text, "image_paths": [str(p1), str(p2)]}
    # 5. log skip count + text-length percentiles (mirroring the RL parse)
```

## Verification

The Apertus checkpoint, IBQ tokenizer, and `cof_sft/raw.jsonl` are **not** present on this box, so the end-to-end script can't run locally. The local check is a structural one in the `ezgatr` conda env (it has the python deps); the real run happens on the cluster.

1. **Local structural check (ezgatr conda env, no GPU, no checkpoint):**
   ```bash
   conda activate ezgatr
   cd ~/work/verl-apertus
   python - <<'PY'
   # Import the new module and exercise its pure-python parsers + message builder
   # against a hand-crafted row matching the example in the task description.
   # Stub image tokens and skip the apply_chat_template / IBQ calls.
   from datasets import prepare_cof_sft_parse as M
   import json
   src = {  # one synthetic row with the 5-message Qwen structure from the task
       "messages": [
           {"role": "system", "content": '<tools>{"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom","parameters":{"properties":{"bbox_2d":{"type":"array","items":{"type":"number"}}},"required":["bbox_2d"],"type":"object"}}}</tools>'},
           {"role": "user", "content": "<image> Question: foo?\nThink in the mind first..."},
           {"role": "assistant", "content": '<think> step1 </think>\n<tool_call>\n{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [1,2,3,4]}}\n</tool_call>'},
           {"role": "user", "content": "<image>\nThink in the mind first..."},
           {"role": "assistant", "content": "<think> step2 </think>\n<answer> C </answer>"},
       ],
       "image_paths": ["images/a.jpg", "images/b.jpg"],
   }
   msgs = M.build_messages(src["messages"], "<IMG1>", "<IMG2>")
   assert [m["role"] for m in msgs] == ["system","user","assistant","tool","assistant"], msgs
   assert msgs[3]["content"] == "<IMG2>", "tool message must be image-only"
   assert "<IMG1>" in msgs[1]["content"] and "Think in the mind first" not in msgs[1]["content"]
   assert msgs[2]["content"]["blocks"][1]["calls"][0]["name"] == "image_zoom_in_tool"
   assert json.loads(msgs[4]["content"]["blocks"][1]["calls"][0]["arguments"]) == {"answer": "C"}
   print("structural check OK")
   PY
   ```
2. **Cluster smoke test (after `prepare_cof_sft_download.py` has run):**
   ```bash
   sbatch slurm/prepare_cof_sft.slurm   # or interactive on a GPU node:
   # python datasets/prepare_cof_sft_parse.py --limit 5
   ```
   On the produced `metadata.jsonl`, assert:
   - `set(r) == {"text", "image_paths"}` and `len(r["image_paths"]) == 2`
   - exactly **two** `<|img_start|>` occurrences in `r["text"]` (one in the user message, one inside the `[ ... ]` tool span)
   - the rendered text contains the expected boundary tokens: `<|system_start|>`, `<|developer_start|>` listing both `image_zoom_in_tool` and `display_answers`, `<|tools_prefix|>...<|tools_suffix|>` for both tool calls, and a single closing `<|assistant_end|>`.
3. **Full run on the cluster:** drop `--limit`; expect ~5.4k records and a text-length histogram printed like the RL parse's.

## Out of scope

- No SLURM script in this plan (the user can clone [slurm/prepare_cof_rl.slurm](../slurm/prepare_cof_rl.slurm) trivially once the parse script is in).
- No changes to existing files; `prepare_cof_rl_parse.py` is imported as-is.
