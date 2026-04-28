# Plan: CoF-RL Apertus pipeline on verl — tools, parquet, reward, trainer

## Context

We are wiring `verl-apertus` into verl's RL trainer for the CoF-RL dataset. Apertus consumes images as inline IBQ token strings (no PIL inputs at the model level), and the SFT data trained the model to call a tool named `image_zoom_in_tool` (parameter `bbox_2d`) plus a terminal `display_answers` tool. The fully-rendered Apertus prompt produced by [datasets/prepare_cof_rl_parse.py](../../../work/verl-apertus/datasets/prepare_cof_rl_parse.py) already advertises both tools in the developer block (lines 54–67, 106–114) and replaces `<image>` with IBQ tokens inline.

This plan covers five things, in order:
1. The verl-side `BaseTool` implementations (`image_zoom_in_tool`, `display_answers`).
2. A small custom agent loop that hard-stops the rollout the instant `display_answers` is called — so the model never sees an OOD empty tool response and verl never generates a phantom extra turn.
3. JSONL → parquet conversion that wires `image_path` into per-sample `tools_kwargs` so the tools work out of the box.
4. A 1/0 string-match reward function that pulls the answer out of the trajectory's `display_answers` tool call.
5. A trainer YAML + launch script that ties everything together.

**Path convention.** Training runs on the cluster, not on this laptop. Throughout this plan, `$PROJECT` = `/users/msayfiddinov/capscratch/verl-apertus` and `$VERL` = `/users/msayfiddinov/capscratch/verl`. Local paths (`/home/muhammadali/work/...`) appear only in the *Critical files referenced* link list, which is for review on this machine.

verl already ships [verl/tools/image_zoom_in_tool.py](../../../work/verl/verl/tools/image_zoom_in_tool.py), but it's wrong for us: it stores a fetched PIL image in instance state, returns `ToolResponse(image=[cropped])`, and includes a `label` parameter the dataset doesn't advertise. Apertus needs the tool response to be the *IBQ-encoded token string* of the cropped region (per the SFT format documented in [plans/cof-sft-parse.md:209](../../../work/verl-apertus/plans/cof-sft-parse.md#L209)). And per [verl/CLAUDE.md](../../../work/verl/CLAUDE.md), we don't add new files to the upstream verl tree — everything lives in `verl-apertus/`.

---

## Files to create

All under [/home/muhammadali/work/verl-apertus/](../../../work/verl-apertus/):

| File | Purpose |
|---|---|
| `tools/image_zoom_in_emu_tool.py` | `ImageZoomInEmuTool(BaseTool)` |
| `tools/display_answers_tool.py` | `DisplayAnswersTool(BaseTool)` |
| `agent_loops/cof_tool_agent_loop.py` | Custom agent loop that short-circuits on `display_answers` |
| `configs/cof_rl_tool_config.yaml` | verl tool registry config |
| `datasets/prepare_cof_rl_to_parquet.py` | metadata.jsonl + raw.jsonl → train/val parquet |
| `rewards/cof_rl_reward.py` | `compute_score(...)` returning 1.0 / 0.0 |
| `configs/cof_rl_grpo.yaml` | top-level trainer config (Hydra, extends `ppo_trainer`) |
| `slurm/run_cof_rl_grpo.sh` | launch script setting `PYTHONPATH` + `python3 -m verl.trainer.main_ppo` |

No edits to existing files are required (parse script untouched, upstream verl untouched).

---

## 1. Tool: `ImageZoomInEmuTool`

**Class:** `tools.image_zoom_in_emu_tool.ImageZoomInEmuTool`, subclasses [`verl.tools.base_tool.BaseTool`](../../../work/verl/verl/tools/base_tool.py).

**Schema** — must match the source `<tools>` block in raw.jsonl (the SFT-trained format uses `bbox_2d`):

```yaml
type: function
function:
  name: image_zoom_in_tool
  description: "Zoom in on a specific region of an image by cropping it on a bounding box."
  parameters:
    type: object
    properties:
      bbox_2d:
        type: array
        items: {type: number}
        description: "[x1, y1, x2, y2]"
    required: [bbox_2d]
```

Only `bbox_2d` — no `label`, no `ratio` (per user spec: "only bbox parameters").

**Image-path resolution.** Standard verl pattern: per-sample `extra_info.tools_kwargs.image_zoom_in_tool.create_kwargs.image_path`. The dataset prep step (§2) populates this. The model never passes `image_path` in the call — just `bbox_2d`.

**`__init__(config, tool_schema)`**
- Read from `config`:
  - `vq_model_path` (path to `Emu3.5-VisionTokenizer`)
  - `vq_device` (default `"cuda:0"`)
  - `target_area` (default `262144` = 512×512, matches [`encode_image` default](../../../work/verl-apertus/inference/vision.py#L90))
  - `min_dimension` (default `28`, mirrors [verl's existing zoom-in tool](../../../work/verl/verl/tools/image_zoom_in_tool.py#L129))
- Init `self._instance_dict = {}`.
- Init `self._vq_model = None`, `self._vq_lock = threading.Lock()` — load IBQ lazily inside `execute` behind a lock so the first call pays the cost and subsequent calls reuse the model. (Eager load at import time would fire on every Ray worker before any rollout starts and may load it multiple times across processes; lazy + lock contains the cost.)

**`async create(instance_id, **kwargs)`**
- Resolve `image_path = kwargs.get("create_kwargs", {}).get("image_path")` (verl forwards `create_kwargs` through to `tool.create`).
- If missing or unreadable, store `{"image": None, "error": "<msg>"}` and return — don't raise; verl wraps tool exceptions into a generic message and we want our specific error visible in the rollout transcript.
- Else `Image.open(image_path).convert("RGB")` and store on `self._instance_dict[instance_id]`.
- Return `(instance_id, ToolResponse())`.

**`async execute(instance_id, parameters, **kwargs)` — all return paths use reward `0.0`. No partial/negative rewards.**

1. If the instance has `error`, return `(ToolResponse(text=f"Error: {error}"), 0.0, {"success": False})`.
2. Pull `bbox_2d` from `parameters`. Validate: list/tuple of 4 numbers — else `(ToolResponse(text="Error: bbox_2d must be a list of 4 numbers."), 0.0, {"success": False})`.
3. **Bbox sanitization** (mirror the proven logic from [`_maybe_resize_bbox`](../../../work/verl/verl/tools/image_zoom_in_tool.py#L205) and the simpler clamp in [`crop_zoom_in`](../../../work/verl-apertus/tools/crop_zoom_tool.py#L6)):
   - cast to float, clamp `[x1, y1, x2, y2]` to `[0, W] × [0, H]`,
   - require `x1 < x2` and `y1 < y2`,
   - reject zero-area / pathological aspect (`max/min > 100`),
   - if either side `< min_dimension`, expand from the center while staying inside the image; if still too small, return `(ToolResponse(text=<descriptive error>), 0.0, {"success": False})`.
4. Lazy-load IBQ if needed (`load_vq_model(self.vq_model_path, device=self.vq_device)`).
5. `cropped = image.crop(sanitized_bbox)`.
6. `token_str = encode_image(cropped, self._vq_model, target_area=self.target_area)` — uses [`encode_image`](../../../work/verl-apertus/inference/vision.py#L89) which already calls `format_image_tokens` and returns `<|img_start|>...<|img_end|>`.
7. Return `(ToolResponse(text=token_str), 0.0, {"success": True, "bbox": sanitized_bbox})`.
   - **Critical**: return as `text=`, not `image=[...]`. The Apertus chat template renders the `tool`-role message body verbatim, so the IBQ token sentinels arrive at the model exactly as in the SFT data. Returning `image=[...]` would dispatch through the HF processor's image branch — wrong for Apertus.

**`async release(instance_id)`** — pop the entry (closes the PIL image).

---

## 2. Tool: `DisplayAnswersTool`

**Class:** `tools.display_answers_tool.DisplayAnswersTool`, subclasses `BaseTool`.

**Schema** — matches `DISPLAY_ANSWERS_TOOL` in [`prepare_cof_rl_parse.py:54`](../../../work/verl-apertus/datasets/prepare_cof_rl_parse.py#L54):

```yaml
type: function
function:
  name: display_answers
  description: "Display the final answer to the user."
  parameters:
    type: object
    properties:
      answer: {type: string, description: "The final answer (a single phrase, word, or letter)."}
    required: [answer]
```

**Behavior**
- `create`: returns a fresh `instance_id`, empty `ToolResponse()`. No state.
- `execute(instance_id, parameters)`:
  - `answer = str(parameters.get("answer", ""))`.
  - Return **immediately** (no I/O, no sleeps, no network) `(ToolResponse(text=""), 0.0, {"success": True, "answer": answer, "is_terminal": True})`.
- `release`: no-op.

The tool itself does nothing functional. The reward `0.0` is fixed; correctness is judged later by the reward function (§5) reading `solution_str`. The `is_terminal: True` metric is the explicit signal the custom agent loop (§3) consumes to stop the rollout.

**Why we cannot rely solely on "the model will emit EOS after an empty tool response."** Apertus's SFT corpus puts `display_answers` as the *last* assistant action — there is no training example where the model sees a `tool`-role message *after* its `display_answers` call. Feeding it one (even empty) is OOD; under temperature > 0, the model could sample more tokens, possibly another `<|tools_prefix|>...<|tools_suffix|>` block (a second `display_answers`, a fresh `image_zoom_in_tool`, or junk). That would (a) corrupt the trajectory the reward function sees, (b) waste compute, and (c) under truncation make the *reward extraction* regex pick up the wrong call. So we don't trust model behavior here — the agent loop hard-stops.

---

## 3. Custom agent loop — `agent_loops/cof_tool_agent_loop.py`

**Goal.** Make the rollout terminate the moment any executed tool returns `metrics["is_terminal"] = True` — i.e. the moment `display_answers` is called. No follow-up generation, no empty tool message exposed to the model.

**Why a new file (and why not edit upstream).** verl ships a default agent loop registered as `tool_agent`. Modifying it would conflict with [verl/CLAUDE.md](../../../work/verl/CLAUDE.md)'s "no agent-only PRs" rule and would also affect every other tool-using project on the cluster. Per the user's instruction, this plan does not look at the upstream loop's implementation; it just registers a new loop name beside it.

**Approach (no upstream edits).** verl's agent-loop registry is exposed via `from verl.experimental.agent_loop.agent_loop import register, AgentLoopBase`. We register a new loop name `cof_tool_agent` and have the dataset reference it (verl's `RLHFDataset` reads `agent_name` from the parquet row and routes to the matching registered loop).

The loop's contract is intentionally tiny — it implements the same surface as the default tool loop (a coroutine `run(sampling_params, **kwargs) -> AgentLoopOutput`), but with one extra invariant:

> After executing any batch of tool calls, if any returned `metrics.get("is_terminal") is True`, append the tool messages, do NOT call `generate` again, and emit `AgentLoopOutput` immediately.

Implementation strategy: import the upstream class as a black-box base and override only the post-tool-execution hook. Concretely the file does this — written without inspecting the loop's internals (we rely only on the public registration decorator and `AgentLoopBase` being importable):

```python
# agent_loops/cof_tool_agent_loop.py
from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop  # default loop class

@register("cof_tool_agent")
class CofToolAgentLoop(ToolAgentLoop):
    """Identical to ToolAgentLoop except: if a tool returns is_terminal=True,
    the rollout stops immediately after that tool's response is appended."""

    async def _handle_processing_tools_state(self, agent_data):
        # Run the parent's tool-processing pass exactly as-is — this includes
        # appending tool messages, updating prompt_ids/response_mask, and
        # (normally) returning AgentState.GENERATING.
        next_state = await super()._handle_processing_tools_state(agent_data)

        # Inspect the tool_rewards / metrics that the parent stashed during
        # the call. If any tool fired with is_terminal, override the next
        # state and finish the rollout.
        terminal = getattr(agent_data, "_cof_terminal_seen", False)
        if terminal:
            from verl.experimental.agent_loop.tool_agent_loop import AgentState
            return AgentState.TERMINATED
        return next_state

    async def _call_tool(self, tool_call, tools_kwargs, agent_data):
        resp, reward, meta = await super()._call_tool(tool_call, tools_kwargs, agent_data)
        if meta and meta.get("is_terminal"):
            agent_data._cof_terminal_seen = True
        return resp, reward, meta
```

This is the *only* contact this plan has with `tool_agent_loop.py` — we touch it through inheritance, not editing. The two methods we override are stable extension points (their names and signatures are part of the public override surface that the existing geo3k/gsm8k tools already rely on).

**Failure mode if the override surface drifts.** If a future verl bump renames `_handle_processing_tools_state` or changes `_call_tool`'s signature, the loop registration will fail loudly at trainer startup (clear ImportError / AttributeError), not silently. Detection is trivial — pin the verl SHA in `requirements.txt` once we settle on a version.

**Wiring.** The parquet's `agent_name` column is set to `"cof_tool_agent"` for every row (§4). verl routes by that name.

---

## 4. Tool config YAML — `configs/cof_rl_tool_config.yaml`

Consumed by [`initialize_tools_from_config`](../../../work/verl/verl/tools/utils/tool_registry.py#L82). `class_name` strings resolve when `verl-apertus/` is on `PYTHONPATH`.

```yaml
tools:
  - class_name: "tools.image_zoom_in_emu_tool.ImageZoomInEmuTool"
    config:
      type: native
      vq_model_path: /capstor/store/cscs/swissai/infra01/MLLM/tokenizer/Emu3.5-VisionTokenizer
      vq_device: cuda:0
      target_area: 262144
      min_dimension: 28
    tool_schema:
      type: function
      function:
        name: image_zoom_in_tool
        description: "Zoom in on a specific region of an image by cropping it on a bounding box."
        parameters:
          type: object
          properties:
            bbox_2d:
              type: array
              items: {type: number}
              description: "[x1, y1, x2, y2]"
          required: [bbox_2d]

  - class_name: "tools.display_answers_tool.DisplayAnswersTool"
    config: {type: native}
    tool_schema:
      type: function
      function:
        name: display_answers
        description: "Display the final answer to the user."
        parameters:
          type: object
          properties:
            answer: {type: string, description: "The final answer (a single phrase, word, or letter)."}
          required: [answer]
```

---

## 5. Parquet conversion — `datasets/prepare_cof_rl_to_parquet.py`

**Why it's needed.** The existing parse script writes a *fully rendered* Apertus prompt as a string. verl's [`RLHFDataset`](../../../work/verl/verl/utils/dataset/rl_dataset.py#L362-L386) expects the `prompt` column to be a list of message dicts — its agent loop then re-applies the chat template at rollout time with `tools=` injected. So the parquet must contain the **structured messages** (without the developer block — verl will render that from the tool schemas) plus `extra_info.tools_kwargs` for image-path injection.

**Inputs.**
- `datasets/cof_rl/raw.jsonl` — original Qwen-style messages with `<image>` placeholder + `image_paths`.
- `datasets/cof_rl/metadata.jsonl` — produced by the existing parse script. We use this only to extract the IBQ token string per row (so we don't have to re-run GPU-bound IBQ encoding).

**Output.**
- `datasets/cof_rl/train.parquet` (default 95%)
- `datasets/cof_rl/val.parquet` (default 5%, configurable via `--val_ratio`, deterministic seed)

**Per-row schema (parquet columns):**

```python
{
    "data_source": "cof_rl",                      # used by reward registry
    "agent_name": "cof_tool_agent",               # routes to our custom loop (§3)
    "prompt": [                                   # list[dict], NOT a string
        {"role": "system", "content": APERTUS_SYSTEM},
        {"role": "user",   "content": user_text_with_ibq_inline},
    ],
    "ability": row["ability"],
    "reward_model": {"style": "rule", "ground_truth": row["reward_model"]["ground_truth"]},
    "extra_info": {
        "index": qid,
        "split": "train" | "val",
        "need_tools_kwargs": True,
        "tools_kwargs": {
            "image_zoom_in_tool": {
                "create_kwargs": {"image_path": <abs path on cluster from metadata.jsonl>},
            },
            # display_answers needs no kwargs
        },
        "answer": row["reward_model"]["ground_truth"],   # convenience for reward fn
    },
}
```

**Pipeline.**
1. Load `raw.jsonl` and `metadata.jsonl` line-aligned (parse script preserves order; `extra_info.index`/`question_id` matches between them — assert equality on every row, fail loud on drift).
2. For each pair `(raw, meta)`:
   a. Pull `raw_user_text = get_user_text(raw["prompt"])` (helper imported from `datasets.prepare_cof_rl_parse`).
   b. Extract `image_token_str` from `meta["prompt"]` via regex `r"<\|img_start\|>.*?<\|img_end\|>"` (greedy not needed — IBQ blocks don't nest).
   c. Build `user_msg = build_user_message(raw_user_text, image_token_str)` (helper imported).
   d. Assemble the parquet row above.
3. Shuffle deterministically (`numpy.random.default_rng(seed).permutation`), split, write parquet via `pyarrow`.

**Imports to reuse** from `datasets/prepare_cof_rl_parse.py`: `APERTUS_SYSTEM`, `build_user_message`, `get_user_text`. No new GPU work.

**Validation in the script** (`--limit 5` smoke mode):
- `len(raw) == len(meta)`, all `index` matches.
- Every output row's `prompt[1]["content"]` contains exactly one `<|img_start|>...<|img_end|>` span.
- Every `image_path` resolves on disk (`Path(...).exists()`).

---

## 6. Reward function — `rewards/cof_rl_reward.py`

**Plumbing.** verl loads custom reward fns via `reward.custom_reward_function.{path, name}` ([trainer/ppo/reward.py:50-86](../../../work/verl/verl/trainer/ppo/reward.py#L50-L86)). The signature called per sample is `compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float | dict`.

**`solution_str` decoding.** Apertus emits tool calls in the native format `<|tools_prefix|>[{"display_answers": {"answer": "<X>"}}]<|tools_suffix|>` (see [run_vtc_bench_tool_agent.py:415-443](../../../work/verl-apertus/inference/run_vtc_bench_tool_agent.py#L415-L443) — that file documents the exact regex `r'<\|tools_prefix\|>\[(.*?)\]<\|tools_suffix\|>'`). Multiple tool-call blocks may appear across turns; we want the **last `display_answers` call**.

**Extraction logic.**

```python
import json, re
TOOLS_BLOCK = re.compile(r'<\|tools_prefix\|>\[(.*?)\]<\|tools_suffix\|>', re.DOTALL)

def _extract_display_answer(solution_str: str) -> str | None:
    for inner in reversed(TOOLS_BLOCK.findall(solution_str)):
        try:
            calls = json.loads(f"[{inner}]")
        except json.JSONDecodeError:
            continue
        for call in reversed(calls):
            if not isinstance(call, dict):
                continue
            if "display_answers" in call:
                args = call["display_answers"]
                if isinstance(args, dict) and "answer" in args:
                    return str(args["answer"])
    return None
```

**Normalization.** Match the existing baseline's loose VQA-style comparison: lowercase, strip whitespace, strip trailing punctuation. Keep it simple — exact string equality after `.strip().lower().rstrip(".,!?")`.

**`compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float`**
- `pred = _extract_display_answer(solution_str)`
- If `pred is None`: return `0.0` (no display_answers → wrong).
- Return `1.0` if `_normalize(pred) == _normalize(ground_truth)` else `0.0`.

The function is registered for `data_source == "cof_rl"` via the `custom_reward_function.path` config knob, so it short-circuits the data-source dispatch in [`default_compute_score`](../../../work/verl/verl/utils/reward_score/__init__.py#L19) without us having to edit upstream verl.

**Self-tests** (in the same module, runnable as `python rewards/cof_rl_reward.py`):
- happy path with one tool block,
- multiple tool blocks (zoom calls before display_answers),
- malformed JSON,
- no display_answers at all,
- case/punctuation normalization.

---

## 7. Trainer config — `configs/cof_rl_grpo.yaml` + `slurm/run_cof_rl_grpo.sh`

Adapted from [examples/sglang_multiturn/config/geo3k_multiturn_grpo.yaml](../../../work/verl/examples/sglang_multiturn/config/geo3k_multiturn_grpo.yaml) and [run_qwen2.5-3b_geo3k_multiturn.sh](../../../work/verl/examples/sglang_multiturn/geo3k/run_qwen2.5-3b_geo3k_multiturn.sh).

All cluster paths use `$PROJECT = /users/msayfiddinov/capscratch/verl-apertus` and `$VERL = /users/msayfiddinov/capscratch/verl`.

**`configs/cof_rl_grpo.yaml`** (Hydra):

```yaml
hydra:
  searchpath:
    - file:///users/msayfiddinov/capscratch/verl/verl/trainer/config

defaults:
  - ppo_trainer
  - _self_

data:
  train_files: /users/msayfiddinov/capscratch/verl-apertus/datasets/cof_rl/train.parquet
  val_files:   /users/msayfiddinov/capscratch/verl-apertus/datasets/cof_rl/val.parquet
  prompt_key: prompt
  max_prompt_length: 8192          # IBQ tokens are long
  max_response_length: 2048
  train_batch_size: 64
  return_raw_chat: true
  return_multi_modal_inputs: false # we encode images as text tokens, not pixels
  # IMPORTANT: re-applying chat template must produce the exact same string
  # prepare_cof_rl_parse.py rendered, so enable_thinking=True is required.
  apply_chat_template_kwargs:
    enable_thinking: true
  filter_overlong_prompts: true
  truncation: error

actor_rollout_ref:
  hybrid_engine: true
  model:
    path: /capstor/store/cscs/swissai/infra01/MLLM/ablations/apertus-8b-img-SFT-32nodes-gbs512-mbs1-steps8030-img-text-seqlen8192-s2onlytxtloss/HF
    trust_remote_code: true
    enable_gradient_checkpointing: true
  rollout:
    name: sglang
    gpu_memory_utilization: 0.5
    n: 8                            # GRPO group size
    multi_turn:
      enable: true
      format: hermes                # native <|tools_prefix|>...<|tools_suffix|> parser
      max_assistant_turns: 5
      max_parallel_calls: 1
      max_tool_response_length: 8192   # IBQ token strings are large; don't truncate them
      tool_response_truncate_side: middle
      tool_config_path: /users/msayfiddinov/capscratch/verl-apertus/configs/cof_rl_tool_config.yaml

algorithm:
  adv_estimator: grpo
  use_kl_in_reward: false

reward:
  reward_manager:
    name: naive
  custom_reward_function:
    path: /users/msayfiddinov/capscratch/verl-apertus/rewards/cof_rl_reward.py
    name: compute_score

trainer:
  project_name: cof_rl_apertus
  experiment_name: apertus8b_grpo_cof_rl
  total_epochs: 5
  n_gpus_per_node: 4
  nnodes: 1
  save_freq: 100
  test_freq: 50
  logger: ["console", "wandb"]
```

The tool config YAML's `vq_model_path` (§4) also points at the cluster path `/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/Emu3.5-VisionTokenizer` — no change needed there since it was already cluster-resident.

Open knobs to confirm with the cluster: GPU count, `n_gpus_per_node`, `tensor_model_parallel_size`, `train_batch_size`, `ppo_mini_batch_size`. These will need tuning, but the structure above is correct.

**Note on `format: hermes`.** verl's tool parser registry maps generation-format names to extractors. `hermes` is the standard one for `<|tools_prefix|>...<|tools_suffix|>` payloads; if Apertus uses a non-standard variant, we may need a thin custom parser registered under a new name. Verify on the smoke run; listed as a known follow-up rather than blocking.

**`slurm/run_cof_rl_grpo.sh`** sets up `PYTHONPATH` so the tool/loop `class_name`s resolve, the custom agent loop registers itself, the reward fn imports cleanly, and `vision_tokenizer` is importable:

```bash
#!/bin/bash
set -x
ulimit -n 65535

PROJECT=/users/msayfiddinov/capscratch/verl-apertus
VERL=/users/msayfiddinov/capscratch/verl
EMU3_SRC=${EMU3_SRC:-/users/msayfiddinov/capscratch/Emu3.5/src}

export PYTHONPATH="$PROJECT:$VERL:$EMU3_SRC:${PYTHONPATH:-}"
# Force registration of the cof_tool_agent loop before main_ppo reads agent_name
export VERL_AGENT_LOOPS_EXTRA="agent_loops.cof_tool_agent_loop"
cd "$VERL"

python3 -m verl.trainer.main_ppo \
    --config-path="$PROJECT/configs" \
    --config-name=cof_rl_grpo "$@"
```

If verl doesn't honor `VERL_AGENT_LOOPS_EXTRA`, fall back to `python3 -c "import agent_loops.cof_tool_agent_loop" && python3 -m verl.trainer.main_ppo ...` (the import side-effect-registers the loop). To be confirmed during the smoke run.

---

## Critical files referenced (read-only)

- [verl-apertus/datasets/prepare_cof_rl_parse.py](../../../work/verl-apertus/datasets/prepare_cof_rl_parse.py) — record schema, tool-name conventions, helpers we import
- [verl-apertus/inference/vision.py](../../../work/verl-apertus/inference/vision.py) — `encode_image`, `format_image_tokens`, `load_vq_model`
- [verl-apertus/inference/run_vtc_bench_tool_agent.py](../../../work/verl-apertus/inference/run_vtc_bench_tool_agent.py) — native tool-call regex
- [verl-apertus/tools/crop_zoom_tool.py](../../../work/verl-apertus/tools/crop_zoom_tool.py) — bbox-clamp reference
- [verl/verl/tools/base_tool.py](../../../work/verl/verl/tools/base_tool.py) — `BaseTool` interface
- [verl/verl/tools/schemas.py](../../../work/verl/verl/tools/schemas.py) — `ToolResponse`, `OpenAIFunctionToolSchema`
- [verl/verl/tools/image_zoom_in_tool.py](../../../work/verl/verl/tools/image_zoom_in_tool.py) — bbox-resize logic to mirror
- [verl/verl/tools/utils/tool_registry.py](../../../work/verl/verl/tools/utils/tool_registry.py) — how `class_name` is loaded
- [verl/verl/utils/dataset/rl_dataset.py](../../../work/verl/verl/utils/dataset/rl_dataset.py) — parquet schema + tools_kwargs forwarding
- [verl/verl/utils/reward_score/__init__.py](../../../work/verl/verl/utils/reward_score/__init__.py) — reward function signature
- [verl/verl/trainer/ppo/reward.py](../../../work/verl/verl/trainer/ppo/reward.py) — custom reward fn loader
- [verl/examples/sglang_multiturn/config/tool_config/gsm8k_tool_config.yaml](../../../work/verl/examples/sglang_multiturn/config/tool_config/gsm8k_tool_config.yaml) — config-shape reference
- [verl/examples/sglang_multiturn/config/geo3k_multiturn_grpo.yaml](../../../work/verl/examples/sglang_multiturn/config/geo3k_multiturn_grpo.yaml) + [run_qwen2.5-3b_geo3k_multiturn.sh](../../../work/verl/examples/sglang_multiturn/geo3k/run_qwen2.5-3b_geo3k_multiturn.sh) — closest analog for our GRPO config

---

## Verification

All commands run on the cluster.

A. **Static / pure-python tests (login node, no GPU):**

```bash
cd /users/msayfiddinov/capscratch/verl-apertus
export PYTHONPATH=$(pwd):/users/msayfiddinov/capscratch/verl:$PYTHONPATH

# (a) tool registry loads both classes
python - <<'PY'
from verl.tools.utils.tool_registry import initialize_tools_from_config
tools = initialize_tools_from_config("configs/cof_rl_tool_config.yaml")
assert {t.name for t in tools} == {"image_zoom_in_tool", "display_answers"}
PY

# (b) custom agent loop registers under name "cof_tool_agent"
python - <<'PY'
import agent_loops.cof_tool_agent_loop  # registers via decorator
from verl.experimental.agent_loop.agent_loop import _AGENT_LOOP_REGISTRY  # or whatever public accessor exists
# fall-back: just confirm the import succeeds without error
PY

# (c) display_answers returns instantly with empty text + is_terminal=True;
#     image_zoom_in_tool with monkeypatched encode_image returns text-only ToolResponse;
#     bbox edge cases (out-of-bounds, zero-area, tiny) all return reward 0.0 and success=False.
python -m pytest tools/  # if tests are added; otherwise inline asserts in a script

# (d) reward fn unit tests
python rewards/cof_rl_reward.py
```

B. **Cluster GPU smoke — end-to-end on 5 rows:**

```bash
cd /users/msayfiddinov/capscratch/verl-apertus
export PYTHONPATH=$(pwd):/users/msayfiddinov/capscratch/verl:/users/msayfiddinov/capscratch/Emu3.5/src:$PYTHONPATH

# parquet conversion (no GPU)
python datasets/prepare_cof_rl_to_parquet.py --limit 5

# real IBQ-encoding round-trip on one parquet row
python - <<'PY'
import asyncio, pandas as pd
from verl.tools.utils.tool_registry import initialize_tools_from_config

row = pd.read_parquet("datasets/cof_rl/train.parquet").iloc[0]
ip = row["extra_info"]["tools_kwargs"]["image_zoom_in_tool"]["create_kwargs"]["image_path"]

zoom = next(t for t in initialize_tools_from_config("configs/cof_rl_tool_config.yaml") if t.name == "image_zoom_in_tool")
iid, _ = asyncio.run(zoom.create(create_kwargs={"image_path": ip}))
resp, reward, meta = asyncio.run(zoom.execute(iid, {"bbox_2d": [10, 10, 200, 200]}))
assert reward == 0.0
assert resp.text and resp.text.startswith("<|img_start|>") and resp.text.endswith("<|img_end|>")
PY
```

C. **Cluster training launch:**

```bash
sbatch /users/msayfiddinov/capscratch/verl-apertus/slurm/run_cof_rl_grpo.sh
```

Watch first 10 steps for:
- tools loading without error,
- `display_answers` returning under 10 ms,
- **rollouts terminate at the assistant turn that emits `display_answers` (not one turn later)** — confirms the custom agent loop's short-circuit is working,
- a non-zero fraction of trajectories receiving reward 1.0 (proves the reward fn is finding `display_answers` calls).

---

## Out of scope

- **Reward shaping / step-level reward.** The current scope is binary final-answer reward. Tool-level partial rewards (returning small negative reward on bbox failures) are already emitted by `ImageZoomInTool` for trace observability but are not aggregated into the trainer reward — verl's `tool_rewards` collection is separate.
- **Custom tool parser** for non-standard Apertus tool-call format. If `format: hermes` doesn't extract calls correctly during the smoke run, add a small parser as a follow-up.
- **SFT warm-start.** Assumed already done; this plan covers RL only.
