"""Standalone multi-turn Chain-of-Focus inference for Apertus, using plain
HuggingFace `transformers.generate` instead of verl / sglang rollouts.

Why this exists
---------------
verl's RL rollout (sglang agent loop) takes ~30 min just to start. For
*inference / eval / debugging* of the zoom-in tool we don't need any of that
machinery. This script loads the model once with `AutoModelForCausalLM`, loads
the Emu3.5 VQ tokenizer once, and runs the SAME multi-turn tool loop that
`data_prep/apertus_rl_dataset.py::ApertusToolAgentLoop` runs inside verl --
reproduced here at the token level so behaviour matches the RL checkpoint.

The loop, mirroring verl's ToolAgentLoop exactly:
  1. Render the initial prompt with the Apertus chat template (system +
     developer/tools + user-with-IBQ-tokens), add_generation_prompt=True.
  2. generate() one assistant turn (stops at </s> / <|assistant_end|>).
  3. Parse tool calls from that turn: <|tools_prefix|>[{tool: args}]<|tools_suffix|>.
  4. Execute each call. image_zoom_in_tool -> crop original image, re-encode to
     IBQ token string. display_answers -> "Answers displayed".
  5. Append the tool observation as the raw string  "[" + ",".join(texts) + "]"
     (add_special_tokens=False) -- NO role wrapper, exactly like the verl loop.
  6. Continue generating until max_assistant_turns or a turn with no tool call.

Crucially this re-applies the chat template only ONCE; subsequent turns are raw
token concatenation, which is what the model saw during GRPO.

Usage (on a GPU node inside the verl_env container; see slurm/run_cof_hf.slurm):
    python inference/run_cof_hf.py \
        --model checkpoints/cof_rl_apertus/apertus8b_grpo_cof_rl/global_step_100/actor_hf_merged \
        --val data_prep/cof_rl/val.parquet \
        --tool-config configs/cof_rl_tool_config.yaml \
        --vq-model /capstor/store/cscs/swissai/infra01/MLLM/tokenizer/Emu3.5-VisionTokenizer \
        --num-samples 8
"""

import argparse
import json
import re
import sys
import time
from math import ceil, floor
from pathlib import Path

import torch
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# verl-free image encoding (same module the verl tool uses).
from inference.vision import encode_image, load_vq_model, smart_resize

TOOL_SUFFIX = "<|tools_suffix|>"
# Strict: a well-formed Apertus tool block is <|tools_prefix|>[...]<|tools_suffix|>.
# Same regex as verl's ApertusToolParser — a missing suffix is a model error, not
# something we paper over.
TOOL_CALL_RE = re.compile(r"<\|tools_prefix\|>(.*?)<\|tools_suffix\|>", re.DOTALL)


# --------------------------------------------------------------------------- #
# Tool schemas + tool-call parsing (standalone copies of the verl logic)
# --------------------------------------------------------------------------- #
def _unwrap_function(schema: dict) -> dict:
    if isinstance(schema, dict) and schema.get("type") == "function" and "function" in schema:
        return schema["function"]
    return schema


def load_tool_schemas(path: str) -> list[dict]:
    """Load + unwrap tool schemas from a verl tool-config YAML (Apertus template
    reads tool.name/description directly, not the OpenAI envelope)."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    schemas = [_unwrap_function(t["tool_schema"]) for t in (cfg.get("tools") or [])]
    names = {s.get("name") for s in schemas}
    missing = {"image_zoom_in_tool", "display_answers"} - names
    if missing:
        raise ValueError(f"tool config {path} missing schemas: {sorted(missing)}")
    return schemas


def parse_tool_calls(text: str) -> list[tuple[str, dict]]:
    """Strict Apertus tool-call parse, identical in spirit to verl's
    ApertusToolParser: only <|tools_prefix|>[...]<|tools_suffix|> blocks count. A
    call without its closing suffix is a malformed generation and yields nothing
    (the episode then ends with no answer — surfacing the model error)."""
    calls: list[tuple[str, dict]] = []
    for match in TOOL_CALL_RE.findall(text):
        try:
            decoded = json.loads(match)
        except Exception:
            continue
        for call in decoded if isinstance(decoded, list) else [decoded]:
            if not isinstance(call, dict):
                continue
            if "name" in call and "arguments" in call:
                calls.append((str(call["name"]), call["arguments"]))
            elif len(call) == 1:
                name, args = next(iter(call.items()))
                calls.append((str(name), args))
    return calls


# --------------------------------------------------------------------------- #
# image_zoom_in_tool, standalone (replicates ImageZoomInEmuTool.execute math)
# --------------------------------------------------------------------------- #
def _sanitize_bbox(bbox_2d, orig_w, orig_h, disp_w, disp_h):
    """Map bbox from displayed (smart_resize) space back to original pixels."""
    try:
        b = [float(v) for v in bbox_2d]
    except (TypeError, ValueError):
        return None
    if len(b) != 4:
        return None
    sx, sy = orig_w / disp_w, orig_h / disp_h
    left = max(0.0, b[0] * sx)
    top = max(0.0, b[1] * sy)
    right = min(float(orig_w), b[2] * sx)
    bottom = min(float(orig_h), b[3] * sy)
    if not (left < right and top < bottom):
        return None
    return [floor(left), floor(top), ceil(right), ceil(bottom)]


def zoom_in(image: Image.Image, bbox_2d, vq_model) -> str:
    """Crop bbox (in displayed coords) from the ORIGINAL image, re-encode to an
    IBQ token string. Returns an "Error: ..." string on bad input (matching the
    verl tool, so the model sees the same observation)."""
    if not isinstance(bbox_2d, (list, tuple)):
        return "Error: bbox_2d must be a list of 4 numbers."
    displayed = smart_resize(image)
    san = _sanitize_bbox(bbox_2d, image.width, image.height, displayed.width, displayed.height)
    if san is None:
        return f"Error: bbox {bbox_2d} is invalid (requires x1 < x2 and y1 < y2)."
    try:
        return encode_image(image.crop(tuple(san)), vq_model)
    except Exception as e:
        return f"Error: failed to encode cropped region: {e}"


# --------------------------------------------------------------------------- #
# Multi-turn episode (replicates ApertusToolAgentLoop state machine)
# --------------------------------------------------------------------------- #
def run_episode(
    model,
    tokenizer,
    vq_model,
    messages: list[dict],
    tool_schemas: list[dict],
    image_path: str,
    *,
    stop_ids: list[int],
    gen_kwargs: dict,
    max_assistant_turns: int = 3,
    max_response_length: int = 2048,
    max_new_tokens: int = 512,
    device: str = "cuda:0",
) -> dict:
    """Run one CoF episode, verl-faithful: generate until eos (which includes
    <|tools_suffix|>=72 and <|assistant_end|>=68), parse the turn STRICTLY, run
    the tool, append the model's real response tokens + the raw "[obs]" string,
    and continue. Mirrors ApertusToolAgentLoop's token-level loop."""
    image = Image.open(image_path).convert("RGB")

    # The Apertus chat template emits its own bos (`{{ bos_token }}`), and the
    # tokenizer has add_bos_token=True — so tokenizing the template with special
    # tokens enabled yields a DOUBLE <s><s> BOS (OOD, erratic generations). Render
    # to text, then tokenize with add_special_tokens=False for exactly one BOS.
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tools=tool_schemas,
        enable_thinking=True,
        add_generation_prompt=True,
        tokenize=False,
    )
    input_ids = tokenizer(
        prompt_text, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)
    prompt_len = input_ids.shape[1]

    turns: list[dict] = []
    final_answer = None
    response_token_count = 0

    for _ in range(max_assistant_turns):
        budget = min(max_new_tokens, max_response_length - response_token_count)
        if budget <= 0:
            break
        seq_before = input_ids
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=budget,
                eos_token_id=stop_ids,
                pad_token_id=tokenizer.pad_token_id,
                **gen_kwargs,
            )
        gen_ids = out[0, seq_before.shape[1]:]
        response_token_count += int(gen_ids.shape[0])
        turn_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
        input_ids = out  # append the model's real tokens (verl-faithful)

        calls = parse_tool_calls(turn_text)
        turns.append({"text": turn_text, "tool_calls": [(n, a) for n, a in calls]})

        if not calls:
            # Mirror verl: a turn that ends in <|tools_suffix|> but has no parseable
            # <|tools_prefix|> call (e.g. an <|inner_prefix|> deliberation block) is
            # given an empty "[]" observation and the rollout CONTINUES. Only a turn
            # with no suffix at all terminates the episode.
            if TOOL_SUFFIX in turn_text:
                obs_ids = tokenizer.encode("[]", add_special_tokens=False, return_tensors="pt").to(device)
                if response_token_count + obs_ids.shape[1] >= max_response_length:
                    break
                input_ids = torch.cat([input_ids, obs_ids], dim=1)
                continue
            break  # no tool call and no suffix -> episode ends

        name, args = calls[0]  # verl runs max_parallel_calls=1
        if name == "display_answers":
            ans = args.get("answers")
            final_answer = ans[0] if isinstance(ans, list) and ans else ans
            break  # final action: answer obtained

        if name == "image_zoom_in_tool":
            obs = zoom_in(image, args.get("bbox_2d"), vq_model)
        else:
            obs = f"Error: unknown tool {name}"

        # Append the tool observation as the raw "[obs]" string, no role wrapper.
        obs_ids = tokenizer.encode("[" + obs + "]", add_special_tokens=False, return_tensors="pt").to(device)
        if response_token_count + obs_ids.shape[1] >= max_response_length:
            break
        input_ids = torch.cat([input_ids, obs_ids], dim=1)

    full_text = tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=False)
    return {
        "turns": turns,
        "final_answer": final_answer,
        "num_turns": len(turns),
        "num_zooms": sum(
            1 for t in turns for n, _ in t["tool_calls"] if n == "image_zoom_in_tool"
        ),
        "full_completion": full_text,
    }


# --------------------------------------------------------------------------- #
def _messages_from_row(prompt_field) -> list[dict]:
    """parquet 'prompt' is a list of {role, content} (possibly numpy types)."""
    return [{"role": str(m["role"]), "content": str(m["content"])} for m in prompt_field]


def _normalize(s) -> str:
    return str(s).strip().lower().rstrip(".") if s is not None else ""


def main():
    ap = argparse.ArgumentParser(description="Standalone HF multi-turn CoF inference")
    ap.add_argument("--model", required=True, help="HF checkpoint dir (RL-merged or SFT)")
    ap.add_argument("--val", default=str(PROJECT_ROOT / "data_prep/cof_rl/val.parquet"))
    ap.add_argument("--tool-config", default=str(PROJECT_ROOT / "configs/cof_rl_tool_config.yaml"))
    ap.add_argument(
        "--vq-model",
        default="/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/Emu3.5-VisionTokenizer",
    )
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--max-assistant-turns", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    # Decoding. The model was RL-optimised under sampling at temperature 1.0
    # (rollout default: do_sample=True, temperature=1.0, top_p=1.0, top_k=-1) and
    # emits proper <|tools_suffix|>-terminated calls there. Greedy (temperature 0)
    # is brittle on this checkpoint. Default to the training-time sampling.
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0, help="0 / -1 => disabled")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--debug", action="store_true", help="dump rendered prompt + raw turns for sample 0")
    ap.add_argument("--output", default=str(PROJECT_ROOT / "results/cof_hf/predictions.jsonl"))
    args = ap.parse_args()

    if args.temperature and args.temperature > 0:
        gen_kwargs = {
            "do_sample": True,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }
        if args.top_k and args.top_k > 0:
            gen_kwargs["top_k"] = args.top_k
    else:
        gen_kwargs = {"do_sample": False}

    import pyarrow.parquet as pq
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    print(f"Decoding: {gen_kwargs}  seed={args.seed}", flush=True)
    print(f"Loading tokenizer + model from {args.model} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device, trust_remote_code=True
    )
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)

    print(f"Loading VQ model from {args.vq_model} ...", flush=True)
    t0 = time.time()
    vq_model = load_vq_model(args.vq_model, device=args.device)
    print(f"VQ model loaded in {time.time() - t0:.1f}s", flush=True)

    tool_schemas = load_tool_schemas(args.tool_config)

    # Stop tokens: </s> + <|assistant_end|> (+ generation_config extras if present)
    stop_ids = {tokenizer.eos_token_id}
    for tok in ("<|assistant_end|>",):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid >= 0 and tid != tokenizer.unk_token_id:
            stop_ids.add(tid)
    gen_eos = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(gen_eos, list):
        stop_ids.update(gen_eos)
    elif isinstance(gen_eos, int):
        stop_ids.add(gen_eos)
    stop_ids = sorted(stop_ids)
    print(f"Stop token ids: {stop_ids}", flush=True)

    table = pq.read_table(args.val).to_pylist()
    rows = table[: args.num_samples]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results = []
    n_correct = 0
    wall = time.time()
    for i, row in enumerate(rows):
        messages = _messages_from_row(row["prompt"])
        image_path = row["extra_info"]["tools_kwargs"]["image_zoom_in_tool"]["create_kwargs"]["image_path"]
        gold = row["extra_info"]["answer"]

        if args.debug and i == 0:
            rendered = tokenizer.apply_chat_template(
                messages, tools=tool_schemas, enable_thinking=True,
                add_generation_prompt=True, tokenize=False,
            )
            short = re.sub(r"(<\|visual token \d+\|>)+", "<VISUAL…>", rendered)
            print("=== DEBUG rendered prompt (sample 0, visual tokens collapsed) ===", flush=True)
            print(short[:1200], flush=True)
            bos = tokenizer.bos_token_id
            n_bos_fixed = int((tokenizer(rendered, add_special_tokens=False, return_tensors="pt").input_ids[0] == bos).sum())
            n_bos_bad = int((tokenizer.apply_chat_template(messages, tools=tool_schemas, enable_thinking=True, add_generation_prompt=True, return_tensors="pt")[0] == bos).sum())
            print("  Deliberation enabled:", "Deliberation: enabled" in rendered,
                  "| ends @assistant_start:", rendered.rstrip().endswith("<|assistant_start|>"),
                  f"| BOS count: fixed-path={n_bos_fixed} tokenize=True-path={n_bos_bad}", flush=True)

        t0 = time.time()
        trace = run_episode(
            model, tokenizer, vq_model, messages, tool_schemas, image_path,
            stop_ids=stop_ids,
            gen_kwargs=gen_kwargs,
            max_assistant_turns=args.max_assistant_turns,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )
        if args.debug and i == 0:
            print("=== DEBUG raw turns (sample 0) ===", flush=True)
            for ti, t in enumerate(trace["turns"]):
                tt = re.sub(r"(<\|visual token \d+\|>)+", "<VISUAL…>", t["text"])
                print(f"  turn{ti}: {tt[:600]}", flush=True)
        dt = time.time() - t0
        pred = trace["final_answer"]
        correct = _normalize(pred) == _normalize(gold) and pred is not None
        n_correct += int(correct)

        rec = {
            "index": row["extra_info"]["index"],
            "gold": gold,
            "prediction": pred,
            "correct": correct,
            "num_turns": trace["num_turns"],
            "num_zooms": trace["num_zooms"],
            "seconds": round(dt, 2),
            "turns": trace["turns"],
        }
        results.append(rec)
        print(
            f"[{i+1}/{len(rows)}] gold={gold!r} pred={pred!r} "
            f"correct={correct} turns={trace['num_turns']} zooms={trace['num_zooms']} ({dt:.1f}s)",
            flush=True,
        )

    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(
        f"\n=== Done: {n_correct}/{len(rows)} correct "
        f"({100*n_correct/max(1,len(rows)):.1f}%) in {time.time()-wall:.1f}s ===",
        flush=True,
    )
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
