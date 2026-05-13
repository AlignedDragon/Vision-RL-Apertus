"""Parse CoF-SFT-Data raw.jsonl into Apertus-formatted SFT examples.

Each raw row encodes an N-image trajectory (N >= 1): one user image plus
(N-1) tool-response images produced by image_zoom_in_tool calls. The
source `messages` array is always `2N + 1` long: system, user, then
(assistant=tool_call, user=tool_response) repeated (N-1) times, then a
final assistant=answer turn.

Per row, render an Apertus conversation:
  - system:    "You are a helpful assistant."
  - developer: source tool (image_zoom_in_tool) + display_answers, rendered via tools=
  - user:      original question with first <image> replaced by IBQ tokens for image_paths[0],
               Qwen "Think in the mind first..." trailer stripped, Apertus instruction appended
  - for k in 1..N-1:
      - assistant: thoughts + tool_calls (from <think>...</think> + <tool_call>...</tool_call>)
      - tool:      IBQ tokens for image_paths[k] (no surrounding text)
  - assistant: thoughts + display_answers tool call (final <answer>...</answer> wrapped)

Trajectories with N > 3 are skipped to keep training sequences bounded.

Output records:
    {
      "text": str,                 # full rendered Apertus conversation
      "image_paths": list[str]     # absolute paths, length N
    }

Usage (interactive on a GPU node):
    python data_prep/prepare_cof_sft_parse.py --limit 5

Usage (SLURM): clone slurm/prepare_cof_rl.slurm and swap the script + dataset dir.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Re-export the constants/helpers we share with the RL parse. Heavy deps
# (torch, PIL, transformers, vision_tokenizer) are imported inside main()
# so the pure-python helpers below can be unit-tested without them.
from data_prep.prepare_cof_rl_parse import (
    APERTUS_SYSTEM,
    QWEN_TRAILER_SENTINEL,
    DISPLAY_ANSWERS_TOOL,
    extract_tool_def,
    load_config,
)
from inference.vision import smart_resize

# SFT keeps multi-word ground-truths, so the instruction must accept any answer
# string (RL drops multi-word answers and instructs "single word"). The single
# element of the `answers` array carries the original answer verbatim.
APERTUS_INSTRUCTION = (
    "Call the display_answers tool exactly once at the end of your response, "
    "passing your final answer as the single element of the `answers` argument."
)

THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


def build_user_text(raw_user_text: str, image_token_str: str) -> str:
    head = raw_user_text.split(QWEN_TRAILER_SENTINEL, 1)[0].rstrip()
    head = head.replace("<image>", image_token_str, 1)
    return f"{head}\n\n{APERTUS_INSTRUCTION}"


def parse_intermediate_assistant(text: str) -> tuple[str, str, dict]:
    """Extract <think> + <tool_call>; return (thoughts, name, args_dict)."""
    think_m = THINK_RE.search(text)
    call_m = TOOL_CALL_RE.search(text)
    if not think_m or not call_m:
        raise ValueError("intermediate assistant missing <think> or <tool_call>")
    call_obj = json.loads(call_m.group(1).strip())
    return think_m.group(1).strip(), call_obj["name"], call_obj["arguments"]


def parse_final_assistant(text: str) -> tuple[str, dict]:
    """Extract <think> + <answer>; convert answer into a display_answers tool call."""
    think_m = THINK_RE.search(text)
    ans_m = ANSWER_RE.search(text)
    if not think_m or not ans_m:
        raise ValueError("final assistant missing <think> or <answer>")
    args_str = json.dumps({"answers": [ans_m.group(1).strip()]}, ensure_ascii=False)
    return think_m.group(1).strip(), {"name": "display_answers", "arguments": args_str}


def _scale_bbox(bbox, src_size, dst_size):
    """Scale [x1,y1,x2,y2] from src (w,h) to dst (w,h)."""
    sw, sh = src_size
    dw, dh = dst_size
    sx, sy = dw / sw, dh / sh
    x1, y1, x2, y2 = bbox
    return [round(x1 * sx), round(y1 * sy), round(x2 * sx), round(y2 * sy)]


_BBOX_LITERAL_RE = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]")


def _rewrite_bboxes_in_text(text, src_size, dst_size):
    """Rewrite [x1,y1,x2,y2] integer tuples in `text` from src to dst pixel space.

    Skips tuples that don't fit src bounds so non-bbox 4-int tuples pass through.
    """
    sw, sh = src_size
    def _sub(m):
        x1, y1, x2, y2 = (int(g) for g in m.groups())
        if not (x1 < x2 and y1 < y2 and x2 <= sw and y2 <= sh):
            return m.group(0)
        scaled = _scale_bbox([x1, y1, x2, y2], src_size, dst_size)
        return f"[{scaled[0]}, {scaled[1]}, {scaled[2]}, {scaled[3]}]"
    return _BBOX_LITERAL_RE.sub(_sub, text)


def build_messages(
    src_msgs: list,
    image_token_strs: list[str],
    main_orig_size: tuple[int, int],
    main_resized_size: tuple[int, int],
    intermediate_calls: list[tuple[str, str, dict]],
) -> list:
    """Render an Apertus conversation for an N-image trajectory (N >= 1).

    Source layout (always 2N+1 messages):
        system, user,
        (assistant=tool_call, user=tool_response) * (N-1),
        assistant=answer.
    `image_token_strs[k]` are the IBQ tokens for `image_paths[k]`.
    `main_orig_size`/`main_resized_size` are (w, h) of image 0 before/after
    smart_resize; used to rewrite bbox literals in `<think>` text.
    `intermediate_calls[k-1]` is the (thoughts, tool_name, args) tuple for the
    assistant turn that produced image k (k>=1); args["bbox_2d"] is already in
    resized-main pixel coords.
    """
    n = len(image_token_strs)
    if n < 1:
        raise ValueError("need at least one image")
    if len(src_msgs) != 2 * n + 1:
        raise ValueError(
            f"expected {2 * n + 1} source messages for {n} images, got {len(src_msgs)}"
        )
    if len(intermediate_calls) != n - 1:
        raise ValueError(
            f"expected {n - 1} intermediate_calls for {n} images, got {len(intermediate_calls)}"
        )

    raw_user = src_msgs[1]["content"]
    out = [
        {"role": "system", "content": APERTUS_SYSTEM},
        {"role": "user", "content": build_user_text(raw_user, image_token_strs[0])},
    ]

    for k in range(1, n):
        th, name, args = intermediate_calls[k - 1]
        th = _rewrite_bboxes_in_text(th, main_orig_size, main_resized_size)
        args_str = json.dumps(args, ensure_ascii=False, separators=(", ", ": "))
        out.append({"role": "assistant", "content": {"blocks": [
            {"type": "thoughts", "text": th},
            {"type": "tool_calls", "calls": [{"name": name, "arguments": args_str}]},
        ]}})
        out.append({"role": "tool", "content": image_token_strs[k]})

    th_final, call_final = parse_final_assistant(src_msgs[2 * n]["content"])
    th_final = _rewrite_bboxes_in_text(th_final, main_orig_size, main_resized_size)
    out.append({"role": "assistant", "content": {"blocks": [
        {"type": "thoughts", "text": th_final},
        {"type": "tool_calls", "calls": [call_final]},
    ]}})
    return out


def main():
    parser = argparse.ArgumentParser(description="Render CoF-SFT trajectories in Apertus format")
    parser.add_argument("--input", default=None, help="Default: data_prep/cof_sft/raw.jsonl")
    parser.add_argument("--output", default=None, help="Default: data_prep/cof_sft/metadata.jsonl")
    parser.add_argument("--config", default="configs/apertus.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N rows (debug)")
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_dir = PROJECT_ROOT / "data_prep" / "cof_sft"
    input_path = Path(args.input) if args.input else dataset_dir / "raw.jsonl"
    output_path = Path(args.output) if args.output else dataset_dir / "metadata.jsonl"

    config = load_config(args.config)

    rows = []
    with open(input_path) as f:
        for line in f:
            rows.append(json.loads(line))
        if args.limit:
            rows = rows[: args.limit]
    print(f"Loaded {len(rows)} rows from {input_path}")

    src_system = next(m["content"] for m in rows[0]["messages"] if m["role"] == "system")
    tool_def = extract_tool_def(src_system)
    print(f"Extracted tool: {tool_def['name']}")
    for i in range(1, min(50, len(rows))):
        other_system = next(m["content"] for m in rows[i]["messages"] if m["role"] == "system")
        if extract_tool_def(other_system) != tool_def:
            raise RuntimeError(f"Tool definition drift at row {i}")
    print(f"Tool definition consistent across first {min(50, len(rows))} rows")

    from PIL import Image
    from transformers import AutoTokenizer
    from inference.vision import encode_image, load_vq_model

    print(f"Loading Apertus tokenizer from {config['model']['checkpoint']} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["checkpoint"], trust_remote_code=True
    )

    print(f"Loading IBQ vision tokenizer from {config['model']['vq_model']} ...")
    vq_model = load_vq_model(config["model"]["vq_model"], device="cuda:0")
    print("Models loaded")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    skipped = 0
    text_lens: list[int] = []
    start = time.time()

    src_dir = dataset_dir / "images_original"
    out_img_dir = dataset_dir / "images"

    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, row in enumerate(rows):
            paths = row.get("image_paths") or []
            if not (1 <= len(paths) <= 3):
                print(f"  SKIP row {i}: image_paths count {len(paths)} outside [1, 3]")
                skipped += 1
                continue

            src_paths = [src_dir / p for p in paths]
            missing = [p for p in src_paths if not p.exists()]
            if missing:
                print(f"  SKIP row {i}: missing image(s) {missing}")
                skipped += 1
                continue

            try:
                src_msgs = row["messages"]
                img_tokens: list[str] = []
                img_paths: list[Path] = []
                intermediate_calls: list[tuple[str, str, dict]] = []
                main_orig_size: tuple[int, int] | None = None
                main_resized_size: tuple[int, int] | None = None
                for k, (p_src, p_rel) in enumerate(zip(src_paths, paths)):
                    img = Image.open(p_src).convert("RGB")
                    if k == 0:
                        main_orig_size = img.size
                        resized = smart_resize(img)
                        main_resized_size = resized.size
                    else:
                        # Size the crop to its bbox region in resized-main pixel
                        # space so the encoded token grid matches the bbox area.
                        th, name, src_args = parse_intermediate_assistant(src_msgs[2 * k]["content"])
                        if "bbox_2d" not in src_args:
                            raise ValueError(f"assistant {k} has no bbox_2d")
                        bbox_res = _scale_bbox(
                            src_args["bbox_2d"], main_orig_size, main_resized_size
                        )
                        target_w = bbox_res[2] - bbox_res[0]
                        target_h = bbox_res[3] - bbox_res[1]
                        if target_w < 16 or target_h < 16:
                            raise ValueError(f"target width or height less than 16: [{target_w}, {target_h}]")
                        main_w, main_h = main_resized_size
                        if target_w * target_h > 0.8 * main_w * main_h:
                            raise ValueError(f"bbox covers >0.8 of image: {target_w}x{target_h} vs {main_w}x{main_h}")
                        scaled_args = {**src_args, "bbox_2d": bbox_res}
                        intermediate_calls.append((th, name, scaled_args))
                        scaled = img.resize((target_w, target_h), Image.BICUBIC)
                        resized = smart_resize(scaled)

                    out_p = out_img_dir / p_rel
                    out_p.parent.mkdir(parents=True, exist_ok=True)
                    resized.save(out_p)
                    img_paths.append(out_p)

                    img_tokens.append(encode_image(resized, vq_model))
            except Exception as e:
                print(f"  SKIP row {i}: image prep failed: {e}")
                skipped += 1
                continue

            try:
                messages = build_messages(
                    row["messages"], img_tokens, main_orig_size, main_resized_size, intermediate_calls
                )
            except Exception as e:
                print(f"  SKIP row {i}: message build failed: {e}")
                skipped += 1
                continue

            text = tokenizer.apply_chat_template(
                messages,
                tools=[tool_def, DISPLAY_ANSWERS_TOOL],
                enable_thinking=True,
                add_generation_prompt=False,
                tokenize=False,
            )
            if not text.rstrip().endswith("<|assistant_end|>"):
                text = text + "<|assistant_end|>"

            out_f.write(json.dumps({
                "text": text,
                "image_paths": [str(p) for p in img_paths],
            }, ensure_ascii=False) + "\n")
            text_lens.append(len(text))

            if (i + 1) % 50 == 0 or i == len(rows) - 1:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                print(f"  [{i+1}/{len(rows)}] {rate:.2f} rows/s  skipped={skipped}")

    written = len(rows) - skipped
    print(f"\nWrote {written} records to {output_path}")
    if skipped:
        print(f"Skipped {skipped} rows")

    if text_lens:
        text_lens.sort()
        n = len(text_lens)
        print(f"\ntext char-length stats: min={text_lens[0]} "
              f"p50={text_lens[n // 2]} p95={text_lens[int(n * 0.95)]} "
              f"max={text_lens[-1]}")

    # Deterministic shuffle + split → train/val parquet (consumed by
    # CoFSFTDataset; metadata.jsonl above stays for human inspection).
    meta_rows: list[dict] = []
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            meta_rows.append(json.loads(line))

    if meta_rows:
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(meta_rows))
        n_val = max(1, int(round(len(meta_rows) * args.val_ratio))) if len(meta_rows) > 1 else 0
        val_set = set(perm[:n_val].tolist())
        train_recs = [m for i, m in enumerate(meta_rows) if i not in val_set]
        val_recs = [m for i, m in enumerate(meta_rows) if i in val_set]

        train_out = output_path.parent / "train.parquet"
        val_out = output_path.parent / "val.parquet"
        pq.write_table(pa.Table.from_pylist(train_recs), train_out)
        pq.write_table(pa.Table.from_pylist(val_recs), val_out)
        print(f"\nWrote {len(train_recs)} rows to {train_out}")
        print(f"Wrote {len(val_recs)} rows to {val_out}")
        print(f"\nReady. Originals are preserved under {src_dir}; processed "
              f"artifacts written to {out_img_dir}.")


if __name__ == "__main__":
    main()
