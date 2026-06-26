"""Direct (no-tool) single-turn MCQ eval on V*Bench, to measure each model's raw
visual-MCQ ability WITHOUT the CoF zoom tool / display_answers protocol.

Reuses the IBQ image tokens already baked into data_prep/vstar_bench/val.parquet
(strips the trailing display_answers instruction), renders the prompt WITHOUT
tools, generates greedily, and parses the answer letter. Lets us compare:
  direct (no-tool) accuracy   vs   tool-agent accuracy
to see whether the zoom tool actually helps.
"""
import argparse, json, re, sys
from pathlib import Path
import torch
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from data_prep.cof_rl_parse import APERTUS_INSTRUCTION

VIS = re.compile(r"(<\|visual token \d+\|>)+")

def parse_letter(t: str):
    t = VIS.sub("", t)
    for stop in ("<|assistant_end|>", "<|tools_suffix|>", "<|tools_prefix|>"):
        t = t.split(stop)[0]
    m = re.search(r"\(?\b([ABCD])\b\)?", t.strip())
    return m.group(1) if m else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--val", default=str(PROJECT_ROOT / "data_prep/vstar_bench/val.parquet"))
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, device_map=a.device, trust_remote_code=True
    ).eval()

    stop = {tok.eos_token_id}
    tid = tok.convert_tokens_to_ids("<|assistant_end|>")
    if isinstance(tid, int) and tid >= 0:
        stop.add(tid)
    stop = sorted(stop)

    rows = pq.read_table(a.val).to_pylist()
    suffix = "\n\n" + APERTUS_INSTRUCTION
    out, corr, got = [], 0, 0
    for r in rows:
        uc = r["prompt"][1]["content"]
        if uc.endswith(suffix):
            uc = uc[: -len(suffix)]               # drop the display_answers instruction
        msgs = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": uc}]
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)  # NO tools
        ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(a.device)
        with torch.no_grad():
            o = model.generate(ids, max_new_tokens=8, do_sample=False,
                               eos_token_id=stop, pad_token_id=tok.pad_token_id)
        gen = tok.decode(o[0, ids.shape[1]:], skip_special_tokens=False)
        p = parse_letter(gen); gold = r["extra_info"]["answer"]
        got += int(p is not None); corr += int(p == gold)
        out.append({"index": r["extra_info"]["index"], "gold": gold, "pred": p,
                    "raw": VIS.sub("", gen)[:40], "correct": p == gold,
                    "category": r["extra_info"].get("category")})
    print(f"{a.label} DIRECT (no-tool, greedy): {corr}/{len(rows)} = {100*corr/len(rows):.1f}%  gave-letter={got} ({100*got/len(rows):.0f}%)", flush=True)
    op = PROJECT_ROOT / f"results/vstar/{a.label}_direct.jsonl"
    op.parent.mkdir(parents=True, exist_ok=True)
    with open(op, "w") as f:
        for x in out:
            f.write(json.dumps(x) + "\n")
    print("wrote", op, flush=True)

if __name__ == "__main__":
    main()
