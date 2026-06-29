"""Direct (no-tool) single-turn MCQ eval on V*Bench using the sglang offline Engine.

sglang-ONLY counterpart of run_direct_mcq.py (the HF backend is NOT used: HF greedy
degenerates on these checkpoints; sglang greedy is authoritative). Reuses the IBQ image
tokens baked into the vstar parquet, strips the trailing display_answers instruction,
renders the prompt WITHOUT tools, generates greedily on sglang, and parses the answer
letter from the free-text response. This is how the BASE model must be scored (it does
not emit the display_answers/zoom tool protocol).

ENV CAVEAT (cof256 run, 2026-06-29): the offline sglang.Engine spawns worker subprocesses
that bare-`import psutil`/`xgrammar`, which are broken compiled C-exts in the current verl
conda env (see memory verl-env-broken-cexts). Prepend an isolated `.pylibs` (psutil) to
PYTHONPATH per slurm; xgrammar still blocks the Engine, so in that run the direct numbers
were obtained via the verl sglang-SERVER path (eval_vstar_verl.slurm) + response re-scoring
instead. This script works once the env C-exts are fixed.

Run on a GPU node inside the verl_env container:
    python inference/run_direct_mcq_sglang.py --model <hf_ckpt> --label base_256 --val data_prep/vstar_bench/val.parquet
"""
import argparse, json, re, sys
from pathlib import Path
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
    ap.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=8)
    ap.add_argument("--mem-fraction", dest="mem_fraction", type=float, default=0.85)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    import sglang as sgl

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    stop_ids = {tok.eos_token_id}
    tid = tok.convert_tokens_to_ids("<|assistant_end|>")
    if isinstance(tid, int) and tid >= 0:
        stop_ids.add(tid)
    stop_ids = sorted(i for i in stop_ids if isinstance(i, int) and i >= 0)

    rows = pq.read_table(a.val).to_pylist()
    suffix = "\n\n" + APERTUS_INSTRUCTION
    input_ids = []
    for r in rows:
        uc = r["prompt"][1]["content"]
        if uc.endswith(suffix):
            uc = uc[: -len(suffix)]                 # drop the display_answers instruction
        msgs = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": uc}]
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)  # NO tools
        ids = tok(text, add_special_tokens=False).input_ids
        input_ids.append(ids)

    print(f"Loading sglang engine for {a.model} ...", flush=True)
    engine = sgl.Engine(model_path=a.model, trust_remote_code=True, tp_size=1,
                        mem_fraction_static=a.mem_fraction, context_length=8192)
    sampling = {"temperature": 0.0, "max_new_tokens": a.max_new_tokens, "stop_token_ids": stop_ids}
    outs = engine.generate(input_ids=input_ids, sampling_params=sampling)
    engine.shutdown()

    out, corr, got = [], 0, 0
    for r, o in zip(rows, outs):
        gen = o["text"] if isinstance(o, dict) else o
        p = parse_letter(gen)
        gold = r["extra_info"]["answer"]
        got += int(p is not None)
        corr += int(p == gold)
        out.append({"index": r["extra_info"]["index"], "gold": gold, "pred": p,
                    "raw": VIS.sub("", gen)[:60], "correct": p == gold,
                    "category": r["extra_info"].get("category")})
    n = len(rows)
    print(f"{a.label} DIRECT-sglang (no-tool, greedy): {corr}/{n} = {100*corr/n:.1f}%  "
          f"gave-letter={got} ({100*got/n:.0f}%)", flush=True)
    op = PROJECT_ROOT / f"results/vstar/{a.label}_direct_sglang.jsonl"
    op.parent.mkdir(parents=True, exist_ok=True)
    with open(op, "w") as f:
        for x in out:
            f.write(json.dumps(x) + "\n")
    print("wrote", op, flush=True)


if __name__ == "__main__":
    main()
