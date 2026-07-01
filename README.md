# verl-apertus

Tool-augmented visual reasoning for **Apertus 8B**, trained with [verl](https://github.com/volcengine/verl) (SFT + GRPO RL).
Two pipelines live here:

- **VCoT (Visual-CoT)** — a *draw-bbox* grounding tool: the model localizes the referred region, then answers. Evaluated on **RefCOCO+**. Trained first; its checkpoint initializes CoF below.
- **CoF (Chain-of-Focus)** — an *image zoom-in* tool that lets the model crop and re-examine a region at higher resolution before answering. Mirrors DeepEyes, adapted from Qwen-VL to Apertus. **Initialized from the VCoT grounding checkpoint** so it zooms the right region. Evaluated on **V\*Bench**.

## How Apertus "sees"

Apertus has **no continuous vision encoder**. Images are turned into discrete **IBQ tokens** by the Emu3.5 `VisionTokenizer` (131k-entry codebook) and spliced into the text prompt as a token string (`<|img_start|> … <|img_end|>`). See `inference/vision.py`.

The consequence: **verl treats Apertus as a text-only model.** A tool call doesn't return a PIL image — it returns a *new IBQ token string*. The zoom tool crops the original full-resolution image and re-encodes the crop; the bbox tool encodes the localized region. `smart_resize` clamps images to `[16, 2048]` patches, but CoF crops are capped at a **256-token budget** at the call sites (the `vision.py` default stays 2048 for other paths).

## Tools

| Tool | Pipeline | Returns |
|------|----------|---------|
| `image_zoom_in_emu_tool` | CoF | IBQ tokens of a cropped/zoomed region (`bbox_2d`) |
| `image_draw_bbox_tool`   | VCoT | acknowledges a localization bbox (`bbox_2d`) |
| `display_answers_tool`   | both | the answer-emission protocol (`answers: [...]`) |

The model emits Apertus-native calls: `<|tools_prefix|>[{"image_zoom_in_tool": {"bbox_2d": [...]}}]<|tools_suffix|>`.

## Repository layout

```
configs/        apertus*.yaml (eval + agent loop), cof_*.yaml, vcot_*.yaml (SFT/RL/tool configs)
data_prep/      download + parse into verl parquet schema (cof, vcot, vstar, refcocop)
tools/          the three verl tools above
rewards/        format-shaped GRPO rewards + eval rewards (cof_rl, vcot_rl, refcocop)
inference/      vision.py (IBQ encoder) + standalone HF eval harnesses
evaluation/     compute_accuracy.py
slurm/          one script per stage (prepare_* / *_sft / *_rl / eval_* / merge_*)
plans/          design notes; viz/ plots; cluster.md cluster setup
```

## Reward shaping

Both RL rewards are format-shaped (max 1.0), so the policy is credited for *using the tool protocol correctly*, not just for the final answer:

- **CoF** (`rewards/cof_rl_reward.py`): `+0.1` valid zoom call · `+0.1` valid `display_answers` · `+0.9` answer match.
- **VCoT** (`rewards/vcot_rl_reward.py`): `+0.1` valid bbox call · `+0.1` valid `display_answers` · `+0.4` answer match · `+0.4` IoU with gold box.

## Workflow

Everything runs on the CSCS GH200 cluster via SLURM (`--account infra01 --environment=verl_env`). Each stage has a script in `slurm/`. Typical CoF run:

```bash
# 1. data
sbatch slurm/prepare_cof_sft.slurm        # CoF-SFT-Data  -> parquet
sbatch slurm/prepare_cof_rl.slurm         # cof_rl (DeepEyes-derived) -> parquet

# 2. train  (CoF SFT initializes from the merged VCoT RL grounding checkpoint)
sbatch slurm/cof_sft.slurm                # SFT (>=3 epochs at the 256 budget)
sbatch slurm/cof_rl.slurm                 # GRPO (verl + sglang rollouts)
sbatch slurm/merge_cof_rl_checkpoint.slurm  # FSDP shards -> HF checkpoint

# 3. eval on V*Bench
sbatch slurm/prepare_vstar_eval.slurm
sbatch slurm/eval_vstar_verl.slurm        # MODEL=base|auto-sft|auto-rl   (sglang, authoritative)
```

The VCoT pipeline is symmetric: `prepare_vcot_{sft,rl}` → `vcot_sft` → `vcot_rl` → `merge_vcot_rl_checkpoint` → `prepare_refcocop_eval` + `eval_refcocop_verl`.

Model paths (base SFT checkpoint, Emu3.5 tokenizer, Emu3.5 source) are set in `configs/apertus.yaml`.

> **Always evaluate under sglang, not HF.** HF greedy/sampling decoding degenerates on these checkpoints and badly under-reports tool use; verl's sglang engine is authoritative. When scoring the **base** model, score it from its free-text response — base never calls `display_answers`, so the tool-gated harness records 0% as an artifact, not an inability.

## Results (sglang, greedy)

The zoom-in policy is **initialized from the VCoT grounding checkpoint** (see order below), so it localizes the region it zooms into.

**V\*Bench** — 191 MCQ, %:

| Model | 256-token budget |
|-------|:---:|
| base (direct)         | 33.0 |
| SFT + zoom (grnd-init) | 45.2 |
| RL + zoom (grnd-init)  | 49.3 |

**In-distribution** held-out cof_rl eval (408 Q): base **35.0** → SFT **54.4** → RL **64.9**.

**RefCOCO+** val, Acc@0.5 IoU: base **0.000** → SFT **0.274** → RL **0.309**.

Takeaways: the zoom tool **causally helps** (no-zoom ablations: +3pt at 256, +8pt at 2048). We first train the VCoT grounding-box capability, then initialize the CoF zoom-in policy from that checkpoint. This grounding init is what makes zoom transfer to V\*Bench: with a base-initialized policy the stock GRPO reward left V\* flat and RL even regressed it (35.6 → 32.5), because the model zoomed the wrong region; the grounding-initialized policy lifts both the CoF eval and V\*Bench (33.0 → 45.2 → 49.3) and reverses that regression.

## Status / caveats

- Inside the `verl_env` container, `$SCRATCH = /iopsstor/scratch/cscs/$USER` (not `/capstor/...`) — hardcode absolute paths for HF caches.
- 256-token SFT needs ≥2–3 epochs; 1 epoch degenerates the tool-call JSON.
