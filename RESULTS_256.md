# CoF @ 256-token image budget — results tracker

Worktree: `/capstor/scratch/cscs/msayfiddinov/verl-apertus-cof256` (branch `cof-256`).
All evals **sglang only** (greedy). Base scored from free-text response (no display gate).

## Key finding: 256-token SFT needs more epochs (format degeneration)
- At 256-token images, a **1-epoch** SFT degenerates the tool-call JSON format: emits
  `{"answers": {...}}` (dict/recursive) + reserved `<SPECIAL_NNN>` tokens → only ~5% of
  display calls are parseable → reward ~0.005, val acc 2.9%. The old 2048 SFT (1 epoch)
  was clean (100% list format, reward 0.40). Same code/parse/reward — only the image
  budget differs (256-tok images are far OOD vs the base img-SFT distribution).
- **Fix: train SFT 3 epochs.** val/loss 0.52 (1ep) → 0.37 (≥2ep). Format probe on the
  3-epoch ckpt (global_step_321): **97% clean `[...]` lists, 0% dicts, 0 special tokens.**

## SFT model
- `checkpoints/apertus8b-cof-sft/global_step_321/huggingface` (3 epochs, val/loss ~0.37).

## V*Bench results (191 MCQ; "acc" = answer-correct fraction; reward = compute_score mean)
| Model | Input | Tool | acc (answer-correct) | reward mean | zoom rate | notes |
|-------|-------|------|----------------------|-------------|-----------|-------|
| SFT-321 | 256 | zoom | **35.6%** (68/191) | 0.459 | 60% | probe=eval, format clean |
| base | 256 | direct | **33.0%** (=256+zoom re-score) | | 0% | base ignores tools -> direct |
| base | 2048 | direct | **41.4%** (79/191) | | 0% | CEILING; base ignores tools -> direct |
| base | 256 | zoom | **33.0%** (63/191) | n/a (resp-scored) | 0% | base ignores tool, answers in text; gave-letter 100% |
| SFT-321 | 2048 | zoom | **37.7%** (72/191) | 0.467 | 49% | 256-trained model on 2048 input (OOD) |
| SFT-321 | 256 | no-zoom (display-only) | **32.5%** (62/191) | 0.381 | 0% | ablation: zoom causally worth +3.1pt (35.6 vs 32.5) |
| RL-50 | 256 | zoom | **32.5%** (62/191) | 0.458 | 99% | RL cof_rl-val 0.529->0.614 (step0/25/50) but V* FLAT/slight-regress vs SFT |
| RL-50 | 2048 | zoom | **44.0%** (84/191) | 0.542 | 78% | best in matrix; RL helps at high-res input |
| RL-50 | 256 | no-zoom (display-only) | **26.7%** (51/191) | 0.320 | 0% | RL became zoom-dependent; worse without it |
| RL-50 | 2048 | no-zoom (display-only) | **36.1%** (69/191) | 0.403 | 0% | causal: zoom worth +7.9pt at 2048 (44.0 vs 36.1) |
| ... | | | | | | |

## IN-DISTRIBUTION cof_rl eval set (408 Qs, sglang greedy, answer-correct %)
| model | acc | zoom | note |
|-------|-----|------|------|
| base  | **35.0%** (143/408; lenient 38.7%) | 0% | base ignores tools; scored from free-text (exact-norm) |
| SFT   | **46.8%** (191/408) | 29% | reward 0.534 |
| RL-50 | **52.0%** (212/408) | 100% | reward 0.615 |

**On the cof_rl (in-distribution) eval set RL DOES improve: base 35.0 -> SFT 46.8 -> RL 52.0**
(RL +17pt over base, +5.2pt over SFT; cof_rl-val reward 0.529->0.586->0.614 over step 0/25/50).
OPPOSITE of V* (RL flat/regresses) -> stock GRPO learns its training distribution but does
NOT transfer to V* (OOD): drives zoom to 100% for the +0.1 zoom bonus + fits cof_rl answers.
(Tool-gated val reads base~0 on both sets due to the no-tool-protocol artifact; base re-scored
from free-text.)

## Jobs in flight
- RL 2640812 (GRPO from SFT-321, total_training_steps=50, save_freq=25)
- base@256+zoom 2640813 -> dump slurm/val_dumps/base_256zoom (re-score from response)
- SFT@2048+zoom 2640814 -> dump slurm/val_dumps/sft_2048zoom

## FINAL ANALYSIS (V*Bench, 191 MCQ, sglang greedy, answer-correct %)

**Resolution gap:** base@256-direct **33.0%** -> base@2048-direct **41.4%** => cutting the
image budget 2048->256 costs **~8.4 pts** of raw accuracy (the gap CoF zoom must recover).

**Zoom causally helps SFT at 256:** SFT@256 no-zoom **32.5%** -> SFT@256+zoom **35.6%** =
**+3.1 pts** from the zoom tool. So at a 256-token budget, zoom recovers ~3 of the ~8 pts
lost to low resolution (SFT+zoom 35.6% sits between base@256 33.0% and base@2048 41.4%).
SFT no-zoom (32.5%) ~= base@256 (33.0%) -> the reasoning scaffold alone adds ~nothing; the
gain is the zoom RESOLUTION.

**The context-length fix worked (no zoom abandonment):** unlike the old 2048 run (RL drove
zoom to ~0%), here RL zooms **99%** of episodes. Shifting the budget to the response
(prompt 2048 / response 6000) so multi-turn zoom isn't truncated prevented the collapse the
user predicted.

**But RL still did NOT improve V*@256:** RL@256+zoom **32.5%** is flat/slightly below
SFT@256+zoom 35.6% (6 questions, ~within noise). RL improved its *training* metric
(cof_rl-val 0.529->0.614) and zooms constantly, but that doesn't transfer to V*: the stock
reward's +0.1 zoom-format bonus makes RL zoom indiscriminately (reward-hacking) and it
overfits the cof_rl answer distribution. RL@256 no-zoom drops to 26.7% -> RL became
zoom-dependent. Net: stock GRPO on cof_rl is not beneficial for V*@256 (matches the prior
2048 finding, but via over-zoom rather than zoom-abandonment).

**RL helps at high-res input — and it IS the zoom (causal):** RL@2048+zoom **44.0%** is the
best cell (> base@2048 41.4%, > SFT@2048 37.7%). Causal ablation: **RL@2048 no-zoom = 36.1%**
=> the zoom tool is worth **+7.9pt** at 2048 (44.0 vs 36.1), and RL+zoom even beats the
no-tool high-res baseline (base@2048 41.4%). NOTE the WITHIN-run correlation is misleading
(zoomed eps 40.9% < non-zoomed eps 54.8%) — that's difficulty selection bias (the model
zooms on the hard small-target questions); the no-zoom ablation removes the confound and
confirms zoom genuinely helps. (Same direction at 256: RL no-zoom 26.7 -> +zoom 32.5 = +5.8pt;
SFT 32.5 -> 35.6 = +3.1pt. Zoom causally helps in every ablation; it's just that at 256 the
RL *policy overall* still doesn't beat SFT.)

**Headline:** at a 256-token budget, the zoom tool (via SFT) recovers ~3/8 of the
resolution gap; stock RL does not add value on V* (over-zoom + train/eval mismatch), though
it is well-behaved (99% zoom, clean format, no crash, trained cleanly to step 50).

**Caveat — SFT needs >=2-3 epochs at 256:** a 1-epoch SFT degenerates the tool-call JSON
(see top); all numbers above use the 3-epoch SFT (global_step_321).

## GOTCHA: @2048+zoom evals need a length override
The training config sets `max_prompt_length=2048` (fine for 256-tok input). The 2048-tok
input prompts (~2600 tok) exceed it, so `filter_overlong_prompts` drops ALL of them ->
"filter dataset len: 0" -> job fails. For any **@2048+zoom** eval (eval_vstar_verl) pass
`data.max_prompt_length=6000 data.max_response_length=2048` (sum 8048<8192, the proven
2048 split). The @2048 **direct** evals (run_direct_mcq_sglang, sglang.Engine) are unaffected.

## ENV breakage: offline sglang.Engine unusable -> direct evals via verl+re-score
The conda env has corrupted compiled C-extensions: bare `import psutil` fails
(`_psutil_linux`), and the sglang offline-Engine scheduler subprocess also fails on
`xgrammar_bindings`. The verl sglang SERVER path (eval_vstar_verl) imports these fine, so
ALL evals go through eval_vstar_verl (sglang). For "direct" numbers: BASE ignores tools
(0% zoom), so base@N+zoom re-scored from the free-text response == base@N-direct. Pure
no-tool direct for the tool-trained SFT/RL is omitted (offline Engine broken); the
**no-zoom (display-only) ablation** is the meaningful isolate-zoom measure instead.
(isolated `--target` psutil fixed psutil but xgrammar remained; not worth chasing.)

## TODO evals (all via eval_vstar_verl + re-score)
- base@2048 re-score (2641100) -> base@2048 direct ceiling
- sft@256 no-zoom ablation (2641350)
- after RL+merge: rl@256zoom, rl@2048zoom(len override), rl@256 no-zoom; rl "direct" = rl@256+zoom re-score if rl emits text
- NOTE no-zoom override path = `actor_rollout_ref.rollout.multi_turn.tool_config_path=<nozoom.yaml>` (NOT ...multi_turn... without .rollout)
