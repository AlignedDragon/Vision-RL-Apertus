# Apertus TextVQA Evaluation — Steps 1 & 2

## What is this project about?

We are evaluating **Apertus 8B**, a vision-language model developed by the SwissAI initiative, to understand how well it answers questions about text in images. The broader goal is a six-step roadmap to improve Apertus's performance through progressively more sophisticated approaches:

1. **Define a test set** (TextVQA, 200 samples) — DONE
2. **Baseline evaluation** with no guidance prompt — DONE (this document)
3. Evaluation with a "thinking" prompt (chain-of-thought)
4. Agentic retrieval-based approach
5. SFT (supervised fine-tuning) with curated data
6. MORL (multi-objective reinforcement learning)

This document covers steps 1–2: what was done, why each choice was made, and what the results tell us.

---

## Background: How Apertus sees images

Apertus is not a typical VLM — it does not have a vision encoder that feeds continuous features into the language model. Instead, it uses **IBQ (Image-Based Quantization)**: images are encoded into discrete tokens from a codebook of 131,072 entries, and these tokens are inserted directly into the text prompt as if they were regular text tokens.

The encoding pipeline works as follows:
1. A PIL image is resized to ~512x512 pixels (aspect-preserving, dimensions divisible by 16)
2. Pixels are normalized to [-1, 1]
3. The **Emu3.5 VisionTokenizer** (an IBQ encoder) encodes the image into a 2D grid of codebook indices (e.g., 32x32 = 1024 tokens for a square image)
4. These indices are formatted into a special token string using **Apertus-specific naming**: `<|img_start|>32*32<|img_token_start|><|visual token 0|><|visual token 42|>...<|img_end|>`

A critical subtlety: **Apertus and Emu3.5 use different token naming conventions**. Emu3.5 uses `<|image start|>` and `<|visual token 000000|>` (zero-padded to 6 digits), while Apertus uses `<|img_start|>` and `<|visual token 0|>` (no padding). Using the wrong convention would produce unknown tokens and garbage output.

The Emu3.5 VisionTokenizer checkpoint lives on the shared project store at `/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/Emu3.5-VisionTokenizer`. It is not a HuggingFace model — it uses a custom `config.yaml + model.ckpt` format and can only be loaded through the Emu3.5 source code's `build_vision_tokenizer("ibq", path)` function. The Emu3.5 source code is cloned at `~/Emu3.5/src/` and added to PYTHONPATH at runtime.

---

## Step 1: Defining the test set

### Why TextVQA?

TextVQA is a visual question answering benchmark where models must read and reason about text visible in images (signs, labels, documents, screens, etc.). It is a good fit for evaluating Apertus because:
- It tests both visual understanding (seeing the image) and text recognition (reading what's in it)
- It has a well-defined accuracy metric used across the community
- The questions are concrete and have short, verifiable answers
- The validation set has ~5,000 samples — plenty to subsample from

### Why 200 samples?

A full evaluation of 5,000 samples would take hours of GPU time. 200 samples are enough to get a meaningful accuracy estimate while keeping each experiment under 10 minutes of inference. This matters because we have 6 steps in the roadmap and need fast iteration.

### How the data was prepared

The script `datasets/prepare_textvqa.py` runs on the login node (no GPU needed):
1. Downloads the official TextVQA v0.5.1 validation annotations JSON from Facebook's CDN
2. Selects 300 candidate samples using `random.seed(42)` for determinism
3. Attempts to download each candidate's image from Flickr
4. Keeps the first 200 that succeed

**Why over-sample 300 to get 200?** Flickr URLs in TextVQA are old and roughly 20% are broken at any given time. On the first attempt with only 200 candidates, we got 163 images. Over-sampling to 300 and taking the first 200 successes solved this.

**Known reproducibility issue:** Because the script skips broken Flickr URLs, running it at a different time may get a slightly different set of 200 samples (different URLs may be broken). The `metadata.jsonl` file is currently gitignored along with the rest of `datasets/textvqa/`, which means a teammate would re-run the script and potentially get a different dataset. This should be fixed — either by committing the metadata, or by storing a canonical list of question IDs.

### What the data looks like

Each sample in `datasets/textvqa/metadata.jsonl` has:
- `question_id`: unique integer
- `image_id`: Flickr image ID
- `question`: natural language question (e.g., "what is the name on the sign?")
- `answers`: list of 10 human annotations
- `image_file`: relative path to the JPEG image

Images are stored as `datasets/textvqa/images/{question_id}.jpg`. Total dataset size: ~8.7 MB.

---

## Step 2: Baseline evaluation

### What "baseline" means

The model is given each image + question with no special instructions — no "answer briefly", no chain-of-thought prompting, no retrieval. Just the raw Apertus chat template with the image tokens and the question. This establishes a floor that future approaches must beat.

### The prompt format

Each prompt follows the Apertus chat template:
```
<s>
<|system_start|>
You are Apertus, a helpful assistant created by the SwissAI initiative.
Knowledge cutoff: 2024-04
Current date: 2026-04-13
<|system_end|>
<|developer_start|>
Deliberation: disabled
Tool Capabilities: disabled
<|developer_end|>
<|user_start|>
<|img_start|>32*32<|img_token_start|><|visual token ...>...<|img_end|>
What is written on the sign?
<|user_end|>
<|assistant_start|>
```

The model's built-in `apply_chat_template()` was used on all 4 GPUs (it worked correctly; the manual fallback was not needed). Deliberation and tool capabilities are disabled because this is a simple Q&A task — we want a direct answer, not an internal reasoning trace or tool call.

Generation parameters: **greedy decoding** (temperature=0.0) for full reproducibility, max 256 new tokens, stopping at `</s>` or `<|assistant_end|>`.

### How inference was parallelized

The job runs on one Clariden node with 4x NVIDIA GH200 120GB GPUs. The inference has two phases:

**Phase 1 — Image encoding (GPU 0 only, ~48 seconds):**
The VQ model is loaded on GPU 0, all 200 images are encoded to token strings, then the VQ model is freed. This is done sequentially on one GPU because the VQ model is lightweight (~300MB) and encoding is fast (~4 images/second). The output is a dict mapping question_id to token string held in CPU memory.

**Phase 2 — Text generation (all 4 GPUs, ~2 minutes):**
The 200 samples are split into 4 chunks of 50. Four processes are spawned via `torch.multiprocessing` (spawn method), each loading its own copy of Apertus 8B (~18GB in bfloat16) on its assigned GPU. Each process generates answers for its 50 samples and writes results to a partial JSONL file. After all processes finish, the partials are merged into `results/textvqa/baseline/predictions.jsonl`, sorted by question_id for determinism.

**Why not pipeline or tensor parallelism?** At 18GB per model copy and 120GB per GPU, we can comfortably fit 4 independent copies. Data parallelism is simpler and gives near-linear speedup for independent samples. Pipeline parallelism would only help if the model didn't fit on one GPU.

**Why spawn 4 processes instead of using DataParallel/FSDP?** For batch-size-1 greedy decoding on independent samples, `torch.multiprocessing` with separate model copies is simpler and avoids all-reduce overhead. Each process is completely independent.

### SLURM configuration

```
#SBATCH --account=infra01
#SBATCH --partition=normal          # up to 12h
#SBATCH --environment=verl_env      # Clariden's container-based env
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=460000                # ~450 GB (4 model copies + tokenizer + overhead)
#SBATCH --time=2:00:00
```

The SLURM script (`slurm/run_baseline.slurm`) includes pre-flight checks that verify all model/data paths exist and that `einops` is installed before starting inference. The `einops` library is required by the IBQ encoder's internals and was not in the verl conda environment by default — it was installed separately via `pip install einops`.

### Timing breakdown (SLURM job 1845890, run 2026-04-13)

| Phase | Duration |
|---|---|
| Pre-flight checks | < 1s |
| VQ model load | ~5s |
| 200 image IBQ encoding | 48s |
| 4x model loading (parallel, from capstor) | ~40s |
| Inference (200 samples, 4 GPUs) | ~2 min |
| Evaluation | < 1s |
| **Total wall time** | **~4 minutes** |

Throughput during inference: 0.4–0.8 samples/s per GPU, ~2.0 samples/s aggregate. The variance between GPUs is likely due to different output lengths (some answers are 5 tokens, others are 150+).

---

## Results

### Headline numbers

| Metric | Value |
|---|---|
| **Overall TextVQA accuracy** | **4.5%** |
| Samples evaluated | 200 |
| Correct (accuracy > 0) | 9 (4.5%) |
| Perfect match (accuracy = 1.0) | 9 (4.5%) |

### How TextVQA accuracy works

Each sample has 10 human-annotated answers. The accuracy for a single prediction is:

```
accuracy = min(1, count_of_matching_answers / 3)
```

Both the prediction and each human answer are normalized before comparison: lowercased, punctuation removed, articles ("a", "an", "the") removed, whitespace collapsed. A prediction needs to match at least 3 of the 10 annotations to get a perfect score of 1.0.

### What went wrong: format mismatch, not visual blindness

The 4.5% accuracy is misleadingly low. **The model frequently understands the image correctly but wraps the answer in a verbose sentence**, which fails the exact-match evaluation:

| Question | Prediction | Expected | Accuracy |
|---|---|---|---|
| "how long has the drink been aged?" | "The drink on the right has been aged for 10 years." | "10 years" | 0.0 |
| "what is the name of the runner on the left?" | "The runner on the left is named Willis." | "willis" | 0.0 |
| "what number is the player looking at?" | "The player is looking at the number 9 on the children's jerseys." | "9" | 0.0 |
| "how much is the coin worth?" | (4-sentence paragraph about the coin's history and value) | "25" | 0.0 |

The 9 samples that scored 1.0 are cases where the model happened to produce a terse answer: "9", "CHEESE", "1", "17", "8", "6-12". These are mostly numeric questions where there's no room for elaboration.

### Two categories of failure

1. **Correct content, wrong format (~60-70% of failures):** The model extracts the right information from the image but answers conversationally. "The number for Southern Homes is 648-4500" instead of "648-home". These are recoverable with better prompting.

2. **Genuinely wrong answers (~30-40% of failures):** The model misreads text, hallucinates, or gives factually incorrect information. "San Pellegrino" instead of "chino" for a beer brand. "Heinrich Boll" instead of "Hemingway" for a book author. These indicate real visual or knowledge errors.

### What this tells us about Step 3

The format mismatch is the dominant failure mode. A simple instruction like "Answer with a single word or short phrase" in the system or user prompt should recover many of the verbose-but-correct predictions. This is exactly what Step 3 (thinking/guidance prompt) should address first — before any chain-of-thought reasoning, we need to fix the output format.

---

## Project structure

```
~/verl-apertus/
├── configs/
│   └── apertus.yaml               # Model paths, generation params, dataset config
├── datasets/
│   ├── prepare_textvqa.py          # Downloads 200 TextVQA samples from Flickr
│   └── textvqa/                    # Images + metadata (local only, not in git)
│       ├── metadata.jsonl
│       └── images/
├── inference/
│   ├── run_baseline.py             # 4-GPU parallel inference
│   └── vision.py                   # IBQ image encoding
├── evaluation/
│   └── compute_accuracy.py         # TextVQA soft accuracy metric
├── results/                        # gitignored
│   └── textvqa/baseline/
│       ├── predictions.jsonl       # 200 predictions (96 KB)
│       └── accuracy.json           # Per-sample + overall accuracy (75 KB)
├── slurm/
│   └── run_baseline.slurm          # SLURM job script
└── .gitignore
```

### Key external dependencies

| What | Where | Why it's there |
|---|---|---|
| Apertus 8B checkpoint | `/capstor/.../apertus-8b-img-SFT-.../HF` | Shared project store, too large for home dir (18 GB) |
| Emu3.5 VisionTokenizer | `/capstor/.../Emu3.5-VisionTokenizer` | IBQ encoder weights (shared project store) |
| Emu3.5 source code | `~/Emu3.5/src/` | Provides `build_vision_tokenizer()` loader — no pip package exists |
| conda verl environment | `~/miniconda3/envs/verl/` | Python 3.12, PyTorch, transformers, einops |

---

## Known issues and open questions

1. **Dataset reproducibility:** `datasets/textvqa/` is gitignored, so a teammate running `prepare_textvqa.py` at a different time may get a slightly different 200-sample subset due to changing Flickr URL availability. A canonical list of question IDs should be committed.

2. **Tokenizer regex warning:** The Apertus tokenizer emits a warning about an incorrect regex pattern inherited from Mistral. It does not seem to affect output quality but could cause subtle tokenization differences. The warning suggests setting `fix_mistral_regex=True`.

3. **`torch_dtype` deprecation:** The transformers library warns that `torch_dtype` is deprecated in favor of `dtype`. Not urgent but should be updated.

4. **rope_scaling warning:** `original_max_position_embeddings` equals `max_position_embeddings` (both 8192), which triggers a warning. This is a checkpoint config issue, not something we control.

---

## Roadmap

| Step | Description | Status |
|---|---|---|
| 1 | Define test set (TextVQA, 200 samples) | Done |
| 2 | Baseline evaluation, no guidance prompt | Done — **4.5% accuracy** |
| 3 | Evaluation with "thinking" / guidance prompt | Next |
| 4 | Agentic retrieval-based approach | Planned |
| 5 | SFT with curated data | Planned |
| 6 | MORL (multi-objective RL) | Planned |
