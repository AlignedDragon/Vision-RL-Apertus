# Clariden Cluster Guide

Personal reference for working on the CSCS Clariden cluster as part of the Swiss AI initiative.

**Cluster:** ~1,200 NVIDIA Grace-Hopper (GH200) nodes, 4 GPUs each  
**Account:** `infra01`  
**OS:** SLES 15-SP6 (aarch64 / ARM-based)  
**Scheduler:** SLURM + Lmod modules  

---

## 1. Storage Architecture

Clariden has four storage tiers. Each one has a specific purpose — putting data in the wrong tier degrades performance for everyone.

### Quick Reference

| Tier | Path | Symlink | Quota | Lifetime | Use For |
|------|------|---------|-------|----------|---------|
| **Home** | `/users/$USER/` | `~/` | 50 GB / 500K inodes | Permanent (daily snapshots, 7-day retention) | Configs, small repos, symlinks |
| **Iopsstor Scratch** | `/iopsstor/scratch/cscs/$USER/` | `~/scratch` (`$SCRATCH`) | 150 TB | 30-day auto-delete | Active datasets, HF cache, high-IOPS work |
| **Capstor Scratch** | `/capstor/scratch/cscs/$USER/` | — | Large | 30-day auto-delete | Checkpoints, large sequential outputs |
| **Project Store** | `/capstor/store/cscs/swissai/infra01/reasoning/users/$USER` | `~/project` | Shared quota | Permanent (tape backup, 3 versions) | Final models, curated data, important results |

There's also a shared team directory:

| Path | Symlink | Contents |
|------|---------|----------|
| `/capstor/store/cscs/swissai/infra01/reasoning/` | `~/shared` | `data/`, `models/`, `containers/`, `imgs/`, `dev/`, user dirs |

### What Goes Where

**Home (`~/`) — config only, never compute**
- Shell profiles (`.bashrc`, `.profile`)
- SSH keys, editor configs, `.vscode/`
- Small git repos (source code only, no data)
- Symlinks to other tiers
- Conda base install (`miniconda3/`)

Do NOT put here: datasets, checkpoints, model weights, training outputs, HuggingFace cache, large logs.

**Iopsstor Scratch (`~/scratch`) — fast I/O for active work**
- HuggingFace cache (`$SCRATCH/hf_home`)
- Training datasets (parquet files with random access)
- Active experiment outputs and intermediate checkpoints
- Container image cache (`.edf_imagestore/`)
- Anything you're actively reading/writing during training

This is NVMe SSD storage — the fastest available. Use it for anything that needs random I/O during compute jobs.

**Capstor Scratch — sequential writes**
- Checkpoint dumps during training (sequential large writes)
- Simulation outputs, large contiguous files
- NOT for datasets with random reads (use iopsstor)

Same 30-day cleanup as iopsstor. Move anything important to project store before it expires.

**Project Store (`~/project`) — permanent archive**
- Final trained model checkpoints
- Curated/generated datasets you want to reuse across experiments
- Important results and logs worth preserving
- Anything you'd be upset to lose

This is the only scratch-safe tier. Files here are backed up to tape (3 most recent versions).

### Storage Warnings

- When scratch hits **60% cluster-wide occupancy**, CSCS asks users to clean up
- At **80%+**, automatic deletion can happen without warning (not just 30-day policy)
- Soft quotas have a grace period; hard quotas block new writes immediately
- Run quota checks regularly (see Section 6)

---

## 2. Compute (SLURM)

### Login Nodes

You're on a login node right now (e.g., `clariden-ln003`). These are shared.

**Allowed:** editing, git, job submission, light file management, `conda install`  
**NOT allowed:** training, inference, GPU work, heavy data processing

Any real compute must go through SLURM.

### Partitions

| Partition | Nodes | Max Time | Use For |
|-----------|-------|----------|---------|
| `debug` | 24 | 1.5 hours | Quick tests, debugging, interactive dev |
| `normal` | 1,204 | 12 hours | Production training runs |
| `xfer` | 2 | 24 hours | Data transfers between storage tiers |

### Job Submission

**Interactive debug session:**
```bash
srun --account=infra01 -p debug --gpus-per-node=4 --pty bash
```

Or use the alias from `.bashrc`:
```bash
sdebug  # srun --account=infra01 -p debug --pty --container-writable
```

**Batch job (production):**
```bash
sbatch slurm/apertus_bbox.slurm
```

**VS Code tunnel on compute node:**
```bash
scode --gpus-per-node=4 --time=12:00:00 --environment=verl_env
```

### GPU Configuration (GH200)

Each node has 4 GPUs with ~98 GB each. Standard config:

```bash
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=460000
```

For multi-node training, increase `--nodes=N` and adjust `ntasks-per-node` if needed.

### SLURM Header Template

```bash
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=460000
#SBATCH --time=12:00:00
#SBATCH --environment=verl_env
#SBATCH --partition=normal
#SBATCH --gpus-per-node=4
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
```

---

## 3. Container Environment

Clariden uses **Enroot** for containers, configured via **Environment Definition Files (EDFs)** in TOML format.

### EDF Basics

EDF files define what container image to use, what to mount, and what environment to set. They live in `~/.edf/` or are specified by absolute path.

```toml
# Example: ~/.edf/verl_env.toml
image = "docker://nvcr.io/nvidia/pytorch:24.01-py3"
workdir = "/users/badralmahouri/verl-vision"
mounts = [
    "${SCRATCH}:/scratch",
    "${HOME}:${HOME}",
    "/capstor:/capstor"
]
env = [
    "HF_HOME=${SCRATCH}/hf_home"
]
writable = true
```

### Using Containers

Specify with `--environment` on `srun` (not `sbatch`):
```bash
srun --environment=verl_env python train.py
```

Or as an `#SBATCH` directive:
```bash
#SBATCH --environment=verl_env
```

Images auto-cache in `$SCRATCH/.edf_imagestore`. Private registry credentials go in `$HOME/.config/enroot/.credentials`.

---

## 4. Training Workflow (End-to-End)

This is the typical cycle for running a VeRL tool-use training experiment.

### Step 1: Edit Code (Login Node)

Work in `~/verl-vision/`. Edit configs, data generation scripts, reward functions, etc.

Key locations:
```
~/verl-vision/
├── examples/sglang_multiturn/config/   # Hydra configs + chat templates
├── verl/experimental/agent_loop/       # Agent loop + tool parsers
├── verl/utils/reward_score/            # Reward functions
├── data/                               # Dataset generators + parquet files
├── data_postprocess/                   # Training entry point (verl_metrics.py)
└── slurm/                              # Job scripts, logs, plots
```

### Step 2: Generate Data (if needed)

Run data generation scripts on the login node (they're lightweight):
```bash
cd ~/verl-vision
python data/generate_random_bbox.py     # bbox task
python data/generate_enhance_tasks.py   # enhance task
python data/generate_random_zoom.py     # zoom task
python data/generate_rotate_tasks.py    # rotate task
```

Output: parquet files in `data/<task>/train.parquet` and `data/<task>/test.parquet`.

### Step 3: Submit Training Job

```bash
cd ~/verl-vision
sbatch slurm/apertus_bbox.slurm        # Apertus on bbox
sbatch slurm/run_image_bbox_tool_example.slurm  # Qwen on bbox
sbatch slurm/run_zoom_tool_example.slurm        # Qwen on zoom (with tool)
sbatch slurm/run_zoom_baseline.slurm            # Qwen on zoom (no tool, baseline)
```

Each SLURM script:
1. Activates conda `verl` environment
2. Sets up env vars (HF cache -> `$SCRATCH`, NCCL config, Ray tmpdir)
3. Runs pre-flight checks (data files, model checkpoints, tokenizer, chat template)
4. Launches training via `python3 data_postprocess/verl_metrics.py --config-name=...`
5. After training: extracts metrics from SLURM output -> `slurm/logs/`
6. Auto-generates plots -> `slurm/plots/`

### Step 4: Monitor Running Jobs

```bash
squeue -u $USER                          # list your jobs
squeue -u $USER -o "%.10i %.20j %.8T %.10M %.6D %.4C %.20R"  # detailed
tail -f verl_image_bbox_tool_<JOBID>.out # live log
scancel <JOBID>                          # cancel a job
```

### Step 5: Analyze Results

After the job completes:

**Metrics log** (step-by-step training stats):
```
slurm/logs/<model>_<task>_<JOBID>.log
```

**Reward debug log** (per-sample details: IoU, answer score, tool calls):
```
slurm/logs/reward_debug_<JOBID>.jsonl
```

**Training plots** (auto-generated):
```
slurm/plots/<model>_<task>_<JOBID>/
```

**Regenerate plots manually:**
```bash
python slurm/plot.py slurm/logs/<metrics_file>.log slurm/plots/<output_dir>
```

**Checkpoints** (written during training):
```
$SCRATCH/checkpoint/<model>_<task>_<version>/
```

### Step 6: Preserve Important Results

Checkpoints on scratch auto-delete after 30 days. Move what matters:
```bash
cp -r $SCRATCH/checkpoint/apertus_bbox_v3 ~/project/checkpoints/
cp slurm/logs/reward_debug_<JOBID>.jsonl ~/project/logs/
```

### Workflow Diagram

```
Login Node                    Compute Node (via SLURM)
──────────                    ────────────────────────
1. Edit code in ~/verl-vision
2. Generate data (lightweight)
3. sbatch slurm/xxx.slurm ──→ 4. Conda activate verl
                                 Set env vars (HF_HOME, NCCL, Ray)
                                 Pre-flight checks
                                 Launch GRPO training
                                 ├─ SGLang rollout (multi-turn agent loop)
                                 ├─ Tool execution (bbox/zoom/enhance/rotate)
                                 ├─ Reward computation (IoU + answer)
                                 └─ Policy update (GRPO)
                                 Write checkpoints → $SCRATCH
                                 Extract metrics → slurm/logs/
                              ←─ Generate plots → slurm/plots/
5. Analyze results
6. Archive to ~/project
```

---

## 5. Key Paths Reference

### Environment Variables

| Variable | Value |
|----------|-------|
| `$HOME` | `/users/badralmahouri` |
| `$SCRATCH` | `/iopsstor/scratch/cscs/badralmahouri` |
| `$PROJECT` | `/capstor/store/cscs/ethz/large-sc-2` |
| `$STORE` | `/capstor/store/cscs/ethz/large-sc-2` |
| `$CLUSTER_NAME` | `clariden` |

### Symlinks in Home

| Symlink | Target |
|---------|--------|
| `~/scratch` | `/iopsstor/scratch/cscs/badralmahouri/` |
| `~/project` | `/capstor/store/cscs/swissai/infra01/reasoning/users/badralmahouri` |
| `~/shared` | `/capstor/store/cscs/swissai/infra01/reasoning` |

### Model Checkpoints

| Model | Path |
|-------|------|
| Apertus 8B (SFT) | `/capstor/store/cscs/swissai/infra01/MLLM/ablations/apertus-8b-img-SFT-32nodes-gbs512-mbs1-steps8030-img-text-seqlen8192-s2onlytxtloss/HF` |
| Emu3.5 VisionTokenizer (IBQ) | `/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/Emu3.5-VisionTokenizer` |
| Qwen2.5-VL-3B | Downloaded via HF to `$SCRATCH/hf_home` |

### Code Repositories

| Repo | Path | Purpose |
|------|------|---------|
| verl-vision | `~/verl-vision/` | Main training codebase (VeRL fork for VLMs) |
| verl-apertus | `~/verl-apertus/` | Launcher configs, this guide |
| verl-main-ML | `~/verl-main-ML/` | Earlier experiments (line, flip, blur, crop) |
| Emu3.5 | `~/Emu3.5/` | IBQ vision tokenizer source |

### Key Files in verl-vision

| File | Purpose |
|------|---------|
| `verl/experimental/agent_loop/tool_agent_loop.py` | Multi-turn agent orchestration |
| `verl/experimental/agent_loop/tool_parser.py` | Tool call extraction (Hermes + Apertus formats) |
| `verl/utils/reward_score/reward_bbox.py` | Reward computation (IoU + answer correctness) |
| `data_postprocess/verl_metrics.py` | Training entry point |
| `examples/sglang_multiturn/config/*.yaml` | Hydra training configs |
| `examples/sglang_multiturn/config/*_chat_template.jinja2` | Chat templates per model |

### Training Data

| Task | Data Path | Generator |
|------|-----------|-----------|
| BBox | `data/bbox/train.parquet` | `data/generate_random_bbox.py` |
| Enhance | `data/enhance_open/train.parquet` | `data/generate_enhance_tasks.py` |
| Zoom | `data/zoom_open/train.parquet` | `data/generate_random_zoom.py` |
| Rotate | `data/rotate_open/train.parquet` | `data/generate_rotate_tasks.py` |

### SLURM Scripts

| Script | Model | Task | Tool? |
|--------|-------|------|-------|
| `apertus_bbox.slurm` | Apertus 8B | BBox | Yes |
| `run_image_bbox_tool_example.slurm` | Qwen2.5-VL-3B | BBox | Yes |
| `run_zoom_tool_example.slurm` | Qwen | Zoom | Yes |
| `run_zoom_baseline.slurm` | Qwen | Zoom | No (baseline) |
| `run_enhance_tool_example.slurm` | Qwen | Enhance | Yes |
| `run_enhance_baseline.slurm` | Qwen | Enhance | No (baseline) |
| `run_rotate_tool_example.slurm` | Qwen | Rotate | Yes |
| `run_rotate_baseline.slurm` | Qwen | Rotate | No (baseline) |

---

## 6. Useful Commands

### SLURM

```bash
squeue -u $USER                  # your running/pending jobs
squeue -u $USER -t RUNNING       # only running
scancel <JOBID>                  # cancel a job
scancel -u $USER                 # cancel all your jobs
sinfo -p normal                  # partition status
sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,MaxRSS  # job history
```

### Storage

```bash
du -sh ~/                        # home usage
du -sh $SCRATCH/*                # scratch breakdown
lfs quota -u $USER /iopsstor     # iopsstor quota (if lustre)
```

### Modules

```bash
module list                      # currently loaded modules
module avail                     # available modules
module load <name>               # load a module
```

### Conda

```bash
conda activate verl              # activate training environment
conda list                       # installed packages
```

### Aliases (from .bashrc)

```bash
sdebug                           # interactive debug node with container
scode                            # VS Code tunnel on compute node
```

---

## 7. Resources

| Resource | URL |
|----------|-----|
| Swiss AI Getting Started | https://github.com/swiss-ai/reasoning_getting-started |
| CSCS Clariden Docs | https://docs.cscs.ch/clusters/clariden/ |
| Storage & Filesystems | https://docs.cscs.ch/storage/filesystems/ |
| SLURM Guide | https://docs.cscs.ch/running/slurm/ |
| Container Engine | https://docs.cscs.ch/software/container-engine/ |
| EDF Reference | https://docs.cscs.ch/software/container-engine/edf/ |
| ML Platform | https://docs.cscs.ch/platforms/mlp/ |
| System Status | https://status.cscs.ch |
| Support Portal | https://support.cscs.ch |
| Knowledge Base | https://confluence.cscs.ch |

**Maintenance window:** Wednesdays 8:00-12:00 CET (services may be unavailable).

**Troubleshooting:** Check Slack channels and status page first — over 90% of issues are resolved through self-service before contacting supervisors.
