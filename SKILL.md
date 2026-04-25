---
name: tao-automl
description: >-
  Run hyperparameter optimization (HPO) for NVIDIA TAO networks using AutoMLRunner.
  Handles algorithm selection (bayesian, hyperband, asha, bohb, llm, hybrid, autoresearch),
  WandB experiment tracking, job execution on Lepton/DGX/Slurm, result interpretation,
  and per-rec custom evaluation hooks. Use when the user mentions TAO AutoML, hyperparameter
  optimization, HPO, automl, automl_settings, AutoMLRunner, tao_automl, bayesian search,
  hyperband, ASHA, LLM-guided search, autoresearch, or wants to tune training hyperparameters
  for any TAO network (cosmos-rl, dino, segformer, clip, etc.).
---

# TAO AutoML Skill

Run automated hyperparameter optimization (HPO) for any TAO network. The agent uses `AutoMLRunner` — a single interface that manages the full loop: generate hyperparameter recommendations, launch training jobs on Lepton/DGX, extract metrics, and feed results back to the optimizer.

The runner is platform-agnostic: platform selection (Lepton, Slurm, K8s) and GPU allocation are handled by the SDK the runner is given, not by the runner itself.

## Prerequisites

Before running AutoML:

1. **SDK credentials**: `secrets.json` must exist with Lepton/DGX credentials. The existing file lives at `~/tao-sdk/secrets.json`:
   ```json
   {
     "LEPTON_WORKSPACE_ID": "...",
     "LEPTON_AUTH_TOKEN": "nvapi-...",
     "NGC_KEY": "nvapi-...",
     "ACCESS_KEY": "...",
     "SECRET_KEY": "...",
     "S3_BUCKET_NAME": "nvcf-storage-handling",
     "CLOUD_REGION": "us-west-1"
   }
   ```
   Pass it via `TaoExecutionSDK(creds_file="~/tao-sdk/secrets.json")`.
2. **Dataset**: Training data accessible from the compute backend. URI format depends on the SDK's platform:
   - Lepton / DGX Cloud: `aws://bucket/path` (S3-compatible)
   - Azure: `azure://container/path`
   - Local / Docker: local filesystem path
3. **Skill bank available**: lives at `~/tao-skills-external`. **CRITICAL**: The runner raises `ValueError: No skill config found for '<network>'` if the skill bank is not set. You MUST set it before importing the runner:
   ```python
   import os
   os.environ["TAO_SKILL_BANK_PATH"] = os.path.expanduser("~/tao-skills-external")
   ```
   Or in bash:
   ```bash
   export TAO_SKILL_BANK_PATH=~/tao-skills-external
   ```
   The bank structure is:
   ```
   tao-skills-external/
   ├── applications/         # workflow configs (normal-train, deft-cosmos-rl, ...)
   ├── models/               # per-network skill packages
   │   ├── dino/
   │   │   ├── config.json           # actions, data_sources, container image
   │   │   ├── defaults-train.json   # default training spec (REQUIRED by AutoML)
   │   │   └── dino.md
   │   ├── cosmos-rl/
   │   ├── segformer/
   │   └── ...
   ├── data/
   └── platform/
   ```
   **CRITICAL**: The `SkillBank.get_default_specs(network, "train")` call requires either `references/spec_template_train.yaml` or `defaults-train.json` in the model directory. If missing, the runner raises `ValueError: No default train specs found`. Create `defaults-train.json` from the network's experiment spec (found at `tao-pytorch/nvidia_tao_pytorch/cv/<network>/experiment_specs/train.yaml`).
4. **Conda environment**: Use the `tao_sdk` conda environment which has `tao-sdk` pre-installed:
   ```bash
   conda activate tao_sdk
   # or prefix commands:
   conda run -n tao_sdk python3 my_script.py
   ```
   For unbuffered output (recommended for long-running AutoML), use the Python binary directly:
   ```bash
   PYTHONUNBUFFERED=1 ~/miniconda3/envs/tao_sdk/bin/python my_script.py
   ```
5. **`nvidia-tao-automl` installed** (editable dev install into `tao_sdk` env):
   ```bash
   conda activate tao_sdk
   # Core only (classical algorithms)
   pip install -e "~/tao-automl[dev]" --extra-index-url https://pypi.nvidia.com

   # With LLM/agentic algorithms
   pip install -e "~/tao-automl[dev,llm]" --extra-index-url https://pypi.nvidia.com

   # Everything
   pip install -e "~/tao-automl[all,dev]" --extra-index-url https://pypi.nvidia.com
   ```

Verify setup:
```bash
conda run -n tao_sdk python3 -c "from tao_automl.runner import AutoMLRunner; print('OK')"

# Verify LLM features (optional)
conda run -n tao_sdk python3 -c "from tao_automl.brain.llm_brain import LLMBrain; print('LLM OK')"

# Verify WandB (optional)
conda run -n tao_sdk python3 -c "import wandb; print('WandB OK')"
```

---

## Concepts: What is TAO AutoML?

TAO AutoML automates the "try different hyperparameter values → train → compare results → repeat" cycle. Instead of manually tweaking learning rate, batch size, or backbone settings, you tell AutoML:

- **What network** to train (e.g. `dino`, `cosmos-rl`, `segformer`)
- **Which hyperparameters** to search over (e.g. `train.optm_lr`, `policy.lora.r`)
- **What metric** to optimize (e.g. `val_loss`, `accuracy`, `mIoU`)
- **How many trials** (budget)

AutoML then:
1. Picks hyperparameter values using a search algorithm (Bayesian, Hyperband, LLM, etc.)
2. Launches a real training job on your compute backend (Lepton, DGX, Slurm)
3. Reads the result metric from training logs
4. Feeds the result back to the algorithm so it learns what works
5. Repeats until budget is exhausted
6. Returns the best configuration found

Each "trial" is called a **recommendation** (rec). One rec = one full training run with a specific set of hyperparameters.

---

## Step 1: Parse User Intent

Extract from the user's request:

| Field | Required | Example | How to get it |
|---|---|---|---|
| `network_arch` | Yes | `"cosmos-rl"`, `"dino"`, `"clip"` | User states the model |
| `train_dataset_uri` | Yes | `"aws://bucket/data/subset"` | User provides the URI |
| `metric` | No | `"loss"` (default) | Ask if unclear — `loss` / `val_loss` for generative tasks, `accuracy` / `mIoU` / custom for classification / segmentation |
| `direction` | No | `"minimize"` or `"maximize"` | **Only needed if your metric name doesn't contain `"loss"` AND you want to minimize, or contains `"loss"` AND you want to maximize.** Otherwise the implicit "contains 'loss' → minimize, else maximize" rule applies. |
| `algorithm` | No | `"bayesian"` (default) | See algorithm guide below |
| `max_recommendations` | No | 5–20 | Ask budget — each rec is one full training run |
| `skill_bank_path` | Yes (if not set) | `"~/tao-skills-external"` | Check if `TAO_SKILL_BANK_PATH` env var is set. If not, ask the user for the path. Default: `~/tao-skills-external`. The runner raises `ValueError: No skill config found` without it. |
| `llm_endpoint` | **Yes** (for `llm`/`hybrid`/`autoresearch`) | `"https://inference-api.nvidia.com"` | **MUST prompt.** The code default `https://integrate.api.nvidia.com/v1` returns 404. Always ask for and pass explicitly. |
| `llm_model` | **Yes** (for `llm`/`hybrid`/`autoresearch`) | `"gcp/google/gemini-3.1-pro-preview"` | **MUST prompt.** Ask which model to use. Default: `meta/llama-3.1-70b-instruct` via NIM. |
| `llm_api_key` | **Yes** (for `llm`/`hybrid`/`autoresearch`) | `"nvapi-..."` or `"sk-..."` | **MUST prompt** if `NVIDIA_API_KEY` / `AUTOML_LLM_API_KEY` env vars are not set. |

If any required field is missing, ask the user. Do NOT guess dataset paths, skill bank paths, or LLM endpoints.

**MANDATORY: Determine user experience level.**

Before diving into configuration, ask the user (or infer from context) whether they want:

**Quick start (first-time / "just run it"):**
- Algorithm: `hybrid` (LLM + bayesian) — intelligent search with no manual tuning
- Experiments: 10 recommendations
- Hyperparameters: `None` (use schema defaults — params with `automl_enabled=True` in the network's dataclass)
- Ranges: schema defaults (no `custom_param_ranges`)
- The agent only needs: `network_arch`, `train_dataset_uri`, and LLM credentials (for hybrid)

When `automl_hyperparameters=None`, the runner automatically discovers all params marked `automl_enabled=True` in the network's JSON schema. For cosmos-rl these are: `policy.lora.r`, `policy.lora.lora_alpha`, `policy.lora.lora_dropout`, `train.epoch`, `train.optm_lr`, `train.optm_decay_type`, `custom.vision.fps`. Each network has its own set — the dataclass definitions in `tao-automl/src/tao_automl/config/<network>/` are the source of truth.

```python
# Quick start — all defaults
result = runner.run(
    network_arch="cosmos-rl",
    train_dataset_uri=S3_TRAIN,
    automl_settings={
        "algorithm": "hybrid",
        "metric": "val_loss",
        "automl_max_recommendations": 10,
        "llm_endpoint": "https://inference-api.nvidia.com",
        "llm_model": "gcp/google/gemini-3.1-pro-preview",
        "llm_api_key": llm_api_key,
    },
    # automl_hyperparameters=None → uses schema defaults
    # custom_param_ranges=None → uses schema defaults
    spec_overrides={...},  # from model MD Typical Spec Overrides
    workspace_path=f"./automl/{TIMESTAMP}",
)
```

**Advanced (customize everything):**
- User can see which params are available and their default ranges
- User picks which params to enable/disable
- User sets custom bounds, option weights, and algorithm-specific settings
- User picks the algorithm

The agent should present the available AutoML parameters from the model's schema. Each param record includes: `parameter` (dotted name), `value_type` (int, float, categorical, list_2, subset_list, etc.), `default_value`, `valid_min`, `valid_max`, `valid_options`, `option_weights`, `math_cond`, `depends_on`, and `automl_enabled`.

To show available params programmatically:

```python
from tao_automl.search_space.params import generate_hyperparams_to_search
from tao_automl.schema.dataclass2json_converter import generate_schema

schema = generate_schema(network_arch, "train")
param_records, param_names = generate_hyperparams_to_search(
    network=network_arch,
    action="train",
    train_specs=schema.get("default", {}),
    automl_hyperparameters=None,  # None = show all automl_enabled params
    override_automl_disabled_params=True,  # True = show ALL params, even disabled ones
)
for rec in param_records:
    if rec:
        print(f"{rec['parameter']:40s} type={rec['value_type']:15s} "
              f"enabled={rec.get('automl_enabled', False)}")
```

Then the user picks from this list and optionally customizes ranges:

```python
result = runner.run(
    ...,
    automl_hyperparameters=[
        "train.optm_lr",
        "policy.lora.r",
        "policy.lora.lora_alpha",
        "train.optm_decay_type",
    ],
    custom_param_ranges={
        "train.optm_lr": {"valid_min": 5e-6, "valid_max": 2e-4},
        "train.epoch": {"valid_min": 1, "valid_max": 3},
        "train.optm_betas": {"valid_min": [0.9, 0.995], "valid_max": [0.95, 0.999]},
        "train.optm_decay_type": {"valid_options": ["cosine", "none"], "option_weights": [0.7, 0.3]},
        "policy.lora.r": {"valid_min": 4, "valid_max": 64},
        "policy.lora.lora_alpha": {"valid_min": 128, "valid_max": 1024},
        "policy.lora.lora_dropout": {"valid_min": 0.03, "valid_max": 0.1},
        "custom.vision.fps": {"valid_min": 1, "valid_max": 2},
    },
)
```

**How to decide:** If the user says "run AutoML", "optimize hyperparameters", or "tune for me" without specifying details, use the **quick start** path. If they say "I want to choose the parameters", "show me what's available", "customize the search space", or mention specific params/ranges, use the **advanced** path.

**MANDATORY prompting for LLM-based algorithms (`llm`, `hybrid`, `autoresearch`):**

When the user requests an LLM-powered algorithm, you MUST explicitly ask for ALL THREE of the following before generating the script. Do not assume defaults — the code defaults are broken (endpoint 404s) and API keys are never pre-configured:

1. **`llm_endpoint`** — "What is your LLM endpoint?" (default: `https://inference-api.nvidia.com`)
2. **`llm_model`** — "Which LLM model?" (default: `meta/llama-3.1-70b-instruct`, or e.g. `gcp/google/gemini-3.1-pro-preview`)
3. **`llm_api_key`** — "What is your API key?" (check env vars first: `NVIDIA_API_KEY` / `AUTOML_LLM_API_KEY`)

If the user doesn't provide these, the LLM brain silently falls back to random sampling — wasting GPU budget on random configs instead of intelligent ones. There is no error message; the only clue is "LLM call failed... Falling back to random" in the logs.

**MANDATORY: Read the model skill before generating the script.**

AutoML runs training. Before generating any AutoML script, read `~/tao-skills-external/models/<network>/<network>.md`. The model skill `.md` contains all model-specific knowledge:

- **Training Requirements** — dataset type, formats, monitoring metric, required dataset URIs to prompt for, required user prompts (data format, num_classes, etc.), and mandatory `spec_overrides`. Prompt the user for every required field. Apply mandatory spec_overrides exactly.
- **Per-Action Dataset Requirements** — table mapping each action to its spec keys, data source, expected files, and whether the field is a list. Use this table to construct the correct data source `spec_overrides` for the requested action. If the model's Typical Spec Overrides mark data sources as "mandatory", construct them from this table and the user's dataset URIs.
- **Typical Spec Overrides** — per-action override suggestions (train, evaluate, export, inference, etc.) extracted from SDK notebooks. Use these as the starting point for `spec_overrides` and suggest them to the user. When overrides are marked "mandatory data sources", they MUST be included — the runner cannot auto-resolve them. Merge with any other mandatory overrides from Training Requirements.
- **AutoML / HPO Notes** — recommended hyperparameters, metric, direction, and AutoML-specific constraints.
- **Error Patterns** — common training failure modes that apply to AutoML recs too.

Do NOT hardcode model-specific knowledge in the AutoML script without reading the model skill first. Each network has different requirements.

**MANDATORY: Timestamped workspace folders.**

ALWAYS generate `workspace_path` with a timestamp suffix. Running the same script twice without a timestamp overwrites the previous experiment. Pattern:

```python
from datetime import datetime
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
workspace_path = f"./experiment_name/{TIMESTAMP}"
```

Do NOT use a flat path like `workspace_path="./my_experiment"`. The user should never have to manually delete old workspace folders.

**Best-practice on metric choice** (learned the hard way from real sweeps):

- `train_loss` is cheap but a footgun for small-dataset fine-tunes — the brain will find configs that memorize rather than generalize. Only use it for large-scale pretraining.
- `val_loss` (held-out slice of train) is cheap AND robust. Best default for fine-tuning. Cosmos-rl and similar frameworks emit this natively when `validation.enable=True` + `validation.freq_in_epoch=1`.
- Real task metric (accuracy / F1 / BLEU / mIoU) via the `eval_fn` hook is the most honest but also the most expensive per rec. Use when val_loss proxy isn't discriminating enough.

---

## Step 2: Select Algorithm

### Classical Algorithms

These require no external services — they use statistical/mathematical methods to pick hyperparameters.

| Algorithm | Use when | Typical budget | How it works |
|---|---|---|---|
| `bayesian` | **Default choice.** Small budgets, few parameters. | 5–20 recs | Builds a Gaussian Process model of metric vs. hyperparameters. Sequential — waits for each result before proposing the next, so it learns fast but can't parallelize. |
| `bfbo` | Alternative to bayesian with different acquisition function. | 5–20 recs | UCB-based Bayesian optimization with local penalization. Good when bayesian gets stuck. |
| `hyperband` | Large search spaces, many parameters. | 20–50+ recs | Trains many configs cheaply for a few epochs, keeps the best, trains longer. Requires `automl_max_epochs` and `automl_reduction_factor`. |
| `hyperband_es` | Hyperband + early stopping. | 20–50+ recs | Like hyperband but adds early-stop thresholds to halt clearly bad runs sooner. |
| `asha` | Async variant of hyperband, supports parallel execution. | 10–30 recs | Same successive-halving idea as hyperband, but trials run concurrently. Best when you have many GPUs. Uses `automl_max_concurrent`. |
| `bohb` | Best of both — Bayesian intelligence + Hyperband efficiency. | 15–40 recs | Combines KDE-based model (like Bayesian) with Hyperband's multi-fidelity scheduling. Good all-rounder for medium budgets. |
| `dehb` | Evolutionary + multi-fidelity. | 15–40 recs | Differential evolution mutations + hyperband scheduling. Good for complex search spaces with many interacting parameters. |
| `pbt` | Dynamic schedules — mutates hyperparameters during training. | population_size × generations | Population-Based Training. Starts N configs in parallel, periodically copies weights from winners and perturbs their hyperparameters. Best for long runs where hyperparameters should change over time (e.g. learning rate schedules). |

### LLM/Agentic Algorithms (NEW)

These use a large language model to reason about hyperparameter choices. They require an LLM endpoint (NVIDIA NIM, OpenAI, vLLM, Ollama, etc.) and the `openai` Python package.

| Algorithm | Use when | Typical budget | How it works |
|---|---|---|---|
| `llm` | Domain knowledge matters more than statistical rigor. | 5–20 recs | An LLM proposes hyperparameter configs based on the search space schema, experiment history, and its training knowledge. Falls back to random sampling on LLM failure. Sequential like bayesian. |
| `hybrid` | You want the LLM to orchestrate multi-phase optimization. | 10–50 recs | An LLM strategist plans optimization phases (e.g. "Phase 1: sweep LR with bayesian for 5 trials, Phase 2: sweep backbone with asha for 10 trials"). Each phase uses a classical sub-algorithm. Stops when the strategist detects diminishing returns. |
| `autoresearch` | Fully autonomous agent loop. | 10–50 recs | The most powerful mode. Combines: (1) RAP knowledge retrieval about the network, (2) LLM-proposed spec modifications, (3) training-free pre-screening of candidates, (4) multi-stage verification (pre-launch + post-result), (5) keep/discard reasoning. Automatically stops on budget exhaustion or consecutive failures. |

**Default to `bayesian` unless** the user specifically asks for something else, has a large GPU budget, or needs early-stopping on cheap intermediate metrics (ASHA / hyperband).

**Use `llm` / `hybrid` / `autoresearch` when** the user wants LLM-guided search, has an API key for NVIDIA NIM or OpenAI, and wants richer reasoning about why certain hyperparameters are chosen.

**Caveat on ASHA with large-checkpoint skills:** ASHA's whole point is running many configs for a cheap 1-epoch rung, then promoting survivors. When the per-epoch checkpoint save is expensive (e.g. cosmos-rl saves a 30+ GB full-model snapshot per epoch), the "cheap rung" stops being cheap. Stick with Bayesian on those workloads until the skill exposes a "skip intermediate checkpoints" knob.

---

## Step 3: Configure and Run

### Minimal Example

```python
import os
from datetime import datetime

# MANDATORY: Set skill bank path before importing the runner.
# Without this, runner.run() raises ValueError: No skill config found.
os.environ["TAO_SKILL_BANK_PATH"] = os.path.expanduser("~/tao-skills-external")

from tao_sdk.sdk import TaoExecutionSDK
from tao_automl.runner import AutoMLRunner

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

sdk = TaoExecutionSDK(creds_file=os.path.expanduser("~/tao-sdk/secrets.json"))
runner = AutoMLRunner(sdk)
result = runner.run(
    network_arch="cosmos-rl",
    train_dataset_uri="aws://bucket/data/my_dataset",
    automl_settings={
        "algorithm": "bayesian",
        "metric": "loss",
        "automl_max_recommendations": 5,
    },
    workspace_path=f"./automl_workspace/{TIMESTAMP}",  # timestamped to avoid collisions
)
```

### Full Example (all options)

```python
def my_eval(rec, train_job_id):
    """Optional post-training evaluator. Return a float (the real metric)
    or None to fall back to the log-based extractor."""
    # e.g. read a results.json uploaded by the container and compute accuracy
    ...
    return 0.71

result = runner.run(
    # --- Required ---
    network_arch="cosmos-rl",
    train_dataset_uri="aws://bucket/data/my_dataset",

    # --- Dataset + resources ---
    eval_dataset_uri="aws://bucket/data/eval",
    base_checkpoint="",
    image="nvcr.io/nvidia/tao/tao-toolkit:6.26.3-cosmos-rl",  # default: from skill
    backend_details={                                # platform-specific
        "backend_type": "lepton",
        "resource_shape": "gpu.h100-sxm",
    },

    # --- AutoML config ---
    automl_settings={
        "algorithm": "bayesian",
        "metric": "val_loss",
        "direction": "minimize",                     # explicit; optional
        "automl_max_recommendations": 10,
    },
    automl_hyperparameters=[                         # validated at launch (typos → error)
        "train.optm_lr",
        "policy.lora.r",
        "policy.lora.lora_alpha",
    ],
    custom_param_ranges={
        "train.optm_lr": {"valid_min": 1e-7, "valid_max": 1e-5},
    },

    # --- Per-rec spec overrides ---
    spec_overrides={                                 # validated — typos raise ValueError
        "train.epoch": 4,
        "validation.enable": True,                   # enable per-epoch val_loss for cosmos-rl
        "validation.freq_in_epoch": 1,
        "train.train_policy.dataset.test_size": 100, # 100 held-out val samples
    },

    # --- State + durability ---
    workspace_path=f"./my_experiment/{TIMESTAMP}",   # ALWAYS timestamp to avoid collisions
    resume=False,                                    # True → recovers in-flight jobs

    # --- WandB tracking (optional) ---
    wandb_config={
        "enabled": True,
        "project": "my-tao-experiments",
        "api_key": "your-wandb-api-key",             # or set WANDB_API_KEY env var
    },

    # --- Hooks (all optional, opt-in) ---
    metric_extractor=None,                           # custom log→metric parser
    eval_fn=my_eval,                                 # post-training real-metric eval
    on_recommendation=lambda r: print(f"launching rec {r.id}: {r.specs}"),
    on_result=lambda r, metric, status: print(f"rec {r.id} {status} → {metric}"),
)
```

### LLM-Powered Algorithm Example (DINO)

This is a complete, runnable example for DINO with LLM brain. It includes every mandatory piece that the agent MUST generate to avoid first-run errors:

```python
import os
from datetime import datetime

os.environ["TAO_SKILL_BANK_PATH"] = os.path.expanduser("~/tao-skills-external")

from tao_sdk.sdk import TaoExecutionSDK
from tao_automl.runner import AutoMLRunner

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
S3_BASE = "s3://bucket/data/coco_subset"
IMAGE_ARCHIVE = "images.tar.gz"

sdk = TaoExecutionSDK(creds_file=os.path.expanduser("~/tao-sdk/secrets.json"))
runner = AutoMLRunner(sdk)
result = runner.run(
    network_arch="dino",
    train_dataset_uri=S3_BASE,
    automl_settings={
        "algorithm": "llm",                          # or "hybrid" or "autoresearch"
        "metric": "kpi",
        "direction": "maximize",
        "automl_max_recommendations": 10,
        # LLM config — MUST pass llm_endpoint explicitly (code default 404s)
        "llm_endpoint": "https://inference-api.nvidia.com",
        "llm_model": "meta/llama-3.1-70b-instruct",
        "llm_api_key": "nvapi-...",                  # or set NVIDIA_API_KEY env var
    },
    automl_hyperparameters=[
        "train.optim.lr",
        "train.optim.weight_decay",
        "model.backbone",
        "model.num_queries",
        "model.dropout_ratio",
    ],
    custom_param_ranges={
        "train.optim.lr": {"valid_min": 1e-5, "valid_max": 5e-4},
        "model.num_queries": {"valid_min": 100, "valid_max": 900},
        "model.dropout_ratio": {"valid_min": 0.0, "valid_max": 0.3},
    },
    # spec_overrides from dino.md AutoML Requirements section
    # Use the remote archive path. The SDK extracts images.tar.gz and rewrites
    # the runtime spec to the extracted images folder.
    spec_overrides={
        "dataset.train_data_sources": [
            {"image_dir": f"{S3_BASE}/{IMAGE_ARCHIVE}", "json_file": f"{S3_BASE}/annotations.json"}
        ],
        "dataset.val_data_sources": [
            {"image_dir": f"{S3_BASE}/{IMAGE_ARCHIVE}", "json_file": f"{S3_BASE}/annotations.json"}
        ],
        "dataset.num_classes": 91,                   # >= max(category_id) + 1
        "train.num_epochs": 12,
        "train.validation_interval": 1,
    },
    workspace_path=f"./dino_llm_automl/{TIMESTAMP}",
)
```

**LLM endpoint configuration** (in order of precedence):
1. `automl_settings` keys: `llm_endpoint`, `llm_model`, `llm_api_key`
2. Environment variables: `AUTOML_LLM_ENDPOINT`, `AUTOML_LLM_MODEL`, `AUTOML_LLM_API_KEY`
3. Fallback env var for API key: `NVIDIA_API_KEY`
4. Defaults: NVIDIA NIM endpoint (`https://inference-api.nvidia.com`) with `meta/llama-3.1-70b-instruct`. **Note:** the code hardcodes `https://integrate.api.nvidia.com/v1` as the fallback which may 404 — always pass `llm_endpoint` explicitly or set `AUTOML_LLM_ENDPOINT`.

### Programmatic API (without runner)

For tighter control, use the `AutoML` class directly:

```python
from tao_automl import AutoML

automl = AutoML(
    workspace="/tmp/my_experiment",
    network="dino",
    train_specs=my_train_spec_dict,
    settings={
        "algorithm": "bayesian",
        "metric": "loss",
        "automl_max_recommendations": 10,
    },
    wandb_config={"enabled": True, "project": "my-project"},
)

while not automl.is_complete():
    recs = automl.next_recommendation()
    for rec in recs:
        metric_value = train_model(rec.specs)    # your training function
        automl.report_result(rec.id, metric_value)

automl.finish()   # close WandB run
print("Best:", automl.get_best().specs)
```

### `automl_settings` keys

| Key | Type | Default | Description |
|---|---|---|---|
| `algorithm` | str | **required** | `bayesian`, `hyperband`, `bohb`, `asha`, `bfbo`, `dehb`, `pbt`, `hyperband_es`, `llm`, `hybrid`, `autoresearch` |
| `metric` | str | `"loss"` | Metric name. The implicit rule for direction is "contains `'loss'` → minimize, else maximize". Override with `direction`. |
| `direction` | `"minimize"` \| `"maximize"` | inferred | Explicit direction. Required only when it disagrees with the implicit rule. The runner transparently inverts reported values so callers always see their metric in its original scale. |
| `automl_max_recommendations` | int | 20 | Max trials (bayesian, bfbo, llm) |
| `automl_max_epochs` | int | 27 | Epoch budget (hyperband, bohb, asha, dehb) |
| `automl_reduction_factor` | int | 3 | Halving factor (hyperband variants) |
| `automl_max_concurrent` | int | 4 | Max parallel configs (asha only) |
| `automl_population_size` | int | 10 | Population size (pbt only) |
| `automl_max_experiments` | int | 50 | Max experiments (autoresearch only) |
| `llm_endpoint` | str | NVIDIA NIM | OpenAI-compatible API endpoint (llm, hybrid, autoresearch) |
| `llm_model` | str | `meta/llama-3.1-70b-instruct` | LLM model name (llm, hybrid, autoresearch) |
| `llm_api_key` | str | from env | API key for the LLM endpoint |
| `research_program` | str | None | Free-text research directives for the autoresearch agent (e.g. "Focus on LoRA rank and learning rate interaction") |
| `automl_delete_intermediate_ckpt` | bool | False | Delete non-best checkpoints to save storage. Hyperband-family algorithms defer deletion until bracket completion for safety. |
| `override_automl_disabled_params` | bool | False | Include params whose schema `automl_enabled` is False. For advanced users who want to search over params the network author didn't flag for AutoML. |

### `kpi` metric resolution

When `metric="kpi"`, the controller resolves the actual metric key from the network config's `metrics.monitoring_metric` field. For example:
- cosmos-rl: `kpi` → `val/avg_loss` (minimized)
- dino: `kpi` → `val_mAP50` (maximized)
- segformer: `kpi` → (network-specific primary metric)

This is the recommended approach — use `metric="kpi"` and the system picks the right metric for the network. Only use explicit metric names when you need a non-default metric.

### `custom_param_ranges` format

Each entry can include:

| Field | Type | Description |
|---|---|---|
| `valid_min` | float/int/list | Min value. For `list_2` types (e.g. `optm_betas`), pass a list: `[0.9, 0.995]` |
| `valid_max` | float/int/list | Max value. Same list rules as min. |
| `valid_options` | list[str] | For categorical/ordered params: restrict to these values |
| `option_weights` | list[float] | Sampling weights for `valid_options`. Must match length. Higher weight = more likely to be sampled. |
| `disable_list` | bool | For params that can be float OR list (e.g. `train.optm_lr` in cosmos-rl): `True` keeps it as a single float for optimization, bypassing network list helpers. |

Example with all features:

```python
custom_param_ranges={
    "train.optm_lr": {"valid_min": 5e-6, "valid_max": 2e-4, "disable_list": True},
    "train.optm_decay_type": {
        "valid_options": ["cosine", "none"],
        "option_weights": [0.7, 0.3],
    },
    "train.optm_betas": {"valid_min": [0.9, 0.995], "valid_max": [0.95, 0.999]},
    "policy.lora.r": {"valid_min": 4, "valid_max": 64},
}
```

### Network-specific auto-exclusions

The search space builder has built-in safety rules:

- **cosmos-rl LoRA exclusion:** If `policy.lora.*` keys are absent from the training spec (i.e. full fine-tuning, not LoRA), all LoRA parameters are auto-excluded from the search space. No action needed — just be aware that LoRA params won't appear if LoRA is disabled in `spec_overrides`.
- **classification_pyt + hyperband:** If `model.head.type` is in the search space and algorithm is `hyperband`, AutoML raises an error. Use `bayesian` instead for head-type search.

### LLM Analyzer (server-side range narrowing)

The controller supports automatic range narrowing via the LLM analyzer. Enable via environment variables before launching:

```python
os.environ["AUTOML_LLM_ANALYZER_ENABLED"] = "true"
os.environ["AUTOML_LLM_ANALYZER_INTERVAL"] = "5"        # analyze every 5 completed recs
os.environ["AUTOML_LLM_ANALYZER_NARROW_RANGES"] = "true" # auto-tighten custom_param_ranges
```

When enabled, after every N completed experiments the analyzer reviews patterns, assesses convergence, and optionally narrows search ranges to focus on promising regions. This happens server-side and persists the narrowed ranges.

### `spec_overrides` common keys

(Every key you pass is validated against the skill's spec schema. Typos that look like existing keys raise `ValueError` with a suggestion; genuinely-new keys are accepted with a warning. This catches `save_freq_in_epochs` → `save_freq_in_epoch` style silent failures.)

| Key | Effect |
|---|---|
| `train.epoch` | Number of training epochs |
| `train.train_batch_per_replica` | Batch size per GPU |
| `train.optm_lr` | Base learning rate |
| `train.ckpt.save_freq_in_epoch` | Save checkpoint every N epochs (default for cosmos-rl is 10 — override to `train.epoch` if running fewer) |
| `train.ckpt.max_keep` | Max checkpoints to keep on disk |
| `validation.enable` / `validation.freq_in_epoch` | Enable per-epoch validation for `val_loss` metric |
| `train.train_policy.dataset.test_size` | Validation split size (int = absolute samples; float = ratio) |
| `policy.model_max_length` | Context window size (40960 for video VLMs) |
| `policy.parallelism.dp_shard_size` | Must equal number of GPUs |

---

## WandB Experiment Tracking

AutoML optionally integrates with [Weights & Biases](https://wandb.ai) to track all experiments in a single dashboard.

### Setup

```bash
pip install wandb
# or: pip install nvidia-tao-automl[wandb]
```

### How it works

When `wandb_config={"enabled": True}` is passed:

1. The controller creates a WandB **run** named `automl_brain` in the specified project.
2. All recommendations are grouped under a WandB **group** (e.g. `automl_abc123`) so parent + child training runs appear together in the dashboard.
3. After every result, a **WandB table** (`automl_experiments`) is logged containing:
   - `experiment_id`, `job_id`, `status`, metric value, `best_epoch_number`
   - All varying hyperparameter values
4. Call `automl.finish()` (or let `runner.run()` complete) to finalize the WandB run.

### Minimal WandB setup

```python
# Option 1: via config dict
result = runner.run(
    ...,
    wandb_config={
        "enabled": True,
        "project": "tao-hpo",
        "api_key": "your-key",  # or set WANDB_API_KEY env var
    },
)

# Option 2: environment variable (simpler)
# export WANDB_API_KEY=your-key
result = runner.run(
    ...,
    wandb_config={"enabled": True, "project": "tao-hpo"},
)
```

### Dashboard features

Once tracking is active, you can:
- **Compare all trials** side-by-side in the WandB table view
- **Sort by metric** to find the best config instantly
- **Group by hyperparameter** to see which values correlate with good results
- **Link to child training runs** if the compute backend also logs to WandB (group name is available via `automl.wandb_group`)

---

## LLM/Agentic Features Deep Dive

### Natural Language Configuration

Don't know which algorithm or parameters to use? The `NLConfigGenerator` translates plain English into a valid AutoML configuration:

```python
from tao_automl.brain.nl_config import NLConfigGenerator

generator = NLConfigGenerator()   # uses NVIDIA NIM by default
config = generator.generate_config(
    user_prompt="I want to maximize detection accuracy on a small custom dataset with 500 images",
    network="dino",
    available_parameters=param_records,  # from generate_hyperparams_to_search()
    hardware_info="2x A100 80GB",
)
# config = {
#   "automl_algorithm": "bayesian",
#   "automl_hyperparameters": ["train.optim.lr", "train.optim.weight_decay", ...],
#   "algorithm_specific_params": {"automl_max_recommendations": 15},
#   "metric": "kpi",
#   "reasoning": "Small dataset + limited budget → bayesian for sample efficiency..."
# }
```

### LLM Analyzer (works with ANY algorithm)

The `LLMAnalyzer` can be used alongside any classical algorithm to provide periodic analysis of experiment results:

```python
from tao_automl.brain.llm_analyzer import LLMAnalyzer

analyzer = LLMAnalyzer(analysis_interval=5, narrow_ranges=True)

# After every 5 completed experiments, call:
analysis = analyzer.analyze(
    experiments=experiment_history,
    parameters=param_records,
    network="dino",
    metric_name="kpi",
    metric_direction="maximize",
    best_metric=0.85,
)
# analysis = {
#   "patterns": ["LR > 0.01 always causes divergence"],
#   "convergence_assessment": "improving",
#   "recommendations": ["Try weight_decay in [0.01, 0.05]"],
#   "suggested_ranges": {"train.optim.lr": {"min": 0.0005, "max": 0.005, ...}},
# }
```

When `narrow_ranges=True`, the analyzer suggests tighter search bounds based on observed patterns. These can be applied to dynamically focus the search.

### Autoresearch Agent Components

The `autoresearch` algorithm integrates five AutoML-Agent concepts:

| Component | What it does | When it runs |
|---|---|---|
| **KnowledgeRetriever** (RAP) | Retrieves built-in tuning knowledge for the network (e.g. "DINO works best with LR 1e-4 to 5e-4") and optionally web-searched papers/benchmarks | Once at initialization |
| **SpecPrescreener** | LLM predicts which of N candidate configs are worth running, WITHOUT training. Saves GPU budget by filtering unlikely-to-improve configs. | Before each trial — proposes 3 candidates, pre-screens to pick the best 1 |
| **MultiStageVerifier** | Pre-launch: validates proposed changes won't crash/OOM. Post-result: checks metrics are plausible (not NaN, not anomalous). | Before launch + after result |
| **ExperimentTracker** | Tracks full history with keep/discard decisions and reasoning | After each result |
| **LLMAnalyzer** | Periodic pattern detection, convergence assessment, and optional range narrowing | Every N completed experiments |

### Research Programs

For complex multi-phase optimization, define a research program:

```python
from tao_automl.brain.research_program import ResearchProgram, ResearchPhase

program = ResearchProgram(
    objective="Maximize detection mAP on custom dataset",
    network="dino",
    phases=[
        ResearchPhase(
            name="LR sweep",
            algorithm="bayesian",
            parameters=["train.optim.lr", "train.optim.weight_decay"],
            trials=8,
        ),
        ResearchPhase(
            name="Architecture search",
            algorithm="asha",
            parameters=["model.backbone", "model.num_queries"],
            trials=15,
            carry_forward="best",   # best LR values carry into this phase
        ),
    ],
)

# Validate before running
issues = program.validate(
    available_parameters=["train.optim.lr", "train.optim.weight_decay", "model.backbone", "model.num_queries"],
    available_algorithms=["bayesian", "asha"],
)
```

---

## Advanced hooks (opt-in)

Both hooks are optional. If neither is provided, the runner uses its built-in log regex extractor.

### `metric_extractor(logs: str, metric_name: str) → float | None`

Called on every poll of the training container's logs. Return the most recent/final metric value seen, or `None` if the metric isn't yet present.

Use it when:
- Your container emits the metric in a non-standard log format the built-in regex misses.
- You want to parse values from log lines instead of using the generic patterns.
- Your metric needs derivation from multiple log fields.

```python
import re

def extract_bleu(logs: str, metric_name: str):
    m = re.search(r"BLEU-4:\s*([0-9.]+)", logs)
    return float(m.group(1)) if m else None

runner.run(..., metric_extractor=extract_bleu)
```

Exceptions raised inside the extractor are caught and logged; the runner continues polling.

### `eval_fn(rec, train_job_id: str) → float | None`

Called once after a rec's training job reaches a terminal state, before the result is reported to the brain. Whatever it returns **overrides** any value captured by `metric_extractor` and becomes what the brain optimizes on.

Use it when:
- The real task metric lives outside the training logs (results.json, external benchmark, merge-then-inference pipeline).
- You want a true-test-metric sweep without building surrounding plumbing yourself.
- Per-rec cost is acceptable relative to `metric_extractor`.

```python
def eval_on_held_out(rec, train_job_id):
    # 1. Find the LoRA adapter the training job saved on S3
    adapter = f"s3://{BUCKET}/results/{train_job_id}/train_output_dir/..."
    # 2. Merge into base + run inference on the eval set
    merged = merge_lora(adapter, base="nvidia/Cosmos-Reason2-8B")
    acc, _ = run_classification_eval(merged, eval_dataset_uri)
    return acc

runner.run(
    ...,
    automl_settings={"metric": "accuracy", "direction": "maximize", ...},
    eval_fn=eval_on_held_out,
)
```

Exceptions from `eval_fn` are caught and logged — the runner falls back to the log-extracted metric for that rec.

---

## Step 4: Monitor Progress

`runner.run()` blocks until all recommendations complete. Use callbacks to report progress to the user:

```python
def on_rec(rec):
    print(f"Rec {rec.id}: trying lr={rec.specs.get('train.optm_lr')}, "
          f"lora_r={rec.specs.get('policy.lora.r')}")

def on_result(rec, metric, status):
    print(f"Rec {rec.id}: {status}, metric={metric}")

result = runner.run(..., on_recommendation=on_rec, on_result=on_result)
```

Each rec takes 10–90 minutes depending on model size, dataset, epochs, and checkpoint save cost. Don't assume failure during long uploads.

### Resume after interruption

If the orchestrator dies mid-run (network timeout, machine sleep, Ctrl-C), re-run with `resume=True` and the **full suffixed path** (including the `run_<timestamp>` directory):

```python
result = runner.run(
    ...,
    workspace_path="./my_experiment/run_20260423_183015",   # full suffixed path
    resume=True,
)
```

When `resume=True`, the runner does NOT append a new timestamp suffix — it reuses the path as-is.

Behaviour on resume:
1. **Brain state** is reloaded from `<workspace>/.automl/*` — all completed rec results are already registered.
2. **Any in-flight jobs** recorded in `<workspace>/active_jobs.json` (persisted after each submission) are polled to terminal, their metrics extracted, and reported to the brain — *before* the main propose-new-rec loop starts. No duplicate submissions; no leaked GPU work from the previous orchestrator.
3. After recovery, the loop continues normally until `automl.is_complete()`.

---

## Step 5: Interpret Results

The result is a plain dict:

```python
{
    "best": {
        "rec_id": 4,
        "specs": {"train.optm_lr": 2.83e-7, "policy.lora.r": 8, "policy.lora.lora_alpha": 512, ...},
        "metric_value": 0.7077,
    },
    "progress": {
        "completed": 8, "total": 8,
        "best_metric": 0.7077, "best_rec_id": 4,
        "algorithm": "bayesian",
    },
    "history": [
        {"rec_id": 0, "metric": 0.6308, "status": "success"},
        {"rec_id": 1, "metric": 0.7077, "status": "success"},
        ...
    ],
}
```

Metric values in `best` and `history` are always in the original scale the user provided — direction inversion (if any) is undone before the dict is returned.

### How to report to the user

1. **Best config** — show the winning hyperparameters and metric value.
2. **Comparison table** — rank all recs by metric, highlight the best.
3. **Insights** — call out what the optimizer learned (e.g. "high-α regime consistently wins; LR in 1-3e-7 range is robust").
4. **WandB link** — if tracking was enabled, provide the dashboard URL.
5. **Next steps** — suggest:
   - More recs (re-run with `resume=True` + higher `automl_max_recommendations`).
   - Train longer with the best config using `sdk.create_job(specs=result["best"]["specs"])`.
   - Run a downstream evaluation on the best checkpoint.
   - Export / merge / deploy the best model.

### If all recs failed

Check common issues:
- **Dataset path wrong** — verify the URI points to a directory with the files the skill expects (e.g. `annotations.json` + `videos.tar.gz` for cosmos-rl).
- **Validation never fires** — if `metric="val_loss"` but `validation.freq_in_epoch > train.epoch`, no val-loss line ever appears in the log. Set `validation.freq_in_epoch=1` and `validation.enable=True`.
- **Checkpoint never saves** — if `train.ckpt.save_freq_in_epoch > train.epoch`, no checkpoint is saved; downstream merge/eval fails.
- **Model or data download timeout** — the first run downloads ~15 GB from HuggingFace; subsequent runs use Lustre cache.
- **OOM** — reduce `train.train_policy.mini_batch` or `custom.vision.total_pixels` via `spec_overrides`.
- **Silent tarball corruption** — a cached, truncated `*.tar.gz` on shared Lustre will silently poison every future run. Symptoms: cryptic data-loader errors like `moov atom not found` / `KeyError: 'video_fps'`. Fix: delete the cached Lustre path and let it re-download.
- **LLM endpoint unreachable** (llm/hybrid/autoresearch only) — the brain falls back to random sampling. Check `AUTOML_LLM_ENDPOINT` and `AUTOML_LLM_API_KEY`. Verify with: `curl -s $AUTOML_LLM_ENDPOINT/models -H "Authorization: Bearer $AUTOML_LLM_API_KEY"`.

---

## Network-Specific Notes

### cosmos-rl

Dataset layout: `annotations.json` + `videos.tar.gz` (or `images.tar.gz`) per split. The runner auto-applies tarball extraction + spec-rewrite inside the container so annotation-relative video paths resolve after download.

**Recommended spec_overrides for fine-tuning:**

```python
spec_overrides={
    "train.epoch": 4,                                 # 3-5 is typical for small datasets
    "train.ckpt.save_freq_in_epoch": 4,               # save exactly at the final epoch
    "train.ckpt.max_keep": 1,
    "validation.enable": True,                        # emit "[SFT] Validation loss: X" lines
    "validation.freq_in_epoch": 1,                    # every epoch — cheap
    "train.train_policy.dataset.test_size": 100,      # 100 held-out samples
}
```

With these overrides, `automl_settings={"metric": "val_loss", ...}` gives the brain a clean, cheap signal and the built-in extractor picks up cosmos-rl's `[SFT] Validation loss: X for train step N/M, epoch E` lines automatically.

**Common hyperparameter search space that works well for small-dataset LoRA:**

```python
automl_hyperparameters=[
    "train.optm_lr",
    "policy.lora.r",
    "policy.lora.lora_alpha",
    "policy.lora.lora_dropout",
    "train.optm_decay_type",
]
custom_param_ranges={
    "policy.lora.r": {"valid_min": 4, "valid_max": 32},
    "policy.lora.lora_dropout": {"valid_min": 0.0, "valid_max": 0.05},
}
```

These constraints avoid the overfitting traps (very-low `r` + very-high `α` + high `dropout` = aggressive memorization on small data).

**LoRA eval pitfall:** Cosmos-rl's direct-LoRA evaluation path assumes a 4-shard base-model filename pattern that the public HF repo doesn't use. Work around by pre-merging: use `peft.merge_and_unload()` to produce a single merged model, then eval that with `model.enable_lora=False`.

### dino / deformable_detr / grounding_dino / rtdetr

Object detection networks using COCO-format data. Read `~/tao-skills-external/models/dino/dino.md` — the **"Training Requirements"** section has required datasets, user prompts, and mandatory spec_overrides. The **"AutoML / HPO Notes"** section has recommended hyperparameters and metric.

### segformer / classification_pyt / other vision

No special handling needed. The runner reads base specs from `SkillBank.get_default_specs(network_arch, "train")`. Pass `automl_hyperparameters` to control which params are searched, or leave `None` to use all `automl_enabled` params from the schema. Read the model's `<network>.md` for any model-specific requirements.

---

## Common Pitfalls

1. **`TAO_SKILL_BANK_PATH` not set.** The #1 first-run error. The runner raises `ValueError: No skill config found for '<network>'`. Fix: `os.environ["TAO_SKILL_BANK_PATH"] = os.path.expanduser("~/tao-skills-external")` before importing the runner. ALWAYS include this in generated scripts.
2. **Wrong LLM endpoint (404).** The code hardcodes `https://integrate.api.nvidia.com/v1` as the default, which returns 404. The correct endpoint is `https://inference-api.nvidia.com`. ALWAYS pass `llm_endpoint` explicitly in `automl_settings`. The LLM brain silently falls back to random sampling on 404, so you won't see a crash — just useless random configs.
3. **Model-specific training failures (data format, missing datasets, invalid params).** Each network has unique training requirements. ALWAYS read `~/tao-skills-external/models/<network>/<network>.md` — the "Training Requirements" and "Error Patterns" sections document model-specific failure modes that apply to AutoML recs too.
4. **Workspace path collisions.** Running the same script twice overwrites the previous experiment. Always include a timestamp: `workspace_path=f"./automl_workspace/{TIMESTAMP}"` where `TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")`.
5. **Using `train_loss` for small-dataset fine-tuning.** The brain will find configs that memorize. Switch to `val_loss` or provide `eval_fn`.
6. **Implicit direction trap.** `metric="perplexity"` → brain maximizes (wrong). Set `direction="minimize"` explicitly.
7. **Spec-override typos.** `save_freq_in_epochs` (plural) used to silently do nothing; now raises `ValueError` with suggestion. If you see that error, it's the fix working.
8. **Orchestrator dies mid-sweep.** Relaunch with the same `workspace_path` and `resume=True`. In-flight jobs are recovered from `active_jobs.json`.
9. **"Rec never reports a metric" with `val_loss`.** Check that `validation.enable=True` and `validation.freq_in_epoch <= train.epoch`. Without this, the container never emits a validation-loss line.
10. **Parallel Bayesian arms.** Bayesian is inherently sequential. If you want parallelism, use `asha`. If you use multiple `AutoMLRunner` instances, give each its own `TaoExecutionSDK(state_file=...)` to avoid SQLite write races.
11. **LLM brain returning random configs.** If every LLM recommendation looks random, the LLM endpoint is probably failing silently. Check the logs for "LLM call failed" warnings. Verify your API key and endpoint are correct. Common cause: using the wrong endpoint URL (see pitfall #2).
12. **`openai` package not installed.** The `llm`, `hybrid`, and `autoresearch` algorithms require the `openai` Python package. Install with `pip install openai` or `pip install nvidia-tao-automl[llm]`.
13. **WandB not logging.** Ensure `wandb_config={"enabled": True}` is passed and either `api_key` is in the config or `WANDB_API_KEY` is set in the environment. Check logs for "WandB initialized" confirmation.
14. **`No default train specs found` for a network.** The skill bank model directory is missing `defaults-train.json` or `references/spec_template_train.yaml`. Create one from the network's experiment spec in `tao-pytorch/nvidia_tao_pytorch/cv/<network>/experiment_specs/train.yaml`.
15. **`conda run` buffers output.** When running AutoML via `conda run -n tao_sdk python script.py`, all output is buffered until completion. Use `PYTHONUNBUFFERED=1 ~/miniconda3/envs/tao_sdk/bin/python script.py` for real-time output.

---

## Querying Experiment Status

Use `query_status()` to check experiment progress from a separate process — no need to read JSON files or parse logs.

```python
from tao_automl import query_status

status = query_status("./my_experiment")

# Progress summary
p = status["progress"]
print(f"{p['completed']}/{p['total']} recs done, "
      f"{p['succeeded']} succeeded, {p['failed']} failed")

# Best config
if status["best"]:
    print(f"Best: rec {status['best']['rec_id']}, "
          f"metric={status['best']['metric_value']}, "
          f"specs={status['best']['specs']}")

# Per-rec details
for rec in status["recommendations"]:
    print(f"  Rec {rec['rec_id']}: {rec['status']} "
          f"metric={rec['metric_value']} specs={rec['specs']}")

# In-flight jobs
for job in status["active_jobs"]:
    print(f"  Active: rec {job['rec_id']} job {job['job_id']}")
```

The function reads from the persisted state store (`<workspace>/.automl/`) and `active_jobs.json`. It is safe to call while the runner is active — no locking conflicts.

The `AutoML` class also exposes `get_status()` for in-process queries:

```python
automl = AutoML(workspace=..., ...)
status = automl.get_status()
```

---

## Quick Reference: Algorithm Decision Tree

```
Is your budget tiny (≤10 recs)?
  YES → bayesian
  NO  ↓

Do you have an LLM API key and want AI-guided search?
  YES → Do you want full autonomy? → autoresearch
        Just LLM proposals?        → llm
        LLM orchestrating phases?  → hybrid
  NO  ↓

Do you need parallel execution?
  YES → asha (or bohb for smarter sampling)
  NO  ↓

Is your search space large (10+ parameters)?
  YES → hyperband or dehb
  NO  ↓

Do hyperparameters need to change during training (schedules)?
  YES → pbt
  NO  → bayesian (safe default)
```

---

## Example Conversations

### User: "Optimize cosmos-rl training on my ITS dataset, using val_loss"

```
Agent: I'll run Bayesian HPO driven by val_loss (cheaper + more overfit-robust than train loss). I need:
1. Your dataset URI (e.g. aws://bucket/data/its_subset)
2. How many trials? (default: 10, each ~60 min on H100)

User: aws://nvcf-storage-handling/data/cosmos_rl_its/, 10 recs

Agent: Running. Enabling per-epoch validation on a 100-sample held-out slice so val_loss actually appears in the logs. I'll merge + eval the best config once the sweep converges.
[executes runner.run(metric="val_loss", spec_overrides={"validation.*": ..., ...})]
```

### User: "I want the real accuracy, not val_loss"

```
Agent: For a real eval metric, I'll use the eval_fn hook — each rec runs train → merge LoRA → inference → accuracy. ~2x cost per rec, so I'd start with 8 recs instead of 10.
[executes runner.run(metric="accuracy", direction="maximize", eval_fn=my_eval, ...)]
```

### User: "Use the LLM to figure out the best hyperparameters"

```
Agent: I'll use the LLM algorithm — it reasons about your network architecture and learns from each experiment.
I need three things for the LLM brain:
1. LLM endpoint URL (default: https://inference-api.nvidia.com)
2. LLM model name (default: meta/llama-3.1-70b-instruct, or e.g. gcp/google/gemini-3.1-pro-preview)
3. API key for the endpoint (or set NVIDIA_API_KEY env var)

User: endpoint "https://inference-api.nvidia.com", model "gcp/google/gemini-3.1-pro-preview", key "sk-abc123"

Agent: Running LLM-guided search with 10 recs. The LLM will explain its reasoning for each config choice in the logs.
[executes runner.run(automl_settings={
    "algorithm": "llm",
    "llm_endpoint": "https://inference-api.nvidia.com",
    "llm_model": "gcp/google/gemini-3.1-pro-preview",
    "llm_api_key": "sk-abc123",
    ...
})]
```

### User: "Run a fully autonomous search — let the AI figure everything out"

```
Agent: I'll use the autoresearch algorithm. It will:
1. Retrieve tuning knowledge for your network
2. Propose modifications, pre-screen them, verify before launch
3. Make keep/discard decisions with reasoning after each experiment
4. Automatically stop when it detects diminishing returns

I'll cap it at 30 experiments. I need:
- Dataset URI
- LLM endpoint (default: https://inference-api.nvidia.com)
- LLM model (default: meta/llama-3.1-70b-instruct)
- LLM API key (or set NVIDIA_API_KEY env var)

User: dataset aws://bucket/data, endpoint https://inference-api.nvidia.com, model gcp/google/gemini-3.1-pro-preview, key sk-abc123

[executes runner.run(automl_settings={
    "algorithm": "autoresearch", "automl_max_experiments": 30,
    "llm_endpoint": "https://inference-api.nvidia.com",
    "llm_model": "gcp/google/gemini-3.1-pro-preview",
    "llm_api_key": "sk-abc123",
    ...
})]
```

### User: "Resume the experiment from yesterday"

```
Agent: Resuming. The runner will poll any jobs that were in flight when we stopped, report their results, then continue.
[executes runner.run(..., workspace_path="./my_experiment", resume=True)]
```

### User: "Switch to ASHA with 30 trials and track in WandB"

```
Agent: Switching to ASHA — parallel + early-stopping. Enabling WandB tracking so you can see all trials in a dashboard. Worth flagging: on cosmos-rl the per-epoch checkpoint upload is expensive, so ASHA's cheap-rung advantage is muted. For 30 trials it's still faster than Bayesian wall-clock, just not 10x faster.
[executes runner.run(
    automl_settings={"algorithm": "asha", "automl_max_epochs": 9, ...},
    wandb_config={"enabled": True, "project": "tao-hpo"},
)]
```

### User: "I don't know what parameters to tune for DINO"

```
Agent: I can generate a configuration from your goal description. What are you trying to achieve?

User: Maximize detection accuracy on my small custom dataset with 200 images on 1 A100.

Agent: Based on your constraints, I recommend:
- Algorithm: bayesian (sample-efficient for small budgets)
- Parameters: train.optim.lr, train.optim.weight_decay, model.num_queries
- Budget: 12 recs
- LR range narrowed to [1e-5, 5e-4] (standard for DINO fine-tuning)
[uses NLConfigGenerator, then executes runner.run with the generated config]
```
