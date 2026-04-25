---
name: tao-automl
description: >-
  Run hyperparameter optimization (HPO) for NVIDIA TAO networks using AutoMLRunner.
  Handles algorithm selection (bayesian, hyperband, asha, bohb, llm, hybrid, autoresearch),
  WandB experiment tracking, job execution on Lepton/DGX/Slurm, result interpretation,
  and per-rec custom evaluation hooks. Use when the user mentions TAO AutoML, hyperparameter
  optimization, HPO, automl, automl_settings, AutoMLRunner, tao_automl, bayesian search,
  hyperband, ASHA, LLM-guided search, autoresearch, or wants to tune training hyperparameters
  for any TAO network.
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
   - Lepton / DGX Cloud: `s3://bucket/path` (S3-compatible; do not generate `aws://...`)
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
   ├── applications/         # workflow configs
   ├── models/               # per-network skill packages
   │   ├── <network>/
   │   │   ├── config.json           # actions, data_sources, container image
   │   │   ├── defaults-train.json   # default training spec (REQUIRED by AutoML)
   │   │   └── <network>.md
   │   ├── <another-network>/
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
   PYTHONPATH=~/tao-sdk:~/tao-automl/src PYTHONUNBUFFERED=1 ~/miniconda3/envs/tao_sdk/bin/python my_script.py
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

TAO AutoML automates the "try different hyperparameter values → train → compare results → repeat" cycle. Instead of manually tweaking training settings, you tell AutoML:

- **What network** to train (`network_arch`)
- **Which hyperparameters** to search over (from the model skill and schema)
- **What metric** to optimize (from the model skill or user request)
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
| `network_arch` | Yes | `"<network_arch>"` | User states the model |
| `train_dataset_uri` | Yes | `"s3://bucket/data/subset"` | User provides the URI |
| `metric` | No | `"<metric_name>"` | Use the model skill recommendation or ask if unclear. Do not choose model-specific metrics from this AutoML skill. |
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

When `automl_hyperparameters=None`, the runner automatically discovers all params marked `automl_enabled=True` in the network's JSON schema. Each network has its own set; the dataclass definitions in `tao-automl/src/tao_automl/config/<network>/` and the model skill's **AutoML / HPO Notes** are the source of truth.

```python
# Quick start — all defaults
result = runner.run(
    network_arch=network_arch,
    train_dataset_uri=S3_TRAIN,
    automl_settings={
        "algorithm": "hybrid",
        "metric": metric,
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
    automl_hyperparameters=selected_param_names,
    custom_param_ranges={
        "<param_name>": {"valid_min": min_value, "valid_max": max_value},
        "<categorical_param>": {"valid_options": ["option_a", "option_b"], "option_weights": [0.7, 0.3]},
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

**MANDATORY: No model-specific constants in this AutoML skill.**

The AutoML skill must not define model-specific hyperparameter names, metric names, dataset layouts, archive names, class-count rules, spec override keys, container images, checkpoint quirks, or custom metric regexes. Those belong in `~/tao-skills-external/models/<network>/<network>.md`, especially the **Training Requirements**, **Typical Spec Overrides**, **AutoML / HPO Notes**, and **Error Patterns** sections. This skill may describe how to read and apply those sections, but not the concrete per-model values.

**MANDATORY: Timestamped workspace folders.**

ALWAYS generate `workspace_path` with a timestamp suffix. Running the same script twice without a timestamp overwrites the previous experiment. Pattern:

```python
from datetime import datetime
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
workspace_path = f"./experiment_name/{TIMESTAMP}"
```

Do NOT use a flat path like `workspace_path="./my_experiment"`. The user should never have to manually delete old workspace folders.

**MANDATORY: Fresh runner per new AutoML request.**

Every new user request to run AutoML MUST create a new runner script and launch a new AutoML job, even if an older runner script for the same network/algorithm already exists. Existing runner files and logs may be read only as references for dataset URIs, credentials patterns, and proven fixes; do not reuse them as the execution target for a new request.

Use a unique timestamp in the new runner filename, log filename, PID filename, SDK `state_file`, and `workspace_path`. Derive path components from the requested `network_arch` and `algorithm`; do not hardcode any model or algorithm name unless it is the actual requested value.

```python
import re

def slug(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_").lower()

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_NAME = f"{slug(network_arch)}_{slug(algorithm)}"
runner_path = f"automl_runs/run_{RUN_NAME}_{TIMESTAMP}.py"
log_path = f"automl_runs/{RUN_NAME}_{TIMESTAMP}.log"
pid_path = f"automl_runs/{RUN_NAME}_{TIMESTAMP}.pid"
state_file = f"tao_session_state_{RUN_NAME}_{TIMESTAMP}.json"
workspace_path = f"./automl_runs/{RUN_NAME}/{TIMESTAMP}"
```

Only resume an existing runner/workspace when the user explicitly asks to resume, continue, recover, or inspect an existing experiment. If the user says "run automl" or asks for a new AutoML run, treat it as a fresh job.

**Best-practice on metric choice**:

- Training loss is cheap, but can overfit on small fine-tuning datasets. Prefer the model skill's recommended validation or task metric when available.
- If the model skill recommends a validation proxy, also apply the model skill's required validation-related `spec_overrides` so the metric is actually emitted.
- A real task metric via `eval_fn` is often the most honest but adds per-rec cost. Use it when the model skill says log-based metrics are insufficient or the user explicitly wants downstream evaluation.

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
| `hybrid` | You want the LLM to orchestrate multi-phase optimization. | 10–50 recs | An LLM strategist plans optimization phases over model-skill parameters. Each phase uses a classical sub-algorithm. Stops when the strategist detects diminishing returns. |
| `autoresearch` | Fully autonomous agent loop. | 10–50 recs | The most powerful mode. Combines: (1) RAP knowledge retrieval about the network, (2) LLM-proposed spec modifications, (3) training-free pre-screening of candidates, (4) multi-stage verification (pre-launch + post-result), (5) keep/discard reasoning. Automatically stops on budget exhaustion or consecutive failures. |

**Default to `bayesian` unless** the user specifically asks for something else, has a large GPU budget, or needs early-stopping on cheap intermediate metrics (ASHA / hyperband).

**Use `llm` / `hybrid` / `autoresearch` when** the user wants LLM-guided search, has an API key for NVIDIA NIM or OpenAI, and wants richer reasoning about why certain hyperparameters are chosen.

**Caveat on ASHA with expensive checkpoints:** ASHA's whole point is running many configs cheaply for early rungs, then promoting survivors. If the model skill warns that checkpoints, validation, or startup cost dominate short trials, prefer the model skill's recommended algorithm instead of assuming ASHA will be cheaper.

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
    network_arch=network_arch,
    train_dataset_uri=train_dataset_uri,
    automl_settings={
        "algorithm": algorithm,
        "metric": metric,
        "automl_max_recommendations": max_recommendations,
    },
    workspace_path=f"./automl_workspace/{TIMESTAMP}",  # timestamped to avoid collisions
)
```

### Full Example (all options)

```python
def my_eval(rec, train_job_id):
    """Optional post-training evaluator. Return a float (the real metric)
    or None to fall back to the log-based extractor."""
    # e.g. read a results file uploaded by the container and compute the requested metric
    ...
    return 0.71

result = runner.run(
    # --- Required ---
    network_arch=network_arch,
    train_dataset_uri=train_dataset_uri,

    # --- Dataset + resources ---
    eval_dataset_uri=eval_dataset_uri,
    base_checkpoint="",
    image=image,                                      # only set if the model skill or user requires it
    backend_details={                                # platform-specific
        "backend_type": "lepton",
        "resource_shape": "gpu.h100-sxm",
    },

    # --- AutoML config ---
    automl_settings={
        "algorithm": algorithm,
        "metric": metric,
        "direction": direction,                       # explicit when needed
        "automl_max_recommendations": max_recommendations,
    },
    automl_hyperparameters=automl_hyperparameters,    # from model skill / schema
    custom_param_ranges=custom_param_ranges,          # from model skill / user constraints

    # --- Per-rec spec overrides ---
    spec_overrides=spec_overrides,                    # mandatory model-specific overrides from model skill

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

### LLM-Powered Algorithm Example

For `llm`, `hybrid`, or `autoresearch`, use the same generic runner shape as above, plus the required LLM endpoint, model, and key in `automl_settings`. All model-specific hyperparameters, metric extractors, and `spec_overrides` must still come from the model skill.

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
    network=network_arch,
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
| `research_program` | str | None | Free-text research directives for the autoresearch agent |
| `automl_delete_intermediate_ckpt` | bool | False | Delete non-best checkpoints to save storage. Hyperband-family algorithms defer deletion until bracket completion for safety. |
| `override_automl_disabled_params` | bool | False | Include params whose schema `automl_enabled` is False. For advanced users who want to search over params the network author didn't flag for AutoML. |

### `kpi` metric resolution

When `metric="kpi"`, the controller resolves the actual metric key from the network config's `metrics.monitoring_metric` field. Whether `kpi` is appropriate, and whether a custom `metric_extractor` is needed, is model-specific. Follow the model skill's **AutoML / HPO Notes**.

### `custom_param_ranges` format

Each entry can include:

| Field | Type | Description |
|---|---|---|
| `valid_min` | float/int/list | Min value. For list-valued parameters, pass the list shape required by the schema. |
| `valid_max` | float/int/list | Max value. Same list rules as min. |
| `valid_options` | list[str] | For categorical/ordered params: restrict to these values |
| `option_weights` | list[float] | Sampling weights for `valid_options`. Must match length. Higher weight = more likely to be sampled. |
| `disable_list` | bool | For params that can be float OR list: `True` keeps it as a single float for optimization, bypassing network list helpers. Use only when supported by the schema/model skill. |

Example with all features:

```python
custom_param_ranges={
    "<float_param>": {"valid_min": min_value, "valid_max": max_value, "disable_list": True},
    "<categorical_param>": {
        "valid_options": ["option_a", "option_b"],
        "option_weights": [0.7, 0.3],
    },
    "<list_param>": {"valid_min": [min_a, min_b], "valid_max": [max_a, max_b]},
}
```

### Model-specific search-space rules

Some networks have built-in search-space exclusions or algorithm restrictions. Do not document them here; read the model skill's **AutoML / HPO Notes** and let schema validation report unsupported combinations.

### LLM Analyzer (server-side range narrowing)

The controller supports automatic range narrowing via the LLM analyzer. Enable via environment variables before launching:

```python
os.environ["AUTOML_LLM_ANALYZER_ENABLED"] = "true"
os.environ["AUTOML_LLM_ANALYZER_INTERVAL"] = "5"        # analyze every 5 completed recs
os.environ["AUTOML_LLM_ANALYZER_NARROW_RANGES"] = "true" # auto-tighten custom_param_ranges
```

When enabled, after every N completed experiments the analyzer reviews patterns, assesses convergence, and optionally narrows search ranges to focus on promising regions. This happens server-side and persists the narrowed ranges.

### `spec_overrides`

`spec_overrides` keys are model-specific. Read the model skill's **Training Requirements**, **Per-Action Dataset Requirements**, and **Typical Spec Overrides** sections, then pass only the keys required or recommended there. Do not infer override keys from examples in this AutoML skill.

Every key you pass is validated against the skill's spec schema. Typos that look like existing keys raise `ValueError` with a suggestion; genuinely-new keys are accepted with a warning.

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
    user_prompt=user_goal,
    network=network_arch,
    available_parameters=param_records,  # from generate_hyperparams_to_search()
    hardware_info=hardware_info,
)
# config = {
#   "automl_algorithm": "bayesian",
#   "automl_hyperparameters": ["<param_from_model_schema>", ...],
#   "algorithm_specific_params": {"automl_max_recommendations": 15},
#   "metric": "<metric_from_model_skill_or_user_request>",
#   "reasoning": "..."
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
    network=network_arch,
    metric_name=metric,
    metric_direction=direction,
    best_metric=best_metric,
)
# analysis = {
#   "patterns": ["..."],
#   "convergence_assessment": "improving",
#   "recommendations": ["..."],
#   "suggested_ranges": {"<param_name>": {"min": ..., "max": ...}},
# }
```

When `narrow_ranges=True`, the analyzer suggests tighter search bounds based on observed patterns. These can be applied to dynamically focus the search.

### Autoresearch Agent Components

The `autoresearch` algorithm integrates five AutoML-Agent concepts:

| Component | What it does | When it runs |
|---|---|---|
| **KnowledgeRetriever** (RAP) | Retrieves built-in tuning knowledge for the requested network and optionally web-searched papers/benchmarks | Once at initialization |
| **SpecPrescreener** | LLM predicts which of N candidate configs are worth running, WITHOUT training. Saves GPU budget by filtering unlikely-to-improve configs. | Before each trial — proposes 3 candidates, pre-screens to pick the best 1 |
| **MultiStageVerifier** | Pre-launch: validates proposed changes won't crash/OOM. Post-result: checks metrics are plausible (not NaN, not anomalous). | Before launch + after result |
| **ExperimentTracker** | Tracks full history with keep/discard decisions and reasoning | After each result |
| **LLMAnalyzer** | Periodic pattern detection, convergence assessment, and optional range narrowing | Every N completed experiments |

### Research Programs

For complex multi-phase optimization, define a research program:

```python
from tao_automl.brain.research_program import ResearchProgram, ResearchPhase

program = ResearchProgram(
    objective=objective,
    network=network_arch,
    phases=[
        ResearchPhase(
            name="Phase 1",
            algorithm="bayesian",
            parameters=["<param_from_model_schema>", "..."],
            trials=8,
        ),
        ResearchPhase(
            name="Phase 2",
            algorithm="asha",
            parameters=["<another_param_from_model_schema>", "..."],
            trials=15,
            carry_forward="best",   # best values carry into this phase
        ),
    ],
)

# Validate before running
issues = program.validate(
    available_parameters=available_parameters,
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

def extract_custom_metric(logs: str, metric_name: str):
    m = re.search(rf"{re.escape(metric_name)}:\s*([0-9.]+)", logs)
    return float(m.group(1)) if m else None

runner.run(..., metric_extractor=extract_custom_metric)
```

Exceptions raised inside the extractor are caught and logged; the runner continues polling.

### `eval_fn(rec, train_job_id: str) → float | None`

Called once after a rec's training job reaches a terminal state, before the result is reported to the brain. Whatever it returns **overrides** any value captured by `metric_extractor` and becomes what the brain optimizes on.

Use it when:
- The real task metric lives outside the training logs.
- You want a true-test-metric sweep without building surrounding plumbing yourself.
- Per-rec cost is acceptable relative to `metric_extractor`.

```python
def eval_on_held_out(rec, train_job_id):
    # Implement the model-specific evaluation flow documented in the model skill.
    metric_value = run_model_specific_eval(rec, train_job_id)
    return metric_value

runner.run(
    ...,
    automl_settings={"metric": task_metric, "direction": direction, ...},
    eval_fn=eval_on_held_out,
)
```

Exceptions from `eval_fn` are caught and logged — the runner falls back to the log-extracted metric for that rec.

---

## Step 4: Monitor Progress

`runner.run()` blocks until all recommendations complete. Use callbacks to report progress to the user:

```python
def on_rec(rec):
    print(f"Rec {rec.id}: trying {rec.specs}")

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
        "specs": {"<param_name>": "<value>", "...": "..."},
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
3. **Insights** — call out what the optimizer learned from the requested parameters and metric.
4. **WandB link** — if tracking was enabled, provide the dashboard URL.
5. **Next steps** — suggest:
   - More recs (re-run with `resume=True` + higher `automl_max_recommendations`).
   - Train longer with the best config using `sdk.create_job(specs=result["best"]["specs"])`.
   - Run a downstream evaluation on the best checkpoint.
   - Run the model skill's recommended export/deploy workflow for the best model.

### If all recs failed

Check common issues:
- **Dataset path wrong** — verify the URI points to the layout required by the model skill.
- **Metric never appears** — verify the model skill's required metric-related overrides and custom extractor are present.
- **Checkpoint or eval artifact missing** — verify the model skill's checkpoint/export/eval requirements.
- **Model or data download timeout** — inspect backend logs and model-skill error patterns.
- **OOM** — reduce the model-specific batch, resolution, sequence length, or memory-heavy knobs recommended by the model skill.
- **Cached data corruption** — inspect the model skill's dataset/cache error patterns and clear only the affected cache path if documented.
- **LLM endpoint unreachable** (llm/hybrid/autoresearch only) — the brain falls back to random sampling. Check `AUTOML_LLM_ENDPOINT` and `AUTOML_LLM_API_KEY`. Verify with: `curl -s $AUTOML_LLM_ENDPOINT/models -H "Authorization: Bearer $AUTOML_LLM_API_KEY"`.

---

## Model-Specific Notes

Model-specific notes do not belong in this AutoML skill. For every requested `network_arch`, read `~/tao-skills-external/models/<network>/<network>.md` and use its **Training Requirements**, **Per-Action Dataset Requirements**, **Typical Spec Overrides**, **AutoML / HPO Notes**, and **Error Patterns** sections as the source of truth.

---

## Common Pitfalls

1. **`TAO_SKILL_BANK_PATH` not set.** The #1 first-run error. The runner raises `ValueError: No skill config found for '<network>'`. Fix: `os.environ["TAO_SKILL_BANK_PATH"] = os.path.expanduser("~/tao-skills-external")` before importing the runner. ALWAYS include this in generated scripts.
2. **Wrong LLM endpoint (404).** The code hardcodes `https://integrate.api.nvidia.com/v1` as the default, which returns 404. The correct endpoint is `https://inference-api.nvidia.com`. ALWAYS pass `llm_endpoint` explicitly in `automl_settings`. The LLM brain silently falls back to random sampling on 404, so you won't see a crash — just useless random configs.
3. **Model-specific training failures (data format, missing datasets, invalid params).** Each network has unique training requirements. ALWAYS read `~/tao-skills-external/models/<network>/<network>.md` — the "Training Requirements" and "Error Patterns" sections document model-specific failure modes that apply to AutoML recs too.
4. **Workspace path collisions.** Running the same script twice overwrites the previous experiment. Always include a timestamp: `workspace_path=f"./automl_workspace/{TIMESTAMP}"` where `TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")`.
5. **Using a weak proxy metric.** The brain can optimize a metric that does not reflect real task quality. Use the metric recommended by the model skill or provide `eval_fn`.
6. **Implicit direction trap.** If the metric name does not imply the desired direction, set `direction` explicitly.
7. **Spec-override typos.** `save_freq_in_epochs` (plural) used to silently do nothing; now raises `ValueError` with suggestion. If you see that error, it's the fix working.
8. **Orchestrator dies mid-sweep.** Relaunch with the same `workspace_path` and `resume=True`. In-flight jobs are recovered from `active_jobs.json`.
9. **Rec never reports a metric.** Check the model skill's metric-emission requirements and custom extractor guidance.
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

### User: "Optimize `<network_arch>` training on my dataset"

```
Agent: I'll read the model skill for `<network_arch>` first, then generate a fresh timestamped runner using its required datasets, spec overrides, metric, and recommended search space. I need:
1. The dataset URI or URIs required by that model skill
2. How many trials? (default: 10)

User: s3://bucket/data/train, 10 recs

Agent: Running a fresh job with a new runner, log, state file, and workspace. The model-specific metric and overrides come from `<network>.md`.
[executes runner.run(...)]
```

### User: "I want the real task metric, not the default proxy"

```
Agent: For a real task metric, I'll use the eval_fn hook described by the model skill. This adds per-rec cost, so I’ll adjust the budget if needed.
[executes runner.run(metric=task_metric, direction=direction, eval_fn=model_specific_eval, ...)]
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

User: dataset s3://bucket/data, endpoint https://inference-api.nvidia.com, model gcp/google/gemini-3.1-pro-preview, key sk-abc123

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
Agent: Switching to ASHA — parallel + early-stopping. I’ll first check the model skill for any warning that short trials are dominated by startup, validation, or checkpoint cost, then generate a fresh runner with WandB tracking.
[executes runner.run(
    automl_settings={"algorithm": "asha", "automl_max_epochs": 9, ...},
    wandb_config={"enabled": True, "project": "tao-hpo"},
)]
```

### User: "I don't know what parameters to tune for `<network_arch>`"

```
Agent: I can generate a configuration from your goal description. What are you trying to achieve?

User: Maximize my task metric on a small custom dataset with 1 GPU.

Agent: Based on your constraints, I recommend:
- Algorithm: bayesian (sample-efficient for small budgets)
- Parameters: from the model skill's AutoML / HPO Notes and the generated schema
- Budget: 12 recs
- Ranges: from the model skill and user constraints
[uses NLConfigGenerator, then executes runner.run with the generated config]
```
