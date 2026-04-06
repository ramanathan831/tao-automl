---
name: TAO AutoML
description: Run hyperparameter optimization for TAO networks using the AutoMLRunner. Handles algorithm selection, experiment configuration, job execution on Lepton/DGX, and result interpretation.
dependencies:
  - bash
  - python3
---

# TAO AutoML Skill

Run automated hyperparameter optimization (HPO) for any TAO network. The agent uses `AutoMLRunner` — a single interface that manages the full loop: generate hyperparameter recommendations, launch training jobs on Lepton/DGX, extract metrics, and feed results back to the optimizer.

## Prerequisites

Before running AutoML:

1. **SDK credentials**: `secrets.json` must exist in the working directory with platform credentials
2. **Dataset**: Training data accessible from the compute backend. URI format depends on platform:
   - Lepton/DGX Cloud: `aws://bucket/path` (S3-compatible)
   - Azure: `azure://container/path`
   - Local/Docker: local filesystem path
3. **nvidia-tao-automl installed**: `pip install nvidia-tao-automl` (or `pip install nvidia-tao-sdk[automl]`)

Verify setup:
```bash
python3 -c "from tao_automl.runner import AutoMLRunner; print('OK')"
```

---

## Step 1: Parse User Intent

Extract from the user's request:

| Field | Required | Example | How to get it |
|---|---|---|---|
| `network_arch` | Yes | `"cosmos-rl"`, `"dino"`, `"clip"` | User states the model |
| `train_dataset_uri` | Yes | `"aws://bucket/data/subset"` | User provides the URI (S3, Azure, or local path) |
| `metric` | No | `"loss"` (default) | Ask if unclear — loss for regression, accuracy for classification |
| `algorithm` | No | `"bayesian"` (default) | See algorithm guide below |
| `max_recommendations` | No | 5–20 | Ask budget — each rec is one full training run |

If any required field is missing, ask the user. Do NOT guess dataset paths.

---

## Step 2: Select Algorithm

| Algorithm | Use when | Typical budget |
|---|---|---|
| `bayesian` | **Default choice.** Small budgets, few parameters. Sequential — learns from each result before generating the next. | 5–20 recs |
| `hyperband` | Large search spaces, many parameters. Trains many configs cheaply, keeps the best, trains longer. | 20–50+ recs |
| `asha` | Need parallel execution. Like Hyperband but async — doesn't wait for all configs in a rung. | 10–30 recs |
| `bohb` | Best of both — Bayesian intelligence + Hyperband efficiency. | 15–40 recs |
| `pbt` | Dynamic schedules — mutates hyperparameters during training. Good for long runs. | population_size * generations |

**Default to `bayesian` unless the user specifically asks for something else or has a large GPU budget.**

---

## Step 3: Configure and Run

### Minimal Example

```python
from tao_sdk import TaoExecutionSDK
from tao_automl.runner import AutoMLRunner

sdk = TaoExecutionSDK(creds_file="secrets.json")
runner = AutoMLRunner(sdk)
result = runner.run(
    network_arch="cosmos-rl",
    train_dataset_uri="aws://bucket/data/my_dataset",
    automl_settings={
        "algorithm": "bayesian",
        "metric": "loss",
        "automl_max_recommendations": 5,
    },
)
```

### Full Example (with all options)

```python
result = runner.run(
    # Required
    network_arch="cosmos-rl",
    train_dataset_uri="aws://bucket/data/my_dataset",

    # Job config (platform is handled by the SDK, not the runner)
    eval_dataset_uri="aws://bucket/data/eval",       # optional eval set
    base_checkpoint="",                                # pretrained checkpoint
    image="nvcr.io/nvidia/tao/tao-toolkit:6.26.3-cosmos-rl",  # container image

    # AutoML config
    automl_settings={
        "algorithm": "bayesian",
        "metric": "loss",
        "automl_max_recommendations": 10,
    },
    automl_hyperparameters=[                           # which params to search
        "train.optm_lr",
        "policy.lora.r",
        "policy.lora.lora_alpha",
    ],
    custom_param_ranges={                              # narrow the search space
        "train.optm_lr": {"valid_min": 1e-6, "valid_max": 1e-4},
    },

    # State & control
    spec_overrides={"train.epoch": 3},                 # override base spec values
    resume=False,                                      # True to resume interrupted run
    workspace_path="./my_experiment",                  # where AutoML state is saved
)
```

### Parameter Reference

**`automl_settings` keys:**

| Key | Type | Default | Description |
|---|---|---|---|
| `algorithm` | str | **required** | `bayesian`, `hyperband`, `bohb`, `asha`, `bfbo`, `dehb`, `pbt`, `hyperband_es` |
| `metric` | str | `"loss"` | Metric to optimize. Contains `"loss"` → lower is better, else higher is better |
| `automl_max_recommendations` | int | 20 | Max trials (bayesian, bfbo) |
| `automl_max_epochs` | int | 27 | Epoch budget (hyperband, bohb, asha, dehb) |
| `automl_reduction_factor` | int | 3 | Halving factor (hyperband variants) |
| `automl_max_concurrent` | int | 4 | Max parallel configs (asha only) |

**`spec_overrides` common keys:**

| Key | Effect |
|---|---|
| `train.epoch` | Number of training epochs |
| `train.train_batch_per_replica` | Batch size per GPU |
| `train.optm_lr` | Base learning rate |
| `policy.model_name_or_path` | Model to fine-tune |
| `policy.model_max_length` | Context window size (40960 for video VLMs) |
| `policy.parallelism.dp_shard_size` | Must equal number of GPUs |

---

## Step 4: Monitor Progress

`runner.run()` blocks until all recommendations complete. Use callbacks to report progress to the user:

```python
def on_rec(rec):
    print(f"Rec {rec.id}: trying lr={rec.specs.get('train.optm_lr')}, "
          f"lora_r={rec.specs.get('policy.lora.r')}")

def on_result(rec, metric, status):
    print(f"Rec {rec.id}: {status}, loss={metric}")

result = runner.run(
    ...,
    on_recommendation=on_rec,
    on_result=on_result,
)
```

**Without callbacks**, periodically tell the user the run is in progress. Each rec takes 5–60 minutes depending on model size, dataset, and epochs. Do NOT assume failure if it takes a while.

**If the run is interrupted** (network timeout, machine sleep), re-run with `resume=True`:

```python
result = runner.run(
    ...,
    workspace_path="./my_experiment",  # same path as before
    resume=True,                        # picks up from last completed rec
)
```

---

## Step 5: Interpret Results

The result is a plain dict:

```python
{
    "best": {
        "rec_id": 0,
        "specs": {"train.optm_lr": 1.42e-7, "policy.lora.r": 4, ...},
        "metric_value": 0.799
    },
    "progress": {
        "completed": 5,
        "total": 5,
        "best_metric": 0.799,
        "best_rec_id": 0,
        "algorithm": "bayesian"
    },
    "history": [
        {"rec_id": 0, "metric": 0.799, "status": "success"},
        {"rec_id": 1, "metric": 4.196, "status": "success"},
        {"rec_id": 2, "metric": 0.0,   "status": "failure"},
        ...
    ]
}
```

### How to report to the user:

1. **Best config** — show the winning hyperparameters and metric value
2. **Comparison table** — rank all recs by metric, highlight the best
3. **Insights** — what the optimizer learned (e.g., "lower learning rate performed better")
4. **Next steps** — suggest:
   - Run more recs if budget allows (re-run with `resume=True`)
   - Train longer with the best config using `sdk.create_job(specs=result["best"]["specs"])`
   - Run evaluation with the best checkpoint
   - Export the model

### If all recs failed:

Check for common issues:
- **Dataset path wrong** — verify the URI points to a directory with `annotations.json` + `images.tar.gz` or `videos.tar.gz`
- **Batch size mismatch** — use `spec_overrides={"train.train_batch_per_replica": 4}`
- **Model download timeout** — the first run downloads ~15GB from HuggingFace; subsequent runs use cache
- **OOM** — reduce batch size or model_max_length via spec_overrides

---

## Network-Specific Notes

### cosmos-rl

The runner auto-applies these fixes for cosmos-rl:
- `train_batch_per_replica` bumped to >= `mini_batch` (batch assertion fix)
- `model_max_length` set to 40960 (video token overflow prevention)
- `dp_shard_size` set to 1 (single GPU)
- `validation.enable` set to true (container bug workaround)

Dataset must contain `annotations.json` + `videos.tar.gz` (or `images.tar.gz`).

Searchable hyperparameters (auto-discovered from schema):
- `train.epoch`, `train.optm_lr`, `train.optm_decay_type`
- `policy.lora.r`, `policy.lora.lora_alpha`, `policy.lora.lora_dropout`
- `custom.vision.fps`

### Other networks (dino, clip, etc.)

No special fixes needed. The runner uses `sdk.get_default_specs()` for base config. Pass `automl_hyperparameters` to control which params are searched, or leave as None to use all `automl_enabled` params from the schema.

---

## Example Conversations

### User: "Optimize cosmos-rl training on my ITS dataset"

```
Agent: I'll run Bayesian HPO on cosmos-rl. I need:
1. Your dataset URI (e.g. aws://bucket/data/its_subset)
2. How many trials? (default: 5, each takes ~10 min on H100)

User: aws://nvcf-storage-handling/data/cosmos_rl_wts_train_subset, try 3 recs

Agent: Running 3 Bayesian recommendations on cosmos-rl...
[executes runner.run()]

Agent: AutoML complete! Results:
| Rec | lr | lora_r | alpha | Loss |
|-----|-----|--------|-------|------|
| 0   | 1.4e-7 | 4 | 512 | 0.799 |
| 1   | 5.7e-8 | 128 | 512 | 4.196 |
| 2   | 2.2e-7 | 64 | 4 | failed |

Best config: Rec 0 (loss=0.799) — low lr (1.4e-7) with small LoRA rank (4)
and high alpha (512). Want me to run eval with this config?
```

### User: "Resume the experiment from yesterday"

```
Agent: I'll resume from the saved state.
[executes runner.run(..., workspace_path="./my_experiment", resume=True)]
```

### User: "Try hyperband with 20 trials instead of bayesian"

```
Agent: Switching to Hyperband with 20 trials.
[executes runner.run(..., automl_settings={
    "algorithm": "hyperband",
    "metric": "loss",
    "automl_max_epochs": 27,
    "automl_reduction_factor": 3,
})]
```
