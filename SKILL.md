---
name: TAO AutoML
description: Run hyperparameter optimization for TAO networks using the AutoMLRunner. Handles algorithm selection, experiment configuration, job execution on Lepton/DGX, result interpretation, and per-rec custom evaluation hooks.
dependencies:
  - bash
  - python3
---

# TAO AutoML Skill

Run automated hyperparameter optimization (HPO) for any TAO network. The agent uses `AutoMLRunner` — a single interface that manages the full loop: generate hyperparameter recommendations, launch training jobs on Lepton/DGX, extract metrics, and feed results back to the optimizer.

The runner is platform-agnostic: platform selection (Lepton, Slurm, K8s) and GPU allocation are handled by the SDK the runner is given, not by the runner itself.

## Prerequisites

Before running AutoML:

1. **SDK credentials**: `secrets.json` must exist in the working directory with the platform credentials required by the SDK.
2. **Dataset**: Training data accessible from the compute backend. URI format depends on the SDK's platform:
   - Lepton / DGX Cloud: `aws://bucket/path` (S3-compatible)
   - Azure: `azure://container/path`
   - Local / Docker: local filesystem path
3. **Skill bank available**: the runner reads skill config + default specs via `SkillBank`. Point it at the bank with:
   ```bash
   export TAO_SKILL_BANK_PATH=/path/to/tao-skills-external
   ```
4. **`nvidia-tao-automl` installed** (editable dev install pulls `tao-sdk` from git):
   ```bash
   pip install -e ".[dev]" --extra-index-url https://pypi.nvidia.com
   ```

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
| `train_dataset_uri` | Yes | `"aws://bucket/data/subset"` | User provides the URI |
| `metric` | No | `"loss"` (default) | Ask if unclear — `loss` / `val_loss` for generative tasks, `accuracy` / `mIoU` / custom for classification / segmentation |
| `direction` | No | `"minimize"` or `"maximize"` | **Only needed if your metric name doesn't contain `"loss"` AND you want to minimize, or contains `"loss"` AND you want to maximize.** Otherwise the implicit "contains 'loss' → minimize, else maximize" rule applies. |
| `algorithm` | No | `"bayesian"` (default) | See algorithm guide below |
| `max_recommendations` | No | 5–20 | Ask budget — each rec is one full training run |

If any required field is missing, ask the user. Do NOT guess dataset paths.

**Best-practice on metric choice** (learned the hard way from real sweeps):

- `train_loss` is cheap but a footgun for small-dataset fine-tunes — the brain will find configs that memorize rather than generalize. Only use it for large-scale pretraining.
- `val_loss` (held-out slice of train) is cheap AND robust. Best default for fine-tuning. Cosmos-rl and similar frameworks emit this natively when `validation.enable=True` + `validation.freq_in_epoch=1`.
- Real task metric (accuracy / F1 / BLEU / mIoU) via the `eval_fn` hook is the most honest but also the most expensive per rec. Use when val_loss proxy isn't discriminating enough.

---

## Step 2: Select Algorithm

| Algorithm | Use when | Typical budget |
|---|---|---|
| `bayesian` | **Default choice.** Small budgets, few parameters. Sequential — learns from each result before generating the next. | 5–20 recs |
| `hyperband` | Large search spaces, many parameters. Trains many configs cheaply, keeps the best, trains longer. | 20–50+ recs |
| `asha` | Async variant of hyperband, supports parallel execution. | 10–30 recs |
| `bohb` | Best of both — Bayesian intelligence + Hyperband efficiency. | 15–40 recs |
| `pbt` | Dynamic schedules — mutates hyperparameters during training. Good for long runs. | population_size × generations |

**Default to `bayesian` unless** the user specifically asks for something else, has a large GPU budget, or needs early-stopping on cheap intermediate metrics (ASHA / hyperband).

**Caveat on ASHA with large-checkpoint skills:** ASHA's whole point is running many configs for a cheap 1-epoch rung, then promoting survivors. When the per-epoch checkpoint save is expensive (e.g. cosmos-rl saves a 30+ GB full-model snapshot per epoch), the "cheap rung" stops being cheap. Stick with Bayesian on those workloads until the skill exposes a "skip intermediate checkpoints" knob.

---

## Step 3: Configure and Run

### Minimal Example

```python
from tao_sdk.sdk import TaoExecutionSDK
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
    workspace_path="./my_experiment",
    resume=False,                                    # True → recovers in-flight jobs

    # --- Hooks (all optional, opt-in) ---
    metric_extractor=None,                           # custom log→metric parser
    eval_fn=my_eval,                                 # post-training real-metric eval
    on_recommendation=lambda r: print(f"launching rec {r.id}: {r.specs}"),
    on_result=lambda r, metric, status: print(f"rec {r.id} {status} → {metric}"),
)
```

### `automl_settings` keys

| Key | Type | Default | Description |
|---|---|---|---|
| `algorithm` | str | **required** | `bayesian`, `hyperband`, `bohb`, `asha`, `bfbo`, `dehb`, `pbt`, `hyperband_es` |
| `metric` | str | `"loss"` | Metric name. The implicit rule for direction is "contains `'loss'` → minimize, else maximize". Override with `direction`. |
| `direction` | `"minimize"` \| `"maximize"` | inferred | Explicit direction. Required only when it disagrees with the implicit rule. The runner transparently inverts reported values so callers always see their metric in its original scale. |
| `automl_max_recommendations` | int | 20 | Max trials (bayesian, bfbo) |
| `automl_max_epochs` | int | 27 | Epoch budget (hyperband, bohb, asha, dehb) |
| `automl_reduction_factor` | int | 3 | Halving factor (hyperband variants) |
| `automl_max_concurrent` | int | 4 | Max parallel configs (asha only) |

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

### Advanced hooks (opt-in)

Both hooks are optional. If neither is provided, the runner uses its built-in log regex extractor.

#### `metric_extractor(logs: str, metric_name: str) → float | None`

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

#### `eval_fn(rec, train_job_id: str) → float | None`

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

If the orchestrator dies mid-run (network timeout, machine sleep, Ctrl-C), re-run with `resume=True`:

```python
result = runner.run(
    ...,
    workspace_path="./my_experiment",   # same path as before
    resume=True,
)
```

Behaviour on resume:
1. **Brain state** is reloaded from `<workspace>/state_store/*` — all completed rec results are already registered.
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
4. **Next steps** — suggest:
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

### Other networks (dino, clip, segformer, etc.)

No special handling needed. The runner reads base specs from `SkillBank.get_default_specs(network_arch, "train")`. Pass `automl_hyperparameters` to control which params are searched, or leave `None` to use all `automl_enabled` params from the schema.

**Segformer-like skills** expose real val metrics (`val_mIoU`) in the training loop — use them directly as the AutoML metric. For generative skills (LLM/VLM) only `val_loss` is cheap; real task accuracy requires `eval_fn`.

---

## Common Pitfalls

1. **Using `train_loss` for small-dataset fine-tuning.** The brain will find configs that memorize. Switch to `val_loss` or provide `eval_fn`.
2. **Implicit direction trap.** `metric="perplexity"` → brain maximizes (wrong). Set `direction="minimize"` explicitly.
3. **Spec-override typos.** `save_freq_in_epochs` (plural) used to silently do nothing; now raises `ValueError` with suggestion. If you see that error, it's the fix working.
4. **Orchestrator dies mid-sweep.** Relaunch with the same `workspace_path` and `resume=True`. In-flight jobs are recovered from `active_jobs.json`.
5. **"Rec never reports a metric" with `val_loss`.** Check that `validation.enable=True` and `validation.freq_in_epoch <= train.epoch`. Without this, the container never emits a validation-loss line.
6. **Parallel Bayesian arms.** Bayesian is inherently sequential. If you want parallelism, use `asha`. If you use multiple `AutoMLRunner` instances, give each its own `TaoExecutionSDK(state_file=...)` to avoid SQLite write races.

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

### User: "Resume the experiment from yesterday"

```
Agent: Resuming. The runner will poll any jobs that were in flight when we stopped, report their results, then continue.
[executes runner.run(..., workspace_path="./my_experiment", resume=True)]
```

### User: "Switch to ASHA with 30 trials"

```
Agent: Switching to ASHA — parallel + early-stopping. Worth flagging: on cosmos-rl the per-epoch checkpoint upload is expensive, so ASHA's cheap-rung advantage is muted. For 30 trials it's still faster than Bayesian wall-clock, just not 10x faster.
[executes runner.run(..., automl_settings={"algorithm": "asha", "automl_max_epochs": 9, ...})]
```
