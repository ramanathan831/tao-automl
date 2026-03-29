# TAO AutoML SDK — Architecture & Interface Report

**Package**: `nvidia-tao-automl` v0.1.0
**Date**: 2026-03-28
**Authors**: Tejas Anand, Arif Ahmed
**Status**: Review Draft — requesting team feedback on interface definitions

---

## 1. What This Is

A standalone Python wheel that provides hyperparameter optimization (HPO) for TAO networks. It decides **what hyperparameters to try next** and **learns from results** — but does NOT execute training jobs. The caller (TAO SDK, LLM-generated script, or any Python code) handles job execution.

**Key principle**: The AutoML wheel is the brain. The Executor SDK is the hands. They never import each other.

### What It Replaces

The AutoML functionality previously lived inside the FTMS (Finetuning Microservice) at `nvidia_tao_core/microservices/automl/`. It was tightly coupled to:
- MongoDB (state persistence)
- Flask API (job creation)
- FTMS workflow engine (job orchestration)
- Execution handlers (Docker/K8s/Lepton)

This wheel extracts the core HPO logic with **zero infrastructure dependencies**.

### Dependencies

```
numpy, scikit-learn, scipy, pandas, omegaconf, requests
```

No: Flask, MongoDB, Docker, torch, kubernetes, leptonai.

---

## 2. Package Structure

```
src/tao_automl/
├── __init__.py                  # AutoML class — top-level public API
├── types.py                     # Recommendation, JobStates, AutoMLContext
│
├── brain/                       # HPO algorithms (the core intelligence)
│   ├── base.py                  # AutoMLAlgorithmBase — value generation + constraints
│   ├── bayesian.py              # Gaussian Process + Expected Improvement
│   ├── hyperband.py             # Successive Halving brackets
│   ├── bohb.py                  # Bayesian Optimization + Hyperband
│   ├── asha.py                  # Async Successive Halving (parallel)
│   ├── bfbo.py                  # Bayesian Function-level BO
│   ├── dehb.py                  # Distributed Evolutionary Hyperband
│   ├── pbt.py                   # Population-Based Training
│   ├── hyperband_es.py          # Hyperband + Early Stopping
│   ├── factory.py               # BrainFactory + AlgorithmParams
│   └── network_utils/           # Per-network parameter logic
│       ├── common.py            # Dispatch to network-specific handlers
│       ├── cosmos_rl.py         # Cosmos-RL: LoRA constraints, multi-part LR
│       └── dino.py              # DINO-specific logic
│
├── controller/                  # Optimization loop manager
│   └── controller.py            # next_recommendation, report_result, is_complete
│
├── state/                       # Persistence layer
│   └── state_store.py           # JSON files + fcntl.flock concurrency
│
├── search_space/                # Schema → searchable parameters
│   └── params.py                # generate_hyperparams_to_search()
│
├── schema/                      # Network config → JSON schema
│   ├── generate_schema.py
│   ├── dataclass2json_converter.py
│   └── enum_constants.py
│
├── config/                      # Network dataclass definitions (41 networks)
│   ├── utils/types.py           # Field type helpers (INT_FIELD, FLOAT_FIELD, etc.)
│   ├── common/                  # Shared base configs
│   ├── cosmos-rl/               # Cosmos Reason model config
│   ├── dino/                    # DINO config
│   └── .../                     # 38 more networks
│
└── utils/                       # Pure utility functions
    ├── math_utils.py            # fix_input_dimension, clamp_value, get_valid_range
    ├── spec_utils.py            # get_flatten_specs, flatten_properties
    ├── network_constants.py     # backbone_mapper, gpu_mapper
    └── automl_helper.py         # automl_list_helper tables
```

---

## 3. How It Works — Data Flow

### Initialization

```
AutoML(workspace, network, train_specs, settings)
  │
  ├─ 1. StateStore(workspace)              → creates .automl/ directory
  ├─ 2. AutoMLContext(id, network, metric)  → lightweight session context
  ├─ 3. save_job_specs(train_specs)         → persist base config for brain to read
  ├─ 4. generate_hyperparams_to_search()    → schema → searchable params list
  │     │
  │     ├─ generate_schema(network, "train") → JSON schema from config dataclass
  │     ├─ flatten_properties()              → flat dict of all params with metadata
  │     └─ filter to automl_enabled=TRUE     → e.g., 7 params for cosmos-rl
  │
  ├─ 5. AlgorithmParams.from_dict(settings) → algorithm-specific settings
  ├─ 6. BrainFactory.create_brain()         → instantiate the right algorithm
  └─ 7. Controller(brain, context, state)   → wire everything together
```

### Optimization Loop

```
while not automl.is_complete():
    recs = automl.next_recommendation()     ← A
    for rec in recs:
        metric = caller_runs_training(rec)  ← B (not our code)
        automl.report_result(rec.id, metric) ← C
```

**Step A — next_recommendation():**
```
Controller.next_recommendation()
  → brain.generate_recommendations(history)
    → [Bayesian] If first call: random [0,1]^d vector
    → [Bayesian] If subsequent: GP.fit(Xs, ys) → optimize_ei() → best [0,1]^d
    → For each dimension: map [0,1] → actual param value
      ├─ float: suggestion * (v_max - v_min) + v_min
      ├─ int: round to nearest, apply math_cond (^ 2 = power of 2)
      ├─ categorical: map to index in valid_options
      └─ network-specific post-processing (e.g., cosmos-rl multi-part LR)
  → Wrap as Recommendation objects with IDs
  → Persist to state_store
```

**Step C — report_result():**
```
Controller.report_result(rec_id, metric)
  → Acquire global file lock (fcntl.flock)
  → Update rec.result, rec.status in history
  → brain.save_state() → persist GP's Xs/ys (or bracket state) to JSON
  → controller.save_state() → persist history to JSON
  → Release lock
```

---

## 4. Interface Definition

### AutoML (top-level API)

```python
class AutoML:

    def __init__(
        self,
        workspace: str,                       # Local path for state persistence
        network: str,                         # Network name ("cosmos-rl", "dino", etc.)
        train_specs: dict,                    # Base training spec (default config)
        settings: dict,                       # Algorithm config (see §4.1)
        automl_hyperparameters: list = None,  # Param names to search, or None for schema defaults
        custom_param_ranges: dict = None,     # Per-param range overrides
        resume: bool = False,                 # Resume from persisted state
    )

    def next_recommendation(self) -> list[Recommendation]
    def report_result(self, rec_id: int, metric_value: float,
                      best_epoch: int = None, status: str = "success") -> None
    def is_complete(self) -> bool
    def get_best(self) -> Recommendation | None
    def get_progress(self) -> dict   # {completed, total, best_metric, best_rec_id, algorithm}
    def get_history(self) -> list[Recommendation]
```

#### 4.1 `settings` Dict

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `algorithm` | str | **Yes** | — | `bayesian` \| `hyperband` \| `bohb` \| `asha` \| `bfbo` \| `dehb` \| `pbt` \| `hyperband_es` |
| `metric` | str | No | `"loss"` | Metric name. Contains `"loss"` → lower is better, else higher is better |
| `automl_max_recommendations` | int | No | 20 | Max trials (bayesian, bfbo) |
| `automl_max_epochs` | int | No | 27 | Epoch budget (hyperband, bohb, asha, dehb) |
| `automl_reduction_factor` | int | No | 3 | Halving factor (hyperband, bohb, asha, dehb) |
| `epoch_multiplier` | int | No | 1 | Multiplier for epoch values |
| `automl_max_concurrent` | int | No | 4 | Max parallel configs (asha) |
| `automl_population_size` | int | No | 10 | Population size (pbt) |
| `automl_max_generations` | int | No | 20 | Max generations (pbt) |
| `automl_eval_interval` | int | No | 10 | Eval interval epochs (pbt) |
| `automl_perturbation_factor` | float | No | 1.2 | Perturbation magnitude (pbt) |

### Recommendation

```python
class Recommendation:
    id: int                         # Unique identifier
    specs: dict                     # Hyperparameter dict {"train.optm_lr": 1e-5, ...}
    metric: str                     # Metric name
    status: str                     # "pending" | "success" | "failure" | "canceled"
    result: float                   # Metric value (0.0 until reported)
    job_id: str | None              # Caller-assigned via rec.assign_job_id()
    resume_from_job_id: str | None  # PBT/Hyperband: checkpoint to resume from
    early_stop_epoch: int | None    # Hyperband: epoch limit for this config
    created_on: str                 # ISO timestamp
    last_modified: str              # ISO timestamp

    def assign_job_id(self, job_id: str) -> None
    def update_result(self, result: float) -> None
    def update_status(self, status: str) -> None
```

### StateStore

```python
class StateStore:
    def __init__(self, workspace_path: str)

    # Locking (for concurrent report_result)
    def lock(self) -> context_manager     # Global exclusive lock

    # Per-entity CRUD
    def get_job_specs(self, job_id) -> dict | None
    def save_job_specs(self, job_id, specs: dict) -> None
    def get_brain_info(self, job_id) -> dict | None
    def save_brain_info(self, job_id, state: dict) -> None
    def get_controller_info(self, job_id) -> list | None
    def save_controller_info(self, job_id, recs: list) -> None
    def get_best_rec_info(self, job_id) -> dict | None
    def save_best_rec_info(self, job_id, rec_number, rec_data) -> None
    def get_custom_param_ranges(self, experiment_id) -> dict | None
    def save_custom_param_ranges(self, experiment_id, ranges: dict) -> None
```

**Concurrency model**: All reads use shared locks (`LOCK_SH`), all writes use exclusive locks (`LOCK_EX`), plus an in-process `threading.Lock`. `Controller.report_result()` wraps the full read-modify-write in the global lock. Safe for threads and processes on local filesystem. Does NOT work on NFS.

**Storage layout:**
```
workspace/.automl/
├── specs/{session_id}.json
├── brain/{session_id}.json
├── controller/{session_id}.json
├── best_rec/{session_id}.json
├── current_rec/{session_id}.json
└── custom_ranges/{experiment_id}.json
```

---

## 5. Algorithm Behaviors

| Algorithm | Recs per call | Completion | Parallelizable | Best for |
|-----------|--------------|------------|----------------|----------|
| **bayesian** | 1 | `completed >= max_recommendations` | No (needs prev result) | Few-shot HPO, small budgets |
| **bfbo** | 1 | `completed >= max_recommendations` | No | Function-level optimization |
| **hyperband** | Batch (per rung) | All brackets exhausted | Yes (within rung) | Large search spaces |
| **bohb** | Batch | All brackets exhausted | Yes | Best of Bayesian + Hyperband |
| **asha** | Up to `max_concurrent` | `brain.done()` | Yes (async) | Async parallel HPO |
| **dehb** | Batch | All brackets exhausted | Yes | Evolutionary + multi-fidelity |
| **pbt** | `population_size` | `generation >= max_generations` | Yes (full population) | Dynamic schedules |
| **hyperband_es** | Batch | All brackets exhausted | Yes | Early stopping + multi-fidelity |

### Search Space per Network

Parameters are discovered from the network's config dataclass. Each field with `automl_enabled="TRUE"` is included. Example for **cosmos-rl**:

| Parameter | Type | Range | Constraint |
|-----------|------|-------|------------|
| `policy.lora.r` | int | 1–256 | Power of 2 |
| `policy.lora.lora_alpha` | int | 1–1024 | Power of 2 |
| `policy.lora.lora_dropout` | float | 0.0–0.1 | — |
| `train.epoch` | int | 1–20 | — |
| `train.optm_lr` | float | 0–∞ | Log-uniform sampling; list for full SFT |
| `train.optm_decay_type` | categorical | linear, sqrt, cosine, none | Weighted: cosine=0.4, none=0.4, linear=0.1, sqrt=0.1 |
| `custom.vision.fps` | int | 1–3 | — |

---

## 6. What Changed from FTMS

| Component | FTMS (old) | Standalone (new) | Change type |
|-----------|-----------|------------------|-------------|
| State persistence | MongoDB via `stateless_handler_utils.py` (15+ functions) | `StateStore` — JSON files + `fcntl.flock` | **Rewritten** |
| Controller | `controller.py` (87KB) — launched jobs, polled backends, read MongoDB, uploaded to cloud, WandB | `controller.py` (315 lines) — only brain loop + state | **Rewritten** |
| Session context | `JobContext` (18 fields, tied to Flask/MongoDB) | `AutoMLContext` (7 fields, plain dataclass) | **New** |
| Public API | None (embedded in FTMS) | `AutoML` class with 6 methods | **New** |
| Search space | `params.py` calling `get_microservices_network_and_action()` | `params.py` with direct network names + fixed `automl_enabled` default logic | **Modified** |
| Brain algorithms | 8 files in `automl/` | Same 8 files — imports + constructor changed, algorithm logic identical | **Modified (imports only)** |
| Brain base class | `automl_algorithm_base.py` | `base.py` — constructor takes `(context, state_store)` instead of `(job_context, root)` | **Modified (constructor)** |
| Network utils | `automl/network_utils/` | Same files, verbatim | **Copied** |
| Utility functions | `utils/automl_utils.py`, `handler_utils.py` | `utils/math_utils.py`, `spec_utils.py` | **Copied** |
| Schema generation | `scripts/generate_schema.py` | Same, import paths updated | **Copied** |
| Network configs | `nvidia_tao_core/config/` (41 networks) | Same 153 files, verbatim | **Copied** |

**Summary**: 2 rewrites (controller, state), 1 new (AutoML class), ~10 import-path changes, everything else verbatim.

---

## 7. Integration with TAO SDK

The AutoML wheel is consumed by the TAO Execution SDK (`tao_sdk`). Neither imports the other directly. A thin runner wires them:

```python
from tao_automl import AutoML
from tao_sdk import TaoExecutionSDK

sdk = TaoExecutionSDK(creds_file="secrets.json")
automl = AutoML(
    workspace="./experiment",
    network="cosmos-rl",
    train_specs=sdk.get_default_specs("cosmos-rl", "train"),
    settings={"algorithm": "bayesian", "metric": "loss",
              "automl_max_recommendations": 10},
)

while not automl.is_complete():
    recs = automl.next_recommendation()
    for rec in recs:
        merged_specs = merge(base_specs, rec.specs)
        job = sdk.create_job(network_arch="cosmos-rl", specs=merged_specs, ...)
        wait_for_completion(job)
        metric = extract_from_logs(sdk.get_job_logs(job.id))
        automl.report_result(rec.id, metric)

best = automl.get_best()
```

---

## 8. Concurrency Model

**Current implementation**: `fcntl.flock` file locking.

- Each data file has a `.lock` companion file
- Reads take `LOCK_SH` (shared — concurrent reads OK)
- Writes take `LOCK_EX` (exclusive — blocks other readers/writers)
- `Controller.report_result()` wraps the entire read-modify-write in a global lock
- In-process `threading.Lock` prevents thread interleaving within the same Python process
- Per-thread unique temp files (`path.tmp.{thread_id}`) prevent `os.replace` collisions

**Tested**: 8 concurrent threads reporting results simultaneously — zero data loss, all 21 tests pass.

**Limitation**: `fcntl.flock` does NOT work on NFS/network filesystems. The AutoML workspace must be on a local disk. This is acceptable because the AutoML state lives on the coordinator machine, not on the training nodes.

---

## 9. Open Questions for Team Review

1. **Interface**: Is the `next_recommendation()` → `report_result()` loop the right abstraction? Or should we support a callback model where the caller registers `on_start_job` / `on_check_job` handlers?

2. **State backend**: JSON files with file locking work for single-machine. If we need multi-machine coordination (e.g., a web service frontend), should `StateStore` become an ABC with pluggable backends (Redis, SQLite)?

3. **Parallel execution**: For batch algorithms (ASHA, Hyperband, PBT), the runner should launch recs concurrently. The locking is ready — do we want this in the AutoML wheel (as a built-in concurrent runner) or keep it in the SDK's runner?

4. **Schema ownership**: The `config/` directory (153 files, 41 networks) is copied from `tao-core`. Should this wheel own a copy, or should it depend on `nvidia-tao-core` for configs and only ship the schema/search_space logic?

5. **Network-specific logic**: `brain/network_utils/cosmos_rl.py` has cosmos-specific LoRA constraint handling. As we add more networks, should this be a plugin system or keep adding files?

---

## 10. Test Results

```
21 passed in 2.06s

✓ All imports (top-level, types, state, controller, 8 brain algorithms, utils)
✓ Recommendation create/update/type-checks
✓ AutoMLContext dataclass
✓ StateStore full CRUD roundtrip + missing returns None
✓ Controller full optimization loop (5 recs, is_complete, get_best)
✓ Controller state persistence + load_state roundtrip
✓ Controller higher-is-better metric (accuracy)
✓ Controller failure status excluded from get_best
✓ AlgorithmParams defaults + from_dict
✓ Concurrent report_result (8 threads, file locking)
✓ Package version and pip metadata
```
