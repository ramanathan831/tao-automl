# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AutoML runner: wires the tao_automl brain to tao_sdk execution.

The runner is platform-agnostic — it has no knowledge of which backend
(Lepton, Slurm, K8s) the SDK is connected to. Platform selection and
resource allocation are handled entirely by the SDK.

Usage::

    from tao_sdk import TaoExecutionSDK
    from tao_automl.runner import AutoMLRunner

    sdk = TaoExecutionSDK(creds_file="secrets.json")  # SDK knows the platform
    runner = AutoMLRunner(sdk)
    result = runner.run(
        network_arch="cosmos-rl",
        train_dataset_uri="aws://bucket/data/subset",
        automl_settings={
            "algorithm": "bayesian",
            "metric": "loss",
            "automl_max_recommendations": 5,
        },
    )
    print(result)

Or execute a plan file::

    python -m tao_automl.runner automl_plan.json secrets.json
"""

import json
import logging
import re
import signal
import sys
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_DEFAULT_POLL_INTERVAL = 30
_TERMINAL_STATUSES = {"Complete", "Error", "Canceled"}


def _extract_metric_from_logs(logs: str, metric_name: str) -> float | None:
    """Extract the final metric value from TAO training logs.

    Searches logs in reverse (last occurrence = final value). Handles:
    - Generic: "loss: 0.123" or "best loss: 0.123"
    - Cosmos-RL: "Step: 107/107, Loss: 8.27675, Grad norm: ..."
    - KPI: "kpi: 0.123"
    - Epoch: "Epoch 10 loss: 0.123"
    """
    if not logs:
        return None
    lines = logs.strip().splitlines()

    # Pattern 1: Cosmos-RL step format "Step: N/M, Loss: X.XXXX" (most specific)
    step_pattern = re.compile(
        r'Step:\s*\d+/\d+.*?Loss:\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)',
        re.IGNORECASE,
    )
    for line in reversed(lines):
        match = step_pattern.search(line)
        if match:
            try:
                val = float(match.group(1))
                if val > 0:  # Skip 0.0 values (empty validation)
                    return val
            except ValueError:
                continue

    # Pattern 2: direct metric match (case-insensitive)
    metric_pattern = re.compile(
        rf'(?:best\s+)?{re.escape(metric_name)}\s*[:=]\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)',
        re.IGNORECASE,
    )
    for line in reversed(lines):
        match = metric_pattern.search(line)
        if match:
            try:
                val = float(match.group(1))
                if val > 0:  # Skip 0.0 values
                    return val
            except ValueError:
                continue

    # Pattern 3: KPI
    kpi_pattern = re.compile(
        r'kpi\s*[:=]\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)', re.IGNORECASE,
    )
    for line in reversed(lines):
        match = kpi_pattern.search(line)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue

    # Pattern 4: Epoch line
    epoch_pattern = re.compile(
        r'[Ee]poch\s+\d+.*?(?:loss|accuracy|mIoU)\s*[:=]\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)',
    )
    for line in reversed(lines):
        match = epoch_pattern.search(line)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


def _check_execution_status(logs: str) -> str | None:
    """Check if logs contain Execution status: PASS or FAIL."""
    if not logs:
        return None
    for line in reversed(logs.strip().splitlines()):
        if "Execution status: PASS" in line:
            return "PASS"
        if "Execution status: FAIL" in line:
            return "FAIL"
    return None


class AutoMLRunner:
    """Wires AutoML brain to SDK execution for automated HPO loops."""

    def __init__(self, sdk, poll_interval: int = _DEFAULT_POLL_INTERVAL):
        self._sdk = sdk
        self._poll_interval = poll_interval
        self._active_jobs = {}

    def run(self, network_arch, train_dataset_uri, eval_dataset_uri="",
            base_checkpoint="", workspace_id=None, image=None,
            automl_settings=None,
            automl_hyperparameters=None, custom_param_ranges=None,
            workspace_path="./automl_workspace",
            spec_overrides=None, resume=False,
            on_recommendation=None, on_result=None) -> dict:
        """Run a full AutoML optimization loop.

        Args:
            network_arch: Model name (e.g. "cosmos-rl", "dino").
            train_dataset_uri: Training dataset URI (e.g. "aws://bucket/data").
            eval_dataset_uri: Eval dataset URI (optional).
            base_checkpoint: Pretrained checkpoint URI (optional).
            workspace_id: Workspace ID (default: from SDK).
            image: Docker image override (optional).
            automl_settings: Algorithm config (see AlgorithmParams).
            automl_hyperparameters: Param names to search, or None for schema defaults.
            custom_param_ranges: Per-param range overrides.
            workspace_path: Local path for AutoML state persistence.
            spec_overrides: Dict of spec overrides applied to base specs before
                AutoML starts. Dotted keys supported (e.g.
                {"train.epoch": 5, "policy.model_max_length": 40960}).
            resume: If True, resume from persisted state in workspace_path.
            on_recommendation: Callback(rec) called when a new rec is generated.
            on_result: Callback(rec, metric, status) called when a result is reported.

        Returns:
            Dict with keys: best, progress, history.
        """
        from tao_automl import AutoML

        automl_settings = automl_settings or {"algorithm": "bayesian", "metric": "loss"}
        workspace_id = workspace_id or self._sdk._workspace_id

        base_specs = self._sdk.get_default_specs(network_arch, "train")

        # Fix cosmos-rl spec defaults for single-GPU Lepton jobs
        # Reference: cosmos-rl training skill config_documentation.md
        if network_arch == "cosmos-rl":
            train = base_specs.setdefault("train", {})
            policy = base_specs.setdefault("policy", {})
            validation = base_specs.setdefault("validation", {})

            # train_batch_per_replica must be divisible by mini_batch
            mini_batch = train.get("train_policy", {}).get("mini_batch", 4)
            if train.get("train_batch_per_replica", 1) < mini_batch:
                train["train_batch_per_replica"] = mini_batch

            # model_max_length must be 40960 for video inputs to avoid
            # "vision_embeds.shape[0] must be equal to n_tokens" errors
            if policy.get("model_max_length", 4096) < 40960:
                policy["model_max_length"] = 40960

            # dp_shard_size must equal number of GPUs (1 on single-GPU Lepton)
            parallelism = policy.setdefault("parallelism", {})
            parallelism["dp_shard_size"] = 1

            # validation.enable must be true (container bug if false)
            validation["enable"] = True

            # Use 2 epochs as default — brain can override via automl params
            if "epoch" not in train or train["epoch"] > 2:
                train["epoch"] = 2

        # Apply user spec overrides on top of defaults + network fixes
        if spec_overrides:
            self._merge_specs(base_specs, spec_overrides)

        automl = AutoML(
            workspace=workspace_path, network=network_arch,
            train_specs=base_specs, settings=automl_settings,
            automl_hyperparameters=automl_hyperparameters,
            custom_param_ranges=custom_param_ranges,
            resume=resume,
        )
        metric_name = automl_settings.get("metric", "loss")
        logger.info("Starting AutoML loop: network=%s, algorithm=%s, metric=%s",
                     network_arch, automl_settings.get("algorithm"), metric_name)

        while not automl.is_complete():
            recs = automl.next_recommendation()
            if not recs:
                logger.info("No recommendations available — waiting for results")
                time.sleep(5)
                continue
            for rec in recs:
                if on_recommendation:
                    on_recommendation(rec)
                logger.info("Recommendation %d: launching job with %d spec overrides",
                            rec.id, len(rec.specs))
                merged_specs = self._merge_specs(base_specs, rec.specs)
                metric_value, status = self._run_one_job(
                    network_arch=network_arch, workspace_id=workspace_id,
                    train_dataset_uri=train_dataset_uri,
                    eval_dataset_uri=eval_dataset_uri,
                    base_checkpoint=base_checkpoint, image=image,
                    specs=merged_specs, rec=rec, metric_name=metric_name,
                )
                automl.report_result(
                    rec_id=rec.id,
                    metric_value=metric_value if metric_value is not None else 0.0,
                    status=status,
                )
                if on_result:
                    on_result(rec, metric_value, status)
                logger.info("Recommendation %d: metric=%.6f, status=%s",
                            rec.id, metric_value if metric_value is not None else 0.0, status)

        best = automl.get_best()
        progress = automl.get_progress()
        history = automl.get_history()
        result = {
            "best": {
                "rec_id": best.id if best else None,
                "specs": best.specs if best else {},
                "metric_value": best.result if best else None,
            },
            "progress": progress,
            "history": [{"rec_id": r.id, "metric": r.result, "status": r.status} for r in history],
        }
        logger.info("AutoML complete: %d recommendations, best metric=%.6f (rec %s)",
                     progress["completed"], best.result if best else 0.0, best.id if best else "N/A")
        return result

    def _run_one_job(self, network_arch, workspace_id, train_dataset_uri,
                     eval_dataset_uri, base_checkpoint, image,
                     specs, rec, metric_name) -> tuple[float | None, str]:
        """Launch a single training job and wait for it to finish."""
        try:
            job = self._sdk.create_job(
                network_arch=network_arch, workspace_id=workspace_id,
                train_dataset_uri=train_dataset_uri,
                eval_dataset_uri=eval_dataset_uri,
                base_checkpoint=base_checkpoint, action="train",
                specs=specs, image=image,
            )
        except Exception as e:
            logger.error("Failed to create job for rec %d: %s", rec.id, e)
            return None, "failure"

        rec.assign_job_id(job.id)
        self._active_jobs[rec.id] = job.id
        logger.info("Rec %d: job %s submitted (backend: %s)", rec.id, job.id, job.backend_job_id)

        # Poll status AND logs simultaneously — Lepton clears logs fast after
        # completion, so we cache the best metric seen during polling.
        cached_metric = None
        cached_exec_status = None
        all_logs = ""

        while True:
            time.sleep(self._poll_interval)

            # Read logs every poll cycle to cache metrics before they expire
            try:
                logs = self._sdk.get_job_logs(job.id)
                if logs:
                    all_logs = logs  # Keep latest snapshot
                    m = _extract_metric_from_logs(logs, metric_name)
                    if m is not None:
                        cached_metric = m
                    es = _check_execution_status(logs)
                    if es:
                        cached_exec_status = es
            except Exception:
                pass

            try:
                job_status = self._sdk.get_job_status(job.id)
            except Exception as e:
                logger.warning("Failed to get status for job %s: %s", job.id, e)
                continue
            if job_status.status in _TERMINAL_STATUSES:
                break

        # Final log read (may be empty if Lepton already cleaned up)
        try:
            final_logs = self._sdk.get_job_logs(job.id)
            if final_logs:
                all_logs = final_logs
                m = _extract_metric_from_logs(final_logs, metric_name)
                if m is not None:
                    cached_metric = m
                es = _check_execution_status(final_logs)
                if es:
                    cached_exec_status = es
        except Exception:
            pass

        exec_status = cached_exec_status or _check_execution_status(all_logs)
        status = job_status.status

        if status == "Error" or exec_status == "FAIL":
            logger.warning("Rec %d: job %s failed", rec.id, job.id)
            return cached_metric, "failure"
        if status == "Canceled":
            return None, "failure"

        metric_value = cached_metric
        if metric_value is None:
            logger.warning("Rec %d: job %s completed but could not extract metric '%s' from logs",
                           rec.id, job.id, metric_name)
            return None, "failure"

        logger.info("Rec %d: job %s succeeded, metric=%f", rec.id, job.id, metric_value)
        return metric_value, "success"

    @staticmethod
    def _merge_specs(base_specs: dict, rec_specs: dict) -> dict:
        """Deep-merge recommendation specs into base specs."""
        import copy
        merged = copy.deepcopy(base_specs)
        for key, value in rec_specs.items():
            parts = key.split(".")
            target = merged
            for part in parts[:-1]:
                if part not in target or not isinstance(target[part], dict):
                    target[part] = {}
                target = target[part]
            target[parts[-1]] = value
        return merged


def run_automl_plan(plan: dict, creds_file: str = None) -> dict:
    """Execute an AutoML plan file."""
    if not plan.get("ready"):
        issues = plan.get("blocking_issues", ["Unknown issue"])
        print("Plan is not ready to execute:")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)

    step = plan["steps"][0]
    params = step["params"]
    automl_settings = plan.get("automl_settings", {})

    from tao_sdk.sdk import TaoExecutionSDK
    sdk = TaoExecutionSDK(creds_file=creds_file)
    runner = AutoMLRunner(sdk)
    result = runner.run(
        network_arch=params["network_arch"],
        train_dataset_uri=params["train_dataset_uri"],
        eval_dataset_uri=params.get("eval_dataset_uri", ""),
        base_checkpoint=params.get("base_checkpoint", ""),
        workspace_id=params.get("workspace_id"),
        image=params.get("image"),
        automl_settings=automl_settings,
        automl_hyperparameters=plan.get("automl_hyperparameters"),
        custom_param_ranges=plan.get("custom_param_ranges"),
        workspace_path=plan.get("automl_workspace_path", "./automl_workspace"),
    )
    print(json.dumps(result, indent=2, default=str))
    return result


_runner = None

def _signal_handler(signum, frame):
    if _runner and _runner._active_jobs:
        for rec_id, job_id in _runner._active_jobs.items():
            try:
                _runner._sdk.cancel_job(job_id)
                print(f"Canceled job {job_id} (rec {rec_id})")
            except Exception as e:
                print(f"Failed to cancel job {job_id}: {e}")
    sys.exit(1)

signal.signal(signal.SIGINT, _signal_handler)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m tao_automl.runner automl_plan.json [secrets.json]")
        sys.exit(1)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    plan_path = sys.argv[1]
    creds_file = sys.argv[2] if len(sys.argv) > 2 else None
    with open(plan_path) as f:
        plan = json.load(f)
    global _runner
    from tao_sdk.sdk import TaoExecutionSDK
    sdk = TaoExecutionSDK(creds_file=creds_file)
    _runner = AutoMLRunner(sdk)
    run_automl_plan(plan, creds_file)


if __name__ == "__main__":
    main()
