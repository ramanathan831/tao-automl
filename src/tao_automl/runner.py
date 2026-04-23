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
import os
import re
import signal
import sys
import time
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_DEFAULT_POLL_INTERVAL = 30
_TERMINAL_STATUSES = {"Complete", "Error", "Canceled"}


def _extract_metric_from_logs(logs: str, metric_name: str) -> float | None:
    """Extract the final metric value from TAO training logs.

    Searches logs in reverse (last occurrence = final value). Handles:
    - Cosmos-RL validation: "[SFT] Validation loss: 0.12 for train step ..."
      (used when metric_name contains "val" — ASHA promotion metric)
    - Generic: "loss: 0.123" or "best loss: 0.123"
    - Cosmos-RL: "Step: 107/107, Loss: 8.27675, Grad norm: ..."
    - KPI: "kpi: 0.123"
    - Epoch: "Epoch 10 loss: 0.123"
    """
    if not logs:
        return None
    lines = logs.strip().splitlines()

    # If caller requested a validation metric, ONLY look for cosmos-rl's
    # per-epoch validation-loss line — don't fall through to train loss.
    if "val" in metric_name.lower():
        val_pattern = re.compile(
            r'\[SFT\]\s+Validation loss:\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)',
            re.IGNORECASE,
        )
        for line in reversed(lines):
            m = val_pattern.search(line)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    continue
        return None

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


# ---------------------------------------------------------------------------
# Key-validation helpers (fix #2: catch typos in spec_overrides /
# automl_hyperparameters at launch time instead of silently accepting them).
# ---------------------------------------------------------------------------

def _flatten_keys(d: dict, prefix: str = "") -> set[str]:
    """Recursively flatten a nested spec dict into dotted keys."""
    keys: set[str] = set()
    if not isinstance(d, dict):
        return keys
    for k, v in d.items():
        full = f"{prefix}.{k}" if prefix else str(k)
        keys.add(full)
        if isinstance(v, dict):
            keys |= _flatten_keys(v, full)
    return keys


def _validate_keys_against_schema(provided_keys, base_specs, kind):
    """Raise ValueError on provided keys that look like typos of existing
    schema keys. Accepts genuinely-new keys (logs a warning) so users who
    intentionally add a new spec field aren't blocked.
    """
    import difflib
    base_keys = _flatten_keys(base_specs)
    unknown = [k for k in provided_keys if k not in base_keys]
    for k in unknown:
        close = difflib.get_close_matches(k, base_keys, n=1, cutoff=0.85)
        if close:
            raise ValueError(
                f"{kind} key {k!r} is not in the skill's spec schema "
                f"but looks very close to existing key {close[0]!r} — "
                "did you mean that? (remove the typo; if you really "
                "intended a brand-new key, rename it so it doesn't collide.)"
            )
        logger.warning(
            "%s key %r is not in the skill's spec schema. "
            "Accepting it, but double-check that it's intentional.", kind, k)


# ---------------------------------------------------------------------------
# Metric direction (fix #1: explicit minimize/maximize).
# ---------------------------------------------------------------------------

def _implicit_direction(metric_name: str) -> str:
    """The brain's existing rule: metric name containing 'loss' is minimized,
    everything else is maximized. Keep this in one place so we can layer an
    explicit override on top.
    """
    return "minimize" if "loss" in (metric_name or "").lower() else "maximize"


def _resolve_direction(metric_name: str, explicit) -> tuple[str, bool]:
    """Return (effective_direction, invert_needed).

    If the caller didn't pass a direction, follow the implicit rule.
    If they did, validate and report whether we need to invert reported
    values to make the brain (which uses the implicit rule internally)
    optimize in the requested direction.
    """
    implicit = _implicit_direction(metric_name)
    if explicit is None:
        return implicit, False
    if explicit not in ("minimize", "maximize"):
        raise ValueError(
            f"automl_settings['direction'] must be 'minimize' or 'maximize', "
            f"got {explicit!r}"
        )
    return explicit, explicit != implicit


# ---------------------------------------------------------------------------
# Active-jobs persistence (fix #3: survive orchestrator crashes without
# leaking in-flight Lepton jobs).
# ---------------------------------------------------------------------------

def _active_jobs_path(workspace_path: str):
    from pathlib import Path
    return Path(workspace_path) / "active_jobs.json"


def _save_active_jobs(workspace_path: str, active: dict) -> None:
    """Atomic write of {rec_id: {rec_id, job_id, submitted_at}} to disk."""
    from pathlib import Path
    p = _active_jobs_path(workspace_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(list(active.values()), indent=2))
    tmp.replace(p)


def _load_active_jobs(workspace_path: str) -> list:
    p = _active_jobs_path(workspace_path)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except Exception as e:
        logger.warning("Couldn't read active_jobs.json: %s; starting fresh", e)
        return []


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
            backend_details=None,
            metric_extractor=None,
            eval_fn=None,
            on_recommendation=None, on_result=None) -> dict:
        """Run a full AutoML optimization loop.

        Args:
            network_arch: Model name (e.g. "cosmos-rl", "dino").
            train_dataset_uri: Training dataset URI (e.g. "aws://bucket/data").
            eval_dataset_uri: Eval dataset URI (optional).
            base_checkpoint: Pretrained checkpoint URI (optional).
            workspace_id: Workspace ID (default: from SDK).
            image: Docker image override. Default: from skill config.
            automl_settings: Algorithm config (see AlgorithmParams).
            automl_hyperparameters: Param names to search, or None for schema defaults.
            custom_param_ranges: Per-param range overrides.
            workspace_path: Local path for AutoML state persistence.
            spec_overrides: Dict of spec overrides applied to base specs before
                AutoML starts. Dotted keys supported (e.g.
                {"train.epoch": 5, "policy.model_max_length": 40960}).
            resume: If True, resume from persisted state in workspace_path.
            backend_details: Dict with 'backend_type', 'resource_shape',
                'dedicated_node_group', 'num_gpus', etc. Passed to sdk.create_job.
            metric_extractor: Optional callable ``(logs: str, metric_name: str) -> float | None``
                invoked on each poll of a rec's training logs to pull the
                current/latest metric value. Return ``None`` if the metric
                isn't yet present in the log snapshot. When ``None`` (default),
                the built-in ``_extract_metric_from_logs`` is used, which
                recognises training-step loss, "Validation loss:" lines,
                generic ``<metric_name>: X`` patterns, and epoch summaries.
                Supply your own extractor when your container's log format
                differs or your metric lives outside the log (e.g. reading
                an accuracy value from a results.json on S3).
            eval_fn: Optional callable ``(rec, train_job_id: str) -> float | None``
                invoked once after a rec's training job reaches a terminal
                state. Intended for workflows where the real metric needs a
                separate pipeline (e.g., merge LoRA + run inference + parse
                results.json). Whatever this returns overrides any value
                captured by ``metric_extractor`` and is what the brain sees
                via ``report_result``. Return ``None`` to fall back to the
                extractor. Raised exceptions are caught and logged; the rec
                is reported with the extractor's value (or None + failure).
            on_recommendation: Callback(rec) called when a new rec is generated.
            on_result: Callback(rec, metric, status) called when a result is reported.

        `automl_settings` additions:
            direction: Optional ``"minimize" | "maximize"``. When set, this
                overrides the implicit "metric name contains 'loss' → minimize,
                else maximize" rule. Useful when your metric name doesn't hint
                at the direction (e.g. ``"bleu_score"``, ``"perplexity"``,
                ``"wer"``). Under the hood the runner negates reported values
                when the explicit direction disagrees with the implicit rule,
                then flips them back in the returned result — callers always
                see their original metric scale.

        Returns:
            Dict with keys: best, progress, history.
        """
        from tao_automl import AutoML
        from tao_sdk.planner import SkillBank

        automl_settings = automl_settings or {"algorithm": "bayesian", "metric": "loss"}
        workspace_id = workspace_id or self._sdk._workspace_id

        if not resume:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            workspace_path = os.path.join(workspace_path, f"run_{ts}")
        os.makedirs(workspace_path, exist_ok=True)
        logger.info("Workspace: %s", workspace_path)

        # Load network knowledge from the skill bank. Set TAO_SKILL_BANK_PATH
        # to point at tao-skills-external (or a submodule). Everything below
        # is driven by skill config — no per-network special cases here.
        skill_bank = SkillBank()
        model_config = skill_bank.get_model_config(network_arch)
        if not model_config:
            raise ValueError(
                f"No skill config found for '{network_arch}'. "
                f"Set TAO_SKILL_BANK_PATH or install a skill bank."
            )
        base_specs = skill_bank.get_default_specs(network_arch, "train")
        if base_specs is None:
            raise ValueError(f"No default train specs found for '{network_arch}'.")

        resolved_image = image or model_config.get("container_image")
        script_runner = model_config.get("actions", {}).get("train")
        data_format = model_config.get("data_format")

        # Skills that declare tarball packing (e.g. path_from_format lists
        # "videos.tar.gz" / "images.tar.gz") ship datasets packed; the
        # script_runner only downloads, so we prepend an extraction step.
        # --strip-components=1 drops the tarball's top-level dir so extracted
        # files land where annotations reference them.
        if script_runner and self._skill_has_tarball_media(model_config):
            # The SDK entrypoint wraps this command in single quotes when
            # shelling out, so we avoid any single-quote characters here.
            # When no .tao_extracted marker is present we force-clean any
            # prior partial extraction (wipes subdirs; keeps the tarball and
            # annotation file) before re-extracting with --overwrite. This
            # recovers from interrupted/raced extractions.
            extract_cmd = (
                "echo [runner] extracting tarball media if present; "
                "find /mnt/lustre /results -type f "
                "\\( -name videos.tar.gz -o -name images.tar.gz \\) "
                "2>/dev/null | while read f; do "
                "d=\"$(dirname \"$f\")\"; "
                "marker=\"$d/.tao_extracted\"; "
                "if [ -f \"$marker\" ]; then echo [runner] already extracted \"$f\"; continue; fi; "
                "echo [runner] cleaning stale subdirs under \"$d\"; "
                "find \"$d\" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {{}} + 2>/dev/null; "
                "echo [runner] tar -xzf \"$f\"; "
                "tar --overwrite -xzf \"$f\" -C \"$d\" "
                "--strip-components=1 && touch \"$marker\"; done; "
                # Rewrite config file: /.../videos.tar.gz → /.../ (parent dir).
                # Runs whether or not extraction found anything (no-op if not).
                "sed -i -e \"s|/videos\\.tar\\.gz|/|g\" "
                "-e \"s|/images\\.tar\\.gz|/|g\" {config_path}; "
            )
            script_runner = dict(script_runner)
            script_runner["command"] = extract_cmd + script_runner["command"]

        # Inject dataset URIs declared by the skill's data_sources config.
        # Generic over any skill: maps spec keys to train/eval URIs using the
        # skill's own rules (source, path template, path_from_format).
        self._apply_data_sources(
            model_config=model_config, specs=base_specs, action="train",
            train_dataset_uri=train_dataset_uri,
            eval_dataset_uri=eval_dataset_uri,
            data_format=data_format,
        )

        # --- fix #2: validate spec_overrides + automl_hyperparameters
        #              against the schema before anything expensive runs.
        if spec_overrides:
            _validate_keys_against_schema(
                list(spec_overrides.keys()), base_specs, "spec_override")
            base_specs = self._merge_specs(base_specs, spec_overrides)
        if automl_hyperparameters:
            _validate_keys_against_schema(
                list(automl_hyperparameters), base_specs, "automl_hyperparameter")

        # --- fix #1: resolve explicit direction. _invert_metric tells us
        #              whether to negate values before reporting to the brain
        #              (and flip them back in the returned result).
        metric_name = automl_settings.get("metric", "loss")
        _effective_dir, invert_metric = _resolve_direction(
            metric_name, automl_settings.get("direction"))

        automl = AutoML(
            workspace=workspace_path, network=network_arch,
            train_specs=base_specs, settings=automl_settings,
            automl_hyperparameters=automl_hyperparameters,
            custom_param_ranges=custom_param_ranges,
            resume=resume,
        )
        logger.info("Starting AutoML loop: network=%s, algorithm=%s, "
                    "metric=%s, direction=%s%s",
                     network_arch, automl_settings.get("algorithm"),
                     metric_name, _effective_dir,
                     " (values will be inverted for the brain)" if invert_metric else "")

        # --- fix #3: if resuming, recover any jobs that were in flight when
        #              the previous orchestrator died. Poll each to terminal,
        #              report to the brain, then continue.
        if resume:
            pending = _load_active_jobs(workspace_path)
            if pending:
                logger.info("Resume: recovering %d in-flight job(s) from prior run",
                            len(pending))
                for entry in pending:
                    self._recover_pending_job(
                        entry=entry, automl=automl, metric_name=metric_name,
                        metric_extractor=metric_extractor, eval_fn=eval_fn,
                        workspace_path=workspace_path, invert_metric=invert_metric,
                        on_result=on_result,
                    )

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
                # Pre-generate a job UUID and rewrite any non-remote declared
                # output spec values to s3://<bucket>/results/<uuid>/<key> so
                # the script_runner uploads checkpoints/artifacts to S3.
                pre_job_id = str(uuid.uuid4())
                self._apply_output_destinations(
                    script_runner=script_runner, specs=merged_specs,
                    bucket=self._sdk._creds.get("S3_BUCKET_NAME", ""),
                    job_id=pre_job_id,
                )
                metric_value, status = self._run_one_job(
                    network_arch=network_arch, workspace_id=workspace_id,
                    train_dataset_uri=train_dataset_uri,
                    eval_dataset_uri=eval_dataset_uri,
                    base_checkpoint=base_checkpoint, image=resolved_image,
                    script_runner=script_runner, data_format=data_format,
                    backend_details=backend_details,
                    specs=merged_specs, rec=rec, metric_name=metric_name,
                    pre_job_id=pre_job_id,
                    metric_extractor=metric_extractor,
                    eval_fn=eval_fn,
                    workspace_path=workspace_path,
                )
                # Report to the brain, inverting if explicit direction disagrees
                # with the brain's implicit metric-name rule.
                report_value = metric_value
                if invert_metric and report_value is not None:
                    report_value = -report_value
                automl.report_result(
                    rec_id=rec.id,
                    metric_value=report_value if report_value is not None else 0.0,
                    status=status,
                )
                if on_result:
                    on_result(rec, metric_value, status)
                logger.info("Recommendation %d: metric=%.6f, status=%s",
                            rec.id, metric_value if metric_value is not None else 0.0, status)

        best = automl.get_best()
        progress = automl.get_progress()
        history = automl.get_history()

        # Unflip values if we inverted them for the brain, so callers see
        # metrics in their original scale regardless of `direction`.
        def _unflip(v):
            if v is None:
                return None
            return -v if invert_metric else v

        result = {
            "best": {
                "rec_id": best.id if best else None,
                "specs": best.specs if best else {},
                "metric_value": _unflip(best.result) if best else None,
            },
            "progress": progress,
            "history": [{"rec_id": r.id, "metric": _unflip(r.result),
                          "status": r.status} for r in history],
        }
        logger.info("AutoML complete: %d recommendations, best metric=%.6f (rec %s)",
                     progress["completed"],
                     _unflip(best.result) if best and best.result is not None else 0.0,
                     best.id if best else "N/A")
        return result

    def _run_one_job(self, network_arch, workspace_id, train_dataset_uri,
                     eval_dataset_uri, base_checkpoint, image,
                     script_runner, data_format, backend_details,
                     specs, rec, metric_name,
                     pre_job_id=None,
                     metric_extractor=None,
                     eval_fn=None,
                     workspace_path=None) -> tuple[float | None, str]:
        """Launch a single training job and wait for it to finish."""
        try:
            job = self._sdk.create_job(
                network_arch=network_arch, workspace_id=workspace_id,
                train_dataset_uri=train_dataset_uri,
                eval_dataset_uri=eval_dataset_uri,
                base_checkpoint=base_checkpoint, action="train",
                specs=specs, image=image,
                script_runner=script_runner,
                data_format=data_format,
                backend_details=backend_details,
                job_id=pre_job_id,
            )
        except Exception as e:
            logger.error("Failed to create job for rec %d: %s", rec.id, e)
            return None, "failure"

        rec.assign_job_id(job.id)
        self._active_jobs[rec.id] = job.id
        # fix #3: persist in-flight state so a resume can recover it.
        if workspace_path:
            self._persist_active_jobs(workspace_path)
        logger.info("Rec %d: job %s submitted (backend: %s)", rec.id, job.id, job.backend_job_id)

        # Caller can plug in a custom extractor; fall back to the built-in.
        extract_fn = metric_extractor or _extract_metric_from_logs

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
                    try:
                        m = extract_fn(logs, metric_name)
                    except Exception as ex:
                        logger.warning("metric_extractor raised for rec %d: %s",
                                       rec.id, ex)
                        m = None
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
                try:
                    m = extract_fn(final_logs, metric_name)
                except Exception as ex:
                    logger.warning("metric_extractor raised for rec %d: %s",
                                   rec.id, ex)
                    m = None
                if m is not None:
                    cached_metric = m
                es = _check_execution_status(final_logs)
                if es:
                    cached_exec_status = es
        except Exception:
            pass

        exec_status = cached_exec_status or _check_execution_status(all_logs)
        status = job_status.status

        # fix #3: job has reached terminal state — clear it from active_jobs.json.
        self._active_jobs.pop(rec.id, None)
        if workspace_path:
            self._persist_active_jobs(workspace_path)

        if status == "Error" or exec_status == "FAIL":
            logger.warning("Rec %d: job %s failed", rec.id, job.id)
            return cached_metric, "failure"
        if status == "Canceled":
            return None, "failure"

        # fix #4: if an eval_fn is provided, run it post-training and let its
        # return override the log-extracted metric. Errors are isolated.
        metric_value = cached_metric
        if eval_fn is not None:
            try:
                eval_metric = eval_fn(rec, job.id)
            except Exception as ex:
                logger.warning("eval_fn raised for rec %d: %s; falling back "
                                "to log-extracted metric", rec.id, ex)
                eval_metric = None
            if eval_metric is not None:
                logger.info("Rec %d: eval_fn returned metric=%f "
                            "(overriding log-extracted %s)",
                            rec.id, eval_metric,
                            f"{cached_metric:.6f}" if cached_metric is not None else "None")
                metric_value = eval_metric

        if metric_value is None:
            logger.warning("Rec %d: job %s completed but no metric could be "
                           "extracted (neither metric_extractor nor eval_fn "
                           "produced a value for '%s')",
                           rec.id, job.id, metric_name)
            return None, "failure"

        logger.info("Rec %d: job %s succeeded, metric=%f", rec.id, job.id, metric_value)
        return metric_value, "success"

    def _persist_active_jobs(self, workspace_path: str) -> None:
        """Dump self._active_jobs to workspace/active_jobs.json atomically."""
        now_iso = datetime.now(timezone.utc).isoformat()
        snapshot = {
            rec_id: {"rec_id": rec_id, "job_id": job_id, "updated_at": now_iso}
            for rec_id, job_id in self._active_jobs.items()
        }
        try:
            _save_active_jobs(workspace_path, snapshot)
        except Exception as e:
            logger.warning("Failed to persist active_jobs.json: %s", e)

    def _recover_pending_job(self, entry, automl, metric_name,
                              metric_extractor, eval_fn, workspace_path,
                              invert_metric, on_result) -> None:
        """Poll an in-flight job (recovered on resume), extract its result,
        and report it to the brain. Mirrors the tail of _run_one_job.
        """
        rec_id = entry["rec_id"]
        job_id = entry["job_id"]

        # Find the matching Recommendation object in the brain's history so
        # we can pass it to on_result/eval_fn and update rec.assign_job_id.
        rec = next((r for r in automl.get_history() if r.id == rec_id), None)
        if rec is None:
            logger.warning("Resume: rec %d not in brain history; dropping pending job %s",
                           rec_id, job_id)
            return

        self._active_jobs[rec_id] = job_id
        extract_fn = metric_extractor or _extract_metric_from_logs

        logger.info("Resume: polling rec %d job %s", rec_id, job_id)
        cached_metric = None
        cached_exec_status = None
        all_logs = ""

        # Poll until terminal.
        while True:
            time.sleep(self._poll_interval)
            try:
                logs = self._sdk.get_job_logs(job_id)
                if logs:
                    all_logs = logs
                    try:
                        m = extract_fn(logs, metric_name)
                    except Exception as ex:
                        logger.warning("metric_extractor raised during resume "
                                        "for rec %d: %s", rec_id, ex)
                        m = None
                    if m is not None:
                        cached_metric = m
                    es = _check_execution_status(logs)
                    if es:
                        cached_exec_status = es
            except Exception:
                pass
            try:
                job_status = self._sdk.get_job_status(job_id)
            except Exception as e:
                logger.warning("Resume: failed to get status for job %s: %s",
                               job_id, e)
                continue
            if job_status.status in _TERMINAL_STATUSES:
                break

        # Final log read
        try:
            final_logs = self._sdk.get_job_logs(job_id)
            if final_logs:
                try:
                    m = extract_fn(final_logs, metric_name)
                except Exception:
                    m = None
                if m is not None:
                    cached_metric = m
                es = _check_execution_status(final_logs)
                if es:
                    cached_exec_status = es
        except Exception:
            pass

        exec_status = cached_exec_status or _check_execution_status(all_logs)
        status = job_status.status
        self._active_jobs.pop(rec_id, None)
        self._persist_active_jobs(workspace_path)

        if status == "Error" or exec_status == "FAIL":
            metric_value = cached_metric
            report_status = "failure"
        elif status == "Canceled":
            metric_value = None
            report_status = "failure"
        else:
            metric_value = cached_metric
            if eval_fn is not None:
                try:
                    em = eval_fn(rec, job_id)
                except Exception as ex:
                    logger.warning("eval_fn raised during resume for rec %d: %s",
                                    rec_id, ex)
                    em = None
                if em is not None:
                    metric_value = em
            report_status = "success" if metric_value is not None else "failure"

        report_value = metric_value
        if invert_metric and report_value is not None:
            report_value = -report_value
        automl.report_result(
            rec_id=rec_id,
            metric_value=report_value if report_value is not None else 0.0,
            status=report_status,
        )
        if on_result:
            try:
                on_result(rec, metric_value, report_status)
            except Exception as ex:
                logger.warning("on_result callback raised during resume: %s", ex)
        logger.info("Resume: rec %d %s metric=%s",
                    rec_id, report_status,
                    f"{metric_value:.6f}" if metric_value is not None else "None")

    @staticmethod
    def _skill_has_tarball_media(model_config: dict) -> bool:
        """True if any data_sources entry declares a *.tar.gz suffix."""
        for action_rules in model_config.get("data_sources", {}).values():
            if not isinstance(action_rules, dict):
                continue
            for rule in action_rules.values():
                pff = rule.get("path_from_format") if isinstance(rule, dict) else None
                if not pff:
                    continue
                for val in pff.values():
                    candidates = val if isinstance(val, list) else [val]
                    if any(isinstance(c, str) and c.endswith(".tar.gz")
                           for c in candidates):
                        return True
        return False

    @staticmethod
    def _set_nested(target: dict, dotted_key: str, value) -> None:
        """Mutate target in-place: set target[a][b][c] for dotted_key 'a.b.c'."""
        parts = dotted_key.split(".")
        cursor = target
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = value

    @staticmethod
    def _get_nested(source: dict, dotted_key: str):
        """Read target[a][b][c] for dotted_key 'a.b.c'; None if missing."""
        parts = dotted_key.split(".")
        cursor = source
        for part in parts:
            if not isinstance(cursor, dict) or part not in cursor:
                return None
            cursor = cursor[part]
        return cursor

    @staticmethod
    def _apply_output_destinations(script_runner, specs, bucket, job_id):
        """For each declared output spec key that isn't already remote,
        point it at s3://<bucket>/results/<job_id>/<key_sanitized> so the
        script_runner uploads it to S3 instead of dropping it on container
        ephemeral disk.
        """
        if not script_runner or not bucket:
            return
        outputs = script_runner.get("outputs") or {}
        if isinstance(outputs, list):
            outputs = {k: {} for k in outputs}
        for spec_key in outputs.keys():
            current = AutoMLRunner._get_nested(specs, spec_key)
            if isinstance(current, str) and "://" in current:
                continue  # already remote
            safe = spec_key.replace(".", "_")
            remote_uri = f"s3://{bucket}/results/{job_id}/{safe}"
            AutoMLRunner._set_nested(specs, spec_key, remote_uri)

    @staticmethod
    def _apply_data_sources(model_config, specs, action,
                            train_dataset_uri, eval_dataset_uri, data_format):
        """Resolve skill's data_sources[action] into concrete URIs on specs.

        The skill declares per-spec-key rules:
          source:     "train_datasets" | "eval_dataset"
          path:       template (e.g. "{train_dataset_annotation}") — substituted
                      from top-level scalar values in *specs*.
          path_from_format: {<format>: <str|list>} — appended when present;
                      a list collapses to the folder URI (script_runner then
                      downloads the full prefix).
        """
        data_sources = model_config.get("data_sources", {}).get(action, {})
        if not data_sources:
            return

        source_to_uri = {
            "train_datasets": train_dataset_uri,
            "eval_dataset": eval_dataset_uri,
        }
        for spec_key, rule in data_sources.items():
            base_uri = source_to_uri.get(rule.get("source"))
            if not base_uri:
                continue
            # Normalize aws:// → s3://; the container's fsspec/s3fs understands
            # s3://, while aws:// is only used by the SDK for cloud_metadata keying.
            if base_uri.startswith("aws://"):
                base_uri = "s3://" + base_uri[len("aws://"):]
            base_uri = base_uri.rstrip("/") + "/"

            path_template = rule.get("path")
            if path_template:
                resolved = path_template
                for k, v in specs.items():
                    if isinstance(v, (str, int, float)):
                        resolved = resolved.replace("{" + k + "}", str(v))
                AutoMLRunner._set_nested(specs, spec_key, base_uri + resolved)
                continue

            path_from_format = rule.get("path_from_format")
            if path_from_format is not None:
                suffix = path_from_format.get(data_format, path_from_format.get("*"))
                chosen = None
                if isinstance(suffix, str) and suffix:
                    chosen = suffix
                elif isinstance(suffix, list):
                    tarballs = [s for s in suffix if isinstance(s, str)
                                and s.endswith(".tar.gz")]
                    # Prefer videos.tar.gz for video-leaning formats (llava),
                    # otherwise take the last tarball (skills list less-preferred
                    # candidates first).
                    chosen = (next((s for s in tarballs if "video" in s), None)
                              or (tarballs[-1] if tarballs else None))
                if chosen:
                    AutoMLRunner._set_nested(specs, spec_key, base_uri + chosen)
                else:
                    AutoMLRunner._set_nested(specs, spec_key, base_uri)

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
        backend_details=params.get("backend_details"),
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
