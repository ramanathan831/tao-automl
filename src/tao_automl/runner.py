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

"""AutoML runner: wires the tao_automl brain to a platform SDK for HPO.

The runner is platform-agnostic: it accepts any of the 5 platform SDKs
(Lepton/Slurm/Kubernetes/Docker/Brev). The caller picks the platform; the
runner doesn't choose for them.

Usage::

    from pathlib import Path
    from tao_sdk.platforms.lepton import LeptonSDK   # or Slurm/K8s/Docker/Brev
    from tao_automl.runner import AutoMLRunner

    sdk = LeptonSDK()                                 # reads creds from env
    runner = AutoMLRunner(
        sdk=sdk,
        skill_dir=Path.home() / "tao-sdk/tao-skills-external/models/cosmos-rl",
        action="train",
    )
    result = runner.run(
        train_dataset_uri="s3://bucket/data/subset",
        automl_settings={
            "algorithm": "bayesian",
            "metric": "loss",
            "automl_max_recommendations": 5,
        },
    )
    print(result)

Or execute a plan file::

    python -m tao_automl.runner automl_plan.json --platform lepton
"""

import argparse
import copy
import json
import logging
import os
import re
import signal
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SkillContext — replaces the deleted SkillBank. Reads skill_info.yaml and
# spec_template_<action>.yaml directly from the skill bank dir, the same way
# agent launch scripts do per platform/tao-sdk/SKILL.md's "Constructing the
# spec / args" guidance.
# ---------------------------------------------------------------------------


@dataclass
class SkillContext:
    """Resolved skill metadata for a single (skill, action) pair.

    The runner used to call ``SkillBank.get_model_config(network_arch)`` and
    ``SkillBank.get_default_specs(network_arch, action)`` — both methods are
    gone. This class replaces both by loading directly from the skill bank
    layout that's documented and validated by tao-skills-external/scripts/
    validate-skills.sh.
    """
    skill_dir: Path
    action: str
    skill_info: dict[str, Any] = field(init=False)
    action_cfg: dict[str, Any] = field(init=False)
    default_specs: dict[str, Any] = field(init=False)
    valid_spec_keys: set[str] = field(init=False)
    container_image: str = field(init=False)
    network_arch: str = field(init=False)

    def __post_init__(self):
        self.skill_dir = Path(self.skill_dir)
        info_path = self.skill_dir / "references/skill_info.yaml"
        if not info_path.exists():
            raise FileNotFoundError(
                f"skill_info.yaml not found at {info_path}. "
                f"skill_dir must point at a model directory inside "
                f"tao-skills-external/models/<name>/."
            )
        self.skill_info = yaml.safe_load(info_path.read_text()) or {}

        actions = self.skill_info.get("actions") or {}
        if self.action not in actions:
            raise KeyError(
                f"Action {self.action!r} not declared in {info_path}. "
                f"Available: {sorted(actions.keys())}"
            )
        self.action_cfg = actions[self.action]
        self.network_arch = self.skill_info.get("network_arch", self.skill_dir.name)

        template_path = self.skill_dir / f"references/spec_template_{self.action}.yaml"
        self.default_specs = (
            yaml.safe_load(template_path.read_text()) if template_path.exists() else {}
        ) or {}
        schema_path = self.skill_dir / f"schemas/{self.action}.schema.json"
        if schema_path.exists():
            with open(schema_path) as f:
                schema = json.load(f) or {}
            self.valid_spec_keys = _schema_property_keys(schema) | _flatten_keys(
                schema.get("default", {})
            ) | _flatten_keys(self.default_specs)
        else:
            self.valid_spec_keys = _flatten_keys(self.default_specs)

        # Container image: action-level image overrides win, then model-level.
        # Values may be versions.yaml keys or absolute URIs.
        from tao_sdk.versions import resolve_container_image
        self.container_image = resolve_container_image(
            self.action_cfg.get("container_image")
            or self.skill_info.get("container_image", "")
        )

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

    # If caller requested a validation metric, prefer cosmos-rl's per-epoch
    # validation-loss line, then fall through to the generic metric patterns
    # below. TAO Core tasks often log plain ``val_loss: ...`` or
    # ``val_acc: ...`` instead of the cosmos-specific sentence.
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

    # Pattern 2: direct metric match (case-insensitive). Lightning progress
    # output may print metrics as ``train_loss_epoch: 18.901`` or split the
    # label and value across wrapped terminal lines, so also scan a
    # whitespace-normalized view of the full log.
    metric_aliases = _metric_aliases(metric_name)
    for suffix in ("_epoch", "_step"):
        if not metric_name.endswith(suffix):
            metric_aliases.append(f"{metric_name}{suffix}")
    if metric_name.lower().startswith("val_"):
        bare_metric = metric_name[4:]
        metric_aliases.extend([
            bare_metric,
            "Validation " + bare_metric.replace("_", " "),
        ])
    normalized_logs = re.sub(r"\s+", " ", logs)
    for alias in metric_aliases:
        metric_pattern = re.compile(
            rf'(?:best\s+)?{re.escape(alias)}\s*[:=]\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)',
            re.IGNORECASE,
        )
        for line in reversed(lines):
            match = metric_pattern.search(line)
            if match:
                try:
                    val = float(match.group(1))
                    if val >= 0:
                        return val
                except ValueError:
                    continue
        matches = list(metric_pattern.finditer(normalized_logs))
        for match in reversed(matches):
            try:
                val = float(match.group(1))
                if val >= 0:
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


def _metric_aliases(metric_name: str) -> list[str]:
    """Return common TAO spellings for a metric name.

    Different TAO entrypoints report the same KPI as ``val/loss`` in
    ``status.json`` or ``val_loss`` in Lightning monitor fields. AutoML callers
    should not have to know that spelling difference to get a valid metric.
    """
    aliases = [metric_name]
    if "/" in metric_name:
        aliases.append(metric_name.replace("/", "_"))
    if "_" in metric_name:
        aliases.append(metric_name.replace("_", "/"))
    seen = set()
    return [alias for alias in aliases if not (alias in seen or seen.add(alias))]


def _extract_metric_from_status_file(status_path: Path, metric_name: str) -> float | None:
    """Read the latest finite KPI value from a TAO line-delimited status file."""
    if not status_path.exists():
        return None
    aliases = _metric_aliases(metric_name)
    try:
        lines = status_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        kpi = payload.get("kpi")
        if not isinstance(kpi, dict):
            continue
        for alias in aliases:
            if alias not in kpi:
                continue
            try:
                value = float(kpi[alias])
            except (TypeError, ValueError):
                continue
            if value == value:
                return value
    return None


def _extract_metric_from_local_results(job_id: str, metric_name: str,
                                       platform_kwargs: dict | None) -> float | None:
    """Fallback for local Docker runs whose metrics are written to status.json.

    The platform SDK mounts a host results directory at ``/results``. When logs
    do not contain the metric, inspect the mounted job result folder and parse
    TAO's status artifacts.
    """
    for mount in (platform_kwargs or {}).get("mounts", []) or []:
        if mount.get("container_path") != "/results":
            continue
        host_root = mount.get("host_path")
        if not host_root:
            continue
        job_root = Path(host_root) / job_id
        for status_path in sorted(job_root.rglob("status.json")):
            metric = _extract_metric_from_status_file(status_path, metric_name)
            if metric is not None:
                return metric
    return None


_RESUME_FILE_EXTENSIONS = (".pth", ".pth.tar", ".pt", ".ckpt", ".hdf5", ".tlt")


def _local_results_mount(platform_kwargs: dict | None) -> tuple[Path, str] | None:
    """Return the host/container results mount pair when a /results bind exists."""
    for mount in (platform_kwargs or {}).get("mounts", []) or []:
        if not isinstance(mount, dict):
            continue
        container_path = str(mount.get("container_path", "")).rstrip("/")
        host_path = mount.get("host_path")
        if container_path == "/results" and host_path:
            return Path(host_path), container_path
    return None


def _as_container_path(path: Path, host_root: Path, container_root: str) -> str:
    """Map a host bind-mount path back to the path visible inside the job."""
    rel = path.relative_to(host_root)
    return f"{container_root.rstrip('/')}/{rel.as_posix()}"


def _checkpoint_epoch(path: Path) -> int:
    """Best-effort epoch/step number used to choose the latest resume artifact."""
    text = "/".join(path.parts[-5:])
    values: list[int] = []
    for pattern in (
        r"model_epoch[_-]?(\d+)",
        r"epoch[_-]?(\d+)",
        r"step[_-]?(\d+)",
        r"iter[_-]?(\d+)",
    ):
        values.extend(
            int(m.group(1)) for m in re.finditer(pattern, text, re.IGNORECASE)
        )
    return max(values) if values else 0


def _find_local_resume_artifact(
    job_id: str,
    platform_kwargs: dict | None,
    prefer_directory: bool,
) -> str | None:
    """Find a parent job checkpoint on a shared local /results mount.

    Multi-fidelity algorithms resume promoted trials from the checkpoint
    produced by a lower-budget job. For local Docker, all trials share the same
    host bind mount at /results, so the next container needs the container-side
    path rather than the host path.
    """
    mount = _local_results_mount(platform_kwargs)
    if not mount:
        return None
    host_root, container_root = mount
    job_root = host_root / job_id
    if not job_root.exists():
        return None

    candidates: list[tuple[tuple[int, int, float], Path]] = []

    def usable(path: Path) -> bool:
        norm = path.as_posix()
        return "/inputs/" not in norm and "/ptm/" not in norm

    for root, dirs, files in os.walk(job_root):
        root_path = Path(root)
        dirs[:] = [d for d in dirs if d not in {"inputs", "ptm", "__pycache__"}]
        if not usable(root_path):
            continue

        name = root_path.name.lower()
        if root_path != job_root and (name.startswith("epoch_") or name.startswith("step_")):
            try:
                if any(root_path.iterdir()):
                    norm = root_path.as_posix()
                    directory_priority = 80
                    if "/checkpoints/" in norm:
                        directory_priority = 120
                    elif "/safetensors/" in norm:
                        directory_priority = 70
                    if not prefer_directory:
                        directory_priority -= 50
                    candidates.append((
                        (
                            directory_priority,
                            _checkpoint_epoch(root_path),
                            root_path.stat().st_mtime,
                        ),
                        root_path,
                    ))
            except OSError:
                pass

        for filename in files:
            file_path = root_path / filename
            if not usable(file_path):
                continue
            lower = filename.lower()
            if not lower.endswith(_RESUME_FILE_EXTENSIONS):
                continue
            norm = file_path.as_posix()
            file_priority = 100
            if "/checkpoints/" in norm:
                file_priority = 115
            if "latest" in lower or "best" in lower:
                file_priority += 5
            if prefer_directory:
                file_priority -= 20
            try:
                candidates.append((
                    (
                        file_priority,
                        _checkpoint_epoch(file_path),
                        file_path.stat().st_mtime,
                    ),
                    file_path,
                ))
            except OSError:
                pass

    if not candidates:
        return None
    _, selected = max(candidates, key=lambda item: item[0])
    return _as_container_path(selected, host_root, container_root)


def _find_sdk_resume_artifact(sdk, job_id: str) -> str | None:
    """Fallback checkpoint lookup for SDKs that expose result listings."""
    try:
        checkpoints = sdk.get_checkpoints(job_id)
    except Exception:
        checkpoints = []
    if not checkpoints:
        return None
    try:
        results_dir = sdk.get_job_results_dir(job_id).rstrip("/")
    except Exception:
        results_dir = ""

    def normalize(path: str) -> str:
        if "://" in path or path.startswith("/"):
            return path
        return f"{results_dir}/{path}" if results_dir else path

    return normalize(sorted(checkpoints)[-1])


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

_PATH_PART_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*)(?:\[(\d+)\])?$")


def _parse_path_part(part: str) -> tuple[str, int | None]:
    match = _PATH_PART_RE.match(str(part))
    if match:
        return match.group(1), int(match.group(2)) if match.group(2) is not None else None
    return str(part), None


def _flatten_keys(d: Any, prefix: str = "") -> set[str]:
    """Recursively flatten a nested spec dict into dotted keys."""
    keys: set[str] = set()
    if isinstance(d, dict):
        for k, v in d.items():
            full = f"{prefix}.{k}" if prefix else str(k)
            keys.add(full)
            keys |= _flatten_keys(v, full)
    elif isinstance(d, list):
        for idx, v in enumerate(d):
            full = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            keys.add(full)
            keys |= _flatten_keys(v, full)
    return keys


def _schema_property_keys(schema: Any, prefix: str = "") -> set[str]:
    """Flatten JSON-schema property names into dotted spec keys.

    The packaged spec template may omit optional fields that are still valid
    according to ``schemas/<action>.schema.json``. Validate against both so
    direct optional overrides such as ``custom.vision.fps`` do not require
    unsafe placeholder defaults in the template.
    """
    keys: set[str] = set()
    if not isinstance(schema, dict):
        return keys
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for name, child in properties.items():
            full = f"{prefix}.{name}" if prefix else str(name)
            keys.add(full)
            keys |= _schema_property_keys(child, full)
    items = schema.get("items")
    if isinstance(items, dict) and prefix:
        indexed = f"{prefix}[0]"
        keys.add(indexed)
        keys |= _schema_property_keys(items, indexed)
    return keys


def _validate_keys_against_schema(provided_keys, base_specs, kind, schema_keys=None):
    """Raise ValueError on provided keys that look like typos of existing
    schema keys. Accepts genuinely-new keys (logs a warning) so users who
    intentionally add a new spec field aren't blocked.
    """
    import difflib
    base_keys = set(schema_keys or ()) | _flatten_keys(base_specs)
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
    """Wires AutoML brain to SDK execution for automated HPO loops.

    The runner accepts any of the 5 platform SDKs (LeptonSDK / SlurmSDK /
    KubernetesSDK / DockerSDK / BrevSDK). It does NOT pick a platform for
    the caller — instantiate the SDK you want and pass it in.

    ``skill_dir`` is the absolute path to a model directory inside the
    skill bank (e.g. ``Path.home() / 'tao-sdk/tao-skills-external/models/dino'``).
    The runner reads ``references/skill_info.yaml`` and
    ``references/spec_template_<action>.yaml`` from there.
    """

    def __init__(self, sdk, skill_dir, action: str = "train",
                 poll_interval: int = _DEFAULT_POLL_INTERVAL):
        self._sdk = sdk
        self.skill_ctx = SkillContext(skill_dir=Path(skill_dir), action=action)
        self._poll_interval = poll_interval
        self._active_jobs = {}

    def run(self, train_dataset_uri, eval_dataset_uri="",
            base_checkpoint="", workspace_id=None, image=None,
            automl_settings=None,
            automl_hyperparameters=None, custom_param_ranges=None,
            workspace_path="./automl_workspace",
            spec_overrides=None, resume=False,
            metric_extractor=None,
            eval_fn=None,
            on_recommendation=None, on_result=None,
            **platform_kwargs) -> dict:
        """Run a full AutoML optimization loop.

        Args:
            train_dataset_uri: Training dataset URI (e.g. "s3://bucket/data").
            eval_dataset_uri: Eval dataset URI (optional).
            base_checkpoint: Pretrained checkpoint URI (optional).
            workspace_id: Workspace ID (default: from SDK).
            image: Docker image override. Default: from skill_info.yaml's
                ``container_image`` (resolved via tao_sdk.versions).
            automl_settings: Algorithm config (see AlgorithmParams).
            automl_hyperparameters: Param names to search, or None for schema defaults.
            custom_param_ranges: Per-param range overrides.
            workspace_path: Local path for AutoML state persistence.
            spec_overrides: Dict of spec overrides applied to base specs before
                AutoML starts. Dotted keys supported (e.g.
                {"train.epoch": 5, "policy.model_max_length": 40960}).
            resume: If True, resume from persisted state in workspace_path.
            **platform_kwargs: Forwarded to ``sdk.create_job(...)``. Pass
                whichever kwargs your platform SDK accepts (Lepton:
                ``dedicated_node_group``, ``resource_shape``, ``num_nodes``;
                SLURM: ``partition``, ``account``, ``num_nodes``;
                Kubernetes: ``namespace``, ``node_selector``, ``num_nodes``;
                Docker: ``mounts``; Brev: ``instance_id``, ``gpu_type``).
                Plus the platform-agnostic ``gpu_count`` (defaults to 1 if
                not specified).
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

        automl_settings = automl_settings or {"algorithm": "bayesian", "metric": "loss"}
        workspace_id = workspace_id or getattr(self._sdk, "_workspace_id", "")
        network_arch = self.skill_ctx.network_arch

        if not resume:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            workspace_path = os.path.join(workspace_path, f"run_{ts}")
        os.makedirs(workspace_path, exist_ok=True)
        logger.info("Workspace: %s", workspace_path)

        # Skill metadata is loaded once at __init__ via SkillContext (replaces
        # the deleted SkillBank). action_cfg carries command/inputs/outputs/
        # config_format/upload_excludes — exactly what build_entrypoint takes.
        base_specs = copy.deepcopy(self.skill_ctx.default_specs)
        resolved_image = image or self.skill_ctx.container_image
        action_cfg = self.skill_ctx.action_cfg
        data_format = self.skill_ctx.skill_info.get("data_format")

        # Tar-extraction note: the in-container script_runner now handles
        # tar/tar.gz extraction inline (commit 661040b on tao-sdk main:
        # "Port tar/tar.gz extraction into script_runner"). We no longer
        # need to prepend an extract_cmd to the action command — the runner
        # detects archive inputs and extracts them as part of input download.
        # The old AutoML extract_cmd is gone; if a future skill's media
        # layout breaks this assumption, fix it in script_runner, not here.

        # Inject dataset URIs declared by the skill's data_sources config.
        # Generic over any skill: maps spec keys to train/eval URIs using the
        # skill's own rules (source, path template, path_from_format).
        self._apply_data_sources(
            skill_info=self.skill_ctx.skill_info, specs=base_specs,
            action=self.skill_ctx.action,
            train_dataset_uri=train_dataset_uri,
            eval_dataset_uri=eval_dataset_uri,
            data_format=data_format,
        )

        # --- fix #2: validate spec_overrides + automl_hyperparameters
        #              against the schema before anything expensive runs.
        if spec_overrides:
            _validate_keys_against_schema(
                list(spec_overrides.keys()), base_specs, "spec_override",
                self.skill_ctx.valid_spec_keys)
            base_specs = self._merge_specs(base_specs, spec_overrides)
        if automl_hyperparameters:
            _validate_keys_against_schema(
                list(automl_hyperparameters), base_specs, "automl_hyperparameter",
                self.skill_ctx.valid_spec_keys)

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
                run_base_specs = base_specs
                try:
                    stored_specs = automl._state_store.get_job_specs(automl._context.id)
                    if stored_specs:
                        run_base_specs = stored_specs
                except Exception as ex:
                    logger.debug("Could not read AutoML-updated base specs: %s", ex)
                merged_specs = self._merge_specs(run_base_specs, rec.specs)
                merged_specs = self._apply_resume_checkpoint(
                    merged_specs, rec, platform_kwargs
                )
                job_platform_kwargs = self._apply_resume_environment(
                    platform_kwargs, rec
                )
                # Output destination is resolved at runtime by script_runner
                # from TAO_RESULTS_ROOT (mount) / S3_BUCKET_NAME (cloud) env
                # vars the SDK injects. The agent doesn't pre-rewrite spec
                # output keys here — that lived in the deleted SDK contract.
                metric_value, status = self._run_one_job(
                    image=resolved_image, action_cfg=action_cfg,
                    specs=merged_specs, rec=rec, metric_name=metric_name,
                    metric_extractor=metric_extractor,
                    eval_fn=eval_fn,
                    workspace_path=workspace_path,
                    platform_kwargs=job_platform_kwargs,
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
        if best is None:
            failed = [r.id for r in history if r.status == "failure"]
            raise RuntimeError(
                "AutoML finished without a successful recommendation; "
                f"failed recommendation ids: {failed}"
            )

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

    def _run_one_job(self, image, action_cfg, specs, rec, metric_name,
                     metric_extractor=None,
                     eval_fn=None,
                     workspace_path=None,
                     platform_kwargs=None) -> tuple[float | None, str]:
        """Launch a single training job and wait for it to finish.

        Builds a container command via ``tao_sdk.script_runner.build_entrypoint``
        (inlines the in-container runner heredoc) and submits via the platform
        SDK's ``create_job(image, command, **platform_kwargs)``. Output
        destinations are resolved at runtime in the container from
        ``TAO_RESULTS_ROOT`` / ``S3_BUCKET_NAME`` env vars the SDK injects.
        """
        from tao_sdk.script_runner import build_entrypoint

        try:
            ep = build_entrypoint(
                command=action_cfg["command"],
                specs=specs,
                inputs=action_cfg.get("inputs"),
                outputs=action_cfg.get("outputs"),
                config_format=action_cfg.get("config_format", "toml"),
                upload_excludes=action_cfg.get("upload_excludes", []),
            )
            job = self._sdk.create_job(
                image=image,
                command=ep["command"],
                **(platform_kwargs or {}),
            )
        except Exception as e:
            logger.error("Failed to create job for rec %d: %s", rec.id, e)
            return None, "failure"

        rec.assign_job_id(job.id)
        self._active_jobs[rec.id] = job.id
        # Persist in-flight state so a resume can recover it.
        if workspace_path:
            self._persist_active_jobs(workspace_path)
        logger.info("Rec %d: job %s submitted (backend: %s)",
                    rec.id, job.id, getattr(job, "backend_job_id", job.id))

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
            metric_value = _extract_metric_from_local_results(
                job.id, metric_name, platform_kwargs
            )
            if metric_value is not None:
                logger.info("Rec %d: recovered metric=%f from local status artifacts",
                            rec.id, metric_value)

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

    # _skill_has_tarball_media was removed: tar/tar.gz extraction is now
    # handled by script_runner inline (tao-sdk commit 661040b "Port tar/tar.gz
    # extraction into script_runner"). The runner no longer prepends an
    # extract command, so the tarball-detection helper has no callers.

    @staticmethod
    def _set_nested(target: dict, dotted_key: str, value) -> None:
        """Mutate target in-place: set target[a][b][c] for dotted_key 'a.b.c'."""
        parts = dotted_key.split(".")
        cursor = target
        for part in parts[:-1]:
            key, idx = _parse_path_part(part)
            if key not in cursor:
                cursor[key] = [] if idx is not None else {}
            cursor = cursor[key]
            if idx is not None:
                if not isinstance(cursor, list):
                    raise TypeError(f"Spec path {dotted_key!r} expected list at {key!r}")
                while len(cursor) <= idx:
                    cursor.append({})
                if cursor[idx] is None:
                    cursor[idx] = {}
                cursor = cursor[idx]
        last_key, last_idx = _parse_path_part(parts[-1])
        if last_idx is None:
            cursor[last_key] = value
            return
        if last_key not in cursor or not isinstance(cursor[last_key], list):
            cursor[last_key] = []
        while len(cursor[last_key]) <= last_idx:
            cursor[last_key].append(None)
        cursor[last_key][last_idx] = value

    @staticmethod
    def _get_nested(source: dict, dotted_key: str):
        """Read target[a][b][c] for dotted_key 'a.b.c'; None if missing."""
        parts = dotted_key.split(".")
        cursor = source
        for part in parts:
            key, idx = _parse_path_part(part)
            if not isinstance(cursor, dict) or key not in cursor:
                return None
            cursor = cursor[key]
            if idx is not None:
                if not isinstance(cursor, list) or idx >= len(cursor):
                    return None
                cursor = cursor[idx]
        return cursor

    def _apply_resume_checkpoint(
        self, specs: dict, rec, platform_kwargs: dict | None
    ) -> dict:
        """Inject parent-checkpoint resume params for promoted recommendations.

        Hyperband-family algorithms and PBT return a recommendation with
        ``resume_from_job_id`` once they promote or exploit a prior trial. The
        brain knows which trial won, but the runner owns platform paths and
        skill specs, so the checkpoint handoff belongs here.
        """
        parent_job_id = getattr(rec, "resume_from_job_id", None)
        if not parent_job_id:
            return specs

        path_key = None
        for candidate in (
            "train.resume_training_checkpoint_path",
            "resume_training_checkpoint_path",
        ):
            if (
                candidate in self.skill_ctx.valid_spec_keys
                or self._get_nested(specs, candidate) is not None
            ):
                path_key = candidate
                break

        bool_or_path_key = None
        for candidate in ("train.resume", "resume"):
            if (
                candidate in self.skill_ctx.valid_spec_keys
                or self._get_nested(specs, candidate) is not None
            ):
                bool_or_path_key = candidate
                break

        if not path_key and not bool_or_path_key:
            logger.warning(
                "Rec %d requested resume from %s, but no resume spec key was "
                "found for %s",
                rec.id, parent_job_id, self.skill_ctx.network_arch,
            )
            return specs

        prefer_directory = bool_or_path_key is not None and path_key is None
        artifact = (
            _find_local_resume_artifact(parent_job_id, platform_kwargs, prefer_directory)
            or _find_sdk_resume_artifact(self._sdk, parent_job_id)
        )
        if not artifact:
            logger.warning(
                "Rec %d requested resume from %s, but no checkpoint artifact "
                "could be resolved",
                rec.id, parent_job_id,
            )
            return specs

        if path_key:
            self._set_nested(specs, path_key, artifact)
            logger.info(
                "Rec %d will resume from parent job %s via %s=%s",
                rec.id, parent_job_id, path_key, artifact,
            )
        else:
            # Cosmos-RL's `train.resume` accepts either True or a concrete
            # checkpoint path. Use the path form so a new output directory can
            # still be used for the resumed trial.
            self._set_nested(specs, bool_or_path_key, artifact)
            logger.info(
                "Rec %d will resume from parent job %s via %s=%s",
                rec.id, parent_job_id, bool_or_path_key, artifact,
            )
        return specs

    def _apply_resume_environment(
        self, platform_kwargs: dict | None, rec
    ) -> dict | None:
        """Add runtime env needed by model-specific checkpoint resume paths."""
        if not getattr(rec, "resume_from_job_id", None):
            return platform_kwargs

        updated = copy.deepcopy(platform_kwargs or {})
        env_vars = dict(updated.get("env_vars") or {})
        if env_vars.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD") != "1":
            env_vars["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
            logger.info(
                "Rec %d enabling PyTorch trusted-checkpoint resume for %s",
                rec.id, self.skill_ctx.network_arch,
            )
        updated["env_vars"] = env_vars
        return updated

    # _apply_output_destinations was removed: output destinations are
    # resolved at runtime by script_runner from TAO_RESULTS_ROOT (mount) /
    # S3_BUCKET_NAME (cloud) env vars the SDK injects in create_job. The
    # runner doesn't pre-rewrite output spec keys here anymore.

    @staticmethod
    def _apply_data_sources(skill_info, specs, action,
                            train_dataset_uri, eval_dataset_uri, data_format):
        """Resolve skill's data_sources[action] into concrete URIs on specs.

        The skill declares per-spec-key rules in ``skill_info.yaml``:
          source:     "train_datasets" | "eval_dataset"
          path:       template (e.g. "{train_dataset_annotation}") — substituted
                      from top-level scalar values in *specs*.
          path_from_format: {<format>: <str|list>} — appended when present;
                      a list collapses to the folder URI (script_runner then
                      downloads the full prefix).
        """
        data_sources = skill_info.get("data_sources", {}).get(action, {})
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

            mapping = rule.get("mapping")
            if isinstance(mapping, dict):
                item = {}
                for field_name, field_cfg in mapping.items():
                    field_cfg = field_cfg or {}
                    path = field_cfg.get("path")
                    if path:
                        item[field_name] = base_uri + str(path).lstrip("/")
                    elif not field_cfg.get("optional"):
                        item[field_name] = base_uri.rstrip("/")
                current = AutoMLRunner._get_nested(specs, spec_key)
                if rule.get("multiple_sources") or isinstance(current, list):
                    AutoMLRunner._set_nested(specs, spec_key, [item])
                else:
                    AutoMLRunner._set_nested(specs, spec_key, item)
                continue

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
            AutoMLRunner._set_nested(merged, key, value)
        return merged


_PLATFORMS = ("lepton", "slurm", "kubernetes", "docker", "brev")


def _make_sdk(platform: str):
    """Construct a platform SDK by name. No default — caller must pick.

    Matches platform/tao-sdk/SKILL.md's "It does not select platforms
    automatically" stance: none of the 5 SDKs is a sensible default
    (Lepton biases DGX Cloud, SLURM biases on-prem clusters, etc.).
    """
    if platform == "lepton":
        from tao_sdk.platforms.lepton import LeptonSDK
        return LeptonSDK()
    if platform == "slurm":
        from tao_sdk.platforms.slurm import SlurmSDK
        return SlurmSDK()
    if platform == "kubernetes":
        from tao_sdk.platforms.kubernetes import KubernetesSDK
        return KubernetesSDK()
    if platform == "docker":
        from tao_sdk.platforms.docker import DockerSDK
        return DockerSDK()
    if platform == "brev":
        from tao_sdk.platforms.brev import BrevSDK
        return BrevSDK()
    raise ValueError(
        f"Unknown platform {platform!r}. Choose one of: {', '.join(_PLATFORMS)}."
    )


def run_automl_plan(plan: dict, platform: str) -> dict:
    """Execute an AutoML plan file on the chosen platform.

    The plan JSON's ``params`` block must include ``skill_dir`` (absolute
    path to a model directory inside tao-skills-external). Per-platform
    create_job kwargs go under ``params.platform_kwargs``.
    """
    if not plan.get("ready"):
        issues = plan.get("blocking_issues", ["Unknown issue"])
        print("Plan is not ready to execute:")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)

    step = plan["steps"][0]
    params = step["params"]
    automl_settings = plan.get("automl_settings", {})

    skill_dir = params.get("skill_dir")
    if not skill_dir:
        raise ValueError("plan.steps[0].params.skill_dir is required "
                         "(absolute path to a model dir in tao-skills-external).")
    action = params.get("action", "train")
    platform_kwargs = params.get("platform_kwargs") or {}

    sdk = _make_sdk(platform)
    runner = AutoMLRunner(sdk=sdk, skill_dir=skill_dir, action=action)
    global _runner
    _runner = runner
    result = runner.run(
        train_dataset_uri=params["train_dataset_uri"],
        eval_dataset_uri=params.get("eval_dataset_uri", ""),
        base_checkpoint=params.get("base_checkpoint", ""),
        workspace_id=params.get("workspace_id"),
        image=params.get("image"),
        automl_settings=automl_settings,
        automl_hyperparameters=plan.get("automl_hyperparameters"),
        custom_param_ranges=plan.get("custom_param_ranges"),
        workspace_path=plan.get("automl_workspace_path", "./automl_workspace"),
        spec_overrides=params.get("spec_overrides"),
        **platform_kwargs,
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
    parser = argparse.ArgumentParser(
        description="Execute an AutoML plan against a chosen platform SDK.",
    )
    parser.add_argument("plan", help="Path to the AutoML plan JSON.")
    parser.add_argument(
        "--platform", required=True, choices=_PLATFORMS,
        help="Target platform SDK. Required — no default. "
             "Pick the backend you want to submit jobs to.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    with open(args.plan) as f:
        plan = json.load(f)
    run_automl_plan(plan, platform=args.platform)


if __name__ == "__main__":
    main()
