#!/usr/bin/env python3

"""Deterministic same-job checkpoint resume for Mask2Former SLURM slices.

The SLURM handler re-executes the same staged command after its 3.8-hour
inner timeout. Each execution reaches this helper after script_runner has
recreated the job's YAML spec and before TAO training starts. Only exact,
regular, non-symlink Mask2Former checkpoints under this job's own
results_dir/train directory are eligible.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shlex
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml


CHECKPOINT_PATTERN = re.compile(
    r"^model_epoch_(?P<epoch>[0-9]+)_step_(?P<step>[0-9]+)[.]pth$"
)
DECISION_FILENAME = "mask2former_checkpoint_resume_decision.json"
TRUSTED_CHECKPOINT_ENV = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"


class CheckpointResumeError(RuntimeError):
    """The generated Mask2Former spec cannot be resumed safely."""


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_identity(path: Path) -> dict[str, Any] | None:
    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    if match is None or path.is_symlink() or not path.is_file():
        return None
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size < 1:
        return None
    return {
        "path": str(path),
        "filename": path.name,
        "epoch": int(match.group("epoch")),
        "step": int(match.group("step")),
        "size_bytes": size,
    }


def select_latest_checkpoint(
    train_dir: str | Path,
    *,
    entries: Iterable[str | Path] | None = None,
) -> tuple[dict[str, Any] | None, int]:
    """Select max (epoch, step, filename) from the exact job directory."""
    directory = Path(train_dir)
    if entries is None:
        try:
            candidates = list(directory.iterdir())
        except FileNotFoundError:
            candidates = []
    else:
        candidates = [Path(item) for item in entries]

    eligible: list[dict[str, Any]] = []
    for path in candidates:
        # Direct-child equality prevents callers from supplying an unrelated
        # checkpoint that merely has a conforming basename.
        if path.parent != directory:
            continue
        identity = _checkpoint_identity(path)
        if identity is not None:
            eligible.append(identity)
    if not eligible:
        return None, 0
    selected = max(
        eligible,
        key=lambda item: (item["epoch"], item["step"], item["filename"]),
    )
    return selected, len(eligible)


def inject_resume_checkpoint(spec_path: str | Path) -> dict[str, Any]:
    """Inject the latest same-job checkpoint into one generated TAO YAML."""
    path = Path(spec_path)
    try:
        specification = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CheckpointResumeError(
            f"generated Mask2Former YAML is unreadable: {path}"
        ) from exc
    if not isinstance(specification, dict):
        raise CheckpointResumeError(
            "generated Mask2Former YAML must contain a mapping"
        )
    train = specification.get("train")
    results_dir = specification.get("results_dir")
    if not isinstance(train, dict):
        raise CheckpointResumeError("generated YAML has no train mapping")
    if (
        not isinstance(results_dir, str)
        or not results_dir
        or not Path(results_dir).is_absolute()
    ):
        raise CheckpointResumeError(
            "generated YAML results_dir must be an absolute same-job path"
        )

    runtime_root = os.environ.get("TAO_RESULTS_ROOT", "").rstrip("/")
    runtime_job_id = os.environ.get("TAO_JOB_ID", "")
    if runtime_root or runtime_job_id:
        if not runtime_root or not runtime_job_id or "/" in runtime_job_id:
            raise CheckpointResumeError(
                "TAO same-job result identity is incomplete or invalid"
            )
        expected_results_dir = (
            Path(runtime_root) / runtime_job_id / "results_dir"
        )
        if Path(results_dir) != expected_results_dir:
            raise CheckpointResumeError(
                "generated YAML results_dir is not this TAO job's output"
            )

    train_dir = Path(results_dir) / "train"
    selected, eligible_count = select_latest_checkpoint(train_dir)
    selected_path = selected["path"] if selected is not None else ""
    train["resume_training_checkpoint_path"] = selected_path

    temporary = path.with_name(f".{path.name}.resume-{os.getpid()}.tmp")
    try:
        temporary.write_text(
            yaml.safe_dump(specification, sort_keys=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass

    decision = {
        "schema_version": 1,
        "policy": "same_job_exact_epoch_step_max_v1",
        "checkpoint_directory": str(train_dir),
        "checkpoint_pattern": CHECKPOINT_PATTERN.pattern,
        "tao_job_id": runtime_job_id or None,
        "eligible_checkpoint_count": eligible_count,
        "selected_checkpoint": selected,
        "resume_enabled": selected is not None,
        "trusted_own_checkpoint_load": selected is not None,
        "missing_checkpoint_behavior": "blank_and_start_fresh",
        "symlinks_eligible": False,
        "selection_key": ["epoch", "step", "filename"],
    }
    decision["decision_sha256"] = _canonical_sha256(decision)
    decision_path = path.parent / DECISION_FILENAME
    decision_tmp = decision_path.with_name(
        f".{decision_path.name}.{os.getpid()}.tmp"
    )
    try:
        decision_tmp.write_text(
            json.dumps(decision, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(decision_tmp, decision_path)
    finally:
        try:
            decision_tmp.unlink()
        except FileNotFoundError:
            pass
    return decision


def wrap_train_command(command: str) -> str:
    """Run the same-job resume decision immediately before TAO training."""
    if not isinstance(command, str) or not command.strip():
        raise CheckpointResumeError("training command must be non-empty")
    source = Path(__file__).read_bytes()
    payload = base64.b64encode(source).decode("ascii")
    injector = " ".join(
        [
            "python3 -c",
            shlex.quote(
                "import base64;exec(base64.b64decode(" + repr(payload) + "))"
            ),
            "{config_path}",
        ]
    )
    return " ".join(
        [
            'MASK2FORMER_RESUME_STATE="$(' + injector + ')"',
            '&& case "$MASK2FORMER_RESUME_STATE" in',
            "resume) export " + TRUSTED_CHECKPOINT_ENV + "=1 ;;",
            "fresh) unset " + TRUSTED_CHECKPOINT_ENV + " ;;",
            "*) exit 86 ;;",
            "esac &&",
            command,
        ]
    )


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        raise SystemExit("usage: checkpoint_resume.py SPEC.yaml")
    decision = inject_resume_checkpoint(arguments[0])
    # This single token is consumed by the shell wrapper. The detailed,
    # immutable audit is stored beside the generated spec.
    print("resume" if decision["resume_enabled"] else "fresh")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through wrapper
    raise SystemExit(main())


__all__ = [
    "CHECKPOINT_PATTERN",
    "CheckpointResumeError",
    "DECISION_FILENAME",
    "TRUSTED_CHECKPOINT_ENV",
    "inject_resume_checkpoint",
    "select_latest_checkpoint",
    "wrap_train_command",
]
