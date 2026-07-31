#!/usr/bin/env python3

"""Direct-full-run qualification gate for SegFormer PTM arms.

The user explicitly disallowed CPU/model smokes and mini-steps.  This adapter
therefore accepts only stronger evidence from real one-node/eight-GPU,
full-VOC2012 training plus standalone evaluation.  Evidence never mutates or
bypasses the repository PTM registry: a successful workflow becomes runtime
eligible only after its exact registry record is independently ``supported``.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tao_automl.ptm_preflight import (
    CheckpointLoadSmokeRequest,
    CheckpointLoadSmokeResult,
)
from tao_automl.ptm_registry import canonical_sha256, load_ptm_registry

from .campaign_contract import (
    AGENT_FLAGS,
    FROZEN_QUALIFICATION_FIDELITY,
    FROZEN_QUALIFICATION_RUNTIME_OVERLAY,
    FROZEN_QUALIFICATION_TRAINING_EPOCHS,
    FROZEN_SQSH,
    FROZEN_VALIDATION_SANITY_MIN_MIOU,
    FROZEN_V1_QUALIFICATION_EVIDENCE,
    QUALIFICATION_CAMPAIGN_ID,
    QUALIFICATION_REVISION,
    segformer_registry_snapshot,
    sha256_file,
)


class QualificationGateError(RuntimeError):
    """Qualification evidence cannot authorize the campaign."""


def _sha(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise QualificationGateError(f"{name} must be lowercase SHA-256")
    return value


def _metric(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise QualificationGateError(f"{name} must be finite in [0, 1]")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise QualificationGateError(
            f"{name} must be finite in [0, 1]"
        ) from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise QualificationGateError(f"{name} must be finite in [0, 1]")
    return number


def _registry_core_identity(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    keys = (
        "id",
        "source",
        "expected_size_bytes",
        "model_family",
        "architecture",
        "backbone",
        "checkpoint_target",
        "input_contract",
        "task_compatibility",
    )
    try:
        return {
            key: copy.deepcopy(record[key])
            for key in keys
        }
    except KeyError as exc:
        raise QualificationGateError(
            "current SegFormer registry lacks immutable PTM identity"
        ) from exc


def _stage_evidence(
    document: Mapping[str, Any],
    *,
    current_registry: Any,
    current_registry_sha256: str,
) -> dict[str, Mapping[str, Any]]:
    """Bind pre-promotion evidence to post-promotion immutable identity."""
    stage_path_value = document.get("ptm_stage_manifest_path")
    stage_sha = document.get("ptm_stage_manifest_sha256")
    if stage_path_value is None and stage_sha is None:
        # Compatibility for pre-controller synthetic evidence. Production
        # controller evidence always takes the stronger cross-promotion path.
        if document.get("registry_sha256") != current_registry_sha256:
            raise QualificationGateError(
                "qualification registry identity changed"
            )
        return {}
    if (
        not isinstance(stage_path_value, str)
        or not Path(stage_path_value).is_absolute()
        or not isinstance(stage_sha, str)
    ):
        raise QualificationGateError(
            "qualification PTM stage identity is incomplete"
        )
    _sha(stage_sha, "ptm_stage_manifest_sha256")
    stage_path = Path(stage_path_value).resolve()
    if not stage_path.is_file():
        raise QualificationGateError(
            "qualification PTM stage manifest is unavailable"
        )
    try:
        stage = json.loads(stage_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise QualificationGateError(
            "qualification PTM stage manifest is invalid"
        ) from exc
    payload = copy.deepcopy(stage)
    supplied = payload.pop("stage_manifest_sha256", None)
    if supplied != stage_sha or supplied != canonical_sha256(payload):
        raise QualificationGateError(
            "qualification PTM stage manifest integrity failed"
        )
    rows = stage.get("ptms")
    if (
        stage.get("schema_version") != 2
        or stage.get("qualification_revision") != QUALIFICATION_REVISION
        or stage.get("campaign_id") != QUALIFICATION_CAMPAIGN_ID
        or stage.get("model") != "segformer"
        or stage.get("task") != "semantic_segmentation"
        or stage.get("registry_sha256")
        != document.get("registry_sha256")
        or stage.get("recipe_fidelity")
        != FROZEN_QUALIFICATION_FIDELITY
        or stage.get("runtime", {}).get("runtime_overlay")
        != FROZEN_QUALIFICATION_RUNTIME_OVERLAY
        or stage.get("prior_revision_evidence")
        != FROZEN_V1_QUALIFICATION_EVIDENCE
        or not isinstance(rows, list)
    ):
        raise QualificationGateError(
            "qualification PTM stage contract changed"
        )
    by_id = {
        item.get("checkpoint_id"): item
        for item in rows
        if isinstance(item, Mapping)
    }
    current_ids = tuple(
        item["id"] for item in segformer_registry_snapshot()["records"]
    )
    if set(by_id) != set(current_ids) or len(rows) != len(current_ids):
        raise QualificationGateError(
            "qualification PTM stage must contain every official arm"
        )
    for checkpoint_id in current_ids:
        row = by_id[checkpoint_id]
        current = current_registry.checkpoint(checkpoint_id)
        core = row.get("registry_core_identity")
        if (
            core != _registry_core_identity(current)
            or row.get("registry_core_identity_sha256")
            != canonical_sha256(core)
        ):
            raise QualificationGateError(
                f"{checkpoint_id} immutable registry identity changed "
                "during independent promotion"
            )
        checkpoint = row.get("checkpoint")
        if (
            not isinstance(checkpoint, Mapping)
            or checkpoint.get("size_bytes")
            != current["expected_size_bytes"]
            or _sha(
                checkpoint.get("sha256"),
                f"{checkpoint_id}.staged_checkpoint.sha256",
            )
            != checkpoint.get("sha256")
            or not isinstance(checkpoint.get("path"), str)
            or not checkpoint["path"].startswith("/lustre/")
        ):
            raise QualificationGateError(
                f"{checkpoint_id} staged checkpoint identity is invalid"
            )
        registered_sha = current.get("sha256")
        if (
            registered_sha is not None
            and registered_sha.lower() != checkpoint["sha256"]
        ):
            raise QualificationGateError(
                f"{checkpoint_id} promoted checksum differs from "
                "qualification bytes"
            )
    return by_id


def _workflow_integrity(
    workflow: Mapping[str, Any],
    *,
    checkpoint_id: str,
) -> str:
    payload = copy.deepcopy(dict(workflow))
    supplied = payload.pop("workflow_sha256", None)
    expected = canonical_sha256(payload)
    if supplied != expected:
        raise QualificationGateError(
            f"{checkpoint_id} workflow integrity failed"
        )
    return expected


@dataclass(frozen=True)
class QualifiedPTM:
    checkpoint_id: str
    source_checkpoint_path: str
    source_checkpoint_sha256: str
    source_checkpoint_size_bytes: int
    terminal_checkpoint_path: str
    terminal_checkpoint_sha256: str
    terminal_checkpoint_size_bytes: int
    val_miou: float
    test_miou: float
    workflow_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "source_checkpoint_path": self.source_checkpoint_path,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "source_checkpoint_size_bytes": self.source_checkpoint_size_bytes,
            "terminal_checkpoint_path": self.terminal_checkpoint_path,
            "terminal_checkpoint_sha256": self.terminal_checkpoint_sha256,
            "terminal_checkpoint_size_bytes": self.terminal_checkpoint_size_bytes,
            "val_miou": self.val_miou,
            "test_miou": self.test_miou,
            "workflow_sha256": self.workflow_sha256,
        }


@dataclass(frozen=True)
class QualificationDecision:
    evidence_path: str
    evidence_sha256: str
    qualification_campaign_id: str
    qualified: tuple[QualifiedPTM, ...]
    exclusions: tuple[Mapping[str, Any], ...]
    blockers: tuple[Mapping[str, Any], ...]
    decision_sha256: str

    @property
    def runtime_ready(self) -> bool:
        return bool(self.qualified) and not self.blockers

    @property
    def checkpoint_ids(self) -> tuple[str, ...]:
        return tuple(item.checkpoint_id for item in self.qualified)

    def stable_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "gate": "segformer_direct_full_gpu_then_supported_registry_v2",
            "evidence_path": self.evidence_path,
            "evidence_sha256": self.evidence_sha256,
            "qualification_campaign_id": self.qualification_campaign_id,
            "qualified": [item.to_dict() for item in self.qualified],
            "exclusions": [copy.deepcopy(dict(item)) for item in self.exclusions],
            "blockers": [copy.deepcopy(dict(item)) for item in self.blockers],
            "runtime_ready": self.runtime_ready,
            "registry_bypass_allowed": False,
            "cpu_or_smoke_model_job_launched": False,
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.stable_dict()
        value["decision_sha256"] = self.decision_sha256
        return value

    def assert_runtime_ready(self) -> None:
        if self.runtime_ready:
            return
        codes = ", ".join(
            f"{item.get('checkpoint_id', 'campaign')}:{item['code']}"
            for item in self.blockers
        )
        raise QualificationGateError(
            "SegFormer AutoML is fail-closed: "
            f"{codes or 'no runtime-qualified PTM'}. Direct GPU evidence "
            "cannot mutate or bypass the repository PTM registry."
        )


def _artifact(
    value: Any,
    *,
    name: str,
    expected_size: int | None = None,
) -> tuple[str, str, int]:
    if not isinstance(value, Mapping):
        raise QualificationGateError(f"{name} artifact is unavailable")
    path = value.get("path")
    size = value.get("size_bytes")
    digest = _sha(value.get("sha256"), f"{name}.sha256")
    if (
        not isinstance(path, str)
        or not path.startswith("/lustre/")
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 1
        or (expected_size is not None and size != expected_size)
    ):
        raise QualificationGateError(f"{name} artifact identity is invalid")
    return path, digest, size


def _successful_workflow(
    workflow: Mapping[str, Any],
    *,
    checkpoint_id: str,
    registry_record: Mapping[str, Any],
) -> QualifiedPTM:
    if (
        workflow.get("schema_version") != 2
        or workflow.get("qualification_revision") != QUALIFICATION_REVISION
        or workflow.get("checkpoint_id") != checkpoint_id
        or workflow.get("status") != "success"
        or workflow.get("terminal") is not True
        or workflow.get("failure_preserved") is not False
        or workflow.get("recipe_fidelity")
        != FROZEN_QUALIFICATION_FIDELITY
        or workflow.get("runtime_overlay")
        != FROZEN_QUALIFICATION_RUNTIME_OVERLAY
    ):
        raise QualificationGateError(
            f"{checkpoint_id} did not finish qualification successfully"
        )
    flags = workflow.get("agent_intervention_flags")
    if (
        not isinstance(flags, Mapping)
        or set(flags) != set(AGENT_FLAGS)
        or any(value is not False for value in flags.values())
    ):
        raise QualificationGateError(
            f"{checkpoint_id} agent-intervention flags are invalid"
        )
    source_path, source_sha, source_size = _artifact(
        workflow.get("source_checkpoint"),
        name=f"{checkpoint_id}.source_checkpoint",
        expected_size=int(registry_record["expected_size_bytes"]),
    )
    registered_sha = registry_record.get("sha256")
    if (
        registered_sha is not None
        and source_sha != str(registered_sha).lower()
    ):
        raise QualificationGateError(
            f"{checkpoint_id} source checkpoint differs from the "
            "promoted registry checksum"
        )
    train = workflow.get("train")
    evaluation = workflow.get("evaluation")
    if (
        not isinstance(train, Mapping)
        or train.get("status") != "Complete"
        or train.get("full_dataset") is not True
        or train.get("training_epochs")
        != FROZEN_QUALIFICATION_TRAINING_EPOCHS
        or train.get("validation_interval") != 1
        or train.get("validation_record_count")
        != FROZEN_QUALIFICATION_TRAINING_EPOCHS
        or train.get("recipe_fidelity")
        != FROZEN_QUALIFICATION_FIDELITY
        or train.get("runtime_overlay")
        != FROZEN_QUALIFICATION_RUNTIME_OVERLAY
        or not isinstance(train.get("job"), Mapping)
        or train["job"].get("runtime_overlay_required") is not True
        or train.get("nodes") != 1
        or train.get("gpus") != 8
        or not isinstance(evaluation, Mapping)
        or evaluation.get("status") != "Complete"
        or evaluation.get("full_validation_split") is not True
        or evaluation.get("runtime_overlay")
        != FROZEN_QUALIFICATION_RUNTIME_OVERLAY
        or not isinstance(evaluation.get("job"), Mapping)
        or evaluation["job"].get("runtime_overlay_required") is not True
        or evaluation.get("nodes") != 1
        or evaluation.get("gpus") != 8
    ):
        raise QualificationGateError(
            f"{checkpoint_id} full train/evaluation contract is incomplete"
        )
    checkpoint_path, checkpoint_sha, checkpoint_size = _artifact(
        train.get("terminal_checkpoint"),
        name=f"{checkpoint_id}.terminal_checkpoint",
    )
    val_miou = _metric(train.get("val_miou"), f"{checkpoint_id}.val_miou")
    test_miou = _metric(
        evaluation.get("test_miou"),
        f"{checkpoint_id}.test_miou",
    )
    if (
        val_miou < FROZEN_VALIDATION_SANITY_MIN_MIOU
        or test_miou < FROZEN_VALIDATION_SANITY_MIN_MIOU
    ):
        raise QualificationGateError(
            f"{checkpoint_id} is below the preregistered 0.10 mIoU "
            "experiment sanity gate"
        )
    expected = _workflow_integrity(
        workflow,
        checkpoint_id=checkpoint_id,
    )
    return QualifiedPTM(
        checkpoint_id=checkpoint_id,
        source_checkpoint_path=source_path,
        source_checkpoint_sha256=source_sha,
        source_checkpoint_size_bytes=source_size,
        terminal_checkpoint_path=checkpoint_path,
        terminal_checkpoint_sha256=checkpoint_sha,
        terminal_checkpoint_size_bytes=checkpoint_size,
        val_miou=val_miou,
        test_miou=test_miou,
        workflow_sha256=expected,
    )


def audit_qualification(path: str | Path) -> QualificationDecision:
    evidence_path = Path(path).resolve()
    if not evidence_path.is_file():
        raise QualificationGateError(
            f"qualification evidence is unavailable: {evidence_path}"
        )
    document = json.loads(evidence_path.read_text(encoding="utf-8"))
    supplied_sha = document.get("evidence_sha256")
    payload = copy.deepcopy(document)
    payload.pop("evidence_sha256", None)
    if supplied_sha != canonical_sha256(payload):
        raise QualificationGateError("qualification evidence integrity failed")
    snapshot = segformer_registry_snapshot()
    registry = load_ptm_registry()
    if (
        document.get("schema_version") != 2
        or document.get("qualification_revision") != QUALIFICATION_REVISION
        or document.get("campaign_id") != QUALIFICATION_CAMPAIGN_ID
        or document.get("model") != "segformer"
        or document.get("task") != "semantic_segmentation"
        or document.get("sqsh_sha256") != FROZEN_SQSH["sha256"]
        or document.get("recipe_fidelity")
        != FROZEN_QUALIFICATION_FIDELITY
        or document.get("runtime_overlay")
        != FROZEN_QUALIFICATION_RUNTIME_OVERLAY
        or document.get("prior_revision_evidence")
        != FROZEN_V1_QUALIFICATION_EVIDENCE
        or document.get("cpu_model_runs") != 0
        or document.get("smoke_model_runs") != 0
        or document.get("mini_step_runs") != 0
    ):
        raise QualificationGateError(
            "qualification campaign identity or execution policy changed"
        )
    _sha(document.get("registry_sha256"), "registry_sha256")
    stage_by_id = _stage_evidence(
        document,
        current_registry=registry,
        current_registry_sha256=snapshot["registry_sha256"],
    )
    workflows = document.get("workflows")
    if not isinstance(workflows, list):
        raise QualificationGateError("qualification workflows are unavailable")
    by_id = {
        item.get("checkpoint_id"): item
        for item in workflows
        if isinstance(item, Mapping)
    }
    expected_ids = tuple(
        item["id"] for item in snapshot["records"]
    )
    if set(by_id) != set(expected_ids) or len(workflows) != len(expected_ids):
        raise QualificationGateError(
            "qualification must preserve exactly one workflow per official PTM"
        )

    qualified: list[QualifiedPTM] = []
    exclusions: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    for checkpoint_id in expected_ids:
        workflow = by_id[checkpoint_id]
        record = registry.checkpoint(checkpoint_id)
        status = workflow.get("status")
        try:
            if (
                workflow.get("schema_version") != 2
                or workflow.get("qualification_revision")
                != QUALIFICATION_REVISION
                or workflow.get("recipe_fidelity")
                != FROZEN_QUALIFICATION_FIDELITY
                or workflow.get("runtime_overlay")
                != FROZEN_QUALIFICATION_RUNTIME_OVERLAY
            ):
                raise QualificationGateError(
                    f"{checkpoint_id} qualification v2 identity changed"
                )
            workflow_sha = _workflow_integrity(
                workflow,
                checkpoint_id=checkpoint_id,
            )
            if stage_by_id:
                staged = stage_by_id[checkpoint_id]["checkpoint"]
                path, digest, size = _artifact(
                    workflow.get("source_checkpoint"),
                    name=f"{checkpoint_id}.source_checkpoint",
                    expected_size=int(record["expected_size_bytes"]),
                )
                if (
                    path != staged["path"]
                    or digest != staged["sha256"]
                    or size != staged["size_bytes"]
                ):
                    raise QualificationGateError(
                        f"{checkpoint_id} workflow source differs from "
                        "the sealed PTM stage"
                    )
        except QualificationGateError as exc:
            blockers.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "code": "invalid_workflow_evidence",
                    "reason": str(exc),
                }
            )
            continue
        if status == "success":
            try:
                item = _successful_workflow(
                    workflow,
                    checkpoint_id=checkpoint_id,
                    registry_record=record,
                )
            except QualificationGateError as exc:
                blockers.append(
                    {
                        "checkpoint_id": checkpoint_id,
                        "code": "invalid_success_evidence",
                        "reason": str(exc),
                    }
                )
                continue
            if record.get("status") != "supported":
                blockers.append(
                    {
                        "checkpoint_id": checkpoint_id,
                        "code": "registry_not_supported",
                        "reason": (
                            "Direct full-run success exists, but the exact "
                            "repository record is not independently supported"
                        ),
                    }
                )
            else:
                qualified.append(item)
            continue
        if (
            status != "failure"
            or workflow.get("terminal") is not True
            or workflow.get("failure_preserved") is not True
            or not isinstance(workflow.get("failure_reason"), str)
            or not workflow["failure_reason"].strip()
        ):
            blockers.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "code": "terminal_record_missing",
                    "reason": (
                        "Every unsuccessful official PTM must retain a "
                        "structured terminal failure"
                    ),
                }
            )
            continue
        exclusion = {
            "checkpoint_id": checkpoint_id,
            "code": workflow.get("failure_code", "direct_full_run_failed"),
            "reason": workflow["failure_reason"],
            "workflow_sha256": workflow_sha,
        }
        if record.get("status") == "supported":
            blockers.append(
                {
                    **exclusion,
                    "code": "supported_registry_record_failed_direct_run",
                }
            )
        else:
            exclusions.append(exclusion)

    if not qualified:
        blockers.append(
            {
                "checkpoint_id": None,
                "code": "no_runtime_qualified_ptm",
                "reason": (
                    "No exact PTM has both successful direct full-run evidence "
                    "and repository-supported status"
                ),
            }
        )
    decision_payload = {
        "evidence_path": str(evidence_path),
        "evidence_sha256": supplied_sha,
        "qualification_campaign_id": document.get("campaign_id"),
        "qualified": [item.to_dict() for item in qualified],
        "exclusions": exclusions,
        "blockers": blockers,
    }
    return QualificationDecision(
        evidence_path=str(evidence_path),
        evidence_sha256=supplied_sha,
        qualification_campaign_id=str(document.get("campaign_id", "")),
        qualified=tuple(qualified),
        exclusions=tuple(exclusions),
        blockers=tuple(blockers),
        decision_sha256=canonical_sha256(decision_payload),
    )


class QualificationLoadEvidence:
    """Production-preflight callback backed by real full GPU workflows."""

    def __init__(self, decision: QualificationDecision):
        decision.assert_runtime_ready()
        self._decision = decision
        self._records = {
            item.checkpoint_id: item for item in decision.qualified
        }

    def __call__(
        self,
        request: CheckpointLoadSmokeRequest,
    ) -> CheckpointLoadSmokeResult:
        record = self._records.get(request.checkpoint_id)
        if record is None:
            return CheckpointLoadSmokeResult(
                ok=False,
                code="qualification_evidence_missing",
                reason="No completed direct full-run qualification exists",
            )
        registry_record = load_ptm_registry().checkpoint(request.checkpoint_id)
        observed_size = request.checkpoint_path.stat().st_size
        observed_sha = sha256_file(request.checkpoint_path)
        if (
            registry_record.get("status") != "supported"
            or observed_size != record.source_checkpoint_size_bytes
            or observed_sha != record.source_checkpoint_sha256
        ):
            return CheckpointLoadSmokeResult(
                ok=False,
                code="qualification_artifact_or_registry_mismatch",
                reason=(
                    "Live artifact or supported registry identity differs "
                    "from direct full-run evidence"
                ),
            )
        return CheckpointLoadSmokeResult(
            ok=True,
            code="direct_full_train_eval_qualification_reused",
            reason=(
                "Exact checkpoint passed full-dataset one-node/eight-GPU "
                "50-epoch training, validation, terminal reload, and "
                "standalone eval with the sealed v2 runtime overlay"
            ),
            details={
                "cpu_or_smoke_model_job_launched": False,
                "qualification_evidence_sha256": (
                    self._decision.evidence_sha256
                ),
                "workflow_sha256": record.workflow_sha256,
                "qualified_val_miou": record.val_miou,
                "qualified_test_miou": record.test_miou,
            },
        )


__all__ = [
    "QualificationDecision",
    "QualificationGateError",
    "QualificationLoadEvidence",
    "QualifiedPTM",
    "audit_qualification",
]
