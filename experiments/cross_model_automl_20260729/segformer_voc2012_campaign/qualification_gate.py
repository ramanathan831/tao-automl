#!/usr/bin/env python3

"""Direct-full-run qualification gate for SegFormer PTM arms.

The user explicitly disallowed CPU/model smokes and mini-steps.  This adapter
therefore accepts only stronger evidence from real one-node/eight-GPU,
full-VOC2012 training plus standalone evaluation.  Evidence never mutates the
repository PTM registry.  Ordinary callers still require a supported record;
an explicitly sealed successor campaign may additionally authorize an exact,
versioned in-memory registry projection bound to immutable qualification
evidence and complete pre-existing registry license metadata.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tao_automl.ptm_preflight import (
    CheckpointLoadSmokeRequest,
    CheckpointLoadSmokeResult,
)
from tao_automl.ptm_registry import (
    PTMRegistry,
    canonical_sha256,
    load_ptm_registry,
)

from .campaign_contract import (
    AGENT_FLAGS,
    FROZEN_QUALIFICATION_FIDELITY,
    FROZEN_QUALIFICATION_RUNTIME_OVERLAY,
    FROZEN_QUALIFICATION_TRAINING_EPOCHS,
    FROZEN_RUNTIME_LOCAL_CHECKPOINT_SPEC_FILE,
    FROZEN_SQSH,
    FROZEN_VALIDATION_SANITY_MIN_MIOU,
    FROZEN_PRIOR_QUALIFICATION_EVIDENCE,
    QUALIFICATION_CAMPAIGN_ID,
    QUALIFICATION_REVISION,
    RUNTIME_LOCAL_ELIGIBILITY_KIND,
    segformer_registry_snapshot,
    sha256_file,
    validate_contract,
)


class QualificationGateError(RuntimeError):
    """Qualification evidence cannot authorize the campaign."""


_PRETRAINED_LOAD_COUNT_FIELDS = (
    "loaded_tensor_count",
    "missing_tensor_count",
    "shape_mismatched_tensor_count",
    "unmatched_tensor_count",
    "non_tensor_count",
)
_RUNTIME_LOCAL_SPEC = FROZEN_RUNTIME_LOCAL_CHECKPOINT_SPEC_FILE


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
        != FROZEN_PRIOR_QUALIFICATION_EVIDENCE
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
    checkpoint_target: str
    registry_core_identity_sha256: str
    source_checkpoint_path: str
    source_checkpoint_sha256: str
    source_checkpoint_size_bytes: int
    terminal_checkpoint_path: str
    terminal_checkpoint_sha256: str
    terminal_checkpoint_size_bytes: int
    val_miou: float
    test_miou: float
    pretrained_load_component: str
    pretrained_loaded_tensor_count: int
    pretrained_loaded_keyset_sha256: str
    pretrained_load_report_sha256: str
    workflow_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_target": self.checkpoint_target,
            "registry_core_identity_sha256": (
                self.registry_core_identity_sha256
            ),
            "source_checkpoint_path": self.source_checkpoint_path,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "source_checkpoint_size_bytes": self.source_checkpoint_size_bytes,
            "terminal_checkpoint_path": self.terminal_checkpoint_path,
            "terminal_checkpoint_sha256": self.terminal_checkpoint_sha256,
            "terminal_checkpoint_size_bytes": self.terminal_checkpoint_size_bytes,
            "val_miou": self.val_miou,
            "test_miou": self.test_miou,
            "pretrained_load_component": self.pretrained_load_component,
            "pretrained_loaded_tensor_count": (
                self.pretrained_loaded_tensor_count
            ),
            "pretrained_loaded_keyset_sha256": (
                self.pretrained_loaded_keyset_sha256
            ),
            "pretrained_load_report_sha256": (
                self.pretrained_load_report_sha256
            ),
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
    runtime_eligibility: Mapping[str, Any]
    decision_sha256: str
    runtime_registry: PTMRegistry = field(repr=False, compare=False)

    @property
    def runtime_ready(self) -> bool:
        return bool(self.qualified) and not self.blockers

    @property
    def checkpoint_ids(self) -> tuple[str, ...]:
        return tuple(item.checkpoint_id for item in self.qualified)

    def stable_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 3,
            "gate": self.runtime_eligibility["kind"],
            "evidence_path": self.evidence_path,
            "evidence_sha256": self.evidence_sha256,
            "qualification_campaign_id": self.qualification_campaign_id,
            "qualified": [item.to_dict() for item in self.qualified],
            "exclusions": [copy.deepcopy(dict(item)) for item in self.exclusions],
            "blockers": [copy.deepcopy(dict(item)) for item in self.blockers],
            "runtime_eligibility": copy.deepcopy(
                dict(self.runtime_eligibility)
            ),
            "runtime_ready": self.runtime_ready,
            "registry_bypass_allowed": False,
            "repository_registry_mutated": False,
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
            f"{codes or 'no runtime-qualified PTM'}. Qualification evidence "
            "cannot mutate the repository PTM registry or admit an arm "
            "outside the sealed runtime-eligibility policy."
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


def _pretrained_load_evidence(
    train: Mapping[str, Any],
    *,
    checkpoint_id: str,
    checkpoint_path: str,
    checkpoint_target: str,
) -> dict[str, Any]:
    """Require the exact positive loader receipt emitted by product code."""
    expected_component = {
        "train.pretrained_model_path": "model",
        "model.backbone.pretrained_backbone_path": "backbone",
    }.get(checkpoint_target)
    status_evidence = train.get("status_evidence")
    report = (
        status_evidence.get("pretrained_load")
        if isinstance(status_evidence, Mapping)
        else None
    )
    if expected_component is None or not isinstance(report, Mapping):
        raise QualificationGateError(
            f"{checkpoint_id} has no exact positive pretrained-load evidence"
        )
    required = {
        "schema_version",
        "checkpoint",
        "component",
        "loaded_keyset_sha256",
        "status_record_occurrences",
        "report_sha256",
        *_PRETRAINED_LOAD_COUNT_FIELDS,
    }
    if (
        set(report) != required
        or report.get("schema_version") != 1
        or report.get("checkpoint") != checkpoint_path
        or report.get("component") != expected_component
        or _sha(
            report.get("loaded_keyset_sha256"),
            f"{checkpoint_id}.loaded_keyset_sha256",
        )
        != report.get("loaded_keyset_sha256")
        or _sha(
            report.get("report_sha256"),
            f"{checkpoint_id}.pretrained_load.report_sha256",
        )
        != report.get("report_sha256")
        or any(
            isinstance(report.get(name), bool)
            or not isinstance(report.get(name), int)
            or report[name] < 0
            for name in _PRETRAINED_LOAD_COUNT_FIELDS
        )
        or report["loaded_tensor_count"] < 1
        or isinstance(report.get("status_record_occurrences"), bool)
        or not isinstance(report.get("status_record_occurrences"), int)
        or report["status_record_occurrences"] < 1
    ):
        raise QualificationGateError(
            f"{checkpoint_id} pretrained-load receipt is invalid or does not "
            "prove a nonzero exact-component load"
        )
    payload = {
        key: copy.deepcopy(value)
        for key, value in report.items()
        if key not in {"status_record_occurrences", "report_sha256"}
    }
    if canonical_sha256(payload) != report["report_sha256"]:
        raise QualificationGateError(
            f"{checkpoint_id} pretrained-load receipt integrity failed"
        )
    return copy.deepcopy(dict(report))


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
    pretrained_load = _pretrained_load_evidence(
        train,
        checkpoint_id=checkpoint_id,
        checkpoint_path=source_path,
        checkpoint_target=str(registry_record.get("checkpoint_target", "")),
    )
    terminal_checkpoint = train.get("terminal_checkpoint")
    checkpoint_path, checkpoint_sha, checkpoint_size = _artifact(
        terminal_checkpoint,
        name=f"{checkpoint_id}.terminal_checkpoint",
    )
    terminal_epoch = FROZEN_QUALIFICATION_TRAINING_EPOCHS - 1
    epoch_token = f"{terminal_epoch:03d}"
    filename = Path(checkpoint_path).name
    if (
        not isinstance(terminal_checkpoint, Mapping)
        or terminal_checkpoint.get("training_epochs")
        != FROZEN_QUALIFICATION_TRAINING_EPOCHS
        or terminal_checkpoint.get("terminal_epoch_index")
        != terminal_epoch
        or terminal_checkpoint.get("naming_contract")
        != f"model_epoch_{epoch_token}_step_numeric"
        or terminal_checkpoint.get("ambiguity_policy") != "fail_closed"
        or not filename.startswith(f"model_epoch_{epoch_token}_step_")
        or not filename.endswith(".pth")
        or not filename[len(f"model_epoch_{epoch_token}_step_"):-4].isdigit()
    ):
        raise QualificationGateError(
            f"{checkpoint_id} terminal checkpoint contract changed"
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
        checkpoint_target=str(registry_record["checkpoint_target"]),
        registry_core_identity_sha256=canonical_sha256(
            _registry_core_identity(registry_record)
        ),
        source_checkpoint_path=source_path,
        source_checkpoint_sha256=source_sha,
        source_checkpoint_size_bytes=source_size,
        terminal_checkpoint_path=checkpoint_path,
        terminal_checkpoint_sha256=checkpoint_sha,
        terminal_checkpoint_size_bytes=checkpoint_size,
        val_miou=val_miou,
        test_miou=test_miou,
        pretrained_load_component=pretrained_load["component"],
        pretrained_loaded_tensor_count=pretrained_load[
            "loaded_tensor_count"
        ],
        pretrained_loaded_keyset_sha256=pretrained_load[
            "loaded_keyset_sha256"
        ],
        pretrained_load_report_sha256=pretrained_load["report_sha256"],
        workflow_sha256=expected,
    )


def _runtime_local_policy(
    expected_contract: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Return a separately sealed, post-qualification authorization policy."""
    if expected_contract is None:
        return None
    value = expected_contract.get("qualification_policy", {}).get(
        "runtime_local_eligibility"
    )
    if not isinstance(value, Mapping) or value.get("enabled") is not True:
        return None
    required_false = (
        "repository_registry_mutation_allowed",
        "missing_license_normalization_allowed",
        "failed_arm_promotion_allowed",
        "unsupported_arm_promotion_allowed",
        "agent_override_allowed",
    )
    if (
        value.get("schema_version") != 1
        or value.get("kind")
        != RUNTIME_LOCAL_ELIGIBILITY_KIND
        or value.get("scope") != "campaign_local_in_memory_projection"
        or value.get("model") != "segformer"
        or value.get("task") != "semantic_segmentation"
        or value.get("tao_version") != "7.1.0"
        or value.get("container_sha256") != FROZEN_SQSH["sha256"]
        or value.get("license_policy")
        != "complete_existing_registry_metadata_only"
        or value.get("checkpoint_spec_file") != _RUNTIME_LOCAL_SPEC
        or any(value.get(name) is not False for name in required_false)
    ):
        raise QualificationGateError(
            "sealed runtime-local SegFormer eligibility policy is invalid"
        )
    for name in (
        "base_registry_sha256",
        "qualification_file_sha256",
        "qualification_evidence_sha256",
        "qualification_contract_sha256",
        "qualification_controller_sha256",
        "eligibility_gate_sha256",
        "runtime_resolver_sha256",
        "wheel_sha256",
    ):
        _sha(value.get(name), f"runtime_local_eligibility.{name}")
    for name in ("eligibility_source_commit", "sdk_commit", "skills_commit"):
        commit = value.get(name)
        if (
            not isinstance(commit, str)
            or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)
        ):
            raise QualificationGateError(
                f"runtime_local_eligibility.{name} must be a Git commit"
            )
    evidence_path = value.get("qualification_evidence_path")
    if (
        not isinstance(evidence_path, str)
        or not Path(evidence_path).is_absolute()
    ):
        raise QualificationGateError(
            "runtime_local_eligibility.qualification_evidence_path must be "
            "absolute"
        )
    return value


def _has_complete_existing_license(record: Mapping[str, Any]) -> bool:
    license_info = record.get("license")
    return bool(
        isinstance(license_info, Mapping)
        and isinstance(license_info.get("name"), str)
        and license_info["name"].strip()
        and isinstance(license_info.get("access_requirements"), list)
        and all(
            isinstance(item, str) and item.strip()
            for item in license_info["access_requirements"]
        )
    )


def _project_runtime_registry(
    *,
    base_registry: PTMRegistry,
    qualified: tuple[QualifiedPTM, ...],
    metadata_incomplete_ids: tuple[str, ...],
    evidence_path: Path,
    evidence_file_sha256: str,
    evidence_sha256: str,
    policy: Mapping[str, Any] | None,
) -> tuple[PTMRegistry, dict[str, Any]]:
    """Build a validated run-local view without changing global registry data."""
    base_sha = base_registry.document_sha256
    base_version = base_registry.registry_version
    base_model_records = base_registry.to_dict()["models"]["segformer"][
        "checkpoints"
    ]
    base_record_hashes = {
        record["id"]: canonical_sha256(record)
        for record in sorted(base_model_records, key=lambda value: value["id"])
    }
    if policy is None:
        return base_registry, {
            "schema_version": 1,
            "kind": "repository_supported_registry",
            "scope": "repository_registry",
            "base_registry_version": base_version,
            "base_registry_sha256": base_sha,
            "projected_registry_version": base_version,
            "projected_registry_sha256": base_sha,
            "qualified_checkpoint_ids": [
                item.checkpoint_id for item in qualified
            ],
            "runtime_metadata_incomplete_checkpoint_ids": list(
                metadata_incomplete_ids
            ),
            "base_record_sha256_by_checkpoint_id": base_record_hashes,
            "transformations": [],
            "repository_registry_mutated": False,
            "failed_arms_preserved": True,
        }
    if (
        policy["base_registry_sha256"] != base_sha
        or policy.get("base_registry_version") != base_version
        or policy["qualification_file_sha256"] != evidence_file_sha256
        or policy["qualification_evidence_sha256"] != evidence_sha256
    ):
        raise QualificationGateError(
            "runtime-local eligibility is not bound to the exact base "
            "registry and qualification evidence"
        )

    document = base_registry.to_dict()
    document["registry_version"] = (
        f"{base_version}+segformer-runtime-local-v1."
        f"{evidence_sha256[:12]}"
    )
    records = {
        record["id"]: record
        for record in document["models"]["segformer"]["checkpoints"]
    }
    transformations = []
    for item in sorted(qualified, key=lambda value: value.checkpoint_id):
        record = records[item.checkpoint_id]
        original = copy.deepcopy(record)
        if record["status"] == "unsupported":
            raise QualificationGateError(
                f"{item.checkpoint_id} is explicitly unsupported and cannot "
                "enter a runtime-local projection"
            )
        if not _has_complete_existing_license(record):
            raise QualificationGateError(
                f"{item.checkpoint_id} lacks complete existing license "
                "metadata and cannot enter a runtime-local projection"
            )
        if (
            canonical_sha256(_registry_core_identity(record))
            != item.registry_core_identity_sha256
            or record.get("sha256")
            not in (None, item.source_checkpoint_sha256)
        ):
            raise QualificationGateError(
                f"{item.checkpoint_id} registry/source identity mismatch"
            )
        record["status"] = "supported"
        record.pop("status_reason", None)
        record["sha256"] = item.source_checkpoint_sha256
        record["compatible_tao_versions"] = ["==7.1.0"]
        record["default_spec_overrides"] = {}
        record["checkpoint_spec_file"] = copy.deepcopy(_RUNTIME_LOCAL_SPEC)
        record["deprecation"] = {
            "is_deprecated": False,
            "reason": None,
            "replacement_id": None,
        }
        record["validation"] = {
            "status": "validated",
            "tao_version": policy["tao_version"],
            "container_identity": (
                "sqsh-sha256:" + policy["container_sha256"]
            ),
            "evidence": (
                f"{evidence_path}#evidence_sha256={evidence_sha256};"
                f"workflow_sha256={item.workflow_sha256};"
                "pretrained_load_report_sha256="
                f"{item.pretrained_load_report_sha256}"
            ),
        }
        transformations.append(
            {
                "checkpoint_id": item.checkpoint_id,
                "action": (
                    "retain_supported_identity"
                    if original["status"] == "supported"
                    else "qualify_exact_unverified_identity"
                ),
                "base_status": original["status"],
                "projected_status": "supported",
                "base_record_sha256": canonical_sha256(original),
                "projected_record_sha256": canonical_sha256(record),
                "source_checkpoint_sha256": (
                    item.source_checkpoint_sha256
                ),
                "workflow_sha256": item.workflow_sha256,
                "pretrained_load_report_sha256": (
                    item.pretrained_load_report_sha256
                ),
                "pretrained_loaded_tensor_count": (
                    item.pretrained_loaded_tensor_count
                ),
                "license_metadata_source": "unchanged_base_registry_record",
            }
        )

    projected = PTMRegistry(document)
    if base_registry.document_sha256 != base_sha:
        raise QualificationGateError(
            "base repository registry changed during projection"
        )
    qualified_ids = [
        item.checkpoint_id
        for item in sorted(qualified, key=lambda value: value.checkpoint_id)
    ]
    eligibility = {
        "schema_version": 1,
        "kind": RUNTIME_LOCAL_ELIGIBILITY_KIND,
        "scope": "campaign_local_in_memory_projection",
        "model": "segformer",
        "task": "semantic_segmentation",
        "tao_version": policy["tao_version"],
        "container_sha256": policy["container_sha256"],
        "qualification_path": str(evidence_path),
        "qualification_file_sha256": evidence_file_sha256,
        "qualification_evidence_sha256": evidence_sha256,
        "qualification_contract_sha256": policy[
            "qualification_contract_sha256"
        ],
        "qualification_controller_sha256": policy[
            "qualification_controller_sha256"
        ],
        "eligibility_gate_sha256": policy["eligibility_gate_sha256"],
        "runtime_resolver_sha256": policy["runtime_resolver_sha256"],
        "eligibility_source_commit": policy["eligibility_source_commit"],
        "wheel_sha256": policy["wheel_sha256"],
        "sdk_commit": policy["sdk_commit"],
        "skills_commit": policy["skills_commit"],
        "base_registry_version": base_version,
        "base_registry_sha256": base_sha,
        "projected_registry_version": projected.registry_version,
        "projected_registry_sha256": projected.document_sha256,
        "qualified_checkpoint_ids": qualified_ids,
        "runtime_metadata_incomplete_checkpoint_ids": list(
            metadata_incomplete_ids
        ),
        "base_record_sha256_by_checkpoint_id": base_record_hashes,
        "unchanged_checkpoint_ids": sorted(
            set(base_record_hashes) - set(qualified_ids)
        ),
        "transformations": transformations,
        "license_policy": "complete_existing_registry_metadata_only",
        "checkpoint_spec_file": copy.deepcopy(_RUNTIME_LOCAL_SPEC),
        "repository_registry_mutated": False,
        "projection_persisted_as_global_registry": False,
        "failed_arms_preserved": True,
        "missing_license_normalization_allowed": False,
        "failed_arm_promotion_allowed": False,
        "unsupported_arm_promotion_allowed": False,
        "agent_override_allowed": False,
        "agent_intervention_flags": {name: False for name in AGENT_FLAGS},
    }
    eligibility["eligibility_sha256"] = canonical_sha256(eligibility)
    return projected, eligibility


def audit_qualification(
    path: str | Path,
    *,
    expected_contract: Mapping[str, Any] | None = None,
) -> QualificationDecision:
    evidence_path = Path(path).resolve()
    if not evidence_path.is_file():
        raise QualificationGateError(
            f"qualification evidence is unavailable: {evidence_path}"
        )
    if expected_contract is not None:
        try:
            expected_contract = validate_contract(expected_contract)
        except Exception as exc:
            raise QualificationGateError(
                "expected successor campaign contract is invalid"
            ) from exc
    evidence_bytes = evidence_path.read_bytes()
    evidence_file_sha256 = hashlib.sha256(evidence_bytes).hexdigest()
    document = json.loads(evidence_bytes.decode("utf-8"))
    supplied_sha = document.get("evidence_sha256")
    payload = copy.deepcopy(document)
    payload.pop("evidence_sha256", None)
    if supplied_sha != canonical_sha256(payload):
        raise QualificationGateError("qualification evidence integrity failed")
    policy = _runtime_local_policy(expected_contract)
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
        != FROZEN_PRIOR_QUALIFICATION_EVIDENCE
        or document.get("cpu_model_runs") != 0
        or document.get("smoke_model_runs") != 0
        or document.get("mini_step_runs") != 0
    ):
        raise QualificationGateError(
            "qualification campaign identity or execution policy changed"
        )
    if policy is not None:
        runtime = expected_contract.get("runtime", {})
        launchers = expected_contract.get("launcher_integrity", {})
        repository = Path(str(runtime.get("repository", ""))).resolve()
        runtime_resolver = repository / "src/tao_automl/ptm_runtime.py"
        if (
            str(evidence_path) != policy["qualification_evidence_path"]
            or document.get("automl_contract_sha256")
            != policy["qualification_contract_sha256"]
            or document.get("qualification_controller_sha256")
            != policy["qualification_controller_sha256"]
            or policy["qualification_controller_sha256"]
            != launchers.get("qualification_campaign_sha256")
            or policy["eligibility_gate_sha256"]
            != launchers.get("qualification_gate_sha256")
            or policy["eligibility_gate_sha256"] != sha256_file(__file__)
            or not runtime_resolver.is_file()
            or policy["runtime_resolver_sha256"]
            != sha256_file(runtime_resolver)
            or policy["eligibility_source_commit"]
            != runtime.get("source_commit")
            or policy["wheel_sha256"] != runtime.get("wheel_sha256")
            or policy["sdk_commit"] != runtime.get("sdk_commit")
            or policy["skills_commit"] != runtime.get("skills_commit")
        ):
            raise QualificationGateError(
                "runtime-local policy is not bound to the exact completed "
                "qualification and sealed successor runtime"
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
    metadata_incomplete_ids: list[str] = []
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
                    f"{checkpoint_id} qualification v4 identity changed"
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
            if record.get("status") == "unsupported":
                blockers.append(
                    {
                        "checkpoint_id": checkpoint_id,
                        "code": "registry_explicitly_unsupported",
                        "reason": (
                            "An explicitly unsupported repository record "
                            "cannot be authorized by campaign-local evidence"
                        ),
                    }
                )
            elif policy is not None and not _has_complete_existing_license(
                record
            ):
                metadata_incomplete_ids.append(checkpoint_id)
                exclusions.append(
                    {
                        "checkpoint_id": checkpoint_id,
                        "code": "runtime_metadata_incomplete",
                        "reason": (
                            "The exact PTM passed technical qualification, "
                            "but its base registry record has no complete "
                            "normalized license metadata; the campaign will "
                            "not invent or normalize a license"
                        ),
                        "workflow_sha256": item.workflow_sha256,
                        "pretrained_load_report_sha256": (
                            item.pretrained_load_report_sha256
                        ),
                    }
                )
            elif record.get("status") != "supported" and policy is None:
                blockers.append(
                    {
                        "checkpoint_id": checkpoint_id,
                        "code": "registry_not_supported",
                        "reason": (
                            "Direct full-run success exists, but neither "
                            "repository support nor a separately sealed "
                            "runtime-local eligibility policy is available"
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
                    "No exact PTM has successful positive-load direct-full-run "
                    "evidence and an authorized runtime eligibility path"
                ),
            }
        )
    runtime_registry, runtime_eligibility = _project_runtime_registry(
        base_registry=registry,
        qualified=tuple(qualified),
        metadata_incomplete_ids=tuple(sorted(metadata_incomplete_ids)),
        evidence_path=evidence_path,
        evidence_file_sha256=evidence_file_sha256,
        evidence_sha256=supplied_sha,
        policy=policy,
    )
    decision_payload = {
        "evidence_path": str(evidence_path),
        "evidence_sha256": supplied_sha,
        "qualification_campaign_id": document.get("campaign_id"),
        "qualified": [item.to_dict() for item in qualified],
        "exclusions": exclusions,
        "blockers": blockers,
        "runtime_eligibility": runtime_eligibility,
    }
    return QualificationDecision(
        evidence_path=str(evidence_path),
        evidence_sha256=supplied_sha,
        qualification_campaign_id=str(document.get("campaign_id", "")),
        qualified=tuple(qualified),
        exclusions=tuple(exclusions),
        blockers=tuple(blockers),
        runtime_eligibility=runtime_eligibility,
        decision_sha256=canonical_sha256(decision_payload),
        runtime_registry=runtime_registry,
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
        registry_record = self._decision.runtime_registry.checkpoint(
            request.checkpoint_id
        )
        observed_size = request.checkpoint_path.stat().st_size
        observed_sha = sha256_file(request.checkpoint_path)
        if (
            registry_record.get("status") != "supported"
            or canonical_sha256(_registry_core_identity(registry_record))
            != record.registry_core_identity_sha256
            or registry_record.get("checkpoint_target")
            != record.checkpoint_target
            or request.registry_record != registry_record
            or observed_size != record.source_checkpoint_size_bytes
            or observed_sha != record.source_checkpoint_sha256
            or record.pretrained_loaded_tensor_count < 1
            or record.pretrained_load_component
            != (
                "model"
                if record.checkpoint_target == "train.pretrained_model_path"
                else "backbone"
            )
        ):
            return CheckpointLoadSmokeResult(
                ok=False,
                code="qualification_artifact_or_registry_mismatch",
                reason=(
                    "Live artifact, positive-load receipt, or bound runtime "
                    "registry identity differs "
                    "from direct full-run evidence"
                ),
            )
        return CheckpointLoadSmokeResult(
            ok=True,
            code="direct_full_train_eval_qualification_reused",
            reason=(
                "Exact checkpoint passed full-dataset one-node/eight-GPU "
                "50-epoch training, validation, terminal reload, and "
                "standalone eval with the sealed positive-load runtime overlay"
            ),
            details={
                "cpu_or_smoke_model_job_launched": False,
                "qualification_evidence_sha256": (
                    self._decision.evidence_sha256
                ),
                "runtime_eligibility_sha256": (
                    self._decision.runtime_eligibility.get(
                        "eligibility_sha256"
                    )
                ),
                "projected_registry_sha256": (
                    self._decision.runtime_registry.document_sha256
                ),
                "workflow_sha256": record.workflow_sha256,
                "pretrained_load_component": (
                    record.pretrained_load_component
                ),
                "pretrained_loaded_tensor_count": (
                    record.pretrained_loaded_tensor_count
                ),
                "pretrained_loaded_keyset_sha256": (
                    record.pretrained_loaded_keyset_sha256
                ),
                "pretrained_load_report_sha256": (
                    record.pretrained_load_report_sha256
                ),
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
