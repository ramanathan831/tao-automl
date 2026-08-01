from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from tao_automl.ptm_registry import canonical_sha256, load_ptm_registry

from . import (
    campaign_contract,
    manifest_generator,
    qualification_campaign,
    qualification_gate,
    run_campaign,
)
from .qualification_gate import (
    QualificationGateError,
    QualificationLoadEvidence,
    audit_qualification,
)


HERE = Path(__file__).resolve().parent
SKILL_DIR = (
    manifest_generator.DEFAULT_SKILLS
    / "skills/models/tao-train-segformer"
)
DATASET_STAGE_MANIFEST = manifest_generator.DEFAULT_STAGE_MANIFEST
if not DATASET_STAGE_MANIFEST.is_file():
    # The dataset-staging branch is an explicit integration dependency. This
    # fallback keeps this isolated worktree testable until both commits land.
    DATASET_STAGE_MANIFEST = Path(
        "/localhome/local-rarunachalam/.tao/worktrees/"
        "tao-automl-segmentation-datasets/experiments/"
        "cross_model_automl_20260729/segmentation_datasets/"
        "dataset_stage_manifest.v1.json"
    )


def _dataset() -> dict:
    value = manifest_generator.dataset_record(
        manifest_generator.DEFAULT_DATASET_MANIFEST,
        DATASET_STAGE_MANIFEST,
    )
    return value


def _runtime(tmp_path: Path) -> dict:
    return {
        "repository": str(Path(__file__).resolve().parents[3]),
        "source_commit": "c" * 40,
        "source_dirty": False,
        "wheel_path": str(manifest_generator.DEFAULT_WHEEL),
        "wheel_sha256": manifest_generator.EXPECTED_WHEEL_SHA256,
        "sdk_dir": str(manifest_generator.DEFAULT_SDK),
        "sdk_commit": manifest_generator.EXPECTED_SDK_COMMIT,
        "skills_repository": str(manifest_generator.DEFAULT_SKILLS),
        "skills_commit": manifest_generator.EXPECTED_SKILLS_COMMIT,
        "skill_dir": str(SKILL_DIR),
        "qualification_evidence_path": str(
            tmp_path / "qualification.json"
        ),
        "ptm_stage_manifest_path": str(tmp_path / "ptms.json"),
        "partition": campaign_contract.FROZEN_SLURM_PARTITION,
        "account": "edgeai_tao-ptm_image-foundation-model-clip",
        "base_results_dir": "/lustre/fsw/portfolios/edgeai/users/rarunachalam",
        "container_mounts": "/lustre",
        "time_hours": campaign_contract.FROZEN_SLURM_TIME_HOURS,
        "timeout_hours": campaign_contract.FROZEN_SLURM_TIMEOUT_HOURS,
        "max_job_retries": 10,
        "hardware_contract": copy.deepcopy(
            campaign_contract.FROZEN_HARDWARE
        ),
    }


@pytest.fixture
def contract(tmp_path: Path) -> dict:
    value = campaign_contract.build_preregistered_contract(
        campaign_id="segformer-test",
        dataset=_dataset(),
        skill_dir=str(SKILL_DIR),
        runtime=_runtime(tmp_path),
    )
    value.pop("contract_sha256")
    value["launcher_integrity"] = {
        "campaign_contract_sha256": campaign_contract.sha256_file(
            HERE / "campaign_contract.py"
        ),
        "qualification_gate_sha256": campaign_contract.sha256_file(
            HERE / "qualification_gate.py"
        ),
        "qualification_campaign_sha256": campaign_contract.sha256_file(
            HERE / "qualification_campaign.py"
        ),
        "run_campaign_sha256": campaign_contract.sha256_file(
            HERE / "run_campaign.py"
        ),
        "segformer_latency_worker_sha256": (
            campaign_contract.sha256_file(
                HERE / "segformer_latency_worker.py"
            )
        )
    }
    value["contract_sha256"] = canonical_sha256(value)
    return campaign_contract.validate_contract(value)


def _workflow(
    checkpoint_id: str,
    *,
    success: bool,
    metric: float = 0.4,
) -> dict:
    record = load_ptm_registry().checkpoint(checkpoint_id)
    if not success:
        value = {
            "schema_version": 2,
            "qualification_revision": (
                campaign_contract.QUALIFICATION_REVISION
            ),
            "checkpoint_id": checkpoint_id,
            "status": "failure",
            "terminal": True,
            "failure_preserved": True,
            "failure_code": "direct_full_run_failed",
            "failure_reason": "frozen test failure",
            "recipe_fidelity": copy.deepcopy(
                campaign_contract.FROZEN_QUALIFICATION_FIDELITY
            ),
            "runtime_overlay": copy.deepcopy(
                campaign_contract.FROZEN_QUALIFICATION_RUNTIME_OVERLAY
            ),
        }
        value["workflow_sha256"] = canonical_sha256(value)
        return value
    value = {
        "schema_version": 2,
        "qualification_revision": campaign_contract.QUALIFICATION_REVISION,
        "checkpoint_id": checkpoint_id,
        "status": "success",
        "terminal": True,
        "failure_preserved": False,
        "source_checkpoint": {
            "path": f"/lustre/ptms/{checkpoint_id}.pth",
            "size_bytes": record["expected_size_bytes"],
            "sha256": "a" * 64,
        },
        "train": {
            "status": "Complete",
            "full_dataset": True,
            "training_epochs": (
                campaign_contract.FROZEN_QUALIFICATION_TRAINING_EPOCHS
            ),
            "validation_interval": 1,
            "validation_record_count": (
                campaign_contract.FROZEN_QUALIFICATION_TRAINING_EPOCHS
            ),
            "recipe_fidelity": copy.deepcopy(
                campaign_contract.FROZEN_QUALIFICATION_FIDELITY
            ),
            "runtime_overlay": copy.deepcopy(
                campaign_contract.FROZEN_QUALIFICATION_RUNTIME_OVERLAY
            ),
            "job": {"runtime_overlay_required": True},
            "nodes": 1,
            "gpus": 8,
            "val_miou": metric,
            "terminal_checkpoint": {
                "path": (
                    "/lustre/results/"
                    f"{checkpoint_id}/model_epoch_049_step_09150.pth"
                ),
                "size_bytes": 123,
                "sha256": "b" * 64,
                "training_epochs": 50,
                "terminal_epoch_index": 49,
                "naming_contract": "model_epoch_049_step_numeric",
                "ambiguity_policy": "fail_closed",
            },
        },
        "evaluation": {
            "status": "Complete",
            "full_validation_split": True,
            "runtime_overlay": copy.deepcopy(
                campaign_contract.FROZEN_QUALIFICATION_RUNTIME_OVERLAY
            ),
            "job": {"runtime_overlay_required": True},
            "nodes": 1,
            "gpus": 8,
            "test_miou": metric,
        },
        "agent_intervention_flags": {
            name: False for name in campaign_contract.AGENT_FLAGS
        },
        "recipe_fidelity": copy.deepcopy(
            campaign_contract.FROZEN_QUALIFICATION_FIDELITY
        ),
        "runtime_overlay": copy.deepcopy(
            campaign_contract.FROZEN_QUALIFICATION_RUNTIME_OVERLAY
        ),
    }
    value["workflow_sha256"] = canonical_sha256(value)
    return value


def _qualification_document(success_id: str | None = None) -> dict:
    snapshot = campaign_contract.segformer_registry_snapshot()
    workflows = [
        _workflow(
            record["id"],
            success=record["id"] == success_id,
        )
        for record in snapshot["records"]
    ]
    value = {
        "schema_version": 2,
        "qualification_revision": campaign_contract.QUALIFICATION_REVISION,
        "campaign_id": campaign_contract.QUALIFICATION_CAMPAIGN_ID,
        "model": "segformer",
        "task": "semantic_segmentation",
        "registry_sha256": snapshot["registry_sha256"],
        "sqsh_sha256": campaign_contract.FROZEN_SQSH["sha256"],
        "recipe_fidelity": copy.deepcopy(
            campaign_contract.FROZEN_QUALIFICATION_FIDELITY
        ),
        "runtime_overlay": copy.deepcopy(
            campaign_contract.FROZEN_QUALIFICATION_RUNTIME_OVERLAY
        ),
        "prior_revision_evidence": copy.deepcopy(
            campaign_contract.FROZEN_PRIOR_QUALIFICATION_EVIDENCE
        ),
        "cpu_model_runs": 0,
        "smoke_model_runs": 0,
        "mini_step_runs": 0,
        "workflows": workflows,
    }
    value["evidence_sha256"] = canonical_sha256(value)
    return value


def _fake_qualification_stage(contract: dict) -> dict:
    rows = []
    registry = load_ptm_registry()
    for record_summary in campaign_contract.segformer_registry_snapshot()[
        "records"
    ]:
        record = registry.checkpoint(record_summary["id"])
        checkpoint_path = (
            "/lustre/segformer-qualification/ptms/"
            f"{record['id']}/{record['source']['member']}"
        )
        specifications = qualification_campaign.qualification_specs(
            contract,
            record,
            checkpoint_path,
        )
        specs = {}
        for action, document in specifications.items():
            content = yaml.safe_dump(document, sort_keys=True).encode()
            digest = hashlib.sha256(content).hexdigest()
            specs[action] = {
                "action": action,
                "document": document,
                "document_sha256": canonical_sha256(document),
                "raw_yaml_sha256": digest,
                "size_bytes": len(content),
                "base_template": {
                    "path": str(
                        SKILL_DIR
                        / "references"
                        / f"spec_template_{action}.yaml"
                    ),
                    "sha256": campaign_contract.sha256_file(
                        SKILL_DIR
                        / "references"
                        / f"spec_template_{action}.yaml"
                    ),
                },
                "local_path": f"/tmp/{record['id']}-{action}.yaml",
                "lustre": {
                    "path": (
                        "/lustre/segformer-qualification/specs/"
                        f"{record['id']}/{action}.yaml"
                    ),
                    "size_bytes": len(content),
                    "sha256": digest,
                    "mode": "444",
                    "cache_hit": False,
                },
            }
        observed_sha = hashlib.sha256(record["id"].encode()).hexdigest()
        rows.append(
            {
                "checkpoint_id": record["id"],
                "workflow_id": qualification_campaign._workflow_id(
                    record["id"]
                ),
                "registry_status_at_stage": record["status"],
                "registry_record_sha256": canonical_sha256(record),
                "registry_core_identity": (
                    qualification_campaign.registry_core_identity(record)
                ),
                "registry_core_identity_sha256": canonical_sha256(
                    qualification_campaign.registry_core_identity(record)
                ),
                "source": copy.deepcopy(record["source"]),
                "checkpoint_target": record["checkpoint_target"],
                "backbone": record["backbone"],
                "expected_size_bytes": record["expected_size_bytes"],
                "registered_sha256": record.get("sha256"),
                "observed_sha256": observed_sha,
                "verification_mode": (
                    "immutable_identity_observed_sha256"
                ),
                "source_identity_sha256": canonical_sha256(
                    record["source"]
                ),
                "access_probe": {
                    "ok": True,
                    "code": "accessible",
                    "remote_size_bytes": record[
                        "expected_size_bytes"
                    ],
                },
                "checkpoint_specific_source_spec": {
                    "available": False,
                    "registry_field_present": (
                        "checkpoint_spec_file" in record
                    ),
                    "reason": (
                        "The official SegFormer registry record publishes no "
                        "checkpoint-specific YAML; the staged specs are "
                        "generated from the sealed TAO templates, frozen VOC "
                        "profile, and exact registry checkpoint target."
                    ),
                },
                "checkpoint": {
                    "path": checkpoint_path,
                    "size_bytes": record["expected_size_bytes"],
                    "sha256": observed_sha,
                    "mode": "444",
                    "cache_hit": False,
                },
                "specs": specs,
            }
        )
    value = {
        "schema_version": 2,
        "qualification_revision": campaign_contract.QUALIFICATION_REVISION,
        "campaign_id": qualification_campaign.QUALIFICATION_CAMPAIGN_ID,
        "automl_contract_sha256": contract["contract_sha256"],
        "created_at_utc": "2026-07-31T00:00:00Z",
        "model": "segformer",
        "task": "semantic_segmentation",
        "registry_sha256": contract["ptm_inventory"]["registry_sha256"],
        "registry_version": contract["ptm_inventory"]["registry_version"],
        "source_policy": (
            "all_13_official_registry_arms_without_manual_exclusion"
        ),
        "dataset": {
            "prepared_root": contract["dataset"]["prepared_root"],
            "content_sha256": contract["dataset"]["content_sha256"],
            "stage_manifest_sha256": contract["dataset"][
                "stage_manifest_sha256"
            ],
            "train_pairs": 1464,
            "validation_pairs": 1449,
        },
        "runtime": {
            "sqsh_path": contract["sqsh"]["path"],
            "sqsh_sha256": contract["sqsh"]["sha256"],
            "sdk_commit": contract["runtime"]["sdk_commit"],
            "skills_commit": contract["runtime"]["skills_commit"],
            "source_commit": contract["runtime"]["source_commit"],
            "partition": contract["runtime"]["partition"],
            "time_hours": contract["runtime"]["time_hours"],
            "timeout_hours": contract["runtime"]["timeout_hours"],
            "nodes_per_workflow": 1,
            "gpus_per_workflow": 8,
            "required_gpu": copy.deepcopy(
                campaign_contract.FROZEN_HARDWARE
            ),
            "runtime_overlay": copy.deepcopy(
                campaign_contract.FROZEN_QUALIFICATION_RUNTIME_OVERLAY
            ),
        },
        "recipe_fidelity": copy.deepcopy(
            campaign_contract.FROZEN_QUALIFICATION_FIDELITY
        ),
        "prior_revision_evidence": copy.deepcopy(
            campaign_contract.FROZEN_PRIOR_QUALIFICATION_EVIDENCE
        ),
        "ptms": rows,
        "execution": {
            "operation": (
                "data_only_download_checksum_spec_generation_and_"
                "lustre_publication"
            ),
            "cpu_model_runs": 0,
            "gpu_model_runs": 0,
            "smoke_model_runs": 0,
            "mini_step_runs": 0,
            "checkpoint_loads": 0,
            "scheduler_jobs_submitted": 0,
            "fallback_ptms_used": 0,
            "manually_excluded_ptms": 0,
        },
        "agent_intervention_flags": {
            name: False for name in campaign_contract.AGENT_FLAGS
        },
    }
    value["stage_manifest_sha256"] = canonical_sha256(value)
    return value


def test_exact_tao_identifier_action_and_primary_metric():
    info = json.loads(
        json.dumps(
            __import__("yaml").safe_load(
                (SKILL_DIR / "references/skill_info.yaml").read_text()
            )
        )
    )
    assert info["network_arch"] == "segformer"
    assert info["actions"]["train"]["command"] == (
        "segformer train -e {config_path}"
    )
    assert campaign_contract.mode_settings("x", "accuracy")[
        "accuracy_metric"
    ] == "val_miou"


def test_only_common_train_parameters_are_searched():
    evidence = campaign_contract.validate_packaged_train_schema(SKILL_DIR)
    assert tuple(evidence["explicit_search_parameters"]) == (
        "train.optim.lr",
        "train.optim.weight_decay",
    )
    assert "dataset.segment.img_size" not in campaign_contract.SEARCH_PARAMETERS
    assert "model.backbone.type" not in campaign_contract.SEARCH_PARAMETERS
    assert campaign_contract.SEARCH_SPACE == {
        "train.optim.lr": {
            "type": "float",
            "minimum": 2e-5,
            "maximum": 6e-4,
            "scale": "linear",
        },
        "train.optim.weight_decay": {
            "type": "float",
            "minimum": 1e-4,
            "maximum": 0.1,
            "scale": "linear",
        },
    }


def test_complete_voc2012_record_and_loss_preserving_palette():
    dataset = _dataset()
    assert dataset["train_image_count"] == dataset["train_mask_count"] == 1464
    assert (
        dataset["validation_image_count"]
        == dataset["validation_mask_count"]
        == 1449
    )
    assert dataset["file_manifest_entry_count"] == 5827
    assert dataset["manifest_sha256"] == (
        "051ab20215b8e6976763ac82a3db20a68264759edef3d62fd0c8553c501123ff"
    )
    assert dataset["content_sha256"] == (
        "815b5d01b625238b449c4bca828bf96107b367f0f4d5d8a31d2f97c6161a5de0"
    )
    assert dataset["stage_manifest_sha256"] == (
        "437ff12490637950707b9b951d820ea34d38b926080a478a5d182c2d284a0c5d"
    )
    assert dataset["remote_read_only"] is True
    assert dataset["remote_writable_entries_after_lock"] == 0
    palette = campaign_contract.voc_palette()
    assert [item["label_id"] for item in palette] == [*range(21), 255]
    assert all(item["rgb"] == [item["label_id"]] for item in palette)


def test_search_profile_remains_frozen_at_v1_fidelity():
    profile = campaign_contract.profile_overrides(
        _dataset()["prepared_root"]
    )
    segment = profile["dataset"]["segment"]
    train = profile["train"]
    assert segment["dataset"] == "SFDataset"
    assert segment["num_classes"] == 21
    assert segment["label_transform"] == "None"
    assert segment["img_size"] == 512
    assert train["num_gpus"] == 8
    assert train["gpu_ids"] == list(range(8))
    assert train["num_nodes"] == 1
    assert train["num_epochs"] == 10
    assert train["use_distributed_sampler"] is False
    assert train["validation_interval"] == 1
    assert train["tensorboard"]["enabled"] is False


def test_qualification_v3_uses_official_multiclass_fidelity_uniformly():
    profile = campaign_contract.qualification_profile_overrides(
        _dataset()["prepared_root"]
    )
    segment = profile["dataset"]["segment"]
    train = profile["train"]
    fidelity = campaign_contract.FROZEN_QUALIFICATION_FIDELITY
    assert fidelity["source_recipe"].endswith(
        "segformer/experiment_specs/experiment_multi-class.yaml"
    )
    assert fidelity["source_recipe_sha256"] == (
        "210b6b6c4952289e3dbc1f025b3f0b8f17a073702290cb565796ed6c6ea36b21"
    )
    assert train["num_epochs"] == 50
    assert train["checkpoint_interval"] == 50
    assert train["validation_interval"] == 1
    assert train["optim"] == {
        "optim": "adamw",
        "lr": 1.0e-4,
        "weight_decay": 5.0e-4,
    }
    assert segment["augmentation"]["random_color"]["enable"] is False
    assert segment["augmentation"]["with_random_blur"] is False
    assert train["use_distributed_sampler"] is True


def test_qualification_v3_paths_preserve_frozen_v1_and_v2_evidence():
    v1 = campaign_contract.FROZEN_V1_QUALIFICATION_EVIDENCE
    v2 = campaign_contract.FROZEN_V2_QUALIFICATION_EVIDENCE
    prior = campaign_contract.FROZEN_PRIOR_QUALIFICATION_EVIDENCE
    assert prior == [v1, v2]
    assert v1["campaign_id"].endswith("-v1")
    assert v1["status"] == "terminal_with_failures"
    assert v1["successful_workflows"] == 0
    assert v1["failed_workflows"] == 13
    assert v1["preserve_immutable"] is True
    assert v1["reuse_for_v2"] is False
    assert "/segformer_voc2012_ptm_qualification_v1/" in (
        v1["completion_path"]
    )
    assert v2["campaign_id"].endswith("-v2")
    assert v2["status"] == "terminal_with_failures"
    assert v2["successful_workflows"] == 0
    assert v2["failed_workflows"] == 13
    assert v2["controller_failure_workflows"] == 12
    assert v2["runtime_failure_workflows"] == 1
    assert v2["preserve_immutable"] is True
    assert v2["reuse_for_v3"] is False
    assert campaign_contract.sha256_file(v2["completion_path"]) == (
        v2["completion_whole_file_sha256"]
    )
    assert campaign_contract.sha256_file(
        v2["ptm_stage_manifest_path"]
    ) == v2["ptm_stage_manifest_whole_file_sha256"]
    assert campaign_contract.sha256_file(
        v2["launch_preflight_path"]
    ) == v2["launch_preflight_whole_file_sha256"]
    assert campaign_contract.sha256_file(
        v2["automatic_handoff_path"]
    ) == v2["automatic_handoff_whole_file_sha256"]
    assert qualification_campaign.QUALIFICATION_CAMPAIGN_ID.endswith("-v3")
    assert qualification_campaign.DEFAULT_CONTRACT.name == "campaign.v3.json"
    assert run_campaign.DEFAULT_CONTRACT.name == "campaign.v3.json"
    assert "qualification_v3" in str(
        qualification_campaign.DEFAULT_RUNTIME_ROOT
    )
    assert "qualification_v3" in str(
        qualification_campaign.DEFAULT_LOCAL_CACHE
    )
    assert "qualification_v3" in str(
        qualification_campaign.DEFAULT_LUSTRE_INPUT_ROOT
    )
    assert manifest_generator.DEFAULT_QUALIFICATION != Path(
        v2["completion_path"]
    )
    assert manifest_generator.DEFAULT_PTM_STAGE_MANIFEST != Path(
        v2["ptm_stage_manifest_path"]
    )


def test_qualification_v3_binds_combined_runtime_overlay(contract):
    overlay = campaign_contract.FROZEN_QUALIFICATION_RUNTIME_OVERLAY
    assert overlay["combined_commit"] == (
        "3b1e073571f3bbf3702b0ae837e9279ad12f4286"
    )
    assert overlay["source_commit"] == overlay["combined_commit"]
    assert overlay["archive_sha256"] == (
        "b055100d0d3e9e8c5daf94dfd4caf3cccacfb54fbebb423129fb5832066e420b"
    )
    assert overlay["required_actions"] == ["train", "evaluate"]
    policy = contract["qualification_policy"]
    assert policy["revision"] == 3
    assert policy["campaign_id"].endswith("-v3")
    assert policy["training_epochs"] == 50
    assert policy["recipe_fidelity"] == (
        campaign_contract.FROZEN_QUALIFICATION_FIDELITY
    )
    assert policy["runtime_overlay"] == overlay
    assert policy["prior_revision_evidence"] == (
        campaign_contract.FROZEN_PRIOR_QUALIFICATION_EVIDENCE
    )
    assert contract["search"]["training_epochs"] == 10


def test_voc_metric_sanity_is_separate_from_product_selection(contract):
    gate = contract["validation_sanity_gate"]
    assert gate["metric"] == "val_miou"
    assert gate["minimum"] == 0.10
    assert gate["role"] == (
        "experiment_correctness_gate_not_product_selection"
    )
    assert gate["low_finite_metric_automatically_accepted"] is False


def test_all_official_ptms_are_hierarchical_arms():
    snapshot = campaign_contract.segformer_registry_snapshot()
    assert snapshot["record_count"] == 13
    assert len({item["id"] for item in snapshot["records"]}) == 13
    assert all(item["source"]["official"] is True for item in snapshot["records"])
    assert all(
        item["checkpoint_target"]
        in {
            "train.pretrained_model_path",
            "model.backbone.pretrained_backbone_path",
        }
        for item in snapshot["records"]
    )
    assert set(snapshot["supported_ids"]).isdisjoint(
        snapshot["unverified_ids"]
    )


def test_budget_covers_two_calibration_points_per_official_arm():
    arm_count = campaign_contract.segformer_registry_snapshot()[
        "record_count"
    ]
    assert campaign_contract.FROZEN_CANDIDATE_BUDGET >= (
        2 * arm_count + 4
    )


def test_mode_objectives_are_independent(contract):
    modes = {
        item["mode"]: item["settings"] for item in contract["modes"]
    }
    assert modes["accuracy"]["selection_mode"] == "accuracy"
    assert "latency_accuracy_retention" not in modes["accuracy"]
    assert modes["latency"]["latency_accuracy_retention"] == {
        "type": "relative",
        "retained_fraction": 0.9,
        "reference": "accuracy_winner",
    }
    assert modes["latency"]["objective_acquisition"][
        "calibration_points"
    ] == 2
    assert modes["multi_objective"]["selection_mode"] == "multi_objective"
    assert "latency_accuracy_retention" not in modes["multi_objective"]
    assert modes["multi_objective"]["multi_objective_min_accuracy"] is None
    assert all(item["observation_sharing"] is False for item in contract["modes"])


def test_contract_is_pinned_sqsh_eight_gpu_and_zero_local_model_runs(contract):
    assert contract["sqsh"] == campaign_contract.FROZEN_SQSH
    assert contract["execution"]["container_mode"] == "pinned_sqsh"
    assert contract["execution"]["nodes_per_child"] == 1
    assert contract["execution"]["gpus_per_child"] == 8
    assert contract["execution"]["cpu_runs"] == 0
    assert contract["execution"]["smoke_runs"] == 0
    assert contract["execution"]["local_model_runs"] == 0
    assert contract["qualification_policy"]["cpu_model_runs"] == 0
    assert contract["qualification_policy"]["smoke_model_runs"] == 0
    assert contract["qualification_policy"]["mini_step_runs"] == 0


def test_latency_protocol_is_4000_real_validation_samples(contract):
    protocol = contract["latency_protocol"]
    assert protocol["warmup_iterations"] == 50
    assert protocol["repeated_rounds"] == 5
    assert protocol["timed_iterations"] == 100
    assert protocol["expected_replicas"] == 8
    assert protocol["raw_samples_per_candidate"] == 4000
    source = (HERE / "segformer_latency_worker.py").read_text()
    assert "SFDataModule" in source
    assert "test_dataloader" in source
    assert "model(preloaded[" in source
    assert "torch.randn" not in source
    assert "torch.rand(" not in source


def test_model_imports_and_execution_live_only_below_worker_main():
    tree = ast.parse(
        (HERE / "segformer_latency_worker.py").read_text(encoding="utf-8")
    )
    module_imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert all(
        not (
            isinstance(node, ast.Import)
            and any(alias.name == "torch" for alias in node.names)
        )
        for node in module_imports
    )
    assert all(
        not (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("nvidia_tao_pytorch")
        )
        for node in module_imports
    )


def test_unverified_full_run_success_cannot_bypass_registry(tmp_path: Path):
    snapshot = campaign_contract.segformer_registry_snapshot()
    success_id = snapshot["records"][0]["id"]
    path = tmp_path / "qualification.json"
    path.write_text(
        json.dumps(_qualification_document(success_id)),
        encoding="utf-8",
    )
    decision = audit_qualification(path)
    if load_ptm_registry().checkpoint(success_id)["status"] == "unverified":
        assert success_id not in decision.checkpoint_ids
        assert any(
            item["checkpoint_id"] == success_id
            and item["code"] == "registry_not_supported"
            for item in decision.blockers
        )
        with pytest.raises(QualificationGateError):
            QualificationLoadEvidence(decision)


def test_prior_evidence_is_preserved_but_cannot_satisfy_v3_gate(
    tmp_path: Path,
):
    document = _qualification_document()
    document["schema_version"] = 1
    document["qualification_revision"] = 1
    document["campaign_id"] = (
        campaign_contract.FROZEN_V1_QUALIFICATION_EVIDENCE["campaign_id"]
    )
    document["evidence_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in document.items()
            if key != "evidence_sha256"
        }
    )
    path = tmp_path / "qualification.v1.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(
        QualificationGateError,
        match="campaign identity or execution policy changed",
    ):
        audit_qualification(path)

    document = _qualification_document()
    document["qualification_revision"] = 2
    document["campaign_id"] = (
        campaign_contract.FROZEN_V2_QUALIFICATION_EVIDENCE["campaign_id"]
    )
    document["evidence_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in document.items()
            if key != "evidence_sha256"
        }
    )
    path = tmp_path / "qualification.v2.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(
        QualificationGateError,
        match="campaign identity or execution policy changed",
    ):
        audit_qualification(path)


def test_terminal_ptm_failures_are_preserved_as_exclusions(tmp_path: Path):
    path = tmp_path / "qualification.json"
    path.write_text(
        json.dumps(_qualification_document()),
        encoding="utf-8",
    )
    decision = audit_qualification(path)
    assert len(decision.exclusions) == 13
    assert all(
        item["code"] == "direct_full_run_failed"
        for item in decision.exclusions
    )
    assert any(
        item["code"] == "no_runtime_qualified_ptm"
        for item in decision.blockers
    )


def test_low_finite_miou_does_not_pass_ptm_qualification(tmp_path: Path):
    document = _qualification_document()
    checkpoint_id = document["workflows"][0]["checkpoint_id"]
    document["workflows"][0] = _workflow(
        checkpoint_id,
        success=True,
        metric=0.09,
    )
    document["evidence_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in document.items()
            if key != "evidence_sha256"
        }
    )
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    decision = audit_qualification(path)
    assert any(
        item["checkpoint_id"] == checkpoint_id
        and item["code"] == "invalid_success_evidence"
        and "0.10 mIoU" in item["reason"]
        for item in decision.blockers
    )


def _write_gate_cell(
    root: Path,
    contract_sha256: str,
    mode: str,
    passed: bool,
) -> None:
    run_campaign.atomic_json(
        root / "first_candidate_gate" / f"{mode}.json",
        {
            "schema_version": 1,
            "contract_sha256": contract_sha256,
            "mode": mode,
            "candidate_id": f"{mode}_rec_0",
            "passed": passed,
        },
    )


def test_first_candidate_gate_releases_automatically_only_after_all_pass(
    tmp_path: Path,
):
    processes = {
        mode: SimpleNamespace(is_alive=lambda: True)
        for mode in campaign_contract.MODES
    }
    for mode in campaign_contract.MODES[:2]:
        _write_gate_cell(tmp_path, "a" * 64, mode, True)
    assert (
        run_campaign._release_first_candidate_gate(
            tmp_path, processes, "a" * 64
        )
        is None
    )
    _write_gate_cell(
        tmp_path, "a" * 64, campaign_contract.MODES[2], True
    )
    release = run_campaign._release_first_candidate_gate(
        tmp_path, processes, "a" * 64
    )
    assert release["release_remaining_budget"] is True
    assert release["generated_automatically"] is True


def test_first_candidate_gate_fails_closed_on_any_failed_mode(
    tmp_path: Path,
):
    processes = {
        mode: SimpleNamespace(is_alive=lambda: True)
        for mode in campaign_contract.MODES
    }
    for index, mode in enumerate(campaign_contract.MODES):
        _write_gate_cell(tmp_path, "b" * 64, mode, index != 1)
    release = run_campaign._release_first_candidate_gate(
        tmp_path, processes, "b" * 64
    )
    assert release["release_remaining_budget"] is False


def test_launch_plan_is_automatic_and_does_not_launch(contract):
    plan = run_campaign.launch_plan(
        contract,
        ready=False,
        blockers=[{"code": "ptm_qualification_not_ready"}],
    )
    assert plan["launch_authorized"] is False
    assert plan["automatic_trigger"] is True
    assert plan["cpu_or_smoke_model_jobs"] == 0
    assert plan["first_candidate_gate"]["automatic_release"] is True
    assert plan["first_candidate_gate"][
        "remaining_candidates_per_mode"
    ] == 29
    assert plan["resources_per_child"]["gpus"] == 8


def test_local_seal_is_revalidated_before_launch(
    contract,
    monkeypatch: pytest.MonkeyPatch,
):
    runtime = contract["runtime"]

    def fake_git(repository: Path, *arguments: str) -> str:
        if arguments == ("status", "--porcelain"):
            return ""
        if Path(repository).resolve() == Path(runtime["sdk_dir"]).resolve():
            return runtime["sdk_commit"]
        if (
            Path(repository).resolve()
            == Path(runtime["skills_repository"]).resolve()
        ):
            return runtime["skills_commit"]
        return runtime["source_commit"]

    monkeypatch.setattr(run_campaign, "_git", fake_git)
    evidence = run_campaign.verify_local_contract(contract)
    assert evidence["source_commit"] == runtime["source_commit"]
    assert evidence["artifacts"]["wheel"]["sha256"] == (
        manifest_generator.EXPECTED_WHEEL_SHA256
    )

    changed = copy.deepcopy(contract)
    changed["runtime"]["wheel_sha256"] = "0" * 64
    with pytest.raises(run_campaign.CampaignExecutionError):
        run_campaign.verify_local_contract(changed)


def test_archive_order_cannot_enter_campaign_search_contract(contract):
    modes = contract["modes"]
    assert [item["mode"] for item in modes] == [
        "accuracy",
        "latency",
        "multi_objective",
    ]
    assert all(item["initial_observation_ids"] == [] for item in modes)
    assert len(
        {item["observation_namespace"] for item in modes}
    ) == 3
    assert contract["execution"]["shared_archive"] is False


def test_contract_integrity_rejects_mutation(contract):
    changed = copy.deepcopy(contract)
    changed["execution"]["gpus_per_child"] = 1
    with pytest.raises(campaign_contract.CampaignContractError):
        campaign_contract.validate_contract(changed)

    changed = copy.deepcopy(contract)
    changed["qualification_policy"]["recipe_fidelity"][
        "learning_rate"
    ] = 2.0e-4
    changed["contract_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in changed.items()
            if key != "contract_sha256"
        }
    )
    with pytest.raises(
        campaign_contract.CampaignContractError,
        match="qualification v3 fidelity or provenance changed",
    ):
        campaign_contract.validate_contract(changed)


def test_qualification_plan_contains_every_official_arm_without_fallback(
    contract,
):
    plan = qualification_campaign.qualification_plan(contract)
    assert plan["workflow_count"] == 13
    assert plan["schema_version"] == 2
    assert plan["qualification_revision"] == 3
    assert plan["workflow"] == (
        "full_voc2012_50_epoch_train_then_standalone_full_validation"
    )
    assert plan["recipe_fidelity"] == (
        campaign_contract.FROZEN_QUALIFICATION_FIDELITY
    )
    assert plan["runtime_overlay"] == (
        campaign_contract.FROZEN_QUALIFICATION_RUNTIME_OVERLAY
    )
    assert plan["checkpoint_ids"] == [
        item["id"]
        for item in campaign_contract.segformer_registry_snapshot()[
            "records"
        ]
    ]
    assert plan["all_workflows_independent"] is True
    assert plan[
        "all_workflows_submitted_without_result_driven_exclusion"
    ] is True
    assert plan["terminal_failures_preserved"] is True
    assert plan["replacement_workflows_submitted"] is False
    assert plan["resources_per_job"] == {
        "nodes": 1,
        "gpus": 8,
        "gpu": "NVIDIA A100-SXM4-80GB",
        "partition": "polar3",
        "time_hours": 4.0,
        "container": campaign_contract.FROZEN_SQSH["path"],
    }
    assert plan["cpu_model_runs"] == 0
    assert plan["smoke_model_runs"] == 0
    assert plan["mini_step_runs"] == 0


def test_qualification_specs_bind_only_the_registered_checkpoint_target(
    contract,
):
    stage = qualification_campaign.validate_stage_manifest(
        _fake_qualification_stage(contract),
        contract=contract,
    )
    target_counts = {
        "train.pretrained_model_path": 0,
        "model.backbone.pretrained_backbone_path": 0,
    }
    for row in stage["ptms"]:
        target_counts[row["checkpoint_target"]] += 1
        checkpoint = row["checkpoint"]["path"]
        train = row["specs"]["train"]["document"]
        train_ptm = train["train"]["pretrained_model_path"]
        backbone_ptm = train["model"]["backbone"][
            "pretrained_backbone_path"
        ]
        if row["checkpoint_target"] == "train.pretrained_model_path":
            assert train_ptm == checkpoint
            assert backbone_ptm == ""
        else:
            assert train_ptm == ""
            assert backbone_ptm == checkpoint
        assert train["train"]["num_epochs"] == 50
        assert train["train"]["checkpoint_interval"] == 50
        assert train["train"]["validation_interval"] == 1
        assert train["train"]["num_gpus"] == 8
        assert train["train"]["optim"]["lr"] == 1.0e-4
        assert train["train"]["optim"]["weight_decay"] == 5.0e-4
        assert train["train"]["use_distributed_sampler"] is True
        assert train["dataset"]["segment"]["augmentation"][
            "random_color"
        ]["enable"] is False
        assert train["dataset"]["segment"]["augmentation"][
            "with_random_blur"
        ] is False
        assert train["dataset"]["segment"]["root_dir"] == (
            contract["dataset"]["prepared_root"]
        )
        evaluate = row["specs"]["evaluate"]["document"]
        assert evaluate["evaluate"]["checkpoint"] == (
            qualification_campaign.EVALUATION_CHECKPOINT_SENTINEL
        )
        assert evaluate["train"]["num_epochs"] == 50
        assert evaluate["train"]["optim"]["lr"] == 1.0e-4
        assert evaluate["train"]["optim"]["weight_decay"] == 5.0e-4
        assert evaluate["train"]["use_distributed_sampler"] is True
        assert evaluate["dataset"]["segment"]["augmentation"][
            "random_color"
        ]["enable"] is False
        assert evaluate["dataset"]["segment"]["augmentation"][
            "with_random_blur"
        ] is False
    assert target_counts == {
        "train.pretrained_model_path": 4,
        "model.backbone.pretrained_backbone_path": 9,
    }


def test_ptm_stage_manifest_rejects_missing_or_writable_inputs(contract):
    stage = _fake_qualification_stage(contract)
    qualification_campaign.validate_stage_manifest(stage, contract=contract)

    missing = copy.deepcopy(stage)
    missing["ptms"].pop()
    missing["stage_manifest_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in missing.items()
            if key != "stage_manifest_sha256"
        }
    )
    with pytest.raises(
        qualification_campaign.CampaignExecutionError,
        match="campaign contract changed",
    ):
        qualification_campaign.validate_stage_manifest(
            missing,
            contract=contract,
        )

    writable = copy.deepcopy(stage)
    writable["ptms"][0]["checkpoint"]["mode"] = "644"
    writable["stage_manifest_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in writable.items()
            if key != "stage_manifest_sha256"
        }
    )
    with pytest.raises(
        qualification_campaign.CampaignExecutionError,
        match="checkpoint identity differs",
    ):
        qualification_campaign.validate_stage_manifest(
            writable,
            contract=contract,
        )


def test_completion_exactly_round_trips_through_qualification_gate(
    contract,
    tmp_path: Path,
):
    stage = _fake_qualification_stage(contract)
    Path(
        contract["qualification_policy"]["ptm_stage_manifest_path"]
    ).write_text(json.dumps(stage), encoding="utf-8")
    success_id = stage["ptms"][0]["checkpoint_id"]
    for row in stage["ptms"]:
        path = (
            tmp_path
            / "workflows"
            / row["workflow_id"]
            / "workflow_completion.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        workflow = _workflow(
            row["checkpoint_id"],
            success=row["checkpoint_id"] == success_id,
        )
        workflow["source_checkpoint"] = copy.deepcopy(
            row["checkpoint"]
        )
        workflow["workflow_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in workflow.items()
                if key != "workflow_sha256"
            }
        )
        path.write_text(json.dumps(workflow), encoding="utf-8")
    completion = qualification_campaign.build_completion(
        contract=contract,
        stage=stage,
        runtime_root=tmp_path,
        exit_codes={
            row["checkpoint_id"]: 0 for row in stage["ptms"]
        },
    )
    output = tmp_path / "qualification.json"
    output.write_text(json.dumps(completion), encoding="utf-8")
    decision = audit_qualification(output)
    assert completion["all_official_arms_attempted"] is True
    assert len(completion["workflows"]) == 13
    assert all(
        item["terminal"] is True for item in completion["workflows"]
    )
    assert not any(
        item["code"] == "invalid_success_evidence"
        for item in decision.blockers
    )
    if load_ptm_registry().checkpoint(success_id)["status"] == "unverified":
        assert any(
            item["checkpoint_id"] == success_id
            and item["code"] == "registry_not_supported"
            for item in decision.blockers
        )
    assert len(decision.exclusions) == 12


def test_qualification_handoff_is_automatic_but_never_promotes_registry(
    contract,
    tmp_path: Path,
):
    stage = _fake_qualification_stage(contract)
    Path(
        contract["qualification_policy"]["ptm_stage_manifest_path"]
    ).write_text(json.dumps(stage), encoding="utf-8")
    for row in stage["ptms"]:
        path = (
            tmp_path
            / "workflows"
            / row["workflow_id"]
            / "workflow_completion.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        workflow = _workflow(row["checkpoint_id"], success=False)
        workflow["source_checkpoint"] = copy.deepcopy(
            row["checkpoint"]
        )
        workflow["workflow_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in workflow.items()
                if key != "workflow_sha256"
            }
        )
        path.write_text(json.dumps(workflow), encoding="utf-8")
    completion = qualification_campaign.build_completion(
        contract=contract,
        stage=stage,
        runtime_root=tmp_path,
        exit_codes={
            row["checkpoint_id"]: 1 for row in stage["ptms"]
        },
    )
    qualification_path = tmp_path / "qualification.json"
    qualification_path.write_text(
        json.dumps(completion),
        encoding="utf-8",
    )
    handoff = qualification_campaign.build_handoff(
        contract=contract,
        completion=completion,
        qualification_path=qualification_path,
    )
    assert handoff["automatic"] is True
    assert handoff["manual_confirmation_required"] is False
    assert handoff["registry_mutated"] is False
    assert handoff["registry_bypass_allowed"] is False
    assert handoff["fallback_ptm_selected"] is False
    assert handoff["failed_workflow_replaced"] is False
    assert handoff["status"] == "terminal_no_successful_ptm"


def test_independent_status_promotion_preserves_pre_promotion_evidence(
    contract,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    stage = _fake_qualification_stage(contract)
    Path(
        contract["qualification_policy"]["ptm_stage_manifest_path"]
    ).write_text(json.dumps(stage), encoding="utf-8")
    success_id = stage["ptms"][0]["checkpoint_id"]
    for row in stage["ptms"]:
        path = (
            tmp_path
            / "workflows"
            / row["workflow_id"]
            / "workflow_completion.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        workflow = _workflow(
            row["checkpoint_id"],
            success=row["checkpoint_id"] == success_id,
        )
        workflow["source_checkpoint"] = copy.deepcopy(
            row["checkpoint"]
        )
        workflow["workflow_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in workflow.items()
                if key != "workflow_sha256"
            }
        )
        path.write_text(json.dumps(workflow), encoding="utf-8")
    completion = qualification_campaign.build_completion(
        contract=contract,
        stage=stage,
        runtime_root=tmp_path,
        exit_codes={
            row["checkpoint_id"]: 0 for row in stage["ptms"]
        },
    )
    output = tmp_path / "qualification.json"
    output.write_text(json.dumps(completion), encoding="utf-8")

    promoted_document = load_ptm_registry().to_dict()
    for record in promoted_document["models"]["segformer"][
        "checkpoints"
    ]:
        if record["id"] == success_id:
            record["status"] = "supported"
            record["status_reason"] = "independent full-run review passed"
            record["sha256"] = stage["ptms"][0]["checkpoint"]["sha256"]
            record["compatible_tao_versions"] = ["==7.1.0"]
            record["validation"] = {
                "status": "validated",
                "tao_version": "7.1.0-rc-245",
                "evidence": str(output),
            }

    class PromotedRegistry:
        registry_version = "test-promoted"
        document_sha256 = canonical_sha256(promoted_document)

        def to_dict(self):
            return copy.deepcopy(promoted_document)

        def checkpoint(self, checkpoint_id):
            for model in promoted_document["models"].values():
                for record in model["checkpoints"]:
                    if record["id"] == checkpoint_id:
                        return copy.deepcopy(record)
            raise KeyError(checkpoint_id)

    promoted = PromotedRegistry()
    monkeypatch.setattr(
        qualification_gate,
        "load_ptm_registry",
        lambda: promoted,
    )
    monkeypatch.setattr(
        campaign_contract,
        "load_ptm_registry",
        lambda: promoted,
    )
    decision = qualification_gate.audit_qualification(output)
    assert decision.checkpoint_ids == (success_id,)
    assert decision.blockers == ()
    assert len(decision.exclusions) == 12


def test_direct_qualification_submission_is_pinned_one_node_eight_gpu(
    contract,
):
    calls = []

    class FakeSDK:
        def create_job(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(id="job")

    qualification_campaign._submit_job(FakeSDK(), contract, "command")
    assert calls == [
        {
            "image": campaign_contract.FROZEN_SQSH["path"],
            "command": "command",
            "gpu_count": 8,
            "num_nodes": 1,
            "partition": "polar3",
            "account": "edgeai_tao-ptm_image-foundation-model-clip",
        }
    ]
    guard = qualification_campaign._gpu_guard(
        "segformer train -e {config_path}"
    )
    assert "NVIDIA A100-SXM4-80GB" in guard
    assert "wc -l)\" -eq 8" in guard
    assert "export MASTER_ADDR=127.0.0.1" in guard
    assert "15000 + SLURM_JOB_ID % 10000" in guard
    assert "s.bind" in guard
    assert "segformer train -e {config_path}" in guard


def test_qualification_gpu_guard_exports_usable_allocation_port(
    tmp_path: Path,
):
    nvidia_smi = tmp_path / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/bin/sh\n"
        "case \"$*\" in\n"
        "  *query-gpu=name*) value='NVIDIA A100-SXM4-80GB' ;;\n"
        "  *query-gpu=compute_cap*) value='8.0' ;;\n"
        "  *query-gpu=memory.total*) value='81920' ;;\n"
        "  *) exit 2 ;;\n"
        "esac\n"
        "i=0; while [ \"$i\" -lt 8 ]; do printf '%s\\n' \"$value\"; "
        "i=$((i + 1)); done\n",
        encoding="utf-8",
    )
    nvidia_smi.chmod(0o755)
    selected_port = None
    for port in range(
        qualification_campaign.QUALIFICATION_MASTER_PORT_BASE,
        qualification_campaign.QUALIFICATION_MASTER_PORT_BASE
        + qualification_campaign.QUALIFICATION_MASTER_PORT_SPAN,
    ):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
        selected_port = port
        break
    assert selected_port is not None
    job_id = str(
        selected_port
        - qualification_campaign.QUALIFICATION_MASTER_PORT_BASE
    )
    guard = qualification_campaign._gpu_guard(
        "printf 'rendezvous=%s:%s\\n' \"$MASTER_ADDR\" "
        "\"$MASTER_PORT\""
    )

    result = subprocess.run(
        ["bash", "-c", guard],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "SLURM_JOB_ID": job_id,
        },
    )

    assert result.stdout == f"rendezvous=127.0.0.1:{selected_port}\n"


def test_qualification_entrypoint_installs_exact_overlay_for_both_actions(
    contract,
):
    overlay = campaign_contract.FROZEN_QUALIFICATION_RUNTIME_OVERLAY
    for action in ("train", "evaluate"):
        command = qualification_campaign._runtime_overlay_install_command(
            contract,
            action_name=action,
        )
        assert overlay["archive_path"] in command
        assert overlay["archive_sha256"] in command
        assert overlay["installer_path"] in command
        assert overlay["installer_sha256"] in command
        assert overlay["receipt_path"] in command
        assert "--expected-sha256" in command
        assert "test -s" in command
        resolved = (
            f"{command} && segformer {action} -e {{config_path}}"
        ).format(config_path="/tmp/spec.yaml")
        assert f"segformer {action} -e /tmp/spec.yaml" in resolved

    changed = copy.deepcopy(contract)
    changed["qualification_policy"]["runtime_overlay"][
        "archive_sha256"
    ] = "0" * 64
    with pytest.raises(
        qualification_campaign.CampaignExecutionError,
        match="runtime overlay is not authorized",
    ):
        qualification_campaign._runtime_overlay_install_command(
            changed,
            action_name="train",
        )


def test_training_status_evidence_counts_one_evaluation_record_per_epoch(
    monkeypatch: pytest.MonkeyPatch,
):
    records = []
    epochs = campaign_contract.FROZEN_QUALIFICATION_TRAINING_EPOCHS
    for epoch in range(epochs):
        metric = 0.10 + epoch / 100
        kpi = {"val_miou": metric}
        records.extend(
            [
                {
                    "message": "Eval metrics generated.",
                    "kpi": copy.deepcopy(kpi),
                },
                {
                    "message": "Training loop in progress",
                    "kpi": copy.deepcopy(kpi),
                },
            ]
        )
    records.append({"message": "Train finished successfully."})
    monkeypatch.setattr(
        qualification_campaign,
        "_status_records",
        lambda *_args, **_kwargs: (
            records,
            {"path": "/immutable/status.json", "record_count": len(records)},
        ),
    )

    evidence = qualification_campaign._training_status_evidence(
        object(),
        "job-id",
    )

    assert evidence["validation_record_count"] == epochs
    assert [row["val_miou"] for row in evidence["validation_metrics"]] == [
        0.10 + epoch / 100 for epoch in range(epochs)
    ]
    assert evidence["val_miou"] == pytest.approx(0.59)
    assert evidence["terminal_success"] is True


def test_training_status_evidence_rejects_missing_epoch_evaluation_record(
    monkeypatch: pytest.MonkeyPatch,
):
    records = [
        {
            "message": "Eval metrics generated.",
            "kpi": {"val_miou": 0.2},
        }
        for _ in range(
            campaign_contract.FROZEN_QUALIFICATION_TRAINING_EPOCHS - 1
        )
    ]
    records.extend(
        [
            {
                "message": "Training loop in progress",
                "kpi": {"val_miou": 0.2},
            },
            {"message": "Train finished successfully."},
        ]
    )
    monkeypatch.setattr(
        qualification_campaign,
        "_status_records",
        lambda *_args, **_kwargs: (records, {"record_count": len(records)}),
    )

    with pytest.raises(
        qualification_campaign.CampaignExecutionError,
        match="emitted 49 val_miou records; expected 50",
    ):
        qualification_campaign._training_status_evidence(object(), "job-id")


def test_qualification_terminal_checkpoint_uses_epoch_49_not_search_epoch_9(
    monkeypatch: pytest.MonkeyPatch,
):
    commands = []

    class FakeSDK:
        def get_job_results_dir(self, job_id):
            assert job_id == "train-job"
            return "/lustre/results/train-job"

    def fake_remote_output(command):
        commands.append(command)
        return json.dumps(
            {
                "path": (
                    "/lustre/results/train-job/results_dir/train/"
                    "model_epoch_049_step_09150.pth"
                ),
                "filename": "model_epoch_049_step_09150.pth",
                "size_bytes": 123,
                "sha256": "a" * 64,
            }
        )

    monkeypatch.setattr(
        qualification_campaign,
        "remote_output",
        fake_remote_output,
    )

    evidence = qualification_campaign._qualification_terminal_checkpoint(
        FakeSDK(),
        "train-job",
    )

    assert len(commands) == 1
    assert "model_epoch_049_step_*.pth" in commands[0]
    assert "model_epoch_009" not in commands[0]
    assert evidence["terminal_epoch_index"] == 49
    assert evidence["training_epochs"] == 50
    assert evidence["naming_contract"] == (
        "model_epoch_049_step_numeric"
    )


def test_qualification_terminal_checkpoint_rejects_search_epoch_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeSDK:
        def get_job_results_dir(self, _job_id):
            return "/lustre/results/train-job"

    monkeypatch.setattr(
        qualification_campaign,
        "remote_output",
        lambda _command: json.dumps(
            {
                "path": (
                    "/lustre/results/train-job/results_dir/train/"
                    "model_epoch_009_step_01830.pth"
                ),
                "filename": "model_epoch_009_step_01830.pth",
                "size_bytes": 123,
                "sha256": "a" * 64,
            }
        ),
    )

    with pytest.raises(
        qualification_campaign.CampaignExecutionError,
        match="checkpoint identity is invalid",
    ):
        qualification_campaign._qualification_terminal_checkpoint(
            FakeSDK(),
            "train-job",
        )


def test_qualification_gate_rejects_epoch_9_terminal_checkpoint():
    checkpoint_id = campaign_contract.segformer_registry_snapshot()[
        "records"
    ][0]["id"]
    workflow = _workflow(checkpoint_id, success=True)
    terminal = workflow["train"]["terminal_checkpoint"]
    terminal.update(
        {
            "path": (
                "/lustre/results/"
                f"{checkpoint_id}/model_epoch_009_step_01830.pth"
            ),
            "training_epochs": 10,
            "terminal_epoch_index": 9,
            "naming_contract": "model_epoch_009_step_numeric",
        }
    )
    workflow["workflow_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in workflow.items()
            if key != "workflow_sha256"
        }
    )
    record = load_ptm_registry().checkpoint(checkpoint_id)
    record["sha256"] = "a" * 64

    with pytest.raises(
        QualificationGateError,
        match="terminal checkpoint contract changed",
    ):
        qualification_gate._successful_workflow(
            workflow,
            checkpoint_id=checkpoint_id,
            registry_record=record,
        )


def test_qualification_slurm_preflight_is_read_only_and_job_free(
    contract,
    monkeypatch: pytest.MonkeyPatch,
):
    configured = []
    commands = []
    monkeypatch.setattr(
        run_campaign,
        "configure_slurm_runtime",
        lambda value: configured.append(value["contract_sha256"]),
    )

    def fake_remote_output(command, **_kwargs):
        commands.append(command)
        return "READY\n"

    monkeypatch.setattr(
        qualification_campaign,
        "remote_output",
        fake_remote_output,
    )
    evidence = qualification_campaign.verify_slurm_preflight(contract)
    assert configured == [contract["contract_sha256"]]
    assert len(commands) == 1
    assert "sbatch squeue sacct srun" in commands[0]
    assert "MaxTime=04:00:00" in commands[0]
    assert campaign_contract.FROZEN_SQSH["path"] in commands[0]
    overlay = campaign_contract.FROZEN_QUALIFICATION_RUNTIME_OVERLAY
    assert overlay["archive_path"] in commands[0]
    assert overlay["archive_sha256"] in commands[0]
    assert overlay["installer_path"] in commands[0]
    assert overlay["installer_sha256"] in commands[0]
    assert evidence["status"] == "ready"
    assert evidence["partition"] == "polar3"
    assert evidence["scheduler_jobs_submitted"] == 0
    assert evidence["qualification_runtime_overlay"] == overlay
    assert evidence["sdk_source"].startswith(
        contract["runtime"]["sdk_dir"]
    )


def test_qualification_controller_has_no_local_model_or_smoke_path():
    source = (HERE / "qualification_campaign.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "torch" not in imported
    assert "nvidia_tao_pytorch" not in imported
    assert "torch.load" not in source
    assert "load_smoke" not in source
    assert "mini_step" in source
    assert "scheduler_jobs_submitted\": 0" in source
