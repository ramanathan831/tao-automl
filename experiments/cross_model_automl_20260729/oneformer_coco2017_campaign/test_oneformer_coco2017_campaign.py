"""Contract tests for the OneFormer/full-COCO2017 campaign."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tao_automl.ptm_registry import canonical_sha256, load_ptm_registry

from . import campaign_contract, manifest_generator, run_campaign
from .qualification_gate import (
    QualificationGateError,
    audit_qualification,
)


SKILLS = Path(
    "/localhome/local-rarunachalam/.tao/worktrees/tao-skills-release-7.1.0"
)
SKILL_DIR = SKILLS / "skills/models/tao-train-oneformer"
DATASET_ROOT = (
    "/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/"
    "cross_model_automl_20260729/coco2017_instance_panoptic_v1"
)


def dataset_record() -> dict:
    return {
        "id": "coco_2017_full_instance_panoptic",
        "official_source": "https://cocodataset.org/",
        "license": "test",
        "root": DATASET_ROOT,
        "train_image_count": 118287,
        "validation_image_count": 5000,
        "train_panoptic_png_count": 118287,
        "validation_panoptic_png_count": 5000,
        "train_panoptic_segment_count": 1329984,
        "validation_panoptic_segment_count": 56728,
        "panoptic_category_count": 133,
        "instance_category_count": 80,
        "panoptic_label_map_sha256": (
            "4b28b3773f0f8e63d836dc20da77276633da72178453458b79e32be8e892ce56"
        ),
        "instance_label_map_sha256": (
            "67f15c4dd7d52aa73025da8307dec17e907f13db6d5d82332a670f73da68c306"
        ),
        "train_panoptic_json_sha256": (
            "560a90a275c65b089d4944fbd8d44d04c57d2e36bf7f66597f367cc4a42bfbbb"
        ),
        "validation_panoptic_json_sha256": (
            "454873a8a01114246066ac841750eb742df3b5e42ce927ef38b49690084ec75a"
        ),
        "content_sha256": (
            "deced9d6766344fe6fc69cd9de3bcff2cba456a14b3391d07bcedb74c250909e"
        ),
        "manifest_path": "/tmp/coco.FILE_MANIFEST.sha256",
        "manifest_sha256": (
            "10566a60498de9998154f44a34445a488c9f030e09f2a7346d20a4a1c55f804e"
        ),
        "file_manifest_entry_count": 246593,
        "remote_sha256sum_check": "passed_all_246593",
        "stage_manifest_path": "/tmp/dataset_stage_manifest.v1.json",
        "stage_manifest_lustre_path": (
            f"{DATASET_ROOT}/dataset_stage_manifest.v1.json"
        ),
        "stage_manifest_sha256": (
            "437ff12490637950707b9b951d820ea34d38b926080a478a5d182c2d284a0c5d"
        ),
        "remote_file_manifest_path": (
            f"{DATASET_ROOT}/coco2017_instance_panoptic_v1.FILE_MANIFEST.sha256"
        ),
        "remote_read_only": True,
        "remote_writable_entries_after_lock": 0,
    }


def runtime() -> dict:
    return {
        "repository": "/localhome/local-rarunachalam/tao-automl",
        "source_commit": "a" * 40,
        "source_dirty": False,
        "wheel_path": "/tmp/automl.whl",
        "wheel_sha256": "b" * 64,
        "sdk_dir": "/tmp/sdk",
        "sdk_commit": "c" * 40,
        "skills_repository": str(SKILLS),
        "skills_commit": "d" * 40,
        "skill_dir": str(SKILL_DIR),
        "qualification_evidence_path": "/tmp/qualification.json",
        "ptm_stage_manifest_path": "/tmp/ptm_stage.json",
        "partition": "polar3",
        "account": "account",
        "base_results_dir": "/lustre/results",
        "container_mounts": "/lustre",
        "time_hours": 4.0,
        "timeout_hours": 3.8,
        "max_job_retries": 10,
        "hardware_contract": copy.deepcopy(campaign_contract.FROZEN_HARDWARE),
    }


def contract() -> dict:
    return campaign_contract.build_preregistered_contract(
        campaign_id="oneformer-test",
        dataset=dataset_record(),
        skill_dir=str(SKILL_DIR),
        runtime=runtime(),
    )


def test_registry_snapshots_all_four_official_nonordinal_arms():
    snapshot = campaign_contract.oneformer_registry_snapshot()
    assert snapshot["record_count"] == 4
    assert snapshot["supported_ids"] == []
    assert len(snapshot["unverified_ids"]) == 4
    assert all(
        record["source"]["official"] is True
        and record["compatible_tao_versions"] == ["==7.1.0"]
        and record["checkpoint_spec_file"]["source"] == "repository"
        for record in snapshot["records"]
    )


def test_packaged_schema_owns_every_frozen_search_parameter():
    evidence = campaign_contract.validate_packaged_train_schema(SKILL_DIR)
    assert evidence["explicit_search_parameters"] == list(
        campaign_contract.SEARCH_PARAMETERS
    )
    assert evidence["non_train_fields_excluded"] is True


def test_profile_uses_native_panoptic_contract_and_correct_label_map():
    profile = campaign_contract.profile_overrides(DATASET_ROOT)
    dataset = profile["dataset"]
    assert dataset["train"]["annotations"].endswith(
        "/annotations/panoptic_train2017.json"
    )
    assert dataset["val"]["panoptic"].endswith(
        "/annotations/panoptic_val2017"
    )
    assert dataset["label_map"].endswith("/tao/label_map_panoptic.json")
    assert "label_map_instance" not in json.dumps(profile)
    assert dataset["contiguous_id"] is True
    assert dataset["task_prob_train"] == {
        "semantic": 0.0,
        "instance": 0.0,
        "panoptic": 1.0,
    }
    assert profile["model"]["sem_seg_head"]["num_classes"] == 133
    assert profile["train"]["num_gpus"] == 8
    assert profile["train"]["num_nodes"] == 1
    assert profile["train"]["precision"] == "32"


def test_campaign_is_three_independent_objective_aware_jobs():
    value = contract()
    assert value["model"] == value["network_arch"] == "oneformer"
    assert value["task"] == "panoptic_segmentation"
    assert value["execution"]["independent_mode_jobs"] is True
    assert value["execution"]["shared_archive"] is False
    assert [item["mode"] for item in value["modes"]] == list(
        campaign_contract.MODES
    )
    acquisitions = [
        item["objective"]["acquisition"] for item in value["modes"]
    ]
    assert acquisitions == [
        "expected_improvement",
        "constrained_expected_improvement",
        "parego_expected_improvement",
    ]
    namespaces = {
        item["observation_namespace"] for item in value["modes"]
    }
    assert len(namespaces) == 3


def test_latency_retention_does_not_leak_into_multi_objective():
    latency = campaign_contract.mode_settings("campaign", "latency")
    multi = campaign_contract.mode_settings("campaign", "multi_objective")
    assert latency["latency_accuracy_retention"] == {
        "type": "relative",
        "retained_fraction": 0.90,
        "reference": "accuracy_winner",
    }
    assert "latency_accuracy_retention" not in multi
    assert multi["multi_objective_min_accuracy"] is None


def test_metric_semantics_never_alias_miou_to_pq():
    value = contract()
    assert value["primary_accuracy_metric"] == "mIoU"
    assert value["metric_semantics"] == {
        "observed_metric": "semantic_miou_from_panoptic_annotations",
        "product_panoptic_metric": "PQ",
        "pq_emitted_by_current_train_evaluate_path": False,
        "pq_claim_authorized": False,
        "mislabel_miou_as_pq": False,
    }


def test_frozen_pilot_fidelity_and_no_model_smoke_contract():
    value = contract()
    assert value["training_fidelity"]["epochs"] == 1
    assert value["training_fidelity"]["kind"].startswith("one_complete")
    assert value["qualification_policy"]["full_dataset"] is True
    assert value["qualification_policy"]["cpu_model_runs"] == 0
    assert value["qualification_policy"]["smoke_model_runs"] == 0
    assert value["qualification_policy"]["mini_step_runs"] == 0
    assert value["execution"]["cpu_runs"] == 0
    assert value["execution"]["smoke_runs"] == 0


def test_latency_protocol_is_eight_replica_four_thousand_sample_protocol():
    protocol = campaign_contract.LATENCY_PROTOCOL
    assert protocol["warmup_iterations"] == 50
    assert protocol["timed_iterations"] == 100
    assert protocol["repeated_rounds"] == 5
    assert protocol["expected_replicas"] == 8
    assert protocol["raw_samples_per_candidate"] == 4000
    assert protocol["timed_scope"] == "oneformer_model_forward"
    assert protocol["measurement_role"] == "selection_time"


def test_contract_integrity_and_agent_flags_are_fail_closed():
    value = contract()
    assert campaign_contract.validate_contract(value) == value
    assert not any(value["agent_intervention_flags"].values())
    assert not any(value["selection_isolation_flags"].values())
    mutated = copy.deepcopy(value)
    mutated["agent_intervention_flags"]["agent_overrode_winner"] = True
    mutated.pop("contract_sha256")
    mutated["contract_sha256"] = canonical_sha256(mutated)
    with pytest.raises(campaign_contract.CampaignContractError):
        campaign_contract.validate_contract(mutated)


def test_invalid_retention_values_are_rejected():
    for value in (True, 0, -0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(campaign_contract.CampaignContractError):
            campaign_contract._finite_fraction(value, "retention")


def test_manifest_constants_bind_final_coco_stage_and_new_wheel():
    assert (
        manifest_generator.EXPECTED_DATASET_FILE_MANIFEST_SHA256
        == dataset_record()["manifest_sha256"]
    )
    assert (
        manifest_generator.EXPECTED_STAGE_MANIFEST_SHA256
        == dataset_record()["stage_manifest_sha256"]
    )
    assert (
        manifest_generator.WHEEL_BUILD_COMMIT
        == "746d8b7a7134f3786c90b87122ddf8421183e871"
    )
    assert manifest_generator.DEFAULT_WHEEL.is_file()
    assert (
        campaign_contract.sha256_file(manifest_generator.DEFAULT_WHEEL)
        == manifest_generator.EXPECTED_WHEEL_SHA256
    )


def test_committed_stage_and_complete_file_manifest_build_exact_dataset_record():
    record = manifest_generator.dataset_record(
        manifest_generator.DEFAULT_DATASET_MANIFEST,
        Path(__file__).parents[1]
        / "segmentation_datasets/dataset_stage_manifest.v1.json",
    )
    assert record["root"] == DATASET_ROOT
    assert record["train_image_count"] == 118287
    assert record["validation_image_count"] == 5000
    assert record["panoptic_category_count"] == 133
    assert record["file_manifest_entry_count"] == 246593
    assert record["remote_read_only"] is True


def test_all_failed_qualification_evidence_is_preserved_and_blocks(tmp_path):
    snapshot = campaign_contract.oneformer_registry_snapshot()
    workflows = []
    for record in snapshot["records"]:
        item = {
            "checkpoint_id": record["id"],
            "status": "failure",
            "terminal": True,
            "failure_preserved": True,
            "failure_code": "direct_full_run_failed",
            "failure_reason": "frozen test failure",
            "agent_intervention_flags": {
                name: False for name in campaign_contract.AGENT_FLAGS
            },
        }
        item["workflow_sha256"] = canonical_sha256(item)
        workflows.append(item)
    evidence = {
        "schema_version": 1,
        "campaign_id": "qualification-test",
        "model": "oneformer",
        "task": "panoptic_segmentation",
        "metric": "mIoU",
        "metric_semantics": (
            "semantic_miou_from_native_coco_panoptic_annotations"
        ),
        "pq_emitted": False,
        "pq_claim_authorized": False,
        "registry_sha256": snapshot["registry_sha256"],
        "sqsh_sha256": campaign_contract.FROZEN_SQSH["sha256"],
        "cpu_model_runs": 0,
        "smoke_model_runs": 0,
        "mini_step_runs": 0,
        "workflows": workflows,
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    path = tmp_path / "completion.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    decision = audit_qualification(path)
    assert decision.runtime_ready is False
    assert len(decision.exclusions) == 4
    assert any(
        blocker["code"] == "no_runtime_qualified_ptm"
        for blocker in decision.blockers
    )
    with pytest.raises(QualificationGateError):
        decision.assert_runtime_ready()


def test_tampered_failed_workflow_is_blocked_instead_of_excluded(tmp_path):
    snapshot = campaign_contract.oneformer_registry_snapshot()
    workflows = []
    for record in snapshot["records"]:
        item = {
            "checkpoint_id": record["id"],
            "status": "failure",
            "terminal": True,
            "failure_preserved": True,
            "failure_code": "direct_full_run_failed",
            "failure_reason": "frozen test failure",
            "agent_intervention_flags": {
                name: False for name in campaign_contract.AGENT_FLAGS
            },
        }
        item["workflow_sha256"] = canonical_sha256(item)
        workflows.append(item)
    workflows[0]["failure_reason"] = "tampered after sealing"
    evidence = {
        "schema_version": 1,
        "campaign_id": "qualification-tamper-test",
        "model": "oneformer",
        "task": "panoptic_segmentation",
        "metric": "mIoU",
        "metric_semantics": (
            "semantic_miou_from_native_coco_panoptic_annotations"
        ),
        "pq_emitted": False,
        "pq_claim_authorized": False,
        "registry_sha256": snapshot["registry_sha256"],
        "sqsh_sha256": campaign_contract.FROZEN_SQSH["sha256"],
        "cpu_model_runs": 0,
        "smoke_model_runs": 0,
        "mini_step_runs": 0,
        "workflows": workflows,
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    path = tmp_path / "completion.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    decision = audit_qualification(path)
    assert len(decision.exclusions) == 3
    assert any(
        blocker["code"] == "invalid_failure_evidence"
        for blocker in decision.blockers
    )


def test_missing_qualification_never_launches_a_model(tmp_path):
    with pytest.raises(QualificationGateError):
        audit_qualification(tmp_path / "missing.json")


def test_static_sqsh_findings_keep_oneformer_fail_closed():
    blockers = run_campaign.static_sqsh_runtime_blockers(contract())
    assert {
        item["runtime_code"] for item in blockers
    } == {
        "oneformer_full_checkpoint_loader_missing",
        "oneformer_panoptic_pq_not_emitted",
        "oneformer_ddp_status_metric_not_globally_reduced",
    }
    assert all(
        item["code"] == "static_oneformer_runtime_blocker"
        for item in blockers
    )


def test_metric_extractor_accepts_only_exact_oneformer_metric():
    logs = "mIoU: 0.125\nmIoU=0.375\n"
    assert run_campaign._metric_extractor(logs, "mIoU") == pytest.approx(
        0.375
    )
    assert run_campaign._metric_extractor(logs, "PQ") is None


def test_runner_source_preserves_objective_aware_and_automatic_gates():
    source = Path(run_campaign.__file__).read_text(encoding="utf-8")
    assert "ptm_aware_runtime=True" in source
    assert "resolved_ptm_inventory=inventory" in source
    assert "first_candidate_gate" in source
    assert "automatic_trigger" in source
    assert "gpu_count=8" in source
    assert "num_nodes=1" in source
    assert "TAO_AUTOML_ONEFORMER_LATENCY_COMPLETE" in source
    assert "model_epoch_000_step_" in source


def test_custom_ranges_equal_frozen_search_space():
    ranges = campaign_contract.custom_ranges()
    assert set(ranges) == set(campaign_contract.SEARCH_PARAMETERS)
    for name, record in ranges.items():
        assert record["valid_min"] == campaign_contract.SEARCH_SPACE[name][
            "minimum"
        ]
        assert record["valid_max"] == campaign_contract.SEARCH_SPACE[name][
            "maximum"
        ]


def test_registry_remains_unverified_until_real_direct_runs():
    model = load_ptm_registry().to_dict()["models"]["oneformer"]
    assert model["default_ptm"] is None
    assert {record["status"] for record in model["checkpoints"]} == {
        "unverified"
    }
