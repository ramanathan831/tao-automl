from __future__ import annotations

import ast
import copy
import json
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tao_automl.ptm_registry import canonical_sha256, load_ptm_registry

from . import (
    campaign_contract,
    manifest_generator,
    qualification_campaign,
    run_campaign,
    runtime_overlay,
)
from .qualification_gate import (
    QualificationGateError,
    QualificationLoadEvidence,
    audit_qualification,
)


HERE = Path(__file__).resolve().parent
SKILL_DIR = (
    manifest_generator.DEFAULT_SKILLS
    / "skills/models/tao-train-mask2former"
)
DATASET_STAGE_MANIFEST = manifest_generator.DEFAULT_STAGE_MANIFEST
if not DATASET_STAGE_MANIFEST.is_file():
    # Dataset staging is an explicit integration dependency. This keeps the
    # isolated campaign worktree testable before the staging commit lands.
    DATASET_STAGE_MANIFEST = Path(
        "/localhome/local-rarunachalam/.tao/worktrees/"
        "tao-automl-segmentation-datasets/experiments/"
        "cross_model_automl_20260729/segmentation_datasets/"
        "dataset_stage_manifest.v1.json"
    )


@lru_cache(maxsize=1)
def _dataset_cached() -> dict:
    return manifest_generator.dataset_record(
        manifest_generator.DEFAULT_DATASET_MANIFEST,
        DATASET_STAGE_MANIFEST,
    )


def _dataset() -> dict:
    return copy.deepcopy(_dataset_cached())


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
        "ptm_stage_manifest_sha256": "e" * 64,
        "ptm_stage_content_sha256": "f" * 64,
        "tao_pytorch_overlay": runtime_overlay.contract_record(),
        "partition": "polar3",
        "account": "edgeai_tao-ptm_image-foundation-model-clip",
        "base_results_dir": (
            "/lustre/fsw/portfolios/edgeai/users/rarunachalam"
        ),
        "container_mounts": "/lustre",
        "time_hours": campaign_contract.FROZEN_SLURM_TIME_HOURS,
        "timeout_hours": campaign_contract.FROZEN_SLURM_TIMEOUT_HOURS,
        "walltime_policy": copy.deepcopy(
            campaign_contract.FROZEN_WALLTIME_POLICY
        ),
        "max_job_retries": campaign_contract.FROZEN_SLURM_RETRY_CAP,
        "hardware_contract": copy.deepcopy(
            campaign_contract.FROZEN_HARDWARE
        ),
    }


@pytest.fixture
def contract(tmp_path: Path) -> dict:
    value = campaign_contract.build_preregistered_contract(
        campaign_id="mask2former-test",
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
        "mask2former_latency_worker_sha256": (
            campaign_contract.sha256_file(
                HERE / "mask2former_latency_worker.py"
            )
        ),
        "runtime_overlay_sha256": campaign_contract.sha256_file(
            HERE / "runtime_overlay.py"
        ),
    }
    value["contract_sha256"] = canonical_sha256(value)
    return campaign_contract.validate_contract(value)


def _workflow(
    checkpoint_id: str,
    *,
    success: bool,
    metric: float = 0.20,
    include_mask_ap: bool = True,
) -> dict:
    record = load_ptm_registry().checkpoint(checkpoint_id)
    if not success:
        value = {
            "checkpoint_id": checkpoint_id,
            "status": "failure",
            "terminal": True,
            "failure_preserved": True,
            "failure_code": "direct_full_run_failed",
            "failure_reason": "frozen test failure",
        }
        value["workflow_sha256"] = canonical_sha256(value)
        return value
    train_metric = (
        {"segm_val_mAP": metric}
        if include_mask_ap
        else {"mIoU": metric}
    )
    eval_metric = (
        {
            "segm_test_mAP": metric,
            "segm_test_mAP50": metric,
            "objective_binding": {
                "reported_metric": "segm_test_mAP",
                "canonical_metric": "segm_val_mAP",
                "value": metric,
            },
        }
        if include_mask_ap
        else {"mIoU": metric}
    )
    value = {
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
                campaign_contract.FROZEN_TRAINING_EPOCHS
            ),
            "validation_interval": 1,
            "validation_record_count": (
                campaign_contract.FROZEN_TRAINING_EPOCHS
            ),
            "nodes": 1,
            "gpus": 8,
            **train_metric,
            "terminal_checkpoint": {
                "path": f"/lustre/results/{checkpoint_id}.pth",
                "size_bytes": 123,
                "sha256": "b" * 64,
            },
        },
        "evaluation": {
            "status": "Complete",
            "full_validation_split": True,
            "nodes": 1,
            "gpus": 8,
            **eval_metric,
        },
        "agent_intervention_flags": {
            name: False for name in campaign_contract.AGENT_FLAGS
        },
    }
    value["workflow_sha256"] = canonical_sha256(value)
    return value


def _qualification_document(
    success_id: str | None = None,
    *,
    metric: float = 0.20,
    include_mask_ap: bool = True,
) -> dict:
    snapshot = campaign_contract.mask2former_registry_snapshot()
    workflows = [
        _workflow(
            record["id"],
            success=record["id"] == success_id,
            metric=metric,
            include_mask_ap=include_mask_ap,
        )
        for record in snapshot["records"]
    ]
    value = {
        "schema_version": 1,
        "campaign_id": "mask2former-direct-full-qualification-test",
        "model": "mask2former",
        "task": "instance_segmentation",
        "primary_metric": "segm_val_mAP",
        "standalone_reported_metric": "segm_test_mAP",
        "standalone_objective_binding": {
            "reported_metric": "segm_test_mAP",
            "canonical_metric": "segm_val_mAP",
        },
        "semantic_miou_accepted_as_mask_ap": False,
        "qualification_contract_sha256": "c" * 64,
        "qualification_campaign_sha256": (
            campaign_contract.sha256_file(
                HERE / "qualification_campaign.py"
            )
        ),
        "ptm_stage_manifest_path": "/tmp/frozen-ptm-stage.json",
        "ptm_stage_manifest_sha256": "d" * 64,
        "registry_sha256": snapshot["registry_sha256"],
        "sqsh_sha256": campaign_contract.FROZEN_SQSH["sha256"],
        "tao_pytorch_overlay": runtime_overlay.contract_record(),
        "cpu_model_runs": 0,
        "smoke_model_runs": 0,
        "mini_step_runs": 0,
        "workflows": workflows,
    }
    value["evidence_sha256"] = canonical_sha256(value)
    return value


def test_exact_tao_identifier_actions_and_task_correct_metric():
    info = yaml.safe_load(
        (SKILL_DIR / "references/skill_info.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert info["network_arch"] == "mask2former"
    assert info["actions"]["train"]["command"] == (
        "mask2former train -e {config_path}"
    )
    assert info["actions"]["evaluate"]["command"] == (
        "mask2former evaluate -e {config_path}"
    )
    settings = campaign_contract.mode_settings("x", "accuracy")
    assert settings["accuracy_metric"] == "segm_val_mAP"
    assert run_campaign._metric_extractor("mIoU: 0.71", "segm_val_mAP") is None
    assert (
        run_campaign._metric_extractor(
            "segm_val_mAP: 0.321", "segm_val_mAP"
        )
        == pytest.approx(0.321)
    )


def test_standalone_metric_is_bound_without_mislabeling(
    monkeypatch: pytest.MonkeyPatch,
):
    status = "\n".join([
        json.dumps({
            "kpi": {
                "segm_test_mAP": "0.42",
                "segm_test_mAP50": "0.61",
                "segm_val_mAP": "0.99",
            }
        }),
    ])
    monkeypatch.setattr(run_campaign, "remote_output", lambda _: status)
    sdk = SimpleNamespace(
        get_job_results_dir=lambda _: "lustre:///lustre/results/job"
    )
    assert run_campaign._status_metric(
        sdk,
        "job",
        action="evaluate",
        names=("segm_test_mAP",),
    ) == pytest.approx(0.42)
    assert qualification_campaign._status_values(
        sdk,
        "job",
        action="evaluate",
        names=("segm_test_mAP50",),
    ) == [pytest.approx(0.61)]
    assert qualification_campaign._status_values(
        sdk,
        "job",
        action="evaluate",
        names=("segm_val_mAP",),
    ) == [pytest.approx(0.99)]


def test_search_parameters_are_packaged_train_parameters():
    evidence = campaign_contract.validate_packaged_train_schema(SKILL_DIR)
    assert tuple(evidence["explicit_search_parameters"]) == (
        "model.mask_former.num_object_queries",
        "model.mask_former.dec_layers",
        "dataset.augmentation.test_min_size",
        "train.optim.lr",
        "train.optim.weight_decay",
    )
    assert (
        campaign_contract.SEARCH_SPACE[
            "model.mask_former.num_object_queries"
        ]
        == {"type": "integer", "minimum": 50, "maximum": 200}
    )
    assert campaign_contract.SEARCH_SPACE[
        "model.mask_former.dec_layers"
    ] == {"type": "integer", "minimum": 4, "maximum": 10}
    assert campaign_contract.SEARCH_SPACE[
        "dataset.augmentation.test_min_size"
    ] == {"type": "integer", "minimum": 480, "maximum": 800}


def test_complete_coco2017_instance_dataset_is_frozen():
    dataset = _dataset()
    assert dataset["prepared_root"].endswith(
        "/coco2017_instance_panoptic_v1"
    )
    assert dataset["train_image_count"] == 118287
    assert dataset["validation_image_count"] == 5000
    assert dataset["train_instance_annotations"] == 860001
    assert dataset["validation_instance_annotations"] == 36781
    assert dataset["num_classes"] == 80
    assert dataset["file_manifest_entry_count"] == 246593
    assert dataset["manifest_sha256"] == (
        "10566a60498de9998154f44a34445a488c9f030e09f2a7346d20a4a1c55f804e"
    )
    assert dataset["stage_manifest_sha256"] == (
        "437ff12490637950707b9b951d820ea34d38b926080a478a5d182c2d284a0c5d"
    )
    assert dataset["remote_read_only"] is True
    assert dataset["remote_writable_entries_after_lock"] == 0


def test_profile_is_instance_coco_eight_gpu_not_smoke():
    root = _dataset()["prepared_root"]
    profile = campaign_contract.profile_overrides(root)
    assert profile["model"]["mode"] == "instance"
    assert profile["model"]["sem_seg_head"]["num_classes"] == 80
    assert profile["dataset"]["contiguous_id"] is True
    assert profile["dataset"]["label_map"] == (
        f"{root}/tao/label_map_instance.json"
    )
    for split, name in (
        ("train", "instances_train2017.json"),
        ("val", "instances_val2017.json"),
        ("test", "instances_val2017.json"),
    ):
        assert profile["dataset"][split]["type"] == "coco"
        assert profile["dataset"][split]["instance_json"].endswith(name)
        assert profile["dataset"][split]["panoptic_json"] == ""
        assert profile["dataset"][split]["batch_size"] == 1
    train = profile["train"]
    assert train["num_gpus"] == 8
    assert train["gpu_ids"] == list(range(8))
    assert train["num_nodes"] == 1
    assert train["num_epochs"] == 3
    assert train["validation_interval"] == 1
    assert train["distributed_strategy"] == "ddp"
    assert train["precision"] == "fp32"


def test_mask_ap_sanity_is_separate_from_product_selection(contract):
    assert contract["task"] == "instance_segmentation"
    metric = contract["metric_contract"]
    assert metric["required"] == "segm_val_mAP"
    assert metric["validation_reported_metric"] == "segm_val_mAP"
    assert metric["standalone_reported_metric"] == "segm_test_mAP"
    assert metric["standalone_reported_metric50"] == "segm_test_mAP50"
    assert metric["standalone_canonical_objective"] == "segm_val_mAP"
    assert metric["semantic_miou_is_not_an_alias"] is True
    assert metric["known_repository_state"] == (
        "runtime_fix_available_pending_gpu_qualification"
    )
    gate = contract["validation_sanity_gate"]
    assert gate["metric"] == "segm_val_mAP"
    assert gate["minimum"] == 0.05
    assert gate["role"] == (
        "experiment_correctness_gate_not_product_selection"
    )


def test_official_ptm_registry_is_exact_and_hierarchical():
    snapshot = campaign_contract.mask2former_registry_snapshot()
    assert snapshot["record_count"] == 1
    assert [item["id"] for item in snapshot["records"]] == [
        "mask2former.coco.swin_tiny.trainable.v1.0"
    ]
    record = snapshot["records"][0]
    assert record["source"]["official"] is True
    assert record["source"]["version"] == (
        "mask2former_swint_trainable_v1.0"
    )
    assert record["checkpoint_target"] == "train.pretrained_model_path"
    assert snapshot["supported_ids"] == []
    assert snapshot["unverified_ids"] == [record["id"]]


def test_mode_acquisitions_and_constraints_are_independent(contract):
    modes = {
        item["mode"]: item for item in contract["modes"]
    }
    assert modes["accuracy"]["objective"]["acquisition"] == (
        "expected_improvement"
    )
    assert "latency_accuracy_retention" not in modes["accuracy"]["settings"]
    assert modes["latency"]["objective"]["acquisition"] == (
        "constrained_expected_improvement"
    )
    assert modes["latency"]["settings"][
        "latency_accuracy_retention"
    ] == {
        "type": "relative",
        "retained_fraction": 0.90,
        "reference": "accuracy_winner",
    }
    assert modes["multi_objective"]["objective"]["acquisition"] == (
        "parego_expected_improvement"
    )
    assert "latency_accuracy_retention" not in (
        modes["multi_objective"]["settings"]
    )
    assert (
        modes["multi_objective"]["settings"][
            "multi_objective_min_accuracy"
        ]
        is None
    )
    assert all(
        item["observation_sharing"] is False
        and item["initial_observation_ids"] == []
        for item in modes.values()
    )


def test_contract_is_pinned_sqsh_eight_gpu_and_zero_local_runs(contract):
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
    assert all(
        value is False
        for value in contract["agent_intervention_flags"].values()
    )
    assert all(
        value is False
        for value in contract["selection_isolation_flags"].values()
    )


def test_v2_walltime_is_frozen_without_changing_scientific_budget(contract):
    runtime = contract["runtime"]
    policy = runtime["walltime_policy"]
    assert runtime["time_hours"] == 8.0
    assert runtime["timeout_hours"] == 7.8
    assert policy == campaign_contract.FROZEN_WALLTIME_POLICY
    assert policy["contract_revision"] == "qualification_runtime_v2"
    assert policy["observed_full_epoch_minutes_approx"] == 90.0
    assert policy["observed_minimum_training_hours_approx"] == 4.5
    assert policy["training_epochs"] == 3
    assert policy["training_budget_changed"] is False
    assert policy["search_space_changed"] is False
    assert policy["candidate_budget_changed"] is False
    assert policy["retry_policy_changed"] is False
    assert policy["v1_runtime_evidence_preserved"] is True
    assert contract["search"]["training_epochs"] == 3
    assert contract["search"]["candidate_budget_per_mode"] == 20
    assert contract["search"]["space"] == campaign_contract.SEARCH_SPACE


def test_v2_runtime_paths_do_not_overwrite_v1_evidence():
    assert str(qualification_campaign.DEFAULT_RUNTIME_ROOT).endswith(
        "mask2former_coco2017_ptm_qualification_v2"
    )
    assert str(qualification_campaign.DEFAULT_STAGE_MANIFEST).endswith(
        "mask2former_coco2017_ptm_qualification_v1/"
        "ptm_stage_manifest.json"
    )
    assert str(run_campaign.DEFAULT_RUNTIME_ROOT).endswith(
        "mask2former_coco2017_three_mode_v2"
    )
    assert run_campaign.DEFAULT_CONTRACT.name == "campaign.v2.json"


def test_latency_protocol_is_4000_real_coco_validation_samples(contract):
    protocol = contract["latency_protocol"]
    assert protocol["warmup_iterations"] == 50
    assert protocol["repeated_rounds"] == 5
    assert protocol["timed_iterations"] == 100
    assert protocol["expected_replicas"] == 8
    assert protocol["raw_samples_per_candidate"] == 4000
    assert "instance_postprocessing" in protocol["excluded_scope"]
    source = (HERE / "mask2former_latency_worker.py").read_text(
        encoding="utf-8"
    )
    assert "COCODataset" in source
    assert "Mask2formerPlModule" in source
    assert "model(preloaded[" in source
    assert "torch.randn" not in source
    assert "torch.rand(" not in source


def test_model_imports_live_only_below_latency_worker_main():
    tree = ast.parse(
        (HERE / "mask2former_latency_worker.py").read_text(
            encoding="utf-8"
        )
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


def test_semantic_miou_cannot_qualify_instance_segmentation(tmp_path: Path):
    checkpoint_id = campaign_contract.mask2former_registry_snapshot()[
        "records"
    ][0]["id"]
    document = _qualification_document(
        checkpoint_id,
        include_mask_ap=False,
    )
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    decision = audit_qualification(path)
    assert checkpoint_id not in decision.checkpoint_ids
    assert any(
        item["checkpoint_id"] == checkpoint_id
        and item["code"] == "invalid_success_evidence"
        and "segm_val_mAP" in item["reason"]
        for item in decision.blockers
    )


def test_mislabeled_standalone_val_metric_cannot_qualify(tmp_path: Path):
    checkpoint_id = campaign_contract.mask2former_registry_snapshot()[
        "records"
    ][0]["id"]
    document = _qualification_document(checkpoint_id)
    evaluation = document["workflows"][0]["evaluation"]
    evaluation.pop("segm_test_mAP")
    evaluation["segm_val_mAP"] = 0.20
    document["workflows"][0]["workflow_sha256"] = canonical_sha256({
        key: value
        for key, value in document["workflows"][0].items()
        if key != "workflow_sha256"
    })
    document["evidence_sha256"] = canonical_sha256({
        key: value
        for key, value in document.items()
        if key != "evidence_sha256"
    })
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    decision = audit_qualification(path)
    assert any(
        item["checkpoint_id"] == checkpoint_id
        and item["code"] == "invalid_success_evidence"
        and "segm_test_mAP" in item["reason"]
        for item in decision.blockers
    )


def test_low_finite_mask_ap_does_not_pass_qualification(tmp_path: Path):
    checkpoint_id = campaign_contract.mask2former_registry_snapshot()[
        "records"
    ][0]["id"]
    document = _qualification_document(checkpoint_id, metric=0.049)
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    decision = audit_qualification(path)
    assert any(
        item["checkpoint_id"] == checkpoint_id
        and item["code"] == "invalid_success_evidence"
        and "0.05 COCO mask AP" in item["reason"]
        for item in decision.blockers
    )


def test_unverified_full_run_success_cannot_bypass_registry(tmp_path: Path):
    checkpoint_id = campaign_contract.mask2former_registry_snapshot()[
        "records"
    ][0]["id"]
    path = tmp_path / "qualification.json"
    path.write_text(
        json.dumps(_qualification_document(checkpoint_id)),
        encoding="utf-8",
    )
    decision = audit_qualification(path)
    assert checkpoint_id not in decision.checkpoint_ids
    assert any(
        item["checkpoint_id"] == checkpoint_id
        and item["code"] == "registry_not_supported"
        for item in decision.blockers
    )
    with pytest.raises(QualificationGateError):
        QualificationLoadEvidence(decision)


def test_qualification_can_precede_registry_promotion_without_bypass(
    tmp_path: Path,
):
    checkpoint_id = campaign_contract.mask2former_registry_snapshot()[
        "records"
    ][0]["id"]
    document = _qualification_document(checkpoint_id)
    # Qualification evidence is naturally produced against the pre-promotion
    # registry. Its immutable checkpoint/workflow identity remains usable
    # after a separate reviewed promotion, but current status is still
    # enforced and therefore blocks in this unverified repository.
    document["registry_sha256"] = "1" * 64
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
        and item["code"] == "registry_not_supported"
        for item in decision.blockers
    )


def test_direct_full_qualification_plan_is_plan_only(contract):
    plan = qualification_campaign.qualification_plan(contract)
    assert plan["campaign_id"] == (
        "mask2former-coco2017-direct-full-qualification-v2-20260801"
    )
    assert plan["contract_revision"] == "qualification_runtime_v2"
    assert plan["official_checkpoint_ids"] == [
        "mask2former.coco.swin_tiny.trainable.v1.0"
    ]
    assert plan["workflow_count"] == 1
    assert plan["primary_metric"] == "segm_val_mAP"
    assert plan["standalone_reported_metric"] == "segm_test_mAP"
    assert plan["standalone_objective_binding"] == {
        "reported_metric": "segm_test_mAP",
        "canonical_metric": "segm_val_mAP",
    }
    assert plan["training_epochs"] == 3
    assert plan["nodes_per_job"] == 1
    assert plan["gpus_per_job"] == 8
    assert plan["walltime_policy"] == (
        campaign_contract.FROZEN_WALLTIME_POLICY
    )
    assert plan["scheduler_client_constructed"] is False
    assert plan["jobs_submitted"] == 0
    assert plan["cpu_model_runs"] == 0
    assert plan["smoke_model_runs"] == 0
    assert plan["mini_step_runs"] == 0
    assert plan["replacement_workflows_allowed"] is False


def test_v2_completion_records_exact_walltime_contract(contract):
    completion = qualification_campaign.build_completion(contract, [])
    assert completion["campaign_id"] == (
        "mask2former-coco2017-direct-full-qualification-v2-20260801"
    )
    assert completion["contract_revision"] == "qualification_runtime_v2"
    assert completion["qualification_contract_sha256"] == (
        contract["contract_sha256"]
    )
    assert completion["walltime_policy"] == (
        campaign_contract.FROZEN_WALLTIME_POLICY
    )
    payload = copy.deepcopy(completion)
    observed = payload.pop("evidence_sha256")
    assert observed == canonical_sha256(payload)


def test_v1_evidence_cannot_satisfy_v2_contract(
    contract,
    tmp_path: Path,
):
    record = contract["ptm_inventory"]["records"][0]
    stage = {
        "schema_version": 1,
        "model": "mask2former",
        "registry_sha256": contract["ptm_inventory"]["registry_sha256"],
        "checkpoints": [
            {
                "id": record["id"],
                "path": "/lustre/ptms/mask2former.pth",
                "size_bytes": record["expected_size_bytes"],
                "sha256": "d" * 64,
            }
        ],
    }
    stage_path = tmp_path / "stage.json"
    stage_path.write_text(json.dumps(stage), encoding="utf-8")

    expected = copy.deepcopy(contract)
    expected["runtime"]["ptm_stage_manifest_path"] = str(stage_path)
    expected["runtime"]["ptm_stage_manifest_sha256"] = (
        campaign_contract.sha256_file(stage_path)
    )
    expected["qualification_policy"]["ptm_stage_manifest_path"] = str(
        stage_path
    )
    expected.pop("contract_sha256")
    expected["contract_sha256"] = canonical_sha256(expected)

    document = _qualification_document()
    document["contract_revision"] = "qualification_runtime_v2"
    document["qualification_contract_sha256"] = expected[
        "contract_sha256"
    ]
    document["qualification_campaign_sha256"] = expected[
        "launcher_integrity"
    ]["qualification_campaign_sha256"]
    document["ptm_stage_manifest_path"] = str(stage_path)
    document["ptm_stage_manifest_sha256"] = campaign_contract.sha256_file(
        stage_path
    )
    document["walltime_policy"] = copy.deepcopy(
        campaign_contract.FROZEN_WALLTIME_POLICY
    )
    document["evidence_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in document.items()
            if key != "evidence_sha256"
        }
    )
    evidence_path = tmp_path / "completion.json"
    evidence_path.write_text(json.dumps(document), encoding="utf-8")
    audit_qualification(evidence_path, expected_contract=expected)

    document["contract_revision"] = "qualification_runtime_v1"
    document["evidence_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in document.items()
            if key != "evidence_sha256"
        }
    )
    evidence_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(
        QualificationGateError,
        match="wall-time policy",
    ):
        audit_qualification(evidence_path, expected_contract=expected)


def test_direct_qualification_spec_precedence_preserves_coco_profile(
    contract,
):
    checkpoint_id = contract["ptm_inventory"]["records"][0]["id"]
    train, evaluate = qualification_campaign._qualification_specs(
        contract,
        checkpoint_id,
        "/lustre/ptms/mask2former.pth",
    )
    for specification in (train, evaluate):
        assert specification["model"]["mode"] == "instance"
        # The PTM YAML says one class; the explicit AutoML COCO profile has
        # higher precedence and correctly supplies the official 80 classes.
        assert specification["model"]["sem_seg_head"]["num_classes"] == 80
        assert specification["dataset"]["contiguous_id"] is True
        assert specification["dataset"]["label_map"].endswith(
            "/tao/label_map_instance.json"
        )
    assert train["train"]["pretrained_model_path"] == (
        "/lustre/ptms/mask2former.pth"
    )


def test_ptm_stage_is_exact_content_addressed_inventory(
    contract,
    tmp_path: Path,
):
    record = contract["ptm_inventory"]["records"][0]
    stage = {
        "schema_version": 1,
        "model": "mask2former",
        "registry_sha256": contract["ptm_inventory"]["registry_sha256"],
        "stage_complete": True,
        "remote_read_only": True,
        "cpu_model_runs": 0,
        "gpu_model_runs": 0,
        "smoke_model_runs": 0,
        "mini_step_runs": 0,
        "scheduler_jobs_submitted": 0,
        "checkpoints": [
            {
                "id": record["id"],
                "path": "/lustre/ptms/mask2former.pth",
                "size_bytes": record["expected_size_bytes"],
                "sha256": "d" * 64,
                "immutable_source_identity": record["source"][
                    "immutable_identity"
                ],
                "remote_read_only": True,
            }
        ],
    }
    stage["manifest_sha256"] = canonical_sha256(stage)
    path = tmp_path / "ptm_stage.json"
    path.write_text(json.dumps(stage), encoding="utf-8")
    frozen_contract = copy.deepcopy(contract)
    frozen_contract["runtime"]["ptm_stage_manifest_sha256"] = (
        campaign_contract.sha256_file(path)
    )
    loaded = qualification_campaign.load_ptm_stage(
        path, frozen_contract, verify_remote=False
    )
    assert loaded[record["id"]]["sha256"] == "d" * 64
    sealed = manifest_generator.ptm_stage_record(path)
    assert sealed["sha256"] == campaign_contract.sha256_file(path)
    assert sealed["manifest_sha256"] == stage["manifest_sha256"]
    assert sealed["checkpoint_ids"] == [record["id"]]

    changed = copy.deepcopy(stage)
    changed["checkpoints"][0]["size_bytes"] -= 1
    changed["manifest_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in changed.items()
            if key != "manifest_sha256"
        }
    )
    path.write_text(json.dumps(changed), encoding="utf-8")
    frozen_contract["runtime"]["ptm_stage_manifest_sha256"] = (
        campaign_contract.sha256_file(path)
    )
    with pytest.raises(run_campaign.CampaignExecutionError):
        qualification_campaign.load_ptm_stage(
            path, frozen_contract, verify_remote=False
        )


def test_qualification_missing_mask_ap_is_terminal_and_not_replaced():
    failure = qualification_campaign._failure_workflow(
        "mask2former.coco.swin_tiny.trainable.v1.0",
        "task-correct segm_val_mAP missing; observed mIoU only",
        code="task_correct_metric_missing",
        diagnostics={"mIoU": [0.7]},
    )
    assert failure["status"] == "failure"
    assert failure["terminal"] is True
    assert failure["failure_preserved"] is True
    assert failure["replacement_submitted"] is False
    payload = copy.deepcopy(failure)
    supplied = payload.pop("workflow_sha256")
    assert supplied == canonical_sha256(payload)


def test_terminal_ptm_failure_is_preserved_as_exclusion(tmp_path: Path):
    path = tmp_path / "qualification.json"
    path.write_text(
        json.dumps(_qualification_document()),
        encoding="utf-8",
    )
    decision = audit_qualification(path)
    assert len(decision.exclusions) == 1
    assert decision.exclusions[0]["code"] == "direct_full_run_failed"
    assert any(
        item["code"] == "no_runtime_qualified_ptm"
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
    ] == 19
    assert plan["resources_per_child"]["gpus"] == 8


def test_validation_descriptor_is_bound_to_coco_records(
    contract,
    monkeypatch: pytest.MonkeyPatch,
):
    records = [
        {
            "image_id": index + 1,
            "name": f"{index + 1:012d}.jpg",
            "size_bytes": 100 + index,
            "sha256": f"{index + 1:064x}",
        }
        for index in range(16)
    ]
    observed_command = ""

    def fake_remote(command: str) -> str:
        nonlocal observed_command
        observed_command = command
        return json.dumps(records)

    monkeypatch.setattr(run_campaign, "remote_output", fake_remote)
    descriptor = run_campaign.validation_input_descriptor(contract)
    assert "instances_val2017.json" in observed_command
    assert descriptor["validation_files"] == records
    assert descriptor["preloaded_batches"] == 16
    assert "image_size" not in descriptor


def test_evaluation_spec_is_full_coco_instance_and_eight_gpu(contract):
    specification = run_campaign.evaluation_spec(
        contract,
        {"model": {"mask_former": {"dec_layers": 6}}},
        "/lustre/checkpoints/final.pth",
    )
    assert specification["model"]["mode"] == "instance"
    assert specification["model"]["sem_seg_head"]["num_classes"] == 80
    assert specification["dataset"]["val"]["type"] == "coco"
    assert specification["dataset"]["val"]["instance_json"].endswith(
        "instances_val2017.json"
    )
    assert specification["evaluate"]["num_gpus"] == 8
    assert specification["evaluate"]["gpu_ids"] == list(range(8))
    assert specification["evaluate"]["checkpoint"] == (
        "/lustre/checkpoints/final.pth"
    )


def test_archive_order_cannot_enter_independent_campaign_jobs(contract):
    modes = contract["modes"]
    assert [item["mode"] for item in modes] == [
        "accuracy",
        "latency",
        "multi_objective",
    ]
    assert len(
        {item["observation_namespace"] for item in modes}
    ) == 3
    assert all(item["initial_observation_ids"] == [] for item in modes)
    assert contract["execution"]["shared_archive"] is False


def test_contract_integrity_rejects_policy_mutation(contract):
    changed = copy.deepcopy(contract)
    changed["execution"]["gpus_per_child"] = 1
    with pytest.raises(campaign_contract.CampaignContractError):
        campaign_contract.validate_contract(changed)

    changed = copy.deepcopy(contract)
    changed["runtime"]["time_hours"] = 4.0
    changed["runtime"]["timeout_hours"] = 3.8
    changed.pop("contract_sha256")
    changed["contract_sha256"] = canonical_sha256(changed)
    with pytest.raises(
        campaign_contract.CampaignContractError,
        match="frozen v2 wall-time policy",
    ):
        campaign_contract.validate_contract(changed)

    changed = copy.deepcopy(contract)
    changed["metric_contract"]["semantic_miou_is_not_an_alias"] = False
    with pytest.raises(campaign_contract.CampaignContractError):
        campaign_contract.validate_contract(changed)

    changed = copy.deepcopy(contract)
    changed["qualification_policy"]["standalone_objective_binding"][
        "reported_metric"
    ] = "segm_val_mAP"
    changed.pop("contract_sha256")
    changed["contract_sha256"] = canonical_sha256(changed)
    with pytest.raises(campaign_contract.CampaignContractError):
        campaign_contract.validate_contract(changed)

    changed = copy.deepcopy(contract)
    changed["modes"][2]["objective"]["acquisition"] = (
        "expected_improvement"
    )
    changed.pop("contract_sha256")
    changed["contract_sha256"] = canonical_sha256(changed)
    with pytest.raises(campaign_contract.CampaignContractError):
        campaign_contract.validate_contract(changed)


def test_runtime_overlay_contract_is_sealed_and_pythonpath_only(contract):
    overlay = contract["runtime"]["tao_pytorch_overlay"]
    assert overlay == runtime_overlay.contract_record()
    assert overlay["source_commit"] == (
        "c2e86fe1646ebe89fc280083797dcc544ce88322"
    )
    assert overlay["archive"]["sha256"] == (
        "c395474592d557e0179066c1f99d5cb8f352e10e501621d57043782440dea8c2"
    )
    assert overlay["injection"]["mechanism"] == "PYTHONPATH"
    assert overlay["injection"]["installed_package_mutated"] is False

    changed = copy.deepcopy(contract)
    changed["runtime"]["tao_pytorch_overlay"]["archive"]["sha256"] = "0" * 64
    changed.pop("contract_sha256")
    changed["contract_sha256"] = canonical_sha256(changed)
    with pytest.raises(runtime_overlay.RuntimeOverlayError):
        campaign_contract.validate_contract(changed)


def test_runtime_overlay_wrap_is_fail_closed_and_precedes_action(contract):
    overlay = contract["runtime"]["tao_pytorch_overlay"]
    command = runtime_overlay.wrap_command(
        "mask2former train -e {config_path}",
        overlay,
    )
    assert overlay["installer"]["path"] in command
    assert overlay["installer"]["sha256"] in command
    assert overlay["archive"]["path"] in command
    assert overlay["archive"]["sha256"] in command
    assert overlay["source_commit"] in command
    assert (
        f"export PYTHONPATH={overlay['injection']['pythonpath_root']}:"
        "\"$(printenv PYTHONPATH || true)\""
    ) in command
    assert command.index(overlay["installer"]["path"]) < command.index(
        "mask2former train"
    )
    changed = copy.deepcopy(overlay)
    changed["source_commit"] = "0" * 40
    with pytest.raises(runtime_overlay.RuntimeOverlayError):
        runtime_overlay.wrap_command("mask2former train", changed)


def test_runtime_overlay_remote_identity_and_readonly_are_launch_gates(
    contract,
    monkeypatch: pytest.MonkeyPatch,
):
    overlay = contract["runtime"]["tao_pytorch_overlay"]

    def identity(path: str) -> dict:
        if path == overlay["archive"]["path"]:
            return {
                "path": path,
                "size_bytes": overlay["archive"]["size_bytes"],
                "sha256": overlay["archive"]["sha256"],
            }
        return {
            "path": path,
            "size_bytes": overlay["installer"]["size_bytes"],
            "sha256": overlay["installer"]["sha256"],
        }

    monkeypatch.setattr(run_campaign, "_remote_file_identity", identity)
    monkeypatch.setattr(run_campaign, "remote_output", lambda command: "")
    evidence = run_campaign.verify_runtime_overlay_remote(contract)
    assert evidence["remote_read_only"] is True
    assert evidence["injection"]["mechanism"] == "PYTHONPATH"

    monkeypatch.setattr(
        run_campaign,
        "remote_output",
        lambda command: f"{overlay['directory']}/writable.py\n",
    )
    with pytest.raises(run_campaign.CampaignExecutionError):
        run_campaign.verify_runtime_overlay_remote(contract)

    def mismatched(path: str) -> dict:
        value = identity(path)
        if path == overlay["installer"]["path"]:
            value["sha256"] = "0" * 64
        return value

    monkeypatch.setattr(run_campaign, "_remote_file_identity", mismatched)
    monkeypatch.setattr(run_campaign, "remote_output", lambda command: "")
    with pytest.raises(run_campaign.CampaignExecutionError):
        run_campaign.verify_runtime_overlay_remote(contract)


def test_runner_training_command_uses_same_sealed_overlay(contract):
    original = "mask2former train -e {config_path}"
    runner = SimpleNamespace(
        skill_ctx=SimpleNamespace(action_cfg={"command": original})
    )
    digest = run_campaign.configure_runner_runtime_overlay(runner, contract)
    wrapped = runner.skill_ctx.action_cfg["command"]
    assert digest == run_campaign.text_sha256(wrapped)
    assert contract["runtime"]["tao_pytorch_overlay"]["archive"][
        "sha256"
    ] in wrapped
    assert wrapped.endswith(original)


def test_data_only_stage_manifest_has_no_model_or_scheduler_execution():
    record = qualification_campaign._records()[0]
    stage = {
        "schema_version": 1,
        "model": "mask2former",
        "registry_sha256": (
            campaign_contract.mask2former_registry_snapshot()[
                "registry_sha256"
            ]
        ),
        "created_at_utc": "2026-07-31T00:00:00Z",
        "stage_complete": True,
        "remote_read_only": True,
        "cpu_model_runs": 0,
        "gpu_model_runs": 0,
        "smoke_model_runs": 0,
        "mini_step_runs": 0,
        "scheduler_jobs_submitted": 0,
        "checkpoints": [
            {
                "id": record["id"],
                "path": "/lustre/ptms/mask2former_swint.pth",
                "size_bytes": record["expected_size_bytes"],
                "sha256": "d" * 64,
                "mode": "444",
                "immutable_source_identity": record["source"][
                    "immutable_identity"
                ],
                "remote_read_only": True,
            }
        ],
    }
    stage["manifest_sha256"] = canonical_sha256(stage)
    validated = qualification_campaign.validate_stage_document(stage)
    assert validated["gpu_model_runs"] == 0
    assert validated["scheduler_jobs_submitted"] == 0
