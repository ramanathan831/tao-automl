from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tao_automl.ptm_registry import canonical_sha256, load_ptm_registry

from . import campaign_contract, manifest_generator, run_campaign
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
        "partition": "polar3",
        "account": "edgeai_tao-ptm_image-foundation-model-clip",
        "base_results_dir": "/lustre/fsw/portfolios/edgeai/users/rarunachalam",
        "container_mounts": "/lustre",
        "time_hours": 8.0,
        "timeout_hours": 7.8,
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
            "checkpoint_id": checkpoint_id,
            "status": "failure",
            "terminal": True,
            "failure_preserved": True,
            "failure_code": "direct_full_run_failed",
            "failure_reason": "frozen test failure",
        }
        value["workflow_sha256"] = canonical_sha256(value)
        return value
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
            "training_epochs": 10,
            "validation_interval": 1,
            "validation_record_count": 10,
            "nodes": 1,
            "gpus": 8,
            "val_miou": metric,
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
            "test_miou": metric,
        },
        "agent_intervention_flags": {
            name: False for name in campaign_contract.AGENT_FLAGS
        },
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
        "schema_version": 1,
        "campaign_id": "segformer-direct-full-qualification-test",
        "model": "segformer",
        "task": "semantic_segmentation",
        "registry_sha256": snapshot["registry_sha256"],
        "sqsh_sha256": campaign_contract.FROZEN_SQSH["sha256"],
        "cpu_model_runs": 0,
        "smoke_model_runs": 0,
        "mini_step_runs": 0,
        "workflows": workflows,
    }
    value["evidence_sha256"] = canonical_sha256(value)
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


def test_profile_is_full_dataset_eight_gpu_not_smoke():
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
    assert train["validation_interval"] == 1
    assert train["tensorboard"]["enabled"] is False


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
