"""Contract tests for the direct full RT-DETR qualification."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from . import manifest_generator
from . import run_campaign


HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / "campaign.v1.json"


@pytest.fixture()
def manifest():
    return manifest_generator.load_manifest(MANIFEST_PATH)


def _successful_workflow(manifest, workflow_id):
    record = run_campaign._initial_workflow(manifest, workflow_id)
    train_root = f"/lustre/results/{workflow_id}/train-job"
    eval_root = f"/lustre/results/{workflow_id}/evaluation-job"
    record.update(
        {
            "status": "success",
            "terminal": True,
            "failure_preserved": False,
            "metrics": {"mAP": 0.31, "mAP50": 0.52},
            "jobs": {
                "train": {
                    "status": "Complete",
                    "result_root": train_root,
                    "status_evidence": {
                        "path": f"{train_root}/results_dir/train/status.json",
                        "sha256": "a" * 64,
                        "size_bytes": 123,
                        "record_count": 11,
                        "validation_record_count": 10,
                        "validation_metrics": [
                            {"mAP": 0.2, "mAP50": 0.4}
                            for _ in range(10)
                        ],
                        "terminal_success_message": (
                            "Train finished successfully."
                        ),
                        "terminal_success": True,
                    },
                },
                "evaluation": {
                    "status": "Complete",
                    "result_root": eval_root,
                    "status_evidence": {
                        "path": (
                            f"{eval_root}/results_dir/evaluate/status.json"
                        ),
                        "sha256": "b" * 64,
                        "size_bytes": 123,
                        "record_count": 2,
                        "test_metric_record_count": 1,
                        "metrics": {"mAP": 0.31, "mAP50": 0.52},
                        "terminal_success_message": (
                            "Evaluate finished successfully."
                        ),
                        "terminal_success": True,
                    },
                },
            },
        }
    )
    return record


def _rehash(value):
    value.pop("completion_sha256", None)
    value["completion_sha256"] = manifest_generator.canonical_sha(value)
    return value


def test_manifest_is_exactly_four_direct_full_gpu_workflows(manifest):
    assert manifest["model"] == "rtdetr"
    assert manifest["cpu_runs"] == 0
    assert manifest["smoke_runs"] == 0
    assert manifest["execution"] == manifest_generator.EXECUTION_CONTRACT
    assert manifest["qualification"] == manifest_generator.QUALIFICATION_CONTRACT
    assert manifest["runtime"]["nodes"] == 1
    assert manifest["runtime"]["tasks_per_node"] == 1
    assert manifest["runtime"]["gpus_per_node"] == 8
    assert manifest["runtime"]["sqsh_path"].endswith(".sqsh")
    assert manifest["runtime"]["sqsh_direct_path"] is True
    assert manifest["runtime"]["slurm_image_conversion"] is False
    assert tuple(item["id"] for item in manifest["ptms"]) == (
        manifest_generator.EXPECTED_PTMS
    )


def test_shared_synthetic_dataset_identity_is_frozen(manifest):
    dataset = manifest["dataset"]
    assert dataset["id"] == "tao_od_synthetic_full_dino_coco"
    assert dataset["num_classes"] == 5
    assert dataset["eval_class_ids"] == [1, 2, 3, 4]
    assert dataset["remap_mscoco_category"] is False
    assert dataset["splits"]["train"]["image_count"] == 1414
    assert dataset["splits"]["validation"]["image_count"] == 353
    assert dataset["splits"]["train"]["annotation_sha256"] == (
        "7401a1245dc0b691c40f9f53cf4f46f9b96a3e0bc3dcfd357de038074acc1994"
    )
    assert dataset["splits"]["validation"]["annotation_sha256"] == (
        "9b715b689e9a17588805faad26ed94597886d28ac687438dcb778de433f997af"
    )


def test_all_ptm_artifacts_match_registry_and_remain_unverified(manifest):
    registry, records = manifest_generator._registry_records()
    assert registry.schema_version == 1
    for ptm in manifest["ptms"]:
        record = records[ptm["id"]]
        assert ptm["registry_status_before_qualification"] == "unverified"
        assert ptm["artifact"]["sha256"] == record["sha256"]
        assert ptm["artifact"]["size_bytes"] == record["expected_size_bytes"]
        assert ptm["source_identity"] == record["source"]["immutable_identity"]
        assert ptm["artifact"]["availability_required_at_launch"] is True
        assert all(
            value is False for value in ptm["agent_intervention_flags"].values()
        )


@pytest.mark.parametrize(
    ("workflow_id", "backbone", "spatial_size"),
    [
        ("trafficcam_resnet50", "resnet_50", [544, 960]),
        ("trafficcam_resnet18", "resnet_18", [544, 960]),
        ("warehouse_resnet50", "resnet_50", [640, 640]),
        ("warehouse_efficientvit_l2", "efficientvit_l2", [544, 960]),
    ],
)
def test_train_spec_is_ten_epoch_full_dataset_torchrun_ddp(
    manifest, workflow_id, backbone, spatial_size
):
    spec = run_campaign.build_train_spec(manifest, workflow_id)
    assert spec["model"]["backbone"] == backbone
    assert spec["dataset"]["num_classes"] == 5
    assert spec["dataset"]["eval_class_ids"] == [1, 2, 3, 4]
    assert spec["dataset"]["remap_mscoco_category"] is False
    assert isinstance(spec["dataset"]["train_data_sources"], list)
    assert len(spec["dataset"]["train_data_sources"]) == 1
    assert isinstance(spec["dataset"]["val_data_sources"], dict)
    assert spec["dataset"]["val_data_sources"]["json_file"].endswith(
        "/val/annotations.json"
    )
    assert spec["dataset"]["augmentation"]["train_spatial_size"] == spatial_size
    assert spec["dataset"]["augmentation"]["eval_spatial_size"] == spatial_size
    assert spec["train"]["num_epochs"] == 10
    assert spec["train"]["validation_interval"] == 1
    assert spec["train"]["checkpoint_interval"] == 10
    assert spec["train"]["num_nodes"] == 1
    assert spec["train"]["num_gpus"] == 8
    assert spec["train"]["gpu_ids"] == list(range(8))
    assert spec["train"]["distributed_strategy"] == "ddp"
    assert spec["train"]["is_dry_run"] is False


@pytest.mark.parametrize(
    "workflow_id",
    [
        "trafficcam_resnet50",
        "trafficcam_resnet18",
        "warehouse_resnet50",
        "warehouse_efficientvit_l2",
    ],
)
def test_standalone_evaluation_uses_dict_source_and_terminal_checkpoint(
    manifest, workflow_id
):
    checkpoint = f"/lustre/results/{workflow_id}/model_epoch_009_step_100.pth"
    train = run_campaign.build_train_spec(manifest, workflow_id)
    evaluate = run_campaign.build_evaluation_spec(
        manifest, workflow_id, checkpoint
    )
    assert evaluate["model"] == train["model"]
    assert evaluate["evaluate"]["checkpoint"] == checkpoint
    assert evaluate["evaluate"]["num_gpus"] == 8
    assert evaluate["evaluate"]["gpu_ids"] == list(range(8))
    assert isinstance(evaluate["dataset"]["test_data_sources"], dict)
    assert evaluate["dataset"]["test_data_sources"]["json_file"].endswith(
        "/val/annotations.json"
    )
    assert evaluate["dataset"]["remap_mscoco_category"] is False


def test_completion_is_automatic_hashed_and_covers_all_workflows(
    manifest, tmp_path
):
    workflow_ids = tuple(item["workflow_id"] for item in manifest["ptms"])
    for workflow_id in workflow_ids:
        run_campaign.atomic_json(
            tmp_path / workflow_id / "workflow_completion.json",
            _successful_workflow(manifest, workflow_id),
        )
    completion = run_campaign.build_completion(
        manifest,
        tmp_path,
        workflow_ids,
        {workflow_id: 0 for workflow_id in workflow_ids},
    )
    assert completion["status"] == "success"
    assert completion["completion_generated_automatically"] is True
    assert completion["logical_workflows_submitted"] == 4
    assert set(completion["outcomes"]) == set(workflow_ids)
    assert run_campaign.validate_completion(completion, manifest) == completion


def test_completion_preserves_a_terminal_failure_without_replacement(
    manifest, tmp_path
):
    workflow_ids = tuple(item["workflow_id"] for item in manifest["ptms"])
    successful = workflow_ids[:-1]
    for workflow_id in successful:
        run_campaign.atomic_json(
            tmp_path / workflow_id / "workflow_completion.json",
            _successful_workflow(manifest, workflow_id),
        )
    completion = run_campaign.build_completion(
        manifest,
        tmp_path,
        workflow_ids,
        {
            **{workflow_id: 0 for workflow_id in successful},
            workflow_ids[-1]: 1,
        },
    )
    assert completion["status"] == "terminal_with_failures"
    assert completion["successful_workflows"] == 3
    assert completion["failed_workflows"] == 1
    assert completion["replacement_workflows_submitted"] is False
    assert run_campaign.validate_completion(completion, manifest) == completion


def test_completion_rejects_tampering(manifest, tmp_path):
    workflow_ids = tuple(item["workflow_id"] for item in manifest["ptms"])
    for workflow_id in workflow_ids:
        run_campaign.atomic_json(
            tmp_path / workflow_id / "workflow_completion.json",
            _successful_workflow(manifest, workflow_id),
        )
    completion = run_campaign.build_completion(
        manifest,
        tmp_path,
        workflow_ids,
        {workflow_id: 0 for workflow_id in workflow_ids},
    )
    tampered = copy.deepcopy(completion)
    tampered["workflows"][0]["metrics"]["mAP50"] = 1.01
    _rehash(tampered)
    with pytest.raises(
        run_campaign.CampaignExecutionError,
        match="invalid mAP50",
    ):
        run_campaign.validate_completion(tampered, manifest)


def test_launch_requires_explicit_direct_full_acknowledgement(manifest):
    with pytest.raises(
        run_campaign.CampaignExecutionError,
        match="acknowledge-direct-full-dataset",
    ):
        run_campaign.main(["--manifest", str(MANIFEST_PATH), "--launch"])


def test_dry_run_is_plan_only_and_has_no_cpu_or_smoke_path(
    manifest, capsys
):
    assert run_campaign.main(["--manifest", str(MANIFEST_PATH)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["launch"] is False
    assert plan["logical_workflows"] == 4
    assert plan["parallel"] is True
    assert plan["cpu_runs"] == 0
    assert plan["smoke_runs"] == 0
    assert plan["ministep_runs"] == 0
    assert plan["completion_generated_automatically"] is True


def test_runtime_disables_conversion_but_submits_pinned_sqsh(
    manifest, monkeypatch
):
    monkeypatch.delenv("SLURM_USE_SQSH", raising=False)
    run_campaign.configure_slurm_runtime(manifest)
    assert __import__("os").environ["SLURM_USE_SQSH"] == "false"

    calls = []

    class SDK:
        @staticmethod
        def create_job(**kwargs):
            calls.append(kwargs)
            return object()

    run_campaign._submit_job(SDK(), manifest, "rtdetr train")
    assert calls == [
        {
            "image": manifest["runtime"]["sqsh_path"],
            "command": "rtdetr train",
            "gpu_count": 8,
            "num_nodes": 1,
            "partition": manifest["runtime"]["partition"],
            "account": manifest["runtime"]["account"],
        }
    ]


def test_manifest_builder_rejects_missing_or_reordered_ptm_cohort():
    inputs = json.loads(
        manifest_generator.DEFAULT_INPUTS.read_text(encoding="utf-8")
    )
    first = next(iter(inputs["ptm_runtime"]))
    inputs["ptm_runtime"].pop(first)
    with pytest.raises(
        manifest_generator.ManifestError,
        match="four registry PTMs in frozen order",
    ):
        manifest_generator.build_manifest(inputs)
