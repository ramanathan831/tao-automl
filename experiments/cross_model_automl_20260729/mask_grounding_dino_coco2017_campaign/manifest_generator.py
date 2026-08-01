#!/usr/bin/env python3

"""Seal the Mask Grounding DINO/COCO2017 campaign after immutable prerequisites exist."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path
from typing import Any

from tao_automl.ptm_registry import canonical_sha256

from . import campaign_contract


HERE = Path(__file__).resolve().parent
DEFAULT_REPOSITORY = Path("/localhome/local-rarunachalam/tao-automl")
DEFAULT_WHEEL = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/wheel/1fa7e75066bf/"
    "nvidia_tao_automl-0.1.0-py3-none-any.whl"
)
DEFAULT_SDK = Path(
    "/localhome/local-rarunachalam/.tao/worktrees/"
    "tao-sdk-bounded-self-requeue"
)
DEFAULT_SKILLS = Path(
    "/localhome/local-rarunachalam/.tao/worktrees/"
    "tao-skills-release-7.1.0"
)
DEFAULT_DATASET_MANIFEST = Path(
    "/localhome/local-rarunachalam/.tao/datasets/"
    "cross_model_automl_20260729/manifests/"
    "coco2017_instance_panoptic_v1.FILE_MANIFEST.sha256"
)
DEFAULT_STAGE_MANIFEST = (
    DEFAULT_REPOSITORY
    / "experiments/cross_model_automl_20260729/"
    "segmentation_datasets/dataset_stage_manifest.v1.json"
)
DEFAULT_CONTIGUOUS_VALIDATION_MANIFEST = (
    HERE / "coco2017_contiguous_validation.v1.json"
)
DEFAULT_TEXT_ENCODER_STAGE = (
    DEFAULT_REPOSITORY
    / "experiments/cross_model_automl_20260729/"
    "grounding_dino_shared_detection/runtime_inputs.stage.v1.json"
)
DEFAULT_QUALIFICATION = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/"
    "mask_grounding_dino_coco2017_ptm_qualification_v2/completion.json"
)
DEFAULT_PREDECESSOR_QUALIFICATION = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/"
    "mask_grounding_dino_coco2017_ptm_qualification_v1/completion.json"
)
DEFAULT_PTM_STAGE_MANIFEST = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/"
    "mask_grounding_dino_coco2017_ptm_qualification_v1/ptm_stage_manifest.json"
)
EXPECTED_DATASET_FILE_MANIFEST_SHA256 = (
    "10566a60498de9998154f44a34445a488c9f030e09f2a7346d20a4a1c55f804e"
)
EXPECTED_STAGE_MANIFEST_SHA256 = (
    "437ff12490637950707b9b951d820ea34d38b926080a478a5d182c2d284a0c5d"
)
EXPECTED_CONTIGUOUS_VALIDATION_MANIFEST_SHA256 = (
    "3c2d09d20211017575a2c51a6797ef91f1939340d978a5d11d1d1edab1a30b2d"
)
EXPECTED_TEXT_ENCODER_STAGE_SHA256 = (
    "ac5b6c12bc7d5abd06beaeb61c79426a6f917d4671551fce202fa63fe6dbe160"
)
EXPECTED_WHEEL_SHA256 = (
    "1fa7e75066bf8a58432e1b2672f86a88a2bf4d7a6b37331ee3ac02e87369275f"
)
EXPECTED_SDK_COMMIT = "1a981d79af40d156735f3d89b98495e7818d0891"
EXPECTED_SKILLS_COMMIT = "2e9c1b25f3c7cb1ae444c75652e36c47eace8229"
EXPECTED_PREDECESSOR_QUALIFICATION_SHA256 = (
    "a48d8d8d2a5c65e35c9d39bd5ed1362be54e2be0b89dcda5471812da331a6996"
)


class ManifestGenerationError(RuntimeError):
    """The campaign cannot be sealed from the supplied artifacts."""


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def dataset_record(
    manifest: str | Path,
    stage_manifest: str | Path = DEFAULT_STAGE_MANIFEST,
    contiguous_validation_manifest: str | Path = (
        DEFAULT_CONTIGUOUS_VALIDATION_MANIFEST
    ),
) -> dict[str, Any]:
    path = Path(manifest).resolve()
    if (
        not path.is_file()
        or campaign_contract.sha256_file(path)
        != EXPECTED_DATASET_FILE_MANIFEST_SHA256
    ):
        raise ManifestGenerationError(
            "canonical COCO2017 file manifest is unavailable or changed"
        )
    stage_path = Path(stage_manifest).resolve()
    if (
        not stage_path.is_file()
        or campaign_contract.sha256_file(stage_path)
        != EXPECTED_STAGE_MANIFEST_SHA256
    ):
        raise ManifestGenerationError(
            "final segmentation dataset stage manifest is unavailable or changed"
        )
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    contiguous_path = Path(contiguous_validation_manifest).resolve()
    if (
        not contiguous_path.is_file()
        or campaign_contract.sha256_file(contiguous_path)
        != EXPECTED_CONTIGUOUS_VALIDATION_MANIFEST_SHA256
    ):
        raise ManifestGenerationError(
            "contiguous COCO validation manifest is unavailable or changed"
        )
    contiguous = json.loads(contiguous_path.read_text(encoding="utf-8"))
    execution = stage.get("execution_contract", {})
    validation = stage.get("validation", {})
    coco = stage.get("datasets", {}).get("coco2017", {})
    file_manifest = coco.get("file_manifest", {})
    source_archives = coco.get("source_archives", {})
    odvg = coco.get("odvg_projection", {})
    contiguous_source = contiguous.get("source", {})
    contiguous_output = contiguous.get("output", {})
    contiguous_verification = contiguous.get("verification", {})
    if (
        stage.get("schema_version") != 1
        or execution.get("data_only") is not True
        or execution.get("model_invoked") is not False
        or execution.get("cpu_model_smoke_run") is not False
        or execution.get("gpu_model_smoke_run") is not False
        or execution.get("training_run") is not False
        or execution.get("evaluation_run") is not False
        or execution.get("latency_benchmark_run") is not False
        or execution.get("slurm_job_submitted") is not False
        or validation.get("coco2017_subreport_content_sha256")
        != "deced9d6766344fe6fc69cd9de3bcff2cba456a14b3391d07bcedb74c250909e"
        or coco.get("dataset_id")
        != "coco2017_instance_panoptic"
        or coco.get("lustre_root")
        != (
            "/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/"
            "cross_model_automl_20260729/"
            "coco2017_instance_panoptic_v1"
        )
        or coco.get("task_scope", {}).get("mask_grounding_dino")
        != "category_prompted_grounded_instance_segmentation"
        or coco.get("splits", {}).get("train_images") != 118287
        or coco.get("splits", {}).get("val_images") != 5000
        or coco.get("splits", {}).get("train_instance_annotations") != 860001
        or coco.get("splits", {}).get("val_instance_annotations") != 36781
        or coco.get("categories", {}).get("instance_things") != 80
        or coco.get("tao_assets", {}).get("instance_label_map", {}).get(
            "categories"
        )
        != 80
        or coco.get("tao_assets", {}).get("instance_label_map", {}).get(
            "sha256"
        )
        != "67f15c4dd7d52aa73025da8307dec17e907f13db6d5d82332a670f73da68c306"
        or odvg.get("converter_commit")
        != "dcea3a39bd3e4709e2325e4b61a4f179efebde4c"
        or odvg.get("official_function")
        != (
            "nvidia_tao_ds.annotations.conversion.coco_to_odvg."
            "convert_coco_to_odvg"
        )
        or odvg.get("projected_images") != 117266
        or odvg.get("projected_annotations") != 860001
        or odvg.get("masks_preserved_exactly") != 860001
        or odvg.get("jsonl_sha256")
        != "d5deb4f5cfe027786fb1ceb52632ad6d3ef027e95e434525ba715d6841fb2921"
        or odvg.get("label_map_sha256")
        != "02075d96f6bf06d061f9329b4775dc7c3bb5ac140c77bc5c0e465d305c46d6c1"
        or contiguous.get("schema_version") != 1
        or contiguous.get("scope")
        != "mask_grounding_dino_category_prompted_od_validation_annotations"
        or contiguous.get("execution_contract", {}).get(
            "model_invoked"
        )
        is not False
        or contiguous.get("execution_contract", {}).get(
            "slurm_job_submitted"
        )
        is not False
        or contiguous_source.get("dataset_stage_manifest_sha256")
        != EXPECTED_STAGE_MANIFEST_SHA256
        or contiguous_source.get("sha256")
        != "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"
        or contiguous_output.get("lustre_path")
        != campaign_contract.FROZEN_CONTIGUOUS_VALIDATION_JSON
        or contiguous_output.get("sha256")
        != campaign_contract.FROZEN_CONTIGUOUS_VALIDATION_SHA256
        or contiguous_output.get("category_ids") != list(range(80))
        or contiguous_output.get("remote_read_only") is not True
        or contiguous_verification.get("annotations_preserved_except_category_id")
        != 36781
        or contiguous_verification.get("segmentations_preserved_exactly")
        != 36781
        or contiguous_verification.get("repeat_conversion_byte_identical")
        is not True
        or contiguous_verification.get("remote_sha256sum_check") != "passed"
        or source_archives.get("all_archive_integrity_checks_passed") is not True
        or source_archives.get("annotations_trainval2017.zip", {}).get("sha256")
        != "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268"
        or source_archives.get(
            "panoptic_annotations_trainval2017.zip", {}
        ).get("sha256")
        != "c05f76d2129b6b561eb70efe16e7006df62f73fb92889132d373b9d90e31a370"
        or source_archives.get("train2017.zip", {}).get("sha256")
        != "69a8bb58ea5f8f99d24875f21416de2e9ded3178e903f1f7603e283b9e06d929"
        or source_archives.get("val2017.zip", {}).get("sha256")
        != "4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05"
        or file_manifest.get("entries") != 246593
        or file_manifest.get("sha256")
        != EXPECTED_DATASET_FILE_MANIFEST_SHA256
        or file_manifest.get("remote_sha256sum_check") != "passed"
        or file_manifest.get("remote_file_set_check") != "passed"
        or coco.get("remote_read_only") is not True
        or coco.get("remote_writable_entries_after_lock") != 0
        or stage.get("transfer_provenance", {}).get(
            "remote_bytes_verified_against_local_manifest"
        )
        is not True
    ):
        raise ManifestGenerationError(
            "final COCO2017 stage provenance does not pass the frozen contract"
        )
    return {
        "id": "coco2017_full_category_prompted_grounded_instance_segmentation",
        "official_sources": {
            name: item["url"] for name, item in source_archives.items()
            if isinstance(item, dict) and "url" in item
        },
        "license": "Creative Commons Attribution 4.0 for COCO annotations",
        "prepared_root": (
            "/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/"
            "cross_model_automl_20260729/"
            "coco2017_instance_panoptic_v1"
        ),
        "train_image_count": 118287,
        "validation_image_count": 5000,
        "train_instance_annotations": 860001,
        "validation_instance_annotations": 36781,
        "num_classes": 80,
        "train_instance_json_sha256": (
            "610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d"
        ),
        "validation_instance_json_sha256": (
            "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"
        ),
        "label_map_sha256": (
            "67f15c4dd7d52aa73025da8307dec17e907f13db6d5d82332a670f73da68c306"
        ),
        "train_odvg_jsonl_sha256": odvg["jsonl_sha256"],
        "train_odvg_label_map_sha256": odvg["label_map_sha256"],
        "train_odvg_projected_images": odvg["projected_images"],
        "train_odvg_projected_annotations": odvg[
            "projected_annotations"
        ],
        "train_odvg_masks_preserved": odvg["masks_preserved_exactly"],
        "contiguous_validation_json_path": contiguous_output["lustre_path"],
        "contiguous_validation_json_sha256": contiguous_output["sha256"],
        "contiguous_validation_manifest_path": (
            "/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/"
            "cross_model_automl_20260729/"
            "mask_grounding_dino_coco2017_od_v1/"
            "coco2017_contiguous_validation.v1.json"
        ),
        "contiguous_validation_manifest_local_path": str(contiguous_path),
        "contiguous_validation_manifest_sha256": (
            EXPECTED_CONTIGUOUS_VALIDATION_MANIFEST_SHA256
        ),
        "contiguous_validation_image_count": contiguous_output["image_count"],
        "contiguous_validation_annotation_count": contiguous_output[
            "annotation_count"
        ],
        "contiguous_validation_category_ids": contiguous_output[
            "category_ids"
        ],
        "contiguous_validation_remote_read_only": True,
        "official_archive_sha256": {
            name: item["sha256"] for name, item in source_archives.items()
            if isinstance(item, dict) and "sha256" in item
        },
        "content_sha256": (
            "deced9d6766344fe6fc69cd9de3bcff2cba456a14b3391d07bcedb74c250909e"
        ),
        "manifest_path": str(path),
        "manifest_sha256": EXPECTED_DATASET_FILE_MANIFEST_SHA256,
        "file_manifest_entry_count": 246593,
        "remote_sha256sum_check": "passed_all_246593",
        "stage_manifest_path": str(stage_path),
        "stage_manifest_lustre_path": (
            "/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/"
            "cross_model_automl_20260729/coco2017_instance_panoptic_v1/"
            "dataset_stage_manifest.v1.json"
        ),
        "stage_manifest_sha256": EXPECTED_STAGE_MANIFEST_SHA256,
        "remote_file_manifest_path": file_manifest["lustre_path"],
        "remote_read_only": True,
        "remote_writable_entries_after_lock": 0,
    }


def ptm_stage_record(path: str | Path) -> dict[str, Any]:
    """Validate and bind the immutable official Mask Grounding DINO PTM stage."""
    stage_path = Path(path).resolve()
    if not stage_path.is_file():
        raise ManifestGenerationError(
            f"PTM stage manifest is unavailable: {stage_path}"
        )
    document = json.loads(stage_path.read_text(encoding="utf-8"))
    supplied = document.get("manifest_sha256")
    payload = copy.deepcopy(document)
    payload.pop("manifest_sha256", None)
    snapshot = campaign_contract.mask_grounding_dino_registry_snapshot()
    records = document.get("checkpoints")
    if (
        supplied != canonical_sha256(payload)
        or document.get("schema_version") != 1
        or document.get("model") != "mask_grounding_dino"
        or not isinstance(document.get("registry_sha256"), str)
        or len(document["registry_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in document["registry_sha256"]
        )
        or document.get("stage_complete") is not True
        or document.get("remote_read_only") is not True
        or document.get("cpu_model_runs") != 0
        or document.get("smoke_model_runs") != 0
        or document.get("mini_step_runs") != 0
        or not isinstance(records, list)
    ):
        raise ManifestGenerationError(
            "official Mask Grounding DINO PTM stage is incomplete or changed"
        )
    by_id = {
        item.get("id"): item
        for item in records
        if isinstance(item, dict)
    }
    expected = {
        item["id"]: item for item in snapshot["records"]
    }
    if set(by_id) != set(expected) or len(records) != len(expected):
        raise ManifestGenerationError(
            "PTM stage must contain every and only official Mask Grounding DINO arm"
        )
    for checkpoint_id, registry_record in expected.items():
        item = by_id[checkpoint_id]
        digest = item.get("sha256")
        if (
            not isinstance(item.get("path"), str)
            or not item["path"].startswith("/lustre/")
            or isinstance(item.get("size_bytes"), bool)
            or item.get("size_bytes")
            != registry_record["expected_size_bytes"]
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or item.get("immutable_source_identity")
            != registry_record["source"]["immutable_identity"]
            or (
                registry_record.get("sha256") is not None
                and digest != registry_record["sha256"]
            )
            or item.get("remote_read_only") is not True
        ):
            raise ManifestGenerationError(
                f"official PTM stage record changed: {checkpoint_id}"
            )
    return {
        "path": str(stage_path),
        "sha256": campaign_contract.sha256_file(stage_path),
        "manifest_sha256": supplied,
        "checkpoint_ids": sorted(by_id),
        "checkpoints": copy.deepcopy(records),
    }


def qualification_evidence_record(path: str | Path) -> dict[str, Any]:
    """Bind the immutable v2 direct-full-run completion for local eligibility."""
    evidence_path = Path(path).resolve()
    if not evidence_path.is_file():
        raise ManifestGenerationError(
            "completed Mask Grounding DINO qualification is unavailable"
        )
    document = json.loads(evidence_path.read_text(encoding="utf-8"))
    supplied = document.get("evidence_sha256")
    payload = copy.deepcopy(document)
    payload.pop("evidence_sha256", None)
    workflows = document.get("workflows")
    if (
        supplied != canonical_sha256(payload)
        or document.get("schema_version") != 1
        or document.get("model") != "mask_grounding_dino"
        or document.get("task")
        != "category_prompted_grounded_instance_segmentation"
        or document.get("primary_metric") != "segm_val_mAP50_95"
        or document.get("sqsh_sha256")
        != campaign_contract.FROZEN_SQSH["sha256"]
        or document.get("cpu_model_runs") != 0
        or document.get("smoke_model_runs") != 0
        or document.get("mini_step_runs") != 0
        or document.get("replacement_workflows_submitted") is not False
        or not isinstance(workflows, list)
        or len(workflows) != 4
        or any(
            not isinstance(item, dict)
            or item.get("terminal") is not True
            or item.get("status") not in {"success", "failure"}
            for item in workflows
        )
    ):
        raise ManifestGenerationError(
            "completed Mask Grounding DINO qualification is invalid"
        )
    for name in (
        "qualification_contract_sha256",
        "qualification_campaign_sha256",
        "registry_sha256",
        "ptm_stage_manifest_sha256",
    ):
        digest = document.get(name)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ManifestGenerationError(
                f"qualification {name} is not lowercase SHA-256"
            )
    return {
        "schema_version": 2,
        "kind": "direct_full_gpu_qualification_runtime_local_v2",
        "enabled": True,
        "scope": "campaign_local_in_memory_projection",
        "model": "mask_grounding_dino",
        "task": "category_prompted_grounded_instance_segmentation",
        "tao_version": "7.1.0",
        "container_sha256": campaign_contract.FROZEN_SQSH["sha256"],
        "base_registry_version": (
            campaign_contract.mask_grounding_dino_registry_snapshot()[
                "registry_version"
            ]
        ),
        "base_registry_sha256": document["registry_sha256"],
        "qualification_file_sha256": campaign_contract.sha256_file(
            evidence_path
        ),
        "qualification_evidence_sha256": supplied,
        "qualification_contract_sha256": document[
            "qualification_contract_sha256"
        ],
        "qualification_campaign_sha256": document[
            "qualification_campaign_sha256"
        ],
        "repository_registry_mutation_allowed": False,
        "failed_arm_promotion_allowed": False,
        "unsupported_arm_promotion_allowed": False,
        "agent_override_allowed": False,
    }


def _runtime(
    *,
    repository: Path,
    wheel: Path,
    sdk: Path,
    skills: Path,
    qualification: Path,
    ptm_stage_manifest: Path,
    text_encoder_stage: Path,
    predecessor_qualification: Path,
) -> dict[str, Any]:
    if not wheel.is_file() or (
        campaign_contract.sha256_file(wheel) != EXPECTED_WHEEL_SHA256
    ):
        raise ManifestGenerationError("production AutoML wheel changed")
    if _git(sdk, "rev-parse", "HEAD") != EXPECTED_SDK_COMMIT:
        raise ManifestGenerationError("TAO SDK commit changed")
    if _git(skills, "rev-parse", "HEAD") != EXPECTED_SKILLS_COMMIT:
        raise ManifestGenerationError("TAO skills commit changed")
    if _git(repository, "status", "--porcelain"):
        raise ManifestGenerationError(
            "AutoML source must be clean before campaign sealing"
        )
    ptm_stage = ptm_stage_record(ptm_stage_manifest)
    if (
        not text_encoder_stage.is_file()
        or campaign_contract.sha256_file(text_encoder_stage)
        != EXPECTED_TEXT_ENCODER_STAGE_SHA256
    ):
        raise ManifestGenerationError(
            "frozen Grounding DINO text-encoder stage changed"
        )
    text_stage = json.loads(
        text_encoder_stage.read_text(encoding="utf-8")
    )
    text_encoder = text_stage.get("text_encoder", {})
    if (
        text_stage.get("execution", {}).get("cpu_model_runs") != 0
        or text_stage.get("execution", {}).get("gpu_model_runs") != 0
        or text_stage.get("execution", {}).get("scheduler_jobs_submitted") != 0
        or text_encoder.get("lustre_root")
        != campaign_contract.FROZEN_TEXT_ENCODER_ROOT
        or text_encoder.get("tree_sha256")
        != campaign_contract.FROZEN_TEXT_ENCODER_TREE_SHA256
        or text_encoder.get("offline_runtime", {}).get("HF_HUB_OFFLINE")
        != "1"
        or text_encoder.get("offline_runtime", {}).get(
            "TRANSFORMERS_OFFLINE"
        )
        != "1"
    ):
        raise ManifestGenerationError(
            "frozen Grounding DINO text-encoder provenance changed"
        )
    predecessor = predecessor_qualification.resolve()
    if (
        not predecessor.is_file()
        or campaign_contract.sha256_file(predecessor)
        != EXPECTED_PREDECESSOR_QUALIFICATION_SHA256
    ):
        raise ManifestGenerationError(
            "preserved v1 qualification evidence is unavailable or changed"
        )
    predecessor_document = json.loads(
        predecessor.read_text(encoding="utf-8")
    )
    predecessor_workflows = predecessor_document.get("workflows")
    if (
        not isinstance(predecessor_workflows, list)
        or len(predecessor_workflows) != 4
        or any(
            not isinstance(item, dict)
            or item.get("status") != "failure"
            or item.get("failure_preserved") is not True
            for item in predecessor_workflows
        )
    ):
        raise ManifestGenerationError(
            "preserved v1 qualification is not the sealed four-arm failure"
        )
    source_commit = _git(repository, "rev-parse", "HEAD")
    eligibility = qualification_evidence_record(qualification)
    eligibility.update(
        {
            "eligibility_source_commit": source_commit,
            "wheel_sha256": EXPECTED_WHEEL_SHA256,
            "sdk_commit": EXPECTED_SDK_COMMIT,
            "skills_commit": EXPECTED_SKILLS_COMMIT,
        }
    )
    return {
        "repository": str(repository.resolve()),
        "source_commit": source_commit,
        "source_dirty": False,
        "wheel_path": str(wheel.resolve()),
        "wheel_sha256": EXPECTED_WHEEL_SHA256,
        "sdk_dir": str(sdk.resolve()),
        "sdk_commit": EXPECTED_SDK_COMMIT,
        "skills_repository": str(skills.resolve()),
        "skills_commit": EXPECTED_SKILLS_COMMIT,
        "skill_dir": str(
            (
                skills
                / "skills/models/tao-train-mask-grounding-dino"
            ).resolve()
        ),
        "text_encoder_stage_path": str(text_encoder_stage.resolve()),
        "text_encoder_stage_sha256": EXPECTED_TEXT_ENCODER_STAGE_SHA256,
        "text_encoder_root": text_encoder["lustre_root"],
        "text_encoder_tree_sha256": text_encoder["tree_sha256"],
        "text_encoder_files": [
            {
                "path": item["lustre"]["path"],
                "size_bytes": item["lustre"]["size_bytes"],
                "sha256": item["lustre"]["sha256"],
                "mode": item["lustre"]["mode"],
            }
            for item in text_encoder["files"]
        ],
        "offline_environment": copy.deepcopy(
            text_encoder["offline_runtime"]
        ),
        "qualification_evidence_path": str(qualification.resolve()),
        "runtime_local_eligibility": eligibility,
        "predecessor_failure_evidence": {
            "path": str(predecessor),
            "sha256": EXPECTED_PREDECESSOR_QUALIFICATION_SHA256,
            "campaign_id": predecessor_document.get("campaign_id"),
            "workflow_count": 4,
            "all_terminal_failures_preserved": True,
            "replacement_submitted": False,
        },
        "ptm_stage_manifest_path": ptm_stage["path"],
        "ptm_stage_manifest_sha256": ptm_stage["sha256"],
        "ptm_stage_content_sha256": ptm_stage["manifest_sha256"],
        "partition": "polar3",
        "account": "edgeai_tao-ptm_image-foundation-model-clip",
        "base_results_dir": (
            "/lustre/fsw/portfolios/edgeai/users/rarunachalam"
        ),
        "container_mounts": "/lustre",
        "time_hours": 4.0,
        "timeout_hours": 3.8,
        "max_job_retries": campaign_contract.FROZEN_SLURM_RETRY_CAP,
        "hardware_contract": copy.deepcopy(
            campaign_contract.FROZEN_HARDWARE
        ),
    }


def build_contract(
    *,
    repository: str | Path = DEFAULT_REPOSITORY,
    wheel: str | Path = DEFAULT_WHEEL,
    sdk: str | Path = DEFAULT_SDK,
    skills: str | Path = DEFAULT_SKILLS,
    dataset_manifest: str | Path = DEFAULT_DATASET_MANIFEST,
    stage_manifest: str | Path = DEFAULT_STAGE_MANIFEST,
    contiguous_validation_manifest: str | Path = (
        DEFAULT_CONTIGUOUS_VALIDATION_MANIFEST
    ),
    qualification: str | Path = DEFAULT_QUALIFICATION,
    ptm_stage_manifest: str | Path = DEFAULT_PTM_STAGE_MANIFEST,
    text_encoder_stage: str | Path = DEFAULT_TEXT_ENCODER_STAGE,
    predecessor_qualification: str | Path = (
        DEFAULT_PREDECESSOR_QUALIFICATION
    ),
) -> dict[str, Any]:
    repository_path = Path(repository).resolve()
    value = campaign_contract.build_preregistered_contract(
        campaign_id=(
            "mask_grounding_dino-coco2017-objective-aware-three-mode-v3-20260801"
        ),
        dataset=dataset_record(
            dataset_manifest,
            stage_manifest,
            contiguous_validation_manifest,
        ),
        skill_dir=(
            Path(skills).resolve()
            / "skills/models/tao-train-mask-grounding-dino"
        ),
        runtime=_runtime(
            repository=repository_path,
            wheel=Path(wheel).resolve(),
            sdk=Path(sdk).resolve(),
            skills=Path(skills).resolve(),
            qualification=Path(qualification),
            ptm_stage_manifest=Path(ptm_stage_manifest),
            text_encoder_stage=Path(text_encoder_stage),
            predecessor_qualification=Path(predecessor_qualification),
        ),
    )
    value.pop("contract_sha256")
    value["launcher_integrity"] = {
        "ddp_strategy_audit_sha256": campaign_contract.sha256_file(
            HERE / "ddp_strategy_audit.v2.json"
        ),
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
        "mask_grounding_dino_latency_worker_sha256": (
            campaign_contract.sha256_file(
                HERE / "mask_grounding_dino_latency_worker.py"
            )
        ),
        "checkpoint_resume_sha256": campaign_contract.sha256_file(
            HERE.parent / "checkpoint_resume.py"
        ),
        "manifest_generator_sha256": campaign_contract.sha256_file(
            HERE / "manifest_generator.py"
        ),
    }
    value["contract_sha256"] = canonical_sha256(value)
    return campaign_contract.validate_contract(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=DEFAULT_REPOSITORY)
    parser.add_argument("--wheel", type=Path, default=DEFAULT_WHEEL)
    parser.add_argument("--sdk", type=Path, default=DEFAULT_SDK)
    parser.add_argument("--skills", type=Path, default=DEFAULT_SKILLS)
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=DEFAULT_DATASET_MANIFEST,
    )
    parser.add_argument(
        "--stage-manifest",
        type=Path,
        default=DEFAULT_STAGE_MANIFEST,
    )
    parser.add_argument(
        "--contiguous-validation-manifest",
        type=Path,
        default=DEFAULT_CONTIGUOUS_VALIDATION_MANIFEST,
    )
    parser.add_argument(
        "--text-encoder-stage",
        type=Path,
        default=DEFAULT_TEXT_ENCODER_STAGE,
    )
    parser.add_argument(
        "--qualification", type=Path, default=DEFAULT_QUALIFICATION
    )
    parser.add_argument(
        "--predecessor-qualification",
        type=Path,
        default=DEFAULT_PREDECESSOR_QUALIFICATION,
    )
    parser.add_argument(
        "--ptm-stage-manifest",
        type=Path,
        default=DEFAULT_PTM_STAGE_MANIFEST,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    contract = build_contract(
        repository=args.repository,
        wheel=args.wheel,
        sdk=args.sdk,
        skills=args.skills,
        dataset_manifest=args.dataset_manifest,
        stage_manifest=args.stage_manifest,
        contiguous_validation_manifest=(
            args.contiguous_validation_manifest
        ),
        qualification=args.qualification,
        ptm_stage_manifest=args.ptm_stage_manifest,
        text_encoder_stage=args.text_encoder_stage,
        predecessor_qualification=args.predecessor_qualification,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "contract_sha256": contract["contract_sha256"],
                "launch_authorized": False,
                "reason": (
                    "dynamic direct-full-run PTM qualification and supported "
                    "registry gates are evaluated by the automatic trigger"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
