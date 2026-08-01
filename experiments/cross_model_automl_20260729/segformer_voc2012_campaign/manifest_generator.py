#!/usr/bin/env python3

"""Seal the SegFormer/VOC2012 campaign after immutable prerequisites exist."""

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
    "cross_model_automl_20260729/wheel/1919228616b8/"
    "nvidia_tao_automl-0.1.0-py3-none-any.whl"
)
DEFAULT_SDK = Path(
    "/localhome/local-rarunachalam/.tao/worktrees/"
    "tao-sdk-slurm-a2e50d0"
)
DEFAULT_SKILLS = Path(
    "/localhome/local-rarunachalam/.tao/worktrees/"
    "tao-skills-release-7.1.0"
)
DEFAULT_DATASET_MANIFEST = Path(
    "/localhome/local-rarunachalam/.tao/datasets/"
    "cross_model_automl_20260729/manifests/"
    "voc2012_segmentation_v1.FILE_MANIFEST.sha256"
)
DEFAULT_STAGE_MANIFEST = (
    DEFAULT_REPOSITORY
    / "experiments/cross_model_automl_20260729/"
    "segmentation_datasets/dataset_stage_manifest.v1.json"
)
DEFAULT_QUALIFICATION = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/"
    "segformer_voc2012_ptm_qualification_v5/completion.json"
)
DEFAULT_PTM_STAGE_MANIFEST = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/"
    "segformer_voc2012_ptm_qualification_v5/ptm_stage_manifest.json"
)
EXPECTED_DATASET_FILE_MANIFEST_SHA256 = (
    "051ab20215b8e6976763ac82a3db20a68264759edef3d62fd0c8553c501123ff"
)
EXPECTED_STAGE_MANIFEST_SHA256 = (
    "437ff12490637950707b9b951d820ea34d38b926080a478a5d182c2d284a0c5d"
)
EXPECTED_WHEEL_SHA256 = (
    "3463187cb76ec3d07c64a21eaf34140e56bf251b46e56ce3c89c33728ee22784"
)
EXPECTED_SDK_COMMIT = "a2e50d0930c3e3785b4b39fa8c3da88b39ff89e5"
EXPECTED_SKILLS_COMMIT = "2e9c1b25f3c7cb1ae444c75652e36c47eace8229"


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
) -> dict[str, Any]:
    path = Path(manifest).resolve()
    if (
        not path.is_file()
        or campaign_contract.sha256_file(path)
        != EXPECTED_DATASET_FILE_MANIFEST_SHA256
    ):
        raise ManifestGenerationError(
            "canonical VOC2012 file manifest is unavailable or changed"
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
    execution = stage.get("execution_contract", {})
    validation = stage.get("validation", {})
    voc = stage.get("datasets", {}).get("voc2012", {})
    file_manifest = voc.get("file_manifest", {})
    source_archive = voc.get("source_archive", {})
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
        or validation.get("voc2012_subreport_content_sha256")
        != "815b5d01b625238b449c4bca828bf96107b367f0f4d5d8a31d2f97c6161a5de0"
        or voc.get("dataset_id")
        != "pascal_voc2012_segmentation_trainval"
        or voc.get("prepared_root")
        != (
            "/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/"
            "cross_model_automl_20260729/"
            "voc2012_segmentation_v1/prepared"
        )
        or voc.get("splits")
        != {
            "train_pairs": 1464,
            "val_pairs": 1449,
            "train_val_disjoint": True,
        }
        or voc.get("label_contract", {}).get("num_model_classes") != 21
        or voc.get("label_contract", {}).get("valid_class_ids")
        != list(range(21))
        or voc.get("label_contract", {}).get("ignore_id") != 255
        or voc.get("label_contract", {}).get("label_transform") != "None"
        or source_archive.get("url")
        != (
            "https://thor.robots.ox.ac.uk/pascal/VOC/"
            "voc2012/VOCtrainval_11-May-2012.tar"
        )
        or source_archive.get("sha256")
        != "e14f763270cf193d0b5f74b169f44157a4b0c6efa708f4dd0ff78ee691763bcb"
        or source_archive.get("archive_integrity_passed") is not True
        or file_manifest.get("entries") != 5827
        or file_manifest.get("sha256")
        != EXPECTED_DATASET_FILE_MANIFEST_SHA256
        or file_manifest.get("remote_sha256sum_check") != "passed"
        or file_manifest.get("remote_file_set_check") != "passed"
        or voc.get("remote_read_only") is not True
        or voc.get("remote_writable_entries_after_lock") != 0
        or stage.get("transfer_provenance", {}).get(
            "remote_bytes_verified_against_local_manifest"
        )
        is not True
    ):
        raise ManifestGenerationError(
            "final VOC2012 stage provenance does not pass the frozen contract"
        )
    return {
        "id": "pascal_voc_2012_full_semantic_segmentation",
        "official_source": (
            "https://thor.robots.ox.ac.uk/pascal/VOC/"
            "voc2012/VOCtrainval_11-May-2012.tar"
        ),
        "license": "PASCAL VOC terms of use",
        "prepared_root": (
            "/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/"
            "cross_model_automl_20260729/"
            "voc2012_segmentation_v1/prepared"
        ),
        "train_image_count": 1464,
        "train_mask_count": 1464,
        "validation_image_count": 1449,
        "validation_mask_count": 1449,
        "num_classes": 21,
        "ignore_label": 255,
        "official_archive_sha256": (
            "e14f763270cf193d0b5f74b169f44157a4b0c6efa708f4dd0ff78ee691763bcb"
        ),
        # This is the stable, VOC-only semantic-validation subreport hash.
        "content_sha256": (
            "815b5d01b625238b449c4bca828bf96107b367f0f4d5d8a31d2f97c6161a5de0"
        ),
        "manifest_path": str(path),
        "manifest_sha256": EXPECTED_DATASET_FILE_MANIFEST_SHA256,
        "file_manifest_entry_count": 5827,
        "remote_sha256sum_check": "passed_all_5827",
        "stage_manifest_path": str(stage_path),
        "stage_manifest_lustre_path": (
            "/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/"
            "cross_model_automl_20260729/voc2012_segmentation_v1/"
            "dataset_stage_manifest.v1.json"
        ),
        "stage_manifest_sha256": EXPECTED_STAGE_MANIFEST_SHA256,
        "remote_file_manifest_path": file_manifest["lustre_path"],
        "remote_read_only": True,
        "remote_writable_entries_after_lock": 0,
    }


def _runtime(
    *,
    repository: Path,
    wheel: Path,
    sdk: Path,
    skills: Path,
    qualification: Path,
    ptm_stage_manifest: Path,
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
    return {
        "repository": str(repository.resolve()),
        "source_commit": _git(repository, "rev-parse", "HEAD"),
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
                / "skills/models/tao-train-segformer"
            ).resolve()
        ),
        "qualification_evidence_path": str(qualification.resolve()),
        "ptm_stage_manifest_path": str(ptm_stage_manifest.resolve()),
        "partition": campaign_contract.FROZEN_SLURM_PARTITION,
        "account": "edgeai_tao-ptm_image-foundation-model-clip",
        "base_results_dir": (
            "/lustre/fsw/portfolios/edgeai/projects/"
            "edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam"
        ),
        "container_mounts": "/lustre",
        "time_hours": campaign_contract.FROZEN_SLURM_TIME_HOURS,
        "timeout_hours": campaign_contract.FROZEN_SLURM_TIMEOUT_HOURS,
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
    qualification: str | Path = DEFAULT_QUALIFICATION,
    ptm_stage_manifest: str | Path = DEFAULT_PTM_STAGE_MANIFEST,
) -> dict[str, Any]:
    repository_path = Path(repository).resolve()
    value = campaign_contract.build_preregistered_contract(
        campaign_id=(
            "segformer-voc2012-objective-aware-three-mode-20260801-v5"
        ),
        dataset=dataset_record(dataset_manifest, stage_manifest),
        skill_dir=(
            Path(skills).resolve()
            / "skills/models/tao-train-segformer"
        ),
        runtime=_runtime(
            repository=repository_path,
            wheel=Path(wheel).resolve(),
            sdk=Path(sdk).resolve(),
            skills=Path(skills).resolve(),
            qualification=Path(qualification),
            ptm_stage_manifest=Path(ptm_stage_manifest),
        ),
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
        "--qualification", type=Path, default=DEFAULT_QUALIFICATION
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
        qualification=args.qualification,
        ptm_stage_manifest=args.ptm_stage_manifest,
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
