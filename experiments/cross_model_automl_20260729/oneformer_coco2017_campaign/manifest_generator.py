#!/usr/bin/env python3

"""Seal the OneFormer/full-COCO2017 campaign from immutable inputs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path
from typing import Any

from tao_automl.ptm_registry import canonical_sha256, load_ptm_registry

from . import campaign_contract, ptm_stage


HERE = Path(__file__).resolve().parent
DEFAULT_REPOSITORY = Path("/localhome/local-rarunachalam/tao-automl")
DEFAULT_WHEEL = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/wheel/746d8b7f/"
    "nvidia_tao_automl-0.1.0-py3-none-any.whl"
)
DEFAULT_SDK = Path(
    "/localhome/local-rarunachalam/.tao/worktrees/tao-sdk-slurm-a2e50d0"
)
DEFAULT_SKILLS = Path(
    "/localhome/local-rarunachalam/.tao/worktrees/tao-skills-release-7.1.0"
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
DEFAULT_QUALIFICATION = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/"
    "oneformer_coco2017_ptm_qualification_v2/completion.json"
)
DEFAULT_PTM_STAGE_MANIFEST = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/"
    "oneformer_coco2017_ptm_qualification_v1/ptm_stage_manifest.json"
)
DEFAULT_RUNTIME_OVERLAY = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "oneformer-runtime-product-fixes-c25a20e0/"
    "oneformer-runtime-overlay.tar"
)
EXPECTED_DATASET_FILE_MANIFEST_SHA256 = (
    "10566a60498de9998154f44a34445a488c9f030e09f2a7346d20a4a1c55f804e"
)
EXPECTED_STAGE_MANIFEST_SHA256 = (
    "437ff12490637950707b9b951d820ea34d38b926080a478a5d182c2d284a0c5d"
)
EXPECTED_WHEEL_SHA256 = (
    "e0ca6ab7efdd3af886b61b312fcf6f28506f440450d137316e075a463fcc7622"
)
WHEEL_BUILD_COMMIT = (
    "746d8b7a7134f3786c90b87122ddf8421183e871"
)
EXPECTED_WHEEL_REGISTRY_FILE_SHA256 = (
    "c5dc1fb5573cfb553150713d94fc2f5ff7f7fb1d4698bc8f9dc14fa4ed664153"
)
EXPECTED_WHEEL_ONEFORMER_REGISTRY_SHA256 = (
    "3872bec8c0e58f79cd2f941d18bcc1bcb5660ed90c9a4500f8ab5cf3004bde2a"
)
EXPECTED_SDK_COMMIT = "a2e50d0930c3e3785b4b39fa8c3da88b39ff89e5"
EXPECTED_SKILLS_COMMIT = "2e9c1b25f3c7cb1ae444c75652e36c47eace8229"


class ManifestGenerationError(RuntimeError):
    """The campaign cannot be sealed from the supplied artifacts."""


def runtime_overlay_record(path: str | Path) -> dict[str, Any]:
    """Validate the reviewed TAO PyTorch overlay without installing it."""
    archive = Path(path).resolve()
    frozen = campaign_contract.FROZEN_RUNTIME_OVERLAY
    if (
        not archive.is_file()
        or archive.stat().st_size != frozen["archive_size_bytes"]
        or campaign_contract.sha256_file(archive) != frozen["archive_sha256"]
    ):
        raise ManifestGenerationError(
            "reviewed OneFormer runtime-overlay archive is unavailable or changed"
        )
    manifest_member = (
        f"{frozen['archive_root']}/MANIFEST.json"
    )
    installer_member = (
        f"{frozen['archive_root']}/install_overlay.py"
    )
    try:
        with tarfile.open(archive, "r") as bundle:
            members = {member.name: member for member in bundle.getmembers()}
            manifest_bytes = bundle.extractfile(members[manifest_member]).read()
            installer_bytes = bundle.extractfile(members[installer_member]).read()
            manifest = json.loads(manifest_bytes)
    except (KeyError, OSError, tarfile.TarError, ValueError) as exc:
        raise ManifestGenerationError(
            "reviewed OneFormer runtime-overlay archive is invalid"
        ) from exc
    if (
        hashlib.sha256(manifest_bytes).hexdigest()
        != frozen["manifest_sha256"]
        or hashlib.sha256(installer_bytes).hexdigest()
        != frozen["installer_sha256"]
        or manifest.get("artifact_type") != frozen["artifact_type"]
        or manifest.get("scope") != frozen["scope"]
        or manifest.get("source", {}).get("commit")
        != frozen["source_commit"]
        or manifest.get("source", {}).get("base_commit")
        != frozen["base_commit"]
        or manifest.get("container", {}).get("sha256")
        != campaign_contract.FROZEN_SQSH["sha256"]
        or manifest.get("container", {}).get("site_packages")
        != frozen["base_site_packages"]
        or manifest.get("runtime_contract", {}).get(
            "panoptic_primary_metric"
        )
        != "PQ"
        or len(manifest.get("files", ())) != frozen["file_count"]
    ):
        raise ManifestGenerationError(
            "reviewed OneFormer runtime-overlay manifest changed"
        )
    return {
        "local_archive_path": str(archive),
        "archive_sha256": frozen["archive_sha256"],
        "archive_size_bytes": frozen["archive_size_bytes"],
        "manifest_sha256": frozen["manifest_sha256"],
        "installer_sha256": frozen["installer_sha256"],
        "source_commit": frozen["source_commit"],
        "file_count": frozen["file_count"],
    }


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
            "canonical COCO2017 file manifest is unavailable or changed"
        )
    stage_path = Path(stage_manifest).resolve()
    if (
        not stage_path.is_file()
        or campaign_contract.sha256_file(stage_path)
        != EXPECTED_STAGE_MANIFEST_SHA256
    ):
        raise ManifestGenerationError(
            "final segmentation stage manifest is unavailable or changed"
        )
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    execution = stage.get("execution_contract", {})
    validation = stage.get("validation", {})
    coco = stage.get("datasets", {}).get("coco2017", {})
    file_manifest = coco.get("file_manifest", {})
    splits = coco.get("splits", {})
    categories = coco.get("categories", {})
    assets = coco.get("tao_assets", {})
    sources = coco.get("source_archives", {})
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
        or coco.get("dataset_id") != "coco2017_instance_panoptic"
        or coco.get("lustre_root")
        != (
            "/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/"
            "cross_model_automl_20260729/coco2017_instance_panoptic_v1"
        )
        or splits.get("train_images") != 118287
        or splits.get("val_images") != 5000
        or splits.get("train_panoptic_pngs") != 118287
        or splits.get("val_panoptic_pngs") != 5000
        or splits.get("train_panoptic_segments") != 1329984
        or splits.get("val_panoptic_segments") != 56728
        or categories.get("panoptic_total") != 133
        or assets.get("panoptic_label_map", {}).get("sha256")
        != "4b28b3773f0f8e63d836dc20da77276633da72178453458b79e32be8e892ce56"
        or assets.get("instance_label_map", {}).get("sha256")
        != "67f15c4dd7d52aa73025da8307dec17e907f13db6d5d82332a670f73da68c306"
        or sources.get("all_archive_integrity_checks_passed") is not True
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
    root = coco["lustre_root"]
    return {
        "id": "coco_2017_full_instance_panoptic",
        "official_source": "https://cocodataset.org/",
        "license": (
            "COCO annotations CC BY 4.0; source images retain individual licenses"
        ),
        "root": root,
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
        "manifest_path": str(path),
        "manifest_sha256": EXPECTED_DATASET_FILE_MANIFEST_SHA256,
        "file_manifest_entry_count": 246593,
        "remote_sha256sum_check": "passed_all_246593",
        "stage_manifest_path": str(stage_path),
        "stage_manifest_lustre_path": (
            f"{root}/dataset_stage_manifest.v1.json"
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
    runtime_overlay: Path,
) -> dict[str, Any]:
    if (
        not wheel.is_file()
        or campaign_contract.sha256_file(wheel) != EXPECTED_WHEEL_SHA256
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
    head = _git(repository, "rev-parse", "HEAD")
    registry_path = repository / "src/tao_automl/data/ptm_registry.v1.json"
    try:
        source_registry = json.loads(registry_path.read_text(encoding="utf-8"))
        oneformer_sha = canonical_sha256(
            source_registry["models"]["oneformer"]
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ManifestGenerationError(
            "campaign source lacks the wheel's OneFormer registry"
        ) from exc
    if oneformer_sha != EXPECTED_WHEEL_ONEFORMER_REGISTRY_SHA256:
        raise ManifestGenerationError(
            "campaign source does not match the wheel's OneFormer inventory"
        )
    overlay = runtime_overlay_record(runtime_overlay)
    if (
        not ptm_stage_manifest.is_file()
        or ptm_stage_manifest.is_symlink()
        or ptm_stage_manifest.stat().st_mode & 0o222
    ):
        raise ManifestGenerationError(
            "immutable OneFormer PTM stage manifest is unavailable"
        )
    stage_document = json.loads(
        ptm_stage_manifest.read_text(encoding="utf-8")
    )
    stage_root = stage_document.get("publication", {}).get(
        "canonical_root"
    )
    try:
        stage_document = ptm_stage.validate_stage_manifest(
            stage_document,
            registry=load_ptm_registry(),
            canonical_root=stage_root,
        )
    except Exception as exc:
        raise ManifestGenerationError(
            "OneFormer PTM stage manifest is invalid"
        ) from exc
    return {
        "repository": str(repository.resolve()),
        "source_commit": head,
        "source_dirty": False,
        "wheel_path": str(wheel.resolve()),
        "wheel_sha256": EXPECTED_WHEEL_SHA256,
        "wheel_build_commit": WHEEL_BUILD_COMMIT,
        "wheel_registry_file_sha256": (
            EXPECTED_WHEEL_REGISTRY_FILE_SHA256
        ),
        "wheel_oneformer_registry_sha256": (
            EXPECTED_WHEEL_ONEFORMER_REGISTRY_SHA256
        ),
        "sdk_dir": str(sdk.resolve()),
        "sdk_commit": EXPECTED_SDK_COMMIT,
        "skills_repository": str(skills.resolve()),
        "skills_commit": EXPECTED_SKILLS_COMMIT,
        "skill_dir": str(
            (skills / "skills/models/tao-train-oneformer").resolve()
        ),
        "qualification_evidence_path": str(qualification.resolve()),
        "ptm_stage_manifest_path": str(ptm_stage_manifest.resolve()),
        "ptm_stage_manifest_sha256": campaign_contract.sha256_file(
            ptm_stage_manifest
        ),
        "ptm_stage_content_sha256": stage_document["manifest_sha256"],
        "runtime_overlay_local_archive_path": overlay[
            "local_archive_path"
        ],
        "runtime_overlay_local_identity": overlay,
        "partition": "polar3",
        "account": "edgeai_tao-ptm_image-foundation-model-clip",
        "base_results_dir": (
            "/lustre/fsw/portfolios/edgeai/projects/"
            "edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam"
        ),
        "container_mounts": "/lustre",
        "time_hours": 4.0,
        "timeout_hours": 3.8,
        "max_job_retries": campaign_contract.FROZEN_SLURM_RETRY_CAP,
        "hardware_contract": copy.deepcopy(campaign_contract.FROZEN_HARDWARE),
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
    runtime_overlay: str | Path = DEFAULT_RUNTIME_OVERLAY,
) -> dict[str, Any]:
    repository_path = Path(repository).resolve()
    value = campaign_contract.build_preregistered_contract(
        campaign_id=(
            "oneformer-coco2017-objective-aware-three-mode-v2-20260801"
        ),
        dataset=dataset_record(dataset_manifest, stage_manifest),
        skill_dir=(
            Path(skills).resolve()
            / "skills/models/tao-train-oneformer"
        ),
        runtime=_runtime(
            repository=repository_path,
            wheel=Path(wheel).resolve(),
            sdk=Path(sdk).resolve(),
            skills=Path(skills).resolve(),
            qualification=Path(qualification),
            ptm_stage_manifest=Path(ptm_stage_manifest),
            runtime_overlay=Path(runtime_overlay),
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
        "oneformer_latency_worker_sha256": campaign_contract.sha256_file(
            HERE / "oneformer_latency_worker.py"
        ),
        "static_sqsh_audit_sha256": campaign_contract.sha256_file(
            HERE / "static_sqsh_audit.v1.json"
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
        "--dataset-manifest", type=Path, default=DEFAULT_DATASET_MANIFEST
    )
    parser.add_argument(
        "--stage-manifest", type=Path, default=DEFAULT_STAGE_MANIFEST
    )
    parser.add_argument(
        "--qualification", type=Path, default=DEFAULT_QUALIFICATION
    )
    parser.add_argument(
        "--ptm-stage-manifest",
        type=Path,
        default=DEFAULT_PTM_STAGE_MANIFEST,
    )
    parser.add_argument(
        "--runtime-overlay",
        type=Path,
        default=DEFAULT_RUNTIME_OVERLAY,
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
        runtime_overlay=args.runtime_overlay,
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
                    "the sealed overlay remediates the immutable base-SQSH "
                    "findings; the automatic trigger waits for the exact "
                    "overlay on Lustre plus direct full-run PTM qualification "
                    "and supported registry status"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
