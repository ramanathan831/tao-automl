#!/usr/bin/env python3
"""Run one TAO model through the skill-based AutoMLRunner workflow.

This is intentionally a single-model runner. Algorithm-level orchestration is
done outside this script so each algorithm run folder can be deleted and
reported independently.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from tao_automl.runner import AutoMLRunner, _extract_metric_from_logs, _metric_aliases
from tao_sdk.platforms.docker import DockerSDK
from tao_sdk.script_runner import build_entrypoint


LOG = logging.getLogger("tao_automl_validation")
BUCKET_ROOT = "s3://nvcf-storage-handling/data"
VISUAL_CHANGENET_BACKBONE_CONTAINER_PATH = "/data/pretrained_models/C-RADIOv2_B.safetensors"
BEVFUSION_CONTAINER_DATA_ROOT = "/data/bevfusion_root"
CLIP_CONTAINER_DATA_ROOT = "/data/clip_fallback"
GROUNDING_DINO_INFER_CONTAINER_ROOT = "/data/grounding_dino_infer"
NVPANOPTIX3D_INFER_CONTAINER_ROOT = "/data/nvpanoptix3d_infer"


@dataclass(frozen=True)
class ModelProfile:
    train_uri: str
    eval_uri: str = ""
    inference_uri: str = ""
    calibration_uri: str = ""
    data_format: str | None = None
    num_classes: int | None = None
    data_source_dataset_name: str | None = None
    dataset_name: str | None = None
    model_type: str | None = None
    blocked: str | None = None
    captions: tuple[str, ...] = ()


MODEL_PROFILES: dict[str, ModelProfile] = {
    "action-recognition": ModelProfile(
        f"{BUCKET_ROOT}/purpose_built_models_action_recognition_train",
        f"{BUCKET_ROOT}/purpose_built_models_action_recognition_train",
    ),
    "bevfusion": ModelProfile(
        f"{BUCKET_ROOT}/purpose_built_models_bevfusion_train",
    ),
    "centerpose": ModelProfile(
        f"{BUCKET_ROOT}/purpose_built_models_centerpose_train",
        f"{BUCKET_ROOT}/purpose_built_models_centerpose_val",
    ),
    "classification-pyt": ModelProfile(
        f"{BUCKET_ROOT}/classification_train",
        f"{BUCKET_ROOT}/classification_val",
        num_classes=20,
    ),
    "clip": ModelProfile(f"{BUCKET_ROOT}/auto_label_train", f"{BUCKET_ROOT}/auto_label_val"),
    "cosmos-rl": ModelProfile(
        f"{BUCKET_ROOT}/cosmos_rl_wts_train",
        f"{BUCKET_ROOT}/cosmos_rl_wts_val",
        data_format="llava",
    ),
    "deformable-detr": ModelProfile(
        f"{BUCKET_ROOT}/object_detection_pyt_train",
        f"{BUCKET_ROOT}/object_detection_pyt_val",
        num_classes=6,
    ),
    "depth-net-mono": ModelProfile(
        f"{BUCKET_ROOT}/purpose_built_models_depth_net_train",
        f"{BUCKET_ROOT}/purpose_built_models_depth_net_val",
        data_source_dataset_name="RelativeMonoDataset",
        dataset_name="MonoDataset",
        model_type="RelativeDepthAnything",
    ),
    "depth-net-stereo": ModelProfile(
        f"{BUCKET_ROOT}/purpose_built_models_depth_net_train",
        f"{BUCKET_ROOT}/purpose_built_models_depth_net_val",
        data_source_dataset_name="Middlebury",
        dataset_name="StereoDataset",
        model_type="FoundationStereo",
    ),
    "dino": ModelProfile(
        f"{BUCKET_ROOT}/tao_od_synthetic_subset_train_no_convert",
        f"{BUCKET_ROOT}/tao_od_synthetic_subset_val_no_convert",
        num_classes=6,
    ),
    "grounding-dino": ModelProfile(
        f"{BUCKET_ROOT}/object_detection_grounding_dino_train",
        f"{BUCKET_ROOT}/object_detection_grounding_dino_val",
        num_classes=6,
        captions=("head", "helmet", "person"),
    ),
    "mae": ModelProfile(
        f"{BUCKET_ROOT}/classification_train",
        f"{BUCKET_ROOT}/classification_val",
        num_classes=20,
    ),
    "mal": ModelProfile(f"{BUCKET_ROOT}/auto_label_train", f"{BUCKET_ROOT}/auto_label_val"),
    "mask-grounding-dino": ModelProfile(
        f"{BUCKET_ROOT}/segmentation_mask_grounding_dino_train",
        f"{BUCKET_ROOT}/segmentation_mask_grounding_dino_val",
        num_classes=6,
        captions=("person", "bicycle", "car"),
    ),
    "mask2former": ModelProfile(
        f"{BUCKET_ROOT}/segmentation_mask2former_train",
        f"{BUCKET_ROOT}/segmentation_mask2former_val",
        num_classes=201,
    ),
    "ml-recog": ModelProfile(
        f"{BUCKET_ROOT}/purpose_built_models_ml_recog_train",
        f"{BUCKET_ROOT}/purpose_built_models_ml_recog_train",
    ),
    "nvdinov2": ModelProfile(
        f"{BUCKET_ROOT}/nvdinov2_train_cats_dogs",
        f"{BUCKET_ROOT}/nvdinov2_val_cats_dogs",
        inference_uri=f"{BUCKET_ROOT}/nvdinov2_test_cats_dogs",
    ),
    "nvpanoptix3d": ModelProfile(
        f"{BUCKET_ROOT}/purpose_built_models_nvpanoptix3d_train",
        f"{BUCKET_ROOT}/purpose_built_models_nvpanoptix3d_val",
        inference_uri=f"{BUCKET_ROOT}/purpose_built_models_nvpanoptix3d_val",
    ),
    "ocdnet": ModelProfile(
        f"{BUCKET_ROOT}/purpose_built_models_ocdnet_train",
        f"{BUCKET_ROOT}/purpose_built_models_ocdnet_val",
    ),
    "ocrnet": ModelProfile(
        f"{BUCKET_ROOT}/purpose_built_models_ocrnet_train",
        f"{BUCKET_ROOT}/purpose_built_models_ocrnet_val",
    ),
    "oneformer": ModelProfile(
        f"{BUCKET_ROOT}/segmentation_oneformer_train",
        f"{BUCKET_ROOT}/segmentation_oneformer_val",
        num_classes=133,
    ),
    "optical-inspection": ModelProfile(
        f"{BUCKET_ROOT}/purpose_built_models_optical_inspection_train",
        f"{BUCKET_ROOT}/purpose_built_models_optical_inspection_val",
    ),
    "pointpillars": ModelProfile(
        f"{BUCKET_ROOT}/purpose_built_models_pointpillars_train",
    ),
    "pose-classification": ModelProfile(
        f"{BUCKET_ROOT}/purpose_built_models_pose_classification_train/nvidia",
        f"{BUCKET_ROOT}/purpose_built_models_pose_classification_train/nvidia",
    ),
    "re-identification": ModelProfile(
        f"{BUCKET_ROOT}/purpose_built_models_re_identification_train",
        f"{BUCKET_ROOT}/purpose_built_models_re_identification_train",
        num_classes=100,
    ),
    "rtdetr": ModelProfile(
        f"{BUCKET_ROOT}/object_detection_pyt_train",
        f"{BUCKET_ROOT}/object_detection_pyt_val",
        num_classes=6,
    ),
    "segformer": ModelProfile(
        f"{BUCKET_ROOT}/segmentation_segformer_train",
        f"{BUCKET_ROOT}/segmentation_segformer_val",
        num_classes=6,
    ),
    "sparse4d": ModelProfile(
        f"{BUCKET_ROOT}/purpose_built_models_sparse4d_train",
    ),
    "vila": ModelProfile(
        f"{BUCKET_ROOT}/vila_lita_ft_youcook2_yaml",
        f"{BUCKET_ROOT}/vila_lita_ft_youcook2_val_yaml",
    ),
    "visual-changenet": ModelProfile(
        f"{BUCKET_ROOT}/purpose_built_models_visual_changenet_classify_train",
        f"{BUCKET_ROOT}/purpose_built_models_visual_changenet_classify_val",
    ),
}


def _profile_key_from_network_arch(network_arch: str) -> str:
    return network_arch.replace("_", "-")


def _resolve_model_dir(skill_bank: Path, requested_model: str) -> tuple[Path, str, str]:
    """Return (model_dir, network_arch, profile_key) for old or current names."""
    direct = skill_bank / "models" / requested_model
    if (direct / "references" / "skill_info.yaml").exists():
        info = _read_yaml(direct / "references" / "skill_info.yaml")
        network_arch = info.get("network_arch") or requested_model
        return direct, network_arch, _profile_key_from_network_arch(network_arch)

    requested_key = requested_model.replace("_", "-")
    for candidate in sorted((skill_bank / "models").iterdir()):
        info_path = candidate / "references" / "skill_info.yaml"
        if not info_path.exists():
            continue
        info = _read_yaml(info_path)
        network_arch = info.get("network_arch") or candidate.name
        profile_key = _profile_key_from_network_arch(network_arch)
        if requested_model in {candidate.name, network_arch, profile_key} or requested_key == profile_key:
            return candidate, network_arch, profile_key
    raise KeyError(f"No model skill found for {requested_model!r} under {skill_bank / 'models'}")


def _profile_key_from_model_dir(model_dir: Path) -> str:
    info = _read_yaml(model_dir / "references" / "skill_info.yaml")
    return _profile_key_from_network_arch(info.get("network_arch") or model_dir.name)


CHECKPOINT_SUFFIXES = (
    ".pth",
    ".pth.tar",
    ".pt",
    ".ckpt",
    ".hdf5",
    ".tlt",
    ".safetensors",
)


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def _set_nested(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor: Any = target
    for part in parts[:-1]:
        name, idx = _parse_part(part)
        cursor = cursor.setdefault(name, [] if idx is not None else {})
        if idx is not None:
            while len(cursor) <= idx:
                cursor.append({})
            cursor = cursor[idx]
    name, idx = _parse_part(parts[-1])
    if idx is None:
        cursor[name] = value
        return
    cursor.setdefault(name, [])
    while len(cursor[name]) <= idx:
        cursor[name].append(None)
    cursor[name][idx] = value


def _get_nested(source: dict[str, Any], dotted_key: str) -> Any:
    cursor: Any = source
    for part in dotted_key.split("."):
        name, idx = _parse_part(part)
        if not isinstance(cursor, dict) or name not in cursor:
            return None
        cursor = cursor[name]
        if idx is not None:
            if not isinstance(cursor, list) or len(cursor) <= idx:
                return None
            cursor = cursor[idx]
    return cursor


def _parse_part(part: str) -> tuple[str, int | None]:
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_-]*)(?:\[(\d+)])?$", part)
    if not match:
        return part, None
    return match.group(1), int(match.group(2)) if match.group(2) else None


def _flatten_keys(value: Any, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            full = f"{prefix}.{key}" if prefix else str(key)
            keys.add(full)
            keys |= _flatten_keys(child, full)
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            full = f"{prefix}[{idx}]"
            keys.add(full)
            keys |= _flatten_keys(child, full)
    return keys


def _schema_keys(skill_dir: Path, action: str = "train") -> set[str]:
    schema_path = skill_dir / "schemas" / f"{action}.schema.json"
    keys: set[str] = set()
    if not schema_path.exists():
        return keys

    def walk(schema: Any, prefix: str = "") -> None:
        if not isinstance(schema, dict):
            return
        props = schema.get("properties")
        if isinstance(props, dict):
            for key, child in props.items():
                full = f"{prefix}.{key}" if prefix else str(key)
                keys.add(full)
                walk(child, full)
        items = schema.get("items")
        if isinstance(items, dict) and prefix:
            walk(items, f"{prefix}[0]")

    schema = json.loads(schema_path.read_text())
    walk(schema)
    keys |= _flatten_keys(schema.get("default", {}))
    return keys


def _monitoring_metric(skill_text: str) -> str:
    match = re.search(r"\*\*Monitoring metric:\*\*\s*([^\n]+)", skill_text)
    if not match:
        return "loss"
    metric = match.group(1).strip()
    return metric.split(",", 1)[0].strip()


def _direction(metric: str) -> str:
    return "minimize" if "loss" in metric.lower() else "maximize"


def _parse_action_rows(skill_text: str, action: str) -> list[dict[str, str]]:
    rows = []
    in_table = False
    for raw in skill_text.splitlines():
        line = raw.strip()
        if line.startswith("| Action | Spec Key | Source | Files |"):
            in_table = True
            continue
        if in_table and (not line.startswith("|") or line.startswith("### ")):
            break
        if not in_table or line.startswith("|---"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 5 or cells[0] != action:
            continue
        rows.append({
            "spec_key": cells[1],
            "source": cells[2],
            "files": cells[3],
            "list": cells[4].lower().startswith("y"),
        })
    return rows


def _parse_train_rows(skill_text: str) -> list[dict[str, str]]:
    return _parse_action_rows(skill_text, "train")


def _source_root(profile: ModelProfile, source: str) -> str:
    if source == "train_datasets":
        return profile.train_uri
    if source == "eval_dataset":
        return profile.eval_uri or profile.train_uri
    if source == "inference_dataset":
        return profile.inference_uri or profile.eval_uri or profile.train_uri
    if source == "calibration_dataset":
        return profile.calibration_uri or profile.train_uri
    return profile.train_uri


def _join_uri(root: str, suffix: str) -> str:
    suffix = _clean_file_spec(suffix)
    if not suffix:
        return root
    if suffix.startswith("/") or "://" in suffix:
        return suffix
    return f"{root.rstrip('/')}/{suffix.lstrip('/')}"


def _clean_file_spec(files: str) -> str:
    stripped = files.strip()
    if stripped.startswith("coco_panoptic:"):
        return stripped.split("coco_panoptic:", 1)[1].split(";", 1)[0].strip()
    if stripped.startswith("dataset root containing"):
        return ""
    if stripped.startswith("root directory containing"):
        return ""
    if stripped.startswith("extracted root containing"):
        return ""
    if stripped.startswith("flat folder of"):
        return ""
    if stripped.startswith("LMDB folder containing"):
        return ""
    if stripped.startswith("one image/video or a media folder/archive"):
        return ""
    if " extracted from " in files:
        return files.split(" extracted from ", 1)[1].strip()
    if " when using " in stripped:
        return stripped.split(" when using ", 1)[0].strip()
    if " or " in stripped:
        return stripped.split(" or ", 1)[0].strip()
    return stripped


def _files_mapping(files: str) -> dict[str, str] | None:
    if files.strip().startswith("coco_panoptic:"):
        return None
    if ":" not in files:
        return None
    mapping: dict[str, str] = {}
    for part in files.split(","):
        name, _, value = part.partition(":")
        name = name.strip()
        value = _clean_file_spec(value)
        if name and value and "{" not in value and "from convert" not in value:
            mapping[name] = value
    return mapping or None


def _add_data_source_overrides(
    overrides: dict[str, Any],
    profile: ModelProfile,
    train_rows: list[dict[str, str]],
) -> None:
    for row in train_rows:
        spec_key = row["spec_key"]
        files = row["files"].strip()
        if "{dataset_convert_job_id}" in files or "from convert" in files:
            continue
        if spec_key.endswith(".type") and files in {"ade", "coco", "coco_panoptic"}:
            overrides[spec_key] = files
            continue
        root = _source_root(profile, row["source"])
        if (
            row["source"] == "inference_dataset"
            and not profile.inference_uri
            and profile.eval_uri
            and files == "images_test.tar.gz"
        ):
            overrides[spec_key] = _join_uri(profile.eval_uri, "images_val.tar.gz")
            continue
        if "data_file:" in files and "+ dataset_name" in files:
            item = {
                "data_file": _join_uri(root, files.split("data_file:", 1)[1].split("+", 1)[0].strip()),
                "dataset_name": profile.data_source_dataset_name or "GenericDataset",
            }
            overrides[spec_key] = [item] if row["list"] else item
            continue
        if files == "prompt list":
            overrides[spec_key] = list(profile.captions or ("object",))
            continue
        mapping = _files_mapping(files)
        if mapping and "captions" in mapping:
            image_dir = mapping.get("image_dir")
            if image_dir:
                overrides[f"{spec_key}.image_dir"] = [_join_uri(root, image_dir)]
            overrides[f"{spec_key}.captions"] = list(profile.captions or ("object",))
            continue
        if mapping:
            item = {field: _join_uri(root, suffix) for field, suffix in mapping.items()}
            overrides[spec_key] = [item] if row["list"] else item
        else:
            value = _join_uri(root, _clean_file_spec(files))
            overrides[spec_key] = [value] if row["list"] else value


def _profile_specific_data_overrides(profile_key: str, profile: ModelProfile) -> dict[str, Any]:
    if profile_key == "bevfusion":
        data_prefix = {"pts": "training/velodyne_reduced", "img": "training/image_2"}
        return {
            "dataset.root_dir": BEVFUSION_CONTAINER_DATA_ROOT,
            "dataset.train_dataset.ann_file": (
                f"{BEVFUSION_CONTAINER_DATA_ROOT}/kitti_person_infos_train.pkl"
            ),
            "dataset.train_dataset.data_prefix": data_prefix,
            "dataset.train_dataset.batch_size": 1,
            "dataset.train_dataset.num_workers": 0,
            "dataset.val_dataset.ann_file": (
                f"{BEVFUSION_CONTAINER_DATA_ROOT}/kitti_person_infos_val.pkl"
            ),
            "dataset.val_dataset.data_prefix": data_prefix,
            "dataset.val_dataset.batch_size": 1,
            "dataset.val_dataset.num_workers": 0,
            "dataset.test_dataset.ann_file": (
                f"{BEVFUSION_CONTAINER_DATA_ROOT}/kitti_person_infos_val.pkl"
            ),
            "dataset.test_dataset.data_prefix": data_prefix,
            "dataset.test_dataset.batch_size": 1,
            "dataset.test_dataset.num_workers": 0,
        }
    if profile_key == "clip":
        return {
            "dataset.train.type": "custom",
            "dataset.train.datasets": [{
                "image_dir": f"{CLIP_CONTAINER_DATA_ROOT}/train/images",
                "caption_dir": f"{CLIP_CONTAINER_DATA_ROOT}/train/captions",
            }],
            "dataset.train.wds.root_dir": None,
            "dataset.train.wds.shard_list_file": None,
            "dataset.train.batch_size": 1,
            "dataset.train.num_workers": 0,
            "dataset.val.datasets": [{
                "image_dir": f"{CLIP_CONTAINER_DATA_ROOT}/val/images",
                "caption_dir": f"{CLIP_CONTAINER_DATA_ROOT}/val/captions",
            }],
            "dataset.val.batch_size": 1,
            "dataset.val.num_workers": 0,
        }
    if profile_key == "cosmos-rl":
        return {
            "custom.train_dataset.annotation_path": _join_uri(profile.train_uri, "annotations.json"),
            "custom.train_dataset.media_path": _join_uri(profile.train_uri, "videos.tar.gz"),
            "custom.val_dataset.annotation_path": _join_uri(profile.eval_uri, "annotations.json"),
            "custom.val_dataset.media_path": _join_uri(profile.eval_uri, "videos.tar.gz"),
        }
    if profile_key == "nvpanoptix3d":
        return {
            "dataset.enable_3d": True,
            "dataset.contiguous_id": True,
            "dataset.test.json_path": _join_uri(profile.eval_uri, "meta/val.json"),
            "dataset.test.base_dir": profile.eval_uri,
            "train.optim.monitor_name": "train_loss",
            "train.precision": "fp32",
        }
    if profile_key == "ocdnet":
        return {
            "dataset.train_dataset.data_path": [_join_uri(profile.train_uri, "train.tar.gz")],
            "dataset.validate_dataset.data_path": [_join_uri(profile.eval_uri, "test.tar.gz")],
        }
    if profile_key == "segformer":
        return {"dataset.segment.root_dir": profile.train_uri}
    return {}


def _action_specific_overrides(
    profile_key: str,
    profile: ModelProfile,
    action: str,
    out_dir: Path,
) -> dict[str, Any]:
    if profile_key == "bevfusion" and action in {"evaluate", "inference"}:
        return _profile_specific_data_overrides(profile_key, profile)
    if profile_key == "clip":
        if action == "evaluate":
            return {
                "dataset.val.datasets": [{
                    "image_dir": f"{CLIP_CONTAINER_DATA_ROOT}/val/images",
                    "caption_dir": f"{CLIP_CONTAINER_DATA_ROOT}/val/captions",
                }],
                "dataset.val.batch_size": 1,
                "dataset.val.num_workers": 0,
                "evaluate.batch_size": 1,
            }
        if action == "inference":
            return {
                "inference.datasets": [{
                    "image_dir": f"{CLIP_CONTAINER_DATA_ROOT}/val/images",
                }],
                "inference.text_file": f"{CLIP_CONTAINER_DATA_ROOT}/prompts.txt",
                "inference.batch_size": 1,
            }
    if profile_key == "grounding-dino" and action == "inference":
        return {
            "dataset.infer_data_sources.image_dir": [GROUNDING_DINO_INFER_CONTAINER_ROOT],
            "dataset.infer_data_sources.captions": list(profile.captions or ("object",)),
        }
    if profile_key == "nvpanoptix3d":
        if action == "evaluate":
            return {
                "dataset.frustum_mask_path": _join_uri(profile.eval_uri, "meta/frustum_mask.npz"),
                "dataset.label_map": _join_uri(profile.eval_uri, "meta/colormap.json"),
                "dataset.val.json_path": _join_uri(profile.eval_uri, "meta/val.json"),
                "dataset.val.base_dir": profile.eval_uri,
                "dataset.test.json_path": _join_uri(profile.eval_uri, "meta/val.json"),
                "dataset.test.base_dir": profile.eval_uri,
                "dataset.enable_3d": True,
                "dataset.contiguous_id": True,
            }
        if action == "inference":
            return {
                "dataset.frustum_mask_path": _join_uri(profile.eval_uri, "meta/frustum_mask.npz"),
                "dataset.label_map": _join_uri(profile.eval_uri, "meta/colormap.json"),
                "dataset.enable_3d": True,
                "inference.images_dir": NVPANOPTIX3D_INFER_CONTAINER_ROOT,
                "inference.batch_size": 1,
            }
    if profile_key == "rtdetr" and action == "inference":
        return {
            "dataset.infer_data_sources": {
                "image_dir": [_join_uri(profile.eval_uri or profile.train_uri, "images.tar.gz")],
                "classmap": _join_uri(profile.eval_uri or profile.train_uri, "label_map.txt"),
            }
        }
    if profile_key == "ocrnet":
        if action == "evaluate":
            return {
                "dataset.character_list_file": _join_uri(profile.eval_uri, "character_list"),
                "evaluate.test_dataset_dir": _join_uri(profile.eval_uri, "test.tar.gz"),
                "evaluate.test_dataset_gt_file": _join_uri(profile.eval_uri, "test/gt_new.txt"),
            }
        if action == "inference":
            return {
                "dataset.character_list_file": _join_uri(profile.eval_uri, "character_list"),
                "inference.inference_dataset_dir": _join_uri(profile.eval_uri, "test.tar.gz"),
            }
    if profile_key == "ocdnet":
        if action == "evaluate":
            return {"dataset.validate_dataset.data_path": [_join_uri(profile.eval_uri, "test.tar.gz")]}
        if action == "inference":
            return {"inference.input_folder": _join_uri(profile.eval_uri, "test/img.tar.gz")}
    if profile_key == "segformer" and action in {"evaluate", "inference"}:
        return {"dataset.segment.root_dir": profile.eval_uri or profile.train_uri}
    return {}


def _valid_set(overrides: dict[str, Any], specs: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    valid = dict(overrides)
    merged_keys = keys | _flatten_keys(specs)
    for key in list(valid):
        if key in merged_keys:
            continue
        # Object-valued overrides such as dataset.train_data_sources are valid
        # when a child key exists in the schema.
        prefix = f"{key}."
        list_prefix = f"{key}["
        if any(k.startswith(prefix) or k.startswith(list_prefix) for k in merged_keys):
            continue
        LOG.info("Dropping override not present in train schema: %s", key)
        valid.pop(key)
    return valid


def _minimal_train_overrides(
    specs: dict[str, Any],
    schema_keys: set[str],
    num_classes: int | None,
    model: str,
) -> dict[str, Any]:
    keys = schema_keys | _flatten_keys(specs)
    candidates: dict[str, Any] = {
        "train.num_epochs": 1,
        "train.epoch": 1,
        "train.max_epochs": 1,
        "train.checkpoint_interval": 1,
        "train.validation_interval": 1,
        "train.num_gpus": 1,
        "train.gpu_ids": [0],
        "train.optim.lr_step_size": 1,
        "validation.freq_in_epoch": 1,
        "train.ckpt.save_freq_in_epoch": 1,
        "train.ckpt.max_keep": 2,
        "train.ckpt.export_safetensors": True,
        "dataset.batch_size": 1,
        "dataset.workers": 0,
        "dataset.num_workers": 0,
    }
    if model == "cosmos-rl":
        candidates.update({
            "policy.model_name_or_path": "hf_model://nvidia/Cosmos3-Nano",
            "policy.parallelism.dp_shard_size": 1,
            "policy.parallelism.dp_replicate_size": 1,
            "train.train_batch_per_replica": 1,
            "train.train_policy.mini_batch": 1,
            "train.train_policy.dataset.name": "wts",
            "train.train_policy.dataset.test_size": 0,
            "validation.batch_size": 1,
            "validation.enable_dataset_cache": False,
            "logging.logger": ["console", "tao"],
        })
    if model == "mae":
        candidates.update({
            "dataset.batch_size": 2,
            "train.stage": "finetune",
        })
    if model == "optical-inspection":
        candidates["dataset.batch_size"] = 2
    if model == "re-identification":
        candidates["dataset.batch_size"] = 16
        candidates["dataset.num_instances"] = 4
    if model == "visual-changenet":
        candidates.update({
            "model.backbone.pretrained_backbone_path": VISUAL_CHANGENET_BACKBONE_CONTAINER_PATH,
            "dataset.classify.batch_size": 2,
            "dataset.classify.workers": 0,
        })
    if model in {"depth-net-mono", "depth-net-stereo"}:
        candidates.update({
            "train.precision": "fp32",
            "dataset.train_dataset.batch_size": 1,
            "dataset.train_dataset.workers": 0,
            "dataset.val_dataset.batch_size": 1,
            "dataset.val_dataset.workers": 0,
            "dataset.test_dataset.batch_size": 1,
            "dataset.test_dataset.workers": 0,
            "dataset.infer_dataset.batch_size": 1,
            "dataset.infer_dataset.workers": 0,
        })
    if model == "depth-net-mono":
        candidates.update({
            "model.model_type": "RelativeDepthAnything",
            "dataset.dataset_name": "MonoDataset",
            "dataset.min_depth": None,
            "dataset.max_depth": None,
        })
    if model == "depth-net-stereo":
        candidates.update({
            "model.model_type": "FoundationStereo",
            "dataset.dataset_name": "StereoDataset",
            "dataset.train_dataset.augmentation.crop_size": [128, 128],
            "dataset.val_dataset.augmentation.crop_size": [128, 128],
            "dataset.test_dataset.augmentation.crop_size": [128, 128],
            "dataset.infer_dataset.augmentation.crop_size": [128, 128],
            "dataset.max_disparity": 128,
            "model.max_disparity": 128,
        })
    if num_classes:
        candidates.update({
            "dataset.num_classes": num_classes,
            "model.num_classes": num_classes,
            "model.sem_seg_head.num_classes": num_classes,
            "num_classes": num_classes,
        })
    if model == "mask2former":
        candidates.update({
            "dataset.train.type": "coco_panoptic",
            "dataset.val.type": "coco_panoptic",
            "dataset.test.type": "coco_panoptic",
            "dataset.contiguous_id": False,
            "dataset.augmentation.train_min_size": [128],
            "dataset.augmentation.train_crop_size": [128, 128],
        })
    if model == "oneformer":
        candidates.update({
            "dataset.augmentation.train_min_size": [128],
            "dataset.augmentation.train_crop_size": [128, 128],
            "dataset.augmentation.test_min_size": 128,
            "dataset.augmentation.test_max_size": 256,
        })
    if model == "nvdinov2":
        candidates.update({
            "wandb.enable": False,
            "model.backbone.teacher_type": "vit_s",
            "model.backbone.student_type": "vit_s",
            "model.backbone.img_size": 224,
            "dataset.workers": 2,
            "train.num_prototypes": 1024,
            "train.precision": "32-true",
            "train.use_custom_attention": False,
        })
    if model == "sparse4d":
        candidates.update({
            "dataset.num_frames": 3,
            "dataset.sequences.split_num": 1,
            "dataset.train_dataset.sequences_split_num": 1,
            "model.head.instance_bank.num_anchor": 72,
            "model.head.instance_bank.num_temp_instances": 48,
            "model.head.num_output": 72,
            "train.precision": "fp32",
        })
    return {key: value for key, value in candidates.items() if key in keys}


def _automl_settings(algorithm: str, metric: str, args: argparse.Namespace) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "algorithm": algorithm,
        "metric": metric,
        "direction": _direction(metric),
        "automl_max_recommendations": 2,
        "automl_max_epochs": 2,
        "automl_reduction_factor": 2,
        "automl_max_concurrent": 1,
        "automl_max_trials": 2,
        "automl_min_top_configs": 1,
        "automl_population_size": 2,
        "automl_max_generations": 1,
        "automl_eval_interval": 1,
        "automl_max_experiments": 2,
    }
    if algorithm in {"llm", "hybrid", "autoresearch"}:
        key = os.environ.get("AUTOML_LLM_API_KEY") or os.environ.get("NVIDIA_API_KEY")
        if not key:
            raise RuntimeError("LLM algorithm requested but no LLM API key is present in the environment")
        endpoint = (
            os.environ.get("AUTOML_LLM_ENDPOINT")
            or os.environ.get("base_url")
            or os.environ.get("BASE_URL")
            or os.environ.get("NVIDIA_INFERENCE_ENDPOINT")
        )
        model = (
            os.environ.get("AUTOML_LLM_MODEL")
            or os.environ.get("model")
            or os.environ.get("MODEL")
            or os.environ.get("NVIDIA_INFERENCE_MODEL")
        )
        if not endpoint or not model:
            raise RuntimeError(
                "LLM algorithm requested but no LLM endpoint/model is present in the environment"
            )
        settings.update({
            "llm_endpoint": endpoint,
            "llm_model": model,
            "llm_api_key": key,
        })
    return settings


def _metric_extractor_for(model: str):
    if model != "dino":
        return None

    def extract_dino_map50(logs: str, metric_name: str) -> float | None:
        patterns = [
            r"Validation\s+mAP50\s*[:=]\s*([0-9.]+)",
            r"mAP50\s*[:=]\s*([0-9.]+)",
        ]
        for pattern in patterns:
            matches = list(re.finditer(pattern, logs, flags=re.IGNORECASE))
            if matches:
                return float(matches[-1].group(1))
        return _extract_metric_from_logs(logs, metric_name)

    return extract_dino_map50


def _find_checkpoints(job_root: Path, model: str) -> list[str]:
    checkpoints: list[str] = []
    if not job_root.exists():
        return checkpoints
    for path in job_root.rglob("*"):
        rel = path.relative_to(job_root).as_posix()
        if rel.startswith("inputs/") or rel.startswith("ptm/") or "/inputs/" in rel or "/ptm/" in rel:
            continue
        if model == "cosmos-rl" and path.is_file() and path.name.endswith(".safetensors"):
            checkpoints.append(str(path))
        elif model != "cosmos-rl" and path.is_file() and path.name.lower().endswith(CHECKPOINT_SUFFIXES):
            checkpoints.append(str(path))
        elif path.is_dir() and path.name.lower().startswith(("epoch_", "step_")) and any(path.iterdir()):
            checkpoints.append(str(path))
    return sorted(set(checkpoints))


def _prefer_epoch_or_step_checkpoint(
    checkpoint_paths: list[str],
    model: str | None = None,
) -> str | None:
    exact = []
    fallback = []
    for path in checkpoint_paths:
        name = Path(path).name.lower()
        if "latest" in name:
            fallback.append(path)
            continue
        if model == "nvdinov2" and name.startswith("student_epoch_"):
            exact.insert(0, path)
            continue
        if re.search(r"(?:^|[_-])(epoch|step)[_-]?\d+", name) or re.search(r"/(?:epoch|step)_\d+", path):
            exact.append(path)
        else:
            fallback.append(path)
    return (exact or fallback or [None])[0]


def _host_to_container_path(host_path: str, host_root: Path, container_root: str = "/results") -> str:
    path = Path(host_path)
    return f"{container_root.rstrip('/')}/{path.relative_to(host_root).as_posix()}"


def _latest_kpi(job_root: Path) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    for status_path in sorted(job_root.rglob("status.json")):
        try:
            lines = status_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            kpi = payload.get("kpi")
            if isinstance(kpi, dict):
                latest.update(kpi)
    return latest


def _metric_from_kpi(kpi: dict[str, Any], metric_name: str) -> float | None:
    for alias in _metric_aliases(metric_name):
        if alias not in kpi:
            continue
        try:
            return float(kpi[alias])
        except (TypeError, ValueError):
            continue
    return None


def _close_enough(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= max(1e-6, 1e-4 * max(abs(a), abs(b), 1.0))


def _metric_sources_close_enough(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= max(1e-3, 1e-3 * max(abs(a), abs(b), 1.0))


def _resume_behavior(jobs: dict[int, dict[str, Any]]) -> dict[str, Any]:
    resumed: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for rec_id, data in jobs.items():
        if not (data.get("resume_from_job_id") or data.get("resume_checkpoint_path")):
            continue
        checkpoint_path = data.get("resume_checkpoint_path")
        name = Path(checkpoint_path or "").name.lower()
        specific = bool(checkpoint_path) and "latest" not in name and (
            re.search(r"(?:^|[_-])(epoch|step)[_-]?\d+", name)
            or re.search(r"/(?:epoch|step)_\d+", checkpoint_path or "")
        )
        item = {
            "rec_id": rec_id,
            "resume_from_job_id": data.get("resume_from_job_id"),
            "resume_from_epoch": data.get("resume_from_epoch"),
            "resume_from_step": data.get("resume_from_step"),
            "resume_checkpoint_path": checkpoint_path,
            "uses_epoch_or_step_checkpoint": bool(specific),
        }
        resumed.append(item)
        if not specific:
            invalid.append(item)
    return {
        "status": "passed" if not invalid else "failed",
        "resume_recommendations": resumed,
        "invalid_resume_checkpoints": invalid,
    }


def _best_selection(
    jobs: dict[int, dict[str, Any]],
    best_rec_id: int | None,
    direction: str,
) -> dict[str, Any]:
    metrics = {
        rec_id: data.get("metric")
        for rec_id, data in jobs.items()
        if data.get("status") == "success" and data.get("metric") is not None
    }
    if not metrics:
        return {
            "status": "failed",
            "reason": "no successful job metrics available",
            "actual_best_rec_id": best_rec_id,
        }
    resumed_metrics = {
        rec_id: metric
        for rec_id, metric in metrics.items()
        if jobs.get(rec_id, {}).get("resume_from_job_id")
    }
    comparable_metrics = resumed_metrics or metrics
    expected = (
        min(comparable_metrics, key=comparable_metrics.get)
        if direction == "minimize"
        else max(comparable_metrics, key=comparable_metrics.get)
    )
    return {
        "status": "passed" if expected == best_rec_id else "failed",
        "direction": direction,
        "expected_best_rec_id": expected,
        "actual_best_rec_id": best_rec_id,
        "metrics_by_rec": comparable_metrics,
        "all_metrics_by_rec": metrics,
        "selection_scope": "resumed_final_rung" if resumed_metrics else "all_successful_jobs",
    }


def _minimal_action_overrides(
    specs: dict[str, Any],
    schema_keys: set[str],
    action: str,
    num_classes: int | None,
) -> dict[str, Any]:
    keys = schema_keys | _flatten_keys(specs)
    candidates: dict[str, Any] = {
        "dataset.batch_size": 1,
        "dataset.workers": 0,
        "dataset.num_workers": 0,
        f"{action}.num_gpus": 1,
        f"{action}.gpu_ids": [0],
        f"{action}.batch_size": 1,
        f"{action}.vis_after_n_batches": 1,
    }
    if num_classes:
        candidates.update({
            "dataset.num_classes": num_classes,
            "model.num_classes": num_classes,
            "model.sem_seg_head.num_classes": num_classes,
            "num_classes": num_classes,
        })
    return {key: value for key, value in candidates.items() if key in keys}


def _build_action_specs(
    model_dir: Path,
    skill_text: str,
    profile: ModelProfile,
    action: str,
    checkpoint_container_path: str,
    num_classes: int | None,
    trial_specs: dict[str, Any] | None = None,
    extra_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile_key = _profile_key_from_model_dir(model_dir)
    specs = _read_yaml(model_dir / "references" / f"spec_template_{action}.yaml")
    schema_keys = _schema_keys(model_dir, action)
    overrides: dict[str, Any] = {}
    _add_data_source_overrides(overrides, profile, _parse_action_rows(skill_text, action))
    overrides.update(_profile_specific_data_overrides(profile_key, profile))
    overrides.update(_minimal_action_overrides(specs, schema_keys, action, num_classes))
    overrides.update(_action_specific_overrides(profile_key, profile, action, Path()))
    if profile.model_type and "model.model_type" in (schema_keys | _flatten_keys(specs)):
        overrides["model.model_type"] = profile.model_type
    if profile.dataset_name and "dataset.dataset_name" in (schema_keys | _flatten_keys(specs)):
        overrides["dataset.dataset_name"] = profile.dataset_name
    if profile_key == "action-recognition" and "dataset.label_map" in (schema_keys | _flatten_keys(specs)):
        overrides["dataset.label_map"] = {"catch": 0, "smile": 1}
    if profile_key == "mask-grounding-dino":
        overrides.update({
            "dataset.val_data_sources.data_type": "OD",
            "dataset.test_data_sources.data_type": "OD",
            "dataset.infer_data_sources.data_type": "OD",
        })
    if profile_key == "mask2former":
        overrides.update({
            "dataset.train.type": "coco_panoptic",
            "dataset.val.type": "coco_panoptic",
            "dataset.test.type": "coco_panoptic",
            "dataset.contiguous_id": False,
        })
    if profile_key == "pose-classification" and action == "inference":
        overrides["inference.output_file"] = "/results/pose_classification_inference.txt"
    if profile_key == "re-identification":
        if action == "evaluate":
            overrides["evaluate.output_cmc_curve_plot"] = "/results/reid_cmc_curve.png"
            overrides["evaluate.output_sampled_matches_plot"] = "/results/reid_sampled_matches.png"
        if action == "inference":
            overrides["inference.output_file"] = "/results/reid_inference.json"
    if profile_key == "visual-changenet":
        overrides["model.backbone.pretrained_backbone_path"] = VISUAL_CHANGENET_BACKBONE_CONTAINER_PATH
    if profile_key == "nvdinov2":
        overrides.update({
            "wandb.enable": False,
            "model.backbone.teacher_type": "vit_s",
            "model.backbone.student_type": "vit_s",
            "model.backbone.img_size": 224,
            "train.num_prototypes": 1024,
            "train.precision": "32-true",
            "train.use_custom_attention": False,
        })
    if trial_specs:
        overrides.update(_valid_set(trial_specs, specs, schema_keys))
    if extra_overrides:
        overrides.update(_valid_set(extra_overrides, specs, schema_keys))
    if profile_key == "mae" and action in {"evaluate", "inference"}:
        overrides["train.stage"] = "finetune"
    for key in (f"{action}.checkpoint", f"{action}.model_path", f"{action}.pretrained_model_path"):
        if key in (schema_keys | _flatten_keys(specs)):
            overrides[key] = checkpoint_container_path
            break
    overrides = _valid_set(overrides, specs, schema_keys)
    if profile_key == "mask-grounding-dino" and action == "inference":
        overrides["dataset.infer_data_sources.captions"] = list(profile.captions or ("object",))
    for dotted_key, value in overrides.items():
        _set_nested(specs, dotted_key, value)
    return specs


def _resolve_action_image(skill_info: dict[str, Any], action_cfg: dict[str, Any]) -> str:
    from tao_sdk.versions import resolve_container_image

    return resolve_container_image(
        action_cfg.get("container_image") or skill_info.get("container_image", "")
    )


def _run_action_job(
    *,
    sdk: DockerSDK,
    image: str,
    action_cfg: dict[str, Any],
    specs: dict[str, Any],
    action: str,
    out_dir: Path,
    args: argparse.Namespace,
    mounts: list[dict[str, str]],
    env_vars: dict[str, str] | None = None,
    gpu_count: int | None = None,
) -> dict[str, Any]:
    outputs = action_cfg.get("outputs")
    if (
        action == "inference"
        and "mal inference" in action_cfg.get("command", "")
        and isinstance(outputs, dict)
    ):
        outputs = {
            key: value
            for key, value in outputs.items()
            if key != "inference.label_dump_path"
        }
    ep = build_entrypoint(
        command=action_cfg["command"],
        specs=specs,
        inputs=action_cfg.get("inputs"),
        outputs=outputs,
        config_format=action_cfg.get("config_format", "toml"),
        upload_excludes=action_cfg.get("upload_excludes", []),
    )
    job = sdk.create_job(
        image=image,
        command=ep["command"],
        gpu_count=args.num_gpus if gpu_count is None else gpu_count,
        env_vars=env_vars,
        mounts=mounts,
    )
    while True:
        time.sleep(args.poll_interval)
        status = sdk.get_job_status(job.id)
        LOG.info("%s job %s status=%s", action, job.id, status.status)
        if status.status in {"Complete", "Error", "Canceled"}:
            break
    logs = sdk.get_job_logs(job.id)
    job_root = out_dir / "results" / job.id
    success = status.status == "Complete" and "Execution status: FAIL" not in logs
    return {
        "action": action,
        "status": "success" if success else "failed",
        "job_id": job.id,
        "docker_status": status.status,
        "checkpoint_spec": _get_nested(specs, f"{action}.checkpoint"),
        "kpi": _latest_kpi(job_root),
        "status_files": [str(path) for path in sorted(job_root.rglob("status.json"))],
    }


def _build_dataset_convert_specs(
    *,
    model_dir: Path,
    skill_text: str,
    profile: ModelProfile,
) -> dict[str, Any]:
    profile_key = _profile_key_from_model_dir(model_dir)
    specs = _read_yaml(model_dir / "references" / "spec_template_dataset_convert.yaml")
    schema_keys = _schema_keys(model_dir, "dataset_convert")
    overrides: dict[str, Any] = {}
    _add_data_source_overrides(overrides, profile, _parse_action_rows(skill_text, "dataset_convert"))
    if profile_key == "bevfusion":
        overrides.update({
            "root_dir": BEVFUSION_CONTAINER_DATA_ROOT,
            "results_dir": BEVFUSION_CONTAINER_DATA_ROOT,
            "mode": "training",
        })
    if profile_key == "sparse4d":
        overrides.update({
            "aicity.num_frames": 3,
            "aicity.anchor_init_config.num_anchor": 72,
        })
    overrides = _valid_set(overrides, specs, schema_keys)
    for dotted_key, value in overrides.items():
        _set_nested(specs, dotted_key, value)
    return specs


def _build_ocrnet_dataset_convert_specs(
    *,
    model_dir: Path,
    image_uri: str,
    gt_uri: str,
) -> dict[str, Any]:
    specs = _read_yaml(model_dir / "references" / "spec_template_dataset_convert.yaml")
    schema_keys = _schema_keys(model_dir, "dataset_convert")
    overrides = _valid_set(
        {
            "dataset_convert.input_img_dir": image_uri,
            "dataset_convert.gt_file": gt_uri,
        },
        specs,
        schema_keys,
    )
    for dotted_key, value in overrides.items():
        _set_nested(specs, dotted_key, value)
    return specs


def _find_lmdb_root(root: Path) -> Path | None:
    if (root / "data.mdb").is_file() and (root / "lock.mdb").exists():
        return root
    for data_file in sorted(root.rglob("data.mdb")):
        candidate = data_file.parent
        if (candidate / "lock.mdb").exists():
            return candidate
    return None


def _normalize_sparse4d_depth_paths(
    *,
    model_dir: Path,
    out_dir: Path,
    convert_root: Path,
) -> dict[str, Any] | None:
    script = model_dir / "scripts" / "normalize_depth_paths.py"
    train_root = out_dir / "aicity_root" / "train"
    train_ann_dir = convert_root / "train"
    if not script.exists() or not train_root.exists() or not train_ann_dir.exists():
        return None
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--data-root",
            str(train_root),
            str(train_ann_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "returncode": result.returncode,
        "stdout_tail": result.stdout.splitlines()[-10:],
        "stderr_tail": result.stderr.splitlines()[-10:],
    }


def _run_dataset_convert_preflight(
    *,
    args: argparse.Namespace,
    model_dir: Path,
    skill_text: str,
    skill_info: dict[str, Any],
    profile: ModelProfile,
    out_dir: Path,
    sdk: DockerSDK,
    mounts: list[dict[str, str]],
) -> dict[str, Any]:
    profile_key = _profile_key_from_model_dir(model_dir)
    actions = skill_info.get("actions") or {}
    action_cfg = actions.get("dataset_convert")
    template = model_dir / "references" / "spec_template_dataset_convert.yaml"
    if not action_cfg or not template.exists():
        return {
            "status": "failed",
            "reason": "dataset_convert action/template is not packaged by skill",
        }

    image = _resolve_action_image(skill_info, action_cfg)

    if profile_key == "ocrnet":
        split_inputs = {
            "train": {
                "image_uri": _join_uri(profile.train_uri, "train.tar.gz"),
                "gt_uri": _join_uri(profile.train_uri, "train/gt_new.txt"),
            },
            "eval": {
                "image_uri": _join_uri(profile.eval_uri, "test.tar.gz"),
                "gt_uri": _join_uri(profile.eval_uri, "test/gt_new.txt"),
            },
        }
        jobs: dict[str, dict[str, Any]] = {}
        artifacts: dict[str, str] = {}
        missing: list[str] = []
        for split, split_data in split_inputs.items():
            specs = _build_ocrnet_dataset_convert_specs(
                model_dir=model_dir,
                image_uri=split_data["image_uri"],
                gt_uri=split_data["gt_uri"],
            )
            job_result = _run_action_job(
                sdk=sdk,
                image=image,
                action_cfg=action_cfg,
                specs=specs,
                action="dataset_convert",
                out_dir=out_dir,
                args=args,
                mounts=mounts,
                gpu_count=0,
            )
            jobs[split] = {
                "job": job_result,
                "specs": specs,
                "input_image_uri": split_data["image_uri"],
                "gt_uri": split_data["gt_uri"],
            }
            if job_result["status"] != "success":
                return {
                    "status": "failed",
                    "reason": f"ocrnet {split} dataset_convert job failed",
                    "jobs": jobs,
                }
            convert_root = out_dir / "results" / job_result["job_id"] / "results_dir"
            lmdb_root = _find_lmdb_root(convert_root)
            if lmdb_root is None:
                missing.append(f"{split}: data.mdb/lock.mdb under {convert_root}")
                continue
            artifacts[f"{split}_lmdb"] = str(lmdb_root)
            jobs[split]["convert_root"] = str(convert_root)
            jobs[split]["lmdb_root"] = str(lmdb_root)

        if missing:
            return {
                "status": "failed",
                "reason": "dataset_convert completed but required converted artifacts are missing",
                "missing_artifacts": missing,
                "artifacts": artifacts,
                "jobs": jobs,
            }

        train_lmdb = _host_to_container_path(artifacts["train_lmdb"], out_dir / "results")
        eval_lmdb = _host_to_container_path(artifacts["eval_lmdb"], out_dir / "results")
        return {
            "status": "passed",
            "jobs": jobs,
            "artifacts": artifacts,
            "train_overrides": {
                "dataset.train_dataset_dir": [train_lmdb],
                "dataset.val_dataset_dir": eval_lmdb,
                "dataset.train_gt_file": "",
                "dataset.val_gt_file": "",
                "dataset.character_list_file": _join_uri(profile.eval_uri, "character_list"),
            },
            "specs": {split: data["specs"] for split, data in jobs.items()},
        }

    specs = _build_dataset_convert_specs(
        model_dir=model_dir,
        skill_text=skill_text,
        profile=profile,
    )
    job_result = _run_action_job(
        sdk=sdk,
        image=image,
        action_cfg=action_cfg,
        specs=specs,
        action="dataset_convert",
        out_dir=out_dir,
        args=args,
        mounts=mounts,
        gpu_count=args.num_gpus if profile_key in {"pointpillars", "sparse4d"} else 0,
    )
    if job_result["status"] != "success":
        return {
            "status": "failed",
            "reason": "dataset_convert job failed",
            "job": job_result,
            "specs": specs,
        }

    convert_root = out_dir / "results" / job_result["job_id"] / "results_dir"
    train_overrides: dict[str, Any] = {}
    artifacts: dict[str, str] = {}
    missing: list[str] = []
    extra: dict[str, Any] = {}

    if profile_key == "pointpillars":
        data_info = convert_root / "data_info"
        required = {
            "dbinfos_train": data_info / "dbinfos_train.pkl",
            "infos_train": data_info / "infos_train.pkl",
            "infos_val": data_info / "infos_val.pkl",
        }
        for name, path in required.items():
            if path.exists():
                artifacts[name] = str(path)
            else:
                missing.append(str(path))
        if not missing:
            train_overrides["dataset.data_info_path"] = _host_to_container_path(
                str(data_info),
                out_dir / "results",
            )

    elif profile_key == "bevfusion":
        bev_root = _prepare_bevfusion_data_mount(profile, out_dir)
        data_prefix = {"pts": "training/velodyne_reduced", "img": "training/image_2"}
        required_bev = {
            "train_ann": bev_root / "kitti_person_infos_train.pkl",
            "val_ann": bev_root / "kitti_person_infos_val.pkl",
            "velodyne_reduced": bev_root / "training" / "velodyne_reduced",
        }
        for name, path in required_bev.items():
            if path.exists():
                artifacts[name] = str(path)
            else:
                missing.append(str(path))
        if not missing:
            train_overrides.update({
                "dataset.root_dir": BEVFUSION_CONTAINER_DATA_ROOT,
                "dataset.train_dataset.ann_file": (
                    f"{BEVFUSION_CONTAINER_DATA_ROOT}/kitti_person_infos_train.pkl"
                ),
                "dataset.train_dataset.data_prefix": data_prefix,
                "dataset.train_dataset.batch_size": 1,
                "dataset.train_dataset.num_workers": 0,
                "dataset.val_dataset.ann_file": (
                    f"{BEVFUSION_CONTAINER_DATA_ROOT}/kitti_person_infos_val.pkl"
                ),
                "dataset.val_dataset.data_prefix": data_prefix,
                "dataset.val_dataset.batch_size": 1,
                "dataset.val_dataset.num_workers": 0,
                "dataset.test_dataset.ann_file": (
                    f"{BEVFUSION_CONTAINER_DATA_ROOT}/kitti_person_infos_val.pkl"
                ),
                "dataset.test_dataset.data_prefix": data_prefix,
                "dataset.test_dataset.batch_size": 1,
                "dataset.test_dataset.num_workers": 0,
            })

    elif profile_key == "sparse4d":
        extra["depth_path_normalization"] = _normalize_sparse4d_depth_paths(
            model_dir=model_dir,
            out_dir=out_dir,
            convert_root=convert_root,
        )
        anchor = next(iter(sorted(convert_root.rglob("anchor_init.npy"))), None)
        train_ann = sorted(convert_root.rglob("*_infos_train.pkl"))
        val_ann = sorted(convert_root.rglob("*_infos_val.pkl"))
        test_ann = sorted(convert_root.rglob("*_infos_test.pkl"))
        train_ann_path = train_ann[0] if train_ann else None
        val_ann_path = val_ann[0] if val_ann else None
        test_ann_path = test_ann[0] if test_ann else None
        if (val_ann_path is None or test_ann_path is None) and len(train_ann) >= 3:
            train_ann_path, val_ann_path, test_ann_path = train_ann[:3]
            extra["train_split_pkls_used_for_smoke_val_test"] = [
                str(path) for path in (train_ann_path, val_ann_path, test_ann_path)
            ]
        required_sparse = {
            "anchor": anchor,
            "train_ann": train_ann_path,
            "val_ann": val_ann_path,
            "test_ann": test_ann_path,
        }
        for name, path in required_sparse.items():
            if path and path.exists():
                artifacts[name] = str(path)
            else:
                missing.append(name)
        if not missing:
            train_overrides.update({
                "dataset.data_root": "/data/aicity_root/train",
                "model.head.instance_bank.anchor": _host_to_container_path(
                    str(required_sparse["anchor"]),
                    out_dir / "results",
                ),
                "dataset.train_dataset.ann_file": _host_to_container_path(
                    str(required_sparse["train_ann"]),
                    out_dir / "results",
                ),
                "dataset.val_dataset.ann_file": _host_to_container_path(
                    str(required_sparse["val_ann"]),
                    out_dir / "results",
                ),
                "dataset.test_dataset.ann_file": _host_to_container_path(
                    str(required_sparse["test_ann"]),
                    out_dir / "results",
                ),
                "dataset.num_frames": 3,
                "dataset.sequences.split_num": 1,
                "dataset.train_dataset.sequences_split_num": 1,
                "model.head.instance_bank.num_anchor": 72,
                "model.head.instance_bank.num_temp_instances": 48,
                "model.head.num_output": 72,
                "train.precision": "fp32",
            })

    if missing:
        return {
            "status": "failed",
            "reason": "dataset_convert completed but required converted artifacts are missing",
            "missing_artifacts": missing,
            "artifacts": artifacts,
            "job": job_result,
            "specs": specs,
            **extra,
        }
    return {
        "status": "passed",
        "job": job_result,
        "convert_root": str(convert_root),
        "artifacts": artifacts,
        "train_overrides": train_overrides,
        "specs": specs,
        **extra,
    }


def _run_post_checks(
    *,
    args: argparse.Namespace,
    model_dir: Path,
    skill_text: str,
    skill_info: dict[str, Any],
    profile: ModelProfile,
    out_dir: Path,
    payload: dict[str, Any],
    sdk: DockerSDK,
    num_classes: int | None,
) -> dict[str, Any]:
    profile_key = _profile_key_from_model_dir(model_dir)
    checkpoints = payload.get("best_checkpoint_paths") or []
    checkpoint_path = _prefer_epoch_or_step_checkpoint(checkpoints, model=profile_key)
    if not checkpoint_path:
        payload["checkpoint_validation"] = {
            "status": "failed",
            "reason": "no real checkpoint path found for best recommendation",
        }
        return payload

    host_root = out_dir / "results"
    checkpoint_container_path = _host_to_container_path(checkpoint_path, host_root)
    actions = skill_info.get("actions") or {}
    post_checks = []
    best_trial_specs = ((payload.get("result") or {}).get("best") or {}).get("specs") or {}
    dataset_convert_overrides = (
        (payload.get("dataset_convert") or {}).get("train_overrides") or {}
    )
    mounts = _mounts_for_model(out_dir, profile_key, profile)
    action_env_vars = (
        {"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"}
        if profile_key in {"clip", "ml-recog", "oneformer", "re-identification"}
        else None
    )
    for action in ("evaluate", "inference"):
        action_cfg = actions.get(action)
        template = model_dir / "references" / f"spec_template_{action}.yaml"
        if not action_cfg or not template.exists():
            post_checks.append({
                "action": action,
                "status": "skipped",
                "reason": "action/template not packaged by skill",
            })
            continue
        image = _resolve_action_image(skill_info, action_cfg)
        specs = _build_action_specs(
            model_dir=model_dir,
            skill_text=skill_text,
            profile=profile,
            action=action,
            checkpoint_container_path=checkpoint_container_path,
            num_classes=num_classes,
            trial_specs=best_trial_specs,
            extra_overrides=dataset_convert_overrides,
        )
        post_checks.append(_run_action_job(
            sdk=sdk,
            image=image,
            action_cfg=action_cfg,
            specs=specs,
            action=action,
            out_dir=out_dir,
            args=args,
            mounts=mounts,
            env_vars=action_env_vars,
        ))

    payload["checkpoint_validation"] = {
        "status": (
            "success"
            if post_checks and all(item["status"] in {"success", "skipped"} for item in post_checks)
            else "failed"
        ),
        "checkpoint_path": checkpoint_path,
        "checkpoint_container_path": checkpoint_container_path,
        "uses_latest": "latest" in Path(checkpoint_path).name.lower(),
        "post_checks": post_checks,
    }
    train_passed = (
        bool(payload.get("jobs"))
        and all(data.get("status") == "success" for data in payload.get("jobs", {}).values())
        and ((payload.get("result") or {}).get("best") or {}).get("metric_value") is not None
        and bool(payload.get("best_checkpoint_paths"))
    )
    if train_passed and payload["checkpoint_validation"]["status"] == "success":
        payload["status"] = "passed"
    elif train_passed:
        payload["status"] = "failed"
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def _write_exception_report(args: argparse.Namespace, exc: Exception) -> Path:
    model = args.model
    network_arch = None
    profile_key = None
    profile = None
    metric_documented = None
    try:
        model_dir, network_arch, profile_key = _resolve_model_dir(args.skill_bank, args.model)
        model = model_dir.name
        profile = MODEL_PROFILES.get(profile_key)
        if profile:
            profile = _profile_with_dataset_overrides(profile, args)
        metric_documented = _monitoring_metric((model_dir / "SKILL.md").read_text())
    except Exception as resolve_exc:
        LOG.debug("Could not resolve full model metadata for exception report: %s", resolve_exc)

    report_path = args.run_root / model / "result.json"
    payload = {
        "model": model,
        "network_arch": network_arch,
        "profile_key": profile_key,
        "algorithm": args.algorithm,
        "status": "failed",
        "metric_documented": metric_documented,
        "metric_used_by_automl": None,
        "direction": None,
        "train_dataset_uri": profile.train_uri if profile else None,
        "eval_dataset_uri": profile.eval_uri if profile else None,
        "spec_overrides": {},
        "result": {"status": "failed", "error": str(exc)},
        "run_error": str(exc),
        "jobs": {},
        "algorithm_behavior": {
            "status": "not_started",
            "reason": "exception before AutoML launch",
        },
        "best_selection": {
            "status": "failed",
            "reason": "AutoML did not start",
            "actual_best_rec_id": None,
        },
        "metric_checks_passed": False,
        "best_checkpoint_paths": [],
        "dataset_convert": None,
        "resume_behavior": {
            "status": "passed",
            "resume_recommendations": [],
            "invalid_resume_checkpoints": [],
        },
    }
    _write_json(report_path, payload)
    return report_path


def _supported_automl_parameters(skill_bank: Path, model: str) -> list[str] | None:
    schema_path = skill_bank / "models" / model / "schemas" / "train.schema.json"

    def schema_defaults() -> list[str] | None:
        if not schema_path.exists():
            return None
        try:
            params = json.loads(schema_path.read_text()).get("automl_default_parameters")
        except json.JSONDecodeError:
            return None
        return params if isinstance(params, list) else None

    support_path = skill_bank / "models" / "automl_support.json"
    if not support_path.exists():
        return schema_defaults()
    try:
        support = json.loads(support_path.read_text())
    except json.JSONDecodeError:
        return schema_defaults()
    for item in support.get("supported", []):
        if item.get("model") == model:
            params = item.get("automl_default_parameters", [])
            return params or schema_defaults() or []
    return schema_defaults()


def _profile_with_dataset_overrides(
    profile: ModelProfile,
    args: argparse.Namespace,
) -> ModelProfile:
    return replace(
        profile,
        train_uri=args.train_dataset_uri or profile.train_uri,
        eval_uri=args.eval_dataset_uri or profile.eval_uri,
    )


def _minimal_custom_ranges(
    params: list[str] | None,
    model: str | None = None,
) -> dict[str, dict[str, Any]] | None:
    ranges: dict[str, dict[str, Any]] = {}
    for param in params or []:
        lower = param.lower()
        if "epoch" in lower:
            ranges[param] = {"valid_min": 1, "valid_max": 1}
        elif "batch_size" in lower or lower.endswith("mini_batch") or ".mini_batch" in lower:
            ranges[param] = {"valid_min": 1, "valid_max": 2}
        elif model == "nvdinov2" and lower == "dataset.workers":
            ranges[param] = {"valid_min": 2, "valid_max": 2}
        elif lower.endswith("workers") or "num_workers" in lower:
            ranges[param] = {"valid_min": 0, "valid_max": 0}
        elif lower == "policy.lora.r":
            ranges[param] = {"valid_min": 2, "valid_max": 8}
        elif lower == "policy.lora.lora_alpha":
            ranges[param] = {"valid_min": 2, "valid_max": 16}
        elif lower == "policy.lora.lora_dropout":
            ranges[param] = {"valid_min": 0.0, "valid_max": 0.05}
        elif lower == "model.corr_radius":
            ranges[param] = {"valid_min": 4, "valid_max": 4}
        elif lower == "model.cv_group":
            ranges[param] = {"valid_min": 8, "valid_max": 8}
        elif lower == "model.volume_dim":
            ranges[param] = {"valid_min": 32, "valid_max": 32}
        elif lower.endswith("num_queries"):
            ranges[param] = {"valid_min": 20, "valid_max": 50}
        elif lower.endswith("num_select"):
            ranges[param] = {"valid_min": 1, "valid_max": 20}
        elif lower.endswith("enc_layers") or lower.endswith("dec_layers"):
            ranges[param] = {"valid_min": 1, "valid_max": 2}
        elif "random_crop" in lower:
            ranges[param] = {"valid_min": 128, "valid_max": 256}
        elif lower.endswith("hidden_dim"):
            ranges[param] = {"valid_min": 256, "valid_max": 256}
        elif lower.endswith("train_max_size") or lower.endswith("test_max_size"):
            ranges[param] = {"valid_min": 256, "valid_max": 256}
        elif lower.endswith("test_min_size"):
            ranges[param] = {"valid_min": 128, "valid_max": 128}
        elif lower in {"train.optim.lr", "train.lr"} or lower.endswith("_lr") or lower.endswith("learning_rate"):
            ranges[param] = {"valid_min": 0.00001, "valid_max": 0.001}
        elif lower in {"train.wd", "train.optim.weight_decay"} or lower.endswith("weight_decay"):
            ranges[param] = {"valid_min": 0.0, "valid_max": 0.0001}
        elif lower == "train.optim.lr_backbone":
            ranges[param] = {"valid_min": 0.000001, "valid_max": 0.0001}
        elif lower == "train.optim.lr_linear_proj_mult":
            ranges[param] = {"valid_min": 0.01, "valid_max": 0.1}
    return ranges or None


def _read_s3_json(uri: str) -> Any:
    result = subprocess.run(
        ["aws", "s3", "cp", uri, "-"],
        check=True,
        capture_output=True,
        text=True,
        env=_aws_subprocess_env(),
    )
    return json.loads(result.stdout)


def _aws_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    alias_map = {
        "ACCESS_KEY": ("AWS_ACCESS_KEY_ID",),
        "SECRET_KEY": ("AWS_SECRET_ACCESS_KEY",),
        "S3_ENDPOINT_URL": ("AWS_ENDPOINT_URL", "AWS_ENDPOINT_URL_S3"),
        "CLOUD_REGION": ("AWS_DEFAULT_REGION", "AWS_REGION"),
    }
    for source, targets in alias_map.items():
        value = env.get(source)
        if not value:
            continue
        for target in targets:
            if not env.get(target):
                env[target] = value
    env.setdefault("AWS_EC2_METADATA_DISABLED", "true")
    return env


def _download_s3_file(uri: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["aws", "s3", "cp", uri, str(destination)],
        check=True,
        env=_aws_subprocess_env(),
    )


def _safe_extractall(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    dest_root = destination.resolve()
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if target != dest_root and dest_root not in target.parents:
                raise RuntimeError(f"Refusing to extract unsafe tar member {member.name!r}")
        tar.extractall(destination)


def _archive_image_members(archive: Path) -> list[str]:
    suffixes = (".jpg", ".jpeg", ".png", ".bmp")
    with tarfile.open(archive) as tar:
        return [
            member.name
            for member in tar.getmembers()
            if member.isfile() and member.name.lower().endswith(suffixes)
        ]


def _extract_first_image_to_flat_dir(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    existing = next(
        (
            path
            for path in sorted(destination.iterdir())
            if path.is_file() and path.name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ),
        None,
    )
    if existing:
        return existing

    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if not member.isfile() or not member.name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            source = tar.extractfile(member)
            if source is None:
                continue
            target = destination / Path(member.name).name
            with source, target.open("wb") as out:
                shutil.copyfileobj(source, out)
            return target
    raise RuntimeError(f"No image file found in {archive}")


def _category_name_map(payload: Any) -> dict[int, str]:
    categories = payload.get("categories") if isinstance(payload, dict) else payload
    if not isinstance(categories, list):
        return {}
    names: dict[int, str] = {}
    for item in categories:
        if not isinstance(item, dict) or "id" not in item:
            continue
        try:
            names[int(item["id"])] = str(item.get("name") or f"class_{item['id']}")
        except (TypeError, ValueError):
            continue
    return names


def _prepare_clip_split(uri: str, out_dir: Path, split: str) -> Path:
    split_root = out_dir / "clip_fallback" / split
    image_dir = split_root / "images"
    captions_dir = split_root / "captions"
    archive = split_root / "images.tar.gz"
    _download_s3_file(_join_uri(uri, "images.tar.gz"), archive)
    if not any(image_dir.rglob("*")):
        _safe_extractall(archive, split_root)

    annotations = _read_s3_json(_join_uri(uri, "annotations.json"))
    try:
        label_map_payload = _read_s3_json(_join_uri(uri, "label_map.json"))
    except Exception:
        label_map_payload = annotations
    names_by_category = _category_name_map(label_map_payload) or _category_name_map(annotations)

    image_names_by_id: dict[int, str] = {}
    image_records = annotations.get("images") if isinstance(annotations, dict) else None
    if isinstance(image_records, list):
        for record in image_records:
            if not isinstance(record, dict) or "id" not in record or "file_name" not in record:
                continue
            try:
                image_names_by_id[int(record["id"])] = Path(str(record["file_name"])).name
            except (TypeError, ValueError):
                continue

    labels_by_image: dict[int, set[str]] = {}
    records = annotations.get("annotations") if isinstance(annotations, dict) else None
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, dict):
                continue
            try:
                image_id = int(record["image_id"])
                category_id = int(record["category_id"])
            except (KeyError, TypeError, ValueError):
                continue
            labels_by_image.setdefault(image_id, set()).add(
                names_by_category.get(category_id, f"class_{category_id}")
            )

    captions_dir.mkdir(parents=True, exist_ok=True)
    archive_members = [Path(name).name for name in _archive_image_members(archive)]
    image_names = sorted(set(image_names_by_id.values()) | set(archive_members))
    labels_by_name = {
        image_names_by_id[image_id]: labels
        for image_id, labels in labels_by_image.items()
        if image_id in image_names_by_id
    }
    for image_name in image_names:
        labels = sorted(labels_by_name.get(image_name) or [])
        caption = ", ".join(labels) if labels else "object"
        (captions_dir / f"{Path(image_name).stem}.txt").write_text(caption + "\n")

    prompts = out_dir / "clip_fallback" / "prompts.txt"
    if not prompts.exists():
        prompts.write_text("\n".join(sorted(set(names_by_category.values())) or ["object"]) + "\n")
    return split_root


def _prepare_clip_data_mount(profile: ModelProfile, out_dir: Path) -> Path:
    root = out_dir / "clip_fallback"
    _prepare_clip_split(profile.train_uri, out_dir, "train")
    _prepare_clip_split(profile.eval_uri or profile.train_uri, out_dir, "val")
    return root


def _prepare_bevfusion_data_mount(profile: ModelProfile, out_dir: Path) -> Path:
    data_root = out_dir / "bevfusion_root"
    data_root.mkdir(parents=True, exist_ok=True)
    for name in ("training.tar.gz", "testing.tar.gz", "ImageSets.tar.gz"):
        archive = data_root / name
        _download_s3_file(_join_uri(profile.train_uri, name), archive)
        marker = data_root / f".extracted_{name}"
        if not marker.exists():
            _safe_extractall(archive, data_root)
            marker.write_text("ok\n")
    return data_root


def _prepare_single_image_mount(
    *,
    uri: str,
    archive_suffix: str,
    out_dir: Path,
    name: str,
) -> Path:
    data_root = out_dir / name
    archive = data_root / Path(archive_suffix).name
    _download_s3_file(_join_uri(uri, archive_suffix), archive)
    _extract_first_image_to_flat_dir(archive, data_root)
    return data_root


def _prepare_depth_data_mount(profile: ModelProfile, out_dir: Path) -> Path:
    data_root = out_dir / "data_mount"
    for uri in (profile.train_uri, profile.eval_uri):
        dataset_name = uri.rstrip("/").rsplit("/", 1)[-1]
        target = data_root / dataset_name
        archive = target / "images.tar.gz"
        _download_s3_file(_join_uri(uri, "images.tar.gz"), archive)
        if not (target / "left").exists():
            _safe_extractall(archive, target)
    return data_root


def _prepare_visual_changenet_backbone(out_dir: Path) -> Path:
    destination = out_dir / "ptm" / "c-radio-v2-b" / "C-RADIOv2_B.safetensors"
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    from huggingface_hub import hf_hub_download

    destination.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(hf_hub_download(
        repo_id="nvidia/C-RADIOv2-B",
        filename="model.safetensors",
        token=os.environ.get("HF_TOKEN") or None,
    ))
    shutil.copy2(downloaded, destination)
    return destination


def _mounts_for_model(out_dir: Path, model: str, profile: ModelProfile) -> list[dict[str, str]]:
    mounts = [{"host_path": str(out_dir / "results"), "container_path": "/results"}]
    if model == "bevfusion":
        mounts.append({
            "host_path": str(_prepare_bevfusion_data_mount(profile, out_dir)),
            "container_path": BEVFUSION_CONTAINER_DATA_ROOT,
        })
    if model == "clip":
        mounts.append({
            "host_path": str(_prepare_clip_data_mount(profile, out_dir)),
            "container_path": CLIP_CONTAINER_DATA_ROOT,
        })
    if model in {"depth-net-mono", "depth-net-stereo"}:
        mounts.append({
            "host_path": str(_prepare_depth_data_mount(profile, out_dir)),
            "container_path": "/data",
        })
    if model == "grounding-dino":
        mounts.append({
            "host_path": str(_prepare_single_image_mount(
                uri=profile.eval_uri or profile.train_uri,
                archive_suffix="images.tar.gz",
                out_dir=out_dir,
                name="grounding_dino_infer",
            )),
            "container_path": GROUNDING_DINO_INFER_CONTAINER_ROOT,
        })
    if model == "nvpanoptix3d":
        mounts.append({
            "host_path": str(_prepare_single_image_mount(
                uri=profile.eval_uri or profile.train_uri,
                archive_suffix="data/images.tar.gz",
                out_dir=out_dir,
                name="nvpanoptix3d_infer",
            )),
            "container_path": NVPANOPTIX3D_INFER_CONTAINER_ROOT,
        })
    if model == "visual-changenet":
        mounts.append({
            "host_path": str(_prepare_visual_changenet_backbone(out_dir)),
            "container_path": VISUAL_CHANGENET_BACKBONE_CONTAINER_PATH,
        })
    if model == "sparse4d":
        data_root = out_dir / "aicity_root"
        data_root.mkdir(parents=True, exist_ok=True)
        mounts.append({
            "host_path": str(data_root),
            "container_path": "/data/aicity_root",
        })
    return mounts


def _sample_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)][:8]
    if isinstance(payload, dict):
        for key in ("annotations", "data", "records", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)][:8]
        if payload:
            return [payload]
    return []


def _cosmos_annotation_preflight(profile: ModelProfile) -> dict[str, Any]:
    checked = []
    for name, uri in {
        "train_annotation": _join_uri(profile.train_uri, "annotations.json"),
        "eval_annotation": _join_uri(profile.eval_uri, "annotations.json"),
    }.items():
        payload = _read_s3_json(uri)
        records = _sample_records(payload)
        checked.append({
            "name": name,
            "uri": uri,
            "sampled_records": len(records),
        })
    return {
        "status": "passed",
        "checked": checked,
    }


def run_model(args: argparse.Namespace) -> int:
    requested_model = args.model
    model_dir, network_arch, profile_key = _resolve_model_dir(args.skill_bank, requested_model)
    model = model_dir.name
    profile = MODEL_PROFILES.get(profile_key)
    if profile is None:
        raise KeyError(f"No validation profile for {model} (network_arch={network_arch})")
    profile = _profile_with_dataset_overrides(profile, args)

    skill_text = (model_dir / "SKILL.md").read_text()
    skill_info = _read_yaml(model_dir / "references" / "skill_info.yaml")
    train_specs = _read_yaml(model_dir / "references" / "spec_template_train.yaml")
    if profile.data_format:
        skill_info["data_format"] = profile.data_format
    data_preflight = None

    out_dir = args.run_root / model
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "result.json"

    effective_num_classes = max(
        [value for value in (args.num_classes, profile.num_classes) if value is not None],
        default=None,
    )

    supported_params = _supported_automl_parameters(args.skill_bank, model)
    if supported_params == [] and not args.post_check_only:
        payload = {
            "model": model,
            "algorithm": args.algorithm,
            "status": "blocked",
            "blocker": "skill is listed as AutoML-enabled but exposes no searchable AutoML train parameters",
        }
        _write_json(report_path, payload)
        print(json.dumps(payload))
        return 2

    if profile_key == "cosmos-rl" and not args.post_check_only:
        if not os.environ.get("HF_TOKEN"):
            payload = {
                "model": model,
                "network_arch": network_arch,
                "profile_key": profile_key,
                "algorithm": args.algorithm,
                "status": "blocked",
                "blocker": "HF_TOKEN is required for the gated Cosmos3-Nano model",
            }
            _write_json(report_path, payload)
            print(json.dumps(payload))
            return 2
        data_preflight = _cosmos_annotation_preflight(profile)

    if args.post_check_only:
        if not report_path.exists():
            raise FileNotFoundError(f"{report_path} does not exist")
        payload = json.loads(report_path.read_text())
        sdk = DockerSDK(
            poll_interval=args.poll_interval,
            state_file=str(out_dir / "sdk_state_post_checks.json"),
        )
        payload = _run_post_checks(
            args=args,
            model_dir=model_dir,
            skill_text=skill_text,
            skill_info=skill_info,
            profile=profile,
            out_dir=out_dir,
            payload=payload,
            sdk=sdk,
            num_classes=effective_num_classes,
        )
        _write_json(report_path, payload)
        print(json.dumps({"model": model, "status": payload["status"], "report": str(report_path)}))
        return 0 if payload["status"] == "passed" else 1

    if profile.blocked and not args.allow_known_blockers:
        payload = {
            "model": model,
            "network_arch": network_arch,
            "profile_key": profile_key,
            "algorithm": args.algorithm,
            "status": "blocked",
            "blocker": profile.blocked,
        }
        _write_json(report_path, payload)
        print(json.dumps(payload))
        return 2

    schema_keys = _schema_keys(model_dir, "train")
    rows = _parse_train_rows(skill_text)
    overrides: dict[str, Any] = {}
    _add_data_source_overrides(overrides, profile, rows)
    overrides.update(_profile_specific_data_overrides(profile_key, profile))
    overrides.update(_minimal_train_overrides(train_specs, schema_keys, effective_num_classes, profile_key))
    if profile_key == "mask-grounding-dino":
        overrides["dataset.val_data_sources.data_type"] = "OD"
    overrides = _valid_set(overrides, train_specs, schema_keys)

    metric = args.metric or _monitoring_metric(skill_text)
    if profile_key == "dino" and metric == "val_mAP50":
        metric = "mAP50"

    jobs: dict[int, dict[str, Any]] = {}

    def on_recommendation(rec) -> None:
        jobs.setdefault(rec.id, {})["specs"] = rec.specs
        LOG.info("model=%s rec=%s recommendation generated", model, rec.id)

    def on_result(rec, metric_value, status) -> None:
        jobs.setdefault(rec.id, {}).update({
            "job_id": getattr(rec, "job_id", None),
            "metric": metric_value,
            "status": status,
            "resume_from_job_id": getattr(rec, "resume_from_job_id", None),
            "resume_from_epoch": getattr(rec, "resume_from_epoch", None),
            "resume_from_step": getattr(rec, "resume_from_step", None),
            "resume_checkpoint_path": getattr(rec, "resume_checkpoint_path", None),
        })
        LOG.info("model=%s rec=%s status=%s metric=%s", model, rec.id, status, metric_value)

    sdk = DockerSDK(
        poll_interval=args.poll_interval,
        state_file=str(out_dir / "sdk_state.json"),
    )
    mounts = _mounts_for_model(out_dir, profile_key, profile)
    dataset_convert_preflight = None
    if profile_key in {"bevfusion", "ocrnet", "pointpillars", "sparse4d"}:
        dataset_convert_preflight = _run_dataset_convert_preflight(
            args=args,
            model_dir=model_dir,
            skill_text=skill_text,
            skill_info=skill_info,
            profile=profile,
            out_dir=out_dir,
            sdk=sdk,
            mounts=mounts,
        )
        if dataset_convert_preflight["status"] != "passed":
            payload = {
                "model": model,
                "network_arch": network_arch,
                "profile_key": profile_key,
                "algorithm": args.algorithm,
                "status": (
                    "blocked"
                    if "required converted artifacts are missing"
                    in dataset_convert_preflight.get("reason", "")
                    else "failed"
                ),
                "metric_documented": _monitoring_metric(skill_text),
                "metric_used_by_automl": metric,
                "train_dataset_uri": profile.train_uri,
                "eval_dataset_uri": profile.eval_uri,
                "dataset_convert": dataset_convert_preflight,
                "blocker": dataset_convert_preflight.get("reason"),
            }
            _write_json(report_path, payload)
            print(json.dumps(payload))
            return 2 if payload["status"] == "blocked" else 1
        overrides.update(dataset_convert_preflight.get("train_overrides") or {})
        overrides = _valid_set(overrides, train_specs, schema_keys)
    runner = AutoMLRunner(
        sdk=sdk,
        skill_dir=model_dir,
        action="train",
        poll_interval=args.poll_interval,
    )
    run_error = None
    try:
        result = runner.run(
            train_dataset_uri=profile.train_uri,
            eval_dataset_uri=profile.eval_uri,
            automl_settings=_automl_settings(args.algorithm, metric, args),
            automl_hyperparameters=supported_params or None,
            custom_param_ranges=_minimal_custom_ranges(supported_params, model=profile_key),
            workspace_path=str(out_dir / "workspace"),
            spec_overrides=overrides,
            metric_extractor=_metric_extractor_for(profile_key),
            on_recommendation=on_recommendation,
            on_result=on_result,
            gpu_count=args.num_gpus,
            mounts=mounts,
        )
    except Exception as exc:
        run_error = str(exc)
        LOG.exception("AutoML run failed for model=%s algorithm=%s", model, args.algorithm)
        result = {
            "best": None,
            "progress": {},
            "history": [],
            "error": run_error,
        }

    for rec_id, data in jobs.items():
        job_id = data.get("job_id")
        job_root = out_dir / "results" / str(job_id) if job_id else Path("")
        data["checkpoint_paths"] = _find_checkpoints(job_root, profile_key) if job_id else []
        data["checkpoint_count"] = len(data["checkpoint_paths"])
        train_kpi = _latest_kpi(job_root) if job_id else {}
        data["train_kpi"] = train_kpi
        log_metric = None
        if job_id:
            try:
                log_metric = _extract_metric_from_logs(sdk.get_job_logs(job_id), metric)
            except Exception:
                log_metric = None
        status_metric = _metric_from_kpi(train_kpi, metric)
        emitted_metric = status_metric if status_metric is not None else log_metric
        status_log_metric_match = None
        if status_metric is not None and log_metric is not None:
            status_log_metric_match = _metric_sources_close_enough(status_metric, log_metric)
        data["metric_verification"] = {
            "metric_used_by_automl": metric,
            "automl_reported_metric": data.get("metric"),
            "status_metric": status_metric,
            "log_metric": log_metric,
            "status_log_metric_match": status_log_metric_match,
            "emitted_metric": emitted_metric,
            "emitted_kpi_keys": sorted(train_kpi.keys()),
            "matches_emitted_metric": _close_enough(data.get("metric"), emitted_metric),
        }
        if (
            profile_key == "sparse4d"
            and not data["metric_verification"]["matches_emitted_metric"]
            and data.get("resume_from_job_id")
        ):
            parent_job_id = str(data["resume_from_job_id"])
            parent_root = out_dir / "results" / parent_job_id
            parent_kpi = _latest_kpi(parent_root)
            parent_log_metric = None
            try:
                parent_log_metric = _extract_metric_from_logs(sdk.get_job_logs(parent_job_id), metric)
            except Exception:
                parent_log_metric = None
            parent_status_metric = _metric_from_kpi(parent_kpi, metric)
            parent_emitted_metric = (
                parent_status_metric if parent_status_metric is not None else parent_log_metric
            )
            parent_match = _close_enough(data.get("metric"), parent_emitted_metric)
            parent_status_log_metric_match = None
            if parent_status_metric is not None and parent_log_metric is not None:
                parent_status_log_metric_match = _metric_sources_close_enough(
                    parent_status_metric,
                    parent_log_metric,
                )
            data["metric_verification"].update({
                "metric_source": "resume_parent" if parent_match else "resume_parent_unmatched",
                "parent_job_id": parent_job_id,
                "parent_status_metric": parent_status_metric,
                "parent_log_metric": parent_log_metric,
                "parent_status_log_metric_match": parent_status_log_metric_match,
                "parent_emitted_metric": parent_emitted_metric,
                "parent_emitted_kpi_keys": sorted(parent_kpi.keys()),
                "matches_emitted_metric": parent_match,
            })

    best = result.get("best") or {}
    best_rec_id = best.get("rec_id")
    best_job = jobs.get(best_rec_id, {})
    ready_for_post_checks = (
        bool(jobs)
        and all(data.get("status") == "success" for data in jobs.values())
        and best.get("metric_value") is not None
        and bool(best_job.get("checkpoint_paths"))
    )
    selection = _best_selection(jobs, best_rec_id, _direction(metric))
    resume_behavior = _resume_behavior(jobs)
    successful_jobs = [data for data in jobs.values() if data.get("status") == "success"]
    def _metric_verification_passed(data: Dict[str, Any]) -> bool:
        verification = data.get("metric_verification", {})
        if not verification.get("matches_emitted_metric"):
            return False
        # Some TAO logs round metrics more aggressively than status artifacts. When
        # AutoML matches the emitted status metric, keep the rounded log mismatch as
        # report evidence without failing the model.
        if (
            verification.get("status_log_metric_match") is False
            and verification.get("status_metric") is None
        ):
            return False
        if (
            verification.get("parent_status_log_metric_match") is False
            and verification.get("parent_status_metric") is None
        ):
            return False
        return True

    metric_checks_passed = bool(successful_jobs) and all(
        _metric_verification_passed(data) for data in successful_jobs
    )
    algorithm_behavior = {
        "recommendation_count": len(jobs),
        "successful_recommendations": sum(1 for data in jobs.values() if data.get("status") == "success"),
        "unique_recommendations": len({
            json.dumps(data.get("specs") or {}, sort_keys=True, default=str)
            for data in jobs.values()
        }),
        "metrics_reported": sum(1 for data in jobs.values() if data.get("metric") is not None),
        "resume_recommendations": sum(1 for data in jobs.values() if data.get("resume_from_job_id")),
    }
    passed = (
        ready_for_post_checks
        and selection["status"] == "passed"
        and metric_checks_passed
        and resume_behavior["status"] == "passed"
    )
    payload = {
        "model": model,
        "network_arch": network_arch,
        "profile_key": profile_key,
        "algorithm": args.algorithm,
        "status": "passed" if passed else "failed",
        "metric_documented": _monitoring_metric(skill_text),
        "metric_used_by_automl": metric,
        "direction": _direction(metric),
        "train_dataset_uri": profile.train_uri,
        "eval_dataset_uri": profile.eval_uri,
        "spec_overrides": overrides,
        "result": result,
        "run_error": run_error,
        "jobs": jobs,
        "algorithm_behavior": algorithm_behavior,
        "best_selection": selection,
        "metric_checks_passed": metric_checks_passed,
        "best_checkpoint_paths": best_job.get("checkpoint_paths", []),
        "dataset_convert": dataset_convert_preflight,
        "data_preflight": data_preflight,
        "resume_behavior": resume_behavior,
    }
    if ready_for_post_checks:
        payload = _run_post_checks(
            args=args,
            model_dir=model_dir,
            skill_text=skill_text,
            skill_info=skill_info,
            profile=profile,
            out_dir=out_dir,
            payload=payload,
            sdk=sdk,
            num_classes=effective_num_classes,
        )
        if not passed and payload.get("status") == "passed":
            payload["status"] = "failed"
    _write_json(report_path, payload)
    print(json.dumps({"model": model, "status": payload["status"], "report": str(report_path)}))
    return 0 if payload["status"] == "passed" else 1


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--skill-bank", default="/localhome/local-rarunachalam/tao-skills-external", type=Path)
    parser.add_argument("--gpu-device-id", default="2")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--metric")
    parser.add_argument("--train-dataset-uri")
    parser.add_argument("--eval-dataset-uri")
    parser.add_argument("--allow-known-blockers", action="store_true")
    parser.add_argument("--post-check-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = parse_args(argv)
    try:
        return run_model(args)
    except Exception as exc:
        report_path = _write_exception_report(args, exc)
        print(json.dumps({"status": "failed", "error": str(exc), "report": str(report_path)}))
        LOG.exception("validation failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
