# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for AutoML validation harness model-specific fixes."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_validator():
    path = Path(__file__).resolve().parents[1] / "scripts" / "validate_skill_automl_model.py"
    spec = importlib.util.spec_from_file_location("validate_skill_automl_model", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cosmos_profile_uses_direct_s3_annotations_and_dataset_overrides():
    validator = _load_validator()
    profile = validator._profile_with_dataset_overrides(
        validator.MODEL_PROFILES["cosmos-rl"],
        SimpleNamespace(
            train_dataset_uri="s3://bucket/wts_train",
            eval_dataset_uri="s3://bucket/wts_eval",
        ),
    )

    overrides = validator._profile_specific_data_overrides("cosmos-rl", profile)

    assert profile.train_uri == "s3://bucket/wts_train"
    assert profile.eval_uri == "s3://bucket/wts_eval"
    assert overrides["custom.train_dataset.annotation_path"] == "s3://bucket/wts_train/annotations.json"
    assert overrides["custom.train_dataset.media_path"] == "s3://bucket/wts_train/videos.tar.gz"
    assert overrides["custom.val_dataset.annotation_path"] == "s3://bucket/wts_eval/annotations.json"
    assert overrides["custom.val_dataset.media_path"] == "s3://bucket/wts_eval/videos.tar.gz"


def test_clip_profile_uses_local_custom_dataset_and_clears_wds():
    validator = _load_validator()
    profile = validator.MODEL_PROFILES["clip"]

    overrides = validator._profile_specific_data_overrides("clip", profile)

    assert overrides["dataset.train.type"] == "custom"
    assert overrides["dataset.train.datasets"][0]["image_dir"] == "/data/clip_fallback/train/images"
    assert overrides["dataset.train.datasets"][0]["caption_dir"] == "/data/clip_fallback/train/captions"
    assert overrides["dataset.train.wds.root_dir"] is None
    assert overrides["dataset.train.wds.shard_list_file"] is None


def test_nvpanoptix3d_reuses_val_json_for_test_split_when_no_test_json_exists():
    validator = _load_validator()
    profile = validator.MODEL_PROFILES["nvpanoptix3d"]

    overrides = validator._profile_specific_data_overrides("nvpanoptix3d", profile)

    assert overrides["dataset.test.json_path"].endswith(
        "/purpose_built_models_nvpanoptix3d_val/meta/val.json"
    )
    assert overrides["dataset.test.base_dir"].endswith("/purpose_built_models_nvpanoptix3d_val")
    assert overrides["train.optim.monitor_name"] == "train_loss"


def test_grounding_dino_inference_uses_one_image_mount_and_real_prompts():
    validator = _load_validator()
    profile = validator.MODEL_PROFILES["grounding-dino"]

    overrides = validator._action_specific_overrides("grounding-dino", profile, "inference", Path())

    assert overrides["dataset.infer_data_sources.image_dir"] == ["/data/grounding_dino_infer"]
    assert overrides["dataset.infer_data_sources.captions"] == ["head", "helmet", "person"]


def test_clean_file_spec_treats_human_folder_descriptions_as_roots():
    validator = _load_validator()

    assert validator._clean_file_spec("root directory containing `.tar` shards") == ""
    assert validator._clean_file_spec("flat folder of `.jpg`/`.png` RGB images") == ""
