#!/usr/bin/env python3

"""Frozen OneFormer/full-COCO2017 objective-aware campaign policy.

The campaign explicitly selects OneFormer's panoptic evaluation task and uses
the task-correct, globally reduced ``PQ`` metric emitted by the pinned TAO
PyTorch source overlay.  The base SQSH remains immutable; every model command
must install the sealed overlay before importing TAO PyTorch.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tao_automl.ptm_registry import canonical_sha256, load_ptm_registry


MODES = ("accuracy", "latency", "multi_objective")
AGENT_FLAGS = (
    "agent_selected_candidate",
    "agent_injected_candidate",
    "agent_modified_search_space_after_results",
    "agent_changed_seed_after_results",
    "agent_changed_budget_after_results",
    "agent_changed_threshold_after_results",
    "agent_changed_ptm_after_results",
    "agent_overrode_winner",
)
SELECTION_FLAGS = (
    "selector_invoked_on_matched_measurements",
    "selection_time_objectives_replaced",
    "measurements_feed_selection",
    "measurements_feed_reselection",
    "algorithm_selected_candidate_overridden",
)

# PTM identity is represented as a hierarchical, nonordinal outer arm.  These
# inner parameters are all explicitly AutoML-enabled in the packaged OneFormer
# train schema.  Test resolution supplies a supported inference-cost variable;
# optimizer and training resize fields supply accuracy-recovery dimensions.
SEARCH_PARAMETERS = (
    "dataset.augmentation.test_min_size",
    "dataset.augmentation.train_max_size",
    "train.optim.lr",
    "train.optim.weight_decay",
    "train.optim.backbone_multiplier",
)
SEARCH_SPACE = {
    "dataset.augmentation.test_min_size": {
        "type": "integer",
        "minimum": 512,
        "maximum": 1024,
    },
    "dataset.augmentation.train_max_size": {
        "type": "integer",
        "minimum": 1024,
        "maximum": 1536,
    },
    "train.optim.lr": {
        "type": "float",
        "minimum": 5.0e-6,
        "maximum": 5.0e-5,
        "scale": "linear",
    },
    "train.optim.weight_decay": {
        "type": "float",
        "minimum": 0.01,
        "maximum": 0.10,
        "scale": "linear",
    },
    "train.optim.backbone_multiplier": {
        "type": "float",
        "minimum": 0.05,
        "maximum": 0.20,
        "scale": "linear",
    },
}

FROZEN_CANDIDATE_BUDGET = 20
FROZEN_TRAINING_EPOCHS = 1
FROZEN_SEARCH_SEED = 271828
FROZEN_TRAINING_SEED = 123
FROZEN_CALIBRATION_POINTS_PER_ARM = 2
FROZEN_INVALID_RECOVERY_ISSUES_PER_ARM = 1
FROZEN_LATENCY_RETENTION = 0.90
FROZEN_LATENCY_TOLERANCE_MS = 0.73553775
FROZEN_VALIDATION_SANITY_MIN_PQ = 0.01
FROZEN_SLURM_RETRY_CAP = 10
FROZEN_BATCH_SIZE_PER_REPLICA = 1
FROZEN_HARDWARE = {
    "gpu_name": "NVIDIA A100-SXM4-80GB",
    "compute_capability": "8.0",
    "total_memory_bytes": 85174583296,
}
FROZEN_SQSH = {
    "path": (
        "/lustre/fsw/portfolios/edgeai/users/rarunachalam/"
        "nvcr.io_nvstaging_tao_tao-toolkit-pyt_7.1.0-rc-245-multiarch.sqsh"
    ),
    "sha256": (
        "e36640f9ae7a03bc80828cf7de93bd6bdbbb0fecf509a71a243be0ab5b497fc2"
    ),
    "size_bytes": 28860358656,
    "image_reference": (
        "nvcr.io/nvstaging/tao/tao-toolkit-pyt:"
        "7.1.0-rc-245-multiarch"
    ),
}
FROZEN_RUNTIME_OVERLAY = {
    "artifact_type": "tao_pytorch_source_overlay",
    "scope": "oneformer_runtime_product_fixes",
    "archive_path": (
        "/lustre/fsw/portfolios/edgeai/users/rarunachalam/artifacts/"
        "oneformer-runtime-product-fixes-c25a20e0/"
        "oneformer-runtime-overlay.tar"
    ),
    "archive_sha256": (
        "6b976090fb264b319ba23e7092445f261fd1b445964400d3f879c2746247a4f3"
    ),
    "archive_size_bytes": 153600,
    "archive_root": "oneformer-runtime-overlay",
    "manifest_sha256": (
        "1ed2721226677e023d8a688f629fa85c997f1ce7f9889b00cda01fc5db899760"
    ),
    "installer_sha256": (
        "c0db61d777cbedc33ffeab795825f30924c7b56faa6996504181684096dfc030"
    ),
    "source_repository": "tao-pytorch",
    "source_commit": "c25a20e0d6e2cf98ccb80c16eb0d4d30bb40f600",
    "product_fix_commit": "e3ebf59a47d0aea365c855919a1de196f8a0432e",
    "base_commit": "99741bc8229617d0d3dd52e30540111d55efd1af",
    "base_site_packages": "/usr/local/lib/python3.12/dist-packages",
    "runtime_site_packages_strategy": "writable_tmp_symlink_tree",
    "runtime_site_packages_suffix": "/site-packages",
    "file_count": 19,
    "remediates_static_findings": [
        "oneformer_full_checkpoint_loader_missing",
        "oneformer_panoptic_pq_not_emitted",
        "oneformer_ddp_status_metric_not_globally_reduced",
    ],
    "evaluation_task": "panoptic",
    "primary_accuracy_metric": "PQ",
    "receipt_required_for_every_model_job": True,
}
LATENCY_PROTOCOL = {
    "warmup_iterations": 50,
    "timed_iterations": 100,
    "repeated_rounds": 5,
    "preloaded_batches": 16,
    "benchmark_seed": 20260727,
    "tail_percentile": 95.0,
    "bootstrap_resamples": 5000,
    "bootstrap_confidence_level": 0.95,
    "bootstrap_seed": 424242,
    "batch_size_per_replica": 1,
    "expected_replicas": 8,
    "precision": "fp32",
    "timed_scope": "oneformer_model_forward",
    "excluded_scope": [
        "checkpoint_load",
        "disk_io",
        "image_decode",
        "resize_normalize",
        "host_to_device_transfer",
        "semantic_or_panoptic_postprocessing",
        "metric_accumulation",
        "distributed_gather",
    ],
    "synchronization": "accelerator_sync_before_and_after_each_sample",
    "replica_alignment": "nccl_barrier_before_each_timed_sample",
    "measurement_role": "selection_time",
    "raw_samples_per_candidate": 4000,
    "validity_thresholds": {
        "max_robust_cv": 0.10,
        "max_round_median_range_fraction": 0.05,
        "max_absolute_round_drift_fraction": 0.05,
        "max_device_median_range_fraction": 0.05,
        "max_bootstrap_ci_width_fraction": 0.03,
    },
}


class CampaignContractError(ValueError):
    """The OneFormer campaign contract is inconsistent."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_fraction(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise CampaignContractError(f"{name} must be finite in (0, 1]")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CampaignContractError(
            f"{name} must be finite in (0, 1]"
        ) from exc
    if not math.isfinite(number) or not 0.0 < number <= 1.0:
        raise CampaignContractError(f"{name} must be finite in (0, 1]")
    return number


def oneformer_registry_snapshot() -> dict[str, Any]:
    """Snapshot every official repository-owned OneFormer PTM record."""
    registry = load_ptm_registry()
    model = registry.to_dict()["models"]["oneformer"]
    records = []
    for record in model["checkpoints"]:
        spec_file = record.get("checkpoint_spec_file", {})
        if (
            record.get("source", {}).get("official") is not True
            or record.get("model_family") != "oneformer"
            or "panoptic_segmentation"
            not in record.get("task_compatibility", ())
            or spec_file.get("source") != "repository"
            or not spec_file.get("path", "").startswith(
                "data/ptm_specs/oneformer/"
            )
        ):
            raise CampaignContractError(
                f"invalid official OneFormer registry record: {record.get('id')}"
            )
        records.append(
            {
                "id": record["id"],
                "status": record["status"],
                "status_reason": record.get("status_reason"),
                "source": copy.deepcopy(record["source"]),
                "sha256": record["sha256"],
                "expected_size_bytes": record["expected_size_bytes"],
                "compatible_tao_versions": copy.deepcopy(
                    record["compatible_tao_versions"]
                ),
                "checkpoint_target": record["checkpoint_target"],
                "architecture": record["architecture"],
                "backbone": record["backbone"],
                "input_contract": copy.deepcopy(record["input_contract"]),
                "default_spec_overrides": copy.deepcopy(
                    record["default_spec_overrides"]
                ),
                "checkpoint_spec_file": copy.deepcopy(spec_file),
                "registry_record_sha256": canonical_sha256(record),
            }
        )
    if len(records) != 4 or len({item["id"] for item in records}) != 4:
        raise CampaignContractError(
            "the frozen repository inventory must contain four OneFormer PTMs"
        )
    records.sort(key=lambda item: item["id"])
    return {
        "registry_version": registry.registry_version,
        "registry_sha256": registry.document_sha256,
        "default_ptm": model["default_ptm"],
        "records": records,
        "record_count": len(records),
        "supported_ids": [
            item["id"] for item in records if item["status"] == "supported"
        ],
        "unverified_ids": [
            item["id"] for item in records if item["status"] == "unverified"
        ],
    }


def validate_packaged_train_schema(skill_dir: str | Path) -> dict[str, Any]:
    root = Path(skill_dir)
    info_path = root / "references/skill_info.yaml"
    schema_path = root / "schemas/train.schema.json"
    template_path = root / "references/spec_template_train.yaml"
    for path in (info_path, schema_path, template_path):
        if not path.is_file():
            raise CampaignContractError(f"missing packaged skill artifact: {path}")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if schema.get("x_tao_schema", {}).get("network_arch") != "oneformer":
        raise CampaignContractError("packaged train schema is not OneFormer")
    defaults = schema.get("automl_default_parameters")
    if not isinstance(defaults, list):
        raise CampaignContractError(
            "packaged train schema lacks AutoML default parameters"
        )
    missing = sorted(set(SEARCH_PARAMETERS) - set(defaults))
    if missing:
        raise CampaignContractError(
            "frozen search parameters are not AutoML enabled: "
            + ", ".join(missing)
        )
    return {
        "skill_info_path": str(info_path),
        "skill_info_sha256": sha256_file(info_path),
        "schema_path": str(schema_path),
        "schema_sha256": sha256_file(schema_path),
        "template_path": str(template_path),
        "template_sha256": sha256_file(template_path),
        "explicit_search_parameters": list(SEARCH_PARAMETERS),
        "non_train_fields_excluded": True,
    }


def mode_objective(mode: str) -> dict[str, Any]:
    objectives = [
        {"metric": "PQ", "direction": "maximize", "role": "accuracy"},
        {"metric": "latency_ms", "direction": "minimize", "role": "latency"},
    ]
    if mode == "accuracy":
        return {
            "selection_mode": mode,
            "objectives": objectives,
            "acquisition": "expected_improvement",
            "latency_accuracy_retention": None,
            "multi_objective_min_accuracy": None,
            "selection_policy": "highest_valid_accuracy",
        }
    if mode == "latency":
        return {
            "selection_mode": mode,
            "objectives": objectives,
            "acquisition": "constrained_expected_improvement",
            "latency_accuracy_retention": {
                "type": "relative",
                "retained_fraction": FROZEN_LATENCY_RETENTION,
                "reference": "accuracy_winner",
            },
            "multi_objective_min_accuracy": None,
            "selection_policy": "equivalent_fastest_accuracy_tiebreak",
        }
    if mode == "multi_objective":
        return {
            "selection_mode": mode,
            "objectives": objectives,
            "acquisition": "parego_expected_improvement",
            "latency_accuracy_retention": None,
            "multi_objective_min_accuracy": None,
            "selection_policy": "normalized_augmented_chebyshev",
        }
    raise CampaignContractError(f"unsupported mode: {mode!r}")


def mode_settings(campaign_id: str, mode: str) -> dict[str, Any]:
    objective = mode_objective(mode)
    settings = {
        "algorithm": "bayesian",
        "automl_max_recommendations": FROZEN_CANDIDATE_BUDGET,
        "automl_max_concurrent": 1,
        "campaign_id": campaign_id,
        "job_id": f"{campaign_id}-{mode}",
        "session_id": f"{campaign_id}-{mode}",
        "experiment_id": f"{campaign_id}-{mode}-observations",
        "random_seed": FROZEN_SEARCH_SEED,
        "objectives": [
            {"metric": item["metric"], "direction": item["direction"]}
            for item in objective["objectives"]
        ],
        "selection_mode": mode,
        "accuracy_metric": "PQ",
        "latency_metric": "latency_ms",
        "objective_acquisition": {
            "calibration_points": FROZEN_CALIBRATION_POINTS_PER_ARM,
            "augmentation_rho": 1.0e-6,
        },
        "objective_normalization": "pareto_front",
        "augmentation_rho": 1.0e-6,
        "accuracy_tolerance": 1.0e-12,
        "latency_tolerance": FROZEN_LATENCY_TOLERANCE_MS,
        "selection_score_tolerance": 1.0e-12,
        "latency_ci_low_metric": "latency_ci95_low_ms",
        "latency_ci_high_metric": "latency_ci95_high_ms",
        "multi_objective_min_accuracy": None,
        "run_baseline": False,
        "run_final_evaluation": False,
        "require_eval_fn_success": True,
        "automl_delete_intermediate_ckpt": False,
        "automl_checkpoint_retention_strategy": "terminal",
    }
    if mode == "latency":
        settings["latency_accuracy_retention"] = copy.deepcopy(
            objective["latency_accuracy_retention"]
        )
    return settings


def custom_ranges() -> dict[str, dict[str, Any]]:
    ranges: dict[str, dict[str, Any]] = {}
    for name in SEARCH_PARAMETERS:
        record = SEARCH_SPACE[name]
        ranges[name] = {
            "valid_min": record["minimum"],
            "valid_max": record["maximum"],
        }
    return ranges


def profile_overrides(dataset_root: str) -> dict[str, Any]:
    """Return the identical native-COCO spec for every mode and PTM arm."""
    if not isinstance(dataset_root, str) or not dataset_root.startswith(
        "/lustre/"
    ):
        raise CampaignContractError("dataset root must be an absolute Lustre path")
    annotations = f"{dataset_root}/annotations"
    images = f"{dataset_root}/images"
    return {
        "model_name": "oneformer_coco2017",
        "results_dir": "",
        "wandb": {"enable": False},
        "model": {
            "sem_seg_head": {"num_classes": 133},
            "test": {
                "semantic_on": True,
                "instance_on": True,
                "panoptic_on": True,
            },
        },
        "dataset": {
            "train": {
                "images": f"{images}/train2017",
                "annotations": f"{annotations}/panoptic_train2017.json",
                "panoptic": f"{annotations}/panoptic_train2017",
                "batch_size": FROZEN_BATCH_SIZE_PER_REPLICA,
                "num_workers": 4,
            },
            "val": {
                "images": f"{images}/val2017",
                "annotations": f"{annotations}/panoptic_val2017.json",
                "panoptic": f"{annotations}/panoptic_val2017",
                "batch_size": FROZEN_BATCH_SIZE_PER_REPLICA,
                "num_workers": 4,
            },
            "test": {
                "images": f"{images}/val2017",
                "annotations": f"{annotations}/panoptic_val2017.json",
                "panoptic": f"{annotations}/panoptic_val2017",
                "batch_size": FROZEN_BATCH_SIZE_PER_REPLICA,
                "num_workers": 4,
            },
            "label_map": f"{dataset_root}/tao/label_map_panoptic.json",
            "contiguous_id": True,
            "task_prob_train": {
                "semantic": 0.0,
                "instance": 0.0,
                "panoptic": 1.0,
            },
            "task_prob_val": {
                "semantic": 0.0,
                "instance": 0.0,
                "panoptic": 1.0,
            },
        },
        "train": {
            "num_gpus": 8,
            "gpu_ids": list(range(8)),
            "num_nodes": 1,
            "seed": FROZEN_TRAINING_SEED,
            "num_epochs": FROZEN_TRAINING_EPOCHS,
            "checkpoint_interval": FROZEN_TRAINING_EPOCHS,
            "checkpoint_interval_unit": "epoch",
            "validation_interval": 1,
            "resume_training_checkpoint_path": "",
            "results_dir": "",
            "precision": "32",
            "cudnn": {"benchmark": False, "deterministic": True},
        },
        "evaluate": {
            "num_gpus": 8,
            "gpu_ids": list(range(8)),
            "num_nodes": 1,
            "checkpoint": "",
            "batch_size": 1,
            "iou_per_class": True,
            "task": "panoptic",
        },
        "inference": {"mode": "panoptic"},
    }


def validate_dataset_record(dataset: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "id": "coco_2017_full_instance_panoptic",
        "train_image_count": 118287,
        "validation_image_count": 5000,
        "train_panoptic_png_count": 118287,
        "validation_panoptic_png_count": 5000,
        "panoptic_category_count": 133,
        "panoptic_label_map_sha256": (
            "4b28b3773f0f8e63d836dc20da77276633da72178453458b79e32be8e892ce56"
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
        "manifest_sha256": (
            "10566a60498de9998154f44a34445a488c9f030e09f2a7346d20a4a1c55f804e"
        ),
        "file_manifest_entry_count": 246593,
        "stage_manifest_sha256": (
            "437ff12490637950707b9b951d820ea34d38b926080a478a5d182c2d284a0c5d"
        ),
        "remote_read_only": True,
        "remote_writable_entries_after_lock": 0,
    }
    for key, expected in required.items():
        if dataset.get(key) != expected:
            raise CampaignContractError(
                f"COCO2017 dataset field {key!r} changed"
            )
    root = dataset.get("root")
    if not isinstance(root, str) or not root.startswith("/lustre/"):
        raise CampaignContractError("COCO2017 root must be on Lustre")
    for key in (
        "manifest_path",
        "stage_manifest_path",
        "stage_manifest_lustre_path",
        "remote_file_manifest_path",
    ):
        value = dataset.get(key)
        if not isinstance(value, str):
            raise CampaignContractError(f"COCO2017 {key} is missing")
        if key in {"manifest_path", "stage_manifest_path"}:
            if not Path(value).is_absolute():
                raise CampaignContractError(f"COCO2017 {key} must be absolute")
        elif not value.startswith("/lustre/"):
            raise CampaignContractError(f"COCO2017 {key} must be on Lustre")
    return copy.deepcopy(dict(dataset))


def build_preregistered_contract(
    *,
    campaign_id: str,
    dataset: Mapping[str, Any],
    skill_dir: str,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Build immutable intent without granting launch authorization."""
    _finite_fraction(FROZEN_LATENCY_RETENTION, "latency retention")
    dataset_record = validate_dataset_record(dataset)
    schema = validate_packaged_train_schema(skill_dir)
    ptm_inventory = oneformer_registry_snapshot()
    value = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "model": "oneformer",
        "network_arch": "oneformer",
        "task": "panoptic_segmentation",
        "primary_accuracy_metric": "PQ",
        "metric_semantics": {
            "observed_metric": "panoptic_quality",
            "metric_scale": "unit_interval",
            "source": "native_coco_panoptic_annotations",
            "pq_emitted_by_overlaid_train_evaluate_path": True,
            "pq_claim_authorized": True,
            "semantic_miou_used_as_panoptic_objective": False,
            "distributed_reduction": (
                "global_additive_sufficient_statistics_before_metric"
            ),
        },
        "dataset": dataset_record,
        "runtime": copy.deepcopy(dict(runtime)),
        "sqsh": copy.deepcopy(FROZEN_SQSH),
        "runtime_overlay": copy.deepcopy(FROZEN_RUNTIME_OVERLAY),
        "schema": schema,
        "ptm_inventory": ptm_inventory,
        "qualification_policy": {
            "kind": "direct_full_gpu_train_eval_then_supported_registry",
            "cpu_model_runs": 0,
            "smoke_model_runs": 0,
            "mini_step_runs": 0,
            "nodes_per_job": 1,
            "gpus_per_job": 8,
            "full_dataset": True,
            "training_epochs": FROZEN_TRAINING_EPOCHS,
            "standalone_evaluation": True,
            "registry_bypass_allowed": False,
            "qualification_evidence_path": runtime[
                "qualification_evidence_path"
            ],
            "ptm_stage_manifest_path": runtime["ptm_stage_manifest_path"],
            "ptm_stage_manifest_sha256": runtime[
                "ptm_stage_manifest_sha256"
            ],
            "ptm_stage_content_sha256": runtime[
                "ptm_stage_content_sha256"
            ],
        },
        "execution": {
            "kind": "objective_aware_three_mode_search",
            "cpu_runs": 0,
            "smoke_runs": 0,
            "local_model_runs": 0,
            "independent_mode_jobs": True,
            "shared_archive": False,
            "first_candidate_gate": True,
            "automatic_remaining_budget_release": True,
            "automatic_trigger": True,
            "nodes_per_child": 1,
            "gpus_per_child": 8,
            "container_mode": "pinned_sqsh",
        },
        "search": {
            "algorithm": "bayesian",
            "implementation": "hierarchical_ptm_objective_aware_bayesian_v1",
            "candidate_budget_per_mode": FROZEN_CANDIDATE_BUDGET,
            "search_seed": FROZEN_SEARCH_SEED,
            "training_seed": FROZEN_TRAINING_SEED,
            "training_epochs": FROZEN_TRAINING_EPOCHS,
            "calibration_points_per_arm": FROZEN_CALIBRATION_POINTS_PER_ARM,
            "invalid_recovery_issues_per_arm": (
                FROZEN_INVALID_RECOVERY_ISSUES_PER_ARM
            ),
            "parameters": list(SEARCH_PARAMETERS),
            "space": copy.deepcopy(SEARCH_SPACE),
            "space_sha256": canonical_sha256(SEARCH_SPACE),
            "latency_accuracy_retention": FROZEN_LATENCY_RETENTION,
            "latency_practical_tolerance_ms": FROZEN_LATENCY_TOLERANCE_MS,
            "ptm_representation": "hierarchical_nonordinal_arms",
            "ptm_policy_by_mode": {
                "accuracy": "all_runtime_supported_explicit_multi_ptm",
                "latency": "all_runtime_supported",
                "multi_objective": "all_runtime_supported",
            },
        },
        "validation_sanity_gate": {
            "metric": "PQ",
            "minimum": FROZEN_VALIDATION_SANITY_MIN_PQ,
            "role": "experiment_correctness_gate_not_product_selection",
            "rationale": (
                "On native 133-category COCO panoptic annotations, a finite "
                "PQ below 0.01 requires data, label, transfer-load, "
                "optimization, fidelity, and metric root-cause analysis."
            ),
            "low_finite_metric_automatically_accepted": False,
        },
        "training_fidelity": {
            "kind": "one_complete_full_dataset_epoch_pilot",
            "epochs": FROZEN_TRAINING_EPOCHS,
            "reason": (
                "One complete 118287-image epoch is the frozen pilot fidelity; "
                "failure to produce a meaningful metric is classified as "
                "training_fidelity_insufficient and does not authorize "
                "result-driven range or budget changes."
            ),
        },
        "latency_protocol": copy.deepcopy(LATENCY_PROTOCOL),
        "modes": [
            {
                "mode": mode,
                "observation_namespace": f"{campaign_id}-{mode}-observations",
                "observation_sharing": False,
                "initial_observation_ids": [],
                "objective": mode_objective(mode),
                "settings": mode_settings(campaign_id, mode),
            }
            for mode in MODES
        ],
        "agent_intervention_flags": {name: False for name in AGENT_FLAGS},
        "selection_isolation_flags": {
            name: False for name in SELECTION_FLAGS
        },
    }
    value["contract_sha256"] = canonical_sha256(value)
    return value


def validate_contract(document: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(document))
    observed = value.pop("contract_sha256", None)
    if observed != canonical_sha256(value):
        raise CampaignContractError("campaign contract integrity failed")
    expected_modes = [
        {
            "mode": mode,
            "observation_namespace": (
                f"{value.get('campaign_id')}-{mode}-observations"
            ),
            "observation_sharing": False,
            "initial_observation_ids": [],
            "objective": mode_objective(mode),
            "settings": mode_settings(str(value.get("campaign_id")), mode),
        }
        for mode in MODES
    ]
    if (
        value.get("model") != "oneformer"
        or value.get("network_arch") != "oneformer"
        or value.get("task") != "panoptic_segmentation"
        or value.get("primary_accuracy_metric") != "PQ"
        or value.get("metric_semantics", {}).get("pq_claim_authorized")
        is not True
        or value.get("metric_semantics", {}).get(
            "semantic_miou_used_as_panoptic_objective"
        )
        is not False
        or value.get("execution", {}).get("cpu_runs") != 0
        or value.get("execution", {}).get("smoke_runs") != 0
        or value.get("execution", {}).get("gpus_per_child") != 8
        or value.get("execution", {}).get("container_mode")
        != "pinned_sqsh"
        or value.get("search", {}).get("space") != SEARCH_SPACE
        or value.get("modes") != expected_modes
        or value.get("validation_sanity_gate", {}).get("metric") != "PQ"
        or value.get("validation_sanity_gate", {}).get("minimum")
        != FROZEN_VALIDATION_SANITY_MIN_PQ
        or value.get("qualification_policy", {}).get(
            "ptm_stage_manifest_sha256"
        )
        != value.get("runtime", {}).get("ptm_stage_manifest_sha256")
        or value.get("qualification_policy", {}).get(
            "ptm_stage_content_sha256"
        )
        != value.get("runtime", {}).get("ptm_stage_content_sha256")
    ):
        raise CampaignContractError("campaign execution policy changed")
    validate_dataset_record(value["dataset"])
    if value.get("sqsh") != FROZEN_SQSH:
        raise CampaignContractError("pinned SQSH identity changed")
    if value.get("runtime_overlay") != FROZEN_RUNTIME_OVERLAY:
        raise CampaignContractError("pinned OneFormer runtime overlay changed")
    if any(value["agent_intervention_flags"].values()):
        raise CampaignContractError("agent intervention flags must remain false")
    if any(value["selection_isolation_flags"].values()):
        raise CampaignContractError("selection isolation flags must remain false")
    value["contract_sha256"] = observed
    return value


__all__ = [
    "AGENT_FLAGS",
    "CampaignContractError",
    "FROZEN_BATCH_SIZE_PER_REPLICA",
    "FROZEN_CALIBRATION_POINTS_PER_ARM",
    "FROZEN_CANDIDATE_BUDGET",
    "FROZEN_HARDWARE",
    "FROZEN_LATENCY_RETENTION",
    "FROZEN_LATENCY_TOLERANCE_MS",
    "FROZEN_RUNTIME_OVERLAY",
    "FROZEN_SEARCH_SEED",
    "FROZEN_SLURM_RETRY_CAP",
    "FROZEN_SQSH",
    "FROZEN_TRAINING_EPOCHS",
    "FROZEN_VALIDATION_SANITY_MIN_PQ",
    "LATENCY_PROTOCOL",
    "MODES",
    "SEARCH_PARAMETERS",
    "SEARCH_SPACE",
    "SELECTION_FLAGS",
    "build_preregistered_contract",
    "custom_ranges",
    "mode_objective",
    "mode_settings",
    "oneformer_registry_snapshot",
    "profile_overrides",
    "sha256_file",
    "validate_contract",
    "validate_dataset_record",
    "validate_packaged_train_schema",
]
