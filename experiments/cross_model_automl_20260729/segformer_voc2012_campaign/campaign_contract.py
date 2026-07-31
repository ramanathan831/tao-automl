#!/usr/bin/env python3

"""Frozen SegFormer/VOC2012 three-mode campaign policy.

This module contains experiment intent only.  It never qualifies a checkpoint,
changes registry status, or chooses a winner.  Runtime eligibility is granted
only when a repository-supported PTM also has immutable evidence from a real
one-node/eight-GPU full-dataset training and evaluation workflow.
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

# PTM identity is a hierarchical categorical outer arm.  These are the two
# minimal, inference-invariant training parameters exposed by the packaged
# SegFormer train schema and useful across every FAN arm.
SEARCH_PARAMETERS = (
    "train.optim.lr",
    "train.optim.weight_decay",
)
SEARCH_SPACE = {
    "train.optim.lr": {
        "type": "float",
        "minimum": 2.0e-5,
        "maximum": 6.0e-4,
        "scale": "linear",
    },
    "train.optim.weight_decay": {
        "type": "float",
        "minimum": 1.0e-4,
        "maximum": 0.10,
        "scale": "linear",
    },
}

FROZEN_CANDIDATE_BUDGET = 30
FROZEN_TRAINING_EPOCHS = 10
FROZEN_SEARCH_SEED = 271828
FROZEN_TRAINING_SEED = 1234
FROZEN_CALIBRATION_POINTS_PER_ARM = 2
FROZEN_INVALID_RECOVERY_ISSUES_PER_ARM = 1
FROZEN_LATENCY_RETENTION = 0.90
FROZEN_LATENCY_TOLERANCE_MS = 0.73553775
FROZEN_VALIDATION_SANITY_MIN_MIOU = 0.10
FROZEN_SLURM_RETRY_CAP = 10
FROZEN_SLURM_PARTITION = "polar3"
FROZEN_SLURM_TIME_HOURS = 4.0
FROZEN_SLURM_TIMEOUT_HOURS = 3.8
FROZEN_IMAGE_SIZE = 512
FROZEN_BATCH_SIZE_PER_REPLICA = 4
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
    "image_reference": (
        "nvcr.io/nvstaging/tao/tao-toolkit-pyt:"
        "7.1.0-rc-245-multiarch"
    ),
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
    "timed_scope": "segformer_model_forward",
    "excluded_scope": [
        "checkpoint_load",
        "disk_io",
        "image_decode",
        "resize_normalize",
        "host_to_device_transfer",
        "argmax_and_mask_serialization",
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

VOC_CLASS_NAMES = (
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)


class CampaignContractError(ValueError):
    """The SegFormer campaign contract is inconsistent."""


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


def voc_palette() -> list[dict[str, Any]]:
    """Return the loss/metric-preserving grayscale VOC palette."""
    records = [
        {
            "label_id": label_id,
            "mapping_class": name,
            "rgb": [label_id],
            "seg_class": name,
        }
        for label_id, name in enumerate(VOC_CLASS_NAMES)
    ]
    records.append(
        {
            "label_id": 255,
            "mapping_class": "ignore",
            "rgb": [255],
            "seg_class": "ignore",
        }
    )
    return records


def segformer_registry_snapshot() -> dict[str, Any]:
    """Snapshot every official repository-owned SegFormer PTM record."""
    registry = load_ptm_registry()
    model = registry.to_dict()["models"]["segformer"]
    records = []
    for record in model["checkpoints"]:
        if (
            record.get("source", {}).get("official") is not True
            or record.get("model_family") != "segformer"
            or "semantic_segmentation"
            not in record.get("task_compatibility", ())
        ):
            raise CampaignContractError(
                f"invalid official SegFormer registry record: {record.get('id')}"
            )
        records.append(
            {
                "id": record["id"],
                "status": record["status"],
                "status_reason": record.get("status_reason"),
                "source": copy.deepcopy(record["source"]),
                "expected_size_bytes": record["expected_size_bytes"],
                "checkpoint_target": record["checkpoint_target"],
                "architecture": record["architecture"],
                "backbone": record["backbone"],
                "input_contract": copy.deepcopy(record["input_contract"]),
                "registry_record_sha256": canonical_sha256(record),
            }
        )
    if len(records) != 13 or len({item["id"] for item in records}) != 13:
        raise CampaignContractError(
            "the frozen repository inventory must contain 13 SegFormer PTMs"
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
    if schema.get("x_tao_schema", {}).get("network_arch") != "segformer":
        raise CampaignContractError("packaged train schema is not SegFormer")
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
        {"metric": "val_miou", "direction": "maximize", "role": "accuracy"},
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
        "accuracy_metric": "val_miou",
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
    return {
        name: {
            "valid_min": SEARCH_SPACE[name]["minimum"],
            "valid_max": SEARCH_SPACE[name]["maximum"],
        }
        for name in SEARCH_PARAMETERS
    }


def profile_overrides(dataset_root: str) -> dict[str, Any]:
    """Return the identical full-dataset spec applied to every mode/PTM arm."""
    if not isinstance(dataset_root, str) or not dataset_root.startswith(
        "/lustre/"
    ):
        raise CampaignContractError("dataset root must be an absolute Lustre path")
    return {
        "model_name": "segformer_voc2012",
        "results_dir": "",
        "wandb": {"enable": False},
        "dataset": {
            "segment": {
                "root_dir": dataset_root,
                "dataset": "SFDataset",
                "num_classes": 21,
                "img_size": FROZEN_IMAGE_SIZE,
                "batch_size": FROZEN_BATCH_SIZE_PER_REPLICA,
                "workers": 8,
                "shuffle": True,
                "train_split": "train",
                "validation_split": "val",
                "test_split": "val",
                "predict_split": "val",
                "label_transform": "None",
                "palette": voc_palette(),
            }
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
            "tensorboard": {"enabled": False},
            "use_distributed_sampler": False,
            "sync_batchnorm": False,
            "cudnn": {"benchmark": False, "deterministic": True},
        },
    }


def validate_dataset_record(dataset: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "id": "pascal_voc_2012_full_semantic_segmentation",
        "train_image_count": 1464,
        "train_mask_count": 1464,
        "validation_image_count": 1449,
        "validation_mask_count": 1449,
        "num_classes": 21,
        "ignore_label": 255,
        "official_archive_sha256": (
            "e14f763270cf193d0b5f74b169f44157a4b0c6efa708f4dd0ff78ee691763bcb"
        ),
        "content_sha256": (
            "815b5d01b625238b449c4bca828bf96107b367f0f4d5d8a31d2f97c6161a5de0"
        ),
        "manifest_sha256": (
            "051ab20215b8e6976763ac82a3db20a68264759edef3d62fd0c8553c501123ff"
        ),
        "file_manifest_entry_count": 5827,
        "stage_manifest_sha256": (
            "437ff12490637950707b9b951d820ea34d38b926080a478a5d182c2d284a0c5d"
        ),
        "remote_read_only": True,
        "remote_writable_entries_after_lock": 0,
    }
    for key, expected in required.items():
        if dataset.get(key) != expected:
            raise CampaignContractError(
                f"VOC2012 dataset field {key!r} changed"
            )
    root = dataset.get("prepared_root")
    if not isinstance(root, str) or not root.startswith("/lustre/"):
        raise CampaignContractError("VOC2012 prepared_root must be on Lustre")
    if (
        not isinstance(dataset.get("manifest_path"), str)
        or not Path(dataset["manifest_path"]).is_absolute()
        or not isinstance(dataset.get("stage_manifest_path"), str)
        or not Path(dataset["stage_manifest_path"]).is_absolute()
        or not isinstance(dataset.get("stage_manifest_lustre_path"), str)
        or not dataset["stage_manifest_lustre_path"].startswith("/lustre/")
        or not isinstance(dataset.get("remote_file_manifest_path"), str)
        or not dataset["remote_file_manifest_path"].startswith("/lustre/")
    ):
        raise CampaignContractError(
            "VOC2012 local and Lustre provenance paths are invalid"
        )
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
    ptm_inventory = segformer_registry_snapshot()
    value = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "model": "segformer",
        "network_arch": "segformer",
        "task": "semantic_segmentation",
        "primary_accuracy_metric": "val_miou",
        "dataset": dataset_record,
        "runtime": copy.deepcopy(dict(runtime)),
        "sqsh": copy.deepcopy(FROZEN_SQSH),
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
            "implementation": (
                "hierarchical_ptm_objective_aware_bayesian_v1"
            ),
            "candidate_budget_per_mode": FROZEN_CANDIDATE_BUDGET,
            "search_seed": FROZEN_SEARCH_SEED,
            "training_seed": FROZEN_TRAINING_SEED,
            "training_epochs": FROZEN_TRAINING_EPOCHS,
            "calibration_points_per_arm": (
                FROZEN_CALIBRATION_POINTS_PER_ARM
            ),
            "invalid_recovery_issues_per_arm": (
                FROZEN_INVALID_RECOVERY_ISSUES_PER_ARM
            ),
            "parameters": list(SEARCH_PARAMETERS),
            "space": copy.deepcopy(SEARCH_SPACE),
            "space_sha256": canonical_sha256(SEARCH_SPACE),
            "latency_accuracy_retention": FROZEN_LATENCY_RETENTION,
            "latency_practical_tolerance_ms": (
                FROZEN_LATENCY_TOLERANCE_MS
            ),
            "ptm_representation": "hierarchical_nonordinal_arms",
            "ptm_policy_by_mode": {
                "accuracy": "all_runtime_supported",
                "latency": "all_runtime_supported",
                "multi_objective": "all_runtime_supported",
            },
        },
        "validation_sanity_gate": {
            "metric": "val_miou",
            "minimum": FROZEN_VALIDATION_SANITY_MIN_MIOU,
            "role": "experiment_correctness_gate_not_product_selection",
            "rationale": (
                "For 21-class VOC semantic segmentation, a value below 0.10 "
                "requires data, label, optimization, fidelity, and metric "
                "root-cause analysis before a campaign can continue."
            ),
            "low_finite_metric_automatically_accepted": False,
        },
        "latency_protocol": copy.deepcopy(LATENCY_PROTOCOL),
        "modes": [
            {
                "mode": mode,
                "observation_namespace": (
                    f"{campaign_id}-{mode}-observations"
                ),
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
    if (
        value.get("model") != "segformer"
        or value.get("network_arch") != "segformer"
        or value.get("task") != "semantic_segmentation"
        or value.get("primary_accuracy_metric") != "val_miou"
        or value.get("execution", {}).get("cpu_runs") != 0
        or value.get("execution", {}).get("smoke_runs") != 0
        or value.get("execution", {}).get("gpus_per_child") != 8
        or value.get("execution", {}).get("container_mode")
        != "pinned_sqsh"
        or value.get("runtime", {}).get("partition")
        != FROZEN_SLURM_PARTITION
        or value.get("runtime", {}).get("time_hours")
        != FROZEN_SLURM_TIME_HOURS
        or value.get("runtime", {}).get("timeout_hours")
        != FROZEN_SLURM_TIMEOUT_HOURS
        or value.get("search", {}).get("space") != SEARCH_SPACE
        or tuple(item.get("mode") for item in value.get("modes", ()))
        != MODES
    ):
        raise CampaignContractError("campaign execution policy changed")
    validate_dataset_record(value["dataset"])
    if value.get("sqsh") != FROZEN_SQSH:
        raise CampaignContractError("pinned SQSH identity changed")
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
    "FROZEN_IMAGE_SIZE",
    "FROZEN_LATENCY_RETENTION",
    "FROZEN_LATENCY_TOLERANCE_MS",
    "FROZEN_SEARCH_SEED",
    "FROZEN_SLURM_RETRY_CAP",
    "FROZEN_SLURM_PARTITION",
    "FROZEN_SLURM_TIME_HOURS",
    "FROZEN_SLURM_TIMEOUT_HOURS",
    "FROZEN_SQSH",
    "FROZEN_TRAINING_EPOCHS",
    "FROZEN_VALIDATION_SANITY_MIN_MIOU",
    "LATENCY_PROTOCOL",
    "MODES",
    "SEARCH_PARAMETERS",
    "SEARCH_SPACE",
    "SELECTION_FLAGS",
    "VOC_CLASS_NAMES",
    "build_preregistered_contract",
    "custom_ranges",
    "mode_objective",
    "mode_settings",
    "profile_overrides",
    "segformer_registry_snapshot",
    "sha256_file",
    "validate_contract",
    "validate_dataset_record",
    "validate_packaged_train_schema",
    "voc_palette",
]
