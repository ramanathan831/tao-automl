"""Tests for fail-closed Grounding DINO shared-detection preparation."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from tao_automl.ptm_registry import load_ptm_registry

from .contract import (
    AGENT_FLAGS,
    MODEL_ID,
    MODES,
    SELECTION_FLAGS,
    PreparationError,
    build_preparation,
    derive_category_contract,
    derive_official_ptms,
    grounding_annotation_contract,
    read_json,
    validate_preparation,
)


HERE = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def inputs():
    return read_json(HERE / "campaign.inputs.v1.json")


@pytest.fixture(scope="module")
def preparation(inputs):
    return build_preparation(experiment_dir=HERE, inputs=inputs)


def test_exact_repository_model_identifier_and_skill(preparation):
    assert preparation["model"]["id"] == "grounding_dino"
    assert preparation["model"]["skill"] == "tao-train-grounding-dino"
    assert preparation["automl"]["search_schema"]["model_id"] == MODEL_ID


def test_category_prompts_are_derived_verbatim_from_source(preparation):
    category = preparation["dataset"]["category_contract"]
    assert category["source_category_ids"] == [1, 2, 3, 4]
    assert category["model_category_ids"] == [0, 1, 2, 3]
    assert category["prompt_list"] == [
        "cone",
        "forklift",
        "cart",
        "fire_extinguisher",
    ]
    assert category["label_map"] == {
        "0": "cone",
        "1": "forklift",
        "2": "cart",
        "3": "fire_extinguisher",
    }
    assert category["manual_prompt_or_synonym_injection"] is False
    assert {
        item["derivation"] for item in category["prompt_mapping"]
    } == {"exact_source_coco_category_name"}


def test_category_mapping_is_input_order_invariant(inputs):
    manifest_path = (HERE / inputs["dataset"]["manifest_file"]).resolve()
    audit_path = (HERE / inputs["dataset"]["audit_file"]).resolve()
    manifest = read_json(manifest_path)
    audit = read_json(audit_path)
    expected = derive_category_contract(manifest, audit)

    changed_manifest = copy.deepcopy(manifest)
    changed_manifest["categories"].reverse()
    changed_audit = copy.deepcopy(audit)
    for split in ("train", "validation"):
        changed_audit["splits"][split]["categories"].reverse()
    assert derive_category_contract(changed_manifest, changed_audit) == expected


def test_duplicate_or_inconsistent_categories_fail_closed(inputs):
    manifest = read_json((HERE / inputs["dataset"]["manifest_file"]).resolve())
    audit = read_json((HERE / inputs["dataset"]["audit_file"]).resolve())
    duplicate = copy.deepcopy(manifest)
    duplicate["categories"][1]["id"] = duplicate["categories"][0]["id"]
    with pytest.raises(PreparationError, match="unique"):
        derive_category_contract(duplicate, audit)

    mismatched = copy.deepcopy(audit)
    mismatched["splits"]["validation"]["categories"][0]["name"] = "invented"
    with pytest.raises(PreparationError, match="identities differ"):
        derive_category_contract(manifest, mismatched)


def test_plain_coco_is_not_misrepresented_as_phrase_grounding(inputs):
    audit = read_json((HERE / inputs["dataset"]["audit_file"]).resolve())
    contract = grounding_annotation_contract(audit)
    assert contract["category_prompted_detection"]["supported_by_source"] is True
    phrase = contract["referring_expression_box_grounding"]
    assert phrase["supported_by_source"] is False
    assert phrase["images_with_caption"] == 0
    assert phrase["annotations_with_tokens_positive"] == 0
    assert "must not be represented as referring expressions" in phrase["blocker"]


def test_every_official_repository_ptm_is_derived_without_manual_filtering():
    records = derive_official_ptms()
    registry_records = load_ptm_registry().to_dict()["models"][MODEL_ID][
        "checkpoints"
    ]
    expected = sorted(
        item["id"]
        for item in registry_records
        if item["source"].get("official") is True
    )
    assert [item["id"] for item in records] == expected
    assert len(records) == 2
    assert all(item["status"] == "unverified" for item in records)


def test_schema_search_space_is_repository_derived(preparation):
    schema = preparation["automl"]["search_schema"]
    assert schema["source"] == "packaged_train_schema_automl_default_parameters"
    assert schema["parameter_names"] == sorted(schema["parameter_names"])
    assert "model.enc_layers" in schema["parameter_names"]
    assert "model.dec_layers" in schema["parameter_names"]
    assert "model.num_select" in schema["parameter_names"]
    assert all(
        node["automl_enabled"] is True
        for node in schema["parameters"].values()
    )


def test_three_independent_objective_aware_jobs_are_prepared(preparation):
    modes = preparation["automl"]["modes"]
    assert tuple(item["mode"] for item in modes) == MODES
    assert len({item["observation_namespace"] for item in modes}) == 3
    assert all(item["observation_sharing"] is False for item in modes)
    assert all(item["initial_observation_ids"] == [] for item in modes)
    assert [item["objective"]["acquisition"] for item in modes] == [
        "expected_improvement",
        "constrained_expected_improvement",
        "parego_expected_improvement",
    ]
    latency = modes[1]["objective"]["quality_constraint"]
    assert latency["retained_fraction"] == pytest.approx(0.90)
    assert latency["reference"] == "best_observed_within_job"
    assert latency["reference_updates"] == "monotonic"
    assert modes[2]["objective"]["quality_constraint"] is None


def test_all_jobs_are_direct_eight_gpu_sqsh_jobs(preparation):
    runtime = preparation["runtime"]
    assert runtime["platform"] == "slurm"
    assert runtime["nodes"] == 1
    assert runtime["gpus_per_node"] == 8
    assert runtime["distributed_workers_per_node"] == 8
    assert runtime["sqsh_direct_path"] is True
    assert runtime["slurm_use_sqsh_conversion"] is False
    assert runtime["sqsh_path"].endswith(".sqsh")
    assert preparation["direct_qualification"]["cpu_model_runs"] is False
    assert preparation["direct_qualification"]["smoke_or_ministep_runs"] is False


def test_qualification_covers_every_official_ptm(preparation):
    inventory = preparation["official_ptm_inventory"]
    jobs = preparation["direct_qualification"]["jobs"]
    assert inventory["manual_ptm_selection"] is False
    assert [job["ptm_id"] for job in jobs] == inventory["candidate_ids"]
    assert all(job["resource"]["gpus_per_node"] == 8 for job in jobs)
    assert all(
        job["workflow"][0] == "full_10_epoch_train_with_validation_each_epoch"
        for job in jobs
    )


def test_metric_and_ptm_trust_boundaries_keep_gate_closed(preparation):
    metric = preparation["metric_contract"]
    assert metric["val_mAP50"]["availability"] == "unregistered"
    assert metric["val_Pr@0.5"]["availability"] == "blocked"
    gate = preparation["automatic_gate"]
    assert gate["launch_authorized"] is False
    codes = {item["code"] for item in gate["blockers"]}
    assert "referring_expression_annotation_contract_missing" in codes
    assert "category_detection_metric_policy_not_supported" in codes
    assert "official_ptms_not_production_qualified" in codes
    assert "converted_dataset_artifacts_not_sealed" in codes


def test_no_agent_or_validation_measurement_can_change_selection(preparation):
    assert set(preparation["agent_intervention_flags"]) == set(AGENT_FLAGS)
    assert set(preparation["selection_isolation_flags"]) == set(SELECTION_FLAGS)
    assert not any(preparation["agent_intervention_flags"].values())
    assert not any(preparation["selection_isolation_flags"].values())


def test_preparation_performs_no_launch_or_model_execution(preparation):
    assert preparation["execution"] == {
        "jobs_submitted": 0,
        "scheduler_mutation_performed": False,
        "model_execution_performed": False,
    }
    validate_preparation(preparation)


def test_preparation_hash_detects_mutation(preparation):
    changed = copy.deepcopy(preparation)
    changed["runtime"]["gpus_per_node"] = 4
    with pytest.raises(PreparationError, match="eight-GPU"):
        validate_preparation(changed)

    changed = copy.deepcopy(preparation)
    changed["execution"]["jobs_submitted"] = 1
    with pytest.raises(PreparationError, match="model execution"):
        validate_preparation(changed)
