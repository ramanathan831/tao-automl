from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from tao_automl.recommendation_audit import (
    algorithmic_campaign_flags,
    build_recommendation_audit,
)
from tao_automl.selection import analyze_archive

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from automatic_successor import (  # noqa: E402
    AutomaticSuccessorError,
    EXPECTED_MODEL_BASED_METHOD,
    EXPECTED_OPTIMIZATION_DIRECTION,
    MODES,
    SELECTION_TIME_ISOLATION,
    _process_identity,
    _selection_config,
    canonical_sha256,
    sha256_file,
    trigger_successor,
    validate_completed_dino,
    validate_successor_descriptor,
    watch_and_trigger,
)
from dino_campaign.manifest_generator import load_manifest  # noqa: E402


MANIFEST_PATH = HERE / "dino_campaign" / "campaign.v1.json"
MANIFEST = load_manifest(MANIFEST_PATH)


class _Objectives:
    def __init__(self, mode: str):
        self.mode = mode

    def to_dict(self):
        return {
            "mode": self.mode,
            "objectives": [
                {"metric": "mAP50", "direction": "maximize"},
                {"metric": "latency_ms", "direction": "minimize"},
            ],
        }


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _successor_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    marker = tmp_path / "successor-ran.json"
    script = tmp_path / "successor.py"
    script.write_text(
        "import json, pathlib, sys, time\n"
        "pathlib.Path(sys.argv[1]).write_text("
        "json.dumps({'launched': True}) + '\\n', encoding='utf-8')\n"
        "time.sleep(0.1)\n",
        encoding="utf-8",
    )
    campaign_manifest = tmp_path / "deformable_detr_campaign.v1.json"
    _write_json(
        campaign_manifest,
        {
            "schema_version": 1,
            "model": "deformable_detr",
            "execution_kind": "direct_full_search",
            "cpu_runs": 0,
            "smoke_runs": 0,
        },
    )
    return script, marker, campaign_manifest


def _descriptor(
    tmp_path: Path,
    runtime_root: Path,
    *,
    script: Path,
    marker: Path,
    campaign_manifest: Path,
) -> tuple[Path, dict]:
    process_identity = _process_identity(os.getpid())
    assert process_identity is not None
    executable = Path(sys.executable).resolve()
    payload = {
        "schema_version": 1,
        "predecessor": {
            "campaign_id": MANIFEST["campaign_id"],
            "manifest_path": str(MANIFEST_PATH),
            "manifest_file_sha256": sha256_file(MANIFEST_PATH),
            "manifest_sha256": MANIFEST["manifest_sha256"],
            "runtime_root": str(runtime_root),
            "required_modes": list(MODES),
            "controller_process": process_identity,
        },
        "successor": {
            "name": "direct-full-deformable-detr",
            "model": "deformable_detr",
            "execution_kind": "direct_full_search",
            "cpu_runs": 0,
            "smoke_runs": 0,
            "manifest_path": str(campaign_manifest),
            "manifest_file_sha256": sha256_file(campaign_manifest),
            "working_directory": str(tmp_path),
            "command": [
                str(executable),
                str(script),
                str(marker),
                "--acknowledge-direct-full-dataset",
            ],
            "environment": {
                "HOME": str(tmp_path),
                "LANG": "C.UTF-8",
                "PATH": "/usr/bin:/bin",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            "completion_artifact": str(marker),
            "required_files": [
                {
                    "path": str(executable),
                    "sha256": sha256_file(executable),
                },
                {
                    "path": str(script),
                    "sha256": sha256_file(script),
                },
                {
                    "path": str(campaign_manifest),
                    "sha256": sha256_file(campaign_manifest),
                },
            ],
        },
    }
    descriptor = {
        **payload,
        "descriptor_sha256": canonical_sha256(payload),
    }
    path = tmp_path / "successor.v1.json"
    _write_json(path, descriptor)
    return path, descriptor


def _objective_values(rec_id: int) -> dict[str, float]:
    latency = 50.0 + rec_id
    return {
        "mAP50": 0.40 + 0.02 * rec_id,
        "latency_ms": latency,
        "latency_p95_ms": latency + 1.0,
        "latency_ci95_low_ms": latency - 0.1,
        "latency_ci95_high_ms": latency + 0.1,
    }


def _visible_history(mode: str, rec_id: int) -> list[dict]:
    values = []
    for previous in range(rec_id):
        specs = {"model.enc_layers": 3 + previous % 4}
        values.append(
            {
                "candidate_id": str(previous),
                "candidate_fingerprint": (
                    __import__(
                        "tao_automl.selection",
                        fromlist=["canonical_spec_fingerprint"],
                    ).canonical_spec_fingerprint(specs)
                ),
                "status": "success",
                "result": _objective_values(previous)["mAP50"],
                "objective_score": None,
                "objective_values": _objective_values(previous),
                "failure_reason": None,
            }
        )
    return values


def _recommendation_audit(mode: str, rec_id: int) -> dict:
    specs = {"model.enc_layers": 3 + rec_id % 4}
    model_based = rec_id >= MANIFEST["search"]["calibration_points"]
    method = (
        EXPECTED_MODEL_BASED_METHOD[mode]
        if model_based
        else "deterministic_low_discrepancy_design"
    )
    stage = "model_based" if model_based else "calibration"
    decision = {
        "stage": stage,
        "active_method": method,
        "mode": mode,
        "uses_raw_objectives": True,
        "selector_score_used": False,
        "observation_count": rec_id,
    }
    if model_based:
        decision["optimization_direction"] = (
            EXPECTED_OPTIMIZATION_DIRECTION[mode]
        )
    return build_recommendation_audit(
        candidate_id=rec_id,
        specs=specs,
        algorithm="bayesian",
        search_seed=MANIFEST["search"]["search_seed"],
        search_space=[{"parameter": "model.enc_layers"}],
        custom_ranges={
            "model.enc_layers": {"valid_min": 3, "valid_max": 6}
        },
        objective_config=_Objectives(mode),
        visible_history=_visible_history(mode, rec_id),
        acquisition={
            "proposal": {
                "stage": stage,
                "acquisition_mode": mode,
                "decision_state": decision,
            }
        },
    )


def _latency_evidence(fingerprint: str, rec_id: int) -> dict:
    protocol = MANIFEST["latency_protocol"]
    contract = {
        "schema_version": 1,
        "warmup_iterations": protocol["warmup_iterations"],
        "timed_iterations": protocol["timed_iterations"],
        "repeated_rounds": protocol["repeated_rounds"],
        "tail_percentile": protocol["tail_percentile"],
        "bootstrap_resamples": protocol["bootstrap_resamples"],
        "bootstrap_confidence_level": protocol[
            "bootstrap_confidence_level"
        ],
        "bootstrap_seed": protocol["bootstrap_seed"],
        "batch_size_per_replica": protocol["batch_size_per_replica"],
        "precision": protocol["precision"],
        "timed_scope": protocol["timed_scope"],
        "input_sha256": protocol["input_sha256"],
        "runtime_sha256": protocol["runtime_sha256"],
        "expected_replicas": protocol["expected_replicas"],
        "measurement_role": protocol["measurement_role"],
        "synchronization": protocol["synchronization"],
        "validity_thresholds": protocol["validity_thresholds"],
    }
    runtime_contract = {
        **MANIFEST["runtime"]["hardware_contract"],
        "python": "3.12.0",
        "torch": "2.8.0",
        "cuda": "12.8",
        "cudnn": 91002,
        "tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
    }
    latency = _objective_values(rec_id)["latency_ms"]
    statistics = {
        "median_ms": latency,
        "p95_ms": latency + 1.0,
        "mad_ms": 0.05,
        "iqr_ms": 0.10,
        "robust_cv": 0.001,
        "bootstrap_median_ci_ms": [latency - 0.1, latency + 0.1],
        "round_median_range_ms": 0.1,
        "round_drift_ms": 0.01,
        "device_median_range_ms": 0.1,
        "raw_sample_count_total": 4000,
        "samples_per_device": 500,
        "is_valid": True,
        "invalid_reasons": [],
    }
    aggregate_payload = {
        "schema_version": 1,
        "contract": contract,
        "contract_sha256": canonical_sha256(contract),
        "candidate_fingerprint": fingerprint,
        "hardware_sha256": canonical_sha256(runtime_contract),
        "replica_record_sha256": [
            f"{rank + 1:064x}" for rank in range(8)
        ],
        "statistics": statistics,
        "selection_isolation": SELECTION_TIME_ISOLATION,
    }
    aggregate = {
        **aggregate_payload,
        "aggregate_sha256": canonical_sha256(aggregate_payload),
    }
    outer_payload = {
        "schema_version": 1,
        "aggregate": aggregate,
        "input_evidence_sha256": "b" * 64,
        "rank_runtime_evidence": [
            {
                "hostname": f"node-{rank}",
                "local_rank": rank,
                "nvidia_smi": "redacted-test-evidence",
                **runtime_contract,
            }
            for rank in range(8)
        ],
    }
    return {
        **outer_payload,
        "evidence_sha256": canonical_sha256(outer_payload),
    }


def _candidate(mode: str, rec_id: int) -> dict:
    from tao_automl.selection import canonical_spec_fingerprint

    specs = {"model.enc_layers": 3 + rec_id % 4}
    fingerprint = canonical_spec_fingerprint(specs)
    return {
        "candidate_id": f"{mode}_rec_{rec_id}",
        "rec_id": str(rec_id),
        "specs": specs,
        "recommendation_audit": _recommendation_audit(mode, rec_id),
        "agent_intervention_flags": algorithmic_campaign_flags(),
        "status": "success",
        "train_job_id": f"{mode}-train-{rec_id}",
        "candidate_fingerprint": fingerprint,
        "objective_values": _objective_values(rec_id),
        "selection_time_latency": {
            "aggregate_evidence": _latency_evidence(
                fingerprint,
                rec_id,
            )
        },
        "matched_validation_selection_isolation_flags": dict(
            MANIFEST["selection_isolation_flags"]
        ),
    }


def _completed_runtime(runtime_root: Path, descriptor: dict) -> None:
    budget = MANIFEST["search"]["candidate_budget_per_mode"]
    _write_json(
        runtime_root / "mode_process_status.json",
        {mode: 0 for mode in MODES},
    )
    for mode in MODES:
        candidates = {
            f"{mode}_rec_{rec_id}": _candidate(mode, rec_id)
            for rec_id in range(budget)
        }
        history = [
            {
                "rec_id": rec_id,
                "specs": candidates[f"{mode}_rec_{rec_id}"]["specs"],
                "job_id": f"{mode}-train-{rec_id}",
                "metric": _objective_values(rec_id)["mAP50"],
                "objective_score": None,
                "objective_values": _objective_values(rec_id),
                "status": "success",
                "failure_reason": None,
                "adjustments": [],
                "selection_audit": None,
            }
            for rec_id in range(budget)
        ]
        archive = [
            {
                "id": item["rec_id"],
                "specs": item["specs"],
                "status": item["status"],
                "objective_values": item["objective_values"],
            }
            for item in history
        ]
        analysis = analyze_archive(
            archive,
            _selection_config(MANIFEST, mode),
        ).to_dict()
        winner_id = int(analysis["selections"][mode]["winner_id"])
        result = {
            "best": {
                "rec_id": winner_id,
                "specs": history[winner_id]["specs"],
                "metric_value": history[winner_id]["metric"],
                "objective_score": None,
                "objective_values": history[winner_id]["objective_values"],
                "adjustments": [],
            },
            "progress": {
                "completed": budget,
                "total": budget,
            },
            "history": history,
            "pareto_front": [],
            "selection_analysis": analysis,
        }
        _write_json(
            runtime_root / mode / "result.json",
            {
                "schema_version": 1,
                "manifest_sha256": descriptor["predecessor"][
                    "manifest_sha256"
                ],
                "mode": mode,
                "status": "success",
                "result": result,
            },
        )
        _write_json(
            runtime_root / mode / "candidate_evidence.json",
            {
                "schema_version": 1,
                "manifest_sha256": descriptor["predecessor"][
                    "manifest_sha256"
                ],
                "mode": mode,
                "candidates": candidates,
            },
        )


def _fixture(tmp_path: Path):
    runtime_root = tmp_path / "runtime"
    script, marker, campaign_manifest = _successor_files(tmp_path)
    descriptor_path, descriptor = _descriptor(
        tmp_path,
        runtime_root,
        script=script,
        marker=marker,
        campaign_manifest=campaign_manifest,
    )
    return (
        runtime_root,
        script,
        marker,
        campaign_manifest,
        descriptor_path,
        descriptor,
    )


def test_completed_gate_triggers_successor_exactly_once(tmp_path):
    (
        runtime_root,
        _script,
        marker,
        _campaign_manifest,
        descriptor_path,
        descriptor,
    ) = _fixture(tmp_path)
    _completed_runtime(runtime_root, descriptor)

    report = validate_completed_dino(descriptor, runtime_root)
    assert report["status"] == "passed"
    assert set(report["modes"]) == set(MODES)
    assert all(
        item["model_based_candidate_ids"]
        == [str(index) for index in range(8, 20)]
        for item in report["modes"].values()
    )

    assert (
        watch_and_trigger(
            descriptor_path,
            runtime_root,
            poll_seconds=0.001,
        )
        == 0
    )
    assert json.loads(marker.read_text()) == {"launched": True}
    state = json.loads(
        (
            runtime_root
            / "automatic_successor"
            / "automatic_successor_state.json"
        ).read_text()
    )
    assert state["status"] == "successor_completed"
    decision = json.loads(
        (
            runtime_root
            / "automatic_successor"
            / "gate_decision.json"
        ).read_text()
    )
    assert decision["successor_submitted"] is True

    with pytest.raises(
        AutomaticSuccessorError,
        match="already has terminal or running state",
    ):
        trigger_successor(
            descriptor,
            state_dir=runtime_root / "automatic_successor",
            gate_report=report,
        )


def test_waiting_gate_does_not_launch_successor(tmp_path):
    (
        runtime_root,
        _script,
        marker,
        _campaign_manifest,
        descriptor_path,
        _descriptor_value,
    ) = _fixture(tmp_path)
    runtime_root.mkdir()

    assert (
        watch_and_trigger(
            descriptor_path,
            runtime_root,
            poll_seconds=0.001,
            once=True,
        )
        == 3
    )
    assert not marker.exists()


def test_dead_controller_blocks_before_terminal_marker(tmp_path):
    (
        runtime_root,
        _script,
        marker,
        _campaign_manifest,
        descriptor_path,
        descriptor,
    ) = _fixture(tmp_path)
    runtime_root.mkdir()
    descriptor["predecessor"]["controller_process"]["start_time_ticks"] += 1
    payload = {
        key: value
        for key, value in descriptor.items()
        if key != "descriptor_sha256"
    }
    descriptor["descriptor_sha256"] = canonical_sha256(payload)
    _write_json(descriptor_path, descriptor)

    with pytest.raises(
        AutomaticSuccessorError,
        match="controller exited or its sealed process identity changed",
    ):
        watch_and_trigger(
            descriptor_path,
            runtime_root,
            poll_seconds=0.001,
            once=True,
        )
    assert not marker.exists()


def test_selector_policy_tampering_blocks_without_launch(tmp_path):
    (
        runtime_root,
        _script,
        marker,
        _campaign_manifest,
        descriptor_path,
        descriptor,
    ) = _fixture(tmp_path)
    _completed_runtime(runtime_root, descriptor)
    result_path = runtime_root / "latency" / "result.json"
    result = json.loads(result_path.read_text())
    result["result"]["selection_analysis"]["selections"]["latency"][
        "winner_id"
    ] = "19"
    result["result"]["best"]["rec_id"] = 19
    _write_json(result_path, result)

    with pytest.raises(
        AutomaticSuccessorError,
        match="selector evidence differs from production replay",
    ):
        watch_and_trigger(
            descriptor_path,
            runtime_root,
            poll_seconds=0.001,
        )
    assert not marker.exists()


def test_latency_provenance_tampering_blocks_without_launch(tmp_path):
    (
        runtime_root,
        _script,
        marker,
        _campaign_manifest,
        descriptor_path,
        descriptor,
    ) = _fixture(tmp_path)
    _completed_runtime(runtime_root, descriptor)
    evidence_path = runtime_root / "accuracy" / "candidate_evidence.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["candidates"]["accuracy_rec_0"]["objective_values"][
        "latency_ms"
    ] += 1.0
    _write_json(evidence_path, evidence)

    with pytest.raises(
        AutomaticSuccessorError,
        match="differs from runner history",
    ):
        watch_and_trigger(
            descriptor_path,
            runtime_root,
            poll_seconds=0.001,
        )
    assert not marker.exists()


def test_tampered_successor_input_is_rejected(tmp_path):
    (
        runtime_root,
        script,
        marker,
        _campaign_manifest,
        descriptor_path,
        descriptor,
    ) = _fixture(tmp_path)
    _completed_runtime(runtime_root, descriptor)
    script.write_text("# changed after sealing\n", encoding="utf-8")

    with pytest.raises(
        AutomaticSuccessorError,
        match="successor input identity changed",
    ):
        watch_and_trigger(
            descriptor_path,
            runtime_root,
            poll_seconds=0.001,
        )
    assert not marker.exists()


def test_descriptor_is_content_addressed(tmp_path):
    (
        _runtime_root,
        _script,
        _marker,
        _campaign_manifest,
        descriptor_path,
        descriptor,
    ) = _fixture(tmp_path)
    descriptor["successor"]["name"] = "result-driven-change"
    _write_json(descriptor_path, descriptor)

    with pytest.raises(
        AutomaticSuccessorError,
        match="descriptor integrity verification failed",
    ):
        validate_successor_descriptor(descriptor_path)
