"""Contract tests for evaluator-only Mask Grounding DINO recovery."""

import json

from . import metric_recovery_campaign as recovery


def test_frozen_predecessor_yields_exact_four_checkpoint_cohort():
    completion = json.loads(
        recovery.DEFAULT_PREDECESSOR_COMPLETION.read_text(encoding="utf-8")
    )

    records = recovery._checkpoint_records(completion)

    assert len(records) == 4
    assert [record["checkpoint_id"] for record in records] == sorted(
        record["checkpoint_id"] for record in records
    )
    assert all(
        record["terminal_checkpoint"]["terminal_epoch_index"] == 2
        and record["terminal_checkpoint"]["training_epochs"] == 3
        for record in records
    )


def test_overlay_command_is_fail_closed_and_does_not_mutate_base():
    command = recovery.overlay_install_command(
        {"overlay": recovery.OVERLAY}
    )

    assert "sha256sum" in command
    assert recovery.OVERLAY["archive_sha256"] in command
    assert "cp -as" in command
    assert "install_overlay.py" in command
    assert "export PYTHONPATH=" in command
    assert (
        f'{recovery.OVERLAY["base_site_packages"]}/nvidia_tao_pytorch/.'
        in command
    )


def test_workflow_submits_evaluation_only_with_frozen_checkpoint(
    monkeypatch, tmp_path
):
    checkpoint = {
        "path": "/lustre/frozen/model_epoch_002_step_10992.pth",
        "sha256": "a" * 64,
        "size_bytes": 123,
        "terminal_epoch_index": 2,
        "training_epochs": 3,
    }
    record = {
        "checkpoint_id": "mask_grounding_dino.test",
        "source_train_job_id": "train-job",
        "failed_evaluation_job_id": "old-eval-job",
        "terminal_checkpoint": checkpoint,
    }
    contract = {
        "overlay": recovery.OVERLAY,
        "runtime": {},
    }
    called = []

    monkeypatch.setattr(
        recovery.qualification_campaign,
        "_qualification_specs",
        lambda *_: ({"train": {}}, {"evaluate": {"checkpoint": ""}}),
    )

    def entrypoint(_contract, action, specification):
        called.append((action, specification["evaluate"]["checkpoint"]))
        return "tao model evaluate -e {config_path}", "b" * 64

    monkeypatch.setattr(recovery.qualification_campaign, "_entrypoint", entrypoint)

    class Job:
        id = "new-eval-job"

    submitted = []

    def submit(_sdk, _contract, command):
        submitted.append(command)
        return Job()

    monkeypatch.setattr(recovery.qualification_campaign, "_submit", submit)
    monkeypatch.setattr(recovery.run_campaign, "_wait_for_job", lambda *_a, **_k: "Complete")
    monkeypatch.setattr(
        recovery.qualification_campaign,
        "_status_values",
        lambda _sdk, _job, *, action, names: (
            [0.11] if names[0].startswith("[segm]") else [0.22]
        ),
    )

    result = recovery._run_one(contract, record, tmp_path, object())

    assert called == [("evaluate", checkpoint["path"])]
    assert len(submitted) == 1
    assert "install_overlay.py" in submitted[0]
    assert " evaluate " in submitted[0]
    assert " train " not in submitted[0]
    assert result["status"] == "success"
    assert result["training_jobs_submitted"] == 0
    assert result["segm_val_mAP50_95"] == 0.11
    assert result["bbox_val_mAP50_95"] == 0.22
