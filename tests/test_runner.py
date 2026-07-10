# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the migrated AutoMLRunner.

Validates:
- SkillContext loads skill_info.yaml and spec_template_<action>.yaml.
- AutoMLRunner.__init__ takes (sdk, skill_dir, action) — no SkillBank.
- _make_sdk returns the right per-platform SDK class; rejects bad names.
- _run_one_job calls build_entrypoint with action_cfg fields and
  sdk.create_job with the resulting command + platform_kwargs.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# SkillContext
# ---------------------------------------------------------------------------

def _write_fake_skill(tmp_path: Path, action: str = "train") -> Path:
    """Create a minimal skill bank layout for tests."""
    skill_dir = tmp_path / "models" / "fake-net"
    refs = skill_dir / "references"
    refs.mkdir(parents=True)
    (refs / "skill_info.yaml").write_text(
        "network_arch: fake-net\n"
        "container_image: nvcr.io/nvidia/tao/fake:0.1\n"
        "data_format: coco\n"
        "actions:\n"
        f"  {action}:\n"
        "    command: fake train -e {config_path}\n"
        "    config_format: yaml\n"
        "    inputs:\n"
        "      dataset.train_data_sources[0].image_dir:\n"
        "        type: file\n"
        "    outputs:\n"
        "      results_dir:\n"
        "        type: folder\n"
        "    upload_excludes: ['inputs/']\n"
    )
    (refs / f"spec_template_{action}.yaml").write_text(
        "train:\n"
        "  num_epochs: 12\n"
        "  optim:\n"
        "    lr: 2.0e-4\n"
        "dataset:\n"
        "  num_classes: 80\n"
    )
    return skill_dir


def _write_fake_action_skill(
    tmp_path: Path,
    *,
    network_arch: str = "dino",
    action: str = "quantize",
) -> Path:
    """Create a minimal non-train action skill that can run a shell command."""
    skill_dir = tmp_path / "models" / f"fake-{action}"
    refs = skill_dir / "references"
    refs.mkdir(parents=True)
    (refs / "skill_info.yaml").write_text(
        f"network_arch: {network_arch}\n"
        "container_image: nvcr.io/nvidia/tao/fake:0.1\n"
        "actions:\n"
        f"  {action}:\n"
        "    command: >-\n"
        "      python -c \"print('action_metric: 0.73')\"\n"
        "    config_format: yaml\n"
        "    inputs: {}\n"
        "    outputs: {}\n"
        "    upload_excludes: []\n"
    )
    results_dir = tmp_path / "action-results"
    (refs / f"spec_template_{action}.yaml").write_text(
        f"results_dir: {results_dir}\n"
        "train:\n"
        "  optim:\n"
        "    lr: 2.0e-4\n"
        "quantize:\n"
        f"  results_dir: {results_dir}\n"
    )
    return skill_dir


class _CompletedProcessSDK:
    """SDK shim that executes submitted jobs as real local subprocesses."""

    def __init__(self):
        self.jobs = {}

    def create_job(self, image, command, **kwargs):
        from tao_sdk.models import Job

        job_id = f"job-{len(self.jobs)}"
        env = dict(**kwargs.pop("env_vars", {}))
        proc_env = None
        if env:
            import os
            proc_env = os.environ.copy()
            proc_env.update(env)
        if proc_env is None:
            import os
            proc_env = os.environ.copy()
        proc_env["TAO_JOB_ID"] = job_id

        completed = subprocess.run(
            command,
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=proc_env,
            check=False,
        )
        self.jobs[job_id] = completed
        return Job(
            id=job_id,
            network_arch="dino",
            action="quantize",
            workspace_id="local",
            backend_job_id=job_id,
            status="Complete" if completed.returncode == 0 else "Error",
        )

    def get_job_status(self, job_id):
        from tao_sdk.models import JobStatus

        completed = self.jobs[job_id]
        status = "Complete" if completed.returncode == 0 else "Error"
        return JobStatus(job_id=job_id, status=status)

    def get_job_logs(self, job_id, tail=None):
        logs = self.jobs[job_id].stdout or ""
        if tail is not None:
            return "\n".join(logs.splitlines()[-tail:])
        return logs


def _write_python_skill(tmp_path: Path) -> Path:
    """Create an external model skill backed by a direct Python script."""
    skill_dir = tmp_path / "models" / "public-random-forest"
    refs = skill_dir / "references"
    schemas = skill_dir / "schemas"
    scripts = skill_dir / "scripts"
    refs.mkdir(parents=True)
    schemas.mkdir()
    scripts.mkdir()
    (scripts / "train.py").write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--config', required=True)\n"
        "parser.parse_args()\n"
        "print('accuracy: 0.875', flush=True)\n"
    )
    (refs / "skill_info.yaml").write_text(
        "network_arch: public_random_forest\n"
        "actions:\n"
        "  train:\n"
        "    config_format: json\n"
        "    execution:\n"
        "      type: python_script\n"
        "      script: scripts/train.py\n"
        "      args: [--config, '{config_path}']\n"
        "      cwd: .\n"
        "    outputs:\n"
        "      results_dir:\n"
        "        type: folder\n"
    )
    default_specs = {
        "model": {"n_estimators": 10, "max_depth": 3},
        "results_dir": "",
    }
    (refs / "spec_template_train.yaml").write_text(
        "model:\n"
        "  n_estimators: 10\n"
        "  max_depth: 3\n"
        "results_dir: ''\n"
    )
    (schemas / "train.schema.json").write_text(json.dumps({
        "type": "object",
        "default": default_specs,
        "properties": {
            "model": {
                "type": "object",
                "properties": {
                    "n_estimators": {
                        "type": "integer", "default": 10,
                        "minimum": 2, "maximum": 20,
                        "automl_enabled": True,
                    },
                    "max_depth": {
                        "type": "integer", "default": 3,
                        "minimum": 1, "maximum": 6,
                        "automl_enabled": True,
                    },
                },
            },
            "results_dir": {"type": "string", "default": ""},
        },
    }))
    return skill_dir


def test_skill_context_loads_skill_info_and_template(tmp_path):
    from tao_automl.runner import SkillContext
    skill_dir = _write_fake_skill(tmp_path)
    ctx = SkillContext(skill_dir=skill_dir, action="train")
    assert ctx.network_arch == "fake-net"
    assert ctx.action_cfg["command"] == "fake train -e {config_path}"
    assert ctx.action_cfg["config_format"] == "yaml"
    assert ctx.default_specs["train"]["num_epochs"] == 12
    assert ctx.default_specs["dataset"]["num_classes"] == 80


def test_skill_context_action_container_image_overrides_model_image(tmp_path):
    from tao_automl.runner import SkillContext
    skill_dir = _write_fake_skill(tmp_path, action="dataset_convert")
    info_path = skill_dir / "references/skill_info.yaml"
    info_path.write_text(
        info_path.read_text().replace(
            "    command: fake train -e {config_path}\n",
            "    container_image: nvcr.io/nvidia/tao/fake-ds:0.1\n"
            "    command: fake convert -e {config_path}\n",
        )
    )
    ctx = SkillContext(skill_dir=skill_dir, action="dataset_convert")
    assert ctx.container_image == "nvcr.io/nvidia/tao/fake-ds:0.1"


def test_skill_context_no_template_yields_empty_specs(tmp_path):
    """Models without a spec_template_<action>.yaml get default_specs={}.
    Caller is responsible for constructing the spec from skill SKILL.md."""
    from tao_automl.runner import SkillContext
    skill_dir = _write_fake_skill(tmp_path)
    (skill_dir / "references/spec_template_train.yaml").unlink()
    ctx = SkillContext(skill_dir=skill_dir, action="train")
    assert ctx.default_specs == {}


def test_skill_context_missing_action_raises(tmp_path):
    from tao_automl.runner import SkillContext
    skill_dir = _write_fake_skill(tmp_path, action="train")
    with pytest.raises(KeyError, match="evaluate"):
        SkillContext(skill_dir=skill_dir, action="evaluate")


def test_skill_context_missing_skill_info_raises(tmp_path):
    from tao_automl.runner import SkillContext
    with pytest.raises(FileNotFoundError, match="skill_info.yaml"):
        SkillContext(skill_dir=tmp_path / "nonexistent", action="train")


def test_generate_hyperparams_uses_selected_action_schema(monkeypatch):
    """Non-train AutoML must read the action schema instead of train."""
    from tao_automl.search_space import params

    seen = {}

    def fake_generate_schema(network, action):
        seen["network"] = network
        seen["action"] = action
        return {
            "default": {
                "quantize": {"calibration_batches": 4},
            },
            "properties": {
                "quantize": {
                    "type": "object",
                    "properties": {
                        "calibration_batches": {
                            "type": "integer",
                            "default": 4,
                            "minimum": 1,
                            "maximum": 8,
                            "automl_enabled": True,
                        }
                    },
                }
            },
        }

    monkeypatch.setattr(params, "generate_schema", fake_generate_schema)

    records, names = params.generate_hyperparams_to_search(
        network="fake-net",
        action="quantize",
        train_specs={"quantize": {"calibration_batches": 4}},
        automl_hyperparameters=None,
    )

    assert seen == {"network": "fake-net", "action": "quantize"}
    assert names == ["quantize.calibration_batches"]
    assert records[0]["parameter"] == "quantize.calibration_batches"


def test_quantize_mode_search_filters_static_ptq_for_fixed_torchao_backend(monkeypatch):
    """TorchAO validates only weight_only_ptq, so AutoML should not suggest static_ptq."""
    from tao_automl.search_space import params

    def fake_generate_schema(network, action):
        return {
            "default": {
                "quantize": {
                    "backend": "torchao",
                    "mode": "weight_only_ptq",
                },
            },
            "properties": {
                "quantize": {
                    "type": "object",
                    "properties": {
                        "backend": {
                            "type": "categorical",
                            "default": "torchao",
                            "enum": ["modelopt.pytorch", "torchao", "modelopt.onnx"],
                        },
                        "mode": {
                            "type": "categorical",
                            "default": "weight_only_ptq",
                            "enum": ["static_ptq", "weight_only_ptq"],
                        },
                    },
                },
            },
        }

    monkeypatch.setattr(params, "generate_schema", fake_generate_schema)

    records, names = params.generate_hyperparams_to_search(
        network="fake-net",
        action="quantize",
        train_specs={"quantize": {"backend": "torchao", "mode": "weight_only_ptq"}},
        automl_hyperparameters=["quantize.mode"],
    )

    assert names == ["quantize.mode"]
    assert records[0]["valid_options"] == ["weight_only_ptq"]


def test_quantize_algorithm_search_filters_invalid_modelopt_static_ptq_algorithms(monkeypatch):
    """ModelOpt PyTorch static PTQ should not suggest unregistered algorithms."""
    from tao_automl.search_space import params

    def fake_generate_schema(network, action):
        return {
            "default": {
                "quantize": {
                    "backend": "modelopt.pytorch",
                    "mode": "static_ptq",
                    "algorithm": "max",
                },
            },
            "properties": {
                "quantize": {
                    "type": "object",
                    "properties": {
                        "backend": {
                            "type": "categorical",
                            "default": "modelopt.pytorch",
                            "enum": ["modelopt.pytorch", "torchao", "modelopt.onnx"],
                        },
                        "mode": {
                            "type": "categorical",
                            "default": "static_ptq",
                            "enum": ["static_ptq", "weight_only_ptq"],
                        },
                        "algorithm": {
                            "type": "categorical",
                            "default": "max",
                            "enum": [
                                "minmax",
                                "max",
                                "entropy",
                                "awq_clip",
                                "awq_lite",
                                "awq_full",
                                "rtn_dq",
                            ],
                        },
                    },
                },
            },
        }

    monkeypatch.setattr(params, "generate_schema", fake_generate_schema)

    records, names = params.generate_hyperparams_to_search(
        network="fake-net",
        action="quantize",
        train_specs={
            "quantize": {
                "backend": "modelopt.pytorch",
                "mode": "static_ptq",
                "algorithm": "max",
            }
        },
        automl_hyperparameters=["quantize.algorithm"],
    )

    assert names == ["quantize.algorithm"]
    assert records[0]["valid_options"] == ["max", "awq_lite", "awq_full"]


def test_quantize_mode_and_algorithm_search_filters_for_fixed_torchao_backend(monkeypatch):
    """Searching mode and algorithm together should stay valid for TorchAO."""
    from tao_automl.search_space import params

    def fake_generate_schema(network, action):
        return {
            "default": {
                "quantize": {
                    "backend": "torchao",
                    "mode": "weight_only_ptq",
                    "algorithm": "minmax",
                },
            },
            "properties": {
                "quantize": {
                    "type": "object",
                    "properties": {
                        "backend": {
                            "type": "categorical",
                            "default": "torchao",
                            "enum": ["modelopt.pytorch", "torchao", "modelopt.onnx"],
                        },
                        "mode": {
                            "type": "categorical",
                            "default": "weight_only_ptq",
                            "enum": ["static_ptq", "weight_only_ptq"],
                        },
                        "algorithm": {
                            "type": "categorical",
                            "default": "minmax",
                            "enum": [
                                "minmax",
                                "max",
                                "entropy",
                                "awq_clip",
                                "awq_lite",
                                "awq_full",
                                "rtn_dq",
                            ],
                        },
                    },
                },
            },
        }

    monkeypatch.setattr(params, "generate_schema", fake_generate_schema)

    records, names = params.generate_hyperparams_to_search(
        network="fake-net",
        action="quantize",
        train_specs={
            "quantize": {
                "backend": "torchao",
                "mode": "weight_only_ptq",
                "algorithm": "minmax",
            }
        },
        automl_hyperparameters=["quantize.mode", "quantize.algorithm"],
    )

    ranges = {record["parameter"]: record["valid_options"] for record in records}
    assert names == ["quantize.mode", "quantize.algorithm"]
    assert ranges["quantize.mode"] == ["weight_only_ptq"]
    assert ranges["quantize.algorithm"] == ["minmax"]


def test_quantize_mode_and_algorithm_search_filters_for_fixed_modelopt_backend(monkeypatch):
    """Searching mode and algorithm together should stay valid for ModelOpt PyTorch."""
    from tao_automl.search_space import params

    def fake_generate_schema(network, action):
        return {
            "default": {
                "quantize": {
                    "backend": "modelopt.pytorch",
                    "mode": "static_ptq",
                    "algorithm": "max",
                },
            },
            "properties": {
                "quantize": {
                    "type": "object",
                    "properties": {
                        "backend": {
                            "type": "categorical",
                            "default": "modelopt.pytorch",
                            "enum": ["modelopt.pytorch", "torchao", "modelopt.onnx"],
                        },
                        "mode": {
                            "type": "categorical",
                            "default": "static_ptq",
                            "enum": ["static_ptq", "weight_only_ptq"],
                        },
                        "algorithm": {
                            "type": "categorical",
                            "default": "max",
                            "enum": [
                                "minmax",
                                "max",
                                "entropy",
                                "awq_clip",
                                "awq_lite",
                                "awq_full",
                                "rtn_dq",
                            ],
                        },
                    },
                },
            },
        }

    monkeypatch.setattr(params, "generate_schema", fake_generate_schema)

    records, names = params.generate_hyperparams_to_search(
        network="fake-net",
        action="quantize",
        train_specs={
            "quantize": {
                "backend": "modelopt.pytorch",
                "mode": "static_ptq",
                "algorithm": "max",
            }
        },
        automl_hyperparameters=["quantize.mode", "quantize.algorithm"],
    )

    ranges = {record["parameter"]: record["valid_options"] for record in records}
    assert names == ["quantize.mode", "quantize.algorithm"]
    assert ranges["quantize.mode"] == ["static_ptq"]
    assert ranges["quantize.algorithm"] == ["max", "awq_lite", "awq_full"]


def test_non_train_action_defaults_include_action_native_params():
    from tao_automl.schema.generate_schema import generate_schema
    from tao_automl.search_space import params

    distill_schema = generate_schema("classification_pyt", "distill")
    _distill_records, distill_names = params.generate_hyperparams_to_search(
        network="classification_pyt",
        action="distill",
        train_specs=distill_schema["default"],
        automl_hyperparameters=None,
    )
    assert {
        "distill.loss_type",
        "distill.loss_lambda",
        "distill.mode",
        "distill.use_mlp",
        "distill.mlp_hidden_size",
        "distill.mlp_num_inner",
    }.issubset(set(distill_names))

    prune_schema = generate_schema("ocrnet", "prune")
    _prune_records, prune_names = params.generate_hyperparams_to_search(
        network="ocrnet",
        action="prune",
        train_specs=prune_schema["default"],
        automl_hyperparameters=None,
    )
    assert {
        "prune.prune_setting.mode",
        "prune.prune_setting.amount",
        "prune.prune_setting.granularity",
        "prune.prune_setting.raw_prune_score",
    }.issubset(set(prune_names))

    quantize_schema = generate_schema("classification_pyt", "quantize")
    quantize_spec = quantize_schema["default"]
    quantize_spec["quantize"]["backend"] = "torchao"
    quantize_spec["quantize"]["mode"] = "weight_only_ptq"
    _quantize_records, quantize_names = params.generate_hyperparams_to_search(
        network="classification_pyt",
        action="quantize",
        train_specs=quantize_spec,
        automl_hyperparameters=None,
    )
    assert "quantize.mode" in quantize_names
    assert "quantize.algorithm" in quantize_names
    assert "quantize.backend" not in quantize_names


def test_cosmos_evaluate_defaults_include_autoprompt_search_space():
    from tao_automl.schema.generate_schema import generate_schema
    from tao_automl.search_space import params

    schema = generate_schema("cosmos-rl", "evaluate")
    records, names = params.generate_hyperparams_to_search(
        network="cosmos-rl",
        action="evaluate",
        train_specs=schema["default"],
        automl_hyperparameters=None,
    )
    records_by_name = {record["parameter"]: record for record in records}

    expected = {
        "dataset.system_prompt",
        "vision.nframes",
        "generation.max_tokens",
        "generation.temperature",
        "generation.repetition_penalty",
        "generation.presence_penalty",
        "generation.frequency_penalty",
    }
    assert expected.issubset(set(names))
    assert "vision.fps" not in names

    prompt_record = records_by_name["dataset.system_prompt"]
    assert prompt_record["value_type"] == "categorical"
    assert len(prompt_record["valid_options"]) >= 4
    assert prompt_record["default_value"] in prompt_record["valid_options"]

    assert records_by_name["vision.nframes"]["value_type"] == "ordered_int"
    assert records_by_name["vision.nframes"]["valid_options"] == [4, 8]


def test_custom_valid_options_cannot_reopen_schema_excluded_options():
    from tao_automl.utils.math_utils import get_valid_options

    options = get_valid_options(
        {
            "parameter": "quantize.mode",
            "valid_options": ["weight_only_ptq"],
        },
        {
            "quantize.mode": {
                "valid_options": ["weight_only_ptq", "static_ptq"],
            }
        },
    )

    assert options == ["weight_only_ptq"]


def test_custom_valid_options_cannot_reopen_invalid_modelopt_algorithms(monkeypatch, tmp_path):
    import json

    from tao_automl import AutoML
    from tao_automl.search_space import params

    def fake_generate_hyperparams_to_search(**kwargs):
        return [
            {
                "parameter": "quantize.algorithm",
                "value_type": "categorical",
                "default_value": "max",
                "valid_min": "",
                "valid_max": "",
                "valid_options": ["max", "awq_lite", "awq_full"],
                "option_weights": None,
                "math_cond": "",
                "parent_param": "",
                "depends_on": "",
            }
        ], ["quantize.algorithm"]

    monkeypatch.setattr(
        params,
        "generate_hyperparams_to_search",
        fake_generate_hyperparams_to_search,
    )

    AutoML(
        workspace=str(tmp_path),
        network="classification_pyt",
        train_specs={
            "quantize": {
                "backend": "modelopt.pytorch",
                "mode": "static_ptq",
                "algorithm": "max",
            }
        },
        settings={
            "algorithm": "bayesian",
            "metric": "val_acc_1",
            "direction": "maximize",
            "automl_max_recommendations": 1,
            "session_id": "fixedsession",
        },
        automl_hyperparameters=["quantize.algorithm"],
        custom_param_ranges={
            "quantize.algorithm": {
                "valid_options": ["minmax", "max", "entropy", "awq_lite"],
            }
        },
        action="quantize",
    )

    ranges = json.loads(
        (tmp_path / ".automl/custom_ranges/fixedsession.json").read_text()
    )
    assert ranges["quantize.algorithm"]["valid_options"] == ["max", "awq_lite"]


def test_automl_persists_sanitized_custom_valid_options(monkeypatch, tmp_path):
    import json

    from tao_automl import AutoML
    from tao_automl.search_space import params

    def fake_generate_hyperparams_to_search(**kwargs):
        return [
            {
                "parameter": "quantize.mode",
                "value_type": "categorical",
                "default_value": "weight_only_ptq",
                "valid_min": "",
                "valid_max": "",
                "valid_options": ["weight_only_ptq"],
                "option_weights": None,
                "math_cond": "",
                "parent_param": "",
                "depends_on": "",
            }
        ], ["quantize.mode"]

    monkeypatch.setattr(
        params,
        "generate_hyperparams_to_search",
        fake_generate_hyperparams_to_search,
    )

    AutoML(
        workspace=str(tmp_path),
        network="classification_pyt",
        train_specs={"quantize": {"backend": "torchao", "mode": "weight_only_ptq"}},
        settings={
            "algorithm": "bayesian",
            "metric": "val_acc_1",
            "direction": "maximize",
            "automl_max_recommendations": 1,
            "session_id": "fixedsession",
        },
        automl_hyperparameters=["quantize.mode"],
        custom_param_ranges={
            "quantize.mode": {
                "valid_options": ["weight_only_ptq", "static_ptq"],
            }
        },
        action="quantize",
    )

    ranges = json.loads(
        (tmp_path / ".automl/custom_ranges/fixedsession.json").read_text()
    )
    assert ranges["quantize.mode"]["valid_options"] == ["weight_only_ptq"]


def test_skill_context_resolves_python_script_execution_and_external_schema(tmp_path):
    from tao_automl.runner import SkillContext

    skill_dir = _write_python_skill(tmp_path)
    ctx = SkillContext(skill_dir=skill_dir, action="train")

    assert ctx.container_image == ""
    assert ctx.execution.script == (skill_dir / "scripts/train.py").resolve()
    assert ctx.execution.script_args == ("--config", "{config_path}")
    assert ctx.execution.config_format == "json"
    assert ctx.execution.cwd == skill_dir.resolve()
    assert ctx.schema["default"]["model"]["n_estimators"] == 10


def test_skill_context_rejects_missing_python_script(tmp_path):
    from tao_automl.runner import SkillContext

    skill_dir = _write_python_skill(tmp_path)
    (skill_dir / "scripts/train.py").unlink()

    with pytest.raises(FileNotFoundError, match="Python action script not found"):
        SkillContext(skill_dir=skill_dir, action="train")


def test_skill_context_requires_external_schema_for_python_script(tmp_path):
    from tao_automl.runner import SkillContext

    skill_dir = _write_python_skill(tmp_path)
    (skill_dir / "schemas/train.schema.json").unlink()

    with pytest.raises(FileNotFoundError, match="require an external AutoML schema"):
        SkillContext(skill_dir=skill_dir, action="train")


@pytest.mark.parametrize(
    ("schema", "message"),
    [
        ({}, "non-empty 'properties'"),
        ({"type": "object", "default": [], "properties": {"x": {"type": "integer"}}},
         "'default' must be a JSON object"),
        ({"type": "object", "properties": {"model": {"type": "object", "properties": {}}}},
         "requires non-empty nested properties"),
        ({"type": "object", "properties": {"x": {"anyOf": True}}},
         "'anyOf' must be a non-empty list"),
        ({"type": "object", "properties": {"x": {"type": ["integer", "null"]}}},
         "list-valued 'type'"),
        ({"type": "object", "properties": {"x": {
            "type": "integer", "default": True, "minimum": 1,
            "maximum": 2, "automl_enabled": True,
        }}}, "expected an integer"),
        ({"type": "object", "properties": {"x": {
            "type": "number", "default": 1.0, "minimum": 2.0,
            "maximum": 1.0, "automl_enabled": True,
        }}}, "minimum cannot exceed maximum"),
        ({"type": "object", "properties": {"x": {
            "type": "string", "default": "x", "automl_enabled": True,
        }}}, "requires an enum"),
        ({"type": "object", "properties": {"x": {
            "anyOf": [{"type": "integer"}, {"type": "number"}],
            "automl_enabled": True,
        }}}, "exactly one non-null anyOf type"),
    ],
)
def test_skill_context_rejects_malformed_python_search_schema(
    tmp_path, schema, message,
):
    from tao_automl.runner import SkillContext

    skill_dir = _write_python_skill(tmp_path)
    (skill_dir / "schemas/train.schema.json").write_text(json.dumps(schema))

    with pytest.raises((TypeError, ValueError), match=message):
        SkillContext(skill_dir=skill_dir, action="train")


def test_optional_anyof_schema_uses_non_null_search_type(tmp_path):
    from tao_automl.runner import SkillContext
    from tao_automl.search_space.params import generate_hyperparams_to_search

    skill_dir = _write_python_skill(tmp_path)
    schema_path = skill_dir / "schemas/train.schema.json"
    schema = json.loads(schema_path.read_text())
    n_estimators = schema["properties"]["model"]["properties"]["n_estimators"]
    n_estimators.pop("type")
    n_estimators["anyOf"] = [{"type": "null"}, {"type": "integer"}]
    n_estimators["properties"] = {}
    schema_path.write_text(json.dumps(schema))

    ctx = SkillContext(skill_dir=skill_dir, action="train")
    _, names = generate_hyperparams_to_search(
        ctx.network_arch,
        "train",
        ctx.default_specs,
        ["model.n_estimators"],
        schema=ctx.schema,
    )

    assert names == ["model.n_estimators"]


def test_optional_anyof_schema_uses_branch_search_metadata(tmp_path):
    from tao_automl.runner import SkillContext
    from tao_automl.search_space.params import generate_hyperparams_to_search

    skill_dir = _write_python_skill(tmp_path)
    schema_path = skill_dir / "schemas/train.schema.json"
    schema = json.loads(schema_path.read_text())
    parameter = schema["properties"]["model"]["properties"]["n_estimators"]
    parameter.clear()
    parameter["anyOf"] = [
        {"type": "null"},
        {
            "type": "integer",
            "default": 10,
            "minimum": 2,
            "maximum": 20,
            "automl_enabled": True,
        },
    ]
    schema_path.write_text(json.dumps(schema))

    ctx = SkillContext(skill_dir=skill_dir, action="train")
    records, names = generate_hyperparams_to_search(
        ctx.network_arch,
        "train",
        ctx.default_specs,
        [],
        schema=ctx.schema,
    )

    assert "model.n_estimators" in names
    record = next(r for r in records if r["parameter"] == "model.n_estimators")
    assert record["default_value"] == 10
    assert record["valid_min"] == 2
    assert record["valid_max"] == 20


def test_python_script_run_rejects_invalid_merged_spec_override(tmp_path):
    from tao_automl.runner import AutoMLRunner

    runner = AutoMLRunner(
        sdk=MagicMock(), skill_dir=_write_python_skill(tmp_path), action="train"
    )

    with pytest.raises(TypeError, match="expected an integer"):
        runner.run(
            automl_settings={"algorithm": "bayesian", "metric": "accuracy"},
            automl_hyperparameters=["model.n_estimators"],
            spec_overrides={"model.n_estimators": "many"},
            workspace_path=str(tmp_path / "workspace"),
        )


def test_runtime_python_execution_override_requires_external_schema(tmp_path):
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_fake_skill(tmp_path)
    script = skill_dir / "train.py"
    script.write_text("print('accuracy: 1.0')\n")
    runner = AutoMLRunner(sdk=MagicMock(), skill_dir=skill_dir, action="train")

    with pytest.raises(FileNotFoundError, match="external AutoML schema"):
        runner.run(
            execution={"type": "python_script", "script": str(script)},
            workspace_path=str(tmp_path / "workspace"),
        )


def test_external_schema_defaults_are_used_when_template_is_absent(tmp_path):
    from tao_automl.runner import SkillContext

    skill_dir = _write_python_skill(tmp_path)
    (skill_dir / "references/spec_template_train.yaml").unlink()

    ctx = SkillContext(skill_dir=skill_dir, action="train")
    assert ctx.default_specs["model"] == {"n_estimators": 10, "max_depth": 3}


def test_python_script_action_does_not_resolve_unused_container_image(tmp_path):
    from tao_automl.runner import SkillContext

    skill_dir = _write_python_skill(tmp_path)
    info_path = skill_dir / "references/skill_info.yaml"
    info_path.write_text(
        "container_image: key.that.does.not.exist\n" + info_path.read_text()
    )

    ctx = SkillContext(skill_dir=skill_dir, action="train")
    assert ctx.container_image == ""


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def test_extract_metric_allows_val_prefix_for_sparse4d_map():
    from tao_automl.runner import _extract_metric_from_logs
    logs = "Calculating metrics...\nmAP: 0.0000\nNDS: 0.0000\nExecution status: PASS\n"
    assert _extract_metric_from_logs(logs, "val_mAP") == 0.0


def test_extract_metric_supports_signed_values_and_exact_aliases():
    from tao_automl.runner import _extract_metric_from_logs

    logs = (
        "val_loss: 0.9\n"
        "loss_scale: 1024\n"
        "loss: -2.5e-1\n"
        "val_loss: 0.1\n"
    )

    assert _extract_metric_from_logs(logs, "loss") == pytest.approx(-0.25)
    assert _extract_metric_from_logs("val_loss: 0.2\n", "loss") is None
    assert _extract_metric_from_logs("loss: 0.2\n", "val_loss") is None


def test_cosmos_validation_loss_competes_by_log_position():
    from tao_automl.runner import _extract_metric_from_logs

    logs = "[SFT] Validation loss: 0.9\nval_loss: 0.2\n"

    assert _extract_metric_from_logs(logs, "val_loss") == pytest.approx(0.2)


def test_cosmos_validation_loss_does_not_satisfy_other_validation_metrics():
    from tao_automl.runner import _extract_metric_from_logs

    logs = "[SFT] Validation loss: 0.9\n"

    assert _extract_metric_from_logs(logs, "val_accuracy") is None


def test_extract_metric_uses_globally_latest_matching_format():
    from tao_automl.runner import _extract_metric_from_logs

    logs = "accuracy: 0.8\nnoise\nkpi: -1.25e-2\n"

    assert _extract_metric_from_logs(logs, "accuracy") == pytest.approx(-0.0125)


def test_extract_metric_reads_sparse4d_status_kpi_alias(tmp_path):
    from tao_automl.runner import _extract_metric_from_status_file

    status_path = tmp_path / "status.json"
    status_path.write_text(
        '{"status": "RUNNING", "kpi": {"img_bbox_NuScenes/mAP": 0.125}}\n'
    )

    assert _extract_metric_from_status_file(status_path, "val_mAP") == 0.125


def test_status_metric_ignores_non_objects_and_nonfinite_values(tmp_path):
    from tao_automl.runner import _extract_metric_from_status_file

    status_path = tmp_path / "status.json"
    status_path.write_text(
        '{"kpi": {"accuracy": 0.75}}\n'
        '[]\n'
        '{"kpi": {"accuracy": true}}\n'
        '{"kpi": {"accuracy": Infinity}}\n'
    )

    assert _extract_metric_from_status_file(status_path, "accuracy") == 0.75


@pytest.mark.parametrize("bad_metric", [True, float("nan"), float("inf"), -float("inf")])
def test_structured_metric_payloads_reject_boolean_and_nonfinite_values(bad_metric):
    from tao_automl.runner import (
        _extract_metric_from_best_score_payload,
        _extract_metric_from_metrics_payload,
        _merge_metric_payload,
    )

    assert _extract_metric_from_best_score_payload(
        {"best_score": bad_metric}, "accuracy"
    ) is None
    assert _extract_metric_from_metrics_payload(
        {"accuracy": bad_metric}, "accuracy"
    ) is None
    target = {}
    assert not _merge_metric_payload(target, {"metric_value": bad_metric})
    assert "metric_value" not in target


def test_multi_objective_callback_payload_requires_all_finite_values():
    from tao_automl.runner import _callback_metric_payload

    assert _callback_metric_payload(
        {"accuracy": 0.8, "latency": 12},
        "eval_fn",
    ) == {"accuracy": 0.8, "latency": 12.0}
    assert _callback_metric_payload(
        {"accuracy": 0.8, "latency": True},
        "eval_fn",
    ) is None
    assert _callback_metric_payload(
        {"accuracy": 0.8, "latency": float("nan")},
        "eval_fn",
    ) is None


def test_generic_log_metric_only_satisfies_primary_objective():
    from tao_automl.runner import _extract_metric_from_logs, _extract_metric_values

    assert _extract_metric_values(
        "kpi: 0.8\n",
        ["accuracy", "latency"],
        _extract_metric_from_logs,
    ) == {"accuracy": 0.8}


def test_extract_latency_aliases_from_logs_and_status(tmp_path):
    from tao_automl.runner import _extract_metric_from_logs, _extract_metric_from_status_file

    logs = "val_mAP: 0.812\ninference_latency_ms: 14.5\n"
    assert _extract_metric_from_logs(logs, "latency") == pytest.approx(14.5)

    status_path = tmp_path / "status.json"
    status_path.write_text(
        '{"status": "RUNNING", "kpi": {"avg_latency_ms": 12.25}}\n'
    )
    assert _extract_metric_from_status_file(status_path, "latency") == pytest.approx(12.25)


def test_latency_does_not_fall_back_to_primary_best_score():
    from tao_automl.runner import _extract_metric_from_best_score_payload

    payload = '{"best_score": 0.93, "metric": "val_mAP"}\n'
    assert _extract_metric_from_best_score_payload(payload, "latency") is None

    payload_with_latency = (
        '{"best_score": 0.93, "metric": "val_mAP", "latency_ms": 18.0}\n'
    )
    assert _extract_metric_from_best_score_payload(
        payload_with_latency, "latency"
    ) == pytest.approx(18.0)
    assert _extract_metric_from_best_score_payload(
        {"metric": "val_mAP", "val_mAP": 0.93},
        "latency",
    ) is None
    assert _extract_metric_from_best_score_payload(
        {"best_score": 0.93},
        "latency",
        allow_generic=False,
    ) is None


def test_execution_status_can_ignore_fatal_cleanup_patterns():
    from tao_automl.runner import _check_execution_status

    logs = "Saved best score to best_score.json\nRendezvousConnectionError\n"

    assert _check_execution_status(logs) == "FAIL"
    assert _check_execution_status(logs, include_fatal_patterns=False) is None


def test_execution_status_detects_nccl_watchdog_as_hard_failure():
    from tao_automl.runner import _check_execution_status, _has_hard_failure_pattern

    logs = (
        "Watchdog caught collective operation timeout: "
        "WorkNCCL(SeqNum=33271, OpType=ALLREDUCE)\n"
    )

    assert _check_execution_status(logs) == "FAIL"
    assert _has_hard_failure_pattern(logs)


def test_extract_metric_reads_cosmos_best_score_json(tmp_path):
    from tao_automl.runner import _extract_metric_from_local_results

    best_score = (
        tmp_path / "results" / "job-1" / "train_output_dir" / "best"
        / "best_score.json"
    )
    best_score.parent.mkdir(parents=True)
    best_score.write_text(
        '{"best_score": 0.8927091135965706, "metric": "val_loss"}\n'
    )

    metric = _extract_metric_from_local_results(
        "job-1",
        "val/avg_loss",
        {"mounts": [{"host_path": str(tmp_path / "results"),
                     "container_path": "/results"}]},
    )

    assert metric == pytest.approx(0.8927091135965706)


def test_extract_metric_reads_direct_script_metrics_json(tmp_path):
    from tao_automl.runner import _extract_metric_from_sdk_results

    results_dir = tmp_path / "job-results"
    metrics_path = results_dir / "results_dir" / "metrics.json"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text('{"accuracy": 0.9375, "score": 0.4}\n')

    class ResultsSDK:
        def get_job_results_dir(self, job_id):
            return str(results_dir)

    assert _extract_metric_from_sdk_results(
        ResultsSDK(), "job-1", "accuracy"
    ) == pytest.approx(0.9375)


def test_streamed_terminal_scan_uses_latest_explicit_status_marker():
    from tao_automl.runner import _extract_metric_from_logs, _scan_terminal_logs

    class StreamingSDK:
        def iter_job_log_chunks(self, job_id, chunk_size=256 * 1024):
            yield "Execution status: FAIL\n"
            yield "noise\n" * 2_000
            yield "accuracy: 0.9\nExecution status: PASS\n"

    metric, status, _ = _scan_terminal_logs(
        StreamingSDK(), "job-1", "accuracy", _extract_metric_from_logs, None, None
    )

    assert metric == pytest.approx(0.9)
    assert status == "PASS"


def test_non_streaming_sdk_retains_full_terminal_log_compatibility():
    from tao_automl.runner import _extract_metric_from_logs, _scan_terminal_logs

    full_logs = "accuracy: 0.875\n" + "noise\n" * 10_001

    class LegacySDK:
        def __init__(self):
            self.requested_tails = []

        def get_job_logs(self, job_id, tail=None):
            self.requested_tails.append(tail)
            if tail is None:
                return full_logs
            return "".join(full_logs.splitlines(keepends=True)[-tail:])

    sdk = LegacySDK()
    metric, status, _ = _scan_terminal_logs(
        sdk, "job-1", "accuracy", _extract_metric_from_logs, None, None
    )

    assert metric == pytest.approx(0.875)
    assert status is None
    assert sdk.requested_tails == [None]


def test_non_streaming_terminal_scan_finds_early_fail_with_cached_metric():
    from tao_automl.runner import _extract_metric_from_logs, _scan_terminal_logs

    full_logs = (
        "Execution status: FAIL\n"
        + "noise\n" * 10_001
        + "accuracy: 0.9\n"
    )

    class LegacySDK:
        def get_job_logs(self, job_id, tail=None):
            assert tail is None
            return full_logs

    metric, status, _ = _scan_terminal_logs(
        LegacySDK(),
        "job-1",
        "accuracy",
        _extract_metric_from_logs,
        0.9,
        None,
    )

    assert metric == pytest.approx(0.9)
    assert status == "FAIL"


def test_llm_config_accepts_provider_aliases():
    from tao_automl.brain.llm_client import LLMConfig

    config = LLMConfig.from_params({
        "base_url": "https://inference-api.nvidia.com",
        "model": "gcp/google/gemini-3.1-pro-preview",
        "api_key": "secret",
    })

    assert config.endpoint == "https://inference-api.nvidia.com"
    assert config.model == "gcp/google/gemini-3.1-pro-preview"
    assert config.api_key == "secret"


def test_algorithm_params_pass_provider_aliases_to_llm_client():
    from tao_automl.brain.factory import AlgorithmParams

    params = AlgorithmParams.from_dict({
        "base_url": "https://inference-api.nvidia.com",
        "model": "gcp/google/gemini-3.1-pro-preview",
        "api_key": "secret",
    })

    assert params.get_llm_params() == {
        "llm_endpoint": "https://inference-api.nvidia.com",
        "llm_model": "gcp/google/gemini-3.1-pro-preview",
        "llm_api_key": "secret",
    }


def test_algorithm_params_parse_hybrid_range_narrowing_flag():
    from tao_automl.brain.factory import AlgorithmParams

    assert AlgorithmParams.from_dict({
        "hybrid_enable_llm_range_narrowing": "true",
    }).hybrid_enable_llm_range_narrowing
    assert not AlgorithmParams.from_dict({}).hybrid_enable_llm_range_narrowing


def _install_sequential_fake_automl(monkeypatch, count=3):
    from tao_automl.types import Recommendation

    reports = []

    class FakeAutoML:
        def __init__(self, *args, **kwargs):
            self.recs = [
                Recommendation(i, {"train.num_epochs": i + 1}, "accuracy")
                for i in range(count)
            ]
            self.index = 0

        def is_complete(self):
            return self.index >= count

        def next_recommendation(self):
            return [self.recs[self.index]]

        def report_result(self, rec_id, metric_value, best_epoch=None, status="success"):
            rec = self.recs[self.index]
            rec.update_result(metric_value)
            rec.update_status(status)
            reports.append((rec_id, metric_value, status))
            self.index += 1

        def get_best(self):
            return None

        def get_progress(self):
            return {"completed": self.index, "best_metric": None}

        def get_history(self):
            return self.recs

    monkeypatch.setattr("tao_automl.AutoML", FakeAutoML)
    return reports


def test_job_creation_failures_do_not_count_as_missing_metrics(tmp_path, monkeypatch):
    from tao_automl.runner import AutoMLRunner

    reports = _install_sequential_fake_automl(monkeypatch)
    monkeypatch.setattr(
        AutoMLRunner, "_run_one_job", lambda self, *args, **kwargs: (None, "failure")
    )
    runner = AutoMLRunner(
        sdk=MagicMock(), skill_dir=_write_fake_skill(tmp_path), action="train"
    )

    with pytest.raises(RuntimeError, match="without a successful recommendation"):
        runner.run(
            image="nvcr.io/test:1",
            automl_settings={
                "algorithm": "bayesian",
                "metric": "accuracy",
                "run_baseline": False,
            },
            workspace_path=str(tmp_path / "workspace"),
        )

    assert len(reports) == 3
    assert {status for _, _, status in reports} == {"failure"}
    assert runner._consecutive_none_metrics == 0


def test_metric_missing_fail_fast_reports_third_failure_before_raising(
    tmp_path, monkeypatch,
):
    from tao_automl.runner import AutoMLRunner, MetricExtractorError

    reports = _install_sequential_fake_automl(monkeypatch)
    callbacks = []
    monkeypatch.setattr(
        AutoMLRunner,
        "_run_one_job",
        lambda self, *args, **kwargs: (None, "metric_missing"),
    )
    runner = AutoMLRunner(
        sdk=MagicMock(), skill_dir=_write_fake_skill(tmp_path), action="train"
    )

    with pytest.raises(MetricExtractorError, match="3 consecutive recs"):
        runner.run(
            image="nvcr.io/test:1",
            automl_settings={
                "algorithm": "bayesian",
                "metric": "accuracy",
                "run_baseline": False,
            },
            on_result=lambda rec, metric, status: callbacks.append(
                (rec.id, metric, status)
            ),
            workspace_path=str(tmp_path / "workspace"),
        )

    assert len(reports) == 3
    assert len(callbacks) == 3
    assert reports[-1] == (2, 0.0, "failure")
    assert callbacks[-1] == (2, None, "failure")


def test_promoted_metric_missing_checkpoint_carries_forward_prior_metric(
    tmp_path, monkeypatch
):
    from tao_automl.runner import AutoMLRunner
    from tao_automl.types import JobStates, Recommendation

    skill_dir = _write_fake_skill(tmp_path)
    results_root = tmp_path / "results"
    for job_id in ("parent-job", "child-job"):
        ckpt_dir = results_root / job_id / "results_dir" / "train"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "model_latest.pth").write_text("checkpoint")

    class FakeAutoML:
        def __init__(self, *args, **kwargs):
            self.rec = Recommendation(0, {"train.num_epochs": 2}, "val_mAP")
            self.rec.resume_from_job_id = "parent-job"
            self.rec.result = 0.42
            self.complete = False

        def is_complete(self):
            return self.complete

        def next_recommendation(self):
            return [self.rec]

        def report_result(self, rec_id, metric_value, best_epoch=None, status="success"):
            self.rec.update_result(metric_value)
            self.rec.update_status(status)
            self.complete = True

        def get_best(self):
            return self.rec if self.rec.status == JobStates.success else None

        def get_progress(self):
            return {"completed": int(self.complete), "best_metric": self.rec.result}

        def get_history(self):
            return [self.rec]

    def fake_run_one_job(self, *args, **kwargs):
        kwargs["rec"].assign_job_id("child-job")
        return None, "metric_missing"

    monkeypatch.setattr("tao_automl.AutoML", FakeAutoML)
    monkeypatch.setattr(AutoMLRunner, "_run_one_job", fake_run_one_job)

    runner = AutoMLRunner(sdk=MagicMock(), skill_dir=skill_dir, action="train")
    result = runner.run(
        image="nvcr.io/test:1",
        automl_settings={
            "algorithm": "dehb",
            "metric": "val_mAP",
            "direction": "maximize",
        },
        automl_hyperparameters=["train.optim.lr"],
        workspace_path=str(tmp_path / "workspace"),
        mounts=[{"host_path": str(results_root), "container_path": "/results"}],
    )

    assert result["best"]["metric_value"] == 0.42
    assert result["history"][0]["status"] == JobStates.success


def test_run_reports_baseline_metric_and_comparison(tmp_path, monkeypatch):
    from tao_automl.runner import AutoMLRunner
    from tao_automl.types import JobStates, Recommendation

    skill_dir = _write_fake_skill(tmp_path)

    class FakeAutoML:
        def __init__(self, *args, **kwargs):
            self.rec = Recommendation(0, {"train.num_epochs": 2}, "accuracy")
            self.complete = False

        def is_complete(self):
            return self.complete

        def next_recommendation(self):
            return [self.rec]

        def report_result(self, rec_id, metric_value, best_epoch=None, status="success"):
            self.rec.update_result(metric_value)
            self.rec.update_status(status)
            self.complete = True

        def get_best(self):
            return self.rec if self.rec.status == JobStates.success else None

        def get_progress(self):
            return {"completed": int(self.complete), "best_metric": self.rec.result}

        def get_history(self):
            return [self.rec]

    def fake_run_one_job(self, *args, **kwargs):
        return 0.62, "success"

    monkeypatch.setattr("tao_automl.AutoML", FakeAutoML)
    monkeypatch.setattr(AutoMLRunner, "_run_one_job", fake_run_one_job)

    runner = AutoMLRunner(sdk=MagicMock(), skill_dir=skill_dir, action="train")
    result = runner.run(
        image="nvcr.io/test:1",
        automl_settings={
            "algorithm": "bayesian",
            "metric": "accuracy",
            "direction": "maximize",
            "automl_max_recommendations": 1,
        },
        baseline_fn=lambda specs: 0.5,
        workspace_path=str(tmp_path / "workspace"),
    )

    assert result["baseline"]["status"] == "measured"
    assert result["baseline"]["metric_value"] == 0.5
    assert result["baseline"]["comparison_to_best"]["delta"] == pytest.approx(0.12)
    assert result["baseline"]["comparison_to_best"]["improved"] is True


def test_run_preserves_multi_objective_payload_and_explicit_primary_metric(
    tmp_path, monkeypatch,
):
    from tao_automl.runner import AutoMLRunner
    from tao_automl.types import JobStates, Recommendation

    captured = {}

    class FakeAutoML:
        def __init__(self, *args, **kwargs):
            self.rec = Recommendation(0, {"train.num_epochs": 2}, "accuracy")
            self.complete = False

        def is_complete(self):
            return self.complete

        def next_recommendation(self):
            return [self.rec]

        def report_result(self, rec_id, metric_value, best_epoch=None, status="success"):
            assert metric_value == {"accuracy": 0.8, "latency": 12.0}
            self.rec.update_objectives(metric_value, 0.68)
            self.rec.update_status(status)
            self.complete = True

        def get_best(self):
            return self.rec if self.rec.status == JobStates.success else None

        def get_progress(self):
            return {"completed": int(self.complete), "best_metric": 0.8}

        def get_history(self):
            return [self.rec]

        def get_status(self):
            return {"pareto_front": [{"rec_id": self.rec.id}]}

    def fake_run_one_job(self, *args, **kwargs):
        captured["metric_name"] = kwargs["metric_name"]
        captured["objective_names"] = kwargs["objective_names"]
        return {"accuracy": 0.8, "latency": 12}, "success"

    monkeypatch.setattr("tao_automl.AutoML", FakeAutoML)
    monkeypatch.setattr(AutoMLRunner, "_run_one_job", fake_run_one_job)

    runner = AutoMLRunner(
        sdk=MagicMock(),
        skill_dir=_write_fake_skill(tmp_path),
        action="train",
    )
    result = runner.run(
        image="nvcr.io/test:1",
        automl_settings={
            "algorithm": "bayesian",
            "objectives": [
                {"metric": "accuracy", "direction": "maximize"},
                {
                    "metric": "latency",
                    "direction": "minimize",
                    "scale": 100,
                },
            ],
            "run_baseline": False,
            "run_final_evaluation": False,
        },
        workspace_path=str(tmp_path / "workspace"),
    )

    assert captured == {
        "metric_name": "accuracy",
        "objective_names": ["accuracy", "latency"],
    }
    assert result["best"]["metric_value"] == pytest.approx(0.8)
    assert result["best"]["objective_score"] == pytest.approx(0.68)
    assert result["best"]["objective_values"] == {
        "accuracy": 0.8,
        "latency": 12.0,
    }
    assert result["pareto_front"] == [{"rec_id": 0}]


def test_run_reports_runner_owned_final_evaluation(tmp_path, monkeypatch):
    from tao_automl.runner import AutoMLRunner
    from tao_automl.types import JobStates, Recommendation

    skill_dir = _write_fake_skill(tmp_path)
    final_eval_calls = []
    final_record_path = tmp_path / "workspace" / "evaluations" / "best_automl.json"

    class FakeAutoML:
        def __init__(self, *args, **kwargs):
            self.rec = Recommendation(0, {"train.num_epochs": 2}, "accuracy")
            self.complete = False

        def is_complete(self):
            return self.complete

        def next_recommendation(self):
            return [self.rec]

        def report_result(self, rec_id, metric_value, best_epoch=None, status="success"):
            self.rec.update_result(metric_value)
            self.rec.update_status(status)
            self.complete = True

        def get_best(self):
            return self.rec if self.rec.status == JobStates.success else None

        def get_progress(self):
            return {"completed": int(self.complete), "best_metric": self.rec.result}

        def get_history(self):
            return [self.rec]

    def fake_run_one_job(self, *args, **kwargs):
        kwargs["rec"].assign_job_id("train-job-0")
        return 0.62, "success"

    def final_eval_fn(best_rec, train_job_id):
        final_eval_calls.append((best_rec.id, train_job_id))
        return {"metric_value": 0.64, "record_path": str(final_record_path)}

    monkeypatch.setattr("tao_automl.AutoML", FakeAutoML)
    monkeypatch.setattr(AutoMLRunner, "_run_one_job", fake_run_one_job)

    runner = AutoMLRunner(sdk=MagicMock(), skill_dir=skill_dir, action="train")
    result = runner.run(
        image="nvcr.io/test:1",
        automl_settings={
            "algorithm": "bayesian",
            "metric": "accuracy",
            "direction": "maximize",
            "automl_max_recommendations": 1,
            "run_final_evaluation": True,
        },
        baseline_fn=lambda specs: 0.5,
        final_eval_fn=final_eval_fn,
        workspace_path=str(tmp_path / "workspace"),
    )

    assert final_eval_calls == [(0, "train-job-0")]
    assert result["best"]["metric_value"] == 0.62
    assert result["final_evaluation"]["status"] == "measured"
    assert result["final_evaluation"]["source"] == "final_eval_fn"
    assert result["final_evaluation"]["metric_value"] == 0.64
    assert result["final_evaluation"]["record_path"] == str(final_record_path)
    assert result["final_evaluation"]["comparison_to_baseline"]["delta"] == pytest.approx(0.14)
    assert result["final_evaluation"]["comparison_to_baseline"]["improved"] is True


def test_effective_batch_is_capped_before_launch(tmp_path, monkeypatch):
    from tao_automl.runner import AutoMLRunner
    from tao_automl.types import JobStates, Recommendation

    skill_dir = _write_fake_skill(tmp_path)
    captured_specs = {}

    class FakeAutoML:
        def __init__(self, *args, **kwargs):
            self.rec = Recommendation(
                0,
                {
                    "train.train_batch_per_replica": 8,
                    "policy.parallelism.dp_shard_size": 8,
                },
                "accuracy",
            )
            self.complete = False

        def is_complete(self):
            return self.complete

        def next_recommendation(self):
            return [self.rec]

        def report_result(self, rec_id, metric_value, best_epoch=None, status="success"):
            self.rec.update_result(metric_value)
            self.rec.update_status(status)
            self.complete = True

        def get_best(self):
            return self.rec if self.rec.status == JobStates.success else None

        def get_progress(self):
            return {"completed": int(self.complete), "best_metric": self.rec.result}

        def get_history(self):
            return [self.rec]

    def fake_run_one_job(self, *args, **kwargs):
        captured_specs.update(kwargs["specs"])
        return 0.6, "success"

    monkeypatch.setattr("tao_automl.AutoML", FakeAutoML)
    monkeypatch.setattr(AutoMLRunner, "_run_one_job", fake_run_one_job)

    runner = AutoMLRunner(sdk=MagicMock(), skill_dir=skill_dir, action="train")
    result = runner.run(
        image="nvcr.io/test:1",
        automl_settings={
            "algorithm": "bayesian",
            "metric": "accuracy",
            "direction": "maximize",
            "automl_max_recommendations": 1,
            "train_sample_count": 31,
        },
        workspace_path=str(tmp_path / "workspace"),
    )

    assert captured_specs["train"]["train_batch_per_replica"] == 3
    assert result["history"][0]["adjustments"][0]["type"] == "effective_batch_cap"
    assert result["history"][0]["adjustments"][0]["from"] == 8
    assert result["history"][0]["adjustments"][0]["to"] == 3


def test_effective_batch_reports_invalid_when_no_rank_has_samples():
    from tao_automl.runner import _maybe_cap_effective_batch
    from tao_automl.types import Recommendation

    rec = Recommendation(
        0,
        {
            "train.train_batch_per_replica": 8,
            "policy.parallelism.dp_shard_size": 8,
        },
        "loss",
    )
    specs = {
        "train": {"train_batch_per_replica": 8},
        "policy": {"parallelism": {"dp_shard_size": 8}},
    }

    reason = _maybe_cap_effective_batch(
        specs, rec, {"train_sample_count": 2}, {}
    )

    assert "invalid_configuration" in reason
    assert rec.failure_reason == reason
    assert specs["train"]["train_batch_per_replica"] == 8


def test_validate_skill_runtime_probes_schema_import_path(tmp_path):
    from tao_automl.runner import validate_skill_runtime

    skill_dir = _write_fake_skill(tmp_path)
    info_path = skill_dir / "references/skill_info.yaml"
    info_path.write_text(
        info_path.read_text().replace("network_arch: fake-net", "network_arch: cosmos-rl")
    )

    result = validate_skill_runtime(skill_dir, action="train")

    assert result["network_arch"] == "cosmos-rl"
    assert result["action"] == "train"
    assert result["parameter_count"] > 0


# ---------------------------------------------------------------------------
# _make_sdk — platform selection
# ---------------------------------------------------------------------------

def test_make_sdk_rejects_unknown_platform():
    from tao_automl.runner import _make_sdk
    with pytest.raises(ValueError, match="Unknown platform"):
        _make_sdk("aws-batch")


def test_make_sdk_lists_all_platforms_in_error():
    from tao_automl.runner import _make_sdk, _PLATFORMS
    assert set(_PLATFORMS) == {
        "lepton", "slurm", "kubernetes", "docker", "brev", "virtualenv",
    }
    try:
        _make_sdk("nope")
    except ValueError as e:
        for p in _PLATFORMS:
            assert p in str(e)


def test_make_sdk_constructs_virtualenv_with_sdk_kwargs(tmp_path):
    from tao_automl.runner import _make_sdk

    with patch("tao_sdk.platforms.virtualenv.VirtualEnvSDK") as sdk_cls:
        instance = _make_sdk(
            "virtualenv",
            venv_path=str(tmp_path / "venv"),
            work_dir=str(tmp_path / "jobs"),
        )

    assert instance is sdk_cls.return_value
    sdk_cls.assert_called_once_with(
        venv_path=str(tmp_path / "venv"),
        work_dir=str(tmp_path / "jobs"),
    )


# ---------------------------------------------------------------------------
# AutoMLRunner — submission path uses build_entrypoint + new create_job shape
# ---------------------------------------------------------------------------

def test_run_one_job_calls_build_entrypoint_with_action_cfg(tmp_path):
    """_run_one_job should pass the action's command/inputs/outputs/config_format/
    upload_excludes to build_entrypoint, and the resulting command string to
    sdk.create_job. No old kwargs (specs=, script_runner=, network_arch=)."""
    from tao_automl.runner import AutoMLRunner, _POLL_LOG_TAIL_LINES

    skill_dir = _write_fake_skill(tmp_path)
    fake_sdk = MagicMock()
    fake_sdk.create_job.return_value = MagicMock(id="job-xyz", backend_job_id="be-xyz")

    runner = AutoMLRunner(sdk=fake_sdk, skill_dir=skill_dir, action="train")
    rec = MagicMock(id=1)

    fake_ep = {"command": "BAKED_HEREDOC_COMMAND", "args_template": ""}

    # Patch build_entrypoint at its import site (inside _run_one_job).
    with patch("tao_sdk.script_runner.build_entrypoint", return_value=fake_ep) as build:
        # Avoid the polling loop — make get_job_status return Complete immediately.
        fake_sdk.get_job_status.return_value = MagicMock(status="Complete")
        fake_sdk.get_job_logs.return_value = "loss: 0.5\n"
        runner._poll_interval = 0  # tight loop
        runner._run_one_job(
            image="nvcr.io/test:1",
            action_cfg=runner.skill_ctx.action_cfg,
            specs={"train": {"num_epochs": 10}},
            rec=rec, metric_name="loss",
            workspace_path=str(tmp_path),
            platform_kwargs={"gpu_count": 4, "num_nodes": 1},
        )

    # build_entrypoint received the schema fields from skill_info
    build.assert_called_once()
    kwargs = build.call_args.kwargs
    assert kwargs["command"] == "fake train -e {config_path}"
    assert kwargs["config_format"] == "yaml"
    assert "dataset.train_data_sources[0].image_dir" in kwargs["inputs"]
    assert "results_dir" in kwargs["outputs"]
    assert kwargs["upload_excludes"] == ["inputs/"]
    assert kwargs["specs"]["train"]["num_epochs"] == 10

    # create_job got the baked command + platform kwargs (no old shape)
    create_kwargs = fake_sdk.create_job.call_args.kwargs
    assert create_kwargs["command"] == "BAKED_HEREDOC_COMMAND"
    assert create_kwargs["image"] == "nvcr.io/test:1"
    assert create_kwargs["gpu_count"] == 4
    assert create_kwargs["num_nodes"] == 1
    # Old kwargs should NOT be present
    for legacy in ("specs", "script_runner", "network_arch", "data_format",
                   "backend_details", "workspace_id", "train_dataset_uri"):
        assert legacy not in create_kwargs, f"legacy kwarg {legacy!r} leaked"
    assert fake_sdk.get_job_logs.call_count >= 1
    assert any(
        call.kwargs.get("tail") == _POLL_LOG_TAIL_LINES
        for call in fake_sdk.get_job_logs.call_args_list
    )
    assert any(not call.kwargs for call in fake_sdk.get_job_logs.call_args_list)


def test_run_one_job_returns_multi_objective_values(tmp_path):
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_fake_skill(tmp_path)
    fake_sdk = MagicMock()
    fake_sdk.create_job.return_value = MagicMock(id="job-mo", backend_job_id="be-mo")
    fake_sdk.get_job_status.return_value = MagicMock(status="Complete")
    fake_sdk.get_job_logs.return_value = "val_mAP: 0.75\nlatency_ms: 21.0\n"

    runner = AutoMLRunner(sdk=fake_sdk, skill_dir=skill_dir, action="train")
    runner._poll_interval = 0
    rec = MagicMock(id=3)

    with patch(
        "tao_sdk.script_runner.build_entrypoint",
        return_value={"command": "BAKED_HEREDOC_COMMAND", "args_template": ""},
    ):
        metric, status = runner._run_one_job(
            image="nvcr.io/test:1",
            action_cfg=runner.skill_ctx.action_cfg,
            specs={"train": {"num_epochs": 1}},
            rec=rec,
            metric_name="val_mAP",
            objective_names=["val_mAP", "latency"],
            workspace_path=str(tmp_path),
            platform_kwargs={},
        )

    assert status == "success"
    assert metric == {"val_mAP": pytest.approx(0.75), "latency": pytest.approx(21.0)}


def test_run_one_job_accepts_multi_objective_eval_callback(tmp_path):
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_fake_skill(tmp_path)
    fake_sdk = MagicMock()
    fake_sdk.create_job.return_value = MagicMock(
        id="job-mo-eval",
        backend_job_id="be-mo-eval",
    )
    fake_sdk.get_job_status.return_value = MagicMock(status="Complete")
    fake_sdk.get_job_logs.return_value = ""
    fake_sdk.get_job_results_dir.return_value = ""
    fake_sdk.read_job_result_file.return_value = ""

    runner = AutoMLRunner(sdk=fake_sdk, skill_dir=skill_dir, action="train")
    runner._poll_interval = 0
    rec = MagicMock(id=4)

    with patch(
        "tao_sdk.script_runner.build_entrypoint",
        return_value={"command": "BAKED_HEREDOC_COMMAND", "args_template": ""},
    ):
        metric, status = runner._run_one_job(
            image="nvcr.io/test:1",
            action_cfg=runner.skill_ctx.action_cfg,
            specs={"train": {"num_epochs": 1}},
            rec=rec,
            metric_name="val_mAP",
            objective_names=["val_mAP", "latency"],
            eval_fn=lambda recommendation, job_id: {
                "val_mAP": 0.76,
                "latency": 20,
            },
            workspace_path=str(tmp_path),
            platform_kwargs={},
        )

    assert status == "success"
    assert metric == {"val_mAP": pytest.approx(0.76), "latency": 20.0}


def test_run_one_job_submits_nested_specs_to_python_script_sdk(tmp_path):
    """Python actions bypass the container entrypoint and image API."""
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_python_skill(tmp_path)
    fake_sdk = MagicMock()
    fake_sdk.create_python_job.return_value = MagicMock(
        id="job-python", backend_job_id="12345"
    )
    fake_sdk.get_job_status.return_value = MagicMock(status="Complete")
    fake_sdk.get_job_logs.return_value = "accuracy: 0.875\n"

    runner = AutoMLRunner(sdk=fake_sdk, skill_dir=skill_dir, action="train")
    runner._poll_interval = 0
    rec = MagicMock(id=3)

    with patch(
        "tao_sdk.script_runner.build_entrypoint",
        side_effect=AssertionError("container entrypoint must not be built"),
    ) as build:
        metric, status = runner._run_one_job(
            image=None,
            action_cfg=runner.skill_ctx.action_cfg,
            specs={"model": {"n_estimators": 17, "max_depth": 4}},
            rec=rec,
            metric_name="accuracy",
            execution=runner.skill_ctx.execution,
            workspace_path=str(tmp_path / "workspace"),
            platform_kwargs={"gpu_count": 0},
        )

    build.assert_not_called()
    fake_sdk.create_job.assert_not_called()
    create_kwargs = fake_sdk.create_python_job.call_args.kwargs
    assert create_kwargs["script"] == str(skill_dir / "scripts/train.py")
    assert create_kwargs["specs"] == {
        "model": {"n_estimators": 17, "max_depth": 4}
    }
    assert create_kwargs["config_format"] == "json"
    assert create_kwargs["script_args"] == ["--config", "{config_path}"]
    assert create_kwargs["cwd"] == str(skill_dir)
    assert create_kwargs["network_arch"] == "public_random_forest"
    assert create_kwargs["action"] == "train"
    assert create_kwargs["gpu_count"] == 0
    assert metric == pytest.approx(0.875)
    assert status == "success"


def test_runner_runs_real_subprocess_job_for_non_train_action(tmp_path):
    """Exercise a real subprocess-backed SDK job for a non-train action."""
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_fake_action_skill(tmp_path, action="quantize")
    runner = AutoMLRunner(
        sdk=_CompletedProcessSDK(),
        skill_dir=skill_dir,
        action="quantize",
        poll_interval=0,
    )

    result = runner.run(
        automl_settings={
            "algorithm": "bayesian",
            "metric": "action_metric",
            "direction": "maximize",
            "automl_max_recommendations": 1,
        },
        automl_hyperparameters=["train.optim.lr"],
        custom_param_ranges={
            "train.optim.lr": {"valid_min": 1e-5, "valid_max": 1e-3},
        },
        workspace_path=str(tmp_path / "workspace"),
        env_vars={"TAO_RESULTS_ROOT": str(tmp_path / "sdk-results")},
    )

    assert result["progress"]["completed"] == 1
    assert result["best"]["metric_value"] == pytest.approx(0.73)
    assert result["history"][0]["status"] == "success"


def test_run_one_job_allows_completed_metric_with_cleanup_rendezvous(tmp_path):
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_fake_skill(tmp_path)
    fake_sdk = MagicMock()
    fake_sdk.create_job.return_value = MagicMock(id="job-xyz", backend_job_id="be-xyz")
    fake_sdk.get_job_status.return_value = MagicMock(status="Complete")
    fake_sdk.get_job_logs.return_value = (
        "[cosmos] Validation rank 0: avg_loss=0.951012, samples=6\n"
        "Saved best score to best_score.json: 0.9510115849549504\n"
        "torch.distributed.elastic.rendezvous.api.RendezvousConnectionError\n"
        "torch.distributed.DistNetworkError: Failed to recv, got 0 bytes.\n"
    )

    runner = AutoMLRunner(sdk=fake_sdk, skill_dir=skill_dir, action="train")
    runner._poll_interval = 0
    rec = MagicMock(id=7)

    with patch(
        "tao_sdk.script_runner.build_entrypoint",
        return_value={"command": "BAKED_HEREDOC_COMMAND", "args_template": ""},
    ):
        metric, status = runner._run_one_job(
            image="nvcr.io/test:1",
            action_cfg=runner.skill_ctx.action_cfg,
            specs={"train": {"num_epochs": 1}},
            rec=rec,
            metric_name="val/avg_loss",
            workspace_path=str(tmp_path),
            platform_kwargs={},
        )

    assert metric == pytest.approx(0.951012)
    assert status == "success"
    fake_sdk.cancel_job.assert_not_called()


def test_terminal_streaming_recovers_metric_before_noisy_tail(tmp_path):
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_fake_skill(tmp_path)
    full_logs = "accuracy: 0.875\n" + "noise\n" * 10_001

    class StreamingSDK:
        def __init__(self):
            self.create_job = MagicMock(
                return_value=MagicMock(id="job-noisy", backend_job_id="be")
            )
            self.log_tails = []

        def get_job_status(self, job_id):
            return MagicMock(status="Complete")

        def get_job_logs(self, job_id, tail=None):
            self.log_tails.append(tail)
            assert tail is not None
            return "".join(full_logs.splitlines(keepends=True)[-tail:])

        def iter_job_log_chunks(self, job_id, chunk_size=256 * 1024):
            for offset in range(0, len(full_logs), 31):
                yield full_logs[offset:offset + 31]

        def cancel_job(self, job_id):
            return False

    fake_sdk = StreamingSDK()
    runner = AutoMLRunner(sdk=fake_sdk, skill_dir=skill_dir, action="train")
    runner._poll_interval = 0
    rec = MagicMock(id=10)

    with patch(
        "tao_sdk.script_runner.build_entrypoint",
        return_value={"command": "BAKED", "args_template": ""},
    ):
        metric, status = runner._run_one_job(
            image="nvcr.io/test:1",
            action_cfg=runner.skill_ctx.action_cfg,
            specs={"train": {"num_epochs": 1}},
            rec=rec,
            metric_name="accuracy",
            workspace_path=str(tmp_path),
            platform_kwargs={},
        )

    assert metric == pytest.approx(0.875)
    assert status == "success"
    assert fake_sdk.log_tails
    assert all(tail is not None for tail in fake_sdk.log_tails)


def test_run_one_job_cancels_hard_failure_and_recovers_remote_best_score(tmp_path):
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_fake_skill(tmp_path)
    fake_sdk = MagicMock()
    fake_sdk.create_job.return_value = MagicMock(id="job-hard", backend_job_id="be-hard")
    fake_sdk.get_job_logs.return_value = (
        "Watchdog caught collective operation timeout: "
        "WorkNCCL(SeqNum=33271, OpType=ALLREDUCE)\n"
    )
    fake_sdk.read_job_result_file.return_value = (
        '{"best_score": 0.8927091135965706, "metric": "val_loss"}\n'
    )

    runner = AutoMLRunner(sdk=fake_sdk, skill_dir=skill_dir, action="train")
    runner._poll_interval = 0
    rec = MagicMock(id=8)

    with patch(
        "tao_sdk.script_runner.build_entrypoint",
        return_value={"command": "BAKED_HEREDOC_COMMAND", "args_template": ""},
    ):
        metric, status = runner._run_one_job(
            image="nvcr.io/test:1",
            action_cfg=runner.skill_ctx.action_cfg,
            specs={"train": {"num_epochs": 10}},
            rec=rec,
            metric_name="val/avg_loss",
            workspace_path=str(tmp_path),
            platform_kwargs={},
        )

    assert metric == pytest.approx(0.8927091135965706)
    assert status == "failure"
    fake_sdk.cancel_job.assert_called_once_with("job-hard")


def test_run_one_job_preserves_metric_when_slurm_reports_canceled(tmp_path):
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_fake_skill(tmp_path)
    fake_sdk = MagicMock()
    fake_sdk.create_job.return_value = MagicMock(id="job-canceled", backend_job_id="be")
    fake_sdk.get_job_status.return_value = MagicMock(status="Canceled")
    fake_sdk.get_job_logs.return_value = (
        "[SFT] Validation loss: 0.751 for train step 8/10, epoch 4\n"
    )

    runner = AutoMLRunner(sdk=fake_sdk, skill_dir=skill_dir, action="train")
    runner._poll_interval = 0
    rec = MagicMock(id=9)

    with patch(
        "tao_sdk.script_runner.build_entrypoint",
        return_value={"command": "BAKED_HEREDOC_COMMAND", "args_template": ""},
    ):
        metric, status = runner._run_one_job(
            image="nvcr.io/test:1",
            action_cfg=runner.skill_ctx.action_cfg,
            specs={"train": {"num_epochs": 10}},
            rec=rec,
            metric_name="val_loss",
            workspace_path=str(tmp_path),
            platform_kwargs={},
        )

    assert metric == pytest.approx(0.751)
    assert status == "failure"


def test_runner_init_replaces_skillbank_with_skillcontext(tmp_path):
    """AutoMLRunner.__init__ no longer takes (sdk, poll_interval) only —
    skill_dir + action are now required."""
    from tao_automl.runner import AutoMLRunner, SkillContext
    skill_dir = _write_fake_skill(tmp_path)
    fake_sdk = MagicMock()
    runner = AutoMLRunner(sdk=fake_sdk, skill_dir=skill_dir, action="train")
    assert isinstance(runner.skill_ctx, SkillContext)
    assert runner.skill_ctx.network_arch == "fake-net"
    # Old API would've worked without skill_dir; new API is explicit.
    with pytest.raises(TypeError):
        AutoMLRunner(sdk=fake_sdk)  # missing skill_dir


def test_execution_parameter_is_appended_after_existing_positional_callbacks():
    """Adding Python execution must not rebind legacy positional arguments."""
    import inspect

    from tao_automl.runner import AutoMLRunner

    parameters = list(inspect.signature(AutoMLRunner.run).parameters)
    assert parameters.index("execution") > parameters.index("on_result")


def test_container_run_preserves_builtin_search_schema_source(tmp_path, monkeypatch):
    """Packaged schemas must not change existing container recommendations."""
    from types import SimpleNamespace

    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_fake_skill(tmp_path)
    schemas = skill_dir / "schemas"
    schemas.mkdir()
    (schemas / "train.schema.json").write_text(json.dumps({
        "type": "object",
        "default": {"external_only": 1},
        "properties": {
            "external_only": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "maximum": 2,
                "automl_enabled": True,
            },
        },
    }))
    captured = {}
    best = SimpleNamespace(id=0, result=0.5, specs={})

    class CompleteAutoML:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        def is_complete(self):
            return True

        def get_best(self):
            return best

        def get_progress(self):
            return {"completed": 0, "best_metric": 0.5}

        def get_history(self):
            return []

    monkeypatch.setattr("tao_automl.AutoML", CompleteAutoML)
    runner = AutoMLRunner(sdk=MagicMock(), skill_dir=skill_dir, action="train")
    runner.run(
        automl_settings={"algorithm": "bayesian", "metric": "loss"},
        workspace_path=str(tmp_path / "workspace"),
    )

    assert captured["search_schema"] is None


# ---------------------------------------------------------------------------
# Override merging
# ---------------------------------------------------------------------------

def test_merge_specs_deep_merges_dotted_keys():
    from tao_automl.runner import AutoMLRunner
    base = {"train": {"num_epochs": 12, "optim": {"lr": 2.0e-4}}}
    overrides = {"train.optim.lr": 5.0e-5, "train.num_gpus": 8}
    merged = AutoMLRunner._merge_specs(base, overrides)
    assert merged["train"]["num_epochs"] == 12       # untouched
    assert merged["train"]["optim"]["lr"] == 5.0e-5  # overridden
    assert merged["train"]["num_gpus"] == 8           # added


def test_merge_specs_does_not_mutate_base():
    from tao_automl.runner import AutoMLRunner
    base = {"train": {"num_epochs": 12}}
    AutoMLRunner._merge_specs(base, {"train.num_epochs": 5})
    assert base["train"]["num_epochs"] == 12  # base stayed pristine


# ---------------------------------------------------------------------------
# Resume checkpoint handoff
# ---------------------------------------------------------------------------

def test_apply_resume_checkpoint_sets_training_checkpoint_path(tmp_path):
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_fake_skill(tmp_path)
    template = skill_dir / "references/spec_template_train.yaml"
    template.write_text(
        "train:\n"
        "  num_epochs: 12\n"
        "  resume_training_checkpoint_path: ''\n"
        "dataset:\n"
        "  num_classes: 80\n"
    )

    results_root = tmp_path / "results"
    checkpoint = (
        results_root / "parent-job" / "results_dir" / "train" / "model_epoch_001.pth"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("checkpoint")
    latest = checkpoint.parent / "classifier_model_latest.pth"
    latest.write_text("latest")

    runner = AutoMLRunner(sdk=MagicMock(), skill_dir=skill_dir, action="train")
    rec = MagicMock(id=2, resume_from_job_id="parent-job", resume_from_epoch=1)

    specs = {"train": {"resume_training_checkpoint_path": ""}}
    updated = runner._apply_resume_checkpoint(
        specs,
        rec,
        {"mounts": [{"host_path": str(results_root), "container_path": "/results"}]},
    )

    assert (
        updated["train"]["resume_training_checkpoint_path"]
        == "/results/parent-job/results_dir/train/model_epoch_001.pth"
    )
    assert rec.resume_checkpoint_path.endswith("model_epoch_001.pth")


def test_apply_resume_checkpoint_does_not_use_latest_for_requested_epoch(tmp_path):
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_fake_skill(tmp_path)
    (skill_dir / "references/spec_template_train.yaml").write_text(
        "train:\n"
        "  resume_training_checkpoint_path: ''\n"
    )

    results_root = tmp_path / "results"
    checkpoint_dir = results_root / "parent-job" / "results_dir" / "train"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model_latest.pth").write_text("latest")

    runner = AutoMLRunner(sdk=MagicMock(), skill_dir=skill_dir, action="train")
    rec = MagicMock(id=2, resume_from_job_id="parent-job", resume_from_epoch=1)

    updated = runner._apply_resume_checkpoint(
        {"train": {"resume_training_checkpoint_path": ""}},
        rec,
        {"mounts": [{"host_path": str(results_root), "container_path": "/results"}]},
    )

    assert updated["train"]["resume_training_checkpoint_path"] == ""
    assert rec.resume_checkpoint_missing is True
    assert rec.resume_checkpoint_path is None


def test_apply_resume_checkpoint_sets_cosmos_resume_to_checkpoint_dir(tmp_path):
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_fake_skill(tmp_path)
    template = skill_dir / "references/spec_template_train.yaml"
    template.write_text("train:\n  resume: false\n  epoch: 2\n")

    results_root = tmp_path / "results"
    checkpoint_dir = (
        results_root / "parent-job" / "train_output_dir" / "run1"
        / "checkpoints" / "epoch_1"
    )
    (checkpoint_dir / "policy").mkdir(parents=True)
    (checkpoint_dir / "policy" / "model_rank_0.pth").write_text("checkpoint")
    safetensor_dir = (
        results_root / "parent-job" / "train_output_dir" / "run1"
        / "safetensors" / "epoch_1"
    )
    safetensor_dir.mkdir(parents=True)
    (safetensor_dir / "adapter_model.safetensors").write_text("adapter")

    runner = AutoMLRunner(sdk=MagicMock(), skill_dir=skill_dir, action="train")
    rec = MagicMock(id=3, resume_from_job_id="parent-job", resume_from_epoch=1)

    updated = runner._apply_resume_checkpoint(
        {"train": {"resume": False, "epoch": 2}},
        rec,
        {"mounts": [{"host_path": str(results_root), "container_path": "/results"}]},
    )

    assert (
        updated["train"]["resume"]
        == "/results/parent-job/train_output_dir/run1/checkpoints/epoch_1"
    )


def test_apply_resume_environment_enables_trusted_checkpoint_resume(tmp_path):
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_fake_skill(tmp_path)
    runner = AutoMLRunner(sdk=MagicMock(), skill_dir=skill_dir, action="train")
    rec = MagicMock(
        id=4,
        resume_from_job_id="parent-job",
        resume_checkpoint_path="/results/parent-job/train/checkpoint.pth",
    )

    updated = runner._apply_resume_environment(
        {"env_vars": {"WANDB_MODE": "disabled"}},
        rec,
    )

    assert updated["env_vars"]["WANDB_MODE"] == "disabled"
    assert updated["env_vars"]["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"


def test_apply_resume_environment_does_not_mutate_non_resume_kwargs(tmp_path):
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_fake_skill(tmp_path)
    runner = AutoMLRunner(sdk=MagicMock(), skill_dir=skill_dir, action="train")
    rec = MagicMock(id=5, resume_from_job_id=None)
    platform_kwargs = {"env_vars": {"WANDB_MODE": "disabled"}}

    assert runner._apply_resume_environment(platform_kwargs, rec) is platform_kwargs


def test_apply_resume_environment_ignores_missing_checkpoint_path(tmp_path):
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_fake_skill(tmp_path)
    runner = AutoMLRunner(sdk=MagicMock(), skill_dir=skill_dir, action="train")
    rec = MagicMock(
        id=6,
        resume_from_job_id="parent-job",
        resume_checkpoint_path=None,
    )
    platform_kwargs = {"env_vars": {"WANDB_MODE": "disabled"}}

    assert runner._apply_resume_environment(platform_kwargs, rec) is platform_kwargs
