"""Comprehensive test for the nvidia-tao-automl wheel."""

import tempfile
import threading
import subprocess

import pytest


# ---------------------------------------------------------------
# 1. Import tests
# ---------------------------------------------------------------

def test_import_top_level():
    from tao_automl import AutoML
    assert AutoML is not None


def test_import_types():
    from tao_automl.types import Recommendation, ResumeRecommendation, JobStates, AutoMLContext
    assert JobStates.success == "success"


def test_import_state_store():
    from tao_automl.state.state_store import StateStore
    assert StateStore is not None


def test_import_controller():
    from tao_automl.controller.controller import Controller
    assert Controller is not None


def test_import_brain_factory():
    from tao_automl.brain.factory import BrainFactory, AlgorithmParams
    assert BrainFactory is not None


def test_import_brain_algorithms():
    from tao_automl.brain.bayesian import Bayesian
    from tao_automl.brain.hyperband import HyperBand
    from tao_automl.brain.bohb import BOHB
    from tao_automl.brain.asha import ASHA
    from tao_automl.brain.bfbo import BFBO
    from tao_automl.brain.dehb import DEHB
    from tao_automl.brain.pbt import PBT
    from tao_automl.brain.hyperband_es import HyperBandES
    assert all([Bayesian, HyperBand, BOHB, ASHA, BFBO, DEHB, PBT, HyperBandES])


def test_import_utils():
    from tao_automl.utils.math_utils import fix_input_dimension, clamp_value
    from tao_automl.utils.spec_utils import get_flatten_specs
    from tao_automl.utils.network_constants import gpu_mapper
    assert callable(fix_input_dimension)


# ---------------------------------------------------------------
# 2. Types tests
# ---------------------------------------------------------------

def test_recommendation_create():
    from tao_automl.types import Recommendation, JobStates
    rec = Recommendation(identifier=0, specs={"lr": 0.01}, metric="loss")
    assert rec.id == 0
    assert rec.status == JobStates.pending
    assert rec.result == 0.0


def test_recommendation_update():
    from tao_automl.types import Recommendation, JobStates
    rec = Recommendation(identifier=1, specs={"lr": 0.01}, metric="loss")
    rec.update_result(0.5)
    assert rec.result == 0.5
    rec.update_status(JobStates.success)
    assert rec.status == "success"
    rec.assign_job_id("job-123")
    assert rec.job_id == "job-123"


def test_recommendation_type_checks():
    from tao_automl.types import Recommendation
    with pytest.raises(AssertionError):
        Recommendation(identifier="bad", specs={}, metric="loss")


def test_automl_context():
    from tao_automl.types import AutoMLContext
    ctx = AutoMLContext(id="t1", network="dino", metric="loss")
    assert ctx.action == "train"


# ---------------------------------------------------------------
# 3. StateStore tests
# ---------------------------------------------------------------

def test_state_store_roundtrip():
    from tao_automl.state.state_store import StateStore
    with tempfile.TemporaryDirectory() as d:
        store = StateStore(d)
        store.save_job_specs("j1", {"lr": 0.01})
        assert store.get_job_specs("j1") == {"lr": 0.01}

        store.save_brain_info("j1", {"Xs": [[0.1]], "ys": [0.5]})
        assert store.get_brain_info("j1")["ys"] == [0.5]

        store.save_controller_info("j1", [{"id": 0}])
        assert store.get_controller_info("j1") == [{"id": 0}]

        store.save_best_rec_info("j1", rec_number=0, rec_data={"m": 0.1})
        assert store.get_best_rec_info("j1")["rec_number"] == 0

        store.save_custom_param_ranges("e1", {"lr": {"min": 1e-5}})
        assert store.get_custom_param_ranges("e1")["lr"]["min"] == 1e-5


def test_state_store_missing_returns_none():
    from tao_automl.state.state_store import StateStore
    with tempfile.TemporaryDirectory() as d:
        store = StateStore(d)
        assert store.get_job_specs("nope") is None
        assert store.get_brain_info("nope") is None


# ---------------------------------------------------------------
# 4. Controller tests
# ---------------------------------------------------------------

class MockBrain:
    def __init__(self, limit=5):
        self.limit = limit
        self.n = 0

    def generate_recommendations(self, history):
        if len(history) < self.limit:
            self.n += 1
            return [{"lr": 0.01 * self.n}]
        return []

    def save_state(self):
        pass


def _make_ctrl(d, limit=5, metric="loss"):
    from tao_automl.controller.controller import Controller
    from tao_automl.state.state_store import StateStore
    from tao_automl.types import AutoMLContext
    store = StateStore(d)
    ctx = AutoMLContext(id="test", network="dino")
    settings = type("P", (), {"automl_max_recommendations": limit})()
    return Controller(
        brain=MockBrain(limit), context=ctx, state_store=store,
        settings=settings, metric=metric, algorithm="bayesian",
    ), store, ctx


def test_controller_full_loop():
    with tempfile.TemporaryDirectory() as d:
        ctrl, _, _ = _make_ctrl(d, 5)
        for i in range(5):
            recs = ctrl.next_recommendation()
            assert recs[0].id == i
            ctrl.report_result(recs[0].id, 0.5 - i * 0.1)
        assert ctrl.is_complete()
        assert abs(ctrl.get_best().result - 0.1) < 1e-9
        assert ctrl.get_progress()["completed"] == 5


def test_controller_persistence():
    from tao_automl.controller.controller import Controller
    with tempfile.TemporaryDirectory() as d:
        ctrl, store, ctx = _make_ctrl(d, 5)
        for i in range(3):
            recs = ctrl.next_recommendation()
            ctrl.report_result(recs[0].id, 0.5 - i * 0.1)

        settings = type("P", (), {"automl_max_recommendations": 5})()
        ctrl2 = Controller.load_state(
            brain=MockBrain(5), context=ctx, state_store=store,
            settings=settings, metric="loss", algorithm="bayesian",
        )
        assert len(ctrl2.history) == 3
        assert ctrl2._next_id == 3


def test_controller_higher_is_better():
    with tempfile.TemporaryDirectory() as d:
        ctrl, _, _ = _make_ctrl(d, 3, metric="accuracy")
        for m in [0.8, 0.95, 0.9]:
            recs = ctrl.next_recommendation()
            ctrl.report_result(recs[0].id, m)
        assert abs(ctrl.get_best().result - 0.95) < 1e-9


def test_controller_failure_skipped_in_best():
    with tempfile.TemporaryDirectory() as d:
        ctrl, _, _ = _make_ctrl(d, 2)
        recs = ctrl.next_recommendation()
        ctrl.report_result(recs[0].id, 0.0, status="failure")
        assert ctrl.get_best() is None


def test_controller_accepts_resume_recommendations():
    from tao_automl.controller.controller import Controller
    from tao_automl.state.state_store import StateStore
    from tao_automl.types import AutoMLContext, JobStates, ResumeRecommendation

    class ResumeBrain(MockBrain):
        def generate_recommendations(self, history):
            if not history:
                return [{"lr": 0.1}]
            return [ResumeRecommendation(0, {"lr": 0.1}, "job-previous")]

    with tempfile.TemporaryDirectory() as d:
        store = StateStore(d)
        ctx = AutoMLContext(id="resume-test", network="fake")
        settings = type("P", (), {"automl_max_recommendations": 2})()
        ctrl = Controller(
            brain=ResumeBrain(1), context=ctx, state_store=store,
            settings=settings, metric="loss", algorithm="hyperband",
        )

        first = ctrl.next_recommendation()
        assert first[0].id == 0
        ctrl.report_result(0, 1.0, status="success")

        resumed = ctrl.next_recommendation()
        assert resumed[0].id == 0
        assert resumed[0].status == JobStates.pending
        assert resumed[0].resume_from_job_id == "job-previous"


def test_hyperband_two_two_one_stop_compare_resume_flow():
    """Hyperband max_epochs=2/reduction_factor=2/epoch_multiplier=1 should:

    - issue two first-rung trials capped at 1 epoch,
    - compare their metrics,
    - promote only the better trial,
    - resume it from the first-rung job,
    - run the promoted trial to epoch 2.
    """
    from tao_automl import AutoML

    base_specs = {
        "train": {
            "epoch": 10,
            "optm_lr": 1e-6,
        }
    }

    with tempfile.TemporaryDirectory() as d:
        automl = AutoML(
            workspace=d,
            network="cosmos-rl",
            train_specs=base_specs,
            settings={
                "algorithm": "hyperband",
                "metric": "val/avg_loss",
                "automl_max_epochs": 2,
                "automl_reduction_factor": 2,
                "epoch_multiplier": 1,
            },
            automl_hyperparameters=["train.optm_lr"],
            custom_param_ranges={
                "train.optm_lr": {"valid_min": 5e-7, "valid_max": 2e-6},
            },
        )

        first_rung = automl.next_recommendation()
        assert len(first_rung) == 2
        assert {rec.id for rec in first_rung} == {0, 1}
        assert all(rec.specs["train.epoch"] == 1 for rec in first_rung)
        first_rung_specs = {rec.id: dict(rec.specs) for rec in first_rung}

        first_rung[0].assign_job_id("job-rec0-epoch1")
        first_rung[1].assign_job_id("job-rec1-epoch1")

        # Lower is better for val/avg_loss, so rec 1 should be promoted.
        automl.report_result(first_rung[0].id, 0.9, status="success")
        automl.report_result(first_rung[1].id, 0.4, status="success")

        promoted = automl.next_recommendation()
        assert len(promoted) == 1
        assert promoted[0].id == 1
        assert promoted[0].specs["train.epoch"] == 2
        assert promoted[0].specs["train.optm_lr"] == first_rung_specs[1]["train.optm_lr"]
        assert promoted[0].resume_from_job_id == "job-rec1-epoch1"

        promoted[0].assign_job_id("job-rec1-epoch2")
        automl.report_result(promoted[0].id, 0.3, status="success")

        assert automl.next_recommendation() == []
        assert automl.is_complete()
        best = automl.get_best()
        assert best.id == 1
        assert best.result == 0.3


# ---------------------------------------------------------------
# 5. AlgorithmParams tests
# ---------------------------------------------------------------

def test_algorithm_params():
    from tao_automl.brain.factory import AlgorithmParams
    p = AlgorithmParams.from_dict({"automl_max_recommendations": 15})
    assert p.automl_max_recommendations == 15
    assert p.automl_reduction_factor == 3  # default


# ---------------------------------------------------------------
# 6. Concurrent report_result
# ---------------------------------------------------------------

def test_concurrent_reports():
    with tempfile.TemporaryDirectory() as d:
        ctrl, _, _ = _make_ctrl(d, 8)
        all_recs = []
        for _ in range(8):
            all_recs.extend(ctrl.next_recommendation())

        errors = []
        def report(rid, m):
            try:
                ctrl.report_result(rid, m)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=report, args=(r.id, 0.5 - r.id * 0.05)) for r in all_recs]
        for t in threads: t.start()
        for t in threads: t.join()

        assert not errors
        assert ctrl.is_complete()


# ---------------------------------------------------------------
# 7. Wheel metadata
# ---------------------------------------------------------------

def test_version():
    import tao_automl
    assert tao_automl.__version__ == "0.1.0"


def test_pip_show():
    r = subprocess.run(["pip", "show", "nvidia-tao-automl"], capture_output=True, text=True)
    assert r.returncode == 0
    assert "0.1.0" in r.stdout
