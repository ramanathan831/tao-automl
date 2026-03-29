# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NVIDIA TAO AutoML - Standalone hyperparameter search library.

Usage::

    from tao_automl import AutoML

    automl = AutoML(
        workspace="/path/to/workspace",
        network="dino",
        train_specs=specs,
        settings={"algorithm": "bayesian", "metric": "loss"},
    )

    while not automl.is_complete():
        rec = automl.next_recommendation()
        if not rec:
            continue
        # caller runs training with rec.specs (or rec[0].specs for parallel)
        metric_value = run_training(rec[0].specs)
        automl.report_result(rec[0].id, metric_value)

    best = automl.get_best()
"""

__version__ = "0.1.0"

import logging
import uuid

from tao_automl.types import AutoMLContext

logger = logging.getLogger(__name__)


class AutoML:
    """Main entry point for TAO AutoML hyperparameter optimization.

    The caller is responsible for running training; this class only manages
    the search loop (generating recommendations, tracking results, deciding
    when to stop).

    Example::

        automl = AutoML(
            workspace="/tmp/my_experiment",
            network="dino",
            train_specs=my_train_spec_dict,
            settings={"algorithm": "bayesian", "metric": "loss",
                       "automl_max_recommendations": 20},
        )

        while not automl.is_complete():
            recs = automl.next_recommendation()
            for rec in recs:
                metric_value = train_model(rec.specs)
                automl.report_result(rec.id, metric_value)

        print("Best:", automl.get_best().specs)
    """

    def __init__(
        self,
        workspace,
        network,
        train_specs,
        settings,
        automl_hyperparameters=None,
        custom_param_ranges=None,
        resume=False,
    ):
        """
        Args:
            workspace: Path to workspace directory for state persistence.
            network: Network architecture name (e.g. ``"dino"``).
            train_specs: Training spec dict (the base configuration).
            settings: Dict with keys ``algorithm``, ``metric``, and any
                algorithm-specific parameters accepted by
                :class:`~tao_automl.brain.factory.AlgorithmParams`.
            automl_hyperparameters: List of dotted parameter names to search.
                If *None*, every parameter marked ``automl_enabled`` in the
                network schema is included.
            custom_param_ranges: Optional dict mapping parameter names to
                custom range overrides (e.g.
                ``{"train.optim.lr": {"valid_min": 1e-5, "valid_max": 1e-2}}``).
            resume: Whether to resume from previously persisted state in
                *workspace*.
        """
        # Lazy imports to avoid pulling in heavy deps (requests, omegaconf)
        # at package import time.
        from tao_automl.brain.factory import AlgorithmParams, BrainFactory
        from tao_automl.controller.controller import Controller
        from tao_automl.search_space.params import generate_hyperparams_to_search
        from tao_automl.state.state_store import StateStore

        if not settings or "algorithm" not in settings:
            raise ValueError("settings must include at least an 'algorithm' key")

        algorithm = settings["algorithm"]
        metric = settings.get("metric", "loss")

        # 1. State store
        self._state_store = StateStore(workspace)

        # 2. Context
        session_id = settings.get("session_id", uuid.uuid4().hex[:12])
        self._context = AutoMLContext(
            id=session_id,
            network=network,
            action="train",
            workspace_path=workspace,
            metric=metric,
            handler_id=settings.get("experiment_id", session_id),
        )

        # 3. Persist the training spec so the brain can read it
        self._state_store.save_job_specs(self._context.id, train_specs)

        # 4. Custom parameter ranges
        if custom_param_ranges:
            self._state_store.save_custom_param_ranges(
                self._context.handler_id, custom_param_ranges
            )

        # 5. Generate search space
        if automl_hyperparameters is None:
            # Caller did not specify; we will pass an empty list so that
            # generate_hyperparams_to_search enables only schema-default params.
            automl_hyperparameters = []

        param_records, param_names = generate_hyperparams_to_search(
            network=network,
            action="train",
            train_specs=train_specs,
            automl_hyperparameters=automl_hyperparameters,
        )

        if not param_records or param_records == [{}]:
            logger.warning(
                "No searchable parameters found for network '%s'. "
                "Check that automl_hyperparameters match the schema.", network
            )

        # 6. Algorithm params
        algo_params = AlgorithmParams.from_dict(settings)

        # 7. Brain
        brain = BrainFactory.create_brain(
            algorithm=algorithm,
            context=self._context,
            state_store=self._state_store,
            network=network,
            parameters=param_records,
            params=algo_params,
            metric=metric,
            resume=resume,
        )

        # 8. Controller
        if resume:
            self._controller = Controller.load_state(
                brain=brain,
                context=self._context,
                state_store=self._state_store,
                settings=algo_params,
                metric=metric,
                algorithm=algorithm,
                parameter_names=param_names,
            )
        else:
            self._controller = Controller(
                brain=brain,
                context=self._context,
                state_store=self._state_store,
                settings=algo_params,
                metric=metric,
                algorithm=algorithm,
                parameter_names=param_names,
            )

        logger.info(
            "AutoML initialized: algorithm=%s, metric=%s, params=%d, resume=%s",
            algorithm, metric, len(param_names), resume,
        )

    # ------------------------------------------------------------------
    # Public API (delegates to controller)
    # ------------------------------------------------------------------

    def next_recommendation(self):
        """Get next recommendation(s).

        Returns:
            list of :class:`~tao_automl.types.Recommendation` objects.
            May be empty if the brain is waiting for results.
        """
        return self._controller.next_recommendation()

    def report_result(self, rec_id, metric_value, best_epoch=None, status="success"):
        """Report a training result back.

        Args:
            rec_id: Recommendation ID (from ``rec.id``).
            metric_value: The metric value achieved.
            best_epoch: Best epoch number (optional).
            status: ``"success"`` or ``"failure"``.
        """
        self._controller.report_result(rec_id, metric_value, best_epoch, status)

    def get_best(self):
        """Get the best Recommendation so far, or None."""
        return self._controller.get_best()

    def get_progress(self):
        """Get a progress summary dict.

        Returns:
            dict with keys ``completed``, ``total``, ``best_metric``,
            ``best_rec_id``, ``algorithm``.
        """
        return self._controller.get_progress()

    def get_history(self):
        """Get all Recommendation objects generated so far."""
        return self._controller.get_history()

    def is_complete(self):
        """Check if the optimization is done."""
        return self._controller.is_complete()
