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

"""AutoML optimization loop controller.

The controller manages the brain algorithm, generates recommendations,
and tracks results.  It does NOT launch jobs -- the caller does that.
"""

import logging

from tao_automl.types import Recommendation, JobStates

logger = logging.getLogger(__name__)

# Algorithms whose completion is determined by brain.done()
_BRAIN_DONE_ALGORITHMS = frozenset({
    "hyperband", "h", "bohb", "asha", "dehb", "hyperband_es", "hes", "pbt",
})

# Algorithms whose completion is determined by max recommendations count
_MAX_REC_ALGORITHMS = frozenset({"bayesian", "b", "bfbo"})


class Controller:
    """AutoML optimization loop controller.

    The controller manages the brain algorithm, generates recommendations,
    and tracks results.  It does NOT launch jobs -- the caller does that.
    """

    def __init__(
        self,
        brain,
        context,
        state_store,
        settings,
        metric,
        algorithm,
        parameter_names=None,
    ):
        """
        Args:
            brain: Algorithm instance (Bayesian, Hyperband, etc.)
            context: AutoMLContext
            state_store: StateStore for persistence
            settings: AlgorithmParams
            metric: Optimization metric name
            algorithm: Algorithm name string
            parameter_names: List of parameter names being searched
        """
        self.brain = brain
        self.context = context
        self.state_store = state_store
        self.settings = settings
        self.metric = metric
        self.algorithm = algorithm.lower()
        self.parameter_names = parameter_names or []
        self.history = []  # list of Recommendation objects
        self._next_id = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def next_recommendation(self):
        """Get next hyperparameter recommendation(s) from the brain.

        Returns:
            list of Recommendation objects.  May be empty if the brain is
            waiting for results (e.g. Bayesian waits for the previous run
            to finish), or contain multiple entries for parallel algorithms
            like Hyperband.
        """
        raw_recs = self.brain.generate_recommendations(self.history)

        if not raw_recs:
            return []

        recommendations = []
        for spec_dict in raw_recs:
            if not spec_dict:
                continue
            rec = Recommendation(
                identifier=self._next_id,
                specs=spec_dict,
                metric=self.metric,
            )
            self.history.append(rec)
            recommendations.append(rec)
            self._next_id += 1

        # Persist after generating new recommendations
        self.save_state()
        return recommendations

    def report_result(self, rec_id, metric_value, best_epoch=None, status="success"):
        """Feed back a training result.

        Thread/process-safe: acquires the state store's global lock so
        concurrent ``report_result`` calls serialize their state writes.

        Args:
            rec_id: Recommendation ID (int).
            metric_value: The metric value achieved (float).
            best_epoch: Best epoch number (optional).
            status: ``"success"`` or ``"failure"``.
        """
        with self.state_store.lock():
            rec = self._find_rec(rec_id)
            if rec is None:
                logger.warning("report_result: recommendation %s not found", rec_id)
                return

            rec.update_result(metric_value)
            rec.update_status(status if status else JobStates.success)
            if best_epoch is not None:
                rec.best_epoch_number = best_epoch

            # Persist brain and controller state (under lock)
            self.brain.save_state()
            self.save_state()

        logger.info(
            "Reported result for rec %d: metric=%.6f status=%s",
            rec_id, metric_value, status,
        )

    def get_best(self):
        """Return the best Recommendation so far, or None.

        Uses the convention that if the metric name contains ``"loss"`` then
        lower is better; otherwise higher is better.
        """
        completed = [
            r for r in self.history
            if r.status in (JobStates.success, JobStates.done)
        ]
        if not completed:
            return None

        lower_is_better = "loss" in self.metric.lower()
        if lower_is_better:
            return min(completed, key=lambda r: r.result)
        return max(completed, key=lambda r: r.result)

    def get_progress(self):
        """Return a progress summary dict.

        Keys: ``completed``, ``total``, ``best_metric``, ``best_rec_id``,
        ``algorithm``.
        """
        completed_recs = [
            r for r in self.history
            if r.status in (JobStates.success, JobStates.done, JobStates.failure, JobStates.error)
        ]
        best = self.get_best()

        total = self._estimate_total()

        return {
            "completed": len(completed_recs),
            "total": total,
            "best_metric": best.result if best else None,
            "best_rec_id": best.id if best else None,
            "algorithm": self.algorithm,
        }

    def get_history(self):
        """Return the full list of Recommendation objects."""
        return list(self.history)

    def is_complete(self):
        """Check if the optimization loop is done.

        * Bayesian / BFBO: completed count >= ``settings.automl_max_recommendations``.
        * Hyperband / BOHB / ASHA / DEHB / HyperBandES / PBT: delegates to
          ``brain.done()`` and verifies no outstanding pending experiments.
        """
        if self.algorithm in _MAX_REC_ALGORITHMS:
            completed = sum(
                1 for r in self.history
                if r.status in (JobStates.success, JobStates.done, JobStates.failure, JobStates.error)
            )
            return completed >= self.settings.automl_max_recommendations

        if self.algorithm in _BRAIN_DONE_ALGORITHMS:
            if hasattr(self.brain, "done"):
                return self.brain.done()
            # Fallback: treat as complete when all history entries are terminal
            if not self.history:
                return False
            return all(
                r.status in (JobStates.success, JobStates.done, JobStates.failure, JobStates.error)
                for r in self.history
            )

        # Unknown algorithm -- conservative default
        logger.warning("is_complete: unknown algorithm '%s', defaulting to False", self.algorithm)
        return False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self):
        """Persist current controller state (history) to workspace."""
        serialized = [self._serialize_rec(r) for r in self.history]
        self.state_store.save_controller_info(self.context.id, serialized)

        # Also persist best recommendation
        best = self.get_best()
        if best is not None:
            self.state_store.save_best_rec_info(
                self.context.id,
                rec_number=best.id,
                rec_data=self._serialize_rec(best),
            )

    @classmethod
    def load_state(
        cls,
        brain,
        context,
        state_store,
        settings,
        metric,
        algorithm,
        parameter_names=None,
    ):
        """Load controller from persisted state.

        Returns a Controller instance with history restored from disk.
        """
        controller = cls(
            brain=brain,
            context=context,
            state_store=state_store,
            settings=settings,
            metric=metric,
            algorithm=algorithm,
            parameter_names=parameter_names,
        )

        saved = state_store.get_controller_info(context.id)
        if saved:
            for rec_dict in saved:
                rec = Recommendation(
                    identifier=int(rec_dict["id"]),
                    specs=rec_dict.get("specs", {}),
                    metric=metric,
                )
                rec.job_id = rec_dict.get("job_id")
                rec.status = rec_dict.get("status", JobStates.pending)
                rec.result = float(rec_dict.get("result", 0.0))
                rec.best_epoch_number = rec_dict.get("best_epoch_number", "")
                rec.resume_from_job_id = rec_dict.get("resume_from_job_id")
                rec.early_stop_epoch = rec_dict.get("early_stop_epoch")
                rec.created_on = rec_dict.get("created_on", "")
                rec.last_modified = rec_dict.get("last_modified", "")
                controller.history.append(rec)

            if controller.history:
                controller._next_id = max(r.id for r in controller.history) + 1

        logger.info(
            "Loaded controller state: %d recommendations, next_id=%d",
            len(controller.history), controller._next_id,
        )
        return controller

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_rec(self, rec_id):
        """Find a recommendation by ID in history."""
        for r in self.history:
            if r.id == rec_id:
                return r
        return None

    def _estimate_total(self):
        """Estimate total number of recommendations for progress reporting."""
        if self.algorithm in _MAX_REC_ALGORITHMS:
            return self.settings.automl_max_recommendations

        # For multi-fidelity algorithms the total is harder to know upfront.
        # Return the number of recommendations generated so far as a lower bound.
        return len(self.history)

    @staticmethod
    def _serialize_rec(rec):
        """Convert a Recommendation to a JSON-safe dict."""
        return {
            "id": rec.id,
            "specs": rec.specs,
            "job_id": rec.job_id,
            "status": rec.status,
            "result": rec.result,
            "best_epoch_number": rec.best_epoch_number,
            "metric": rec.metric,
            "resume_from_job_id": rec.resume_from_job_id,
            "early_stop_epoch": rec.early_stop_epoch,
            "created_on": rec.created_on,
            "last_modified": rec.last_modified,
        }
