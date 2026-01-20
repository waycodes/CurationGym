"""Optuna TPE optimizer for sample-efficient policy search."""

import json
from pathlib import Path
from typing import Any, Callable

from curationgym.optim.search_space import SearchSpace, Constraints
from curationgym.optim.random_search import TrialResult, OptimizationResult


class OptunaTPEOptimizer:
    """Tree-structured Parzen Estimator optimizer using Optuna."""

    def __init__(
        self,
        search_space: SearchSpace,
        constraints: Constraints | None = None,
        storage: str | None = None,  # SQLite path for persistence
        study_name: str = "curationgym_study",
        seed: int = 42,
    ):
        self.search_space = search_space
        self.constraints = constraints or Constraints()
        self.storage = storage
        self.study_name = study_name
        self.seed = seed
        self._study = None

    def _create_study(self):
        """Create or load Optuna study."""
        try:
            import optuna
        except ImportError:
            raise ImportError("optuna required: pip install optuna")

        storage_url = f"sqlite:///{self.storage}" if self.storage else None

        self._study = optuna.create_study(
            study_name=self.study_name,
            storage=storage_url,
            load_if_exists=True,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
        )

    def _suggest_config(self, trial) -> dict[str, Any]:
        """Suggest configuration using Optuna trial."""
        config = {}
        for param in self.search_space.parameters:
            if param.param_type == "categorical":
                config[param.name] = trial.suggest_categorical(param.name, param.choices)
            elif param.param_type == "int":
                config[param.name] = trial.suggest_int(
                    param.name, int(param.low), int(param.high), log=param.log_scale
                )
            elif param.param_type == "float":
                config[param.name] = trial.suggest_float(
                    param.name, param.low, param.high, log=param.log_scale
                )
        return config

    def optimize(
        self,
        objective_fn: Callable[[dict[str, Any]], tuple[float, dict[str, Any]]],
        n_trials: int = 50,
        timeout: float | None = None,
        cost_weight: float = 0.0,  # Weight for compute cost in objective
    ) -> OptimizationResult:
        """Run TPE optimization.

        Args:
            objective_fn: Function(config) -> (score, metrics)
            n_trials: Number of trials
            timeout: Optional timeout in seconds
            cost_weight: Weight for penalizing compute cost (0 = ignore cost)

        Returns:
            OptimizationResult with best configuration
        """
        import optuna

        self._create_study()
        trials: list[TrialResult] = []

        def optuna_objective(trial):
            config = self._suggest_config(trial)

            try:
                score, metrics = objective_fn(config)
            except Exception as e:
                raise optuna.TrialPruned(f"Error: {e}")

            # Check constraints
            compute_cost = metrics.get("compute_hours", 0.0)
            contamination = metrics.get("contamination_rate", 0.0)
            tokens = metrics.get("dataset_tokens", 0)
            diversity = metrics.get("diversity_score", 1.0)

            is_feasible, violations = self.constraints.is_feasible(
                compute_cost, contamination, tokens, diversity
            )

            # Store trial result
            trial_result = TrialResult(
                trial_id=trial.number,
                config=config,
                score=score,
                is_feasible=is_feasible,
                compute_cost=compute_cost,
                dataset_tokens=tokens,
                violations=violations,
                metrics=metrics,
            )
            trials.append(trial_result)

            # Report metrics to Optuna
            trial.set_user_attr("is_feasible", is_feasible)
            trial.set_user_attr("compute_cost", compute_cost)

            if not is_feasible:
                raise optuna.TrialPruned(f"Infeasible: {violations}")

            # Incorporate compute cost into objective
            if cost_weight > 0:
                import math
                score = score - cost_weight * math.log(1 + compute_cost)

            return score

        # Run optimization
        self._study.optimize(
            optuna_objective,
            n_trials=n_trials,
            timeout=timeout,
            catch=(Exception,),
        )

        # Get best result
        best_trial = self._study.best_trial
        best_config = {p.name: best_trial.params.get(p.name, p.default) for p in self.search_space.parameters}

        feasible_trials = [t for t in trials if t.is_feasible]

        return OptimizationResult(
            best_config=best_config,
            best_score=best_trial.value if best_trial.value else 0.0,
            all_trials=sorted(trials, key=lambda t: t.score, reverse=True)[:10],
            num_feasible=len(feasible_trials),
            total_compute_hours=sum(t.compute_cost for t in trials),
        )

    def save_study(self, path: str | Path) -> None:
        """Export study results to JSON."""
        if self._study is None:
            return

        data = {
            "study_name": self.study_name,
            "n_trials": len(self._study.trials),
            "best_value": self._study.best_value,
            "best_params": self._study.best_params,
            "trials": [
                {
                    "number": t.number,
                    "value": t.value,
                    "params": t.params,
                    "state": str(t.state),
                }
                for t in self._study.trials
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2))
