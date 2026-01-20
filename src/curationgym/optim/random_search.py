"""Random search optimizer - baseline for policy search."""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from curationgym.optim.search_space import SearchSpace, Constraints
from curationgym.policy.schema import Policy


@dataclass
class TrialResult:
    """Result from a single optimization trial."""

    trial_id: int
    config: dict[str, Any]
    score: float
    is_feasible: bool
    compute_cost: float = 0.0
    dataset_tokens: int = 0
    violations: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Result from optimization run."""

    best_config: dict[str, Any]
    best_score: float
    all_trials: list[TrialResult]
    num_feasible: int
    total_compute_hours: float


class RandomSearchOptimizer:
    """Random search over policy configurations."""

    def __init__(
        self,
        search_space: SearchSpace,
        constraints: Constraints | None = None,
        seed: int = 42,
    ):
        self.search_space = search_space
        self.constraints = constraints or Constraints()
        self.rng = random.Random(seed)
        self._trials: list[TrialResult] = []

    def optimize(
        self,
        objective_fn: Callable[[dict[str, Any]], tuple[float, dict[str, Any]]],
        n_trials: int = 20,
        top_k: int = 5,
    ) -> OptimizationResult:
        """Run random search optimization.

        Args:
            objective_fn: Function that takes config and returns (score, metrics_dict)
            n_trials: Number of random configurations to try
            top_k: Number of top results to track

        Returns:
            OptimizationResult with best configuration
        """
        self._trials = []

        for trial_id in range(n_trials):
            # Sample configuration
            config = self.search_space.sample(self.rng)

            # Evaluate
            try:
                score, metrics = objective_fn(config)
            except Exception as e:
                score = float("-inf")
                metrics = {"error": str(e)}

            # Check constraints
            compute_cost = metrics.get("compute_hours", 0.0)
            contamination = metrics.get("contamination_rate", 0.0)
            tokens = metrics.get("dataset_tokens", 0)
            diversity = metrics.get("diversity_score", 1.0)

            is_feasible, violations = self.constraints.is_feasible(
                compute_cost, contamination, tokens, diversity
            )

            trial = TrialResult(
                trial_id=trial_id,
                config=config,
                score=score if is_feasible else float("-inf"),
                is_feasible=is_feasible,
                compute_cost=compute_cost,
                dataset_tokens=tokens,
                violations=violations,
                metrics=metrics,
            )
            self._trials.append(trial)

        # Find best feasible result
        feasible_trials = [t for t in self._trials if t.is_feasible]
        if feasible_trials:
            best = max(feasible_trials, key=lambda t: t.score)
        else:
            # Fall back to least-violating
            best = max(self._trials, key=lambda t: t.score)

        return OptimizationResult(
            best_config=best.config,
            best_score=best.score,
            all_trials=sorted(self._trials, key=lambda t: t.score, reverse=True)[:top_k],
            num_feasible=len(feasible_trials),
            total_compute_hours=sum(t.compute_cost for t in self._trials),
        )

    def save_results(self, result: OptimizationResult, path: str | Path) -> None:
        """Save optimization results to JSON."""
        data = {
            "best_config": result.best_config,
            "best_score": result.best_score,
            "num_feasible": result.num_feasible,
            "total_compute_hours": result.total_compute_hours,
            "top_trials": [
                {
                    "trial_id": t.trial_id,
                    "config": t.config,
                    "score": t.score,
                    "is_feasible": t.is_feasible,
                    "compute_cost": t.compute_cost,
                }
                for t in result.all_trials
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2))
