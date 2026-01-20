"""Multi-fidelity scheduler for efficient policy search."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from curationgym.optim.search_space import SearchSpace
from curationgym.optim.random_search import TrialResult


@dataclass
class FidelityLevel:
    """Configuration for a fidelity level."""

    name: str
    budget_fraction: float  # Fraction of full budget
    promotion_fraction: float  # Fraction of trials to promote


@dataclass
class MultiFidelityConfig:
    """Multi-fidelity scheduler configuration."""

    levels: list[FidelityLevel] = field(default_factory=lambda: [
        FidelityLevel("small", 0.1, 0.3),   # 10% budget, promote top 30%
        FidelityLevel("medium", 0.3, 0.5),  # 30% budget, promote top 50%
        FidelityLevel("final", 1.0, 1.0),   # Full budget
    ])


class MultiFidelityScheduler:
    """Successive halving / multi-fidelity optimization.

    Evaluates many candidates cheaply, promotes promising ones to higher fidelity.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        config: MultiFidelityConfig | None = None,
        seed: int = 42,
    ):
        self.search_space = search_space
        self.config = config or MultiFidelityConfig()
        self.seed = seed
        self._promotion_log: list[dict[str, Any]] = []

    def optimize(
        self,
        objective_fn: Callable[[dict[str, Any], float], tuple[float, dict[str, Any]]],
        n_initial: int = 50,
    ) -> tuple[dict[str, Any], float, list[TrialResult]]:
        """Run multi-fidelity optimization.

        Args:
            objective_fn: Function(config, budget_fraction) -> (score, metrics)
            n_initial: Number of initial candidates

        Returns:
            (best_config, best_score, all_trials)
        """
        import random
        rng = random.Random(self.seed)

        # Generate initial candidates
        candidates = [self.search_space.sample(rng) for _ in range(n_initial)]
        all_trials: list[TrialResult] = []
        trial_id = 0

        for level_idx, level in enumerate(self.config.levels):
            level_results: list[TrialResult] = []

            for config in candidates:
                try:
                    score, metrics = objective_fn(config, level.budget_fraction)
                except Exception as e:
                    score = float("-inf")
                    metrics = {"error": str(e)}

                trial = TrialResult(
                    trial_id=trial_id,
                    config=config,
                    score=score,
                    is_feasible=score > float("-inf"),
                    compute_cost=metrics.get("compute_hours", 0.0),
                    dataset_tokens=metrics.get("dataset_tokens", 0),
                    metrics={**metrics, "fidelity_level": level.name},
                )
                level_results.append(trial)
                all_trials.append(trial)
                trial_id += 1

            # Sort by score and promote top fraction
            level_results.sort(key=lambda t: t.score, reverse=True)
            n_promote = max(1, int(len(level_results) * level.promotion_fraction))
            candidates = [t.config for t in level_results[:n_promote]]

            # Log promotion
            self._promotion_log.append({
                "level": level.name,
                "candidates_in": len(level_results),
                "candidates_promoted": n_promote,
                "best_score": level_results[0].score if level_results else 0,
            })

            if level_idx == len(self.config.levels) - 1:
                # Final level - return best
                best = level_results[0] if level_results else all_trials[0]
                return best.config, best.score, all_trials

        # Should not reach here
        best = max(all_trials, key=lambda t: t.score)
        return best.config, best.score, all_trials

    def save_promotion_log(self, path: str | Path) -> None:
        """Save promotion log to JSON."""
        Path(path).write_text(json.dumps(self._promotion_log, indent=2))
