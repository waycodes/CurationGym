"""Pareto frontier tracking for multi-objective optimization."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from curationgym.optim.random_search import TrialResult


@dataclass
class ParetoPoint:
    """A point on the Pareto frontier."""

    trial_id: int
    config: dict[str, Any]
    objectives: dict[str, float]  # objective_name -> value
    is_pareto_optimal: bool = True


class ParetoFrontier:
    """Track Pareto-optimal solutions for multi-objective optimization.

    Default objectives: (score_total, -compute_cost, dataset_tokens)
    All objectives are maximized (negate costs).
    """

    def __init__(self, objective_names: list[str] | None = None):
        self.objective_names = objective_names or ["score", "neg_compute", "tokens"]
        self._points: list[ParetoPoint] = []
        self._frontier: list[ParetoPoint] = []

    def add_trial(self, trial: TrialResult) -> bool:
        """Add trial and update Pareto frontier.

        Returns:
            True if trial is Pareto-optimal
        """
        objectives = {
            "score": trial.score,
            "neg_compute": -trial.compute_cost,  # Negate so higher is better
            "tokens": float(trial.dataset_tokens),
        }

        point = ParetoPoint(
            trial_id=trial.trial_id,
            config=trial.config,
            objectives=objectives,
        )

        # Check if dominated by any existing point
        is_dominated = False
        for existing in self._frontier:
            if self._dominates(existing.objectives, objectives):
                is_dominated = True
                break

        if is_dominated:
            point.is_pareto_optimal = False
            self._points.append(point)
            return False

        # Remove points dominated by new point
        self._frontier = [
            p for p in self._frontier
            if not self._dominates(objectives, p.objectives)
        ]

        # Add to frontier
        point.is_pareto_optimal = True
        self._frontier.append(point)
        self._points.append(point)
        return True

    def _dominates(self, a: dict[str, float], b: dict[str, float]) -> bool:
        """Check if point a dominates point b.

        a dominates b if a is >= b in all objectives and > in at least one.
        """
        dominated_in_all = True
        strictly_better_in_one = False

        for obj in self.objective_names:
            a_val = a.get(obj, float("-inf"))
            b_val = b.get(obj, float("-inf"))

            if a_val < b_val:
                dominated_in_all = False
                break
            if a_val > b_val:
                strictly_better_in_one = True

        return dominated_in_all and strictly_better_in_one

    @property
    def frontier(self) -> list[ParetoPoint]:
        """Get current Pareto frontier."""
        return list(self._frontier)

    @property
    def all_points(self) -> list[ParetoPoint]:
        """Get all tracked points."""
        return list(self._points)

    def get_frontier_configs(self) -> list[dict[str, Any]]:
        """Get configurations on the Pareto frontier."""
        return [p.config for p in self._frontier]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "objective_names": self.objective_names,
            "frontier": [
                {
                    "trial_id": p.trial_id,
                    "objectives": p.objectives,
                    "config": p.config,
                }
                for p in self._frontier
            ],
            "all_points": [
                {
                    "trial_id": p.trial_id,
                    "objectives": p.objectives,
                    "is_pareto_optimal": p.is_pareto_optimal,
                }
                for p in self._points
            ],
        }

    def save(self, path: str | Path) -> None:
        """Save Pareto frontier to JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ParetoFrontier":
        """Load Pareto frontier from JSON."""
        data = json.loads(Path(path).read_text())
        frontier = cls(objective_names=data.get("objective_names"))

        for p_data in data.get("frontier", []):
            point = ParetoPoint(
                trial_id=p_data["trial_id"],
                config=p_data["config"],
                objectives=p_data["objectives"],
                is_pareto_optimal=True,
            )
            frontier._frontier.append(point)
            frontier._points.append(point)

        return frontier

    def summary(self) -> str:
        """Get summary of Pareto frontier."""
        if not self._frontier:
            return "Empty Pareto frontier"

        lines = [f"Pareto frontier: {len(self._frontier)} points"]
        for p in self._frontier:
            obj_str = ", ".join(f"{k}={v:.4f}" for k, v in p.objectives.items())
            lines.append(f"  Trial {p.trial_id}: {obj_str}")
        return "\n".join(lines)
