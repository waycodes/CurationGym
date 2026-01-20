"""Constrained optimization with feasibility-first ranking."""

from dataclasses import dataclass
from typing import Any

from curationgym.optim.search_space import Constraints
from curationgym.optim.random_search import TrialResult


@dataclass
class ConstraintViolation:
    """Details of a constraint violation."""

    constraint_name: str
    actual_value: float
    limit_value: float
    violation_amount: float  # How much over/under the limit


def compute_violations(
    trial: TrialResult,
    constraints: Constraints,
) -> list[ConstraintViolation]:
    """Compute detailed constraint violations for a trial."""
    violations = []

    compute_hours = trial.compute_cost
    if compute_hours > constraints.max_compute_hours:
        violations.append(ConstraintViolation(
            "compute_hours",
            compute_hours,
            constraints.max_compute_hours,
            compute_hours - constraints.max_compute_hours,
        ))

    contamination = trial.metrics.get("contamination_rate", 0.0)
    if contamination > constraints.max_contamination_rate:
        violations.append(ConstraintViolation(
            "contamination_rate",
            contamination,
            constraints.max_contamination_rate,
            contamination - constraints.max_contamination_rate,
        ))

    tokens = trial.dataset_tokens
    if tokens < constraints.min_dataset_tokens:
        violations.append(ConstraintViolation(
            "dataset_tokens",
            float(tokens),
            float(constraints.min_dataset_tokens),
            constraints.min_dataset_tokens - tokens,
        ))

    diversity = trial.metrics.get("diversity_score", 1.0)
    if diversity < constraints.min_diversity_score:
        violations.append(ConstraintViolation(
            "diversity_score",
            diversity,
            constraints.min_diversity_score,
            constraints.min_diversity_score - diversity,
        ))

    return violations


def feasibility_first_rank(
    trials: list[TrialResult],
    constraints: Constraints,
) -> list[TrialResult]:
    """Rank trials with feasibility-first ordering.

    Feasible solutions always rank above infeasible ones.
    Among feasible: rank by score (higher is better).
    Among infeasible: rank by total violation (lower is better).
    """
    feasible = []
    infeasible = []

    for trial in trials:
        violations = compute_violations(trial, constraints)
        if not violations:
            feasible.append((trial, 0.0))
        else:
            total_violation = sum(v.violation_amount for v in violations)
            infeasible.append((trial, total_violation))

    # Sort feasible by score (descending)
    feasible.sort(key=lambda x: x[0].score, reverse=True)

    # Sort infeasible by violation (ascending - less violation is better)
    infeasible.sort(key=lambda x: x[1])

    # Feasible first, then infeasible
    return [t for t, _ in feasible] + [t for t, _ in infeasible]


def apply_constraint_penalty(
    score: float,
    trial: TrialResult,
    constraints: Constraints,
    penalty_weight: float = 1.0,
) -> float:
    """Apply penalty to score based on constraint violations.

    For use in unconstrained optimizers that need soft constraints.
    """
    violations = compute_violations(trial, constraints)

    if not violations:
        return score

    # Compute normalized penalty
    total_penalty = 0.0
    for v in violations:
        # Normalize violation by limit
        if v.limit_value != 0:
            normalized = v.violation_amount / abs(v.limit_value)
        else:
            normalized = v.violation_amount
        total_penalty += normalized

    return score - penalty_weight * total_penalty


def reject_infeasible(
    trials: list[TrialResult],
    constraints: Constraints,
) -> list[TrialResult]:
    """Filter out infeasible trials."""
    return [t for t in trials if not compute_violations(t, constraints)]
