"""Evaluation aggregation with confidence intervals."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from curationgym.eval.aggregators import get_aggregator
from curationgym.eval.runner import EvalResult


@dataclass
class AggregateResult:
    """Aggregated evaluation result."""

    score_total: float
    per_task: dict[str, float]
    confidence_intervals: dict[str, tuple[float, float]]
    aggregation_method: str
    num_tasks: int


def aggregate_eval_results(
    result: EvalResult,
    weights: dict[str, float] | None = None,
    method: str = "weighted_mean",
) -> AggregateResult:
    """Aggregate evaluation results into single score.

    Args:
        result: EvalResult from evaluation run
        weights: Optional task weights (default: equal)
        method: Aggregation method (weighted_mean, geometric_mean)

    Returns:
        AggregateResult with score_total and breakdown
    """
    if not result.task_scores:
        return AggregateResult(
            score_total=0.0,
            per_task={},
            confidence_intervals={},
            aggregation_method=method,
            num_tasks=0,
        )

    # Use provided weights or default to 1.0
    task_weights = weights or {t: 1.0 for t in result.task_scores}

    # Get aggregator
    aggregator = get_aggregator(method)
    score_total = aggregator.aggregate(result.task_scores, task_weights)

    # Propagate confidence intervals
    ci = {}
    for task, (low, high) in result.confidence_intervals.items():
        ci[task] = (low, high)

    # Compute aggregate CI using error propagation (simplified)
    if ci:
        total_weight = sum(task_weights.get(t, 1.0) for t in result.task_scores)
        variance_sum = 0.0
        for task, (low, high) in ci.items():
            stderr = (high - low) / (2 * 1.96)
            weight = task_weights.get(task, 1.0) / total_weight
            variance_sum += (weight * stderr) ** 2

        aggregate_stderr = math.sqrt(variance_sum)
        ci["aggregate"] = (
            score_total - 1.96 * aggregate_stderr,
            score_total + 1.96 * aggregate_stderr,
        )

    return AggregateResult(
        score_total=score_total,
        per_task=result.task_scores,
        confidence_intervals=ci,
        aggregation_method=method,
        num_tasks=len(result.task_scores),
    )


def save_aggregate_results(result: AggregateResult, path: str | Path) -> None:
    """Save aggregate results to JSON."""
    data = {
        "score_total": result.score_total,
        "per_task": result.per_task,
        "confidence_intervals": {k: list(v) for k, v in result.confidence_intervals.items()},
        "aggregation_method": result.aggregation_method,
        "num_tasks": result.num_tasks,
    }
    Path(path).write_text(json.dumps(data, indent=2))
