"""Evaluation score aggregation methods."""

from typing import Protocol


class Aggregator(Protocol):
    """Protocol for score aggregation strategies."""

    def aggregate(self, scores: dict[str, float], weights: dict[str, float]) -> float:
        """Aggregate task scores into a single metric."""
        ...


class WeightedMeanAggregator:
    """Weighted arithmetic mean of task scores."""

    def aggregate(self, scores: dict[str, float], weights: dict[str, float]) -> float:
        total_weight = sum(weights.get(k, 1.0) for k in scores)
        if total_weight == 0:
            return 0.0
        return sum(scores[k] * weights.get(k, 1.0) for k in scores) / total_weight


class GeometricMeanAggregator:
    """Geometric mean of task scores."""

    def aggregate(self, scores: dict[str, float], weights: dict[str, float]) -> float:
        if not scores:
            return 0.0
        product = 1.0
        for score in scores.values():
            product *= max(score, 1e-10)
        return product ** (1.0 / len(scores))


AGGREGATORS: dict[str, type[Aggregator]] = {
    "weighted_mean": WeightedMeanAggregator,
    "geometric_mean": GeometricMeanAggregator,
}


def get_aggregator(method: str) -> Aggregator:
    """Get aggregator by name."""
    if method not in AGGREGATORS:
        raise ValueError(f"Unknown aggregation method: {method}")
    return AGGREGATORS[method]()
