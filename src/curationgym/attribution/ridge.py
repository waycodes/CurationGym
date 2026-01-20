"""Ridge regression attribution for slice-to-benchmark analysis."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from curationgym.attribution.composition import CompositionVector


@dataclass
class AttributionCoefficient:
    """Attribution coefficient with uncertainty."""

    slice_name: str
    value: float
    ci_low: float
    ci_high: float
    significant: bool  # CI does not cross zero


@dataclass
class RidgeAttributionResult:
    """Result from ridge regression attribution."""

    benchmark: str
    coefficients: list[AttributionCoefficient]
    r_squared: float
    n_runs: int
    intercept: float
    method: str = "ridge_regression"

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "benchmark": self.benchmark,
            "r_squared": self.r_squared,
            "n_runs": self.n_runs,
            "intercept": self.intercept,
            "coefficients": {
                c.slice_name: {
                    "value": c.value,
                    "ci_low": c.ci_low,
                    "ci_high": c.ci_high,
                    "significant": c.significant,
                }
                for c in self.coefficients
            },
        }


class RidgeAttribution:
    """Ridge regression for slice-to-benchmark attribution."""

    def __init__(self, alpha: float = 1.0, n_bootstrap: int = 1000, seed: int = 42):
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.seed = seed

    def fit(
        self,
        compositions: list[CompositionVector],
        scores: list[float],
        benchmark: str,
    ) -> RidgeAttributionResult:
        """Fit ridge regression and compute bootstrapped CIs.

        Args:
            compositions: List of composition vectors (one per run)
            scores: Corresponding benchmark scores
            benchmark: Name of benchmark

        Returns:
            RidgeAttributionResult with coefficients and uncertainty
        """
        # Get all slice names
        all_slices = set()
        for comp in compositions:
            all_slices.update(comp.slice_token_fractions.keys())
        slice_names = sorted(all_slices)

        if not slice_names:
            return RidgeAttributionResult(
                benchmark=benchmark,
                coefficients=[],
                r_squared=0.0,
                n_runs=len(compositions),
                intercept=np.mean(scores) if scores else 0.0,
            )

        # Build feature matrix
        X = np.array([comp.to_feature_vector(slice_names) for comp in compositions])
        y = np.array(scores)

        # Fit ridge regression
        coef, intercept, r_squared = self._fit_ridge(X, y)

        # Bootstrap for confidence intervals
        coef_samples = self._bootstrap(X, y)

        # Compute CIs
        coefficients = []
        for i, slice_name in enumerate(slice_names):
            samples = coef_samples[:, i]
            ci_low = float(np.percentile(samples, 2.5))
            ci_high = float(np.percentile(samples, 97.5))
            significant = (ci_low > 0) or (ci_high < 0)

            coefficients.append(AttributionCoefficient(
                slice_name=slice_name,
                value=float(coef[i]),
                ci_low=ci_low,
                ci_high=ci_high,
                significant=significant,
            ))

        return RidgeAttributionResult(
            benchmark=benchmark,
            coefficients=coefficients,
            r_squared=r_squared,
            n_runs=len(compositions),
            intercept=float(intercept),
        )

    def _fit_ridge(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Fit ridge regression."""
        n, p = X.shape

        # Add regularization
        XtX = X.T @ X + self.alpha * np.eye(p)
        Xty = X.T @ y

        # Solve
        coef = np.linalg.solve(XtX, Xty)
        intercept = np.mean(y) - np.mean(X @ coef)

        # R-squared
        y_pred = X @ coef + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return coef, intercept, r_squared

    def _bootstrap(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bootstrap coefficient estimates."""
        rng = np.random.RandomState(self.seed)
        n = len(y)
        coef_samples = []

        for _ in range(self.n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            X_boot = X[idx]
            y_boot = y[idx]
            coef, _, _ = self._fit_ridge(X_boot, y_boot)
            coef_samples.append(coef)

        return np.array(coef_samples)


def run_ridge_attribution(
    compositions: list[CompositionVector],
    eval_results: dict[str, list[float]],  # benchmark -> scores per run
    alpha: float = 1.0,
) -> dict[str, RidgeAttributionResult]:
    """Run ridge attribution for all benchmarks.

    Args:
        compositions: Composition vectors (one per run)
        eval_results: Dict mapping benchmark name to list of scores
        alpha: Ridge regularization strength

    Returns:
        Dict mapping benchmark to attribution result
    """
    ridge = RidgeAttribution(alpha=alpha)
    results = {}

    for benchmark, scores in eval_results.items():
        if len(scores) != len(compositions):
            continue
        results[benchmark] = ridge.fit(compositions, scores, benchmark)

    return results


def save_ridge_results(
    results: dict[str, RidgeAttributionResult],
    path: str | Path,
) -> None:
    """Save ridge attribution results to JSON."""
    data = {benchmark: result.to_dict() for benchmark, result in results.items()}
    Path(path).write_text(json.dumps(data, indent=2))
