"""Sanity checks for attribution validity."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from curationgym.attribution.composition import CompositionVector
from curationgym.attribution.ridge import RidgeAttribution


@dataclass
class SanityCheckResult:
    """Result from attribution sanity checks."""

    permutation_p_value: float  # p-value from permutation test
    seed_stability: float  # Correlation across seeds (0-1)
    negative_control_passed: bool  # Random slice has ~0 coefficient
    overall_valid: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "permutation_p_value": self.permutation_p_value,
            "seed_stability": self.seed_stability,
            "negative_control_passed": self.negative_control_passed,
            "overall_valid": self.overall_valid,
        }


class AttributionSanityChecker:
    """Run sanity checks on attribution results."""

    def __init__(self, n_permutations: int = 1000, seed: int = 42):
        self.n_permutations = n_permutations
        self.seed = seed

    def run_all_checks(
        self,
        compositions: list[CompositionVector],
        scores: list[float],
        observed_r_squared: float,
    ) -> SanityCheckResult:
        """Run all sanity checks.

        Args:
            compositions: Composition vectors
            scores: Benchmark scores
            observed_r_squared: R² from actual attribution

        Returns:
            SanityCheckResult with all check outcomes
        """
        perm_p = self.permutation_test(compositions, scores, observed_r_squared)
        stability = self.seed_stability_check(compositions, scores)
        neg_control = self.negative_control_check(compositions, scores)

        # Overall validity: p < 0.05, stability > 0.7, negative control passes
        overall = (perm_p < 0.05) and (stability > 0.7) and neg_control

        return SanityCheckResult(
            permutation_p_value=perm_p,
            seed_stability=stability,
            negative_control_passed=neg_control,
            overall_valid=overall,
        )

    def permutation_test(
        self,
        compositions: list[CompositionVector],
        scores: list[float],
        observed_r_squared: float,
    ) -> float:
        """Test if attribution is better than random.

        Permute scores and refit; count how often permuted R² >= observed.
        """
        rng = np.random.RandomState(self.seed)
        scores_array = np.array(scores)

        count_better = 0
        ridge = RidgeAttribution(n_bootstrap=10)  # Fewer bootstraps for speed

        for _ in range(self.n_permutations):
            permuted_scores = rng.permutation(scores_array).tolist()
            result = ridge.fit(compositions, permuted_scores, "permuted")
            if result.r_squared >= observed_r_squared:
                count_better += 1

        return (count_better + 1) / (self.n_permutations + 1)

    def seed_stability_check(
        self,
        compositions: list[CompositionVector],
        scores: list[float],
        n_seeds: int = 5,
    ) -> float:
        """Check if coefficients are stable across bootstrap seeds.

        Returns correlation between coefficient vectors from different seeds.
        """
        coef_vectors = []

        for seed in range(n_seeds):
            ridge = RidgeAttribution(seed=seed, n_bootstrap=100)
            result = ridge.fit(compositions, scores, "stability_check")
            coefs = [c.value for c in result.coefficients]
            if coefs:
                coef_vectors.append(coefs)

        if len(coef_vectors) < 2:
            return 1.0

        # Compute average pairwise correlation
        correlations = []
        for i in range(len(coef_vectors)):
            for j in range(i + 1, len(coef_vectors)):
                corr = np.corrcoef(coef_vectors[i], coef_vectors[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

        return float(np.mean(correlations)) if correlations else 1.0

    def negative_control_check(
        self,
        compositions: list[CompositionVector],
        scores: list[float],
    ) -> bool:
        """Check that random noise slice has ~0 coefficient.

        Add a random slice and verify its coefficient is not significant.
        """
        rng = np.random.RandomState(self.seed)

        # Add random slice to compositions
        augmented = []
        for comp in compositions:
            new_comp = CompositionVector(
                run_id=comp.run_id,
                slice_token_fractions={
                    **comp.slice_token_fractions,
                    "_random_control": rng.random(),
                },
                slice_doc_fractions=comp.slice_doc_fractions,
                slice_avg_quality=comp.slice_avg_quality,
                total_tokens=comp.total_tokens,
                total_docs=comp.total_docs,
            )
            augmented.append(new_comp)

        # Fit and check random slice coefficient
        ridge = RidgeAttribution(n_bootstrap=500)
        result = ridge.fit(augmented, scores, "negative_control")

        for coef in result.coefficients:
            if coef.slice_name == "_random_control":
                # Should not be significant
                return not coef.significant

        return True  # Random slice not found (shouldn't happen)


def save_sanity_checks(
    results: dict[str, SanityCheckResult],
    path: str | Path,
) -> None:
    """Save sanity check results to JSON."""
    data = {benchmark: result.to_dict() for benchmark, result in results.items()}
    Path(path).write_text(json.dumps(data, indent=2))
