"""Ablation-based attribution for causal slice analysis."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class AblationResult:
    """Result from ablating a single slice."""

    slice_name: str
    baseline_score: float
    ablated_score: float
    delta: float  # baseline - ablated (positive = slice helps)
    relative_delta: float  # delta / baseline


@dataclass
class AblationStudyResult:
    """Complete ablation study results."""

    benchmark: str
    baseline_score: float
    ablations: list[AblationResult]
    method: str = "ablation"

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "benchmark": self.benchmark,
            "baseline_score": self.baseline_score,
            "ablations": [
                {
                    "slice": a.slice_name,
                    "baseline": a.baseline_score,
                    "ablated": a.ablated_score,
                    "delta": a.delta,
                    "relative_delta": a.relative_delta,
                }
                for a in self.ablations
            ],
        }


class AblationAttribution:
    """Ablation-based attribution by removing slices and measuring impact."""

    def __init__(self, top_n_slices: int = 10):
        self.top_n_slices = top_n_slices

    def run_ablation_study(
        self,
        baseline_policy_config: dict[str, Any],
        baseline_score: float,
        slice_masses: dict[str, float],  # slice -> token fraction
        train_and_eval_fn: Callable[[dict[str, Any]], float],
        benchmark: str,
    ) -> AblationStudyResult:
        """Run ablation study on top slices.

        Args:
            baseline_policy_config: Configuration of best policy
            baseline_score: Score of baseline policy
            slice_masses: Token fraction per slice
            train_and_eval_fn: Function that trains and evaluates, returns score
            benchmark: Benchmark name

        Returns:
            AblationStudyResult with per-slice deltas
        """
        # Select top N slices by mass
        sorted_slices = sorted(slice_masses.items(), key=lambda x: x[1], reverse=True)
        top_slices = [s for s, _ in sorted_slices[:self.top_n_slices]]

        ablations = []
        for slice_name in top_slices:
            # Create ablated config (set slice weight to 0)
            ablated_config = self._create_ablated_config(baseline_policy_config, slice_name)

            # Train and evaluate
            try:
                ablated_score = train_and_eval_fn(ablated_config)
            except Exception:
                ablated_score = 0.0

            delta = baseline_score - ablated_score
            relative_delta = delta / baseline_score if baseline_score != 0 else 0.0

            ablations.append(AblationResult(
                slice_name=slice_name,
                baseline_score=baseline_score,
                ablated_score=ablated_score,
                delta=delta,
                relative_delta=relative_delta,
            ))

        # Sort by delta (most impactful first)
        ablations.sort(key=lambda a: abs(a.delta), reverse=True)

        return AblationStudyResult(
            benchmark=benchmark,
            baseline_score=baseline_score,
            ablations=ablations,
        )

    def _create_ablated_config(
        self,
        config: dict[str, Any],
        slice_to_remove: str,
    ) -> dict[str, Any]:
        """Create config with slice removed (weight = 0)."""
        ablated = config.copy()

        # Set slice weight to 0
        slice_weights = ablated.get("slice_weights", {}).copy()
        slice_weights[slice_to_remove] = 0.0
        ablated["slice_weights"] = slice_weights

        return ablated


def save_ablation_results(
    results: dict[str, AblationStudyResult],
    path: str | Path,
) -> None:
    """Save ablation results to CSV-friendly format."""
    rows = []
    for benchmark, result in results.items():
        for ablation in result.ablations:
            rows.append({
                "benchmark": benchmark,
                "slice": ablation.slice_name,
                "baseline_score": ablation.baseline_score,
                "ablated_score": ablation.ablated_score,
                "delta": ablation.delta,
                "relative_delta": ablation.relative_delta,
            })

    # Save as JSON (can convert to CSV)
    Path(path).write_text(json.dumps(rows, indent=2))


def create_ablation_table(results: dict[str, AblationStudyResult]) -> str:
    """Create markdown table of ablation results."""
    lines = ["| Slice | Benchmark | Delta | Relative Delta |", "|-------|-----------|-------|----------------|"]

    for benchmark, result in results.items():
        for ablation in result.ablations:
            lines.append(
                f"| {ablation.slice_name} | {benchmark} | "
                f"{ablation.delta:.4f} | {ablation.relative_delta:.2%} |"
            )

    return "\n".join(lines)
