"""Visualization plots for CurationGym experiments."""

from pathlib import Path
from typing import Any
import json


def plot_slice_distribution(
    slice_stats: dict[str, dict[str, Any]],
    output_path: str | Path | None = None,
    metric: str = "token_count",
) -> Any:
    """Plot slice distribution as pie chart.

    Args:
        slice_stats: Slice name -> stats dict
        output_path: Optional path to save figure
        metric: Metric to plot ('token_count' or 'doc_count')

    Returns:
        matplotlib figure or None
    """
    try:
        import matplotlib.pyplot as plt

        labels = []
        sizes = []
        for name, stats in sorted(slice_stats.items()):
            val = stats.get(metric, 0)
            if val > 0:
                labels.append(name)
                sizes.append(val)

        # Combine small slices
        total = sum(sizes)
        threshold = total * 0.02
        main_labels, main_sizes = [], []
        other = 0
        for label, size in zip(labels, sizes):
            if size >= threshold:
                main_labels.append(label)
                main_sizes.append(size)
            else:
                other += size
        if other > 0:
            main_labels.append("other")
            main_sizes.append(other)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.pie(main_sizes, labels=main_labels, autopct="%1.1f%%", startangle=90)
        ax.set_title(f"Slice Distribution ({metric})")

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")

        return fig

    except ImportError:
        return None


def plot_eval_radar(
    eval_scores: dict[str, float],
    output_path: str | Path | None = None,
    baseline_scores: dict[str, float] | None = None,
) -> Any:
    """Plot evaluation scores as radar chart.

    Args:
        eval_scores: Benchmark -> score mapping
        output_path: Optional path to save figure
        baseline_scores: Optional baseline for comparison

    Returns:
        matplotlib figure or None
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        benchmarks = list(eval_scores.keys())
        scores = [eval_scores[b] for b in benchmarks]

        angles = np.linspace(0, 2 * np.pi, len(benchmarks), endpoint=False).tolist()
        scores += scores[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.plot(angles, scores, "o-", linewidth=2, label="Current")
        ax.fill(angles, scores, alpha=0.25)

        if baseline_scores:
            base = [baseline_scores.get(b, 0) for b in benchmarks]
            base += base[:1]
            ax.plot(angles, base, "o--", linewidth=2, label="Baseline", alpha=0.7)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(benchmarks)
        ax.set_title("Evaluation Scores")
        ax.legend(loc="upper right")

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")

        return fig

    except ImportError:
        return None


def plot_attribution_bars(
    coefficients: dict[str, dict[str, Any]],
    output_path: str | Path | None = None,
    top_k: int = 15,
) -> Any:
    """Plot attribution coefficients as bar chart with CIs.

    Args:
        coefficients: Slice -> {mean, ci_95} mapping
        output_path: Optional path to save figure
        top_k: Number of top slices to show

    Returns:
        matplotlib figure or None
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # Sort by absolute value
        sorted_coef = sorted(
            coefficients.items(),
            key=lambda x: abs(x[1].get("mean", 0)),
            reverse=True
        )[:top_k]

        names = [c[0] for c in sorted_coef]
        means = [c[1].get("mean", 0) for c in sorted_coef]
        cis = [c[1].get("ci_95", [0, 0]) for c in sorted_coef]
        errors = [[m - ci[0] for m, ci in zip(means, cis)],
                  [ci[1] - m for m, ci in zip(means, cis)]]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["green" if m > 0 else "red" for m in means]
        y_pos = np.arange(len(names))

        ax.barh(y_pos, means, xerr=errors, color=colors, alpha=0.7, capsize=3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.set_xlabel("Attribution Coefficient")
        ax.set_title("Slice Attribution (with 95% CI)")
        ax.invert_yaxis()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")

        return fig

    except ImportError:
        return None


def plot_pareto_frontier(
    trials: list[dict[str, Any]],
    x_metric: str = "compute_cost",
    y_metric: str = "avg_score",
    output_path: str | Path | None = None,
) -> Any:
    """Plot Pareto frontier of trials.

    Args:
        trials: List of trial dicts with metrics
        x_metric: X-axis metric name
        y_metric: Y-axis metric name
        output_path: Optional path to save figure

    Returns:
        matplotlib figure or None
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        x = [t.get(x_metric, 0) for t in trials]
        y = [t.get(y_metric, 0) for t in trials]
        is_pareto = [t.get("is_pareto", False) for t in trials]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Non-Pareto points
        non_pareto_x = [xi for xi, p in zip(x, is_pareto) if not p]
        non_pareto_y = [yi for yi, p in zip(y, is_pareto) if not p]
        ax.scatter(non_pareto_x, non_pareto_y, alpha=0.5, label="Dominated")

        # Pareto points
        pareto_x = [xi for xi, p in zip(x, is_pareto) if p]
        pareto_y = [yi for yi, p in zip(y, is_pareto) if p]
        ax.scatter(pareto_x, pareto_y, color="red", s=100, label="Pareto Optimal")

        # Connect Pareto frontier
        if pareto_x:
            sorted_pareto = sorted(zip(pareto_x, pareto_y))
            ax.plot([p[0] for p in sorted_pareto], [p[1] for p in sorted_pareto],
                    "r--", alpha=0.5)

        ax.set_xlabel(x_metric)
        ax.set_ylabel(y_metric)
        ax.set_title("Pareto Frontier")
        ax.legend()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")

        return fig

    except ImportError:
        return None


def generate_all_plots(
    experiment_dir: str | Path,
    output_dir: str | Path | None = None,
) -> list[Path]:
    """Generate all plots for an experiment.

    Args:
        experiment_dir: Directory with experiment results
        output_dir: Output directory for plots

    Returns:
        List of generated plot paths
    """
    exp_dir = Path(experiment_dir)
    out_dir = Path(output_dir) if output_dir else exp_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = []

    # Load data
    manifest_path = exp_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        slice_stats = manifest.get("slice_stats", {})
        if slice_stats:
            path = out_dir / "slice_distribution.png"
            if plot_slice_distribution(slice_stats, path):
                generated.append(path)

    # Eval scores
    eval_path = exp_dir / "eval_results.json"
    if eval_path.exists():
        eval_data = json.loads(eval_path.read_text())
        scores = eval_data.get("scores", {})
        if scores:
            path = out_dir / "eval_radar.png"
            if plot_eval_radar(scores, path):
                generated.append(path)

    # Attribution
    attr_path = exp_dir / "attribution.json"
    if attr_path.exists():
        attr_data = json.loads(attr_path.read_text())
        coefficients = attr_data.get("coefficients", {})
        if coefficients:
            path = out_dir / "attribution_bars.png"
            if plot_attribution_bars(coefficients, path):
                generated.append(path)

    return generated
