"""Experiment report generator."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ExperimentReport:
    """Complete experiment report."""

    experiment_id: str
    created_at: str

    # Policy
    policy_summary: dict[str, Any]

    # Results
    eval_scores: dict[str, float]
    attribution: dict[str, Any] | None

    # Compute
    compute_usage: dict[str, Any]

    # Metadata
    best_trial_id: str | None = None
    total_trials: int = 0


def generate_experiment_report(
    experiment_id: str,
    policy: dict[str, Any],
    eval_results: dict[str, float],
    compute_stats: dict[str, Any],
    attribution: dict[str, Any] | None = None,
    best_trial_id: str | None = None,
    total_trials: int = 0,
) -> str:
    """Generate markdown experiment report.

    Args:
        experiment_id: Unique experiment identifier
        policy: Policy configuration dict
        eval_results: Benchmark -> score mapping
        compute_stats: Compute usage statistics
        attribution: Optional attribution results
        best_trial_id: ID of best trial
        total_trials: Total trials run

    Returns:
        Markdown report string
    """
    lines = []

    # Header
    lines.append(f"# Experiment Report: {experiment_id}")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    if best_trial_id:
        lines.append(f"- **Best Trial**: {best_trial_id}")
    lines.append(f"- **Total Trials**: {total_trials}")
    avg_score = sum(eval_results.values()) / len(eval_results) if eval_results else 0
    lines.append(f"- **Average Score**: {avg_score:.4f}")
    lines.append("")

    # Policy Summary
    lines.append("## Policy Configuration")
    lines.append("")
    _add_policy_summary(lines, policy)

    # Evaluation Scores
    lines.append("## Evaluation Scores")
    lines.append("")
    lines.append("| Benchmark | Score |")
    lines.append("|-----------|-------|")
    for bench, score in sorted(eval_results.items(), key=lambda x: -x[1]):
        lines.append(f"| {bench} | {score:.4f} |")
    lines.append("")

    # Attribution Highlights
    if attribution:
        lines.append("## Attribution Highlights")
        lines.append("")
        _add_attribution_summary(lines, attribution)

    # Compute Usage
    lines.append("## Compute Usage")
    lines.append("")
    _add_compute_summary(lines, compute_stats)

    return "\n".join(lines)


def _add_policy_summary(lines: list[str], policy: dict[str, Any]) -> None:
    """Add policy summary section."""
    # Filters
    filters = policy.get("filters", [])
    if filters:
        lines.append("### Filters")
        lines.append("")
        for f in filters:
            name = f.get("name", "unknown")
            lines.append(f"- {name}")
        lines.append("")

    # Dedup
    dedup = policy.get("dedup", {})
    if dedup:
        lines.append("### Deduplication")
        lines.append("")
        lines.append(f"- Method: {dedup.get('method', 'none')}")
        lines.append("")

    # Mixing
    mixing = policy.get("mixing", {})
    if mixing:
        lines.append("### Top Mixing Weights")
        lines.append("")
        sorted_mix = sorted(mixing.items(), key=lambda x: -x[1])[:5]
        for slice_name, weight in sorted_mix:
            lines.append(f"- {slice_name}: {weight:.3f}")
        lines.append("")


def _add_attribution_summary(lines: list[str], attribution: dict[str, Any]) -> None:
    """Add attribution highlights."""
    coefficients = attribution.get("coefficients", {})
    if coefficients:
        lines.append("### Top Positive Contributors")
        lines.append("")
        sorted_coef = sorted(coefficients.items(), key=lambda x: -x[1].get("mean", 0))
        for slice_name, coef in sorted_coef[:5]:
            mean = coef.get("mean", 0)
            ci = coef.get("ci_95", [0, 0])
            lines.append(f"- **{slice_name}**: {mean:.4f} (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])")
        lines.append("")

        lines.append("### Top Negative Contributors")
        lines.append("")
        for slice_name, coef in sorted_coef[-3:]:
            mean = coef.get("mean", 0)
            if mean < 0:
                ci = coef.get("ci_95", [0, 0])
                lines.append(f"- **{slice_name}**: {mean:.4f} (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])")
        lines.append("")

    # R² if available
    r2 = attribution.get("r2")
    if r2 is not None:
        lines.append(f"**Model R²**: {r2:.4f}")
        lines.append("")


def _add_compute_summary(lines: list[str], compute: dict[str, Any]) -> None:
    """Add compute usage summary."""
    lines.append(f"- **Total GPU Hours**: {compute.get('gpu_hours', 0):.2f}")
    lines.append(f"- **Total FLOPs**: {compute.get('total_flops', 0):.2e}")
    lines.append(f"- **Training Time**: {compute.get('training_time_hours', 0):.2f}h")
    lines.append(f"- **Eval Time**: {compute.get('eval_time_hours', 0):.2f}h")
    lines.append("")

    # Budget utilization
    budget = compute.get("budget_limit")
    used = compute.get("gpu_hours", 0)
    if budget:
        pct = (used / budget) * 100
        lines.append(f"**Budget Utilization**: {pct:.1f}% ({used:.1f}/{budget:.1f} GPU-hours)")
        lines.append("")


def save_experiment_report(
    report: str,
    output_path: str | Path,
    format: str = "md",
) -> Path:
    """Save experiment report to file.

    Args:
        report: Markdown report string
        output_path: Output file path
        format: Output format ('md' or 'html')

    Returns:
        Path to saved report
    """
    path = Path(output_path)

    if format == "html":
        html = _markdown_to_html(report)
        path = path.with_suffix(".html")
        path.write_text(html)
    else:
        path = path.with_suffix(".md")
        path.write_text(report)

    return path


def _markdown_to_html(md: str) -> str:
    """Convert markdown to HTML with basic styling."""
    try:
        import markdown
        body = markdown.markdown(md, extensions=["tables"])
    except ImportError:
        # Fallback: wrap in pre
        body = f"<pre>{md}</pre>"

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Experiment Report</title>
    <style>
        body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f5f5f5; }}
        code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; }}
        h1 {{ border-bottom: 2px solid #333; padding-bottom: 10px; }}
        h2 {{ color: #444; margin-top: 30px; }}
    </style>
</head>
<body>
{body}
</body>
</html>"""
