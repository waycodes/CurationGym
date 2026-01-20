"""Report module for experiment reporting and visualization."""

from curationgym.report.experiment_report import (
    ExperimentReport,
    generate_experiment_report,
    save_experiment_report,
)
from curationgym.report.plots import (
    plot_slice_distribution,
    plot_eval_radar,
    plot_attribution_bars,
    plot_pareto_frontier,
    generate_all_plots,
)
from curationgym.report.policy_diff import (
    PolicyDiff,
    diff_policies,
    format_policy_diff,
)

__all__ = [
    "ExperimentReport",
    "generate_experiment_report",
    "save_experiment_report",
    "plot_slice_distribution",
    "plot_eval_radar",
    "plot_attribution_bars",
    "plot_pareto_frontier",
    "generate_all_plots",
    "PolicyDiff",
    "diff_policies",
    "format_policy_diff",
]
