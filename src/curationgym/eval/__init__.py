"""Evaluation module for CurationGym."""

from curationgym.eval.aggregators import get_aggregator, AGGREGATORS
from curationgym.eval.runner import EvalRunner, EvalResult, save_eval_results, load_eval_results
from curationgym.eval.aggregate import aggregate_eval_results, AggregateResult, save_aggregate_results
from curationgym.eval.text import LMEvalAdapter

__all__ = [
    "get_aggregator", "AGGREGATORS",
    "EvalRunner", "EvalResult", "save_eval_results", "load_eval_results",
    "aggregate_eval_results", "AggregateResult", "save_aggregate_results",
    "LMEvalAdapter",
]
