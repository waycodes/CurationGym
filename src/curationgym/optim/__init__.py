"""Optimization module for CurationGym."""

from curationgym.optim.search_space import SearchSpace, ParameterRange, Constraints
from curationgym.optim.random_search import RandomSearchOptimizer, TrialResult, OptimizationResult
from curationgym.optim.multifidelity import MultiFidelityScheduler, MultiFidelityConfig, FidelityLevel
from curationgym.optim.pruning import EarlyStoppingCallback, TrialPruner, PruningConfig
from curationgym.optim.optuna_tpe import OptunaTPEOptimizer
from curationgym.optim.constraints import (
    feasibility_first_rank,
    apply_constraint_penalty,
    reject_infeasible,
    compute_violations,
)
from curationgym.optim.pareto import ParetoFrontier, ParetoPoint

__all__ = [
    "SearchSpace", "ParameterRange", "Constraints",
    "RandomSearchOptimizer", "TrialResult", "OptimizationResult",
    "MultiFidelityScheduler", "MultiFidelityConfig", "FidelityLevel",
    "EarlyStoppingCallback", "TrialPruner", "PruningConfig",
    "OptunaTPEOptimizer",
    "feasibility_first_rank", "apply_constraint_penalty", "reject_infeasible", "compute_violations",
    "ParetoFrontier", "ParetoPoint",
]