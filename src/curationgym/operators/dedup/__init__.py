"""Deduplication operators for CurationGym."""

from curationgym.operators.dedup.exact_doc import ExactDedup
from curationgym.operators.dedup.minhash import MinHashDedup, MinHashConfig
from curationgym.operators.dedup.scope import ScopedDedup, DedupScope
from curationgym.operators.dedup.keep_rules import KeepRule, KeepRuleSelector, ClusterDedup
from curationgym.operators.dedup.semantic import SemanticDedup, SemanticDedupConfig

__all__ = [
    "ExactDedup",
    "MinHashDedup", "MinHashConfig",
    "ScopedDedup", "DedupScope",
    "KeepRule", "KeepRuleSelector", "ClusterDedup",
    "SemanticDedup", "SemanticDedupConfig",
]
