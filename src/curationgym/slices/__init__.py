"""Slice system for document categorization and attribution."""

from curationgym.slices.registry import SliceDefinition, SliceRegistry, get_registry
from curationgym.slices.assign import assign_slices, assign_and_store, get_slice_code_version
from curationgym.slices.stats import SliceStats, SliceStatsCollector

__all__ = [
    "SliceDefinition", "SliceRegistry", "get_registry",
    "assign_slices", "assign_and_store", "get_slice_code_version",
    "SliceStats", "SliceStatsCollector",
]