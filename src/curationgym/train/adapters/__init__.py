"""Training adapters for CurationGym."""

from curationgym.train.adapters.base import TrainingAdapter, TrainingBudget, TrainingResult
from curationgym.train.adapters.text_hf import HFTextAdapter

__all__ = ["TrainingAdapter", "TrainingBudget", "TrainingResult", "HFTextAdapter"]
