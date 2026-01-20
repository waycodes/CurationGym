"""Training module for CurationGym."""

from curationgym.train.adapters import TrainingAdapter, TrainingBudget, TrainingResult, HFTextAdapter
from curationgym.train.dataloader import (
    DeterministicTextDataset,
    StreamingTextDataset,
    create_dataset_from_manifest,
)
from curationgym.train.compute_meter import ComputeMeter, ComputeMetrics

__all__ = [
    "TrainingAdapter", "TrainingBudget", "TrainingResult", "HFTextAdapter",
    "DeterministicTextDataset", "StreamingTextDataset", "create_dataset_from_manifest",
    "ComputeMeter", "ComputeMetrics",
]