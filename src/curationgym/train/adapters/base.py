"""Base adapter interface for proxy model training."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrainingBudget:
    """Compute budget for training."""

    max_tokens: int
    max_steps: int | None = None
    max_hours: float | None = None
    batch_size_tokens: int = 100_000
    model_params: int = 50_000_000  # 50M default


@dataclass
class TrainingResult:
    """Result from training run."""

    checkpoint_path: str
    final_loss: float
    tokens_trained: int
    steps_completed: int
    wall_time_seconds: float
    metrics: dict[str, Any]


class TrainingAdapter(ABC):
    """Abstract base class for model training adapters."""

    @abstractmethod
    def train(
        self,
        dataset_manifest_path: str | Path,
        budget: TrainingBudget,
        output_dir: str | Path,
        seed: int = 42,
    ) -> TrainingResult:
        """Train a model on the dataset.

        Args:
            dataset_manifest_path: Path to dataset manifest
            budget: Compute budget constraints
            output_dir: Directory for checkpoints and logs
            seed: Random seed for reproducibility

        Returns:
            TrainingResult with checkpoint path and metrics
        """
        pass

    @abstractmethod
    def get_model_config(self, params: int) -> dict[str, Any]:
        """Get model configuration for target parameter count."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Adapter name."""
        pass
