"""Evaluation runner abstraction."""

import json
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalResult:
    """Result from evaluation run."""

    task_scores: dict[str, float]  # task_name -> score
    aggregate_score: float
    confidence_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    eval_code_version: str = ""
    task_versions: dict[str, str] = field(default_factory=dict)
    raw_results: dict[str, Any] = field(default_factory=dict)


class EvalRunner(ABC):
    """Abstract base class for evaluation runners."""

    @abstractmethod
    def evaluate(
        self,
        checkpoint_path: str | Path,
        eval_suite_config: str | Path,
        output_dir: str | Path,
    ) -> EvalResult:
        """Run evaluation on a checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
            eval_suite_config: Path to eval suite configuration
            output_dir: Directory for results

        Returns:
            EvalResult with scores and metadata
        """
        pass

    @abstractmethod
    def get_code_version(self) -> str:
        """Get version of evaluation code."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Runner name."""
        pass


def save_eval_results(result: EvalResult, path: str | Path) -> None:
    """Save evaluation results to JSON."""
    data = {
        "task_scores": result.task_scores,
        "aggregate_score": result.aggregate_score,
        "confidence_intervals": {k: list(v) for k, v in result.confidence_intervals.items()},
        "eval_code_version": result.eval_code_version,
        "task_versions": result.task_versions,
    }
    Path(path).write_text(json.dumps(data, indent=2))


def load_eval_results(path: str | Path) -> EvalResult:
    """Load evaluation results from JSON."""
    data = json.loads(Path(path).read_text())
    return EvalResult(
        task_scores=data["task_scores"],
        aggregate_score=data["aggregate_score"],
        confidence_intervals={k: tuple(v) for k, v in data.get("confidence_intervals", {}).items()},
        eval_code_version=data.get("eval_code_version", ""),
        task_versions=data.get("task_versions", {}),
    )
