"""Compute accounting for training runs."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ComputeMetrics:
    """Tracked compute metrics."""

    tokens_processed: int = 0
    steps_completed: int = 0
    wall_time_seconds: float = 0.0
    gpu_memory_peak_mb: float = 0.0
    flops_estimate: float = 0.0


class ComputeMeter:
    """Track compute usage during training."""

    def __init__(self, model_params: int):
        self.model_params = model_params
        self._start_time: float | None = None
        self._metrics = ComputeMetrics()
        self._step_times: list[float] = []

    def start(self) -> None:
        """Start timing."""
        self._start_time = time.time()

    def step(self, tokens: int) -> None:
        """Record a training step."""
        self._metrics.tokens_processed += tokens
        self._metrics.steps_completed += 1

        if self._start_time:
            self._metrics.wall_time_seconds = time.time() - self._start_time

        # Estimate FLOPs (rough: 6 * params * tokens for forward+backward)
        self._metrics.flops_estimate = 6 * self.model_params * self._metrics.tokens_processed

        # Track GPU memory if available
        try:
            import torch
            if torch.cuda.is_available():
                peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                self._metrics.gpu_memory_peak_mb = max(self._metrics.gpu_memory_peak_mb, peak_mb)
        except ImportError:
            pass

    def stop(self) -> None:
        """Stop timing."""
        if self._start_time:
            self._metrics.wall_time_seconds = time.time() - self._start_time

    @property
    def metrics(self) -> ComputeMetrics:
        return self._metrics

    @property
    def throughput_tokens_per_sec(self) -> float:
        if self._metrics.wall_time_seconds > 0:
            return self._metrics.tokens_processed / self._metrics.wall_time_seconds
        return 0.0

    @property
    def throughput_steps_per_sec(self) -> float:
        if self._metrics.wall_time_seconds > 0:
            return self._metrics.steps_completed / self._metrics.wall_time_seconds
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tokens_processed": self._metrics.tokens_processed,
            "steps_completed": self._metrics.steps_completed,
            "wall_time_seconds": round(self._metrics.wall_time_seconds, 2),
            "gpu_memory_peak_mb": round(self._metrics.gpu_memory_peak_mb, 2),
            "flops_estimate": self._metrics.flops_estimate,
            "throughput_tokens_per_sec": round(self.throughput_tokens_per_sec, 2),
            "throughput_steps_per_sec": round(self.throughput_steps_per_sec, 4),
        }

    def save(self, path: str | Path) -> None:
        """Save metrics to JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    def check_budget(
        self,
        max_tokens: int | None = None,
        max_steps: int | None = None,
        max_hours: float | None = None,
    ) -> bool:
        """Check if within budget. Returns True if should continue."""
        if max_tokens and self._metrics.tokens_processed >= max_tokens:
            return False
        if max_steps and self._metrics.steps_completed >= max_steps:
            return False
        if max_hours and self._metrics.wall_time_seconds >= max_hours * 3600:
            return False
        return True
