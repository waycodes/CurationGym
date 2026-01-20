"""Early stopping and pruning for optimization trials."""

from dataclasses import dataclass
from typing import Any


@dataclass
class PruningConfig:
    """Configuration for trial pruning."""

    # Loss-based pruning
    loss_patience: int = 3  # Steps without improvement before pruning
    loss_min_delta: float = 0.01  # Minimum improvement to reset patience

    # Score-based pruning (compare to running best)
    score_threshold_fraction: float = 0.5  # Prune if score < best * fraction
    min_steps_before_pruning: int = 100  # Don't prune before this many steps


class EarlyStoppingCallback:
    """Callback for early stopping during training."""

    def __init__(self, config: PruningConfig | None = None):
        self.config = config or PruningConfig()
        self._best_loss = float("inf")
        self._patience_counter = 0
        self._step = 0

    def __call__(self, loss: float) -> bool:
        """Check if training should stop.

        Args:
            loss: Current training loss

        Returns:
            True if should stop, False to continue
        """
        self._step += 1

        if loss < self._best_loss - self.config.loss_min_delta:
            self._best_loss = loss
            self._patience_counter = 0
        else:
            self._patience_counter += 1

        if self._patience_counter >= self.config.loss_patience:
            return True

        return False

    def reset(self) -> None:
        """Reset callback state."""
        self._best_loss = float("inf")
        self._patience_counter = 0
        self._step = 0


class TrialPruner:
    """Prune unpromising trials based on interim results."""

    def __init__(self, config: PruningConfig | None = None):
        self.config = config or PruningConfig()
        self._best_score = float("-inf")
        self._trial_scores: dict[int, list[float]] = {}

    def report(self, trial_id: int, step: int, score: float) -> bool:
        """Report interim score and check if trial should be pruned.

        Args:
            trial_id: Trial identifier
            step: Current step number
            score: Interim evaluation score

        Returns:
            True if trial should be pruned, False to continue
        """
        # Track scores
        if trial_id not in self._trial_scores:
            self._trial_scores[trial_id] = []
        self._trial_scores[trial_id].append(score)

        # Update best
        if score > self._best_score:
            self._best_score = score

        # Don't prune too early
        if step < self.config.min_steps_before_pruning:
            return False

        # Prune if significantly worse than best
        threshold = self._best_score * self.config.score_threshold_fraction
        if score < threshold:
            return True

        return False

    def should_prune(self, trial_id: int, current_score: float) -> bool:
        """Check if trial should be pruned based on current score."""
        if self._best_score == float("-inf"):
            return False

        threshold = self._best_score * self.config.score_threshold_fraction
        return current_score < threshold

    def reset(self) -> None:
        """Reset pruner state."""
        self._best_score = float("-inf")
        self._trial_scores.clear()

    @property
    def best_score(self) -> float:
        return self._best_score
