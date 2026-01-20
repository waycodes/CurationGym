"""Search space definition for policy optimization."""

from dataclasses import dataclass, field
from typing import Any
import random


@dataclass
class ParameterRange:
    """Range for a single parameter."""

    name: str
    param_type: str  # int, float, categorical
    low: float | None = None
    high: float | None = None
    choices: list[Any] | None = None
    default: Any = None
    log_scale: bool = False

    def sample(self, rng: random.Random) -> Any:
        """Sample a value from this range."""
        if self.param_type == "categorical":
            return rng.choice(self.choices)
        elif self.param_type == "int":
            if self.log_scale:
                log_val = rng.uniform(math.log(self.low), math.log(self.high))
                return int(round(math.exp(log_val)))
            return rng.randint(int(self.low), int(self.high))
        elif self.param_type == "float":
            if self.log_scale:
                import math
                log_val = rng.uniform(math.log(self.low), math.log(self.high))
                return math.exp(log_val)
            return rng.uniform(self.low, self.high)
        return self.default


@dataclass
class SearchSpace:
    """Complete search space for policy optimization."""

    parameters: list[ParameterRange] = field(default_factory=list)

    @classmethod
    def default(cls) -> "SearchSpace":
        """Create default search space."""
        import math
        return cls(parameters=[
            # Quality filtering
            ParameterRange("min_words", "int", 10, 500, default=50),
            ParameterRange("max_words", "int", 1000, 500000, default=100000, log_scale=True),
            ParameterRange("min_alpha_ratio", "float", 0.3, 0.9, default=0.6),
            ParameterRange("max_word_rep_ratio", "float", 0.05, 0.5, default=0.2),

            # Deduplication
            ParameterRange("dedup_method", "categorical", choices=["exact", "minhash"], default="minhash"),
            ParameterRange("dedup_scope", "categorical", choices=["global", "per_dump"], default="per_dump"),
            ParameterRange("minhash_bands", "int", 5, 20, default=14),
            ParameterRange("minhash_rows", "int", 4, 16, default=8),

            # Decontamination
            ParameterRange("decontam_enabled", "categorical", choices=[True, False], default=True),
            ParameterRange("decontam_mode", "categorical", choices=["drop", "tag"], default="drop"),
            ParameterRange("overlap_threshold", "float", 0.5, 0.95, default=0.8),

            # Budget
            ParameterRange("max_tokens", "int", 1_000_000_000, 100_000_000_000, default=10_000_000_000, log_scale=True),
        ])

    def sample(self, rng: random.Random | None = None) -> dict[str, Any]:
        """Sample a configuration from the search space."""
        rng = rng or random.Random()
        return {p.name: p.sample(rng) for p in self.parameters}

    def get_default(self) -> dict[str, Any]:
        """Get default configuration."""
        return {p.name: p.default for p in self.parameters}

    def to_optuna_space(self) -> dict[str, Any]:
        """Convert to Optuna-compatible search space definition."""
        space = {}
        for p in self.parameters:
            if p.param_type == "categorical":
                space[p.name] = {"type": "categorical", "choices": p.choices}
            elif p.param_type == "int":
                space[p.name] = {"type": "int", "low": int(p.low), "high": int(p.high), "log": p.log_scale}
            elif p.param_type == "float":
                space[p.name] = {"type": "float", "low": p.low, "high": p.high, "log": p.log_scale}
        return space


@dataclass
class Constraints:
    """Optimization constraints."""

    max_compute_hours: float = 100.0
    max_contamination_rate: float = 0.01
    min_dataset_tokens: int = 1_000_000_000
    min_diversity_score: float = 0.5

    def is_feasible(
        self,
        compute_hours: float,
        contamination_rate: float,
        dataset_tokens: int,
        diversity_score: float,
    ) -> tuple[bool, list[str]]:
        """Check if solution satisfies constraints.

        Returns:
            (is_feasible, list of violated constraints)
        """
        violations = []

        if compute_hours > self.max_compute_hours:
            violations.append(f"compute_hours={compute_hours:.1f} > {self.max_compute_hours}")

        if contamination_rate > self.max_contamination_rate:
            violations.append(f"contamination_rate={contamination_rate:.4f} > {self.max_contamination_rate}")

        if dataset_tokens < self.min_dataset_tokens:
            violations.append(f"dataset_tokens={dataset_tokens} < {self.min_dataset_tokens}")

        if diversity_score < self.min_diversity_score:
            violations.append(f"diversity_score={diversity_score:.2f} < {self.min_diversity_score}")

        return len(violations) == 0, violations
