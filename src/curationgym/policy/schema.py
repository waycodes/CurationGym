"""Policy schema for curation experiments."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QualityThresholds:
    """Quality filtering thresholds."""

    min_words: int = 50
    max_words: int = 100000
    min_avg_word_length: float = 3.0
    max_word_rep_ratio: float = 0.2
    min_alpha_ratio: float = 0.6
    enabled_rules: list[str] = field(default_factory=lambda: [
        "word_rep", "min_words", "max_words", "alpha_ratio"
    ])


@dataclass
class DedupConfig:
    """Deduplication configuration."""

    method: str = "minhash"  # exact, minhash, semantic
    scope: str = "per_dump"  # global, per_dump
    ngram_size: int = 5
    num_bands: int = 14
    rows_per_band: int = 8
    keep_rule: str = "first"  # first, longest, highest_quality


@dataclass
class DecontamConfig:
    """Decontamination configuration."""

    enabled: bool = True
    mode: str = "drop"  # drop, redact, downweight, tag
    ngram_size: int = 13
    overlap_threshold: float = 0.8
    targets: list[str] = field(default_factory=lambda: [
        "hellaswag", "arc_easy", "piqa", "winogrande", "mmlu"
    ])


@dataclass
class Policy:
    """Complete curation policy specification."""

    name: str
    version: str = "1.0"

    # Input
    input_source: str = ""  # HF dataset name or path pattern
    input_config: dict[str, Any] = field(default_factory=dict)

    # Filtering
    language: str = "en"
    min_language_score: float = 0.65
    quality: QualityThresholds = field(default_factory=QualityThresholds)

    # Deduplication
    dedup: DedupConfig = field(default_factory=DedupConfig)

    # Decontamination
    decontam: DecontamConfig = field(default_factory=DecontamConfig)

    # Mixing
    slice_weights: dict[str, float] = field(default_factory=dict)
    max_tokens: int = 10_000_000_000  # 10B default
    max_tokens_per_slice: dict[str, int] = field(default_factory=dict)

    # Reproducibility
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "input_source": self.input_source,
            "input_config": self.input_config,
            "language": self.language,
            "min_language_score": self.min_language_score,
            "quality": {
                "min_words": self.quality.min_words,
                "max_words": self.quality.max_words,
                "min_avg_word_length": self.quality.min_avg_word_length,
                "max_word_rep_ratio": self.quality.max_word_rep_ratio,
                "min_alpha_ratio": self.quality.min_alpha_ratio,
                "enabled_rules": self.quality.enabled_rules,
            },
            "dedup": {
                "method": self.dedup.method,
                "scope": self.dedup.scope,
                "ngram_size": self.dedup.ngram_size,
                "num_bands": self.dedup.num_bands,
                "rows_per_band": self.dedup.rows_per_band,
                "keep_rule": self.dedup.keep_rule,
            },
            "decontam": {
                "enabled": self.decontam.enabled,
                "mode": self.decontam.mode,
                "ngram_size": self.decontam.ngram_size,
                "overlap_threshold": self.decontam.overlap_threshold,
                "targets": self.decontam.targets,
            },
            "slice_weights": self.slice_weights,
            "max_tokens": self.max_tokens,
            "max_tokens_per_slice": self.max_tokens_per_slice,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Policy":
        """Create from dictionary."""
        quality_data = data.get("quality", {})
        dedup_data = data.get("dedup", {})
        decontam_data = data.get("decontam", {})

        return cls(
            name=data.get("name", "unnamed"),
            version=data.get("version", "1.0"),
            input_source=data.get("input_source", ""),
            input_config=data.get("input_config", {}),
            language=data.get("language", "en"),
            min_language_score=data.get("min_language_score", 0.65),
            quality=QualityThresholds(
                min_words=quality_data.get("min_words", 50),
                max_words=quality_data.get("max_words", 100000),
                min_avg_word_length=quality_data.get("min_avg_word_length", 3.0),
                max_word_rep_ratio=quality_data.get("max_word_rep_ratio", 0.2),
                min_alpha_ratio=quality_data.get("min_alpha_ratio", 0.6),
                enabled_rules=quality_data.get("enabled_rules", ["word_rep", "min_words", "max_words", "alpha_ratio"]),
            ),
            dedup=DedupConfig(
                method=dedup_data.get("method", "minhash"),
                scope=dedup_data.get("scope", "per_dump"),
                ngram_size=dedup_data.get("ngram_size", 5),
                num_bands=dedup_data.get("num_bands", 14),
                rows_per_band=dedup_data.get("rows_per_band", 8),
                keep_rule=dedup_data.get("keep_rule", "first"),
            ),
            decontam=DecontamConfig(
                enabled=decontam_data.get("enabled", True),
                mode=decontam_data.get("mode", "drop"),
                ngram_size=decontam_data.get("ngram_size", 13),
                overlap_threshold=decontam_data.get("overlap_threshold", 0.8),
                targets=decontam_data.get("targets", ["hellaswag", "arc_easy", "piqa", "winogrande", "mmlu"]),
            ),
            slice_weights=data.get("slice_weights", {}),
            max_tokens=data.get("max_tokens", 10_000_000_000),
            max_tokens_per_slice=data.get("max_tokens_per_slice", {}),
            seed=data.get("seed", 42),
        )
