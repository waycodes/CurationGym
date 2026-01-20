"""Slice statistics computation for attribution."""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from curationgym.core.document import Document


@dataclass
class SliceStats:
    """Statistics for a single slice."""

    doc_count: int = 0
    token_count: int = 0
    quality_score_sum: float = 0.0
    dedup_dropped: int = 0
    decontam_dropped: int = 0

    @property
    def avg_quality_score(self) -> float:
        return self.quality_score_sum / self.doc_count if self.doc_count > 0 else 0.0

    @property
    def dedup_drop_rate(self) -> float:
        total = self.doc_count + self.dedup_dropped
        return self.dedup_dropped / total if total > 0 else 0.0

    @property
    def decontam_drop_rate(self) -> float:
        total = self.doc_count + self.decontam_dropped
        return self.decontam_dropped / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_count": self.doc_count,
            "token_count": self.token_count,
            "avg_quality_score": round(self.avg_quality_score, 4),
            "dedup_drop_rate": round(self.dedup_drop_rate, 4),
            "decontam_drop_rate": round(self.decontam_drop_rate, 4),
        }


class SliceStatsCollector:
    """Collect statistics per slice."""

    def __init__(self):
        self._stats: dict[str, SliceStats] = defaultdict(SliceStats)
        self._total = SliceStats()

    def add_document(self, doc: Document, kept: bool = True) -> None:
        """Add document to statistics."""
        slice_tags = doc.metadata.get("slice_tags", [])
        token_count = doc.metadata.get("token_count", 0)

        # Compute average quality score
        quality_scores = doc.metadata.get("quality_scores", {})
        avg_quality = sum(quality_scores.values()) / len(quality_scores) if quality_scores else 0.0

        # Check drop reasons
        dedup_dropped = doc.metadata.get("dedup_dropped", False)
        decontam_dropped = doc.metadata.get("contamination_flags", {}).get("dropped", False)

        for tag in slice_tags:
            stats = self._stats[tag]
            if kept and not dedup_dropped and not decontam_dropped:
                stats.doc_count += 1
                stats.token_count += token_count
                stats.quality_score_sum += avg_quality
            if dedup_dropped:
                stats.dedup_dropped += 1
            if decontam_dropped:
                stats.decontam_dropped += 1

        # Update totals
        if kept and not dedup_dropped and not decontam_dropped:
            self._total.doc_count += 1
            self._total.token_count += token_count
            self._total.quality_score_sum += avg_quality
        if dedup_dropped:
            self._total.dedup_dropped += 1
        if decontam_dropped:
            self._total.decontam_dropped += 1

    def get_stats(self, slice_tag: str) -> SliceStats:
        """Get stats for a specific slice."""
        return self._stats.get(slice_tag, SliceStats())

    def get_all_stats(self) -> dict[str, SliceStats]:
        """Get stats for all slices."""
        return dict(self._stats)

    @property
    def total_stats(self) -> SliceStats:
        """Get aggregate stats across all documents."""
        return self._total

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total": self._total.to_dict(),
            "by_slice": {tag: stats.to_dict() for tag, stats in sorted(self._stats.items())},
        }

    def save(self, path: str | Path) -> None:
        """Save stats to JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "SliceStatsCollector":
        """Load stats from JSON file."""
        data = json.loads(Path(path).read_text())
        collector = cls()

        # Restore total
        total = data.get("total", {})
        collector._total.doc_count = total.get("doc_count", 0)
        collector._total.token_count = total.get("token_count", 0)

        # Restore per-slice
        for tag, stats_dict in data.get("by_slice", {}).items():
            stats = SliceStats()
            stats.doc_count = stats_dict.get("doc_count", 0)
            stats.token_count = stats_dict.get("token_count", 0)
            collector._stats[tag] = stats

        return collector

    def reset(self) -> None:
        """Clear all statistics."""
        self._stats.clear()
        self._total = SliceStats()
