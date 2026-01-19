"""Dedup keep rule strategies for selecting which duplicate to retain."""

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from curationgym.core.document import Document


class KeepRule(Enum):
    """Strategy for selecting which document to keep from duplicates."""

    FIRST = "first"  # Keep first encountered
    LONGEST = "longest"  # Keep longest by character count
    MOST_TOKENS = "most_tokens"  # Keep most tokens
    HIGHEST_QUALITY = "highest_quality"  # Keep highest quality score
    LOWEST_TOXICITY = "lowest_toxicity"  # Keep lowest toxicity
    MOST_RECENT = "most_recent"  # Keep most recent (by timestamp)


@dataclass
class DedupStats:
    """Statistics from deduplication."""

    clusters_seen: int = 0
    docs_kept: int = 0
    docs_dropped: int = 0
    dropped_by_rule: dict[str, int] = field(default_factory=lambda: defaultdict(int))


class KeepRuleSelector:
    """Select which document to keep from a cluster based on keep rule."""

    def __init__(
        self,
        rule: KeepRule = KeepRule.FIRST,
        quality_field: str = "quality_score",
        toxicity_field: str = "toxicity_score",
        timestamp_field: str = "timestamp",
    ):
        self.rule = rule
        self.quality_field = quality_field
        self.toxicity_field = toxicity_field
        self.timestamp_field = timestamp_field

    def _get_score(self, doc: Document) -> float:
        """Get comparison score for document (higher = more likely to keep)."""
        if self.rule == KeepRule.FIRST:
            return 0  # Not used
        elif self.rule == KeepRule.LONGEST:
            return len(doc.text)
        elif self.rule == KeepRule.MOST_TOKENS:
            return doc.metadata.get("token_count", len(doc.text.split()))
        elif self.rule == KeepRule.HIGHEST_QUALITY:
            scores = doc.metadata.get("quality_scores", {})
            if scores:
                return sum(scores.values()) / len(scores)
            return doc.metadata.get(self.quality_field, 0)
        elif self.rule == KeepRule.LOWEST_TOXICITY:
            return -doc.metadata.get(self.toxicity_field, 0)  # Negate for "lowest"
        elif self.rule == KeepRule.MOST_RECENT:
            return doc.metadata.get(self.timestamp_field, 0)
        return 0

    def select(self, docs: list[Document]) -> tuple[Document, list[Document]]:
        """Select best document from cluster.

        Returns:
            (kept_doc, dropped_docs)
        """
        if not docs:
            raise ValueError("Empty document list")

        if len(docs) == 1 or self.rule == KeepRule.FIRST:
            return docs[0], docs[1:]

        # Score all documents
        scored = [(self._get_score(d), i, d) for i, d in enumerate(docs)]
        scored.sort(reverse=True)

        kept = scored[0][2]
        dropped = [s[2] for s in scored[1:]]
        return kept, dropped


class ClusterDedup:
    """Deduplicate by collecting clusters and applying keep rules."""

    def __init__(
        self,
        keep_rule: KeepRule = KeepRule.FIRST,
        cluster_field: str = "dedup_cluster_id",
    ):
        self.selector = KeepRuleSelector(keep_rule)
        self.cluster_field = cluster_field
        self._clusters: dict[str, list[Document]] = defaultdict(list)
        self.stats = DedupStats()

    def add(self, doc: Document) -> None:
        """Add document to its cluster."""
        cluster_id = doc.metadata.get(self.cluster_field, doc.id)
        self._clusters[cluster_id].append(doc)

    def process_clusters(self) -> Iterator[Document]:
        """Process all clusters and yield kept documents."""
        self.stats = DedupStats()

        for cluster_id, docs in self._clusters.items():
            self.stats.clusters_seen += 1

            if len(docs) == 1:
                self.stats.docs_kept += 1
                yield docs[0]
                continue

            kept, dropped = self.selector.select(docs)
            kept.metadata["dedup_cluster_size"] = len(docs)
            self.stats.docs_kept += 1
            self.stats.docs_dropped += len(dropped)

            for d in dropped:
                d.metadata["dedup_dropped"] = True
                d.metadata["dedup_kept_id"] = kept.id
                self.stats.dropped_by_rule[self.selector.rule.value] += 1

            yield kept

    def reset(self) -> None:
        """Clear clusters."""
        self._clusters.clear()
        self.stats = DedupStats()
