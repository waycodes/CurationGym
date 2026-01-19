"""Dedup scope management - per-crawl vs global deduplication."""

from collections.abc import Iterator
from enum import Enum
from typing import Callable

from curationgym.core.document import Document
from curationgym.operators.dedup.exact_doc import ExactDedup
from curationgym.operators.dedup.minhash import MinHashConfig, MinHashDedup


class DedupScope(Enum):
    """Deduplication scope."""

    GLOBAL = "global"  # Dedup across all documents
    PER_DUMP = "per_dump"  # Dedup within each CC dump


class ScopedDedup:
    """Deduplication with configurable scope (global vs per-dump)."""

    def __init__(
        self,
        scope: DedupScope = DedupScope.PER_DUMP,
        method: str = "minhash",  # "exact" or "minhash"
        minhash_config: MinHashConfig | None = None,
        dump_field: str = "dump",
    ):
        self.scope = scope
        self.method = method
        self.minhash_config = minhash_config
        self.dump_field = dump_field
        self._dedupers: dict[str, ExactDedup | MinHashDedup] = {}
        self._global_deduper: ExactDedup | MinHashDedup | None = None

    def _create_deduper(self) -> ExactDedup | MinHashDedup:
        """Create a new deduper instance."""
        if self.method == "exact":
            return ExactDedup()
        return MinHashDedup(self.minhash_config)

    def _get_deduper(self, doc: Document) -> ExactDedup | MinHashDedup:
        """Get appropriate deduper for document."""
        if self.scope == DedupScope.GLOBAL:
            if self._global_deduper is None:
                self._global_deduper = self._create_deduper()
            return self._global_deduper

        # Per-dump scope
        dump = doc.metadata.get(self.dump_field, "unknown")
        if dump not in self._dedupers:
            self._dedupers[dump] = self._create_deduper()
        return self._dedupers[dump]

    def process(self, docs: Iterator[Document]) -> Iterator[Document]:
        """Process documents with scoped deduplication."""
        for doc in docs:
            deduper = self._get_deduper(doc)

            if self.method == "exact":
                is_unique, doc_hash = deduper(doc)
                doc.metadata["content_hash"] = doc_hash
                doc.metadata["dedup_cluster_id"] = doc_hash[:16]
            else:
                is_unique, cluster_id = deduper.add_document(doc)
                doc.metadata["dedup_cluster_id"] = cluster_id[:16]

            doc.metadata["dedup_method"] = self.method
            doc.metadata["dedup_scope"] = self.scope.value

            if is_unique:
                yield doc
            else:
                doc.metadata["dedup_dropped"] = True

    def reset(self) -> None:
        """Clear all dedupers."""
        self._dedupers.clear()
        self._global_deduper = None

    @property
    def stats(self) -> dict[str, dict[str, int]]:
        """Return per-partition stats."""
        if self.scope == DedupScope.GLOBAL and self._global_deduper:
            return {"global": self._global_deduper.stats}
        return {dump: d.stats for dump, d in self._dedupers.items()}
