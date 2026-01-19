"""Exact document-level deduplication."""

import hashlib
import re
from collections.abc import Iterator

from curationgym.core.document import Document


class ExactDedup:
    """Remove exact duplicate documents based on normalized text hash."""

    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self._seen: dict[str, str] = {}  # hash -> first doc_id

    def _normalize_text(self, text: str) -> str:
        """Normalize text for hashing."""
        if not self.normalize:
            return text
        # Lowercase, collapse whitespace, strip
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _hash_text(self, text: str) -> str:
        """Compute hash of normalized text."""
        normalized = self._normalize_text(text)
        return hashlib.sha256(normalized.encode()).hexdigest()

    def __call__(self, doc: Document) -> tuple[bool, str]:
        """Check if document is duplicate.

        Returns:
            (is_unique, hash): True if first occurrence, False if duplicate.
        """
        doc_hash = self._hash_text(doc.text)

        if doc_hash in self._seen:
            return False, doc_hash

        self._seen[doc_hash] = doc.id
        return True, doc_hash

    def process(self, docs: Iterator[Document]) -> Iterator[Document]:
        """Process documents, yielding only unique ones."""
        for doc in docs:
            is_unique, doc_hash = self(doc)
            doc.metadata["content_hash"] = doc_hash
            doc.metadata["dedup_cluster_id"] = doc_hash[:16]

            if is_unique:
                yield doc
            else:
                doc.metadata["dedup_dropped"] = True
                doc.metadata["dedup_method"] = "exact"

    def reset(self) -> None:
        """Clear seen hashes."""
        self._seen.clear()

    @property
    def stats(self) -> dict[str, int]:
        """Return dedup statistics."""
        return {"unique_docs": len(self._seen)}
