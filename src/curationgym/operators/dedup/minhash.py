"""MinHash-based near-duplicate deduplication."""

import hashlib
import struct
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass

from curationgym.core.document import Document


@dataclass
class MinHashConfig:
    """MinHash configuration."""

    num_bands: int = 14  # FineWeb default
    rows_per_band: int = 8  # FineWeb default (14x8 = 112 hash functions)
    ngram_size: int = 5
    seed: int = 42


class MinHashDedup:
    """Near-duplicate detection using MinHash LSH."""

    def __init__(self, config: MinHashConfig | None = None):
        self.config = config or MinHashConfig()
        self.num_hashes = self.config.num_bands * self.config.rows_per_band
        self._buckets: dict[int, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
        self._doc_clusters: dict[str, str] = {}  # doc_id -> cluster_id
        self._cluster_docs: dict[str, list[str]] = defaultdict(list)  # cluster_id -> [doc_ids]

    def _get_ngrams(self, text: str) -> set[str]:
        """Extract character n-grams from text."""
        text = text.lower()
        n = self.config.ngram_size
        if len(text) < n:
            return {text}
        return {text[i : i + n] for i in range(len(text) - n + 1)}

    def _compute_minhash(self, ngrams: set[str]) -> list[int]:
        """Compute MinHash signature."""
        if not ngrams:
            return [0] * self.num_hashes

        signature = []
        for i in range(self.num_hashes):
            min_hash = float("inf")
            for ngram in ngrams:
                h = hashlib.md5(f"{i}:{ngram}".encode()).digest()
                hash_val = struct.unpack("<Q", h[:8])[0]
                min_hash = min(min_hash, hash_val)
            signature.append(int(min_hash))
        return signature

    def _get_band_hashes(self, signature: list[int]) -> list[int]:
        """Split signature into bands and hash each band."""
        band_hashes = []
        rows = self.config.rows_per_band
        for band_idx in range(self.config.num_bands):
            start = band_idx * rows
            band = tuple(signature[start : start + rows])
            band_hash = hash(band)
            band_hashes.append(band_hash)
        return band_hashes

    def _find_cluster(self, doc_id: str, band_hashes: list[int]) -> str | None:
        """Find existing cluster for document based on LSH buckets."""
        for band_idx, band_hash in enumerate(band_hashes):
            candidates = self._buckets[band_idx].get(band_hash, [])
            for candidate_id in candidates:
                if candidate_id in self._doc_clusters:
                    return self._doc_clusters[candidate_id]
        return None

    def _add_to_buckets(self, doc_id: str, band_hashes: list[int]) -> None:
        """Add document to LSH buckets."""
        for band_idx, band_hash in enumerate(band_hashes):
            self._buckets[band_idx][band_hash].append(doc_id)

    def add_document(self, doc: Document) -> tuple[bool, str]:
        """Add document and return (is_first_in_cluster, cluster_id)."""
        ngrams = self._get_ngrams(doc.text)
        signature = self._compute_minhash(ngrams)
        band_hashes = self._get_band_hashes(signature)

        # Find existing cluster
        cluster_id = self._find_cluster(doc.id, band_hashes)

        if cluster_id is None:
            # New cluster
            cluster_id = doc.id
            self._doc_clusters[doc.id] = cluster_id
            self._cluster_docs[cluster_id].append(doc.id)
            self._add_to_buckets(doc.id, band_hashes)
            return True, cluster_id
        else:
            # Existing cluster
            self._doc_clusters[doc.id] = cluster_id
            self._cluster_docs[cluster_id].append(doc.id)
            self._add_to_buckets(doc.id, band_hashes)
            return False, cluster_id

    def process(self, docs: Iterator[Document]) -> Iterator[Document]:
        """Process documents, yielding only first occurrence in each cluster."""
        for doc in docs:
            is_first, cluster_id = self.add_document(doc)
            doc.metadata["dedup_cluster_id"] = cluster_id[:16]
            doc.metadata["dedup_method"] = "minhash"

            if is_first:
                yield doc
            else:
                doc.metadata["dedup_dropped"] = True

    def reset(self) -> None:
        """Clear all state."""
        self._buckets.clear()
        self._doc_clusters.clear()
        self._cluster_docs.clear()

    @property
    def stats(self) -> dict[str, int]:
        """Return dedup statistics."""
        return {
            "num_clusters": len(self._cluster_docs),
            "total_docs": len(self._doc_clusters),
            "duplicates_found": len(self._doc_clusters) - len(self._cluster_docs),
        }
