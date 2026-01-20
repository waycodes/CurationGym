"""N-gram overlap decontamination for benchmark protection."""

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum

from curationgym.core.document import Document


class DecontamMode(Enum):
    """Action to take on contaminated documents."""

    DROP = "drop"  # Remove document entirely
    REDACT = "redact"  # Remove contaminated spans
    DOWNWEIGHT = "downweight"  # Mark for reduced sampling
    TAG = "tag"  # Mark but keep unchanged


@dataclass
class ContaminationResult:
    """Result of contamination check for a document."""

    is_contaminated: bool
    overlap_score: float
    eval_source: str | None = None
    matched_ngrams: list[str] = field(default_factory=list)
    action_taken: str | None = None


@dataclass
class DecontamStats:
    """Statistics from decontamination."""

    docs_checked: int = 0
    docs_contaminated: int = 0
    docs_dropped: int = 0
    docs_redacted: int = 0
    docs_downweighted: int = 0
    docs_tagged: int = 0
    by_eval_source: dict[str, int] = field(default_factory=dict)


class NgramDecontaminator:
    """Detect and handle benchmark contamination using n-gram overlap."""

    def __init__(
        self,
        ngram_size: int = 13,
        overlap_threshold: float = 0.8,
        mode: DecontamMode = DecontamMode.DROP,
        max_matched_ngrams_stored: int = 10,
    ):
        self.ngram_size = ngram_size
        self.overlap_threshold = overlap_threshold
        self.mode = mode
        self.max_matched_ngrams_stored = max_matched_ngrams_stored
        self._eval_ngrams: dict[str, set[int]] = {}  # source -> set of ngram hashes
        self._ngram_to_source: dict[int, str] = {}  # hash -> first source
        self.stats = DecontamStats()

    def _hash_ngram(self, ngram: str) -> int:
        """Hash an n-gram to int for efficient storage."""
        return int(hashlib.md5(ngram.lower().encode()).hexdigest()[:16], 16)

    def _extract_ngrams(self, text: str) -> list[str]:
        """Extract word-level n-grams from text."""
        words = text.lower().split()
        if len(words) < self.ngram_size:
            return [" ".join(words)] if words else []
        return [" ".join(words[i : i + self.ngram_size]) for i in range(len(words) - self.ngram_size + 1)]

    def add_eval_data(self, texts: list[str], source: str) -> int:
        """Add evaluation data to the contamination index.

        Returns:
            Number of unique n-grams added.
        """
        if source not in self._eval_ngrams:
            self._eval_ngrams[source] = set()

        added = 0
        for text in texts:
            for ngram in self._extract_ngrams(text):
                h = self._hash_ngram(ngram)
                if h not in self._eval_ngrams[source]:
                    self._eval_ngrams[source].add(h)
                    if h not in self._ngram_to_source:
                        self._ngram_to_source[h] = source
                    added += 1

        return added

    def check(self, doc: Document) -> ContaminationResult:
        """Check document for contamination."""
        if not self._eval_ngrams:
            return ContaminationResult(is_contaminated=False, overlap_score=0.0)

        doc_ngrams = self._extract_ngrams(doc.text)
        if not doc_ngrams:
            return ContaminationResult(is_contaminated=False, overlap_score=0.0)

        # Check overlap with each eval source
        all_eval_hashes = set()
        for hashes in self._eval_ngrams.values():
            all_eval_hashes.update(hashes)

        matched_hashes = set()
        matched_ngrams = []
        primary_source = None

        for ngram in doc_ngrams:
            h = self._hash_ngram(ngram)
            if h in all_eval_hashes:
                matched_hashes.add(h)
                if len(matched_ngrams) < self.max_matched_ngrams_stored:
                    matched_ngrams.append(ngram)
                if primary_source is None and h in self._ngram_to_source:
                    primary_source = self._ngram_to_source[h]

        overlap_score = len(matched_hashes) / len(doc_ngrams) if doc_ngrams else 0.0
        is_contaminated = overlap_score >= self.overlap_threshold

        return ContaminationResult(
            is_contaminated=is_contaminated,
            overlap_score=overlap_score,
            eval_source=primary_source,
            matched_ngrams=matched_ngrams,
        )

    def process(self, doc: Document) -> tuple[Document | None, ContaminationResult]:
        """Process document and apply decontamination mode.

        Returns:
            (processed_doc or None, result)
        """
        self.stats.docs_checked += 1
        result = self.check(doc)

        if not result.is_contaminated:
            return doc, result

        self.stats.docs_contaminated += 1
        if result.eval_source:
            self.stats.by_eval_source[result.eval_source] = self.stats.by_eval_source.get(result.eval_source, 0) + 1

        # Apply mode
        if self.mode == DecontamMode.DROP:
            result.action_taken = "dropped"
            self.stats.docs_dropped += 1
            doc.metadata["contamination_flags"] = {
                "dropped": True,
                "overlap_score": result.overlap_score,
                "eval_source": result.eval_source,
            }
            return None, result

        elif self.mode == DecontamMode.TAG:
            result.action_taken = "tagged"
            self.stats.docs_tagged += 1
            doc.metadata["contamination_flags"] = {
                "contaminated": True,
                "overlap_score": result.overlap_score,
                "eval_source": result.eval_source,
            }
            return doc, result

        elif self.mode == DecontamMode.DOWNWEIGHT:
            result.action_taken = "downweighted"
            self.stats.docs_downweighted += 1
            doc.metadata["contamination_flags"] = {
                "downweight": True,
                "overlap_score": result.overlap_score,
                "eval_source": result.eval_source,
            }
            doc.metadata["sample_weight"] = max(0.1, 1.0 - result.overlap_score)
            return doc, result

        elif self.mode == DecontamMode.REDACT:
            result.action_taken = "redacted"
            self.stats.docs_redacted += 1
            # Simple redaction: remove matched n-grams
            text = doc.text
            for ngram in result.matched_ngrams:
                text = text.replace(ngram, "[REDACTED]")
            doc.metadata["contamination_flags"] = {
                "redacted": True,
                "overlap_score": result.overlap_score,
                "eval_source": result.eval_source,
            }
            return Document(text=text, id=doc.id, metadata=doc.metadata), result

        return doc, result

    def process_stream(self, docs: Iterator[Document]) -> Iterator[Document]:
        """Process document stream, yielding non-dropped documents."""
        for doc in docs:
            processed, _ = self.process(doc)
            if processed is not None:
                yield processed

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = DecontamStats()

    def clear_index(self) -> None:
        """Clear eval data index."""
        self._eval_ngrams.clear()
        self._ngram_to_source.clear()
