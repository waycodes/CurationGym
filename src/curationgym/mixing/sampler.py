"""Slice-based sampling primitives for controlled mixing."""

import random
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from curationgym.core.document import Document


@dataclass
class SamplingConfig:
    """Configuration for slice-based sampling."""

    weights: dict[str, float] | None = None  # slice_tag -> weight
    max_tokens_per_slice: dict[str, int] | None = None  # slice_tag -> max tokens
    temperature: float = 1.0  # Temperature for weight smoothing
    seed: int = 42


class SliceSampler:
    """Sample documents with slice-based weighting and caps."""

    def __init__(self, config: SamplingConfig | None = None):
        self.config = config or SamplingConfig()
        self._rng = random.Random(self.config.seed)
        self._slice_tokens: dict[str, int] = defaultdict(int)
        self._slice_docs: dict[str, list[Document]] = defaultdict(list)

    def _get_weight(self, slice_tag: str) -> float:
        """Get sampling weight for a slice."""
        if self.config.weights is None:
            return 1.0
        base_weight = self.config.weights.get(slice_tag, 1.0)

        # Apply temperature smoothing
        if self.config.temperature != 1.0:
            return base_weight ** (1.0 / self.config.temperature)
        return base_weight

    def _is_slice_capped(self, slice_tag: str) -> bool:
        """Check if slice has reached token cap."""
        if self.config.max_tokens_per_slice is None:
            return False
        max_tokens = self.config.max_tokens_per_slice.get(slice_tag)
        if max_tokens is None:
            return False
        return self._slice_tokens[slice_tag] >= max_tokens

    def add_document(self, doc: Document) -> bool:
        """Add document to sampler.

        Returns:
            True if document was added, False if rejected due to cap.
        """
        slice_tags = doc.metadata.get("slice_tags", [])
        token_count = doc.metadata.get("token_count", 0)

        # Check if any slice is capped
        for tag in slice_tags:
            if self._is_slice_capped(tag):
                return False

        # Add to all relevant slices
        for tag in slice_tags:
            self._slice_docs[tag].append(doc)
            self._slice_tokens[tag] += token_count

        return True

    def sample(self, n: int) -> list[Document]:
        """Sample n documents using weighted slice sampling."""
        if not self._slice_docs:
            return []

        # Build weighted pool
        all_docs: list[tuple[Document, float]] = []
        seen_ids: set[str] = set()

        for tag, docs in self._slice_docs.items():
            weight = self._get_weight(tag)
            for doc in docs:
                if doc.id not in seen_ids:
                    all_docs.append((doc, weight))
                    seen_ids.add(doc.id)

        if not all_docs:
            return []

        # Weighted sampling without replacement
        total_weight = sum(w for _, w in all_docs)
        sampled: list[Document] = []
        remaining = list(all_docs)

        for _ in range(min(n, len(remaining))):
            if not remaining:
                break

            # Select based on weights
            r = self._rng.random() * total_weight
            cumulative = 0.0
            selected_idx = 0

            for i, (_, w) in enumerate(remaining):
                cumulative += w
                if cumulative >= r:
                    selected_idx = i
                    break

            doc, weight = remaining.pop(selected_idx)
            total_weight -= weight
            sampled.append(doc)

        return sampled

    def sample_stream(self, docs: Iterator[Document], target_tokens: int) -> Iterator[Document]:
        """Stream documents until target token count reached."""
        total_tokens = 0

        for doc in docs:
            if not self.add_document(doc):
                continue

            token_count = doc.metadata.get("token_count", 0)
            total_tokens += token_count
            yield doc

            if total_tokens >= target_tokens:
                break

    def get_slice_token_counts(self) -> dict[str, int]:
        """Get current token counts per slice."""
        return dict(self._slice_tokens)

    def reset(self) -> None:
        """Clear sampler state."""
        self._slice_tokens.clear()
        self._slice_docs.clear()
        self._rng = random.Random(self.config.seed)
