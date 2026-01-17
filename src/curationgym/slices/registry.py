"""Slice registry and definitions for attribution tracking."""

from dataclasses import dataclass
from typing import Any, Callable

from curationgym.core.document import Document


@dataclass
class SliceDefinition:
    """Definition of a document slice for attribution."""

    name: str
    description: str
    extractor: Callable[[Document], list[str]]


class SliceRegistry:
    """Registry of slice definitions."""

    def __init__(self) -> None:
        self._slices: dict[str, SliceDefinition] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register built-in slice extractors."""
        self.register(SliceDefinition(
            name="dump",
            description="Common Crawl dump identifier",
            extractor=lambda d: [f"dump={d.dump}"] if d.dump else [],
        ))
        self.register(SliceDefinition(
            name="domain",
            description="URL domain",
            extractor=self._extract_domain,
        ))
        self.register(SliceDefinition(
            name="language",
            description="Document language",
            extractor=lambda d: [f"lang={d.language}"] if d.language else [],
        ))
        self.register(SliceDefinition(
            name="token_length_bin",
            description="Token count bin",
            extractor=self._extract_token_bin,
        ))
        self.register(SliceDefinition(
            name="quality_bin",
            description="Quality score bin",
            extractor=self._extract_quality_bin,
        ))

    def register(self, slice_def: SliceDefinition) -> None:
        """Register a slice definition."""
        self._slices[slice_def.name] = slice_def

    def get(self, name: str) -> SliceDefinition | None:
        """Get slice definition by name."""
        return self._slices.get(name)

    def list_slices(self) -> list[str]:
        """List registered slice names."""
        return list(self._slices.keys())

    def assign_slices(self, doc: Document, slice_names: list[str] | None = None) -> list[str]:
        """Assign slice tags to a document."""
        names = slice_names or list(self._slices.keys())
        tags: list[str] = []
        for name in names:
            if slice_def := self._slices.get(name):
                tags.extend(slice_def.extractor(doc))
        return tags

    @staticmethod
    def _extract_domain(doc: Document) -> list[str]:
        url = doc.url
        if not url:
            return []
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            # Extract TLD category
            if domain.endswith(".edu"):
                return [f"domain={domain}", "domain_type=edu"]
            elif domain.endswith(".gov"):
                return [f"domain={domain}", "domain_type=gov"]
            return [f"domain={domain}"]
        except Exception:
            return []

    @staticmethod
    def _extract_token_bin(doc: Document) -> list[str]:
        count = doc.token_count
        if count is None:
            return []
        if count < 128:
            return ["token_bin=tiny"]
        elif count < 512:
            return ["token_bin=small"]
        elif count < 2048:
            return ["token_bin=medium"]
        return ["token_bin=large"]

    @staticmethod
    def _extract_quality_bin(doc: Document) -> list[str]:
        scores = doc.quality_scores
        if not scores:
            return []
        # Use average of available scores
        avg = sum(scores.values()) / len(scores) if scores else 0
        if avg >= 0.8:
            return ["quality_bin=high"]
        elif avg >= 0.5:
            return ["quality_bin=medium"]
        return ["quality_bin=low"]


# Global registry instance
_registry: SliceRegistry | None = None


def get_registry() -> SliceRegistry:
    """Get or create global slice registry."""
    global _registry
    if _registry is None:
        _registry = SliceRegistry()
    return _registry
