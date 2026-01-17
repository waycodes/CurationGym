"""Canonical Document model for CurationGym.

Mirrors DataTrove's Document concept with extended metadata for
curation tracking, dedup, decontamination, and slice attribution.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A single document in the curation pipeline.

    Attributes:
        text: The document content.
        id: Unique identifier (typically content hash or source ID).
        metadata: Extended metadata for tracking provenance and curation state.
    """

    text: str
    id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    # Convenience accessors for common metadata fields
    @property
    def source(self) -> str | None:
        return self.metadata.get("source")

    @property
    def dump(self) -> str | None:
        """Common Crawl dump identifier (e.g., CC-MAIN-2024-10)."""
        return self.metadata.get("dump")

    @property
    def url(self) -> str | None:
        return self.metadata.get("url")

    @property
    def language(self) -> str | None:
        return self.metadata.get("language")

    @property
    def language_score(self) -> float | None:
        return self.metadata.get("language_score")

    @property
    def token_count(self) -> int | None:
        return self.metadata.get("token_count")

    @property
    def quality_scores(self) -> dict[str, float]:
        return self.metadata.get("quality_scores", {})

    @property
    def dedup_cluster_id(self) -> str | None:
        return self.metadata.get("dedup_cluster_id")

    @property
    def contamination_flags(self) -> dict[str, Any]:
        return self.metadata.get("contamination_flags", {})

    @property
    def slice_tags(self) -> list[str]:
        return self.metadata.get("slice_tags", [])

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {"text": self.text, "id": self.id, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        """Deserialize from dictionary."""
        return cls(text=data["text"], id=data["id"], metadata=data.get("metadata", {}))
