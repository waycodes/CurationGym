"""Deterministic slice assignment with versioning."""

import hashlib
import inspect
from typing import Callable

from curationgym.core.document import Document
from curationgym.slices.registry import SliceRegistry, get_registry


# Version hash of slice assignment code
_SLICE_CODE_VERSION: str | None = None


def _compute_code_version() -> str:
    """Compute hash of slice assignment code for versioning."""
    global _SLICE_CODE_VERSION
    if _SLICE_CODE_VERSION is not None:
        return _SLICE_CODE_VERSION

    # Hash the source of key functions
    sources = [
        inspect.getsource(assign_slices),
        inspect.getsource(SliceRegistry._extract_domain),
        inspect.getsource(SliceRegistry._extract_token_bin),
        inspect.getsource(SliceRegistry._extract_quality_bin),
    ]
    combined = "\n".join(sources)
    _SLICE_CODE_VERSION = hashlib.sha256(combined.encode()).hexdigest()[:12]
    return _SLICE_CODE_VERSION


def assign_slices(
    doc: Document,
    registry: SliceRegistry | None = None,
    slice_names: list[str] | None = None,
) -> list[str]:
    """Deterministically assign slice tags to a document.

    This is a pure function - same input always produces same output.

    Args:
        doc: Document to assign slices to
        registry: Slice registry (uses global if None)
        slice_names: Specific slices to assign (all if None)

    Returns:
        List of slice tags (e.g., ["dump=CC-MAIN-2024-10", "quality_bin=high"])
    """
    reg = registry or get_registry()
    tags = reg.assign_slices(doc, slice_names)

    # Add extended slices not in base registry
    tags.extend(_assign_language_score_bin(doc))
    tags.extend(_assign_toxicity_bin(doc))

    return sorted(set(tags))  # Dedupe and sort for determinism


def _assign_language_score_bin(doc: Document) -> list[str]:
    """Assign language score bin."""
    score = doc.language_score
    if score is None:
        return []

    if score >= 0.95:
        return ["lang_score_bin=very_high"]
    elif score >= 0.85:
        return ["lang_score_bin=high"]
    elif score >= 0.7:
        return ["lang_score_bin=medium"]
    return ["lang_score_bin=low"]


def _assign_toxicity_bin(doc: Document) -> list[str]:
    """Assign toxicity bin if score available."""
    score = doc.metadata.get("toxicity_score")
    if score is None:
        return []

    if score < 0.1:
        return ["toxicity_bin=safe"]
    elif score < 0.3:
        return ["toxicity_bin=low_risk"]
    elif score < 0.5:
        return ["toxicity_bin=medium_risk"]
    return ["toxicity_bin=high_risk"]


def assign_and_store(doc: Document, registry: SliceRegistry | None = None) -> Document:
    """Assign slices and store in document metadata."""
    tags = assign_slices(doc, registry)
    doc.metadata["slice_tags"] = tags
    doc.metadata["slice_code_version"] = _compute_code_version()
    return doc


def get_slice_code_version() -> str:
    """Get version hash of slice assignment code."""
    return _compute_code_version()
