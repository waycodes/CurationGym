"""Composition vector extraction for attribution analysis."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CompositionVector:
    """Composition of a dataset run by slice."""

    run_id: str
    slice_token_fractions: dict[str, float]  # slice -> fraction of total tokens
    slice_doc_fractions: dict[str, float]  # slice -> fraction of total docs
    slice_avg_quality: dict[str, float]  # slice -> average quality score
    total_tokens: int = 0
    total_docs: int = 0

    def to_feature_vector(self, slice_names: list[str]) -> list[float]:
        """Convert to fixed-order feature vector for regression."""
        return [self.slice_token_fractions.get(s, 0.0) for s in slice_names]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "slice_token_fractions": self.slice_token_fractions,
            "slice_doc_fractions": self.slice_doc_fractions,
            "slice_avg_quality": self.slice_avg_quality,
            "total_tokens": self.total_tokens,
            "total_docs": self.total_docs,
        }


def extract_composition_from_stats(
    run_id: str,
    slice_stats_path: str | Path,
) -> CompositionVector:
    """Extract composition vector from slice stats file."""
    stats = json.loads(Path(slice_stats_path).read_text())

    total_tokens = stats.get("total", {}).get("token_count", 0)
    total_docs = stats.get("total", {}).get("doc_count", 0)

    token_fractions = {}
    doc_fractions = {}
    avg_quality = {}

    for slice_tag, slice_data in stats.get("by_slice", {}).items():
        slice_tokens = slice_data.get("token_count", 0)
        slice_docs = slice_data.get("doc_count", 0)

        if total_tokens > 0:
            token_fractions[slice_tag] = slice_tokens / total_tokens
        if total_docs > 0:
            doc_fractions[slice_tag] = slice_docs / total_docs

        avg_quality[slice_tag] = slice_data.get("avg_quality_score", 0.0)

    return CompositionVector(
        run_id=run_id,
        slice_token_fractions=token_fractions,
        slice_doc_fractions=doc_fractions,
        slice_avg_quality=avg_quality,
        total_tokens=total_tokens,
        total_docs=total_docs,
    )


def extract_composition_from_manifest(
    manifest_path: str | Path,
) -> CompositionVector:
    """Extract composition from dataset manifest."""
    manifest = json.loads(Path(manifest_path).read_text())
    run_id = manifest.get("dataset_id", "unknown")

    # Look for slice_stats in same directory
    manifest_dir = Path(manifest_path).parent
    stats_path = manifest_dir / "slice_stats.json"

    if stats_path.exists():
        return extract_composition_from_stats(run_id, stats_path)

    # Fallback: empty composition
    return CompositionVector(
        run_id=run_id,
        slice_token_fractions={},
        slice_doc_fractions={},
        slice_avg_quality={},
        total_tokens=manifest.get("stats", {}).get("total_tokens", 0),
        total_docs=manifest.get("stats", {}).get("total_docs", 0),
    )


def collect_compositions(
    run_dirs: list[str | Path],
) -> list[CompositionVector]:
    """Collect composition vectors from multiple runs."""
    compositions = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            compositions.append(extract_composition_from_manifest(manifest_path))
    return compositions


def save_composition_vector(comp: CompositionVector, path: str | Path) -> None:
    """Save composition vector to JSON."""
    Path(path).write_text(json.dumps(comp.to_dict(), indent=2))
