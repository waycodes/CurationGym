"""Decontamination audit report generation."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from curationgym.operators.decontam.ngram_overlap import ContaminationResult, DecontamStats


@dataclass
class AuditEntry:
    """Single audit entry for a flagged document."""

    doc_id: str
    eval_source: str | None
    overlap_score: float
    matched_ngrams: list[str]
    action_taken: str
    doc_preview: str  # First N chars of document


class DecontamAuditor:
    """Generate audit reports for decontamination."""

    def __init__(self, output_dir: str | Path, preview_length: int = 200):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.preview_length = preview_length
        self._entries: list[AuditEntry] = []

    def add_entry(self, doc_id: str, doc_text: str, result: ContaminationResult) -> None:
        """Add audit entry for a flagged document."""
        if not result.is_contaminated:
            return

        entry = AuditEntry(
            doc_id=doc_id,
            eval_source=result.eval_source,
            overlap_score=result.overlap_score,
            matched_ngrams=result.matched_ngrams[:10],  # Limit stored
            action_taken=result.action_taken or "unknown",
            doc_preview=doc_text[: self.preview_length],
        )
        self._entries.append(entry)

    def save_report(self, stats: DecontamStats, filename: str = "decontam_report.json") -> Path:
        """Save audit report to JSON."""
        report = {
            "summary": {
                "docs_checked": stats.docs_checked,
                "docs_contaminated": stats.docs_contaminated,
                "contamination_rate": stats.docs_contaminated / max(1, stats.docs_checked),
                "docs_dropped": stats.docs_dropped,
                "docs_redacted": stats.docs_redacted,
                "docs_downweighted": stats.docs_downweighted,
                "docs_tagged": stats.docs_tagged,
                "by_eval_source": stats.by_eval_source,
            },
            "flagged_documents": [asdict(e) for e in self._entries],
        }

        path = self.output_dir / filename
        path.write_text(json.dumps(report, indent=2))
        return path

    def save_flagged_ids(self, filename: str = "flagged_doc_ids.txt") -> Path:
        """Save list of flagged document IDs."""
        path = self.output_dir / filename
        path.write_text("\n".join(e.doc_id for e in self._entries))
        return path

    def clear(self) -> None:
        """Clear audit entries."""
        self._entries.clear()

    @property
    def num_entries(self) -> int:
        """Number of audit entries."""
        return len(self._entries)
