"""Automatic dataset card generation."""

from datetime import datetime
from pathlib import Path
from typing import Any

from curationgym.core.manifest import DatasetManifest
from curationgym.release.run_stamp import RunStamp


def generate_dataset_card(
    manifest: DatasetManifest,
    stamp: RunStamp | None = None,
    extra_info: dict[str, Any] | None = None,
) -> str:
    """Generate a dataset card in markdown format.

    Args:
        manifest: Dataset manifest
        stamp: Optional run stamp
        extra_info: Additional info (license, limitations, etc.)

    Returns:
        Markdown string
    """
    extra = extra_info or {}
    lines = []

    # Header
    lines.append(f"# {manifest.dataset_id}")
    lines.append("")
    lines.append(f"Generated: {manifest.created_at}")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Documents**: {manifest.total_docs:,}")
    lines.append(f"- **Tokens**: {manifest.total_tokens:,}")
    lines.append(f"- **Shards**: {len(manifest.shards)}")
    lines.append("")

    # Source Data
    lines.append("## Source Data")
    lines.append("")
    for src in manifest.input_sources:
        sig = src.get("signature", "unknown")
        lines.append(f"- `{sig}`")
    lines.append("")

    # Processing Pipeline
    lines.append("## Processing Pipeline")
    lines.append("")

    policy = manifest.policy_config
    if policy:
        # Filters
        filters = policy.get("filters", [])
        if filters:
            lines.append("### Filters")
            lines.append("")
            for f in filters:
                name = f.get("name", "unknown")
                params = {k: v for k, v in f.items() if k != "name"}
                lines.append(f"- **{name}**: `{params}`")
            lines.append("")

        # Deduplication
        dedup = policy.get("dedup", {})
        if dedup:
            lines.append("### Deduplication")
            lines.append("")
            lines.append(f"- Method: `{dedup.get('method', 'none')}`")
            if dedup.get("scope"):
                lines.append(f"- Scope: `{dedup.get('scope')}`")
            lines.append("")

        # Decontamination
        decontam = policy.get("decontam", {})
        if decontam:
            lines.append("### Decontamination")
            lines.append("")
            lines.append(f"- Mode: `{decontam.get('mode', 'none')}`")
            lines.append(f"- N-gram: `{decontam.get('ngram_size', 13)}`")
            if decontam.get("benchmarks"):
                lines.append(f"- Benchmarks: `{decontam.get('benchmarks')}`")
            lines.append("")

        # Mixing
        mixing = policy.get("mixing", {})
        if mixing:
            lines.append("### Mixing Weights")
            lines.append("")
            lines.append("| Slice | Weight |")
            lines.append("|-------|--------|")
            for slice_name, weight in mixing.items():
                lines.append(f"| {slice_name} | {weight:.3f} |")
            lines.append("")

    # Slice Statistics
    if manifest.slice_stats:
        lines.append("## Slice Distribution")
        lines.append("")
        lines.append("| Slice | Documents | Tokens |")
        lines.append("|-------|-----------|--------|")
        for slice_name, stats in sorted(manifest.slice_stats.items()):
            docs = stats.get("doc_count", 0)
            tokens = stats.get("token_count", 0)
            lines.append(f"| {slice_name} | {docs:,} | {tokens:,} |")
        lines.append("")

    # Known Limitations
    limitations = extra.get("limitations", [])
    if limitations:
        lines.append("## Known Limitations")
        lines.append("")
        for lim in limitations:
            lines.append(f"- {lim}")
        lines.append("")

    # License
    license_info = extra.get("license", "See source data licenses")
    lines.append("## License")
    lines.append("")
    lines.append(license_info)
    lines.append("")

    # Reproducibility
    lines.append("## Reproducibility")
    lines.append("")
    lines.append(f"- Code commit: `{manifest.code_commit}`")
    if stamp:
        lines.append(f"- Git dirty: `{stamp.git_dirty}`")
        lines.append(f"- Dependency hash: `{stamp.dependency_lock_hash}`")
    lines.append("")
    lines.append("Rebuild with:")
    lines.append("```bash")
    lines.append(f"curationgym reproduce --manifest manifest.json --output ./rebuilt/")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def save_dataset_card(
    manifest: DatasetManifest,
    output_path: str | Path,
    stamp: RunStamp | None = None,
    extra_info: dict[str, Any] | None = None,
) -> Path:
    """Generate and save dataset card."""
    card = generate_dataset_card(manifest, stamp, extra_info)
    path = Path(output_path)
    path.write_text(card)
    return path
