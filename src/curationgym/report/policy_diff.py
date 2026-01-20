"""Policy diff tool for comparing two policies."""

from dataclasses import dataclass
from typing import Any


@dataclass
class PolicyDiff:
    """Differences between two policies."""

    filters_added: list[str]
    filters_removed: list[str]
    filters_changed: dict[str, tuple[Any, Any]]

    dedup_changes: dict[str, tuple[Any, Any]]
    decontam_changes: dict[str, tuple[Any, Any]]

    mixing_added: dict[str, float]
    mixing_removed: dict[str, float]
    mixing_changed: dict[str, tuple[float, float]]

    def is_empty(self) -> bool:
        return not any([
            self.filters_added, self.filters_removed, self.filters_changed,
            self.dedup_changes, self.decontam_changes,
            self.mixing_added, self.mixing_removed, self.mixing_changed,
        ])


def diff_policies(policy_a: dict[str, Any], policy_b: dict[str, Any]) -> PolicyDiff:
    """Compare two policies and return differences.

    Args:
        policy_a: First policy (baseline)
        policy_b: Second policy (comparison)

    Returns:
        PolicyDiff with all differences
    """
    # Filters
    filters_a = {f.get("name"): f for f in policy_a.get("filters", [])}
    filters_b = {f.get("name"): f for f in policy_b.get("filters", [])}

    filters_added = [n for n in filters_b if n not in filters_a]
    filters_removed = [n for n in filters_a if n not in filters_b]
    filters_changed = {}
    for name in set(filters_a) & set(filters_b):
        if filters_a[name] != filters_b[name]:
            filters_changed[name] = (filters_a[name], filters_b[name])

    # Dedup
    dedup_a = policy_a.get("dedup", {})
    dedup_b = policy_b.get("dedup", {})
    dedup_changes = _diff_dicts(dedup_a, dedup_b)

    # Decontam
    decontam_a = policy_a.get("decontam", {})
    decontam_b = policy_b.get("decontam", {})
    decontam_changes = _diff_dicts(decontam_a, decontam_b)

    # Mixing
    mixing_a = policy_a.get("mixing", {})
    mixing_b = policy_b.get("mixing", {})

    mixing_added = {k: v for k, v in mixing_b.items() if k not in mixing_a}
    mixing_removed = {k: v for k, v in mixing_a.items() if k not in mixing_b}
    mixing_changed = {}
    for k in set(mixing_a) & set(mixing_b):
        if abs(mixing_a[k] - mixing_b[k]) > 1e-6:
            mixing_changed[k] = (mixing_a[k], mixing_b[k])

    return PolicyDiff(
        filters_added=filters_added,
        filters_removed=filters_removed,
        filters_changed=filters_changed,
        dedup_changes=dedup_changes,
        decontam_changes=decontam_changes,
        mixing_added=mixing_added,
        mixing_removed=mixing_removed,
        mixing_changed=mixing_changed,
    )


def _diff_dicts(a: dict, b: dict) -> dict[str, tuple[Any, Any]]:
    """Find differences between two dicts."""
    changes = {}
    all_keys = set(a) | set(b)
    for k in all_keys:
        va, vb = a.get(k), b.get(k)
        if va != vb:
            changes[k] = (va, vb)
    return changes


def format_policy_diff(diff: PolicyDiff, policy_a_name: str = "A", policy_b_name: str = "B") -> str:
    """Format policy diff as readable markdown.

    Args:
        diff: PolicyDiff object
        policy_a_name: Name for first policy
        policy_b_name: Name for second policy

    Returns:
        Markdown string
    """
    if diff.is_empty():
        return "No differences found."

    lines = [f"# Policy Diff: {policy_a_name} → {policy_b_name}", ""]

    # Filters
    if diff.filters_added or diff.filters_removed or diff.filters_changed:
        lines.append("## Filters")
        lines.append("")
        for f in diff.filters_added:
            lines.append(f"+ **Added**: {f}")
        for f in diff.filters_removed:
            lines.append(f"- **Removed**: {f}")
        for f, (old, new) in diff.filters_changed.items():
            lines.append(f"~ **Changed**: {f}")
            lines.append(f"  - Before: `{old}`")
            lines.append(f"  - After: `{new}`")
        lines.append("")

    # Dedup
    if diff.dedup_changes:
        lines.append("## Deduplication")
        lines.append("")
        for k, (old, new) in diff.dedup_changes.items():
            lines.append(f"- **{k}**: `{old}` → `{new}`")
        lines.append("")

    # Decontam
    if diff.decontam_changes:
        lines.append("## Decontamination")
        lines.append("")
        for k, (old, new) in diff.decontam_changes.items():
            lines.append(f"- **{k}**: `{old}` → `{new}`")
        lines.append("")

    # Mixing
    if diff.mixing_added or diff.mixing_removed or diff.mixing_changed:
        lines.append("## Mixing Weights")
        lines.append("")
        for s, w in diff.mixing_added.items():
            lines.append(f"+ **{s}**: {w:.4f}")
        for s, w in diff.mixing_removed.items():
            lines.append(f"- **{s}**: {w:.4f}")
        for s, (old, new) in sorted(diff.mixing_changed.items(), key=lambda x: abs(x[1][1] - x[1][0]), reverse=True):
            delta = new - old
            sign = "+" if delta > 0 else ""
            lines.append(f"~ **{s}**: {old:.4f} → {new:.4f} ({sign}{delta:.4f})")
        lines.append("")

    return "\n".join(lines)
