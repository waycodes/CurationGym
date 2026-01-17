"""Dataset manifest for reproducibility and provenance tracking."""

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class DatasetManifest:
    """Manifest describing a curated dataset artifact.

    Captures all information needed to reproduce or understand
    the dataset's provenance.
    """

    # Identity
    dataset_id: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Provenance
    input_sources: list[dict[str, Any]] = field(default_factory=list)
    policy_config: dict[str, Any] = field(default_factory=dict)
    policy_config_hash: str = ""
    code_commit: str = ""
    code_dirty: bool = False
    random_seed: int = 0

    # Storage
    format: str = "jsonl.zst"  # jsonl.zst, parquet, webdataset
    shards: list[dict[str, str]] = field(default_factory=list)  # [{path, checksum, doc_count}]

    # Statistics
    stats: dict[str, Any] = field(default_factory=dict)

    def compute_policy_hash(self) -> str:
        """Compute deterministic hash of policy config."""
        canonical = json.dumps(self.policy_config, sort_keys=True)
        self.policy_config_hash = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        return self.policy_config_hash

    def capture_code_version(self) -> None:
        """Capture git commit and dirty state."""
        try:
            self.code_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            dirty = subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            self.code_dirty = bool(dirty)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.code_commit = "unknown"
            self.code_dirty = True

    def add_shard(self, path: str, checksum: str, doc_count: int) -> None:
        """Register a data shard."""
        self.shards.append({"path": path, "checksum": checksum, "doc_count": doc_count})

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "dataset_id": self.dataset_id,
            "created_at": self.created_at,
            "input_sources": self.input_sources,
            "policy_config": self.policy_config,
            "policy_config_hash": self.policy_config_hash,
            "code_commit": self.code_commit,
            "code_dirty": self.code_dirty,
            "random_seed": self.random_seed,
            "format": self.format,
            "shards": self.shards,
            "stats": self.stats,
        }

    def save(self, path: Path | str) -> None:
        """Save manifest to JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path | str) -> "DatasetManifest":
        """Load manifest from JSON file."""
        data = json.loads(Path(path).read_text())
        return cls(
            dataset_id=data["dataset_id"],
            created_at=data.get("created_at", ""),
            input_sources=data.get("input_sources", []),
            policy_config=data.get("policy_config", {}),
            policy_config_hash=data.get("policy_config_hash", ""),
            code_commit=data.get("code_commit", ""),
            code_dirty=data.get("code_dirty", False),
            random_seed=data.get("random_seed", 0),
            format=data.get("format", "jsonl.zst"),
            shards=data.get("shards", []),
            stats=data.get("stats", {}),
        )
