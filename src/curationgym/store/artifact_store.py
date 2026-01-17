"""Content-addressed artifact store for caching and reuse."""

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from curationgym.core.manifest import DatasetManifest


class ArtifactStore:
    """Content-addressed store mapping policy+code+input to artifacts.

    Artifacts are stored at: {base_path}/{artifact_hash}/
    Each artifact directory contains:
      - manifest.json: Full provenance
      - shards/: Data files
      - logs/: Processing logs
    """

    def __init__(self, base_path: str | Path = "artifacts"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def compute_artifact_hash(
        self,
        policy_config: dict[str, Any],
        code_version: str,
        input_signature: str,
    ) -> str:
        """Compute stable hash for artifact lookup."""
        key = {
            "policy": json.dumps(policy_config, sort_keys=True),
            "code": code_version,
            "input": input_signature,
        }
        canonical = json.dumps(key, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def get_artifact_path(self, artifact_hash: str) -> Path:
        """Get path for an artifact by hash."""
        return self.base_path / artifact_hash

    def exists(self, artifact_hash: str) -> bool:
        """Check if artifact exists and is complete."""
        path = self.get_artifact_path(artifact_hash)
        return (path / "manifest.json").exists()

    def get_manifest(self, artifact_hash: str) -> DatasetManifest | None:
        """Load manifest for existing artifact."""
        path = self.get_artifact_path(artifact_hash)
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            return None
        return DatasetManifest.load(manifest_path)

    def create_artifact_dir(self, artifact_hash: str) -> Path:
        """Create directory structure for new artifact."""
        path = self.get_artifact_path(artifact_hash)
        (path / "shards").mkdir(parents=True, exist_ok=True)
        (path / "logs").mkdir(parents=True, exist_ok=True)
        return path

    def save_manifest(self, artifact_hash: str, manifest: DatasetManifest) -> None:
        """Save manifest to artifact directory."""
        path = self.get_artifact_path(artifact_hash)
        manifest.save(path / "manifest.json")

    def delete_artifact(self, artifact_hash: str) -> bool:
        """Delete an artifact and its contents."""
        path = self.get_artifact_path(artifact_hash)
        if path.exists():
            shutil.rmtree(path)
            return True
        return False

    def list_artifacts(self) -> list[str]:
        """List all artifact hashes in store."""
        return [p.name for p in self.base_path.iterdir() if p.is_dir()]
