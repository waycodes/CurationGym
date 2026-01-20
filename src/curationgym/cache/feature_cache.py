"""Feature cache for reusing annotations across policies."""

import hashlib
import json
from pathlib import Path
from typing import Any

from curationgym.core.document import Document


class FeatureCache:
    """Cache intermediate annotations to avoid recomputation.

    Separates annotation (expensive) from selection (cheap) so that
    different policies can reuse the same annotations.
    """

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, dict[str, Any]] = {}

    def _get_cache_key(self, doc_id: str, feature: str) -> str:
        """Get cache key for a document feature."""
        return f"{doc_id}:{feature}"

    def _get_file_path(self, doc_id: str) -> Path:
        """Get file path for document cache."""
        # Use first 2 chars of hash for sharding
        hash_prefix = hashlib.md5(doc_id.encode()).hexdigest()[:2]
        shard_dir = self.cache_dir / hash_prefix
        shard_dir.mkdir(exist_ok=True)
        return shard_dir / f"{doc_id[:50]}.json"

    def get(self, doc_id: str, feature: str) -> Any | None:
        """Get cached feature value."""
        key = self._get_cache_key(doc_id, feature)

        # Check memory cache
        if key in self._memory_cache:
            return self._memory_cache[key]

        # Check disk cache
        file_path = self._get_file_path(doc_id)
        if file_path.exists():
            data = json.loads(file_path.read_text())
            if feature in data:
                self._memory_cache[key] = data[feature]
                return data[feature]

        return None

    def set(self, doc_id: str, feature: str, value: Any) -> None:
        """Set cached feature value."""
        key = self._get_cache_key(doc_id, feature)
        self._memory_cache[key] = value

        # Write to disk
        file_path = self._get_file_path(doc_id)
        if file_path.exists():
            data = json.loads(file_path.read_text())
        else:
            data = {}
        data[feature] = value
        file_path.write_text(json.dumps(data))

    def get_or_compute(
        self,
        doc: Document,
        feature: str,
        compute_fn: callable,
    ) -> Any:
        """Get cached value or compute and cache it."""
        cached = self.get(doc.id, feature)
        if cached is not None:
            return cached

        value = compute_fn(doc)
        self.set(doc.id, feature, value)
        return value

    def has(self, doc_id: str, feature: str) -> bool:
        """Check if feature is cached."""
        return self.get(doc_id, feature) is not None

    def clear_memory(self) -> None:
        """Clear memory cache (disk cache remains)."""
        self._memory_cache.clear()
