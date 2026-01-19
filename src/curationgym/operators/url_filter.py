"""URL filtering operator with blocklist and heuristic support."""

import re
from pathlib import Path
from urllib.parse import urlparse

from curationgym.core.document import Document


class URLFilter:
    """Filter documents based on URL blocklists and heuristics."""

    # Default suspicious patterns
    DEFAULT_PATTERNS = [
        r"\.xxx$",
        r"porn",
        r"adult",
        r"casino",
        r"gambling",
        r"viagra",
        r"cialis",
        r"pharm",
        r"\.onion$",
        r"torrent",
        r"warez",
        r"crack",
        r"keygen",
    ]

    def __init__(
        self,
        blocklist_paths: list[str | Path] | None = None,
        patterns: list[str] | None = None,
        use_default_patterns: bool = True,
    ):
        self.blocklist: set[str] = set()
        self.patterns: list[re.Pattern] = []

        # Load blocklists
        if blocklist_paths:
            for path in blocklist_paths:
                self._load_blocklist(Path(path))

        # Compile patterns
        all_patterns = []
        if use_default_patterns:
            all_patterns.extend(self.DEFAULT_PATTERNS)
        if patterns:
            all_patterns.extend(patterns)

        self.patterns = [re.compile(p, re.IGNORECASE) for p in all_patterns]

    def _load_blocklist(self, path: Path) -> None:
        """Load domains from blocklist file."""
        if not path.exists():
            return
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    self.blocklist.add(line.lower())

    def __call__(self, doc: Document) -> tuple[bool, str | None]:
        """Check if document passes URL filter.

        Returns:
            (passes, reason): True if passes, False with reason code if rejected.
        """
        url = doc.url
        if not url:
            return True, None  # No URL to filter

        url_lower = url.lower()

        # Parse domain
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
        except Exception:
            return False, "invalid_url"

        # Check blocklist
        if domain in self.blocklist:
            return False, "blocklist_domain"

        # Check if any parent domain is blocked
        parts = domain.split(".")
        for i in range(len(parts)):
            parent = ".".join(parts[i:])
            if parent in self.blocklist:
                return False, "blocklist_parent_domain"

        # Check patterns
        for pattern in self.patterns:
            if pattern.search(url_lower):
                return False, f"pattern:{pattern.pattern}"

        return True, None

    def filter(self, doc: Document) -> Document | None:
        """Filter document, returning None if rejected."""
        passes, reason = self(doc)
        if passes:
            return doc

        # Store rejection reason in metadata
        doc.metadata["url_filter_rejected"] = True
        doc.metadata["url_filter_reason"] = reason
        return None

    def filter_with_reason(self, doc: Document) -> tuple[Document | None, str | None]:
        """Filter document, returning (doc, reason)."""
        passes, reason = self(doc)
        if passes:
            return doc, None
        return None, reason


def create_url_filter(
    blocklist_dir: str | Path | None = None,
    extra_patterns: list[str] | None = None,
) -> URLFilter:
    """Create URL filter with optional blocklist directory."""
    blocklist_paths = []
    if blocklist_dir:
        blocklist_dir = Path(blocklist_dir)
        if blocklist_dir.exists():
            blocklist_paths = list(blocklist_dir.glob("*.txt"))

    return URLFilter(blocklist_paths=blocklist_paths, patterns=extra_patterns)
