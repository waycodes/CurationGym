"""Common Crawl WARC/WET reader for CurationGym."""

import gzip
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from curationgym.core.document import Document


class CommonCrawlReader:
    """Read documents from Common Crawl WARC/WET files."""

    def __init__(
        self,
        paths: list[str | Path],
        file_type: str = "wet",  # wet or warc
        dump: str | None = None,
    ):
        self.paths = [Path(p) for p in paths]
        self.file_type = file_type.lower()
        self.dump = dump

    def __iter__(self) -> Iterator[Document]:
        for path in self.paths:
            yield from self._read_file(path)

    def _read_file(self, path: Path) -> Iterator[Document]:
        """Read a single WET/WARC file."""
        opener = gzip.open if path.suffix == ".gz" else open

        with opener(path, "rt", encoding="utf-8", errors="replace") as f:
            if self.file_type == "wet":
                yield from self._parse_wet(f, path)
            else:
                yield from self._parse_warc(f, path)

    def _parse_wet(self, f: Any, path: Path) -> Iterator[Document]:
        """Parse WET format (pre-extracted text)."""
        current_url = None
        current_id = None
        content_lines: list[str] = []
        in_content = False

        for line in f:
            line = line.rstrip("\n")

            if line.startswith("WARC/"):
                # New record - yield previous if exists
                if current_id and content_lines:
                    yield self._make_document(current_id, current_url, content_lines, path)
                current_url = None
                current_id = None
                content_lines = []
                in_content = False

            elif line.startswith("WARC-Target-URI:"):
                current_url = line.split(":", 1)[1].strip()

            elif line.startswith("WARC-Record-ID:"):
                current_id = line.split(":", 1)[1].strip().strip("<>")

            elif line == "" and current_id and not in_content:
                in_content = True

            elif in_content:
                content_lines.append(line)

        # Yield last record
        if current_id and content_lines:
            yield self._make_document(current_id, current_url, content_lines, path)

    def _parse_warc(self, f: Any, path: Path) -> Iterator[Document]:
        """Parse WARC format (raw HTML) - simplified parser."""
        # For full WARC parsing, use warcio library
        # This is a minimal implementation
        current_url = None
        current_id = None
        content_lines: list[str] = []
        in_content = False

        for line in f:
            line = line.rstrip("\n")

            if line.startswith("WARC/"):
                if current_id and content_lines:
                    yield self._make_document(current_id, current_url, content_lines, path)
                current_url = None
                current_id = None
                content_lines = []
                in_content = False

            elif line.startswith("WARC-Target-URI:"):
                current_url = line.split(":", 1)[1].strip()

            elif line.startswith("WARC-Record-ID:"):
                current_id = line.split(":", 1)[1].strip().strip("<>")

            elif line.startswith("Content-Type: text/html"):
                in_content = True

            elif in_content:
                content_lines.append(line)

        if current_id and content_lines:
            yield self._make_document(current_id, current_url, content_lines, path)

    def _make_document(
        self, doc_id: str, url: str | None, lines: list[str], path: Path
    ) -> Document:
        """Create Document from parsed content."""
        text = "\n".join(lines).strip()
        metadata: dict[str, Any] = {
            "source": "commoncrawl",
            "file_type": self.file_type,
            "source_file": str(path.name),
        }
        if url:
            metadata["url"] = url
        if self.dump:
            metadata["dump"] = self.dump

        return Document(text=text, id=doc_id, metadata=metadata)

    def read(self, limit: int | None = None) -> Iterator[Document]:
        """Read documents with optional limit."""
        for i, doc in enumerate(self):
            if limit and i >= limit:
                break
            yield doc
