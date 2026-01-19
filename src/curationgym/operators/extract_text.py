"""HTML-to-text extraction operator using Trafilatura."""

from curationgym.core.document import Document


class TextExtractor:
    """Extract main text content from HTML using Trafilatura."""

    def __init__(
        self,
        include_comments: bool = False,
        include_tables: bool = True,
        favor_precision: bool = True,
        min_extracted_size: int = 100,
    ):
        self.include_comments = include_comments
        self.include_tables = include_tables
        self.favor_precision = favor_precision
        self.min_extracted_size = min_extracted_size

    def __call__(self, doc: Document) -> Document | None:
        """Extract text from HTML document. Returns None if extraction fails."""
        try:
            import trafilatura
        except ImportError:
            raise ImportError("trafilatura required: pip install trafilatura")

        html = doc.text
        if not html or not html.strip():
            return None

        extracted = trafilatura.extract(
            html,
            include_comments=self.include_comments,
            include_tables=self.include_tables,
            favor_precision=self.favor_precision,
        )

        if not extracted or len(extracted) < self.min_extracted_size:
            doc.metadata["extraction_failed"] = True
            doc.metadata["extraction_reason"] = "too_short" if extracted else "no_content"
            return None

        # Compute quality signals
        html_len = len(html)
        text_len = len(extracted)
        extraction_ratio = text_len / html_len if html_len > 0 else 0

        new_metadata = {
            **doc.metadata,
            "original_html_length": html_len,
            "extracted_text_length": text_len,
            "extraction_ratio": round(extraction_ratio, 4),
            "extraction_method": "trafilatura",
        }

        return Document(text=extracted, id=doc.id, metadata=new_metadata)

    def process_batch(self, docs: list[Document]) -> list[Document]:
        """Process a batch of documents."""
        results = []
        for doc in docs:
            result = self(doc)
            if result:
                results.append(result)
        return results


def extract_text(doc: Document, **kwargs) -> Document | None:
    """Convenience function for text extraction."""
    extractor = TextExtractor(**kwargs)
    return extractor(doc)
