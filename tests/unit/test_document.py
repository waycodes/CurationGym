"""Unit tests for core document model."""

import pytest
from curationgym.core.document import Document


class TestDocument:
    def test_create_document(self):
        doc = Document(id="test-1", text="Hello world", source="test")
        assert doc.id == "test-1"
        assert doc.text == "Hello world"
        assert doc.source == "test"

    def test_document_with_metadata(self):
        doc = Document(
            id="test-2",
            text="Sample text",
            source="web",
            metadata={"url": "https://example.com", "lang": "en"},
        )
        assert doc.metadata["url"] == "https://example.com"
        assert doc.metadata["lang"] == "en"

    def test_document_to_dict(self):
        doc = Document(id="test-3", text="Text", source="src")
        d = doc.to_dict()
        assert d["id"] == "test-3"
        assert d["text"] == "Text"
        assert d["source"] == "src"

    def test_document_from_dict(self):
        data = {"id": "test-4", "text": "From dict", "source": "dict", "metadata": {"k": "v"}}
        doc = Document.from_dict(data)
        assert doc.id == "test-4"
        assert doc.text == "From dict"
        assert doc.metadata["k"] == "v"

    def test_document_token_count(self):
        doc = Document(id="test-5", text="one two three four five", source="test")
        # Approximate token count (whitespace split)
        assert doc.approx_token_count() >= 5

    def test_document_hash_stable(self):
        doc1 = Document(id="test-6", text="Same text", source="a")
        doc2 = Document(id="test-6", text="Same text", source="a")
        assert doc1.content_hash() == doc2.content_hash()

    def test_document_hash_differs(self):
        doc1 = Document(id="test-7", text="Text A", source="a")
        doc2 = Document(id="test-7", text="Text B", source="a")
        assert doc1.content_hash() != doc2.content_hash()
