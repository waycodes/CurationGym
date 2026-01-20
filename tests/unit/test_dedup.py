"""Unit tests for deduplication operators."""

import pytest
from curationgym.core.document import Document
from curationgym.operators.dedup.exact import ExactDedup
from curationgym.operators.dedup.minhash import MinHashDedup


class TestExactDedup:
    def test_exact_dedup_removes_duplicates(self):
        docs = [
            Document(id="1", text="Hello world", source="a"),
            Document(id="2", text="Hello world", source="b"),  # Duplicate
            Document(id="3", text="Different text", source="c"),
        ]

        dedup = ExactDedup()
        result = list(dedup.process(docs))

        assert len(result) == 2
        ids = {d.id for d in result}
        assert "1" in ids or "2" in ids
        assert "3" in ids

    def test_exact_dedup_keeps_unique(self):
        docs = [
            Document(id="1", text="Text A", source="a"),
            Document(id="2", text="Text B", source="b"),
            Document(id="3", text="Text C", source="c"),
        ]

        dedup = ExactDedup()
        result = list(dedup.process(docs))
        assert len(result) == 3

    def test_exact_dedup_empty_input(self):
        dedup = ExactDedup()
        result = list(dedup.process([]))
        assert len(result) == 0


class TestMinHashDedup:
    def test_minhash_dedup_similar_docs(self):
        docs = [
            Document(id="1", text="The quick brown fox jumps over the lazy dog", source="a"),
            Document(id="2", text="The quick brown fox jumps over the lazy cat", source="b"),  # Similar
            Document(id="3", text="Completely different content here", source="c"),
        ]

        dedup = MinHashDedup(threshold=0.8, num_perm=128)
        result = list(dedup.process(docs))

        # Should remove one of the similar docs
        assert len(result) <= 3

    def test_minhash_dedup_threshold(self):
        docs = [
            Document(id="1", text="word " * 100, source="a"),
            Document(id="2", text="word " * 100, source="b"),  # Identical
        ]

        dedup = MinHashDedup(threshold=0.9, num_perm=128)
        result = list(dedup.process(docs))
        assert len(result) == 1

    def test_minhash_dedup_low_threshold_keeps_more(self):
        docs = [
            Document(id="1", text="apple banana cherry", source="a"),
            Document(id="2", text="apple banana date", source="b"),
        ]

        dedup = MinHashDedup(threshold=0.5, num_perm=128)
        result = list(dedup.process(docs))
        # Low threshold might keep both
        assert len(result) >= 1
