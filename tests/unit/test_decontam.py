"""Unit tests for decontamination operators."""

import pytest
from curationgym.core.document import Document
from curationgym.operators.decontam.ngram import NgramDecontaminator, DecontamMode


class TestNgramDecontaminator:
    def test_decontam_removes_contaminated(self):
        benchmark_texts = ["The answer is forty two"]
        docs = [
            Document(id="1", text="The answer is forty two exactly", source="a"),  # Contaminated
            Document(id="2", text="Completely unrelated content here", source="b"),
        ]

        decontam = NgramDecontaminator(
            benchmark_texts=benchmark_texts,
            ngram_size=5,
            mode=DecontamMode.REMOVE,
        )
        result = list(decontam.process(docs))

        assert len(result) == 1
        assert result[0].id == "2"

    def test_decontam_flag_mode(self):
        benchmark_texts = ["test benchmark text"]
        docs = [
            Document(id="1", text="This contains test benchmark text inside", source="a"),
        ]

        decontam = NgramDecontaminator(
            benchmark_texts=benchmark_texts,
            ngram_size=3,
            mode=DecontamMode.FLAG,
        )
        result = list(decontam.process(docs))

        assert len(result) == 1
        assert result[0].metadata.get("contaminated") is True

    def test_decontam_no_match(self):
        benchmark_texts = ["specific benchmark phrase"]
        docs = [
            Document(id="1", text="Unrelated document content", source="a"),
            Document(id="2", text="Another clean document", source="b"),
        ]

        decontam = NgramDecontaminator(
            benchmark_texts=benchmark_texts,
            ngram_size=5,
            mode=DecontamMode.REMOVE,
        )
        result = list(decontam.process(docs))
        assert len(result) == 2

    def test_decontam_ngram_size(self):
        benchmark_texts = ["one two three four five"]
        docs = [
            Document(id="1", text="one two three four five six", source="a"),
        ]

        # With ngram_size=5, should match
        decontam5 = NgramDecontaminator(benchmark_texts, ngram_size=5, mode=DecontamMode.REMOVE)
        result5 = list(decontam5.process(docs))
        assert len(result5) == 0

        # With ngram_size=10, should not match
        decontam10 = NgramDecontaminator(benchmark_texts, ngram_size=10, mode=DecontamMode.REMOVE)
        result10 = list(decontam10.process(docs))
        assert len(result10) == 1

    def test_decontam_empty_benchmark(self):
        decontam = NgramDecontaminator(benchmark_texts=[], ngram_size=5, mode=DecontamMode.REMOVE)
        docs = [Document(id="1", text="Any text", source="a")]
        result = list(decontam.process(docs))
        assert len(result) == 1
