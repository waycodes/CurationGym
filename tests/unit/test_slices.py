"""Unit tests for slice assignment and registry."""

import pytest
from curationgym.core.document import Document
from curationgym.slices.registry import SliceRegistry
from curationgym.slices.assign import assign_slices


class TestSliceRegistry:
    def test_register_slice(self):
        registry = SliceRegistry()
        registry.register("domain:news", lambda doc: "news" in doc.source)
        assert "domain:news" in registry.list_slices()

    def test_assign_single_slice(self):
        registry = SliceRegistry()
        registry.register("domain:wiki", lambda doc: "wiki" in doc.source)

        doc = Document(id="1", text="Test", source="wikipedia")
        slices = registry.assign(doc)
        assert "domain:wiki" in slices

    def test_assign_multiple_slices(self):
        registry = SliceRegistry()
        registry.register("domain:wiki", lambda doc: "wiki" in doc.source)
        registry.register("lang:en", lambda doc: doc.metadata.get("lang") == "en")

        doc = Document(id="1", text="Test", source="wikipedia", metadata={"lang": "en"})
        slices = registry.assign(doc)
        assert "domain:wiki" in slices
        assert "lang:en" in slices

    def test_no_matching_slices(self):
        registry = SliceRegistry()
        registry.register("domain:news", lambda doc: "news" in doc.source)

        doc = Document(id="1", text="Test", source="blog")
        slices = registry.assign(doc)
        assert "domain:news" not in slices


class TestAssignSlices:
    def test_assign_slices_batch(self):
        docs = [
            Document(id="1", text="News article", source="news.com"),
            Document(id="2", text="Wiki page", source="wikipedia"),
            Document(id="3", text="Blog post", source="blog.com"),
        ]

        def domain_classifier(doc):
            if "news" in doc.source:
                return ["domain:news"]
            elif "wiki" in doc.source:
                return ["domain:wiki"]
            return ["domain:other"]

        results = assign_slices(docs, domain_classifier)
        assert results["1"] == ["domain:news"]
        assert results["2"] == ["domain:wiki"]
        assert results["3"] == ["domain:other"]
