"""Unit tests for dataset manifest."""

import json
import tempfile
from pathlib import Path

import pytest
from curationgym.core.manifest import DatasetManifest


class TestDatasetManifest:
    def test_create_manifest(self):
        manifest = DatasetManifest(
            dataset_id="test-dataset-001",
            total_docs=1000,
            total_tokens=50000,
        )
        assert manifest.dataset_id == "test-dataset-001"
        assert manifest.total_docs == 1000
        assert manifest.total_tokens == 50000

    def test_manifest_with_shards(self):
        manifest = DatasetManifest(
            dataset_id="test-002",
            total_docs=2000,
            total_tokens=100000,
            shards=[
                {"path": "shard_0.jsonl", "doc_count": 1000, "size_bytes": 50000},
                {"path": "shard_1.jsonl", "doc_count": 1000, "size_bytes": 50000},
            ],
        )
        assert len(manifest.shards) == 2
        assert manifest.shards[0]["doc_count"] == 1000

    def test_manifest_to_dict(self):
        manifest = DatasetManifest(
            dataset_id="test-003",
            total_docs=500,
            total_tokens=25000,
            code_commit="abc123",
        )
        d = manifest.to_dict()
        assert d["dataset_id"] == "test-003"
        assert d["code_commit"] == "abc123"

    def test_manifest_save_load(self):
        manifest = DatasetManifest(
            dataset_id="test-004",
            total_docs=100,
            total_tokens=5000,
            policy_config={"filters": [{"name": "length"}]},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            manifest.save(path)

            loaded = DatasetManifest.load(path)
            assert loaded.dataset_id == manifest.dataset_id
            assert loaded.total_docs == manifest.total_docs
            assert loaded.policy_config == manifest.policy_config

    def test_manifest_slice_stats(self):
        manifest = DatasetManifest(
            dataset_id="test-005",
            total_docs=1000,
            total_tokens=50000,
            slice_stats={
                "domain:news": {"doc_count": 300, "token_count": 15000},
                "domain:wiki": {"doc_count": 700, "token_count": 35000},
            },
        )
        assert manifest.slice_stats["domain:news"]["doc_count"] == 300
        assert manifest.slice_stats["domain:wiki"]["token_count"] == 35000

    def test_manifest_input_sources(self):
        manifest = DatasetManifest(
            dataset_id="test-006",
            total_docs=100,
            total_tokens=5000,
            input_sources=[
                {"signature": "hf:wikipedia", "doc_count": 100},
            ],
        )
        assert manifest.input_sources[0]["signature"] == "hf:wikipedia"
