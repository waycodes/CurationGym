"""Integration test for full CurationGym pipeline."""

import json
import tempfile
from pathlib import Path

import pytest
from curationgym.core.document import Document
from curationgym.core.manifest import DatasetManifest
from curationgym.policy.schema import Policy
from curationgym.policy.execute import execute_policy
from curationgym.release.run_stamp import create_run_stamp
from curationgym.release.dataset_card import generate_dataset_card
from curationgym.report.experiment_report import generate_experiment_report


class TestFullPipeline:
    """End-to-end pipeline test: load → filter → dedup → output."""

    @pytest.fixture
    def sample_docs(self):
        """Create sample documents for testing."""
        return [
            Document(id="1", text="This is a news article about technology. " * 20, source="news.com", metadata={"lang": "en"}),
            Document(id="2", text="Wikipedia article about science and research. " * 20, source="wikipedia.org", metadata={"lang": "en"}),
            Document(id="3", text="This is a news article about technology. " * 20, source="news2.com", metadata={"lang": "en"}),  # Duplicate content
            Document(id="4", text="Short text", source="blog.com", metadata={"lang": "en"}),  # Too short
            Document(id="5", text="Another unique blog post with interesting content. " * 15, source="blog.com", metadata={"lang": "en"}),
        ]

    @pytest.fixture
    def sample_policy(self):
        """Create sample policy."""
        return Policy(
            filters=[
                {"name": "length", "min_tokens": 50},
            ],
            dedup={"method": "exact"},
            mixing={"domain:news": 0.5, "domain:wiki": 0.3, "domain:other": 0.2},
        )

    def test_execute_policy(self, sample_docs, sample_policy):
        """Test policy execution produces valid manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = execute_policy(
                policy=sample_policy,
                input_docs=iter(sample_docs),
                output_dir=tmpdir,
                input_signature="test:sample",
            )

            assert manifest is not None
            assert manifest.total_docs > 0
            assert manifest.total_docs < len(sample_docs)  # Some filtered/deduped
            assert len(manifest.shards) > 0

    def test_manifest_persistence(self, sample_docs, sample_policy):
        """Test manifest can be saved and loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = execute_policy(
                policy=sample_policy,
                input_docs=iter(sample_docs),
                output_dir=tmpdir,
                input_signature="test:sample",
            )

            # Save
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest.save(manifest_path)

            # Load
            loaded = DatasetManifest.load(manifest_path)
            assert loaded.dataset_id == manifest.dataset_id
            assert loaded.total_docs == manifest.total_docs

    def test_run_stamp_creation(self):
        """Test run stamp captures environment."""
        stamp = create_run_stamp(run_id="test-run-001", command="pytest")

        assert stamp.run_id == "test-run-001"
        assert stamp.python_version
        assert stamp.hostname
        assert stamp.cpu_count > 0

    def test_dataset_card_generation(self, sample_docs, sample_policy):
        """Test dataset card generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = execute_policy(
                policy=sample_policy,
                input_docs=iter(sample_docs),
                output_dir=tmpdir,
                input_signature="test:sample",
            )

            card = generate_dataset_card(manifest)

            assert manifest.dataset_id in card
            assert "## Summary" in card
            assert "## Processing Pipeline" in card

    def test_experiment_report_generation(self, sample_policy):
        """Test experiment report generation."""
        eval_results = {
            "hellaswag": 0.45,
            "arc_easy": 0.52,
            "winogrande": 0.51,
        }
        compute_stats = {
            "gpu_hours": 2.5,
            "total_flops": 1e15,
            "training_time_hours": 2.0,
            "eval_time_hours": 0.5,
        }

        report = generate_experiment_report(
            experiment_id="test-exp-001",
            policy=sample_policy.to_dict(),
            eval_results=eval_results,
            compute_stats=compute_stats,
            total_trials=10,
        )

        assert "test-exp-001" in report
        assert "hellaswag" in report
        assert "0.45" in report


class TestPipelineRecovery:
    """Test pipeline recovery from failures."""

    def test_empty_input_handling(self):
        """Test pipeline handles empty input gracefully."""
        policy = Policy(filters=[{"name": "length", "min_tokens": 10}])

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = execute_policy(
                policy=policy,
                input_docs=iter([]),
                output_dir=tmpdir,
                input_signature="test:empty",
            )

            assert manifest.total_docs == 0

    def test_all_filtered_handling(self):
        """Test pipeline handles all docs being filtered."""
        docs = [
            Document(id="1", text="short", source="a"),
            Document(id="2", text="tiny", source="b"),
        ]
        policy = Policy(filters=[{"name": "length", "min_tokens": 1000}])

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = execute_policy(
                policy=policy,
                input_docs=iter(docs),
                output_dir=tmpdir,
                input_signature="test:filtered",
            )

            assert manifest.total_docs == 0
