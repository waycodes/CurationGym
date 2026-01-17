"""Tests for pipeline builder."""

import pytest

from curationgym.pipeline.builder import PipelineBuilder


def test_builder_validates_missing_pipeline_key():
    builder = PipelineBuilder()
    with pytest.raises(ValueError, match="must have 'pipeline' key"):
        builder.from_dict({})


def test_builder_validates_block_type():
    builder = PipelineBuilder()
    config = {
        "pipeline": {
            "blocks": [{"type": "invalid", "name": "test"}]
        }
    }
    with pytest.raises(ValueError, match="Unknown block type"):
        builder.from_dict(config)


def test_builder_validates_block_name():
    builder = PipelineBuilder()
    config = {
        "pipeline": {
            "blocks": [{"type": "filter", "name": "invalid_filter"}]
        }
    }
    with pytest.raises(ValueError, match="Unknown filter block"):
        builder.from_dict(config)


def test_builder_accepts_valid_config():
    builder = PipelineBuilder()
    config = {
        "pipeline": {
            "blocks": [
                {"type": "filter", "name": "language", "params": {"lang": "en"}},
                {"type": "mapper", "name": "token_count", "params": {}},
            ]
        }
    }
    builder.from_dict(config)
    assert len(builder.config["pipeline"]["blocks"]) == 2


def test_builder_builds_adapter():
    builder = PipelineBuilder()
    config = {
        "pipeline": {
            "blocks": [
                {"type": "filter", "name": "language", "params": {}},
            ]
        }
    }
    adapter = builder.from_dict(config).build()
    assert adapter is not None
