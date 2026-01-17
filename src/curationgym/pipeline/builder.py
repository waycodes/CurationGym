"""Pipeline builder from YAML configuration."""

from pathlib import Path
from typing import Any

import yaml

from curationgym.pipeline.datatrove_adapter import DataTroveAdapter


class PipelineBuilder:
    """Build pipelines from YAML configuration files."""

    # Registry of available block types
    BLOCK_TYPES = {
        "reader": ["hf_dataset", "jsonl", "parquet"],
        "filter": ["language", "url", "quality", "length"],
        "mapper": ["extract_text", "token_count", "pii_mask"],
        "dedup": ["exact", "minhash", "semantic"],
        "decontam": ["ngram_overlap"],
    }

    def __init__(self) -> None:
        self._config: dict[str, Any] = {}

    def load_config(self, path: str | Path) -> "PipelineBuilder":
        """Load pipeline configuration from YAML file."""
        with open(path) as f:
            self._config = yaml.safe_load(f)
        self._validate_config()
        return self

    def from_dict(self, config: dict[str, Any]) -> "PipelineBuilder":
        """Load pipeline configuration from dictionary."""
        self._config = config
        self._validate_config()
        return self

    def _validate_config(self) -> None:
        """Validate configuration structure and block parameters."""
        if "pipeline" not in self._config:
            raise ValueError("Config must have 'pipeline' key")

        blocks = self._config["pipeline"].get("blocks", [])
        for i, block in enumerate(blocks):
            if "type" not in block:
                raise ValueError(f"Block {i} missing 'type' field")
            if "name" not in block:
                raise ValueError(f"Block {i} missing 'name' field")

            block_type = block["type"]
            block_name = block["name"]

            # Validate block type exists
            if block_type not in self.BLOCK_TYPES:
                raise ValueError(f"Unknown block type: {block_type}")

            # Validate block name is registered for type
            if block_name not in self.BLOCK_TYPES[block_type]:
                raise ValueError(
                    f"Unknown {block_type} block: {block_name}. "
                    f"Available: {self.BLOCK_TYPES[block_type]}"
                )

    def build(self) -> DataTroveAdapter:
        """Build pipeline from loaded configuration."""
        if not self._config:
            raise ValueError("No configuration loaded")

        adapter = DataTroveAdapter()
        blocks = self._config["pipeline"].get("blocks", [])

        for block in blocks:
            block_type = block["type"]
            block_name = block["name"]
            params = block.get("params", {})

            # Build block based on type (stubs for now)
            if block_type == "filter":
                adapter.add_filter(
                    lambda doc, p=params: True,  # Placeholder
                    name=block_name,
                )
            elif block_type == "mapper":
                adapter.add_mapper(
                    lambda doc, p=params: doc,  # Placeholder
                    name=block_name,
                )

        return adapter

    @property
    def config(self) -> dict[str, Any]:
        """Get loaded configuration."""
        return self._config

    @classmethod
    def register_block(cls, block_type: str, block_name: str) -> None:
        """Register a new block type."""
        if block_type not in cls.BLOCK_TYPES:
            cls.BLOCK_TYPES[block_type] = []
        if block_name not in cls.BLOCK_TYPES[block_type]:
            cls.BLOCK_TYPES[block_type].append(block_name)
