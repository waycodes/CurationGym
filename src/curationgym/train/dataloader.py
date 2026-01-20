"""Deterministic data loader for reproducible training."""

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import Dataset, IterableDataset


class DeterministicTextDataset(Dataset):
    """Dataset with deterministic shuffling by shard and sample ID."""

    def __init__(
        self,
        shard_paths: list[Path],
        tokenizer: Any,
        max_length: int = 512,
        seed: int = 42,
    ):
        self.shard_paths = shard_paths
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.seed = seed
        self._samples: list[dict] = []
        self._load_and_shuffle()

    def _load_and_shuffle(self) -> None:
        """Load all samples and shuffle deterministically."""
        import random

        # Load samples with shard info
        for shard_idx, shard_path in enumerate(self.shard_paths):
            with open(shard_path) as f:
                for line_idx, line in enumerate(f):
                    sample = json.loads(line)
                    # Create deterministic ID for shuffling
                    sample["_shuffle_key"] = self._compute_shuffle_key(shard_idx, line_idx)
                    self._samples.append(sample)

        # Sort by shuffle key for deterministic ordering
        self._samples.sort(key=lambda x: x["_shuffle_key"])

        # Shuffle with seed
        rng = random.Random(self.seed)
        rng.shuffle(self._samples)

    def _compute_shuffle_key(self, shard_idx: int, sample_idx: int) -> str:
        """Compute deterministic shuffle key."""
        key = f"{self.seed}:{shard_idx}:{sample_idx}"
        return hashlib.md5(key.encode()).hexdigest()

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self._samples[idx]
        text = sample.get("text", "")

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze(),
        }


class StreamingTextDataset(IterableDataset):
    """Streaming dataset for large-scale training."""

    def __init__(
        self,
        shard_paths: list[Path],
        tokenizer: Any,
        max_length: int = 512,
        seed: int = 42,
    ):
        self.shard_paths = shard_paths
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.seed = seed
        self._tokens_yielded = 0

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        import random

        # Shuffle shards deterministically
        rng = random.Random(self.seed)
        shards = list(self.shard_paths)
        rng.shuffle(shards)

        for shard_path in shards:
            with open(shard_path) as f:
                lines = f.readlines()

            # Shuffle within shard
            rng.shuffle(lines)

            for line in lines:
                sample = json.loads(line)
                text = sample.get("text", "")

                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt",
                )

                self._tokens_yielded += encoding["attention_mask"].sum().item()

                yield {
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                    "labels": encoding["input_ids"].squeeze(),
                }

    @property
    def tokens_yielded(self) -> int:
        return self._tokens_yielded


def create_dataset_from_manifest(
    manifest_path: str | Path,
    tokenizer: Any,
    max_length: int = 512,
    seed: int = 42,
    streaming: bool = False,
) -> Dataset:
    """Create dataset from manifest file."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())

    # Get shard paths
    manifest_dir = manifest_path.parent
    shard_paths = [manifest_dir / s["path"] for s in manifest.get("shards", [])]

    if streaming:
        return StreamingTextDataset(shard_paths, tokenizer, max_length, seed)
    return DeterministicTextDataset(shard_paths, tokenizer, max_length, seed)
