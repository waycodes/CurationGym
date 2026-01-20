"""Export dataset to HuggingFace datasets format."""

import json
from pathlib import Path
from typing import Any, Iterator

from curationgym.core.manifest import DatasetManifest


def export_to_hf_dataset(
    manifest: DatasetManifest,
    output_dir: str | Path,
    features_schema: dict[str, str] | None = None,
) -> Path:
    """Export shards to HuggingFace Dataset format.

    Args:
        manifest: Dataset manifest
        output_dir: Output directory for HF dataset
        features_schema: Optional feature type mapping

    Returns:
        Path to exported dataset
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default schema
    schema = features_schema or {
        "text": "string",
        "id": "string",
        "source": "string",
        "slice": "string",
    }

    # Create dataset_info.json
    info = {
        "description": f"Dataset {manifest.dataset_id}",
        "features": {k: {"dtype": v, "_type": "Value"} for k, v in schema.items()},
        "splits": {
            "train": {
                "num_examples": manifest.total_docs,
                "num_bytes": sum(s.get("size_bytes", 0) for s in manifest.shards),
            }
        },
        "download_size": sum(s.get("size_bytes", 0) for s in manifest.shards),
        "dataset_size": sum(s.get("size_bytes", 0) for s in manifest.shards),
    }

    (output_dir / "dataset_info.json").write_text(json.dumps(info, indent=2))

    # Create state.json for Arrow format
    state = {
        "_data_files": [{"filename": f"data-{i:05d}.arrow"} for i in range(len(manifest.shards))],
        "_fingerprint": manifest.dataset_id,
        "_format_type": None,
        "_split": "train",
    }
    (output_dir / "state.json").write_text(json.dumps(state, indent=2))

    # Convert shards to Arrow (simplified - actual impl would use pyarrow)
    converted = []
    for i, shard in enumerate(manifest.shards):
        shard_path = Path(shard.get("path", ""))
        if shard_path.exists():
            # Copy/convert shard
            arrow_path = output_dir / f"data-{i:05d}.arrow"
            _convert_shard_to_arrow(shard_path, arrow_path, schema)
            converted.append(str(arrow_path))

    return output_dir


def _convert_shard_to_arrow(
    input_path: Path,
    output_path: Path,
    schema: dict[str, str],
) -> None:
    """Convert a shard to Arrow format."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        # Read input (jsonl or parquet)
        if input_path.suffix == ".parquet":
            table = pq.read_table(input_path)
        else:
            # JSONL
            records = []
            with open(input_path) as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            table = pa.Table.from_pylist(records)

        # Write Arrow
        with pa.OSFile(str(output_path), "wb") as f:
            writer = pa.ipc.new_file(f, table.schema)
            writer.write_table(table)
            writer.close()

    except ImportError:
        # Fallback: just copy as JSONL
        import shutil
        shutil.copy(input_path, output_path.with_suffix(".jsonl"))


def push_to_hub(
    manifest: DatasetManifest,
    repo_id: str,
    token: str | None = None,
    private: bool = True,
) -> str:
    """Push dataset to HuggingFace Hub.

    Args:
        manifest: Dataset manifest
        repo_id: HF repo ID (e.g., "username/dataset-name")
        token: HF token (uses cached if None)
        private: Whether repo should be private

    Returns:
        URL of pushed dataset
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)

        # Create repo
        api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

        # Upload shards
        for shard in manifest.shards:
            shard_path = Path(shard.get("path", ""))
            if shard_path.exists():
                api.upload_file(
                    path_or_fileobj=str(shard_path),
                    path_in_repo=f"data/{shard_path.name}",
                    repo_id=repo_id,
                    repo_type="dataset",
                )

        # Upload manifest
        manifest_path = Path(manifest.shards[0].get("path", "")).parent / "manifest.json"
        if manifest_path.exists():
            api.upload_file(
                path_or_fileobj=str(manifest_path),
                path_in_repo="manifest.json",
                repo_id=repo_id,
                repo_type="dataset",
            )

        return f"https://huggingface.co/datasets/{repo_id}"

    except ImportError:
        raise ImportError("huggingface_hub required: pip install huggingface_hub")
