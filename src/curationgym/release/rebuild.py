"""Rebuild dataset from manifest for reproducibility verification."""

import json
import subprocess
from pathlib import Path
from typing import Any

from curationgym.core.manifest import DatasetManifest
from curationgym.policy.schema import Policy
from curationgym.release.run_stamp import RunStamp


def verify_environment(manifest: DatasetManifest, stamp: RunStamp | None) -> list[str]:
    """Verify current environment matches manifest requirements.

    Returns list of warnings/errors.
    """
    warnings = []

    # Check git commit
    if stamp:
        try:
            current_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            if current_commit != stamp.git_commit:
                warnings.append(
                    f"Git commit mismatch: current={current_commit[:8]}, "
                    f"manifest={stamp.git_commit[:8]}"
                )
        except (subprocess.CalledProcessError, FileNotFoundError):
            warnings.append("Cannot verify git commit")

    # Check code version from manifest
    if manifest.code_commit and manifest.code_commit != "unknown":
        try:
            current = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            if not current.startswith(manifest.code_commit):
                warnings.append(f"Code version mismatch: {manifest.code_commit}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    return warnings


def rebuild_from_manifest(
    manifest_path: str | Path,
    output_dir: str | Path,
    verify_only: bool = False,
) -> dict[str, Any]:
    """Rebuild dataset from manifest.

    Args:
        manifest_path: Path to manifest.json
        output_dir: Directory for rebuilt dataset
        verify_only: If True, only verify without rebuilding

    Returns:
        Dict with rebuild status and any warnings
    """
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)

    # Load manifest
    manifest = DatasetManifest.load(manifest_path)

    # Load run stamp if available
    stamp_path = manifest_path.parent / "run_stamp.json"
    stamp = RunStamp.load(stamp_path) if stamp_path.exists() else None

    # Verify environment
    warnings = verify_environment(manifest, stamp)

    result = {
        "manifest_id": manifest.dataset_id,
        "warnings": warnings,
        "verified": len(warnings) == 0,
    }

    if verify_only:
        return result

    # Rebuild
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reconstruct policy
    policy = Policy.from_dict(manifest.policy_config)

    # Get input source
    input_sources = manifest.input_sources
    if not input_sources:
        result["error"] = "No input sources in manifest"
        return result

    # Import and execute
    from curationgym.policy.execute import execute_policy
    from curationgym.io import HFDatasetReader

    # Attempt to recreate input
    input_source = input_sources[0]
    source_sig = input_source.get("signature", "")

    # For HF datasets, try to reload
    if source_sig.startswith("hf:"):
        dataset_name = source_sig.replace("hf:", "")
        reader = HFDatasetReader(dataset_name)
        input_docs = iter(reader)
    else:
        result["error"] = f"Cannot recreate input source: {source_sig}"
        return result

    # Execute policy
    try:
        new_manifest = execute_policy(
            policy=policy,
            input_docs=input_docs,
            output_dir=output_dir,
            input_signature=source_sig,
        )
        result["rebuilt"] = True
        result["new_manifest_id"] = new_manifest.dataset_id
    except Exception as e:
        result["error"] = str(e)
        result["rebuilt"] = False

    return result


def add_reproduce_command(cli):
    """Add reproduce command to CLI."""
    import click

    @cli.command()
    @click.option("--manifest", "-m", required=True, help="Path to manifest.json")
    @click.option("--output", "-o", required=True, help="Output directory")
    @click.option("--verify-only", is_flag=True, help="Only verify, don't rebuild")
    def reproduce(manifest: str, output: str, verify_only: bool):
        """Rebuild dataset from manifest for reproducibility."""
        result = rebuild_from_manifest(manifest, output, verify_only)

        if result.get("warnings"):
            click.echo("Warnings:")
            for w in result["warnings"]:
                click.echo(f"  - {w}")

        if verify_only:
            if result["verified"]:
                click.echo("✓ Environment verified")
            else:
                click.echo("✗ Environment mismatch")
        else:
            if result.get("rebuilt"):
                click.echo(f"✓ Rebuilt: {result['new_manifest_id']}")
            else:
                click.echo(f"✗ Failed: {result.get('error', 'Unknown error')}")

    return reproduce
