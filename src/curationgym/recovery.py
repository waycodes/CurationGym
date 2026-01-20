"""Failure recovery mechanisms for CurationGym."""

import json
import signal
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import atexit


@dataclass
class Checkpoint:
    """Pipeline checkpoint for recovery."""

    stage: str
    progress: int
    total: int
    state: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "stage": self.stage,
            "progress": self.progress,
            "total": self.total,
            "state": self.state,
        }, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Checkpoint | None":
        p = Path(path)
        if not p.exists():
            return None
        data = json.loads(p.read_text())
        return cls(
            stage=data["stage"],
            progress=data["progress"],
            total=data["total"],
            state=data.get("state", {}),
        )


class CheckpointManager:
    """Manages checkpointing for pipeline recovery."""

    def __init__(self, checkpoint_dir: str | Path, checkpoint_interval: int = 100):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = checkpoint_interval
        self._current: Checkpoint | None = None
        self._counter = 0

    @property
    def checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "checkpoint.json"

    def start(self, stage: str, total: int, state: dict[str, Any] | None = None) -> None:
        """Start a new checkpoint stage."""
        self._current = Checkpoint(stage=stage, progress=0, total=total, state=state or {})
        self._counter = 0
        self._current.save(self.checkpoint_path)

    def update(self, progress: int, state: dict[str, Any] | None = None) -> None:
        """Update checkpoint progress."""
        if self._current is None:
            return

        self._current.progress = progress
        if state:
            self._current.state.update(state)

        self._counter += 1
        if self._counter >= self.checkpoint_interval:
            self._current.save(self.checkpoint_path)
            self._counter = 0

    def complete(self) -> None:
        """Mark current stage as complete."""
        if self._current:
            self._current.progress = self._current.total
            self._current.save(self.checkpoint_path)
        self._current = None

    def load(self) -> Checkpoint | None:
        """Load existing checkpoint."""
        return Checkpoint.load(self.checkpoint_path)

    def clear(self) -> None:
        """Clear checkpoint."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
        self._current = None


class GracefulShutdown:
    """Handle graceful shutdown on signals."""

    def __init__(self):
        self._shutdown_requested = False
        self._callbacks: list[Callable[[], None]] = []

    def register(self, callback: Callable[[], None]) -> None:
        """Register shutdown callback."""
        self._callbacks.append(callback)

    def install_handlers(self) -> None:
        """Install signal handlers."""
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        atexit.register(self._cleanup)

    def _handle_signal(self, signum: int, frame: Any) -> None:
        """Handle shutdown signal."""
        if self._shutdown_requested:
            sys.exit(1)  # Force exit on second signal

        self._shutdown_requested = True
        print(f"\nShutdown requested (signal {signum}), finishing current work...")
        self._cleanup()

    def _cleanup(self) -> None:
        """Run cleanup callbacks."""
        for callback in self._callbacks:
            try:
                callback()
            except Exception as e:
                print(f"Cleanup error: {e}")

    @property
    def should_stop(self) -> bool:
        return self._shutdown_requested


def validate_shard(shard_path: str | Path) -> tuple[bool, str]:
    """Validate a shard file for corruption.

    Returns:
        (is_valid, error_message)
    """
    path = Path(shard_path)

    if not path.exists():
        return False, f"Shard not found: {path}"

    if path.stat().st_size == 0:
        return False, f"Empty shard: {path}"

    # Check JSONL format
    if path.suffix == ".jsonl":
        try:
            with open(path) as f:
                line_count = 0
                for i, line in enumerate(f):
                    if line.strip():
                        json.loads(line)
                        line_count += 1
            if line_count == 0:
                return False, f"No valid records in shard: {path}"
        except json.JSONDecodeError as e:
            return False, f"JSON error at line {i+1}: {e}"
        except Exception as e:
            return False, f"Read error: {e}"

    # Check Parquet format
    elif path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(path)
            if table.num_rows == 0:
                return False, f"Empty parquet: {path}"
        except ImportError:
            pass  # Skip if pyarrow not available
        except Exception as e:
            return False, f"Parquet error: {e}"

    return True, ""


def recover_from_corrupt_shard(
    shard_path: str | Path,
    output_path: str | Path | None = None,
) -> int:
    """Attempt to recover valid records from corrupt shard.

    Returns:
        Number of recovered records
    """
    path = Path(shard_path)
    out_path = Path(output_path) if output_path else path.with_suffix(".recovered.jsonl")

    recovered = 0

    if path.suffix == ".jsonl":
        with open(path) as f_in, open(out_path, "w") as f_out:
            for line in f_in:
                try:
                    if line.strip():
                        json.loads(line)  # Validate
                        f_out.write(line)
                        recovered += 1
                except json.JSONDecodeError:
                    continue

    return recovered
