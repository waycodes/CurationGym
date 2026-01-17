"""Structured run logging for CurationGym."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RunLogger:
    """JSONL-based structured logger for experiment runs."""

    def __init__(self, run_dir: Path | str | None = None, run_id: str | None = None):
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
        self.run_dir = Path(run_dir) if run_dir else Path("runs") / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self.run_dir / "events.jsonl"

    def log(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        """Log a structured event."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "event": event_type,
            **(data or {}),
        }
        with open(self._log_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def log_config(self, config: dict[str, Any]) -> None:
        """Log run configuration."""
        self.log("config", {"config": config})
        (self.run_dir / "config.json").write_text(json.dumps(config, indent=2))

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        """Log a metric value."""
        self.log("metric", {"name": name, "value": value, "step": step})

    def log_artifact(self, name: str, path: str) -> None:
        """Log artifact path."""
        self.log("artifact", {"name": name, "path": path})


def get_run_dir(run_id: str, base: str = "runs") -> Path:
    """Get standard run directory path."""
    return Path(base) / run_id
