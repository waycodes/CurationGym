"""Strict run stamping for reproducibility."""

import hashlib
import json
import os
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class RunStamp:
    """Complete run stamp for reproducibility."""

    # Timing
    timestamp: str
    timezone: str

    # Code version
    git_commit: str
    git_branch: str
    git_dirty: bool
    git_diff_hash: str | None  # Hash of uncommitted changes

    # Dependencies
    python_version: str
    dependency_lock_hash: str

    # Hardware
    hostname: str
    platform: str
    cpu_count: int
    gpu_info: list[str]

    # Run info
    run_id: str
    command: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "timezone": self.timezone,
            "git": {
                "commit": self.git_commit,
                "branch": self.git_branch,
                "dirty": self.git_dirty,
                "diff_hash": self.git_diff_hash,
            },
            "python_version": self.python_version,
            "dependency_lock_hash": self.dependency_lock_hash,
            "hardware": {
                "hostname": self.hostname,
                "platform": self.platform,
                "cpu_count": self.cpu_count,
                "gpu_info": self.gpu_info,
            },
            "run_id": self.run_id,
            "command": self.command,
        }

    def save(self, path: str | Path) -> None:
        """Save run stamp to JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "RunStamp":
        """Load run stamp from JSON."""
        data = json.loads(Path(path).read_text())
        git = data.get("git", {})
        hw = data.get("hardware", {})
        return cls(
            timestamp=data.get("timestamp", ""),
            timezone=data.get("timezone", ""),
            git_commit=git.get("commit", ""),
            git_branch=git.get("branch", ""),
            git_dirty=git.get("dirty", False),
            git_diff_hash=git.get("diff_hash"),
            python_version=data.get("python_version", ""),
            dependency_lock_hash=data.get("dependency_lock_hash", ""),
            hostname=hw.get("hostname", ""),
            platform=hw.get("platform", ""),
            cpu_count=hw.get("cpu_count", 0),
            gpu_info=hw.get("gpu_info", []),
            run_id=data.get("run_id", ""),
            command=data.get("command", ""),
        )


def create_run_stamp(run_id: str, command: str = "") -> RunStamp:
    """Create a complete run stamp capturing current environment."""
    return RunStamp(
        timestamp=datetime.now(timezone.utc).isoformat(),
        timezone=str(datetime.now().astimezone().tzinfo),
        git_commit=_get_git_commit(),
        git_branch=_get_git_branch(),
        git_dirty=_is_git_dirty(),
        git_diff_hash=_get_diff_hash() if _is_git_dirty() else None,
        python_version=platform.python_version(),
        dependency_lock_hash=_get_dependency_hash(),
        hostname=platform.node(),
        platform=platform.platform(),
        cpu_count=os.cpu_count() or 0,
        gpu_info=_get_gpu_info(),
        run_id=run_id,
        command=command,
    )


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _get_git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _is_git_dirty() -> bool:
    try:
        result = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        return bool(result)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return True


def _get_diff_hash() -> str | None:
    """Get hash of uncommitted changes."""
    try:
        diff = subprocess.check_output(
            ["git", "diff", "HEAD"], text=True, stderr=subprocess.DEVNULL
        )
        if diff:
            return hashlib.sha256(diff.encode()).hexdigest()[:16]
        return None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _get_dependency_hash() -> str:
    """Get hash of dependency lock file."""
    lock_files = ["uv.lock", "poetry.lock", "requirements.txt", "Pipfile.lock"]

    for lock_file in lock_files:
        if Path(lock_file).exists():
            content = Path(lock_file).read_bytes()
            return hashlib.sha256(content).hexdigest()[:16]

    return "no_lock_file"


def _get_gpu_info() -> list[str]:
    """Get GPU information if available."""
    gpus = []

    # Try nvidia-smi
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL
        )
        for line in result.strip().split("\n"):
            if line:
                gpus.append(line.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Try torch
    if not gpus:
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpus.append(torch.cuda.get_device_name(i))
        except ImportError:
            pass

    return gpus if gpus else ["none"]
