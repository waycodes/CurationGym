"""Performance profiling utilities for CurationGym."""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator
import functools


@dataclass
class ProfileResult:
    """Result of a profiling session."""

    name: str
    duration_seconds: float
    memory_mb: float | None = None
    calls: int = 1
    extra: dict[str, Any] = field(default_factory=dict)


class Profiler:
    """Simple profiler for tracking execution time and memory."""

    def __init__(self):
        self.results: list[ProfileResult] = []
        self._start_times: dict[str, float] = {}

    def start(self, name: str) -> None:
        """Start timing a section."""
        self._start_times[name] = time.perf_counter()

    def stop(self, name: str, extra: dict[str, Any] | None = None) -> ProfileResult:
        """Stop timing and record result."""
        if name not in self._start_times:
            raise ValueError(f"Timer '{name}' was not started")

        duration = time.perf_counter() - self._start_times.pop(name)
        memory = _get_memory_mb()

        result = ProfileResult(
            name=name,
            duration_seconds=duration,
            memory_mb=memory,
            extra=extra or {},
        )
        self.results.append(result)
        return result

    @contextmanager
    def section(self, name: str) -> Generator[None, None, None]:
        """Context manager for profiling a section."""
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def summary(self) -> dict[str, Any]:
        """Get profiling summary."""
        by_name: dict[str, list[ProfileResult]] = {}
        for r in self.results:
            by_name.setdefault(r.name, []).append(r)

        summary = {}
        for name, results in by_name.items():
            total_time = sum(r.duration_seconds for r in results)
            avg_time = total_time / len(results)
            max_memory = max((r.memory_mb or 0) for r in results)

            summary[name] = {
                "total_seconds": total_time,
                "avg_seconds": avg_time,
                "calls": len(results),
                "max_memory_mb": max_memory,
            }

        return summary

    def report(self) -> str:
        """Generate profiling report."""
        lines = ["# Performance Profile", ""]
        lines.append("| Section | Calls | Total (s) | Avg (s) | Max Mem (MB) |")
        lines.append("|---------|-------|-----------|---------|--------------|")

        summary = self.summary()
        for name, stats in sorted(summary.items(), key=lambda x: -x[1]["total_seconds"]):
            lines.append(
                f"| {name} | {stats['calls']} | {stats['total_seconds']:.2f} | "
                f"{stats['avg_seconds']:.4f} | {stats['max_memory_mb']:.1f} |"
            )

        return "\n".join(lines)


def _get_memory_mb() -> float | None:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return None


def profile(func):
    """Decorator to profile a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start

        # Store in function attribute
        if not hasattr(wrapper, "_profile_data"):
            wrapper._profile_data = []
        wrapper._profile_data.append(duration)

        return result
    return wrapper


class MemoryTracker:
    """Track memory usage over time."""

    def __init__(self):
        self.snapshots: list[tuple[str, float]] = []

    def snapshot(self, label: str) -> None:
        """Take memory snapshot."""
        mem = _get_memory_mb()
        if mem is not None:
            self.snapshots.append((label, mem))

    def peak(self) -> float:
        """Get peak memory usage."""
        if not self.snapshots:
            return 0.0
        return max(s[1] for s in self.snapshots)

    def report(self) -> str:
        """Generate memory report."""
        if not self.snapshots:
            return "No memory snapshots recorded."

        lines = ["# Memory Usage", ""]
        for label, mem in self.snapshots:
            lines.append(f"- {label}: {mem:.1f} MB")
        lines.append(f"\nPeak: {self.peak():.1f} MB")

        return "\n".join(lines)


# Global profiler instance
_global_profiler = Profiler()


def get_profiler() -> Profiler:
    """Get global profiler instance."""
    return _global_profiler
