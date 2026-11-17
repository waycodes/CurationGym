"""Local pipeline executor with resumability support."""

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from curationgym.core.document import Document
from curationgym.pipeline.datatrove_adapter import DataTroveAdapter

# Terminal statuses. A task in any other status has no completion evidence.
_TERMINAL = ("completed", "failed")

# Marker recorded on a task reclaimed from a crashed run, so an operator can
# tell a reclaimed task apart from one that failed inside the pipeline.
RECLAIM_ERROR = "reclaimed: owner process died while task was running"


@dataclass
class TaskStatus:
    """Status of a pipeline task."""

    task_id: int
    status: str = "pending"  # pending, running, completed, failed
    docs_processed: int = 0
    error: str | None = None
    # PID of the process that last marked this task running. Lets a resume
    # tell a crashed owner apart from a live concurrent one.
    owner_pid: int | None = None


@dataclass
class ExecutionState:
    """Persistent state for resumable execution."""

    run_id: str
    total_tasks: int
    tasks: dict[int, TaskStatus] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        """Save state to file.

        Writes to a temp file in the same directory and then renames, so a
        crash mid-save cannot leave a truncated state file. `os.replace` is
        atomic within a filesystem.
        """
        data = {
            "run_id": self.run_id,
            "total_tasks": self.total_tasks,
            "tasks": {
                k: {
                    "task_id": v.task_id,
                    "status": v.status,
                    "docs_processed": v.docs_processed,
                    "error": v.error,
                    "owner_pid": v.owner_pid,
                }
                for k, v in self.tasks.items()
            },
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "ExecutionState":
        """Load state from file."""
        data = json.loads(path.read_text())
        state = cls(run_id=data["run_id"], total_tasks=data["total_tasks"])
        for k, v in data["tasks"].items():
            state.tasks[int(k)] = TaskStatus(**v)
        return state

    def is_complete(self) -> bool:
        """True only when every task reached `completed`.

        A caller cannot infer completion from `execute` returning, because a
        crashed run resumes and returns normally while some tasks are still
        unfinished. This is the single question a caller should ask.
        """
        return all(t.status == "completed" for t in self.tasks.values()) and len(self.tasks) == self.total_tasks

    def unfinished(self) -> list[int]:
        """Task ids that have no completion evidence, sorted."""
        return sorted(tid for tid, t in self.tasks.items() if t.status != "completed")


class LocalExecutor:
    """Local pipeline executor with multi-worker support and resumability."""

    def __init__(
        self,
        output_dir: str | Path,
        num_workers: int = 1,
        tasks_per_worker: int = 1,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_workers = num_workers
        self.tasks_per_worker = tasks_per_worker
        self._state_path = self.output_dir / "execution_state.json"

    def execute(
        self,
        pipeline: DataTroveAdapter,
        input_shards: list[Callable[[], Iterator[Document]]],
        run_id: str,
    ) -> ExecutionState:
        """Execute pipeline on input shards with resumability."""
        # Load or create state
        if self._state_path.exists():
            state = ExecutionState.load(self._state_path)
            if state.run_id != run_id:
                state = self._create_state(run_id, len(input_shards))
        else:
            state = self._create_state(run_id, len(input_shards))

        # Reclaim tasks orphaned by a crashed owner BEFORE computing work.
        # A "running" task whose owner process is gone has no completion
        # evidence, so it must re-run. Without this, a SIGKILLed run resumes
        # and silently skips every in-flight task.
        reclaimed = self._reclaim_orphaned(state)
        if reclaimed:
            state.save(self._state_path)

        # Find work. A task must run when it has no completion evidence, but
        # NOT when a live foreign process still owns it: reclaim deliberately
        # spared those, and scheduling them here would double-execute and let
        # two writers race on the same shard path.
        held = self._held_by_live_owner(state)
        pending = [
            i for i, t in state.tasks.items()
            if t.status != "completed" and i not in held
        ]

        if not pending:
            return state

        # Execute tasks
        if self.num_workers == 1:
            self._execute_sequential(pipeline, input_shards, pending, state)
        else:
            self._execute_parallel(pipeline, input_shards, pending, state)

        return state

    def _held_by_live_owner(self, state: ExecutionState) -> set[int]:
        """Task ids currently owned by a different, live process."""
        me = os.getpid()
        return {
            tid
            for tid, t in state.tasks.items()
            if t.status == "running"
            and t.owner_pid is not None
            and t.owner_pid != me
            and self._pid_alive(t.owner_pid)
        }

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        """True when a process with this pid exists and is not a zombie."""
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            # Exists, owned by another user.
            return True
        # A zombie answers signal 0 but is not doing work. Treat a zombie as
        # dead so its tasks are reclaimed rather than waited on forever.
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            return stat.rsplit(")", 1)[1].split()[0] != "Z"
        except (OSError, IndexError):
            return True

    def _reclaim_orphaned(self, state: ExecutionState) -> list[int]:
        """Move `running` tasks whose owner process died back to `failed`.

        Returns the reclaimed task ids. A task owned by a LIVE pid is left
        alone, so a concurrent healthy run is never stolen from.
        """
        reclaimed: list[int] = []
        me = os.getpid()
        for tid, t in sorted(state.tasks.items()):
            if t.status != "running":
                continue
            pid = t.owner_pid
            if pid is not None and pid != me and self._pid_alive(pid):
                # Someone else is genuinely working on this task.
                continue
            t.status = "failed"
            t.error = RECLAIM_ERROR
            t.docs_processed = 0
            t.owner_pid = None
            # A crashed writer can leave a TRUNCATED shard that is still valid
            # JSONL, so the stale output must be removed, not trusted.
            stale = self.output_dir / f"shard_{tid:05d}.jsonl"
            if stale.exists():
                stale.unlink()
            reclaimed.append(tid)
        return reclaimed

    def _create_state(self, run_id: str, num_tasks: int) -> ExecutionState:
        """Create fresh execution state."""
        state = ExecutionState(run_id=run_id, total_tasks=num_tasks)
        for i in range(num_tasks):
            state.tasks[i] = TaskStatus(task_id=i)
        state.save(self._state_path)
        return state

    def _execute_sequential(
        self,
        pipeline: DataTroveAdapter,
        input_shards: list[Callable[[], Iterator[Document]]],
        task_ids: list[int],
        state: ExecutionState,
    ) -> None:
        """Execute tasks sequentially."""
        for task_id in task_ids:
            state.tasks[task_id].status = "running"
            state.tasks[task_id].owner_pid = os.getpid()
            state.save(self._state_path)

            try:
                docs = list(pipeline.run(input_shards[task_id]()))
                self._write_output(task_id, docs)
                state.tasks[task_id].status = "completed"
                state.tasks[task_id].docs_processed = len(docs)
                state.tasks[task_id].error = None
            except Exception as e:
                state.tasks[task_id].status = "failed"
                state.tasks[task_id].error = str(e)

            state.tasks[task_id].owner_pid = None
            state.save(self._state_path)

    def _execute_parallel(
        self,
        pipeline: DataTroveAdapter,
        input_shards: list[Callable[[], Iterator[Document]]],
        task_ids: list[int],
        state: ExecutionState,
    ) -> None:
        """Execute tasks in parallel using process pool."""
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {}
            for task_id in task_ids:
                state.tasks[task_id].status = "running"
                # The pool owner, not the child, is the process whose death
                # orphans this task, so record the owner pid.
                state.tasks[task_id].owner_pid = os.getpid()
                future = executor.submit(
                    self._run_task, pipeline, input_shards[task_id], task_id
                )
                futures[future] = task_id

            state.save(self._state_path)

            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    docs_count = future.result()
                    state.tasks[task_id].status = "completed"
                    state.tasks[task_id].docs_processed = docs_count
                    state.tasks[task_id].error = None
                except Exception as e:
                    state.tasks[task_id].status = "failed"
                    state.tasks[task_id].error = str(e)

                state.tasks[task_id].owner_pid = None
                state.save(self._state_path)

    def _run_task(
        self,
        pipeline: DataTroveAdapter,
        input_fn: Callable[[], Iterator[Document]],
        task_id: int,
    ) -> int:
        """Run a single task (for parallel execution)."""
        docs = list(pipeline.run(input_fn()))
        self._write_output(task_id, docs)
        return len(docs)

    def _write_output(self, task_id: int, docs: list[Document]) -> None:
        """Write task output to shard file atomically.

        A crash partway through a direct `open(path, "w")` leaves a SHORT but
        syntactically valid JSONL file, which no downstream reader can tell
        apart from a complete shard. Writing to a temp file and renaming means
        the final path either does not exist or is whole.
        """
        output_path = self.output_dir / f"shard_{task_id:05d}.jsonl"
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        with open(tmp_path, "w") as f:
            for doc in docs:
                f.write(json.dumps(doc.to_dict()) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, output_path)
