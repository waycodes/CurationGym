"""Local pipeline executor with resumability support."""

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from curationgym.core.document import Document
from curationgym.pipeline.datatrove_adapter import DataTroveAdapter


@dataclass
class TaskStatus:
    """Status of a pipeline task."""

    task_id: int
    status: str = "pending"  # pending, running, completed, failed
    docs_processed: int = 0
    error: str | None = None


@dataclass
class ExecutionState:
    """Persistent state for resumable execution."""

    run_id: str
    total_tasks: int
    tasks: dict[int, TaskStatus] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        """Save state to file."""
        data = {
            "run_id": self.run_id,
            "total_tasks": self.total_tasks,
            "tasks": {
                k: {"task_id": v.task_id, "status": v.status, "docs_processed": v.docs_processed, "error": v.error}
                for k, v in self.tasks.items()
            },
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "ExecutionState":
        """Load state from file."""
        data = json.loads(path.read_text())
        state = cls(run_id=data["run_id"], total_tasks=data["total_tasks"])
        for k, v in data["tasks"].items():
            state.tasks[int(k)] = TaskStatus(**v)
        return state


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

        # Find incomplete tasks
        pending = [
            i for i, t in state.tasks.items()
            if t.status in ("pending", "failed")
        ]

        if not pending:
            return state

        # Execute tasks
        if self.num_workers == 1:
            self._execute_sequential(pipeline, input_shards, pending, state)
        else:
            self._execute_parallel(pipeline, input_shards, pending, state)

        return state

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
            state.save(self._state_path)

            try:
                docs = list(pipeline.run(input_shards[task_id]()))
                self._write_output(task_id, docs)
                state.tasks[task_id].status = "completed"
                state.tasks[task_id].docs_processed = len(docs)
            except Exception as e:
                state.tasks[task_id].status = "failed"
                state.tasks[task_id].error = str(e)

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
                except Exception as e:
                    state.tasks[task_id].status = "failed"
                    state.tasks[task_id].error = str(e)

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
        """Write task output to shard file."""
        output_path = self.output_dir / f"shard_{task_id:05d}.jsonl"
        with open(output_path, "w") as f:
            for doc in docs:
                f.write(json.dumps(doc.to_dict()) + "\n")
