"""Crash-consistency tests for LocalExecutor resume.

Each test targets one post-crash shape proven with a real SIGKILL against the
pre-fix executor:

  1. Sequential crash: one task left `running`, resume silently skipped it and
     produced 2 of 3 shards while raising nothing.
  2. Parallel crash: ALL tasks left `running`, so `pending` was empty and
     resume returned at the early-return line having run zero tasks.
  3. Mid-write crash: a truncated shard remained on disk as fully valid JSONL
     (1600 parsable lines, 0 torn), so neither file existence nor JSON validity
     is completion evidence.

The tests use a crashed-owner pid rather than really killing a process, so they
run fast and deterministically in CI. The SIGKILL probes that discovered these
shapes live in the drill artifact.
"""

import json
import os
import subprocess
import sys

import pytest

from curationgym.core.document import Document
from curationgym.pipeline.datatrove_adapter import DataTroveAdapter
from curationgym.pipeline.executors import local as local_mod
from curationgym.pipeline.executors.local import (
    ExecutionState,
    LocalExecutor,
    TaskStatus,
)

# Imported defensively ON PURPOSE. If these were hard imports, running this
# module against pre-fix source would fail at COLLECTION with an ImportError,
# which proves only that the module changed. Degrading to a sentinel makes the
# BEHAVIORAL assertions below the thing that fails, which is the non-vacuity
# proof that matters.
RECLAIM_ERROR = getattr(local_mod, "RECLAIM_ERROR", "<no reclaim marker in this build>")


class Shard:
    """Picklable shard callable.

    A nested closure cannot cross a ProcessPoolExecutor boundary, so the
    parallel path requires a module-level callable. See
    `test_parallel_path_rejects_unpicklable_closure_shards`.
    """

    def __init__(self, task_id: int, n_docs: int = 1):
        self.task_id = task_id
        self.n_docs = n_docs

    def __call__(self):
        for i in range(self.n_docs):
            yield Document(text=f"doc {self.task_id}-{i}", id=f"t{self.task_id}-{i}")


def dead_pid() -> int:
    """Return a pid that is guaranteed not to be running.

    Spawn a trivial child, wait for it, then reap it with `wait()` so no
    zombie remains. After reaping, the pid answers no signal.
    """
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    pid = proc.pid
    proc.wait()
    # Popen.wait reaps the child, so the pid is fully gone, not a zombie.
    return pid


def crashed_state(tmp_path, run_id: str, running_ids, total: int, owner=None):
    """Write a state file in the shape a SIGKILL leaves behind.

    `owner_pid` is set via setattr so this helper also runs against a pre-fix
    TaskStatus that has no such field, keeping the fail-before proof about
    behavior rather than about attribute errors.
    """
    state = ExecutionState(run_id=run_id, total_tasks=total)
    for i in range(total):
        state.tasks[i] = TaskStatus(task_id=i)
    for i in running_ids:
        state.tasks[i].status = "running"
        setattr(state.tasks[i], "owner_pid", owner)
    state.save(tmp_path / "execution_state.json")
    return state


def reclaim_via(ex, state) -> list[int]:
    """Call the reclaim transition, or report none when the build lacks it.

    Pre-fix source has no `_reclaim_orphaned`, so returning [] here makes the
    assertion (not the missing attribute) the thing that fails.
    """
    fn = getattr(ex, "_reclaim_orphaned", None)
    if fn is None:
        return []
    return fn(state)


def unfinished_of(state) -> list[int]:
    """Task ids without completion evidence, computed without the new API."""
    return sorted(tid for tid, t in state.tasks.items() if t.status != "completed")


def is_complete_of(state) -> bool:
    """Completion, computed without depending on the new `is_complete`."""
    return len(state.tasks) == state.total_tasks and not unfinished_of(state)


class TestSequentialCrashResume:
    def test_orphaned_running_task_is_rerun_not_skipped(self, tmp_path):
        """Pre-fix this produced 2 of 3 shards with task 1 stuck at running."""
        crashed_state(tmp_path, "run-a", running_ids=[1], total=3, owner=dead_pid())
        # Task 0 had completed before the crash, so its shard is already on
        # disk. A completed task with no shard would be an impossible state.
        st = ExecutionState.load(tmp_path / "execution_state.json")
        st.tasks[0].status = "completed"
        st.tasks[0].docs_processed = 1
        st.save(tmp_path / "execution_state.json")
        (tmp_path / "shard_00000.jsonl").write_text(
            json.dumps({"text": "doc 0-0", "id": "t0-0", "metadata": {}}) + "\n"
        )

        ex = LocalExecutor(output_dir=tmp_path, num_workers=1)
        state = ex.execute(DataTroveAdapter(), [Shard(i) for i in range(3)], run_id="run-a")

        assert state.tasks[1].status == "completed", "orphaned task was skipped"
        assert unfinished_of(state) == []
        assert is_complete_of(state)
        shards = sorted(p.name for p in tmp_path.glob("shard_*.jsonl"))
        assert len(shards) == 3, f"expected 3 shards, got {shards}"
        # The already-completed task must NOT be redone.
        assert state.tasks[0].docs_processed == 1

    def test_reclaim_is_recorded_so_an_operator_can_see_it(self, tmp_path):
        crashed_state(tmp_path, "run-a", running_ids=[0], total=1, owner=dead_pid())
        ex = LocalExecutor(output_dir=tmp_path, num_workers=1)
        reclaimed = reclaim_via(ex, ExecutionState.load(tmp_path / "execution_state.json"))
        assert reclaimed == [0]

    def test_live_owner_task_is_not_stolen(self, tmp_path):
        """A running task owned by a LIVE pid must be left alone.

        pid 1 always exists and is never this test process, so it reads as a
        live owner. This is what keeps a concurrent healthy run safe.
        """
        crashed_state(tmp_path, "run-a", running_ids=[0], total=2, owner=1)
        st = ExecutionState.load(tmp_path / "execution_state.json")
        ex = LocalExecutor(output_dir=tmp_path, num_workers=1)
        reclaimed = reclaim_via(ex, st)
        assert reclaimed == [], "stole a task from a live owner"
        assert st.tasks[0].status == "running"


class TestParallelCrashResume:
    def test_all_running_state_is_not_treated_as_complete(self, tmp_path):
        """Pre-fix the worst case: every task running, so pending was empty.

        `execute` then returned at its early-return line having run zero tasks
        and written zero shards, while raising nothing.
        """
        crashed_state(tmp_path, "run-p", running_ids=[0, 1, 2], total=3, owner=dead_pid())

        ex = LocalExecutor(output_dir=tmp_path, num_workers=2)
        state = ex.execute(DataTroveAdapter(), [Shard(i) for i in range(3)], run_id="run-p")

        assert is_complete_of(state), f"unfinished after resume: {unfinished_of(state)}"
        shards = sorted(p.name for p in tmp_path.glob("shard_*.jsonl"))
        assert len(shards) == 3, f"expected 3 shards, got {shards}"

    def test_parallel_path_rejects_unpicklable_closure_shards(self, tmp_path):
        """Documents a separate real defect found while building the probe.

        `execute` accepts `list[Callable[[], Iterator[Document]]]`, but the
        parallel path pickles each shard, so a closure fails at runtime while
        satisfying the annotation. This test pins the CURRENT behavior so a
        future fix has to change it deliberately.
        """

        def closure_shard(task_id: int):
            def gen():
                yield Document(text="x", id=f"t{task_id}")

            return gen

        ex = LocalExecutor(output_dir=tmp_path, num_workers=2)
        state = ex.execute(DataTroveAdapter(), [closure_shard(i) for i in range(2)], run_id="run-c")
        assert all(t.status == "failed" for t in state.tasks.values())
        assert any("pickle" in (t.error or "") for t in state.tasks.values())
        assert not is_complete_of(state)


class TestTruncatedOutput:
    def test_stale_truncated_shard_is_removed_on_reclaim(self, tmp_path):
        """A truncated shard is valid JSONL, so it must be deleted not trusted."""
        crashed_state(tmp_path, "run-w", running_ids=[0], total=1, owner=dead_pid())
        # Simulate the crash-truncated shard: valid JSONL, wrong row count.
        stale = tmp_path / "shard_00000.jsonl"
        stale.write_text(json.dumps({"text": "partial", "id": "t0-0", "metadata": {}}) + "\n")
        for line in stale.read_text().splitlines():
            json.loads(line)  # proves the stale file parses cleanly

        ex = LocalExecutor(output_dir=tmp_path, num_workers=1)
        state = ex.execute(DataTroveAdapter(), [Shard(0, n_docs=5)], run_id="run-w")

        assert is_complete_of(state)
        rows = [json.loads(x) for x in stale.read_text().splitlines()]
        assert len(rows) == 5, "resume kept the truncated shard instead of rewriting it"

    def test_write_output_leaves_no_partial_file_at_final_path(self, tmp_path):
        ex = LocalExecutor(output_dir=tmp_path, num_workers=1)
        ex._write_output(7, [Document(text="a", id="1"), Document(text="b", id="2")])
        out = tmp_path / "shard_00007.jsonl"
        assert out.exists()
        assert not list(tmp_path.glob("*.tmp")), "temp file was left behind"
        assert len(out.read_text().splitlines()) == 2


class TestCompletionContract:
    def test_execute_returning_is_not_evidence_of_completion(self, tmp_path):
        """The caller-facing lesson. `is_complete` is the question to ask."""
        crashed_state(tmp_path, "run-x", running_ids=[0], total=2, owner=1)
        ex = LocalExecutor(output_dir=tmp_path, num_workers=1)
        state = ex.execute(DataTroveAdapter(), [Shard(i) for i in range(2)], run_id="run-x")
        # Task 0 is owned by live pid 1, so it is correctly not reclaimed and
        # the run is honestly incomplete. execute() still returned normally.
        assert not is_complete_of(state)
        assert unfinished_of(state) == [0]

    def test_a_live_owners_task_is_not_double_executed(self, tmp_path):
        """Regression guard for a bug the FIX itself introduced.

        First cut of the reclaim change spared live-owned tasks in
        `_reclaim_orphaned` but then re-included them in `pending` because they
        were "not completed", so two processes would have run the same task and
        raced on one shard path. Sparing a task must also mean not scheduling
        it.
        """
        crashed_state(tmp_path, "run-h", running_ids=[0], total=2, owner=1)
        ex = LocalExecutor(output_dir=tmp_path, num_workers=1)
        state = ex.execute(DataTroveAdapter(), [Shard(i) for i in range(2)], run_id="run-h")

        assert state.tasks[0].status == "running", "ran a task a live owner holds"
        assert not (tmp_path / "shard_00000.jsonl").exists()
        assert state.tasks[1].status == "completed", "healthy task did not run"
        assert not is_complete_of(state)
        assert unfinished_of(state) == [0]

    def test_state_save_is_atomic(self, tmp_path):
        state = ExecutionState(run_id="r", total_tasks=1)
        state.tasks[0] = TaskStatus(task_id=0)
        path = tmp_path / "execution_state.json"
        state.save(path)
        assert json.loads(path.read_text())["run_id"] == "r"
        assert not list(tmp_path.glob("*.tmp"))

    def test_owner_pid_round_trips_through_save_and_load(self, tmp_path):
        state = ExecutionState(run_id="r", total_tasks=1)
        state.tasks[0] = TaskStatus(task_id=0, status="running", owner_pid=4242)
        path = tmp_path / "execution_state.json"
        state.save(path)
        assert ExecutionState.load(path).tasks[0].owner_pid == 4242

    def test_reclaimed_task_carries_the_reclaim_marker(self, tmp_path):
        crashed_state(tmp_path, "run-m", running_ids=[0], total=1, owner=dead_pid())
        st = ExecutionState.load(tmp_path / "execution_state.json")
        reclaim_via(LocalExecutor(output_dir=tmp_path, num_workers=1), st)
        assert st.tasks[0].error == RECLAIM_ERROR


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
