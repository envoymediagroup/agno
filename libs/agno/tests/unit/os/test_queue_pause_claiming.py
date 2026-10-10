"""Pausing a queue worker's claiming without stopping the worker."""

import asyncio
from types import SimpleNamespace
from typing import Any, Callable, List

import pytest

from agno.db.schemas.jobs import QueuedJob
from agno.job_queue.config import QueueConfig
from agno.job_queue.store import InMemoryQueueStore
from agno.os.job_queue import QueueWorker
from agno.run.base import RunStatus


class GatedAgent:
    """Each run blocks until the test releases it."""

    id = "agent-1"

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.started: List[str] = []

    async def arun(self, **kwargs: Any) -> SimpleNamespace:
        self.started.append(kwargs["run_id"])
        await self.release.wait()
        return SimpleNamespace(status=RunStatus.completed, content="done")


class SlowClaimStore(InMemoryQueueStore):
    """claim_job blocks until the test lets it return."""

    def __init__(self) -> None:
        super().__init__()
        self.claim_started = asyncio.Event()
        self.release_claim = asyncio.Event()

    async def claim_job(self, *args: Any, **kwargs: Any) -> Any:
        self.claim_started.set()
        await self.release_claim.wait()
        return await super().claim_job(*args, **kwargs)


def make_job(job_id: str, max_attempts: int = 1) -> dict:
    return QueuedJob(
        id=job_id,
        component_type="agent",
        component_id="agent-1",
        session_id="s1",
        payload={"input": "hello", "kwargs": {}},
        max_attempts=max_attempts,
    ).to_dict()


def make_worker(store: InMemoryQueueStore, agent: GatedAgent) -> QueueWorker:
    return QueueWorker(
        store=store,
        resolve_component=lambda ctype, cid: agent if (ctype, cid) == ("agent", "agent-1") else None,
        config=QueueConfig(durable=True, poll_interval=0.02, lock_grace_seconds=60, timeout_seconds=None),
        worker_id="live-worker",
    )


async def wait_until(condition: Callable[[], Any], timeout: float = 3.0) -> None:
    async def poll() -> None:
        while not condition():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout=timeout)


async def job_status(store: InMemoryQueueStore, job_id: str) -> str:
    job = await store.get_job(job_id)
    assert job is not None
    return job["status"]


@pytest.mark.asyncio
async def test_paused_claiming_finishes_in_flight_runs_and_leaves_new_ones_queued():
    store, agent = InMemoryQueueStore(), GatedAgent()
    worker = make_worker(store, agent)
    await store.enqueue_job(make_job("r1"))
    await worker.start()
    try:
        await wait_until(lambda: agent.started == ["r1"])

        worker.pause_claiming()
        await store.enqueue_job(make_job("r2"))
        await asyncio.sleep(0.2)

        assert worker.claiming_paused
        assert agent.started == ["r1"]
        assert await job_status(store, "r2") == "queued"
        assert worker.in_flight_count == 1

        agent.release.set()
        await wait_until(lambda: worker.in_flight_count == 0)
        assert await job_status(store, "r1") == "completed"
        await asyncio.sleep(0.1)
        assert await job_status(store, "r2") == "queued"

        worker.resume_claiming()
        assert not worker.claiming_paused
        await wait_until(lambda: agent.started == ["r1", "r2"])
        await wait_until(lambda: worker.in_flight_count == 0)
        assert await job_status(store, "r2") == "completed"
    finally:
        agent.release.set()
        await worker.stop()


@pytest.mark.asyncio
async def test_a_claim_under_way_when_claiming_pauses_counts_as_in_flight():
    store, agent = SlowClaimStore(), GatedAgent()
    worker = make_worker(store, agent)
    await store.enqueue_job(make_job("r1"))
    await worker.start()
    try:
        await asyncio.wait_for(store.claim_started.wait(), timeout=3.0)

        worker.pause_claiming()
        assert worker.in_flight_count == 1

        store.release_claim.set()
        await wait_until(lambda: agent.started == ["r1"])
        assert worker.in_flight_count == 1

        agent.release.set()
        await wait_until(lambda: worker.in_flight_count == 0)
        assert await job_status(store, "r1") == "completed"
    finally:
        store.release_claim.set()
        agent.release.set()
        await worker.stop()


@pytest.mark.asyncio
async def test_paused_worker_still_sweeps_runs_whose_worker_died(monkeypatch: pytest.MonkeyPatch):
    async def persisted(*args: Any, **kwargs: Any) -> Any:
        from agno.run.status_persist import RunPersistOutcome

        return RunPersistOutcome.UPDATED

    store, agent = InMemoryQueueStore(), GatedAgent()
    worker = make_worker(store, agent)
    monkeypatch.setattr(worker, "_persist_run_error_outcome", persisted)
    await store.enqueue_job(make_job("dead"))
    assert await store.claim_job("dead-worker") is not None
    store._jobs["dead"]["locked_at"] -= 1000

    worker.pause_claiming()
    await worker.start()
    try:
        await wait_until(lambda: store._jobs["dead"]["status"] == "failed")
        assert agent.started == []
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_a_wake_while_claiming_is_paused_claims_nothing():
    store, agent = InMemoryQueueStore(), GatedAgent()
    worker = QueueWorker(
        store=store,
        resolve_component=lambda ctype, cid: agent if (ctype, cid) == ("agent", "agent-1") else None,
        # A poll tick far beyond the test: only a wake could claim the run
        config=QueueConfig(durable=True, poll_interval=60, lock_grace_seconds=60, timeout_seconds=None),
        worker_id="live-worker",
    )
    await worker.start()
    try:
        await asyncio.sleep(0.05)
        worker.pause_claiming()
        await store.enqueue_job(make_job("r1"))

        worker.wake()
        await asyncio.sleep(0.2)

        assert agent.started == []
        assert await job_status(store, "r1") == "queued"

        worker.resume_claiming()
        worker.wake()
        await wait_until(lambda: agent.started == ["r1"])
    finally:
        agent.release.set()
        await worker.stop()
