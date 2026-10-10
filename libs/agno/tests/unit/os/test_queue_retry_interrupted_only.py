"""QueueConfig(retry_interrupted_only=True): only interrupted runs are re-run."""

import asyncio
from types import SimpleNamespace
from typing import Any, AsyncIterator, Iterator, List, Union

import pytest

from agno.db.schemas.jobs import QueuedJob
from agno.job_queue.config import QueueConfig
from agno.job_queue.store import InMemoryQueueStore
from agno.os.event_streams import InMemoryEventStream
from agno.os.job_queue import QueueWorker
from agno.run.base import RunStatus

HANG = "hang"
Outcome = Union[RunStatus, BaseException, str]


@pytest.fixture(autouse=True)
def _stub_run_row_persist(monkeypatch: pytest.MonkeyPatch) -> None:
    from agno.session import AgentSession

    async def fake_read(component, session_id=None, user_id=None):
        return AgentSession(session_id=session_id or "s1", runs=[])

    async def fake_save(component, session=None):
        pass

    async def fake_save_run(component, run=None, session_id=None, user_id=None, run_index=None):
        pass

    monkeypatch.setattr("agno.agent._storage.aread_or_create_session", fake_read)
    monkeypatch.setattr("agno.agent._session.asave_session", fake_save)
    monkeypatch.setattr("agno.agent._session.asave_run", fake_save_run)


@pytest.fixture()
def event_stream() -> Iterator[InMemoryEventStream]:
    import agno.os.event_streams as event_streams_module
    from agno.os.event_streams import set_event_stream
    from agno.os.managers import EventsBuffer, SSESubscriberManager

    original = event_streams_module._event_stream
    original_explicit = getattr(event_streams_module, "_event_stream_explicitly_set", False)
    stream = InMemoryEventStream(events_buffer=EventsBuffer(), subscriber_manager=SSESubscriberManager())
    set_event_stream(stream)
    try:
        yield stream
    finally:
        event_streams_module._event_stream = original
        event_streams_module._event_stream_explicitly_set = original_explicit


class ScriptedAgent:
    """Plays one outcome per attempt: a status to return, an error to raise, or HANG until cancelled."""

    id = "agent-1"
    db = None

    def __init__(self, *outcomes: Outcome) -> None:
        self.outcomes = list(outcomes)
        self.calls: List[dict] = []

    def _next(self, kwargs: dict) -> Outcome:
        self.calls.append(kwargs)
        return self.outcomes[min(len(self.calls), len(self.outcomes)) - 1]

    def arun(self, **kwargs: Any) -> Any:
        outcome = self._next(kwargs)
        if kwargs.get("stream"):
            return self._stream(outcome, kwargs["run_id"])
        return self._result(outcome)

    async def _play(self, outcome: Outcome) -> RunStatus:
        if outcome == HANG:
            await asyncio.sleep(3600)
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, RunStatus)
        return outcome

    async def _result(self, outcome: Outcome) -> SimpleNamespace:
        status = await self._play(outcome)
        return SimpleNamespace(status=status, content=status.value)

    async def _stream(self, outcome: Outcome, run_id: str) -> AsyncIterator[SimpleNamespace]:
        status = await self._play(outcome)
        yield SimpleNamespace(run_id=run_id, status=status)


def make_config(**overrides: Any) -> QueueConfig:
    defaults: dict = dict(
        durable=True,
        poll_interval=0.02,
        lock_grace_seconds=60,
        timeout_seconds=None,
        retry_delay_seconds=0,
        retry_interrupted_only=True,
    )
    defaults.update(overrides)
    return QueueConfig(**defaults)


def make_job(job_id: str = "r1", max_attempts: int = 3, stream: bool = False) -> dict:
    payload: dict = {"input": "hello", "kwargs": {}}
    if stream:
        payload["stream"] = True
    return QueuedJob(
        id=job_id,
        component_type="agent",
        component_id="agent-1",
        session_id="s1",
        payload=payload,
        max_attempts=max_attempts,
    ).to_dict()


def make_worker(
    store: InMemoryQueueStore, agent: ScriptedAgent, config: QueueConfig, worker_id: str = "live-worker"
) -> QueueWorker:
    return QueueWorker(
        store=store,
        resolve_component=lambda ctype, cid: agent if (ctype, cid) == ("agent", "agent-1") else None,
        config=config,
        worker_id=worker_id,
    )


async def wait_for_status(store: InMemoryQueueStore, job_id: str, status: str, timeout: float = 3.0) -> dict:
    async def poll() -> dict:
        while True:
            job = await store.get_job(job_id)
            if job is not None and job["status"] == status:
                return job
            await asyncio.sleep(0.01)

    return await asyncio.wait_for(poll(), timeout=timeout)


async def wait_for_calls(agent: ScriptedAgent, count: int, timeout: float = 3.0) -> None:
    async def poll() -> None:
        while len(agent.calls) < count:
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout=timeout)


async def drain_mid_run(store: InMemoryQueueStore, agent: ScriptedAgent, config: QueueConfig) -> None:
    stopping = make_worker(store, agent, config, worker_id="stopping-worker")
    stopping.stop_timeout = 0
    await stopping.start()
    await wait_for_calls(agent, 1)
    await stopping.stop()


@pytest.mark.asyncio
async def test_a_run_whose_worker_died_reruns_with_a_higher_attempt():
    store, agent = InMemoryQueueStore(), ScriptedAgent(RunStatus.completed)
    await store.enqueue_job(make_job())
    assert await store.claim_job("dead-worker") is not None
    store._jobs["r1"]["locked_at"] -= 1000

    worker = make_worker(store, agent, make_config())
    await worker.start()
    try:
        job = await wait_for_status(store, "r1", "completed")
        assert job["attempt"] == 2
        assert len(agent.calls) == 1
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_a_run_interrupted_by_a_shutdown_reruns_on_the_next_worker():
    store, agent = InMemoryQueueStore(), ScriptedAgent(HANG, RunStatus.completed)
    config = make_config()
    await store.enqueue_job(make_job())

    await drain_mid_run(store, agent, config)
    assert (await store.get_job("r1"))["status"] == "queued"

    worker = make_worker(store, agent, config)
    await worker.start()
    try:
        job = await wait_for_status(store, "r1", "completed")
        assert job["attempt"] == 2
        assert len(agent.calls) == 2
    finally:
        await worker.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure, error_text",
    [(RunStatus.error, "ERROR"), (RuntimeError("model exploded"), "model exploded")],
    ids=["error-result", "raised-error"],
)
async def test_a_run_that_fails_on_its_own_is_not_rerun(failure: Outcome, error_text: str):
    store, agent = InMemoryQueueStore(), ScriptedAgent(failure, RunStatus.completed)
    worker = make_worker(store, agent, make_config())
    await store.enqueue_job(make_job())
    await worker.start()
    try:
        job = await wait_for_status(store, "r1", "failed")
        assert job["attempt"] == 1
        assert error_text in job["error"]
        await asyncio.sleep(0.1)
        assert len(agent.calls) == 1
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_a_run_that_times_out_is_not_rerun():
    store, agent = InMemoryQueueStore(), ScriptedAgent(HANG, RunStatus.completed)
    worker = make_worker(store, agent, make_config(timeout_seconds=1))
    worker.config.timeout_seconds = 0.05  # type: ignore[assignment]
    await store.enqueue_job(make_job())
    await worker.start()
    try:
        job = await wait_for_status(store, "r1", "failed")
        assert job["attempt"] == 1
        assert "timeout" in job["error"].lower()
        await asyncio.sleep(0.1)
        assert len(agent.calls) == 1
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_without_the_setting_a_run_that_errors_is_retried():
    store, agent = InMemoryQueueStore(), ScriptedAgent(RunStatus.error, RunStatus.completed)
    worker = make_worker(store, agent, make_config(retry_interrupted_only=False))
    await store.enqueue_job(make_job())
    await worker.start()
    try:
        job = await wait_for_status(store, "r1", "completed")
        assert job["attempt"] == 2
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_a_streaming_run_that_errors_closes_its_stream(event_stream: InMemoryEventStream):
    store, agent = InMemoryQueueStore(), ScriptedAgent(RunStatus.error, RunStatus.completed)
    worker = make_worker(store, agent, make_config())
    await store.enqueue_job(make_job(stream=True))
    await worker.start()
    try:
        await wait_for_status(store, "r1", "failed")
        assert await event_stream.get_run_status("r1") == RunStatus.error
        assert len(agent.calls) == 1
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_a_streaming_run_interrupted_by_a_shutdown_keeps_its_stream_open_for_the_rerun(
    event_stream: InMemoryEventStream,
):
    store, agent = InMemoryQueueStore(), ScriptedAgent(HANG, RunStatus.completed)
    config = make_config()
    await store.enqueue_job(make_job(stream=True))

    await drain_mid_run(store, agent, config)
    assert await event_stream.get_run_status("r1") == RunStatus.running

    worker = make_worker(store, agent, config)
    await worker.start()
    try:
        await wait_for_status(store, "r1", "completed")
        assert await event_stream.get_run_status("r1") == RunStatus.completed
    finally:
        await worker.stop()
