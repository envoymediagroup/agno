"""A durable submission keeps the run id its caller supplies.

The inline paths already run under a caller's run_id form field; the queue
used to mint its own. Now the ticket takes the caller's id, so the worker runs
under it, and a repeated id is answered like a repeated Idempotency-Key.
"""

from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional

import pytest
from fastapi.testclient import TestClient

from agno.agent import Agent
from agno.db.schemas.jobs import QueuedJob
from agno.db.sqlite import SqliteDb
from agno.job_queue.config import QueueConfig
from agno.job_queue.store import InMemoryQueueStore
from agno.os import AgentOS
from agno.run.agent import RunOutput
from agno.run.base import RunStatus
from agno.team import Team
from agno.workflow import Workflow


@pytest.fixture()
def harness(tmp_path):
    db = SqliteDb(db_file=str(tmp_path / "t.db"))
    agent = Agent(id="qa-agent", name="QA Agent", db=db)
    team = Team(id="qa-team", name="QA Team", members=[agent], db=db)
    workflow = Workflow(id="qa-workflow", name="QA Workflow", db=db)
    app = AgentOS(agents=[agent], teams=[team], workflows=[workflow], telemetry=False).get_app()
    store = InMemoryQueueStore()
    app.state.queue_worker = SimpleNamespace(store=store, config=QueueConfig(durable=True))
    client = TestClient(app, raise_server_exceptions=False)
    return SimpleNamespace(store=store, client=client, db=db)


def submit(client: TestClient, path: str, run_id: str, stream: bool = False, **fields: str) -> Any:
    data = {"message": "hi", "stream": str(stream).lower(), "background": "true", "run_id": run_id, **fields}
    return client.post(path, data=data)


class TestNonStreamingSubmission:
    def test_a_caller_supplied_run_id_names_the_queued_run(self, harness):
        response = submit(harness.client, "/agents/qa-agent/runs", "run-1")

        assert response.status_code == 202
        assert response.json()["run_id"] == "run-1"
        assert list(harness.store._jobs) == ["run-1"]

    def test_a_repeated_run_id_returns_the_original_run(self, harness):
        first = submit(harness.client, "/agents/qa-agent/runs", "run-1", message="first")
        repeat = submit(harness.client, "/agents/qa-agent/runs", "run-1", message="again")

        assert repeat.status_code == 202
        assert repeat.json() == {"run_id": "run-1", "session_id": first.json()["session_id"], "status": "PENDING"}
        assert list(harness.store._jobs) == ["run-1"]
        assert harness.store._jobs["run-1"]["payload"]["input"] == "first"

    def test_a_repeat_reports_how_far_the_original_run_got(self, harness):
        submit(harness.client, "/agents/qa-agent/runs", "run-1")
        harness.store._jobs["run-1"]["status"] = "completed"

        repeat = submit(harness.client, "/agents/qa-agent/runs", "run-1")

        assert repeat.status_code == 202
        assert repeat.json()["status"] == "COMPLETED"

    def test_a_run_id_another_user_submitted_is_refused(self, harness):
        submit(harness.client, "/agents/qa-agent/runs", "run-1", user_id="alice")

        response = submit(harness.client, "/agents/qa-agent/runs", "run-1", user_id="bob")

        assert response.status_code == 409
        assert "alice" not in response.json()["detail"]
        assert harness.store._jobs["run-1"]["user_id"] == "alice"

    def test_a_run_id_another_component_used_is_refused(self, harness):
        submit(harness.client, "/agents/qa-agent/runs", "run-1")

        response = submit(harness.client, "/teams/qa-team/runs", "run-1")

        assert response.status_code == 409
        assert "different component" in response.json()["detail"]
        assert harness.store._jobs["run-1"]["component_type"] == "agent"

    def test_a_run_id_a_run_outside_the_queue_holds_is_refused(self, harness):
        from agno.session import AgentSession

        harness.db.upsert_session(AgentSession(session_id="inline-session", agent_id="qa-agent", user_id="alice"))
        inline_run = RunOutput(
            run_id="run-1", session_id="inline-session", agent_id="qa-agent", status=RunStatus.completed
        )
        harness.db.upsert_run(inline_run, session_id="inline-session", user_id="alice", run_index=0)

        response = submit(harness.client, "/agents/qa-agent/runs", "run-1", user_id="bob")

        assert response.status_code == 409
        assert "alice" not in response.json()["detail"]
        assert harness.store._jobs == {}

    def test_an_overlong_run_id_is_refused(self, harness):
        response = submit(harness.client, "/agents/qa-agent/runs", "r" * 513)

        assert response.status_code == 422
        assert harness.store._jobs == {}

    @pytest.mark.parametrize("path", ["/teams/qa-team/runs", "/workflows/qa-workflow/runs"])
    def test_teams_and_workflows_keep_the_caller_run_id_too(self, harness, path):
        first = submit(harness.client, path, "run-1")
        repeat = submit(harness.client, path, "run-1")

        assert first.status_code == 202
        assert first.json()["run_id"] == "run-1"
        assert repeat.json()["run_id"] == "run-1"
        assert list(harness.store._jobs) == ["run-1"]


class RacingStore(InMemoryQueueStore):
    """Another submission of the same run id commits between this one's lookup and its insert."""

    def __init__(self, winner: Dict[str, Any]) -> None:
        super().__init__()
        self.winner: Optional[Dict[str, Any]] = winner

    async def enqueue_job(self, job: Dict[str, Any], max_depth: int = 0) -> Dict[str, Any]:
        if self.winner is not None:
            winner, self.winner = self.winner, None
            await super().enqueue_job(winner)
        return await super().enqueue_job(job, max_depth)


def test_a_concurrent_submission_of_the_same_run_id_gets_the_winning_run(harness):
    winner = QueuedJob(
        id="run-1",
        component_type="agent",
        component_id="qa-agent",
        session_id="winner-session",
        payload={"input": "winner", "kwargs": {}},
    ).to_dict()
    store = RacingStore(winner)
    harness.client.app.state.queue_worker.store = store

    response = submit(harness.client, "/agents/qa-agent/runs", "run-1")

    assert response.status_code == 202
    assert response.json()["session_id"] == "winner-session"
    assert store._jobs["run-1"]["payload"]["input"] == "winner"


class TestStreamingSubmission:
    def test_a_streaming_submission_runs_under_the_caller_run_id(self, live):
        response = submit(live.client, "/agents/qa-agent/runs", "run-s1", stream=True)

        assert response.status_code == 200
        assert live.calls == [{"run_id": "run-s1", "stream": True}]
        assert live.store._jobs["run-s1"]["status"] == "completed"

    def test_a_repeated_run_id_does_not_attach_a_stream_to_a_non_streaming_run(self, live):
        submit(live.client, "/agents/qa-agent/runs", "run-1")

        response = submit(live.client, "/agents/qa-agent/runs", "run-1", stream=True)

        assert response.status_code == 409
        assert "poll run run-1" in response.json()["detail"]
        assert list(live.store._jobs) == ["run-1"]


@pytest.fixture()
def live(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    """AgentOS with its queue worker running, and agent runs that complete at once."""
    import agno.os.event_streams as event_streams_module
    from agno.os.event_streams import InMemoryEventStream, set_event_stream
    from agno.os.managers import EventsBuffer, SSESubscriberManager

    calls: List[Dict[str, Any]] = []

    def fake_arun(self: Agent, input: Any = None, *, stream: bool = False, **kwargs: Any) -> Any:
        calls.append({"run_id": kwargs.get("run_id"), "stream": stream})
        output = RunOutput(
            run_id=kwargs.get("run_id"),
            session_id=kwargs.get("session_id"),
            agent_id=self.id,
            status=RunStatus.completed,
            content="done",
        )

        async def events() -> Any:
            yield output

        async def result() -> RunOutput:
            return output

        return events() if stream else result()

    original_stream = event_streams_module._event_stream
    original_explicit = getattr(event_streams_module, "_event_stream_explicitly_set", False)
    set_event_stream(InMemoryEventStream(events_buffer=EventsBuffer(), subscriber_manager=SSESubscriberManager()))
    monkeypatch.setattr(Agent, "arun", fake_arun)
    store = InMemoryQueueStore()
    agent = Agent(id="qa-agent", name="QA Agent", db=SqliteDb(db_file=str(tmp_path / "live.db")))
    queue = QueueConfig(durable=True, db=store, poll_interval=0.02)
    app = AgentOS(agents=[agent], queue=queue, telemetry=False).get_app()
    try:
        with TestClient(app) as client:
            yield SimpleNamespace(client=client, store=store, calls=calls)
    finally:
        event_streams_module._event_stream = original_stream
        event_streams_module._event_stream_explicitly_set = original_explicit
