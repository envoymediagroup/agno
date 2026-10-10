"""A stored agent's background runs ride the durable queue, at the version found at submit."""

import time
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List

import pytest
from fastapi.testclient import TestClient

from agno.agent import Agent
from agno.db.base import ComponentType
from agno.db.sqlite import SqliteDb
from agno.job_queue.config import QueueConfig
from agno.job_queue.store import InMemoryQueueStore
from agno.os import AgentOS
from agno.registry import Registry
from agno.run.agent import RunOutput
from agno.run.base import RunStatus

AGENT_ID = "stored-agent"


def save_version(db: SqliteDb, instructions: str, stage: str = "published") -> None:
    config = {"name": AGENT_ID, "instructions": instructions}
    if db.get_component(component_id=AGENT_ID) is None:
        db.create_component_with_config(
            component_id=AGENT_ID, component_type=ComponentType.AGENT, name=AGENT_ID, config=config, stage=stage
        )
    else:
        db.upsert_config(AGENT_ID, config=config, stage=stage)


class PublishingDb(SqliteDb):
    """A db where v2 can be published the moment a reader holds v1's config, as a concurrent save would."""

    publish_v2_on_reading_v1 = False

    def get_config(self, *args: Any, **kwargs: Any) -> Any:
        row = super().get_config(*args, **kwargs)
        if self.publish_v2_on_reading_v1 and isinstance(row, dict) and row.get("version") == 1:
            self.publish_v2_on_reading_v1 = False
            save_version(self, "v2")
        return row


class StoredAgentRuns:
    """Stands in for the model: records the instructions every run and continue executed with.

    ``script`` holds one outcome per run: "complete", "pause", or "publish-v2-then-fail".
    """

    def __init__(self, db: SqliteDb) -> None:
        self.db = db
        self.executed: List[str] = []
        self.script: List[str] = []

    def arun(self, agent: Agent, stream: bool = False, **kwargs: Any) -> Any:
        self.executed.append(f"run:{agent.instructions}")
        outcome = self.script.pop(0) if self.script else "complete"
        return self._deliver(self._play(agent, outcome, kwargs), stream)

    def acontinue_run(self, agent: Agent, stream: bool = False, **kwargs: Any) -> Any:
        self.executed.append(f"continue:{agent.instructions}")
        return self._deliver(self._play(agent, "complete", kwargs), stream)

    async def _play(self, agent: Agent, outcome: str, kwargs: Dict[str, Any]) -> RunOutput:
        if outcome == "publish-v2-then-fail":
            save_version(self.db, "v2")
            raise RuntimeError("transient model error")
        output = RunOutput(
            run_id=kwargs["run_id"],
            session_id=kwargs["session_id"],
            agent_id=agent.id,
            status=RunStatus.paused if outcome == "pause" else RunStatus.completed,
            content=outcome,
        )
        if outcome == "pause":
            from agno.agent._session import asave_run

            await asave_run(agent, run=output, session_id=kwargs["session_id"], user_id=kwargs.get("user_id"))
        return output

    @staticmethod
    def _deliver(result: Any, stream: bool) -> Any:
        if not stream:
            return result

        async def events() -> Any:
            yield await result

        return events()


@pytest.fixture()
def harness(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    import agno.os.event_streams as event_streams_module
    from agno.os.event_streams import InMemoryEventStream, set_event_stream
    from agno.os.managers import EventsBuffer, SSESubscriberManager

    db = PublishingDb(db_file=str(tmp_path / "stored.db"))
    save_version(db, "v1")
    runs = StoredAgentRuns(db)
    monkeypatch.setattr(Agent, "arun", lambda self, input=None, **kwargs: runs.arun(self, **kwargs))
    monkeypatch.setattr(
        Agent, "acontinue_run", lambda self, run_response=None, **kwargs: runs.acontinue_run(self, **kwargs)
    )
    original_stream = event_streams_module._event_stream
    original_explicit = getattr(event_streams_module, "_event_stream_explicitly_set", False)
    set_event_stream(InMemoryEventStream(events_buffer=EventsBuffer(), subscriber_manager=SSESubscriberManager()))
    store = InMemoryQueueStore()
    queue = QueueConfig(durable=True, db=store, poll_interval=0.02, max_attempts=2, retry_delay_seconds=0)
    app = AgentOS(db=db, registry=Registry(name="r", dbs=[db]), queue=queue, telemetry=False).get_app()
    try:
        with TestClient(app) as client:
            yield SimpleNamespace(client=client, store=store, db=db, runs=runs)
    finally:
        event_streams_module._event_stream = original_stream
        event_streams_module._event_stream_explicitly_set = original_explicit


def submit(client: TestClient, stream: bool = False, **fields: str) -> Any:
    data = {"message": "hi", "stream": str(stream).lower(), "background": "true", **fields}
    return client.post(f"/agents/{AGENT_ID}/runs", data=data)


def wait_for_ticket(store: InMemoryQueueStore, status: str, run_id: str = "", timeout: float = 5.0) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        jobs = [job for job in list(store._jobs.values()) if not run_id or job["id"] == run_id]
        if len(jobs) == 1 and jobs[0]["status"] == status:
            return jobs[0]
        time.sleep(0.02)
    raise AssertionError(f"no single ticket reached {status!r}: {list(store._jobs.values())}")


def test_a_background_run_is_executed_by_the_queue_worker(harness):
    response = submit(harness.client)

    assert response.status_code == 202
    job = wait_for_ticket(harness.store, "completed", response.json()["run_id"])
    assert (job["component_type"], job["component_id"]) == ("agent", AGENT_ID)
    assert harness.runs.executed == ["run:v1"]


@pytest.mark.parametrize("stream", [False, True], ids=["non-stream", "stream"])
def test_a_rerun_executes_the_version_found_at_submit(harness, stream):
    harness.runs.script = ["publish-v2-then-fail", "complete"]

    response = submit(harness.client, stream=stream)

    assert response.status_code == (200 if stream else 202)
    job = wait_for_ticket(harness.store, "completed")
    assert job["attempt"] == 2
    assert harness.runs.executed == ["run:v1", "run:v1"]

    later = submit(harness.client)
    wait_for_ticket(harness.store, "completed", later.json()["run_id"])
    assert harness.runs.executed[-1] == "run:v2"


def test_a_version_published_while_the_submit_resolves_does_not_take_over_the_run(harness):
    harness.db.publish_v2_on_reading_v1 = True

    response = submit(harness.client)

    assert response.status_code == 202
    wait_for_ticket(harness.store, "completed", response.json()["run_id"])
    assert harness.runs.executed == ["run:v1"]


def test_a_pinned_draft_version_runs_on_the_queue(harness):
    save_version(harness.db, "draft v2", stage="draft")

    response = submit(harness.client, version="2")

    assert response.status_code == 202
    wait_for_ticket(harness.store, "completed", response.json()["run_id"])
    assert harness.runs.executed == ["run:draft v2"]


def test_a_paused_run_continues_through_the_queue_at_its_submit_version(harness):
    harness.runs.script = ["pause"]
    submitted = submit(harness.client).json()
    wait_for_ticket(harness.store, "paused", submitted["run_id"])
    save_version(harness.db, "v2")

    response = harness.client.post(
        f"/agents/{AGENT_ID}/runs/{submitted['run_id']}/continue",
        data={"session_id": submitted["session_id"], "background": "true", "stream": "false"},
    )

    assert response.status_code == 202
    wait_for_ticket(harness.store, "completed", submitted["run_id"])
    assert harness.runs.executed == ["run:v1", "continue:v1"]
