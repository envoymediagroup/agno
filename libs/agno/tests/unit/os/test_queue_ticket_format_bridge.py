"""Workers of the fork's v3.1.2 and v3.1.12 builds run each other's stored-agent tickets.

v3.1.2's fork recorded a stored agent's version as ``payload.component_version``;
v3.1.12 resolves a stored agent only from ``payload.scope`` with ``source: "db"``.
Both versions share one queue table while a deploy drains, so a worker reads an
old ticket as a db-backed one, and a new stored-agent ticket also carries the old key.
"""

import asyncio
import time
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

from agno.agent import Agent
from agno.db.base import ComponentType
from agno.db.schemas.jobs import QueuedJob
from agno.db.sqlite import SqliteDb
from agno.job_queue.config import QueueConfig
from agno.job_queue.store import InMemoryQueueStore
from agno.os import AgentOS
from agno.os.job_queue import QueueWorker
from agno.registry import Registry
from agno.run.base import RunStatus


class RecordingAgent:
    id = "agent-1"

    def __init__(self) -> None:
        self.ran: List[str] = []

    async def arun(self, **kwargs: Any) -> SimpleNamespace:
        self.ran.append(kwargs["run_id"])
        return SimpleNamespace(status=RunStatus.completed, content="done")


def ticket(job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return QueuedJob(
        id=job_id,
        component_type="agent",
        component_id="agent-1",
        session_id="s1",
        user_id="someone",
        payload={"input": "hello", "kwargs": {}, **payload},
    ).to_dict()


async def wait_until(condition: Callable[[], Any], timeout: float = 3.0) -> None:
    async def poll() -> None:
        while not condition():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout=timeout)


async def scope_the_worker_resolves(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The scope a worker hands its resolver when it runs a ticket with ``payload``."""
    agent = RecordingAgent()
    scopes: List[Optional[Dict[str, Any]]] = []

    def resolve(component_type: str, component_id: str, scope: Optional[Dict[str, Any]] = None) -> Any:
        scopes.append(scope)
        return agent

    store = InMemoryQueueStore()
    worker = QueueWorker(
        store=store,
        resolve_component=resolve,
        config=QueueConfig(durable=True, poll_interval=0.02, lock_grace_seconds=60, timeout_seconds=None),
        worker_id="worker",
    )
    await store.enqueue_job(ticket("r1", payload))
    await worker.start()
    try:
        await wait_until(lambda: agent.ran == ["r1"])
    finally:
        await worker.stop()
    assert scopes
    assert all(scope == scopes[0] for scope in scopes)
    return scopes[0]


@pytest.mark.asyncio
async def test_an_old_format_ticket_resolves_as_a_db_backed_agent_at_its_version_with_no_user_scope():
    scope = await scope_the_worker_resolves({"component_version": 3})

    assert scope == {"source": "db", "version": 3, "user_id": None, "pinned": None}


@pytest.mark.asyncio
async def test_a_registry_ticket_without_a_scope_is_left_alone():
    assert await scope_the_worker_resolves({}) is None


@pytest.mark.asyncio
async def test_a_ticket_with_a_scope_is_left_alone():
    scope = {"source": "db", "version": 4, "user_id": "owner", "pinned": 4}

    assert await scope_the_worker_resolves({"scope": scope, "component_version": 4}) == scope


def save_stored_agent(db: SqliteDb, instructions: str) -> None:
    config = {"name": "stored-agent", "instructions": instructions}
    if db.get_component(component_id="stored-agent") is None:
        db.create_component_with_config(
            component_id="stored-agent",
            component_type=ComponentType.AGENT,
            name="stored-agent",
            config=config,
            stage="published",
        )
    else:
        db.upsert_config("stored-agent", config=config, stage="published")


@pytest.fixture()
def accepting_os(tmp_path) -> SimpleNamespace:
    """AgentOS with a stored agent at version 2 and a code-defined one; the worker never runs."""
    db = SqliteDb(db_file=str(tmp_path / "bridge.db"))
    save_stored_agent(db, "v1")
    save_stored_agent(db, "v2")
    code_agent = Agent(id="code-agent", name="Code Agent", db=db)
    app = AgentOS(agents=[code_agent], db=db, registry=Registry(name="r", dbs=[db]), telemetry=False).get_app()
    store = InMemoryQueueStore()
    app.state.queue_worker = SimpleNamespace(store=store, config=QueueConfig(durable=True))
    return SimpleNamespace(client=TestClient(app, raise_server_exceptions=False), store=store)


def submit(client: TestClient, agent_id: str) -> Dict[str, Any]:
    data = {"message": "hi", "stream": "false", "background": "true", "run_id": f"{agent_id}-run"}
    response = client.post(f"/agents/{agent_id}/runs", data=data)
    assert response.status_code in (200, 202), response.text
    return data


def test_a_new_stored_agent_ticket_also_carries_its_version_in_the_old_format(accepting_os):
    data = submit(accepting_os.client, "stored-agent")

    payload = accepting_os.store._jobs[data["run_id"]]["payload"]
    assert payload["scope"]["source"] == "db"
    assert payload["scope"]["version"] == 2
    assert payload["component_version"] == 2


def test_a_code_defined_agent_ticket_carries_no_old_format_version(accepting_os):
    data = submit(accepting_os.client, "code-agent")

    assert "component_version" not in accepting_os.store._jobs[data["run_id"]]["payload"]


def test_an_old_format_ticket_runs_the_stored_agent_at_its_version(tmp_path, monkeypatch: pytest.MonkeyPatch):
    from agno.run.agent import RunOutput

    executed: List[Any] = []

    async def fake_arun(self: Agent, input: Any = None, **kwargs: Any) -> RunOutput:
        executed.append(self.instructions)
        return RunOutput(
            run_id=kwargs.get("run_id"),
            session_id=kwargs.get("session_id"),
            agent_id=self.id,
            status=RunStatus.completed,
            content="done",
        )

    monkeypatch.setattr(Agent, "arun", fake_arun)
    db = SqliteDb(db_file=str(tmp_path / "old.db"))
    save_stored_agent(db, "v1")
    save_stored_agent(db, "v2")
    store = InMemoryQueueStore()
    queue = QueueConfig(durable=True, db=store, poll_interval=0.02)
    app = AgentOS(db=db, registry=Registry(name="r", dbs=[db]), queue=queue, telemetry=False).get_app()
    old_ticket = QueuedJob(
        id="old-run",
        component_type="agent",
        component_id="stored-agent",
        session_id="s1",
        user_id="someone",
        payload={"input": "hello", "kwargs": {}, "component_version": 1},
    ).to_dict()

    with TestClient(app):
        asyncio.run(store.enqueue_job(old_ticket))
        for _ in range(300):
            if store._jobs["old-run"]["status"] == "completed":
                break
            time.sleep(0.01)

    assert store._jobs["old-run"]["status"] == "completed"
    assert executed == ["v1"]
