from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest
from fakeredis import FakeAsyncRedis
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic_ai import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from agentdesk.api import create_app
from agentdesk.config import ROOT, Settings
from agentdesk.knowledge import KnowledgeBase

Step = tuple[str, dict[str, Any]]


class ScriptedModel:
    """A fake LLM that replays a fixed list of steps, one per model call.

    ("search_help_center", {...}) calls a tool; ("final", {...}) returns the structured reply.
    It also records the message history it was given, so tests can check memory.
    """

    def __init__(self, steps: list[Step]) -> None:
        self.steps = list(steps)
        self.seen: list[list[ModelMessage]] = []
        self.model = FunctionModel(self._respond)

    def _respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.seen.append(list(messages))
        if not self.steps:
            raise AssertionError("ScriptedModel ran out of steps")
        name, args = self.steps.pop(0)
        if name == "final":
            name = info.output_tools[0].name
        return ModelResponse(parts=[ToolCallPart(name, args)])


@pytest.fixture
def kb() -> KnowledgeBase:
    return KnowledgeBase.from_dir(ROOT / "data" / "kb")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
        rate_limit_per_minute=5,
        history_max_turns=3,
    )


@pytest.fixture
def redis() -> FakeAsyncRedis:
    return FakeAsyncRedis()


@pytest.fixture
def make_client(settings: Settings, redis: FakeAsyncRedis) -> Callable[[ScriptedModel], Any]:
    """Return an async context manager factory: `async with make_client(model) as client:`."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _make(scripted: ScriptedModel) -> AsyncIterator[AsyncClient]:
        app: FastAPI = create_app(settings, redis=redis, model=scripted.model)
        async with app.router.lifespan_context(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                yield client

    return _make
