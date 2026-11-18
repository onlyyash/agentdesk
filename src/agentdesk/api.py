"""FastAPI app: chat endpoint plus ticket endpoints."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import AgentRunError, ModelAPIError, RunTimedOut, ToolCallPart, UserError
from pydantic_ai.models import Model
from redis.asyncio import Redis
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agentdesk.agent import Deps, SupportReply, support_agent
from agentdesk.config import Settings, get_settings
from agentdesk.db import Ticket, init_db, make_engine, make_sessionmaker, seed_demo_data
from agentdesk.guardrails import GuardrailViolation, check_user_message
from agentdesk.knowledge import KnowledgeBase
from agentdesk.memory import ConversationStore
from agentdesk.ratelimit import RateLimiter


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: SupportReply
    tools_used: list[str]


class TicketOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    category: str
    summary: str
    order_id: str | None
    status: str


def create_app(
    settings: Settings | None = None,
    *,
    redis: Redis | None = None,
    model: Model | str | None = None,
) -> FastAPI:
    """Build the app. Tests pass a fake Redis and a fake model; production uses settings."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = make_engine(settings.database_url)
        await init_db(engine)
        sessions = make_sessionmaker(engine)
        if settings.seed_demo_data:
            await seed_demo_data(sessions)
        r = redis or Redis.from_url(settings.redis_url)
        app.state.sessions = sessions
        app.state.redis = r
        app.state.kb = KnowledgeBase.from_dir(settings.kb_dir)
        app.state.memory = ConversationStore(
            r, settings.history_max_turns, settings.history_ttl_seconds
        )
        app.state.limiter = RateLimiter(r, settings.rate_limit_per_minute)
        app.state.model = model or settings.agent_model
        yield
        if redis is None:
            await r.aclose()
        await engine.dispose()

    app = FastAPI(title="AgentDesk", version="0.1.0", lifespan=lifespan)

    async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
        sessions: async_sessionmaker[AsyncSession] = request.app.state.sessions
        async with sessions() as session:
            yield session

    SessionDep = Annotated[AsyncSession, Depends(get_session)]

    @app.get("/health")
    async def health(request: Request, session: SessionDep) -> dict[str, str]:
        await session.execute(text("SELECT 1"))
        await request.app.state.redis.ping()
        return {"status": "ok"}

    @app.post("/chat")
    async def chat(body: ChatRequest, request: Request, session: SessionDep) -> ChatResponse:
        client = request.client.host if request.client else "unknown"
        if not await request.app.state.limiter.allow(client):
            raise HTTPException(429, "Too many messages. Try again in a minute.")
        try:
            message = check_user_message(body.message, settings.max_message_chars)
        except GuardrailViolation as exc:
            raise HTTPException(400, str(exc)) from exc

        memory: ConversationStore = request.app.state.memory
        history = await memory.load(body.session_id)
        deps = Deps(session=session, kb=request.app.state.kb)
        try:
            result = await support_agent.run(
                message,
                deps=deps,
                message_history=history,
                model=request.app.state.model,
                timeout=settings.agent_timeout_seconds,
            )
        except RunTimedOut as exc:
            await session.rollback()
            raise HTTPException(504, "The assistant took too long to answer. Try again.") from exc
        except (ModelAPIError, AgentRunError) as exc:
            await session.rollback()
            raise HTTPException(502, "The assistant is unavailable right now.") from exc
        except UserError as exc:  # e.g. missing API key or unknown model name
            await session.rollback()
            raise HTTPException(503, "The assistant is not configured.") from exc

        await session.commit()  # tickets created by tools are saved only if the run succeeded
        await memory.save(body.session_id, result.all_messages())
        return ChatResponse(
            session_id=body.session_id,
            reply=result.output,
            tools_used=_tool_names(result.new_messages()),
        )

    @app.delete("/chat/{session_id}", status_code=204)
    async def reset_chat(session_id: str, request: Request) -> None:
        await request.app.state.memory.clear(session_id)

    @app.get("/tickets")
    async def list_tickets(session: SessionDep, email: str | None = None) -> list[TicketOut]:
        query = select(Ticket).order_by(Ticket.id.desc()).limit(100)
        if email:
            query = query.where(Ticket.email == email.lower())
        tickets = (await session.scalars(query)).all()
        return [TicketOut.model_validate(t) for t in tickets]

    @app.get("/tickets/{ticket_id}")
    async def get_ticket(ticket_id: int, session: SessionDep) -> TicketOut:
        ticket = await session.get(Ticket, ticket_id)
        if ticket is None:
            raise HTTPException(404, "Ticket not found.")
        return TicketOut.model_validate(ticket)

    return app


def _tool_names(messages: list[Any]) -> list[str]:
    output_tools = {"final_result"}
    return [
        part.tool_name
        for m in messages
        for part in getattr(m, "parts", [])
        if isinstance(part, ToolCallPart) and part.tool_name not in output_tools
    ]
