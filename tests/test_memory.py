from fakeredis import FakeAsyncRedis
from pydantic_ai import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from agentdesk.memory import ConversationStore, trim_to_last_turns


def turn(n: int, with_tool: bool = False) -> list[ModelMessage]:
    msgs: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(f"question {n}")])]
    if with_tool:
        msgs += [
            ModelResponse(parts=[ToolCallPart("search_help_center", {"q": "x"}, "call-1")]),
            ModelRequest(parts=[ToolReturnPart("search_help_center", "result", "call-1")]),
        ]
    msgs.append(ModelResponse(parts=[TextPart(f"answer {n}")]))
    return msgs


def test_trim_keeps_last_turns_and_never_splits_a_tool_call() -> None:
    history = turn(1) + turn(2, with_tool=True) + turn(3)
    trimmed = trim_to_last_turns(history, max_turns=2)
    assert trimmed == history[2:]  # drops turn 1 (2 messages), keeps turn 2 with its tool call
    assert isinstance(trimmed[0], ModelRequest)


def test_trim_is_a_no_op_for_short_histories() -> None:
    history = turn(1)
    assert trim_to_last_turns(history, max_turns=5) == history


async def test_store_round_trips_and_trims() -> None:
    store = ConversationStore(FakeAsyncRedis(), max_turns=2, ttl_seconds=60)
    await store.save("s1", turn(1) + turn(2) + turn(3))
    loaded = await store.load("s1")
    assert len(loaded) == 4
    first = loaded[0]
    assert isinstance(first, ModelRequest)
    assert first.parts[0].content == "question 2"  # type: ignore[union-attr]

    await store.clear("s1")
    assert await store.load("s1") == []
