"""Conversation memory in Redis, trimmed to the last N user turns."""

from pydantic_ai import ModelMessage, ModelMessagesTypeAdapter, ModelRequest, UserPromptPart
from redis.asyncio import Redis


def _starts_turn(message: ModelMessage) -> bool:
    return isinstance(message, ModelRequest) and any(
        isinstance(part, UserPromptPart) for part in message.parts
    )


def trim_to_last_turns(messages: list[ModelMessage], max_turns: int) -> list[ModelMessage]:
    """Keep the last `max_turns` user turns.

    We only cut at the start of a user turn, so a tool call is never separated
    from its tool result (models reject histories like that).
    """
    turn_starts = [i for i, m in enumerate(messages) if _starts_turn(m)]
    if len(turn_starts) <= max_turns:
        return messages
    return messages[turn_starts[-max_turns] :]


class ConversationStore:
    def __init__(self, redis: Redis, max_turns: int, ttl_seconds: int) -> None:
        self._redis = redis
        self._max_turns = max_turns
        self._ttl = ttl_seconds

    @staticmethod
    def _key(session_id: str) -> str:
        return f"history:{session_id}"

    async def load(self, session_id: str) -> list[ModelMessage]:
        raw = await self._redis.get(self._key(session_id))
        if raw is None:
            return []
        return ModelMessagesTypeAdapter.validate_json(raw)

    async def save(self, session_id: str, messages: list[ModelMessage]) -> None:
        trimmed = trim_to_last_turns(messages, self._max_turns)
        payload = ModelMessagesTypeAdapter.dump_json(trimmed)
        await self._redis.set(self._key(session_id), payload, ex=self._ttl)

    async def clear(self, session_id: str) -> None:
        await self._redis.delete(self._key(session_id))
