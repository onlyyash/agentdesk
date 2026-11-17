"""Fixed-window rate limiting with Redis INCR + EXPIRE."""

import time

from redis.asyncio import Redis


class RateLimiter:
    def __init__(self, redis: Redis, limit_per_minute: int) -> None:
        self._redis = redis
        self._limit = limit_per_minute

    async def allow(self, client_id: str) -> bool:
        window = int(time.time() // 60)
        key = f"ratelimit:{client_id}:{window}"
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.incr(key)
            pipe.expire(key, 60)
            count, _ = await pipe.execute()
        return int(count) <= self._limit
