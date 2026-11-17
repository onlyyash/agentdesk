from fakeredis import FakeAsyncRedis

from agentdesk.ratelimit import RateLimiter


async def test_limit_is_per_client() -> None:
    limiter = RateLimiter(FakeAsyncRedis(), limit_per_minute=2)
    assert [await limiter.allow("a") for _ in range(3)] == [True, True, False]
    assert await limiter.allow("b") is True
