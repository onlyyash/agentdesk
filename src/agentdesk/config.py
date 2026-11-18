"""Application settings, read from environment variables or a .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite+aiosqlite:///./agentdesk.db"
    redis_url: str = "redis://localhost:6379/0"
    agent_model: str = "google:gemini-3.8-flash"
    kb_dir: Path = ROOT / "data" / "kb"

    agent_timeout_seconds: float = 45
    max_message_chars: int = 2000
    history_max_turns: int = 10
    history_ttl_seconds: int = 60 * 60 * 24
    rate_limit_per_minute: int = 20
    seed_demo_data: bool = True

    @field_validator("database_url")
    @classmethod
    def use_async_driver(cls, url: str) -> str:
        """Hosting providers give 'postgres://' URLs; SQLAlchemy async needs '+asyncpg'."""
        for prefix in ("postgres://", "postgresql://"):
            if url.startswith(prefix):
                return "postgresql+asyncpg://" + url.removeprefix(prefix)
        return url


@lru_cache
def get_settings() -> Settings:
    return Settings()
