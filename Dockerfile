FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY data ./data
COPY README.md ./
RUN uv sync --frozen --no-dev

RUN useradd --create-home app
USER app

EXPOSE 8000
CMD ["sh", "-c", "/app/.venv/bin/uvicorn agentdesk.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
