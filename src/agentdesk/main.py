"""ASGI entry point: `uvicorn agentdesk.main:app`."""

from agentdesk.api import create_app

app = create_app()
