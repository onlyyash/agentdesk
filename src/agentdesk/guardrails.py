"""Input guardrails that run before a message reaches the model."""

import re

_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore (all |any )?(previous|prior|above) (instructions|prompts?)",
        r"(reveal|show|print) (your|the) (system )?(prompt|instructions)",
        r"you are now (a|an|in) ",
        r"\bjailbreak\b",
        r"developer mode",
    )
]


class GuardrailViolation(ValueError):
    """Raised when a user message is rejected before it reaches the agent."""


def check_user_message(message: str, max_chars: int) -> str:
    """Return the cleaned message, or raise GuardrailViolation."""
    cleaned = message.strip()
    if not cleaned:
        raise GuardrailViolation("Message is empty.")
    if len(cleaned) > max_chars:
        raise GuardrailViolation(f"Message is longer than {max_chars} characters.")
    if any(p.search(cleaned) for p in _INJECTION_PATTERNS):
        raise GuardrailViolation("Message looks like an attempt to change the assistant's rules.")
    return cleaned
