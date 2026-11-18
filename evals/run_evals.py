"""Run the agent against a real model and score its tool use.

Usage:  GOOGLE_API_KEY=... uv run python evals/run_evals.py

For each case we check that the agent called every expected tool, avoided forbidden
tools, created a ticket only when expected, and cited only real sources.
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from pydantic_ai import ToolCallPart

from agentdesk.agent import Deps, support_agent
from agentdesk.config import get_settings
from agentdesk.db import init_db, make_engine, make_sessionmaker, seed_demo_data
from agentdesk.knowledge import KnowledgeBase

HERE = Path(__file__).parent


async def run_case(case: dict[str, Any], deps: Deps, model: str) -> dict[str, Any]:
    try:
        result = await support_agent.run(case["message"], deps=deps, model=model, timeout=60)
    except Exception as exc:  # count a crash or timeout as a failed case, keep going
        print(f"ERROR {case['message'][:50]}: {type(exc).__name__}", flush=True)
        return {"message": case["message"], "tools_used": [], "answer": "", "error": repr(exc)}
    used = [
        p.tool_name
        for m in result.new_messages()
        for p in getattr(m, "parts", [])
        if isinstance(p, ToolCallPart) and p.tool_name != "final_result"
    ]
    reply = result.output
    checks = {
        "expected_tools": all(t in used for t in case["expected_tools"]),
        "no_forbidden_tools": not any(t in used for t in case["forbidden_tools"]),
        "ticket": (reply.ticket_id is not None) == case.get("expect_ticket", False),
    }
    return {"message": case["message"], "tools_used": used, "answer": reply.answer, **checks}


async def main() -> int:
    settings = get_settings()
    cases = json.loads((HERE / "cases.json").read_text())
    kb = KnowledgeBase.from_dir(settings.kb_dir)

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        engine = make_engine(f"sqlite+aiosqlite:///{tmp}/evals.db")
        await init_db(engine)
        sessions = make_sessionmaker(engine)
        await seed_demo_data(sessions)
        for case in cases:
            async with sessions() as session:  # fresh session per case, rolled back after
                results.append(
                    await run_case(case, Deps(session=session, kb=kb), settings.agent_model)
                )
        await engine.dispose()

    metrics = ["expected_tools", "no_forbidden_tools", "ticket"]
    for r in results:
        for m in metrics:
            r.setdefault(m, False)
    for r in results:
        mark = "PASS" if all(r[m] for m in metrics) else "FAIL"
        print(f"{mark}  {r['message'][:60]:<60}  tools={r['tools_used']}")
    print()
    for m in metrics:
        score = sum(r[m] for r in results) / len(results)
        print(f"{m:<20} {score:.0%}")
    passed = sum(all(r[m] for m in metrics) for r in results)
    print(f"{'all checks':<20} {passed}/{len(results)} cases")

    (HERE / "results.json").write_text(json.dumps(results, indent=2))
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
