"""The support agent: instructions, tools and a validated, structured reply."""

import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agentdesk.db import Customer, Order, Ticket, TicketCategory
from agentdesk.knowledge import KnowledgeBase

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

INSTRUCTIONS = """\
You are the customer-support assistant for Northwind Gear, an online electronics store.

Rules:
- For questions about policies (shipping, returns, refunds, warranty, payments, accounts),
  call `search_help_center` first and answer only from what it returns. List the `source`
  of every section you used in `sources`.
- For questions about a specific order, call `get_order_status`. You need both the order id
  and the customer's email. If either is missing, ask for it instead of guessing.
- Call `create_ticket` only when the customer wants something a person must handle
  (a return, a warranty claim, a lost parcel, an account deletion), and only once you have
  their email. Put the returned ticket id in `ticket_id`.
- If you cannot help, say so and set `needs_human` to true.
- Never invent order details, policies, ticket ids or sources. Keep answers short and friendly.
"""


class SupportReply(BaseModel):
    """The structured output every agent run must produce."""

    answer: str = Field(description="The reply shown to the customer.")
    sources: list[str] = Field(
        default_factory=list, description="Help-centre sources used, e.g. 'returns.md#refunds'."
    )
    ticket_id: int | None = Field(default=None, description="Id of a ticket created in this turn.")
    needs_human: bool = Field(default=False, description="True if a person should take over.")


@dataclass
class Deps:
    session: AsyncSession
    kb: KnowledgeBase
    created_ticket_ids: list[int] = field(default_factory=list)


support_agent = Agent(
    deps_type=Deps,
    output_type=SupportReply,
    instructions=INSTRUCTIONS,
    retries=2,
    defer_model_check=True,
)


@support_agent.tool
async def search_help_center(ctx: RunContext[Deps], query: str) -> list[dict[str, str]]:
    """Search the help-centre articles. Returns the most relevant sections with their source."""
    hits = ctx.deps.kb.search(query, k=3)
    if not hits:
        return [{"source": "", "title": "", "text": "No matching help-centre article."}]
    return [{"source": s.source, "title": s.title, "text": s.text} for s in hits]


@support_agent.tool
async def get_order_status(ctx: RunContext[Deps], order_id: str, email: str) -> dict[str, str]:
    """Look up an order. The email must belong to the customer who placed the order."""
    row = (
        await ctx.deps.session.execute(
            select(Order, Customer.email).join(Customer).where(Order.id == order_id.strip().upper())
        )
    ).first()
    # Same answer for "no such order" and "not your order", so we never leak whether
    # someone else's order id exists.
    if row is None or row.email.lower() != email.strip().lower():
        return {"error": "No order found for that order id and email."}
    order = row.Order
    return {
        "order_id": order.id,
        "status": str(order.status),
        "items": order.items,
        "tracking_number": order.tracking_number or "not available yet",
    }


@support_agent.tool
async def create_ticket(
    ctx: RunContext[Deps],
    email: str,
    category: TicketCategory,
    summary: str,
    order_id: str | None = None,
) -> dict[str, int | str]:
    """Open a support ticket for a person to handle. Returns the new ticket id."""
    if not _EMAIL.match(email.strip()):
        raise ModelRetry("That is not a valid email address. Ask the customer for their email.")
    if len(summary.strip()) < 10:
        raise ModelRetry("Write a summary of at least one full sentence.")
    ticket = Ticket(
        email=email.strip().lower(),
        category=category,
        summary=summary.strip(),
        order_id=order_id.strip().upper() if order_id else None,
    )
    ctx.deps.session.add(ticket)
    await ctx.deps.session.flush()
    ctx.deps.created_ticket_ids.append(ticket.id)
    return {"ticket_id": ticket.id, "status": ticket.status}


@support_agent.output_validator
async def validate_reply(ctx: RunContext[Deps], reply: SupportReply) -> SupportReply:
    """Output guardrail: reject made-up sources or ticket ids and make the model retry."""
    unknown = [s for s in reply.sources if s not in ctx.deps.kb.sources]
    if unknown:
        raise ModelRetry(f"These sources do not exist: {unknown}. Only cite returned sources.")
    if reply.ticket_id is not None and reply.ticket_id not in ctx.deps.created_ticket_ids:
        raise ModelRetry("ticket_id must be an id returned by create_ticket in this conversation.")
    return reply
