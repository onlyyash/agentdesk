"""Database models and async session setup (SQLAlchemy 2.x, asyncpg or aiosqlite)."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import DateTime, ForeignKey, String, Text, select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class OrderStatus(StrEnum):
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    ON_HOLD = "on_hold"


class TicketCategory(StrEnum):
    RETURN = "return"
    WARRANTY = "warranty"
    DELIVERY = "delivery"
    BILLING = "billing"
    ACCOUNT = "account"
    OTHER = "other"


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    orders: Mapped[list["Order"]] = relationship(back_populates="customer")


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[str] = mapped_column(String(20), primary_key=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"))
    status: Mapped[OrderStatus] = mapped_column(String(20))
    items: Mapped[str] = mapped_column(Text)
    tracking_number: Mapped[str | None] = mapped_column(String(50), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    customer: Mapped[Customer] = relationship(back_populates="orders")


class Ticket(Base):
    __tablename__ = "tickets"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), index=True)
    category: Mapped[TicketCategory] = mapped_column(String(20))
    summary: Mapped[str] = mapped_column(Text)
    order_id: Mapped[str | None] = mapped_column(String(20), default=None)
    status: Mapped[str] = mapped_column(String(20), default="open")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


def make_engine(url: str) -> AsyncEngine:
    return create_async_engine(url, pool_pre_ping=True)


def make_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_db(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def seed_demo_data(sessions: async_sessionmaker[AsyncSession]) -> None:
    """Insert a few demo customers and orders so the agent has something to look up."""
    async with sessions() as session, session.begin():
        if (await session.scalar(select(Customer).limit(1))) is not None:
            return
        anna = Customer(email="anna@example.com", name="Anna Berg")
        erik = Customer(email="erik@example.com", name="Erik Holm")
        session.add_all([anna, erik])
        await session.flush()
        session.add_all(
            [
                Order(
                    id="ORD-1001",
                    customer_id=anna.id,
                    status=OrderStatus.SHIPPED,
                    items="Wireless headphones",
                    tracking_number="PN123456789SE",
                ),
                Order(
                    id="ORD-1002",
                    customer_id=anna.id,
                    status=OrderStatus.DELIVERED,
                    items="USB-C charger",
                ),
                Order(
                    id="ORD-1003",
                    customer_id=erik.id,
                    status=OrderStatus.ON_HOLD,
                    items="Mechanical keyboard",
                ),
            ]
        )


async def session_scope(sessions: async_sessionmaker[AsyncSession]) -> AsyncIterator[AsyncSession]:
    async with sessions() as session:
        yield session
