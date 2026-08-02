from __future__ import annotations

from datetime import datetime
from sqlalchemy import String, DateTime, Float, Integer, ForeignKey, Index, Boolean, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import TYPE_CHECKING, Optional

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.decision import Decision


class Trade(Base):
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(primary_key=True)
    trade_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    ticker: Mapped[str] = mapped_column(String(20), index=True)
    direction: Mapped[str] = mapped_column(String(10), index=True)

    entry_price: Mapped[float] = mapped_column(Float)
    exit_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    quantity: Mapped[int] = mapped_column(Integer)
    entry_value: Mapped[float] = mapped_column(Float)
    exit_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    entry_commission: Mapped[float] = mapped_column(Float, default=0.0)
    exit_commission: Mapped[float] = mapped_column(Float, default=0.0)
    borrow_fee: Mapped[float] = mapped_column(Float, default=0.0)
    total_fees: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pnl_percent: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    status: Mapped[str] = mapped_column(String(20), default="OPEN", index=True)
    is_profitable: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)

    # === Связь с Decision (убираем type hint) ===
    decision_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("decisions.id"), nullable=True
    )
    # Связь с Decision (viewonly, без back_populates):
    # у обеих таблиц FK друг в друга, поэтому пары направлений не разрешаются
    decision = relationship(
        "Decision",
        foreign_keys=[decision_id],
        uselist=False,
        viewonly=True,
    )

    stop_loss_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    take_profit_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    highest_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    lowest_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    opened_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    closed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    duration_hours: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    broker_order_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("idx_trade_ticker_status", "ticker", "status"),
        Index("idx_trade_opened", "opened_at"),
        Index("idx_trade_pnl", "pnl"),
    )

    def __repr__(self) -> str:
        pnl_str = f"{self.pnl:.2f}" if self.pnl is not None else "None"
        if self.is_profitable:
            status_marker = "profit"
        elif self.pnl is not None:
            status_marker = "loss"
        else:
            status_marker = "open"
        return (
            f"<Trade(id={self.id}, ticker='{self.ticker}', direction='{self.direction}', "
            f"pnl={pnl_str} {status_marker})>"
        )

    def close(self, exit_price: float, commission: float = 0.0, notes: str = None):
        self.exit_price = exit_price
        self.exit_value = exit_price * self.quantity
        self.exit_commission = commission
        self.closed_at = datetime.utcnow()
        self.status = "CLOSED"
        self.duration_hours = (self.closed_at - self.opened_at).total_seconds() / 3600

        if self.direction == "SHORT":
            self.pnl = (self.entry_price - exit_price) * self.quantity - self.total_fees
        else:
            self.pnl = (exit_price - self.entry_price) * self.quantity - self.total_fees

        self.pnl_percent = (self.pnl / self.entry_value) * 100 if self.entry_value else 0
        self.is_profitable = self.pnl > 0 if self.pnl is not None else False

        if notes:
            self.notes = notes

    def update_stop_loss(self, new_stop_price: float):
        self.stop_loss_price = new_stop_price

    def update_price_monitoring(self, current_price: float):
        if self.highest_price is None or current_price > self.highest_price:
            self.highest_price = current_price
        if self.lowest_price is None or current_price < self.lowest_price:
            self.lowest_price = current_price