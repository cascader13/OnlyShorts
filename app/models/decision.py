from __future__ import annotations

from datetime import datetime
from sqlalchemy import String, DateTime, Float, Integer, JSON, Text, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import TYPE_CHECKING, Optional

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.trade import Trade


class Decision(Base):
    __tablename__ = "decisions"

    id: Mapped[int] = mapped_column(primary_key=True)
    decision_id: Mapped[str] = mapped_column(
        String(36), unique=True, index=True,
        comment="UUID решения"
    )
    ticker: Mapped[str] = mapped_column(String(20), index=True)
    action: Mapped[str] = mapped_column(String(20), index=True)
    quantity: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    confidence: Mapped[float] = mapped_column(Float)
    probability_down: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    probability_up: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    sentiment_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sentiment_news_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    rsi: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    volatility: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sma_50: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sma_200: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    bayesian_network_structure: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    bayesian_network_cpds: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    bayesian_inference_result: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    bayesian_visualization: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    llm_prompt_sentiment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    llm_response_sentiment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    llm_prompt_network: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    llm_response_network: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    llm_prompt_explanation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    llm_response_explanation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # === Связь с Trade (убираем type hint) ===
    trade_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("trades.id"), nullable=True
    )
    # Связь с Trade (viewonly, без back_populates):
    # у обеих таблиц FK друг в друга, поэтому пары направлений не разрешаются
    trade = relationship(
        "Trade",
        foreign_keys=[trade_id],
        uselist=False,
        viewonly=True,
    )

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    evaluated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_decision_ticker_action", "ticker", "action"),
        Index("idx_decision_created", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Decision(id={self.id}, ticker='{self.ticker}', action='{self.action}', confidence={self.confidence:.2f})>"

    @property
    def is_short_signal(self) -> bool:
        return self.action == "SHORT" and self.confidence > 0.6

    @property
    def is_strong_signal(self) -> bool:
        return self.confidence > 0.75