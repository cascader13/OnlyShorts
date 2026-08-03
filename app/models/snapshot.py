"""
Модель снапшота для обучения модели шортов.

Строка — момент времени T с признаками, известными на T:
- технические: свечи SNAPSHOT_TIMEFRAME as-of T (RSI/SMA/волатильность);
- новостные: сентимент по тикеру за окно [T - window_hours, T);
и целевой переменной target = 1, если path-симуляция шорта (те же правила
выхода, что исполняет PositionManager: стоп/тейк/trailing/лимит времени)
закрылась в плюс. Метка и исполнение живут в одной стратегии — модель учится
на том, что агент реально сделает, а не на «close(T+N) < price».

target = NULL означает «будущих свечей ещё нет или окно неполное (рынок
закрылся раньше горизонта)» — строка будет размечена позже (label_expired в
app/services/training_data.py). Так работает единый конвейер «снимок сейчас
-> симуляция пути» и для backfill, и для live.

Время хранится как naive МСК (UTC+3, конвенция проекта).
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.core.timeutil import msk_now


class TrainingSnapshot(Base):
    """Признаки и метка в момент времени T для обучения модели шортов."""

    __tablename__ = "training_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "ticker", "timestamp", "window_hours", "lookahead_hours",
            name="uq_snapshot_ticker_ts_window_lookahead",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, index=True, comment="Время решения (naive МСК)"
    )
    ticker: Mapped[str] = mapped_column(String(20), index=True)

    # --- Технические признаки (as-of timestamp, свечи SNAPSHOT_TIMEFRAME) ---
    rsi: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sma_20: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sma_50: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    price_vs_sma_20: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    price_vs_sma_50: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    volatility: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    volume: Mapped[Optional[float]] = mapped_column(Float, nullable=True, comment="Volume последней свечи")

    # --- Новостные признаки (окно [timestamp - window_hours, timestamp)) ---
    news_sentiment_avg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    news_count: Mapped[int] = mapped_column(Integer, default=0)
    news_confidence_avg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sentiment_change_2h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # --- Целевая переменная: 1 если path-симуляция шорта закрылась в плюс ---
    target: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # --- Результат path-симуляции выхода (правила PositionManager) ---
    target_return_pct: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="Реализованный P&L симуляции выхода, % от входа (плюс = прибыль)",
    )
    exit_reason: Mapped[Optional[str]] = mapped_column(
        String(20), nullable=True,
        comment="stop_loss | take_profit | trailing_stop | time_stop",
    )
    sim_exit_price: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Цена выхода в симуляции"
    )
    sim_duration_hours: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Длительность удержания в симуляции, ч"
    )
    mae_pct: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="Max adverse excursion, % от входа (>=0; для шорта рост цены)",
    )
    mfe_pct: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="Max favorable excursion, % от входа (<=0; для шорта падение цены)",
    )
    max_hold_hours: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="Горизонт симуляции (MAX_HOLD_HOURS на момент разметки)",
    )
    # Параметры правил выхода, которыми размечена строка (самодокументация).
    # label_version кодирует их же; эти колонки — для фильтрации/запросов.
    label_stop_loss_pct: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="STOP_LOSS_PERCENT на момент разметки"
    )
    label_take_profit_pct: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="TAKE_PROFIT_PERCENT на момент разметки"
    )
    label_trail_activation_pct: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="TRAILING_STOP_ACTIVATION_PERCENT на момент разметки",
    )
    label_trail_distance_pct: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="TRAILING_STOP_DISTANCE_PERCENT на момент разметки",
    )
    label_version: Mapped[Optional[str]] = mapped_column(
        String(40), nullable=True,
        comment="Рецепт разметки path_sim_sl5_tp3_ta2_td1.5_h4 (NULL — старая метка)",
    )

    # --- Служебные ---
    window_hours: Mapped[int] = mapped_column(Integer, default=2)
    lookahead_hours: Mapped[int] = mapped_column(Integer, default=2)
    source: Mapped[Optional[str]] = mapped_column(
        String(10), nullable=True, comment="live | backfill"
    )
    labeled_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True, comment="Когда размечен target"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=msk_now, comment="Время создания строки (naive МСК)"
    )

    def __repr__(self) -> str:  # pragma: no cover — для отладки
        return (
            f"<TrainingSnapshot {self.ticker} {self.timestamp:%Y-%m-%d %H:%M} "
            f"target={self.target} price={self.price}>"
        )
