"""
Модели рыночных данных (T-Invest).

Candle   — японские свечи по тикеру и таймфрейму (идемпотентный ключ
           (ticker, timeframe, ts): повторный сбор перезаписывает свечу).
Instrument — кэш метаданных инструмента (для figi, имени, лота).

Время хранится как naive МСК (UTC+3, конвенция проекта: SQLite хранит
DateTime без offset). Конвертацию из timezone-aware значений SDK в МСК
делают в сервисе (app/core/timeutil.to_naive_msk).
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    String,
    DateTime,
    Float,
    Integer,
    Boolean,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.core.timeutil import msk_now


class Instrument(Base):
    """Кэш метаданных инструмента для дашборда и резолва figi."""

    __tablename__ = "instruments"

    ticker: Mapped[str] = mapped_column(
        String(20), primary_key=True, comment="Тикер (PK), напр. SBER"
    )
    figi: Mapped[str] = mapped_column(
        String(32), index=True, comment="FIGI инструмента"
    )
    name: Mapped[str] = mapped_column(
        String(256), comment="Название (кириллица; в консоли Windows кракозябры — косметика)"
    )
    currency: Mapped[Optional[str]] = mapped_column(
        String(10), nullable=True, comment="Валюта (нет в фолбэке InstrumentShort)"
    )
    lot: Mapped[int] = mapped_column(Integer, default=1, comment="Размер лота")
    class_code: Mapped[str] = mapped_column(
        String(10), comment="Код площадки, напр. TQBR"
    )
    instrument_type: Mapped[Optional[str]] = mapped_column(
        String(32), nullable=True, comment="Тип инструмента: share/bond/..."
    )
    uid: Mapped[Optional[str]] = mapped_column(
        String(36), nullable=True, comment="UID инструмента"
    )
    first_1day_candle_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True, comment="Дата самых ранних дневных свечей"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=msk_now, onupdate=msk_now,
        comment="Время обновления кэша (naive МСК, для TTL)",
    )

    def __repr__(self) -> str:
        return f"<Instrument(ticker='{self.ticker}', figi='{self.figi}', name='{self.name}')>"


class Candle(Base):
    """Японская свеча по тикеру и таймфрейму."""

    __tablename__ = "candles"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(String(20), comment="Тикер")
    figi: Mapped[str] = mapped_column(String(32), comment="FIGI (для отладки)")
    timeframe: Mapped[str] = mapped_column(
        String(10), comment="Таймфрейм: 1d / 1h / 15m"
    )
    ts: Mapped[datetime] = mapped_column(
        DateTime, comment="Время начала свечи, naive МСК"
    )
    open: Mapped[float] = mapped_column(Float, comment="Цена открытия")
    high: Mapped[float] = mapped_column(Float, comment="Максимум")
    low: Mapped[float] = mapped_column(Float, comment="Минимум")
    close: Mapped[float] = mapped_column(Float, comment="Цена закрытия")
    volume: Mapped[float] = mapped_column(
        Float, default=0.0, comment="Объём (Float для единообразия с pandas)"
    )
    is_complete: Mapped[bool] = mapped_column(
        Boolean, default=False, comment="Свеча завершена (у последней может быть False)"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=msk_now, comment="naive МСК")

    __table_args__ = (
        # Ключ идемпотентного upsert и дедупликации; заодно создаёт индекс
        UniqueConstraint("ticker", "timeframe", "ts", name="uq_candle_ticker_tf_ts"),
    )

    def __repr__(self) -> str:
        return (
            f"<Candle(ticker='{self.ticker}', tf='{self.timeframe}', "
            f"ts={self.ts}, close={self.close:.2f})>"
        )
