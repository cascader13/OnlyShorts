"""
TInvestMarketCollector — сбор рыночных данных (свечей) через T-Invest SDK.

НЕ наследует BaseCollector: тот заточен под новости (RawNews, дедуп по
external_id/хешу). Для свечей дедупликация идемпотентная — по уникальному
ключу (ticker, timeframe, ts) с перезаписью существующих.

Совместим с DataCollector: он вызывает только collector.collect() и читает
collector.source_name, поэтому коллектор попадает в ALL_COLLECTORS и
подхватывается фоновым планировщиком (интервал COLLECT_INTERVAL_SECONDS).
"""

import logging
from typing import Optional

from sqlalchemy.orm import Session

from app.core.config import settings
from app.services.market_data import collect_market_data

logger = logging.getLogger(__name__)


class TInvestMarketCollector:
    """Собирает свечи по отслеживаемым тикерам в таблицу candles."""

    # Имя источника в статистике DataCollector
    source_name = "tinvest"

    def __init__(self, db: Session, tickers: Optional[list[str]] = None):
        self.db = db
        self.tickers = tickers or settings.TRACKED_TICKERS

    def collect(self) -> int:
        """Один проход сбора. Возвращает число новых свечей."""
        stats = collect_market_data(self.db, self.tickers)
        total = int(stats.get("total", 0))
        logger.info("T-Invest: новых свечей за проход: %d", total)
        return total
