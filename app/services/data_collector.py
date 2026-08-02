"""
DataCollector — оркестратор сбора новостей из всех источников.

Обходит коллекторы из app/collectors, каждый из которых сохраняет новые
записи в raw_news с дедупликацией. Возвращает статистику по источникам.

Коллектор отвечает только за сбор и сохранение сырых новостей.
Анализ (сентимент, тикеры) — задача следующего этапа агента.
"""

import logging
from typing import Dict

from sqlalchemy.orm import Session

from app.collectors import ALL_COLLECTORS
from app.core.config import settings
from app.core.database import get_db_context

logger = logging.getLogger(__name__)


class DataCollector:
    """Собирает новости из всех источников и сохраняет их в raw_news."""

    def __init__(self, db: Session):
        self.db = db
        self.collectors = [collector_cls(db) for collector_cls in ALL_COLLECTORS]

    def collect_all(self) -> Dict[str, object]:
        """
        Запускает сбор из всех источников, затем анализирует необработанные
        новости через LLM и запускает байесовского агента.

        Returns:
            Dict: статистика по каждому источнику + analyzed_news + bayesian
        """
        stats: Dict[str, object] = {}
        total = 0

        for collector in self.collectors:
            try:
                count = collector.collect()
            except Exception:
                logger.exception(
                    "Коллектор %s завершился ошибкой", collector.source_name
                )
                count = 0
            stats[collector.source_name] = count
            total += count

        stats["total_new"] = total

        # Анализ необработанных новостей через LLM
        try:
            from app.services.news_analyzer import process_unprocessed
            analyzed = process_unprocessed(self.db, limit=5)
            stats["analyzed_news"] = analyzed
            if analyzed:
                logger.info("LLM обработал %d новых новостей", analyzed)
        except Exception:
            logger.exception("Ошибка анализа новостей через LLM")

        # Байесовская сеть: технические агенты (RSI/SMA/волатильность) + новости
        # за 3 часа → LLM → валидный JSON. Запускается после обновления данных.
        if settings.BAYESIAN_AGENT_ENABLED:
            try:
                from app.services.bayesian_network import run_bayesian_for_tickers
                bayesian = run_bayesian_for_tickers(self.db)
                stats["bayesian"] = bayesian
                valid_count = sum(
                    1 for e in bayesian.values() if e.get("valid")
                )
                logger.info(
                    "Байесовская сеть: валидных %d из %d (%s)",
                    valid_count, len(bayesian), bayesian,
                )
            except Exception:
                logger.exception("Ошибка запуска байесовской сети")

        logger.info("Сбор завершен: %s", stats)
        return stats


def collect_once() -> Dict[str, int]:
    """
    Удобная обёртка: один проход сбора в своей сессии БД.

    Используется из app.main и скриптов.
    """
    with get_db_context() as db:
        return DataCollector(db).collect_all()
