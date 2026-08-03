"""
DataCollector — оркестратор сбора новостей из всех источников.

Обходит коллекторы из app/collectors, каждый из которых сохраняет новые
записи в raw_news с дедупликацией. Возвращает статистику по источникам.

Коллектор отвечает только за сбор и сохранение сырых новостей.
Анализ (сентимент, тикеры) — задача следующего этапа агента. Здесь же, в
collect_all, выполняется весь остальной конвейер: LLM-анализ новостей,
снапшоты для обучения и (если settings.TRADING_ENABLED) торговый цикл через
run_trading_cycle() — решения -> сделки, мониторинг позиций.
"""

import logging
from datetime import timedelta
from typing import Dict

from sqlalchemy.orm import Session

from app.collectors import ALL_COLLECTORS
from app.core.config import settings
from app.core.database import get_db_context
from app.core.timeutil import msk_now

logger = logging.getLogger(__name__)


def _resolve_account_id() -> str:
    """Счёт для торгового агента: settings.ACCOUNT_ID или первый аккаунт.

    ACCOUNT_ID из .env имеет приоритет. Если он пуст, в песочнице берём первый
    аккаунт (как на дашборде); в боевом режиме счёт задаётся только через
    ACCOUNT_ID — иначе возвращаем пустую строку (торговый цикл будет пропущен).
    """
    if settings.ACCOUNT_ID:
        return settings.ACCOUNT_ID
    if not settings.TINKOFF_SANDBOX:
        logger.warning(
            "Боевой режим: задайте ACCOUNT_ID в .env, чтобы торговый агент знал счёт"
        )
        return ""
    try:
        from app.services.sandbox_account import get_accounts
        accounts = get_accounts()
        if accounts:
            return accounts[0]["account_id"]
        logger.warning("Песочница: аккаунт не открыт — торговый цикл будет пропущен")
    except Exception:
        logger.exception("Не удалось определить счёт для торгового агента")
    return ""


def run_trading_cycle(db: Session) -> Dict[str, object]:
    """Запускает торговый цикл TradingAgent в переданной сессии БД.

    TradingAgent.run_cycle() читает свежие решения, открывает шорты по
    разрешённым решениям и прогоняет мониторинг открытых позиций
    (стоп/тейк/trailing/лимит времени).

    Returns:
        Dict: статистика run_cycle ({opened, closed, errors, ...}),
        {} если торговля выключена, или {"skipped": причина} если счёт
        не определён.
    """
    if not settings.TRADING_ENABLED:
        return {}
    account_id = _resolve_account_id()
    if not account_id:
        return {"skipped": "account_id не определён"}

    from app.services.trading_agent import TradingAgent
    agent = TradingAgent(db, account_id)
    stats = agent.run_cycle()
    logger.info("Торговый цикл (%s): %s", agent.execution.mode, stats)
    return stats


class DataCollector:
    """Собирает новости из всех источников и сохраняет их в raw_news."""

    def __init__(self, db: Session):
        self.db = db
        self.collectors = [collector_cls(db) for collector_cls in ALL_COLLECTORS]

    def collect_all(self) -> Dict[str, object]:
        """
        Запускает сбор из всех источников, затем анализирует необработанные
        новости через LLM, собирает снапшоты для обучения и (если торговля
        включена) торговый цикл: решения -> сделки, мониторинг позиций.

        Returns:
            Dict: статистика по каждому источнику + analyzed_news + snapshots
            + trading (если settings.TRADING_ENABLED)
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

        # Анализ необработанных новостей через LLM: несколько батчей за проход,
        # чтобы охватить большую часть свежих новостей. Лимиты (число новостей,
        # бюджет времени, размер батча) — настройки NEWS_* в .env.
        try:
            from app.services.news_analyzer import process_many
            analyzed = process_many(
                self.db,
                batch_size=settings.NEWS_BATCH_SIZE,
                max_news=settings.NEWS_MAX_PER_CYCLE,
                max_seconds=settings.NEWS_MAX_SECONDS_PER_CYCLE,
            )
            stats["analyzed_news"] = analyzed
            if analyzed:
                logger.info("LLM обработал %d новых новостей", analyzed)
        except Exception:
            logger.exception("Ошибка анализа новостей через LLM")

        # Снапшоты для обучения: live-слепок текущего часа + разметка прошлых
        # + рефреш новостных колонок у снапшотов, чьё окно публикаций накрыло
        # новости, которые LLM только что обработал (слепок снят раньше
        # анализа — «догоняем» их постфактум по published_at).
        if settings.SNAPSHOT_ENABLED:
            try:
                from app.services.training_data import (
                    capture_live,
                    label_expired,
                    refresh_news_features,
                )
                captured = capture_live(self.db)
                labeled = label_expired(self.db)
                refreshed = refresh_news_features(
                    self.db,
                    since=msk_now() - timedelta(hours=settings.SNAPSHOT_REFRESH_HOURS),
                )
                stats["snapshots"] = {
                    "captured": captured,
                    "labeled": labeled,
                    "refreshed": refreshed,
                }
                if captured or labeled or refreshed:
                    logger.info(
                        "Датасет: снапшотов +%d, размечено +%d, новостей пересчитано +%d",
                        captured, labeled, refreshed,
                    )
            except Exception:
                logger.exception("Ошибка захвата снапшотов для обучения")

        # Торговый цикл: решения -> сделки, мониторинг и закрытие позиций.
        if settings.TRADING_ENABLED:
            try:
                stats["trading"] = run_trading_cycle(self.db)
            except Exception:
                logger.exception("Ошибка запуска торгового цикла")
                stats["trading"] = {"error": "торговый цикл упал"}

        logger.info("Сбор завершен: %s", stats)
        return stats


def collect_once() -> Dict[str, object]:
    """
    Удобная обёртка: один проход сбора в своей сессии БД.

    Используется из app.main и скриптов.
    """
    with get_db_context() as db:
        return DataCollector(db).collect_all()
