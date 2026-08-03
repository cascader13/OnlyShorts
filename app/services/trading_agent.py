"""
Торговый агент: превращает решения (Decision) в реальные сделки.

Цикл работы (run_cycle):
1. Прочитать свежие решения (последние N минут) и обработать каждое:
   process_decision — шорт, если решение сигналит вниз и риск-менеджер разрешил.
2. Прогнать мониторинг открытых позиций (position_manager.monitor_and_close):
   стоп-лосс / тейк-профит / лимит времени / trailing stop.
3. Вернуть статистику {opened, closed, errors}.

Paper-режим: settings.PAPER_TRADING=True (по умолчанию) — ордера НЕ отправляются
брокеру. ExecutionService имитирует исполнение (status=FILL), сделки пишутся
только в БД, broker_order_id получает префикс PAPER-. Тот же ExecutionService
разделяется с PositionManager, поэтому и закрытие позиций тоже не ходит на
брокера. Для реальной торговли выставите PAPER_TRADING=False в .env.

Ключевые решения:
- Одна сделка на одно решение: decision.trade_id становится ссылкой на созданный
  Trade, повторная обработка того же решения исключается.
- quantity в таблице trades хранится в АКЦИЯХ (согласовано с Trade.close() и
  балансом брокера); на брокера уходит число лотов = max_lots от RiskManager.
- Открываем полный разрешённый размер (max_lots) — риск-менеджер уже ограничил
  позицию и по капиталу, и по волатильности.
"""

import logging
from datetime import timedelta
from typing import Optional
from uuid import uuid4

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.timeutil import msk_now
from app.models.decision import Decision
from app.models.trade import Trade
from app.services.execution import ExecutionService
from app.services.position_manager import PositionManager
from app.services.risk_manager import RiskManager

logger = logging.getLogger(__name__)

# Свежесть решений: обрабатываем только те, что созданы за последние N минут
DECISION_AGE_MINUTES = 10

# Статусы исполнения, при которых позиция считается открытой
_FILLED_STATUSES = {"FILL", "PARTIALLYFILL"}


class TradingAgent:
    """Агент: решение -> сделка, мониторинг позиций, статистика цикла."""

    def __init__(self, db: Session, account_id: str):
        self.db = db
        self.account_id = account_id

        # Единый ExecutionService разделяется с PositionManager, чтобы и
        # закрытие позиций уважало paper-режим (не слало ордера на брокера).
        self.execution = ExecutionService(
            account_id=account_id, paper=settings.PAPER_TRADING,
        )
        self.risk_manager = RiskManager(db=db, account_id=account_id)
        self.position_manager = PositionManager(
            db=db, account_id=account_id, execution=self.execution,
        )

    # === Обработка одного решения ===

    def process_decision(self, decision: Decision) -> Optional[Trade]:
        """Превращает решение в открытую SHORT-позицию, если это разрешено.

        Шаги: проверка сигнала -> риск-менеджер -> размер позиции ->
        стоп/тейк -> исполнение на брокере (или paper) -> запись в trades ->
        связь decision.trade_id.

        Returns:
            Trade или None (сигнал слабый, риск-менеджер отказал, ордер не
            исполнился, решение уже обработано).
        """
        if not decision.is_short_signal:
            logger.debug("process_decision: %s — не SHORT-сигнал", decision.ticker)
            return None
        if decision.trade_id is not None:
            logger.debug(
                "process_decision: %s уже обработано (trade_id=%s)",
                decision.decision_id, decision.trade_id,
            )
            return None

        ticker = decision.ticker.strip().upper()

        # Цена для расчёта размера: свежая, фолбэк на снапшот из решения
        price = self.position_manager.get_current_price(ticker) or decision.price
        if not price or price <= 0:
            logger.warning("process_decision: нет цены для %s — пропускаем", ticker)
            return None

        # Риск-менеджер: уверенность, лимиты, маржинальная доступность
        ok, reason, max_lots = self.risk_manager.can_open_short(
            ticker, decision.confidence, price,
        )
        if not ok:
            logger.info(
                "process_decision: %s отклонено риск-менеджером: %s", ticker, reason,
            )
            return None

        # Размер: quantity (акции) = лоты * размер лота
        try:
            lot_size = self.risk_manager.get_lot_size(ticker)
        except Exception as exc:
            logger.error("process_decision: размер лота %s недоступен: %s", ticker, exc)
            return None
        quantity_lots = max_lots
        quantity = quantity_lots * lot_size

        # Стоп-лосс / тейк-профит из настроек
        stop_loss, take_profit = self.risk_manager.get_stop_levels(price)

        # Исполнение: реальный ордер или paper-имитация
        order = self.execution.open_short(ticker, quantity_lots)
        status = order.get("status", "")
        if status not in _FILLED_STATUSES:
            logger.warning(
                "process_decision: ордер %s (%s) не исполнен: status=%s message=%s",
                ticker, order.get("order_id"), status, order.get("message", ""),
            )
            return None

        entry_price = order.get("average_price") or price
        trade = Trade(
            trade_id=str(uuid4()),
            ticker=ticker,
            direction="SHORT",
            entry_price=entry_price,
            quantity=quantity,
            entry_value=entry_price * quantity,
            status="OPEN",
            decision_id=decision.id,
            broker_order_id=order.get("order_id"),
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
        )
        self.db.add(trade)
        self.db.commit()

        # Связь решение -> сделка (защита от повторной обработки)
        decision.trade_id = trade.id
        self.db.commit()

        logger.info(
            "process_decision (%s): открыт шорт %s lots=%d price=%.2f "
            "stop=%.2f tp=%.2f trade=%s",
            self.execution.mode, ticker, quantity_lots, entry_price,
            stop_loss, take_profit, trade.trade_id,
        )
        return trade

    # === Полный цикл ===

    def run_cycle(self) -> dict:
        """Один торговый цикл: решения -> открытие -> мониторинг.

        Returns:
            dict: {opened, closed, errors} — число открытых/закрытых позиций
            и ошибок за проход. Дополнительно: decisions (рассмотрено),
            paper (режим).
        """
        stats: dict = {"opened": 0, "closed": 0, "errors": 0}

        # 1. Свежие решения
        decisions: list[Decision] = []
        try:
            decisions = self._recent_decisions()
        except Exception:
            stats["errors"] += 1
            logger.exception("run_cycle: не удалось прочитать решения")
        stats["decisions"] = len(decisions)

        # 2. Обработка решений
        for decision in decisions:
            try:
                if self.process_decision(decision) is not None:
                    stats["opened"] += 1
            except Exception:
                stats["errors"] += 1
                logger.exception(
                    "run_cycle: ошибка по решению %s (%s)",
                    decision.decision_id, decision.ticker,
                )

        # 3. Мониторинг и закрытие открытых позиций
        try:
            closed = self.position_manager.monitor_and_close()
            stats["closed"] = len(closed)
        except Exception:
            stats["errors"] += 1
            logger.exception("run_cycle: monitor_and_close упал")

        stats["paper"] = settings.PAPER_TRADING
        logger.info(
            "run_cycle: opened=%d closed=%d errors=%d",
            stats["opened"], stats["closed"], stats["errors"],
        )
        return stats

    # === Приватные помощники ===

    def _recent_decisions(self) -> list[Decision]:
        """SHORT-решения за последние DECISION_AGE_MINUTES минут, свежие сверху."""
        cutoff = msk_now() - timedelta(minutes=DECISION_AGE_MINUTES)
        return (
            self.db.query(Decision)
            .filter(
                Decision.action == "SHORT",
                Decision.created_at >= cutoff,
            )
            .order_by(Decision.created_at.desc())
            .all()
        )
