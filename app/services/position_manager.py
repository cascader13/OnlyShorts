"""
Управление открытыми позициями (SHORT) на T-Invest.

Три задачи:
1. sync_open_trades — сверяет собственную учётную запись (таблица trades) с
   реальными позициями брокера: если позиции у брокера нет (закрыта вручную,
   маржин-коллом или сбойным прогоном), закрываем запись и в БД.
2. monitor_and_close — главный цикл: для каждой открытой позиции обновляет
   экстремумы (highest/lowest), проверяет стоп-лосс, тейк-профит, лимит
   времени удержания (MAX_HOLD_HOURS) и trailing stop; при срабатывании
   закрывает позицию через ExecutionService и записывает результат в trades.
3. force_close_all — аварийное закрытие всех открытых позиций.

Ключевые решения:
- Торгуем лотами, но в таблице trades quantity хранится в АКЦИЯХ: так
  согласуются Trade.close() (entry_value = entry_price * quantity) и баланс
  брокера get_positions (balance — число акций). Перед close_short (принимает
  лоты) акции переводятся в лоты через размер лота.
- Цена для мониторинга: сначала живая LastPrice из API (get_last_price),
  фолбэк — последняя свеча из БД; если цены нет вовсе, позицию не трогаем
  (лучше переждать, чем закрыть по неизвестной цене).
- Позиция считается закрытой, только если брокер подтвердил исполнение
  (FILL / PARTIALLYFILL); отклонённый или отменённый ордер запись в БД
  не закрывает.
- Песочница нестабильна — все обращения к брокеру идут через существующие
  хелперы с ретраями (execution, sandbox_account, market_data).
"""

import logging
import math
from datetime import timedelta
from typing import Optional

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.timeutil import msk_now
from app.models.market import Instrument
from app.models.trade import Trade
from app.services import sandbox_account
from app.services.execution import ExecutionService
from app.services.market_data import get_candles_from_db, get_last_price

logger = logging.getLogger(__name__)

# --- Trailing stop (настройки TRAILING_STOP_ACTIVATION/DISTANCE_PERCENT) ---
# Для шорта "в нашу пользу" — падение цены, лучшая точка — минимум.
# Trailing активируется, когда цена упала от входа на
# settings.TRAILING_STOP_ACTIVATION_PERCENT%, и срабатывает при отскоке на
# settings.TRAILING_STOP_DISTANCE_PERCENT% от минимума.

# Статусы исполнения, при которых позицию считаем закрытой
_FILLED_STATUSES = {"FILL", "PARTIALLYFILL"}


class PositionManager:
    """Мониторинг и закрытие открытых SHORT-позиций на T-Invest."""

    def __init__(self, db: Session, account_id: str, execution: ExecutionService):
        self.db = db
        self.account_id = account_id
        self.execution = execution

    # === Синхронизация с брокером ===

    def sync_open_trades(self) -> None:
        """Сверяет открытые сделки в БД с реальными позициями брокера.

        Брокер отдаёт позиции по тикеру (число акций в шорте), поэтому сверка
        идёт по тикеру с "бюджетом": открытыми могут остаться не больше акций,
        чем держит брокер. Свежие сделки считаем ещё живыми; излишек (внешнее
        закрытие — вручную, маржин-коллом, сбойным прогоном) закрываем в БД по
        текущей цене или урезаем quantity. Позиции, которых нет в нашей БД,
        не трогаем — мы управляем только своими сделками.
        """
        open_trades = (
            self.db.query(Trade)
            .filter(Trade.status == "OPEN", Trade.direction == "SHORT")
            .all()
        )
        if not open_trades:
            return

        try:
            positions = sandbox_account.get_positions(self.account_id)
        except Exception:
            logger.exception("sync_open_trades: не удалось получить позиции брокера")
            return

        # Карта: ticker -> число акций в шорте (баланс < 0)
        broker_shorts: dict[str, int] = {}
        for sec in positions.get("securities", []):
            balance = int(sec.get("balance", 0))
            if balance < 0:
                broker_shorts[sec["ticker"]] = abs(balance)

        by_ticker: dict[str, list[Trade]] = {}
        for trade in open_trades:
            by_ticker.setdefault(trade.ticker, []).append(trade)

        for ticker, trades in by_ticker.items():
            broker_qty = broker_shorts.get(ticker, 0)
            total_open = sum(t.quantity for t in trades)
            # Бюджет открытых акций: сколько может остаться у нас в БД.
            # Проходим от свежих сделок к старым: свежие оставляем, излишек убираем.
            budget = broker_qty
            for trade in sorted(trades, key=lambda t: t.opened_at, reverse=True):
                if budget <= 0:
                    self._close_externally(trade)
                elif trade.quantity > budget:
                    logger.warning(
                        "sync: %s #%s записано %d акций, брокер держит %d — "
                        "частичное внешнее закрытие, снижаем до %d",
                        ticker, trade.trade_id, trade.quantity, broker_qty, budget,
                    )
                    trade.quantity = budget
                    self.db.commit()
                    budget = 0
                else:
                    budget -= trade.quantity
            logger.debug(
                "sync: %s БД=%d брокер=%d (%s)",
                ticker, total_open, broker_qty,
                "в порядке" if broker_qty >= total_open else "скорректировано",
            )

    # === Мониторинг и закрытие ===

    def monitor_and_close(self) -> list[dict]:
        """Главный цикл: мониторит открытые позиции и закрывает по триггерам.

        Для каждой позиции со status="OPEN": получить текущую цену, обновить
        highest/lowest, проверить стоп-лосс / тейк-профит / лимит времени /
        trailing stop. При срабатывании — закрыть через брокера и записать
        Trade.close().

        Returns:
            list[dict]: закрытые за проход позиции (trade_id, ticker, reason,
            pnl, pnl_percent, ...). Если закрыто нечего — пустой список.
        """
        closed: list[dict] = []
        open_trades = self.db.query(Trade).filter(Trade.status == "OPEN").all()
        if not open_trades:
            return closed

        for trade in open_trades:
            if trade.direction != "SHORT":
                continue
            try:
                price = self.get_current_price(trade.ticker)
                if not price:
                    logger.warning(
                        "monitor: нет цены для %s (#%s) — пропускаем проход",
                        trade.ticker, trade.trade_id,
                    )
                    continue

                trade.update_price_monitoring(price)
                reason = self._check_exit_reason(trade, price)
                if reason:
                    result = self._close_via_broker(trade, reason, fallback_price=price)
                    if result:
                        closed.append(result)
                else:
                    self.db.commit()  # экстремумы обновлены, триггеров нет
            except Exception:
                logger.exception(
                    "monitor: позиция %s (#%s) пропущена", trade.ticker, trade.trade_id,
                )

        logger.info("monitor_and_close: закрыто %d позиций", len(closed))
        return closed

    def force_close_all(self, reason: str) -> None:
        """Аварийно закрывает все открытые позиции (по текущей цене)."""
        open_trades = self.db.query(Trade).filter(Trade.status == "OPEN").all()
        if not open_trades:
            logger.info("force_close_all: открытых позиций нет")
            return

        closed = 0
        for trade in open_trades:
            if trade.direction != "SHORT":
                continue
            try:
                price = self.get_current_price(trade.ticker) or trade.entry_price
                result = self._close_via_broker(
                    trade, f"force_close: {reason}", fallback_price=price,
                )
                if result:
                    closed += 1
            except Exception:
                logger.exception("force_close_all: %s (#%s) не закрыт",
                                 trade.ticker, trade.trade_id)

        logger.warning(
            "force_close_all (%s): закрыто %d/%d позиций",
            reason, closed, len(open_trades),
        )

    # === Триггеры выхода ===

    def _check_exit_reason(self, trade: Trade, price: float) -> Optional[str]:
        """Возвращает причину выхода или None, если триггеров нет.

        Триггеры для SHORT: стоп-лосс (цена >= стопа), тейк-профит
        (цена <= цели), лимит времени удержания, trailing stop. Стоп и цель
        берутся из полей сделки, а если не заданы — из настроек.
        """
        # Стоп-лосс: шорт убыточен при росте цены, стоп ВЫШЕ входа
        stop = trade.stop_loss_price
        if stop is None:
            stop = trade.entry_price * (1 + settings.STOP_LOSS_PERCENT / 100.0)
        if price >= stop:
            return f"stop_loss (цена {price:.2f} >= {stop:.2f})"

        # Тейк-профит: цель НИЖЕ входа
        target = trade.take_profit_price
        if target is None:
            target = trade.entry_price * (1 - settings.TAKE_PROFIT_PERCENT / 100.0)
        if price <= target:
            return f"take_profit (цена {price:.2f} <= {target:.2f})"

        # Лимит времени удержания
        if trade.opened_at and msk_now() - trade.opened_at > timedelta(
            hours=settings.MAX_HOLD_HOURS
        ):
            return f"time_stop (> {settings.MAX_HOLD_HOURS} ч удержания)"

        # Trailing stop: защита прибыли после движения в нашу пользу
        lowest = trade.lowest_price
        if lowest and trade.entry_price > 0:
            move_down_pct = (trade.entry_price - lowest) / trade.entry_price * 100.0
            if move_down_pct >= settings.TRAILING_STOP_ACTIVATION_PERCENT:
                trail_level = lowest * (
                    1 + settings.TRAILING_STOP_DISTANCE_PERCENT / 100.0
                )
                if price >= trail_level:
                    return (
                        f"trailing_stop (отскок с {lowest:.2f} до {price:.2f}, "
                        f"ход вниз {move_down_pct:.1f}%)"
                    )

        return None

    # === Закрытие ===

    def _close_via_broker(self, trade: Trade, reason: str,
                          fallback_price: float) -> Optional[dict]:
        """Закрывает позицию через брокера и, при подтверждении, в БД.

        БД закрывается только по исполненному ордеру (FILL/PARTIALLYFILL);
        exit_price берётся из фактической средней цены исполнения.
        Возвращает dict результата или None, если брокер не исполнил.
        """
        lot_size = self._lot_size(trade.ticker)
        lots = self._shares_to_lots(trade.quantity, lot_size)

        try:
            result = self.execution.close_short(trade.ticker, lots)
        except Exception as exc:
            logger.error("close_short %s (#%s) не удался: %s",
                         trade.ticker, trade.trade_id, exc)
            return None

        status = result.get("status", "")
        if status not in _FILLED_STATUSES:
            logger.warning(
                "close %s (#%s) не исполнен (status=%s): %s",
                trade.ticker, trade.trade_id, status, result.get("message", ""),
            )
            return None

        exit_price = result.get("average_price") or fallback_price
        return self._close_trade(
            trade,
            exit_price=exit_price,
            reason=reason,
            note=f"ордер {result.get('order_id', '')} status={status}",
        )

    def _close_externally(self, trade: Trade) -> None:
        """Закрывает запись trades, если позиции у брокера уже нет."""
        price = self.get_current_price(trade.ticker)
        if not price:
            price = trade.entry_price
            logger.warning(
                "sync: цена %s недоступна, закрываем по цене входа", trade.ticker,
            )
        self._close_trade(
            trade, exit_price=price, reason="внешнее закрытие",
            note="позиции нет у брокера",
        )

    def _close_trade(self, trade: Trade, exit_price: float, reason: str,
                     commission: float = 0.0, note: str = "") -> dict:
        """Закрывает позицию в БД через Trade.close() и возвращает результат."""
        trade.close(
            exit_price=exit_price,
            commission=commission,
            notes="; ".join(x for x in (reason, note) if x),
        )
        self.db.commit()
        result = {
            "trade_id": trade.trade_id,
            "ticker": trade.ticker,
            "direction": trade.direction,
            "quantity": trade.quantity,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "reason": reason,
            "pnl": trade.pnl,
            "pnl_percent": trade.pnl_percent,
            "duration_hours": trade.duration_hours,
            "is_profitable": trade.is_profitable,
        }
        logger.info(
            "позиция закрыта: %s (#%s) reason=%s pnl=%.2f (%.2f%%)",
            trade.ticker, trade.trade_id, reason, trade.pnl or 0.0,
            trade.pnl_percent or 0.0,
        )
        return result

    # === Приватные помощники ===

    def get_current_price(self, ticker: str) -> Optional[float]:
        """Текущая цена: живая через API; фолбэк — последняя свеча из БД."""
        figi = self._figi(ticker)
        if figi:
            price = get_last_price(figi)
            if price and price > 0:
                return price

        tf = next((x for x in ("1d", "1h", "15m") if x in settings.MARKET_TIMEFRAMES), "1d")
        try:
            candles = get_candles_from_db(self.db, ticker, tf, limit=1)
            if candles:
                return candles[-1].close
        except Exception:
            logger.warning("_current_price: свечи %s (%s) недоступны", ticker, tf)
        return None

    def _figi(self, ticker: str) -> Optional[str]:
        """FIGI по тикеру: из кэша Instrument, фолбэк — через execution."""
        inst = self.db.get(Instrument, ticker)
        if inst is not None and inst.figi:
            return inst.figi
        try:
            return self.execution.resolve_figi(ticker)
        except Exception:
            logger.warning("_figi: не удалось определить FIGI для %s", ticker)
            return None

    def _lot_size(self, ticker: str) -> int:
        """Размер лота: из кэша Instrument, фолбэк — через execution."""
        inst = self.db.get(Instrument, ticker)
        if inst is not None and inst.lot and inst.lot > 0:
            return inst.lot
        try:
            return self.execution.get_lot_size(ticker)
        except Exception:
            logger.warning("_lot_size: размер лота %s недоступен, принимаем 1", ticker)
            return 1

    def _shares_to_lots(self, shares: int, lot_size: int) -> int:
        """Акции -> целое число лотов (для close_short)."""
        lot = max(lot_size, 1)
        if shares % lot == 0:
            return shares // lot
        logger.warning(
            "shares=%d не кратно лоту %d — округляем вверх", shares, lot,
        )
        return max(1, math.ceil(shares / lot))
