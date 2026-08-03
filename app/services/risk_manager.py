"""
Риск-менеджмент для SHORT-позиций на T-Invest.

Решает два вопроса агента:
1. "Можно ли открыть шорт?" — уверенность модели, маржинальная доступность,
   лимиты по числу сделок/позиций и убытку за день.
2. "Сколько лотов?" — размер позиции считается от капитала и волатильности.

Ключевые решения:
- Размер позиции ограничен двумя независимыми способами, берём минимум:
    1) Нельный лимит: не более MAX_POSITION_SIZE_PERCENT% капитала в позиции.
    2) Волатильность (ATR): если ожидаемый стоп (2*ATR) шире фиксированного
       STOP_LOSS_PERCENT, позицию пропорционально уменьшаем, чтобы убыток при
       стопе не выходил за заложенную норму.
- Данные счёта берутся через существующие хелперы sandbox_account (build_client
  выбирает песочницу/боевой режим по settings.TINKOFF_SANDBOX). Стоимость
  портфеля запрашивается один раз за проверку — песочница нестабильна.
- Лимиты позиций и убытка читаются из settings (.env); константа модуля
  осталась только там, где настройки нет (число сделок за день).

Стиль: типизированный Python 3.11+, docstring, логирование и обработка ошибок
везде, как в остальных сервисах проекта.
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
from app.services.execution import get_lot_size, resolve_figi
from app.services.market_data import get_candles_from_db

logger = logging.getLogger(__name__)

# --- Внутренние лимиты ---
# Число сделок за день вынесено в константу модуля; лимиты числа позиций и
# реализованного убытка читаются из settings (MAX_OPEN_POSITIONS,
# DAILY_LOSS_LIMIT_PERCENT).
MAX_TRADES_PER_DAY = 10          # макс. шортов, открытых за день

# --- ATR: период и множитель для стопа ---
ATR_PERIOD = 14                  # период расчёта ATR
ATR_STOP_MULTIPLIER = 2.0        # стоп отступает от входа на 2*ATR

# Причина отказа в успешном случае (сигнал для агента: можно открывать)
_OK_REASON = "ok"


class RiskManager:
    """Проверки рисков и расчёт размера позиции для шортов на T-Invest."""

    def __init__(self, db: Session, account_id: str):
        self.db = db
        self.account_id = account_id

    # === Проверка "можно ли открыть шорт" ===

    def can_open_short(self, ticker: str, confidence: float,
                       current_price: float) -> tuple[bool, str, int]:
        """Проверяет, можно ли открыть шорт, и считает максимальное число лотов.

        Returns:
            tuple[bool, str, int]: (можно ли, причина отказа, макс. число лотов).
            При успехе причина — "ok", лотов — максимально допустимое число.
        """
        ticker = ticker.strip().upper()

        # 1) Уверенность модели
        if confidence < settings.CONFIDENCE_THRESHOLD:
            return (
                False,
                f"уверенность {confidence:.2f} ниже порога "
                f"{settings.CONFIDENCE_THRESHOLD:.2f}",
                0,
            )

        # 2) Маржинальная доступность
        if not self.is_ticker_shortable(ticker):
            return False, f"{ticker} недоступен для шорта", 0

        # 3) Дневные лимиты (число сделок, убыток, зависшие позиции)
        if not self.check_daily_limits():
            return False, "дневные лимиты исчерпаны", 0

        # 4) Лимит открытых позиций
        if not self.check_max_positions():
            return False, f"достигнут лимит открытых позиций ({settings.MAX_OPEN_POSITIONS})", 0

        # 5) Капитал
        equity = self.get_current_equity()
        if equity <= 0:
            return False, f"недостаточно средств (equity={equity:.2f})", 0

        # 6) Размер позиции в лотах
        try:
            lot_size = self.get_lot_size(ticker)
            atr = self._get_atr(ticker)
            max_lots = self.calculate_position_size(
                equity, current_price, atr, lot_size=lot_size,
            )
        except Exception as exc:
            logger.exception("can_open_short: не удалось рассчитать размер (%s)", ticker)
            return False, f"ошибка расчёта размера позиции: {exc}", 0

        if max_lots <= 0:
            return False, "размер позиции меньше одного лота", 0

        logger.info(
            "can_open_short: %s allowed, confidence=%.2f equity=%.2f "
            "price=%.2f atr=%s max_lots=%d",
            ticker, confidence, equity, current_price,
            f"{atr:.2f}" if atr else "n/a", max_lots,
        )
        return True, _OK_REASON, max_lots

    # === Размер позиции ===

    def calculate_position_size(self, equity: float, price: float,
                                atr: Optional[float] = None,
                                lot_size: int = 1) -> int:
        """Максимальное число лотов под шорт по заданному капиталу.

        Ограничения (берём минимум):
          * нельный лимит — не более MAX_POSITION_SIZE_PERCENT% капитала;
          * волатильность — если ATR-стоп (2*ATR) шире фиксированного
            STOP_LOSS_PERCENT, позиция масштабируется пропорционально, чтобы
            убыток при стопе не превышал заложенную норму.

        lot_size — размер лота в акциях (в T-Invest торгуют лотами). Значение
        по умолчанию 1 удобно для вызова без брокерских данных; can_open_short
        всегда передаёт реальный размер лота.
        """
        if equity <= 0 or price <= 0:
            return 0

        max_value = equity * settings.MAX_POSITION_SIZE_PERCENT / 100.0

        # Волатильность: если ATR-стоп шире фиксированного — уменьшаем позицию.
        # scale = 1, когда 2*ATR укладывается в STOP_LOSS_PERCENT, и <1 иначе.
        if atr and atr > 0:
            atr_stop_pct = ATR_STOP_MULTIPLIER * atr / price * 100.0
            stop_pct = max(settings.STOP_LOSS_PERCENT, atr_stop_pct)
            scale = settings.STOP_LOSS_PERCENT / stop_pct
            max_value *= scale
            logger.debug(
                "size: price=%.2f atr=%.2f atr_stop_pct=%.2f%% stop_pct=%.2f%% scale=%.3f",
                price, atr, atr_stop_pct, stop_pct, scale,
            )

        lots = math.floor(max_value / (price * max(lot_size, 1)))
        return max(lots, 0)

    # === Лимиты ===

    def check_daily_limits(self) -> bool:
        """Дневные лимиты: число сделок, реализованный убыток, зависшие позиции."""
        start_of_day = msk_now().replace(hour=0, minute=0, second=0, microsecond=0)

        # 1) Лимит сделок за день
        opened_today = (
            self.db.query(Trade)
            .filter(
                Trade.direction == "SHORT",
                Trade.opened_at >= start_of_day,
            )
            .count()
        )
        if opened_today >= MAX_TRADES_PER_DAY:
            logger.warning(
                "check_daily_limits: %d сделок за день (лимит %d)",
                opened_today, MAX_TRADES_PER_DAY,
            )
            return False

        # 2) Дневной лимит убытка по реализованному PnL закрытых сегодня сделок
        closed_today = (
            self.db.query(Trade)
            .filter(Trade.status == "CLOSED", Trade.closed_at >= start_of_day)
            .all()
        )
        if closed_today:
            realized_pnl = sum(t.pnl or 0.0 for t in closed_today)
            equity = self.get_current_equity()
            if equity > 0:
                max_loss = equity * settings.DAILY_LOSS_LIMIT_PERCENT / 100.0
                if realized_pnl <= -max_loss:
                    logger.warning(
                        "check_daily_limits: убыток дня %.2f превышает лимит %.2f",
                        realized_pnl, -max_loss,
                    )
                    return False

        # 3) Нет позиций, "зависших" дольше MAX_HOLD_HOURS — сначала их закрыть
        for trade in self.db.query(Trade).filter(Trade.status == "OPEN").all():
            if trade.opened_at and msk_now() - trade.opened_at > timedelta(
                hours=settings.MAX_HOLD_HOURS
            ):
                logger.warning(
                    "check_daily_limits: позиция #%d (%s) держится дольше %d ч "
                    "(открыта %s) — сначала закрыть",
                    trade.id, trade.ticker, settings.MAX_HOLD_HOURS, trade.opened_at,
                )
                return False

        return True

    def check_max_positions(self) -> bool:
        """Лимит одновременно открытых позиций (по нашей учётной записи в БД)."""
        open_count = self.get_open_trades_count()
        if open_count >= settings.MAX_OPEN_POSITIONS:
            logger.warning(
                "check_max_positions: открыто %d позиций (лимит %d)",
                open_count, settings.MAX_OPEN_POSITIONS,
            )
            return False
        return True

    def is_ticker_shortable(self, ticker: str) -> bool:
        """Заглушка: акции TQBR считаем доступными для шорта.

        Настоящий источник — маржинальный список T-Invest, тянется из API:
          * sdk.instruments.get_trading_status(figi=...) — статус торгов и
            доступность шортов по инструменту (short_enabled_flag в новых
            версиях SDK);
          * sdk.users.get_margin_attributes(account_id=...) — достаточность
            маржи (minimal_margin / funds_sufficiency_level) под заявку.
        Пока в песочнице ликвидные TQBR шортуются без ограничений, поэтому
        полагаемся на кэш инструмента: share на TQBR -> True, прочее -> False.
        Для неизвестных тикеров (нет в кэше) — пермиссивная заглушка True,
        чтобы не блокировать торговлю на первом же прогоне.
        """
        inst = self.db.get(Instrument, ticker)
        if inst is not None:
            return inst.instrument_type == "share" and inst.class_code == "TQBR"
        logger.debug("is_ticker_shortable: %s нет в кэше — считаем шортуемым", ticker)
        return True

    # === Данные счёта ===

    def get_current_equity(self) -> float:
        """Текущая стоимость портфеля (total_amount_portfolio) со счёта T-Invest."""
        try:
            portfolio = sandbox_account.get_portfolio(self.account_id)
            equity = portfolio.get("total_amount_portfolio", 0.0)
            if equity > 0:
                return equity
            logger.warning("get_portfolio вернул пустую стоимость: %s", portfolio)
        except Exception:
            logger.warning("get_portfolio не удался — фолбэк на get_positions",
                           exc_info=True)

        # Фолбэк: сумма свободных денег по валютам (без стоимости бумаг)
        try:
            positions = sandbox_account.get_positions(self.account_id)
            return sum(m["value"] for m in positions.get("money", []))
        except Exception:
            logger.exception("get_positions тоже не удался — капитал неизвестен")
            return 0.0

    def get_open_trades_count(self) -> int:
        """Число открытых позиций по нашей учётной записи (Trade.status == 'OPEN').

        Учёт ведём в собственной таблице trades — она отражает решения агента.
        Брокерский вид позиций (баланс < 0 у бумаги) доступен отдельно через
        sandbox_account.get_positions, если понадобится сверить.
        """
        return self.db.query(Trade).filter(Trade.status == "OPEN").count()

    # === Уровни для выхода (полезно агенту при открытии) ===

    def get_stop_levels(self, entry_price: float) -> tuple[float, float]:
        """Уровни для шорта: (стоп-лосс, тейк-профит) по настройкам.

        Шорт зарабатывает на падении: стоп-лосс ВЫШЕ цены входа, тейк-профит НИЖЕ.
        """
        stop_loss = entry_price * (1 + settings.STOP_LOSS_PERCENT / 100.0)
        take_profit = entry_price * (1 - settings.TAKE_PROFIT_PERCENT / 100.0)
        return round(stop_loss, 2), round(take_profit, 2)

    # === Приватные помощники ===

    def get_lot_size(self, ticker: str) -> int:
        """Размер лота инструмента: из кэша Instrument (без API), иначе execution."""
        inst = self.db.get(Instrument, ticker)
        if inst is not None and inst.lot and inst.lot > 0:
            return inst.lot
        return get_lot_size(resolve_figi(ticker))

    def _get_atr(self, ticker: str) -> Optional[float]:
        """ATR по свечам из БД (без вызова API). None, если свечей мало.

        Свечи берём по дневному таймфрейму, если он собирается, иначе — первому
        из settings.MARKET_TIMEFRAMES. Среднее истинного диапазона за ATR_PERIOD.
        """
        tf = next((x for x in ("1d", "1h", "15m") if x in settings.MARKET_TIMEFRAMES), "1d")
        try:
            candles = get_candles_from_db(self.db, ticker, tf, limit=ATR_PERIOD + 2)
        except Exception:
            logger.warning("_get_atr: не удалось прочитать свечи %s (%s)", ticker, tf)
            return None

        if len(candles) < ATR_PERIOD + 1:
            return None

        true_ranges: list[float] = []
        for i in range(1, len(candles)):
            high, low = candles[i].high, candles[i].low
            prev_close = candles[i - 1].close
            true_ranges.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))

        atr = sum(true_ranges[-ATR_PERIOD:]) / ATR_PERIOD
        return round(atr, 4)
