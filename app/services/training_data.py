"""
Генерация датасета для обучения модели шортов.

Каждая строка TrainingSnapshot — момент времени T с признаками, известными
на T (техника по свечам 15m as-of T + новостной сентимент за окно
[T-window_hours, T)), и целевой переменной target = 1, если путь цены после
T, прогнанный через РОВНО те же правила выхода, что исполняет агент
(PositionManager: стоп-лосс, тейк-профит, trailing stop, лимит времени
MAX_HOLD_HOURS), закрыл шорт в плюс. Метка и исполнение живут в одной
стратегии — модель обучается на том, что агент реально сделает (вариант A
path-simulation, см. simulate_short_exit).

Дополнительно сохраняются: реализованный P&L (target_return_pct), причина
выхода (exit_reason), длительность (sim_duration_hours), MAE/MFE (mae_pct,
mfe_pct), горизонт (max_hold_hours) и сами параметры правил выхода
(label_stop_loss_pct / label_take_profit_pct / label_trail_*). Метка
самодокументирующая: label_version кодирует «рецепт» (path_sim_sl5_tp3_...),
так что после смены процентов и --relabel по БД видно, по каким параметрам
размечена каждая строка. Старые строки со старой меткой «close(T+N) <
price» отличаются (label_version = NULL) и переразмечаются через --relabel.

Единый конвейер «снимок -> отложенная разметка»:
- backfill()      — исторические снапшоты за период; target ставится сразу,
                    если будущие свечи уже есть, иначе NULL;
- capture_live()  — снимает слепок на текущий час (1 строка/час/тикер),
                    target = NULL;
- label_expired() — заполняет target у строк, у которых прошёл горизонт
                    симуляции и появились будущие свечи (и live, и хвост
                    backfill);
- relabel()       — переразмечает ВСЕ строки по текущим правилам (--relabel);
- refresh_news_features() — пересчитывает новостные колонки снапшотов, чьё
                    окно публикаций накрыло новости, которые LLM обработал
                    уже после снятия слепка (атрибуция — по published_at,
                    поэтому запоздавший анализ «догоняется» постфактум).

Идемпотентность: уникальный ключ (ticker, timestamp, window_hours,
lookahead_hours) — повторные запуски не плодят дубли; пересчёт новостных
колонок тоже идемпотентен (считается по всем статьям окна).

Запуск:
    python -m app.services.training_data --backfill --days 14
    python -m app.services.training_data --label
    python -m app.services.training_data --relabel   # переразметка path-симуляцией
    python -m app.services.training_data --refresh   # пересчёт новостных колонок
    python -m app.services.training_data --stats
"""

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import and_, func, or_
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db_context
from app.core.timeutil import msk_now
from app.models.market import Candle
from app.models.news import NewsArticle, ticker_matches
from app.models.snapshot import TrainingSnapshot
from app.services.agents.rsi import RSIAgent
from app.services.agents.sma import SMAAgent
from app.services.agents.volatility import VolatilityAgent
from app.services.charting import candles_to_df

logger = logging.getLogger(__name__)

# Индикаторам нужно >= 50 свечей (SMA50) + текущая
MIN_CANDLES = 51

# Снапшот берём только при «свежих» свечах: если последняя свеча старее этого
# окна (ночь/выходные), признаков как-of T нет — строку пропускаем.
MAX_CANDLE_AGE = timedelta(hours=2)

_INTERVAL_OFFSET = timedelta(minutes=1)


def _safe_float(value) -> Optional[float]:
    """float без NaN/Inf -> None."""
    try:
        if value is None:
            return None
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):  # NaN
        return None
    return round(f, 6)


# --- Свечи ---------------------------------------------------------------

def _candles_asof(
    db: Session, ticker: str, at: datetime, timeframe: str, limit: int
) -> list[Candle]:
    """Свечи по тикеру с ts <= at в хронологическом порядке."""
    rows = (
        db.query(Candle)
        .filter(
            Candle.ticker == ticker,
            Candle.timeframe == timeframe,
            Candle.ts <= at,
        )
        .order_by(Candle.ts.desc())
        .limit(limit)
        .all()
    )
    return list(reversed(rows))


def _close_asof(db: Session, ticker: str, at: datetime, timeframe: str) -> Optional[float]:
    """Цена закрытия последней свечи с ts <= at."""
    row = (
        db.query(Candle.close)
        .filter(
            Candle.ticker == ticker,
            Candle.timeframe == timeframe,
            Candle.ts <= at,
        )
        .order_by(Candle.ts.desc())
        .first()
    )
    return row[0] if row else None


# --- Новости -------------------------------------------------------------

def _in_publication_window(start: datetime, end: datetime):
    """SQL-условие: статья в окне [start, end) по ВРЕМЕНИ ПУБЛИКАЦИИ.

    Новости атрибутируются к моменту публикации, а не обработки LLM. Иначе
    задержка анализа «переносит» новость в более поздний снапшот: модель
    выучит ложную связку «в T был негатив -> цена падала T..T+N», хотя
    негатив вышел раньше (look-ahead в обучающих данных). Обратная сторона —
    слепок, снятый до обработки новости, её не увидит: это чинит
    refresh_news_features(), пересчитывая колонки постфактум.

    Статьи без published_at в окно не попадают: приписать их к моменту
    нельзя, а ошибочная привязка хуже пропуска.
    """
    pub = NewsArticle.published_at
    return and_(
        pub.isnot(None),
        pub >= start,
        pub < end,
    )


def _news_stats(
    db: Session, ticker: str, start: datetime, end: datetime
) -> tuple[float, float, int]:
    """
    Агрегирует новости тикера в окне [start, end).

    Возвращает (avg_sentiment, avg_confidence, count). Фильтр и способ поиска
    тикера — как в news_aggregator.aggregate_news, но по published_at
    (см. _in_publication_window).
    """
    articles = (
        db.query(NewsArticle.sentiment_score, NewsArticle.sentiment_confidence)
        .filter(
            or_(
                ticker_matches(NewsArticle.primary_ticker, ticker),
                ticker_matches(NewsArticle.tickers, ticker),
                NewsArticle.title.ilike(f"%{ticker}%"),
            ),
            _in_publication_window(start, end),
            NewsArticle.sentiment_score.isnot(None),
        )
        .all()
    )
    if not articles:
        return 0.0, 0.0, 0

    scores = [a[0] for a in articles if a[0] is not None]
    confs = [a[1] for a in articles if a[1] is not None]
    avg_sent = sum(scores) / len(scores) if scores else 0.0
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    return round(avg_sent, 4), round(avg_conf, 4), len(articles)


def _news_features(
    db: Session, ticker: str, at: datetime, window_hours: int
) -> dict:
    """Новостные признаки на момент at.

    Текущее окно [at-w, at) и предыдущее [at-2w, at-w) для сентимент-дрейфа.
    Единая логика для compute_features (снятие слепка) и
    refresh_news_features (пересчёт уже записанных снапшотов) — обе обязаны
    давать одинаковые значения.
    """
    start = at - timedelta(hours=window_hours)
    prev_start = at - timedelta(hours=2 * window_hours)
    avg_now, avg_conf_now, count_now = _news_stats(db, ticker, start, at)
    avg_prev, _, _ = _news_stats(db, ticker, prev_start, start)
    sentiment_change = round(avg_now - avg_prev, 4) if count_now else 0.0
    return {
        "news_sentiment_avg": round(avg_now, 4),
        "news_count": count_now,
        "news_confidence_avg": avg_conf_now,
        "sentiment_change_2h": sentiment_change,
    }


# --- Признаки ------------------------------------------------------------

def compute_features(
    db: Session,
    ticker: str,
    at: datetime,
    timeframe: str = "15m",
    window_hours: int = 2,
) -> Optional[dict]:
    """
    Признаки, известные на момент at.

    Технические — по свечам <= at (RSI14, SMA20/50, ATR-волатильность),
    новостные — по сентименту в окне [at-window_hours, at).

    Returns:
        dict признаков или None, если свечей недостаточно (< MIN_CANDLES)
        или они устарели (рынок закрыт).
    """
    candles = _candles_asof(db, ticker, at, timeframe, MIN_CANDLES)
    if len(candles) < MIN_CANDLES:
        logger.debug(
            "%s: %d свечей <= %s < %d — пропускаем",
            ticker, len(candles), at, MIN_CANDLES,
        )
        return None

    last_ts = candles[-1].ts
    if at - last_ts > MAX_CANDLE_AGE:
        logger.debug(
            "%s: свежих свечей нет (последняя %s, снапшот %s) — пропускаем",
            ticker, last_ts, at,
        )
        return None

    df = candles_to_df(candles)

    rsi = _safe_float(RSIAgent().calculate(df).iloc[-1])
    sma_fast, sma_slow = SMAAgent().calculate(df)
    sma20 = _safe_float(sma_fast.iloc[-1])
    sma50 = _safe_float(sma_slow.iloc[-1])
    price = _safe_float(df["close"].iloc[-1])

    price_vs_sma20 = (
        _safe_float((price / sma20 - 1) * 100) if price is not None and sma20 else None
    )
    price_vs_sma50 = (
        _safe_float((price / sma50 - 1) * 100) if price is not None and sma50 else None
    )

    atr = _safe_float(VolatilityAgent().calculate_atr(df).iloc[-1])
    volatility = _safe_float(atr / price * 100) if atr is not None and price else None
    volume = _safe_float(df["volume"].iloc[-1])

    # Новостные признаки: окно [at-w, at) + предыдущее для сентимент-дрейфа
    news = _news_features(db, ticker, at, window_hours)

    return {
        "rsi": rsi,
        "price": price,
        "sma_20": sma20,
        "sma_50": sma50,
        "price_vs_sma_20": price_vs_sma20,
        "price_vs_sma_50": price_vs_sma50,
        "volatility": volatility,
        "volume": volume,
        **news,
    }


# --- Симуляция выхода (path-simulation, вариант A) -----------------------

# Причины выхода — те же, что PositionManager пишет в notes закрытой сделки.
REASON_STOP_LOSS = "stop_loss"
REASON_TAKE_PROFIT = "take_profit"
REASON_TRAILING_STOP = "trailing_stop"
REASON_TIME_STOP = "time_stop"


@dataclass(frozen=True)
class ExitParams:
    """Параметры правил выхода, которыми размечена строка (самодокументация).

    Метка кодирует ровно эти значения: изменение любого параметра даёт другую
    метку. version() — компактный «рецепт» в label_version, чтобы в БД было
    видно, по каким параметрам размечена каждая строка (и её можно было
    отличить после смены настроек и повторного --relabel).
    """

    stop_loss_pct: float
    take_profit_pct: float
    trail_activation_pct: float
    trail_distance_pct: float
    max_hold_hours: float

    @classmethod
    def from_settings(cls) -> "ExitParams":
        """Параметры из текущего settings (читаются в момент разметки)."""
        return cls(
            stop_loss_pct=settings.STOP_LOSS_PERCENT,
            take_profit_pct=settings.TAKE_PROFIT_PERCENT,
            trail_activation_pct=settings.TRAILING_STOP_ACTIVATION_PERCENT,
            trail_distance_pct=settings.TRAILING_STOP_DISTANCE_PERCENT,
            max_hold_hours=float(settings.MAX_HOLD_HOURS),
        )

    def version(self) -> str:
        """Рецепт метки, напр. path_sim_sl5_tp3_ta2_td1.5_h4 (без нулей)."""
        g = lambda v: f"{v:g}"  # noqa: E731 — короткий формат 5.0 -> "5"
        return (
            f"path_sim_sl{g(self.stop_loss_pct)}_tp{g(self.take_profit_pct)}"
            f"_ta{g(self.trail_activation_pct)}_td{g(self.trail_distance_pct)}"
            f"_h{g(self.max_hold_hours)}"
        )


@dataclass
class SimExit:
    """Результат path-симуляции шорта по свечам (без комиссий и слиппеда)."""

    exit_price: float
    pnl_pct: float          # (entry - exit) / entry * 100; плюс = прибыль
    reason: str             # одна из REASON_*
    duration_hours: float
    mae_pct: float          # худшая точка пути, % от входа (>=0 для шорта = рост)
    mfe_pct: float          # лучшая точка пути, % от входа (<=0 для шорта = падение)


def _timeframe_minutes(timeframe: str) -> int:
    """'15m' -> 15, '1h' -> 60, '1d' -> 1440. Незнакомое значение -> 15."""
    s = (timeframe or "").strip().lower()
    mult = {"m": 1, "h": 60, "d": 1440}
    if len(s) < 2 or s[-1] not in mult:
        return 15
    try:
        return int(s[:-1]) * mult[s[-1]]
    except ValueError:
        return 15


def _sim_exit(
    entry_price: float,
    exit_price: float,
    reason: str,
    entry_ts: datetime,
    exit_ts: datetime,
    lowest: float,
    highest: float,
) -> SimExit:
    """Собирает SimExit из цены входа/выхода и экстремумов пути."""
    pnl_pct = (entry_price - exit_price) / entry_price * 100.0
    duration = (exit_ts - entry_ts).total_seconds() / 3600.0
    return SimExit(
        exit_price=exit_price,
        pnl_pct=pnl_pct,
        reason=reason,
        duration_hours=duration,
        mae_pct=(highest - entry_price) / entry_price * 100.0,
        mfe_pct=(lowest - entry_price) / entry_price * 100.0,
    )


def simulate_short_exit(
    entry_price: float,
    entry_ts: datetime,
    candles: list[Candle],
    stop_loss_pct: float,
    take_profit_pct: float,
    max_hold_hours: float,
    trail_activation_pct: float,
    trail_distance_pct: float,
    timeframe_minutes: int,
) -> Optional[SimExit]:
    """Прогоняет свечной путь (entry_ts, entry_ts+max_hold_hours] через те же
    правила выхода, что исполняет PositionManager, и возвращает симуляцию
    шорта. Вход — по entry_price (цена снапшота), выход — по уровню триггера.

    Триггеры и порядок — как в PositionManager._check_exit_reason:
    stop_loss (цена >= стопа), take_profit (цена <= цели), trailing stop,
    time_stop (лимит времени). Шорт убыточен при росте цены, поэтому стоп
    выше входа, цель ниже.

    Допущения (задокументированы, чтобы метка не казалась точнее, чем есть):
    * Порядок движения внутри одной свечи неизвестен. Для шорта консервативно
      считаем неблагоприятный ход (стоп) более ранним, чем благоприятный
      (тейк) — это не завышает долю прибыльных меток.
    * Исполнение по триггеру: тейк — лимитная заявка (min(open, цель) при
      гэпе вниз), стоп и трейлинг — стоп-рынок (max(open, уровень) при гэпе
      вверх). Реальный слиппед стоп-ордера не моделируем.
    * Trailing активируется минимумом с учётом текущей свечи и срабатывает по
      её максимуму (предполагаем низ -> отскок в пределах свечи).
    * Момент ценового выхода — середина свечи (длительность — оценка).
    * Граничная свеча на ts == горизонт в триггерах не участвует: по времени
      позиция закрывается в её открытии (цена ровно в момент лимита).
    * Если рынок закрылся раньше горизонта и триггер не сработал — позиция
      ушла бы в overnight, исход неизвестен: возвращаем None, метку не ставим.

    Args:
        candles: свечи в хронологическом порядке, ts в (entry_ts, horizon].

    Returns:
        SimExit или None при неполном окне (рынок закрыт до горизонта).
    """
    if not candles or entry_price <= 0:
        return None

    horizon = entry_ts + timedelta(hours=max_hold_hours)
    stop_level = entry_price * (1 + stop_loss_pct / 100.0)
    target_level = entry_price * (1 - take_profit_pct / 100.0)
    half_tf = timedelta(minutes=timeframe_minutes // 2)

    lowest = entry_price
    highest = entry_price
    last_trigger: Optional[Candle] = None
    boundary: Optional[Candle] = None

    for c in candles:
        if c.ts >= horizon:
            boundary = c
            break
        last_trigger = c

        # 1) Стоп-лосс: рост неблагоприятен, проверяем первым (по максимуму).
        if c.high >= stop_level:
            return _sim_exit(
                entry_price, max(c.open, stop_level), REASON_STOP_LOSS,
                entry_ts, c.ts + half_tf, lowest, max(highest, c.high),
            )
        # 2) Тейк-профит: цель ниже входа (по минимуму).
        if c.low <= target_level:
            return _sim_exit(
                entry_price, min(c.open, target_level), REASON_TAKE_PROFIT,
                entry_ts, c.ts + half_tf, min(lowest, c.low), highest,
            )
        # 3) Trailing stop: отскок от минимума после хода в нашу пользу.
        new_low = min(lowest, c.low)
        if (entry_price - new_low) / entry_price * 100.0 >= trail_activation_pct:
            trail_level = new_low * (1 + trail_distance_pct / 100.0)
            if c.high >= trail_level:
                return _sim_exit(
                    entry_price, max(c.open, trail_level), REASON_TRAILING_STOP,
                    entry_ts, c.ts + half_tf, new_low, max(highest, c.high),
                )
        lowest = new_low
        highest = max(highest, c.high)

    # Окно исчерпано без ценового триггера.
    if boundary is not None:
        # Рынок был открыт на горизонте — выходим по времени в открытии
        # граничной свечи (цена ровно в момент лимита удержания).
        return _sim_exit(
            entry_price, boundary.open, REASON_TIME_STOP,
            entry_ts, horizon, lowest, highest,
        )
    if last_trigger is not None and (
        last_trigger.ts + timedelta(minutes=timeframe_minutes) >= horizon
    ):
        # Последняя свеча покрывает горизонт — цена на его конце ≈ её close.
        return _sim_exit(
            entry_price, last_trigger.close, REASON_TIME_STOP,
            entry_ts, last_trigger.ts + timedelta(minutes=timeframe_minutes),
            lowest, highest,
        )
    return None  # рынок закрылся раньше горизонта — overnight, метку не ставим


# --- Целевая переменная --------------------------------------------------

def compute_target(
    db: Session,
    ticker: str,
    at: datetime,
    entry_price: Optional[float],
    lookahead_hours: int = 2,
    timeframe: str = "15m",
    params: Optional[ExitParams] = None,
) -> Optional[tuple[int, SimExit, ExitParams]]:
    """Метка по path-симуляции шорта на пути (at, at + N].

    Прогоняет свечи после at через те же правила выхода, что агент
    (см. simulate_short_exit), и метит по реализованному P&L: target = 1,
    если симуляция закрылась в плюс. Горизонт и уровни — из params
    (по умолчанию текущие settings); lookahead_hours из строки на симуляцию
    не влияет (оставлен для совместимости вызова и уникального ключа).

    Returns:
        (target, SimExit, params) или None, если будущие свечи ещё не собраны
        (горизонт в будущем) либо окно неполное (рынок закрылся раньше
        горизонта). В таком случае строка останется target=NULL и будет
        размечена label_expired позже.
    """
    if entry_price is None or entry_price <= 0:
        return None

    params = params or ExitParams.from_settings()
    horizon = at + timedelta(hours=params.max_hold_hours)
    if horizon > msk_now():
        return None  # будущие свечи ещё не собраны

    candles = (
        db.query(Candle)
        .filter(
            Candle.ticker == ticker,
            Candle.timeframe == timeframe,
            Candle.ts > at,
            Candle.ts <= horizon,
        )
        .order_by(Candle.ts.asc())
        .all()
    )
    sim = simulate_short_exit(
        entry_price=entry_price,
        entry_ts=at,
        candles=list(candles),
        stop_loss_pct=params.stop_loss_pct,
        take_profit_pct=params.take_profit_pct,
        max_hold_hours=params.max_hold_hours,
        trail_activation_pct=params.trail_activation_pct,
        trail_distance_pct=params.trail_distance_pct,
        timeframe_minutes=_timeframe_minutes(timeframe),
    )
    if sim is None:
        return None

    target = 1 if sim.pnl_pct > 0 else 0
    return target, sim, params


# --- Сборка и сохранение -------------------------------------------------

def _row_exists(
    db: Session,
    ticker: str,
    at: datetime,
    window_hours: int,
    lookahead_hours: int,
) -> bool:
    """Проверка уникального ключа (ticker, timestamp, window, lookahead)."""
    return (
        db.query(TrainingSnapshot.id)
        .filter(
            TrainingSnapshot.ticker == ticker,
            TrainingSnapshot.timestamp == at,
            TrainingSnapshot.window_hours == window_hours,
            TrainingSnapshot.lookahead_hours == lookahead_hours,
        )
        .first()
        is not None
    )


def _build_snapshot(
    db: Session,
    features: dict,
    ticker: str,
    at: datetime,
    window_hours: int,
    lookahead_hours: int,
    source: str,
) -> TrainingSnapshot:
    """Создаёт строку TrainingSnapshot из признаков, target ставим отдельно."""
    return TrainingSnapshot(
        timestamp=at,
        ticker=ticker,
        rsi=features.get("rsi"),
        price=features.get("price"),
        sma_20=features.get("sma_20"),
        sma_50=features.get("sma_50"),
        price_vs_sma_20=features.get("price_vs_sma_20"),
        price_vs_sma_50=features.get("price_vs_sma_50"),
        volatility=features.get("volatility"),
        volume=features.get("volume"),
        news_sentiment_avg=features.get("news_sentiment_avg"),
        news_count=features.get("news_count", 0),
        news_confidence_avg=features.get("news_confidence_avg"),
        sentiment_change_2h=features.get("sentiment_change_2h"),
        window_hours=window_hours,
        lookahead_hours=lookahead_hours,
        source=source,
    )


def _clear_label(snap: TrainingSnapshot) -> None:
    """Сбрасывает колонки метки (для переразметки и при неполном окне)."""
    snap.target = None
    snap.target_return_pct = None
    snap.exit_reason = None
    snap.sim_exit_price = None
    snap.sim_duration_hours = None
    snap.mae_pct = None
    snap.mfe_pct = None
    snap.max_hold_hours = None
    snap.label_stop_loss_pct = None
    snap.label_take_profit_pct = None
    snap.label_trail_activation_pct = None
    snap.label_trail_distance_pct = None
    snap.label_version = None


def _label_snapshot(
    db: Session,
    snap: TrainingSnapshot,
    timeframe: str,
) -> bool:
    """Размечает строку path-симуляцией. True, если target заполнен.

    При неполном окне метка сбрасывается — так --relabel стирает старые
    правила разметки (label_version=NULL), а не оставляет их «висеть».
    """
    result = compute_target(
        db, snap.ticker, snap.timestamp, snap.price,
        snap.lookahead_hours, timeframe,
    )
    if result is None:
        _clear_label(snap)
        return False
    target, sim, params = result
    snap.target = target
    snap.target_return_pct = round(sim.pnl_pct, 4)
    snap.exit_reason = sim.reason
    snap.sim_exit_price = round(sim.exit_price, 6)
    snap.sim_duration_hours = round(sim.duration_hours, 4)
    snap.mae_pct = round(sim.mae_pct, 4)
    snap.mfe_pct = round(sim.mfe_pct, 4)
    snap.max_hold_hours = params.max_hold_hours
    snap.label_stop_loss_pct = params.stop_loss_pct
    snap.label_take_profit_pct = params.take_profit_pct
    snap.label_trail_activation_pct = params.trail_activation_pct
    snap.label_trail_distance_pct = params.trail_distance_pct
    snap.label_version = params.version()
    snap.labeled_at = msk_now()
    return True


# --- Публичный API -------------------------------------------------------

def backfill(
    db: Session,
    tickers: list[str],
    start: datetime,
    end: datetime,
    timeframe: str = "15m",
    window_hours: int = 2,
    lookahead_hours: int = 2,
    step_minutes: int = 60,
) -> int:
    """
    Исторические снапшоты за [start, end] с шагом step_minutes.

    target проставляется сразу, если будущие свечи уже собраны; иначе NULL —
    дозаполнит label_expired(). Идемпотентен по уникальному ключу.
    """
    if start >= end:
        logger.warning("backfill: start %s >= end %s — ничего не делаем", start, end)
        return 0

    step = timedelta(minutes=max(step_minutes, 1))
    created = 0
    skipped = 0
    # Выравниваем старт на целый час — иначе между запусками
    # timestamps сдвигаются на секунды и _row_exists не находит дубль.
    t = start.replace(minute=0, second=0, microsecond=0)
    if t < start:
        t += timedelta(hours=1)
    while t <= end:
        for ticker in tickers:
            if _row_exists(db, ticker, t, window_hours, lookahead_hours):
                skipped += 1
                continue
            features = compute_features(db, ticker, t, timeframe, window_hours)
            if features is None:
                continue
            snap = _build_snapshot(
                db, features, ticker, t, window_hours, lookahead_hours, "backfill"
            )
            _label_snapshot(db, snap, timeframe)  # может и не размечиться
            db.add(snap)
            created += 1
            if created % 100 == 0:
                db.commit()
        t += step

    db.commit()
    logger.info(
        "backfill: создано %d снапшотов, пропущено существующих %d",
        created, skipped,
    )
    return created


def capture_live(
    db: Session,
    tickers: Optional[list[str]] = None,
    timeframe: str = "15m",
    window_hours: int = 2,
    lookahead_hours: int = 2,
) -> int:
    """
    Снимает live-слепок на текущий час по каждому тикеру.

    Цикл сбора запускается каждые ~5 минут, но благодаря бакету в начало часа
    в таблицу идёт 1 строка/час/тикер. Строка пишется с target=NULL, разметит
    её label_expired() через lookahead_hours.
    """
    now = msk_now()
    bucket = now.replace(minute=0, second=0, microsecond=0)
    tickers = tickers or settings.TRACKED_TICKERS

    captured = 0
    for ticker in tickers:
        if _row_exists(db, ticker, bucket, window_hours, lookahead_hours):
            continue
        features = compute_features(db, ticker, bucket, timeframe, window_hours)
        if features is None:
            continue  # нет свежих свечей — ночь/выходные, пропускаем
        db.add(_build_snapshot(
            db, features, ticker, bucket, window_hours, lookahead_hours, "live"
        ))
        captured += 1

    db.commit()
    if captured:
        logger.info("capture_live: %d новых снапшотов", captured)
    return captured


def label_expired(db: Session, timeframe: str = "15m") -> int:
    """Размечает строки с target=NULL, у которых прошёл горизонт симуляции
    и появились будущие свечи. Идемпотентно.

    Метке нужен полный путь до лимита удержания стратегии (MAX_HOLD_HOURS),
    поэтому пробуем не раньше, чем прошёл этот горизонт — даже если старый
    lookahead_hours строки меньше.
    """
    now = msk_now()
    rows = db.query(TrainingSnapshot).filter(TrainingSnapshot.target.is_(None)).all()
    labeled = 0
    for snap in rows:
        wait_hours = max(snap.lookahead_hours or 0, settings.MAX_HOLD_HOURS)
        if snap.timestamp > now - timedelta(hours=wait_hours):
            continue  # полный путь до горизонта ещё не собран
        if _label_snapshot(db, snap, timeframe):
            labeled += 1
    db.commit()
    if labeled:
        logger.info("label_expired: размечено %d снапшотов", labeled)
    return labeled


def relabel(db: Session, timeframe: str = "15m") -> int:
    """Переразмечает ВСЕ строки текущими правилами (path-симуляция).

    Отличие от label_expired: перезаписывает уже размеченные строки, в т.ч.
    со старым правилом «close(T+N) < price» (label_version=NULL). Строки с
    неполным окном (рынок закрылся раньше горизонта) сбрасываются в
    target=NULL. Запускать после изменения правил выхода или для миграции
    старого датасета: python -m app.services.training_data --relabel.
    """
    rows = db.query(TrainingSnapshot).all()
    relabeled = 0
    for snap in rows:
        if _label_snapshot(db, snap, timeframe):
            relabeled += 1
    db.commit()
    logger.info("relabel: переразмечено %d снапшотов", relabeled)
    return relabeled


def _refresh_snapshot_news(
    db: Session, snap: TrainingSnapshot, window_hours: int
) -> None:
    """Пересчитывает новостные колонки снапшота (окно по его timestamp).

    Полный пересчёт по всем статьям окна — идемпотентен.
    """
    w = snap.window_hours or window_hours
    news = _news_features(db, snap.ticker, snap.timestamp, w)
    snap.news_sentiment_avg = news["news_sentiment_avg"]
    snap.news_count = news["news_count"]
    snap.news_confidence_avg = news["news_confidence_avg"]
    snap.sentiment_change_2h = news["sentiment_change_2h"]


def _article_tickers(article: NewsArticle) -> list[str]:
    """Тикеры статьи для поиска затронутых снапшотов (primary + CSV).

    Снапшот по тикеру T считается затронутым, если статья попала в его окно:
    _news_stats матчит и primary_ticker, и CSV tickers, поэтому берём оба.
    """
    tickers: list[str] = []
    if article.primary_ticker:
        tickers.append(article.primary_ticker)
    if article.tickers:
        tickers.extend(
            t.strip().upper() for t in article.tickers.split(",") if t.strip()
        )
    seen: set[str] = set()
    unique: list[str] = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def refresh_news_features(
    db: Session,
    since: Optional[datetime] = None,
    window_hours: int = 2,
) -> int:
    """
    Пересчитывает новостные колонки снапшотов, затронутых новыми статьями.

    Аналог label_expired для новостных признаков. Снапшот снят в момент T, но
    часть новостей его окна [T - window_hours, T) LLM проанализировал позже
    (NewsArticle.created_at > T): на момент снятия слепка их ещё не было в
    БД, и колонки неполные. Здесь находим статьи, обработанные после `since`,
    и для каждой пересчитываем колонки снапшотов того же тикера, чьё окно
    публикаций накрыло published_at (P < T <= P + W). Так слепок «догоняет»
    новости, вышедшие в его окне, постфактум — без look-ahead, т.к. новость
    была публичной уже на момент T.

    Пересчёт полный (по всем статьям окна) и потому идемпотентный: повторный
    запуск даёт те же значения. since=None — полный рефреш по всем статьям
    (одноразовая «миграция» уже записанных снапшотов, CLI --refresh).

    Returns:
        int: число пересчитанных снапшотов.
    """
    q = db.query(NewsArticle).filter(NewsArticle.published_at.isnot(None))
    if since is not None:
        q = q.filter(NewsArticle.created_at >= since)
    articles = q.all()
    if not articles:
        return 0

    # Верхняя граница поиска кандидатов — максимальное окно среди снапшотов.
    max_win = db.query(func.max(TrainingSnapshot.window_hours)).scalar() or window_hours

    refreshed = 0
    seen_ids: set[int] = set()
    for article in articles:
        published = article.published_at
        hi = published + timedelta(hours=max_win)
        for ticker in _article_tickers(article):
            candidates = (
                db.query(TrainingSnapshot)
                .filter(
                    TrainingSnapshot.ticker == ticker,
                    TrainingSnapshot.timestamp > published,
                    TrainingSnapshot.timestamp <= hi,
                )
                .all()
            )
            for snap in candidates:
                w = snap.window_hours or window_hours
                if not (snap.timestamp - timedelta(hours=w) <= published < snap.timestamp):
                    continue  # окно этого снапшота статью не накрыло
                if snap.id in seen_ids:
                    continue
                seen_ids.add(snap.id)
                _refresh_snapshot_news(db, snap, window_hours)
                refreshed += 1

    db.commit()
    if refreshed:
        logger.info("refresh_news_features: пересчитано %d снапшотов", refreshed)
    return refreshed


def get_snapshot_stats(db: Session) -> dict:
    """Сводка по датасету для контроля выборки."""
    rows = db.query(TrainingSnapshot).all()
    total = len(rows)
    labeled = [r for r in rows if r.target is not None]
    returns = [r.target_return_pct for r in labeled if r.target_return_pct is not None]
    wins = sum(1 for r in labeled if r.target == 1)
    return {
        "total": total,
        "labeled": len(labeled),
        "target_1": wins,
        "target_0": len(labeled) - wins,
        "win_rate": round(wins / len(labeled), 4) if labeled else None,
        "avg_return_pct": round(sum(returns) / len(returns), 4) if returns else None,
        "unlabeled": total - len(labeled),
        "by_exit_reason": {
            reason: sum(1 for r in labeled if r.exit_reason == reason)
            for reason in sorted({r.exit_reason for r in labeled if r.exit_reason})
        },
        "by_label_version": {
            ver: sum(1 for r in rows if r.label_version == ver)
            for ver in sorted({r.label_version for r in rows if r.label_version})
        },
        "by_source": {
            src: sum(1 for r in rows if r.source == src)
            for src in sorted({r.source for r in rows if r.source})
        },
        "by_ticker": {
            t: sum(1 for r in rows if r.ticker == t)
            for t in sorted({r.ticker for r in rows})
        },
    }


# --- CLI: python -m app.services.training_data ---------------------------

def _parse_tickers(value: Optional[str]) -> list[str]:
    if not value:
        return settings.TRACKED_TICKERS
    return [t.strip().upper() for t in value.split(",") if t.strip()]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Снапшоты для обучения модели шортов"
    )
    parser.add_argument("--backfill", action="store_true", help="Исторические снапшоты")
    parser.add_argument("--days", type=int, default=14, help="Глубина бэкфилла в днях")
    parser.add_argument("--tickers", type=str, default=None, help="SBER,GAZP (по умолчанию TRACKED_TICKERS)")
    parser.add_argument("--timeframe", type=str, default=settings.SNAPSHOT_TIMEFRAME)
    parser.add_argument("--window", type=int, default=settings.NEWS_WINDOW_HOURS)
    parser.add_argument("--lookahead", type=int, default=settings.TARGET_HOURS)
    parser.add_argument("--step-min", type=int, default=settings.SNAPSHOT_STEP_MINUTES)
    parser.add_argument("--label", action="store_true", help="Разметить истёкшие")
    parser.add_argument(
        "--relabel", action="store_true",
        help="Переразметить ВСЕ строки path-симуляцией (перезаписывает target)",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Пересчитать новостные колонки затронутых снапшотов (по всем статьям)",
    )
    parser.add_argument("--stats", action="store_true", help="Сводка по датасету")

    args = parser.parse_args()

    with get_db_context() as db:
        if args.backfill:
            now = msk_now()
            created = backfill(
                db,
                tickers=_parse_tickers(args.tickers),
                start=now - timedelta(days=args.days),
                end=now,
                timeframe=args.timeframe,
                window_hours=args.window,
                lookahead_hours=args.lookahead,
                step_minutes=args.step_min,
            )
            print(f"Создано снапшотов: {created}")

        if args.label:
            labeled = label_expired(db, timeframe=args.timeframe)
            print(f"Размечено: {labeled}")

        if args.relabel:
            relabeled = relabel(db, timeframe=args.timeframe)
            print(f"Переразмечено: {relabeled}")

        if args.refresh:
            refreshed = refresh_news_features(
                db, since=None, window_hours=args.window
            )
            print(f"Пересчитано снапшотов: {refreshed}")

        if args.stats or (
            not args.backfill and not args.label and not args.relabel and not args.refresh
        ):
            stats = get_snapshot_stats(db)
            print("Датасет:")
            for k, v in stats.items():
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
