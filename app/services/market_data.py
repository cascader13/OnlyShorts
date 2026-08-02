"""
Рыночные данные через T-Invest SDK (t_tech.invest).

Загрузка японских свечей по отслеживаемым тикерам, идемпотентное сохранение
в таблицу candles (ключ ticker+timeframe+ts) и кэш метаданных инструментов.

Ключевые решения:
- Время храним как naive UTC (SQLite хранит DateTime без offset);
  на вход SDK отдаём timezone-aware UTC (иначе TypeError в get_intervals).
- Инкрементальный сбор: при наличии свечей в БД запрашиваем от
  (последний ts - 2 периода) — перекрытие позволяет перезаписать
  незавершённую свечу (is_complete=False) и её поздние правки.
- Первичный backfill по глубине HISTORY_DEPTH (один раз, ~225 вызовов).
"""

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import func

from t_tech.invest import Client, CandleInterval, InstrumentIdType
from t_tech.invest.utils import get_intervals

from app.core.config import settings
from app.models.market import Candle, Instrument

logger = logging.getLogger(__name__)

# --- Таймфреймы и глубина истории ---

TIMEFRAME_ENUMS: dict[str, CandleInterval] = {
    "1d": CandleInterval.CANDLE_INTERVAL_DAY,
    "1h": CandleInterval.CANDLE_INTERVAL_HOUR,
    "15m": CandleInterval.CANDLE_INTERVAL_15_MIN,
}

TIMEFRAME_LABELS: dict[str, str] = {
    "1d": "День",
    "1h": "Час",
    "15m": "15 минут",
}

INTERVAL_DURATION: dict[str, timedelta] = {
    "1d": timedelta(days=1),
    "1h": timedelta(hours=1),
    "15m": timedelta(minutes=15),
}

# Глубина первичной загрузки (если в БД ещё нет свечей по тикеру+ТФ)
HISTORY_DEPTH: dict[str, timedelta] = {
    "1d": timedelta(days=400),
    "1h": timedelta(days=90),
    "15m": timedelta(days=30),
}

DEFAULT_CLASS_CODE = "TQBR"
INSTRUMENT_CACHE_TTL = timedelta(hours=24)
# Пауза между запросами свечей, чтобы не превысить rate-limit
API_CALL_DELAY = 0.05


def _utcnow() -> datetime:
    """Наивное UTC-время."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _quotation_to_float(q) -> float:
    """Quotation(units, nano) -> float. Nano может быть отрицательным (-5e8)."""
    return float(q.units) + float(q.nano) / 1e9


def _to_naive_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Переводит timezone-aware datetime в naive UTC; naive оставляет как есть."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


# --- Подключение ---

def build_client() -> Client:
    """Создаёт синхронный Client к T-Invest (песочница или боевой режим).

    Адрес выбирается по settings.TINKOFF_SANDBOX: True — песочница,
    False — боевой API. Токен всегда берётся из settings.TINKOFF_TOKEN.
    """
    # Страховка: флаг уже выставляется в config.py при импорте, но канал
    # создаётся в Client.__init__, поэтому дублируем setdefault.
    os.environ.setdefault("SSL_TBANK_VERIFY", "true")
    target = (
        settings.TINKOFF_SANDBOX_ADDRESS
        if settings.TINKOFF_SANDBOX
        else settings.TINKOFF_LIVE_ADDRESS
    )
    return Client(settings.TINKOFF_TOKEN, target=target)


# --- Инструменты ---

def _instrument_obj_to_dict(ticker: str, inst) -> dict:
    """Полный Instrument (get_instrument_by) -> dict для кэша."""
    return {
        "ticker": ticker,
        "figi": inst.figi,
        "name": inst.name or ticker,
        "currency": inst.currency or None,
        "lot": int(inst.lot) or 1,
        "class_code": inst.class_code or DEFAULT_CLASS_CODE,
        "instrument_type": inst.instrument_type or None,
        "uid": inst.uid or None,
        "first_1day_candle_date": _to_naive_utc(inst.first_1day_candle_date),
    }


def _instrument_short_to_dict(ticker: str, item) -> dict:
    """InstrumentShort (find_instrument, фолбэк): нет currency/instrument_type."""
    return {
        "ticker": ticker,
        "figi": item.figi,
        "name": item.name or ticker,
        "currency": None,
        "lot": int(item.lot) or 1,
        "class_code": item.class_code or DEFAULT_CLASS_CODE,
        "instrument_type": None,
        "uid": item.uid or None,
        "first_1day_candle_date": _to_naive_utc(item.first_1day_candle_date),
    }


def _instrument_to_dict(inst: Instrument) -> dict:
    """Строка кэша -> dict."""
    return {
        "ticker": inst.ticker,
        "figi": inst.figi,
        "name": inst.name,
        "currency": inst.currency,
        "lot": inst.lot or 1,
        "class_code": inst.class_code,
        "instrument_type": inst.instrument_type,
        "uid": inst.uid,
        "first_1day_candle_date": inst.first_1day_candle_date,
    }


def resolve_instrument(client, db: Session, ticker: str) -> dict:
    """
    Метаданные инструмента по тикеру.

    Приоритет: свежий кэш -> get_instrument_by(TICKER, TQBR) ->
    find_instrument(query) -> устаревший кэш -> ошибка.
    """
    cached = db.get(Instrument, ticker)
    if cached is not None and _utcnow() - cached.updated_at < INSTRUMENT_CACHE_TTL:
        return _instrument_to_dict(cached)

    data = None

    # 1) Точный запрос по тикеру и площадке
    try:
        resp = client.instruments.get_instrument_by(
            id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_TICKER,
            class_code=DEFAULT_CLASS_CODE,
            id=ticker,
        )
        data = _instrument_obj_to_dict(ticker, resp.instrument)
    except Exception as e:
        logger.warning("T-Invest: get_instrument_by %s не удался (%s)", ticker, e)

    # 2) Фолбэк — поиск по имени/тикеру
    if data is None:
        try:
            resp = client.instruments.find_instrument(query=ticker)
            for item in resp.instruments:
                if item.ticker.lower() == ticker.lower():
                    data = _instrument_short_to_dict(ticker, item)
                    break
        except Exception as e:
            logger.warning("T-Invest: find_instrument %s не удался (%s)", ticker, e)

    # 3) Устаревший кэш — лучше, чем ничего
    if data is None and cached is not None:
        data = _instrument_to_dict(cached)

    if data is None:
        raise ValueError(f"Инструмент {ticker} не найден в T-Invest (песочница)")

    return data


def upsert_instrument(db: Session, ticker: str, data: dict) -> None:
    """Сохраняет/обновляет метаданные инструмента по PK ticker."""
    inst = db.get(Instrument, ticker)
    if inst is None:
        inst = Instrument(ticker=ticker)
        db.add(inst)
    inst.figi = data["figi"]
    inst.name = data["name"]
    inst.currency = data.get("currency")
    inst.lot = data.get("lot", 1)
    inst.class_code = data.get("class_code", DEFAULT_CLASS_CODE)
    inst.instrument_type = data.get("instrument_type")
    inst.uid = data.get("uid")
    inst.first_1day_candle_date = data.get("first_1day_candle_date")
    inst.updated_at = _utcnow()
    db.commit()


def delete_instrument(db: Session, ticker: str) -> None:
    """Удаляет инструмент из кэша (таблица instruments)."""
    inst = db.get(Instrument, ticker)
    if inst:
        db.delete(inst)
        db.commit()


# --- Свечи ---

def _to_candle_dict(c) -> dict:
    """HistoricCandle -> dict с naive UTC ts."""
    return {
        "ts": _to_naive_utc(c.time),
        "open": _quotation_to_float(c.open),
        "high": _quotation_to_float(c.high),
        "low": _quotation_to_float(c.low),
        "close": _quotation_to_float(c.close),
        "volume": float(c.volume),
        "is_complete": bool(c.is_complete),
    }


def fetch_candles(client, figi: str, timeframe: str,
                  from_: datetime, to: datetime) -> list[dict]:
    """
    Загружает свечи из API, разбивая диапазон по лимитам SDK (get_intervals).
    from_/to должны быть timezone-aware UTC. Возвращает список dict.
    """
    interval = TIMEFRAME_ENUMS[timeframe]
    out: list[dict] = []
    for chunk_from, chunk_to in get_intervals(interval, from_, to):
        try:
            resp = client.market_data.get_candles(
                figi=figi,
                from_=chunk_from,
                to=chunk_to,
                interval=interval,
            )
        except Exception:
            logger.exception(
                "T-Invest: get_candles %s (%s..%s) не удался",
                figi, chunk_from, chunk_to,
            )
            raise
        for c in resp.candles:
            out.append(_to_candle_dict(c))
        time.sleep(API_CALL_DELAY)
    return out


def upsert_candles(db: Session, ticker: str, figi: str, timeframe: str,
                   candles: list[dict]) -> int:
    """
    Портативный upsert (без диалект-специфичного SQL):
    существующие свечи обновляются, новые — вставляются.
    Возвращает число вставленных новых свечей.
    """
    if not candles:
        return 0

    # Дедуп внутри батча: последний элемент с тем же ts выигрывает
    deduped: dict = {}
    for c in candles:
        deduped[c["ts"]] = c
    candles = list(deduped.values())

    ts_list = [c["ts"] for c in candles]
    existing = {
        row[0]
        for row in db.query(Candle.ts).filter(
            Candle.ticker == ticker,
            Candle.timeframe == timeframe,
            Candle.ts.in_(ts_list),
        ).all()
    }

    new_count = 0
    for c in candles:
        if c["ts"] in existing:
            db.query(Candle).filter(
                Candle.ticker == ticker,
                Candle.timeframe == timeframe,
                Candle.ts == c["ts"],
            ).update({
                Candle.open: c["open"],
                Candle.high: c["high"],
                Candle.low: c["low"],
                Candle.close: c["close"],
                Candle.volume: c["volume"],
                Candle.is_complete: c["is_complete"],
            })
        else:
            db.add(Candle(
                ticker=ticker,
                figi=figi,
                timeframe=timeframe,
                ts=c["ts"],
                open=c["open"],
                high=c["high"],
                low=c["low"],
                close=c["close"],
                volume=c["volume"],
                is_complete=c["is_complete"],
            ))
            new_count += 1

    db.commit()
    return new_count


def get_last_ts(db: Session, ticker: str, timeframe: str) -> Optional[datetime]:
    """Последний ts свечи в БД (naive UTC) или None."""
    return db.query(func.max(Candle.ts)).filter(
        Candle.ticker == ticker,
        Candle.timeframe == timeframe,
    ).scalar()


def _collect_timeframe(db: Session, client, ticker: str, figi: str,
                       timeframe: str) -> int:
    """Инкрементальный сбор одного тикера и таймфрейма."""
    last_ts = get_last_ts(db, ticker, timeframe)
    now = datetime.now(timezone.utc)

    if last_ts is None:
        from_ = now - HISTORY_DEPTH[timeframe]
    else:
        # Перекрытие в 2 периода: перезапишем незавершённую и правки свечи
        from_ = last_ts - 2 * INTERVAL_DURATION[timeframe]
        if from_.tzinfo is None:
            from_ = from_.replace(tzinfo=timezone.utc)

    candles = fetch_candles(client, figi, timeframe, from_, now)
    return upsert_candles(db, ticker, figi, timeframe, candles)


# --- Оркестрация ---

def get_tracked_tickers(db: Session) -> list[str]:
    """Все отслеживаемые тикеры: из таблицы instruments + TRACKED_TICKERS из .env."""
    db_tickers = [row[0] for row in db.query(Instrument.ticker).all()]
    # Объединяем: тикеры из БД + тикеры из .env (на случай первый запуск без instruments)
    combined = list(dict.fromkeys(db_tickers + settings.TRACKED_TICKERS))
    return combined


def collect_market_data(db: Session, tickers: Optional[list[str]] = None) -> dict:
    """
    Один проход сбора свечей по всем тикерам и таймфреймам.
    Если tickers не задан — берём все из instruments + TRACKED_TICKERS.

    Returns:
        dict: {"SBER": {"1d": n, "1h": n}, ..., "total": N}, где N — число
        новых (вставленных) свечей за проход.
    """
    tickers = tickers or get_tracked_tickers(db)
    timeframes = [tf for tf in settings.MARKET_TIMEFRAMES if tf in TIMEFRAME_ENUMS]
    if not timeframes:
        logger.warning("MARKET_TIMEFRAMES не содержит известных таймфреймов: %s",
                       settings.MARKET_TIMEFRAMES)
        return {"total": 0}

    stats: dict = {ticker: {} for ticker in tickers}
    total = 0

    try:
        client = build_client()
    except Exception:
        logger.exception("T-Invest: не удалось создать клиент")
        return {"total": 0, "error": "не удалось создать клиент"}

    with client as sdk:
        for ticker in tickers:
            try:
                inst = resolve_instrument(sdk, db, ticker)
                upsert_instrument(db, ticker, inst)
                for tf in timeframes:
                    count = _collect_timeframe(db, sdk, ticker, inst["figi"], tf)
                    stats[ticker][tf] = count
                    total += count
            except Exception:
                # Тикер вне песочницы (напр. делистинг) не должен валить проход.
                # rollback важен: сбой flush "отравляет" сессию, иначе упадут
                # все последующие тикеры (PendingRollbackError).
                db.rollback()
                logger.exception("T-Invest: тикер %s пропущен", ticker)
                stats[ticker]["error"] = "ошибка сбора"

    stats["total"] = total
    logger.info("T-Invest: собрано свечей: %s", stats)
    return stats


# --- Чтение для дашборда ---

def get_candles_from_db(db: Session, ticker: str, timeframe: str,
                        limit: int = 120) -> list[Candle]:
    """Последние N свечей по тикеру и таймфрейму в хронологическом порядке."""
    rows = (
        db.query(Candle)
        .filter(Candle.ticker == ticker, Candle.timeframe == timeframe)
        .order_by(Candle.ts.desc())
        .limit(max(limit, 1))
        .all()
    )
    return list(reversed(rows))


def get_instruments(db: Session) -> list[Instrument]:
    """Все инструменты из кэша (для выпадашки в дашборде)."""
    return db.query(Instrument).order_by(Instrument.ticker).all()


def get_last_update_time(db: Session) -> Optional[datetime]:
    """Время последней вставки свечей (для подписи на дашборде)."""
    return db.query(func.max(Candle.created_at)).scalar()


# --- Поиск инструментов (для автодополнения) ---

def search_instruments(query: str) -> list[dict]:
    """
    Поиск инструментов в T-Invest API по запросу (find_instrument).
    Возвращает список dict{ticker, figi, name, class_code, lot}.
    Используется дашбордом для автодополнения; кэшируется через @st.cache_data.
    """
    if not query or len(query.strip()) < 1:
        return []
    try:
        client = build_client()
    except Exception:
        logger.warning("T-Invest: не удалось создать клиент для поиска")
        return []
    results: list[dict] = []
    with client as sdk:
        try:
            resp = sdk.instruments.find_instrument(query=query.strip())
            for item in resp.instruments:
                results.append({
                    "ticker": item.ticker,
                    "figi": item.figi,
                    "name": item.name or item.ticker,
                    "class_code": item.class_code or DEFAULT_CLASS_CODE,
                    "instrument_type": getattr(item, "instrument_type", None) or "",
                    "lot": int(item.lot) or 1,
                })
        except Exception:
            logger.exception("T-Invest: find_instrument('%s') не удался", query)
    # Сортировка: акции (TQBR) primeiro, потом остальные
    results.sort(key=lambda x: (0 if x["class_code"] == "TQBR" else 1, x["ticker"]))
    return results
