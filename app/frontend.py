"""
Streamlit-дашборд рыночных данных.

Запуск:
    streamlit run app/frontend.py

Показывает свечные графики по тикеру (SMA/объём/RSI) из таблицы candles и
ленту последних новостей по тикеру из raw_news. Кнопка «Обновить данные»
вызывает разовый сбор свечей через T-Invest SDK.

Построение графиков — в app/services/charting.py (чистая логика).
"""

import os
import sys

# Корень проекта — родительская папка app/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import logging

import streamlit as st
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db_context
from app.models.market import Instrument
from app.models.news import RawNews
from app.services.charting import (
    DARK_PALETTE,
    LIGHT_PALETTE,
    add_indicators,
    build_chart,
    candles_to_df,
)
from app.services.market_data import (
    TIMEFRAME_LABELS,
    collect_market_data,
    delete_instrument,
    get_candles_from_db,
    get_instruments,
    get_last_update_time,
    resolve_instrument,
    search_instruments,
    upsert_instrument,
    build_client,
)

logger = logging.getLogger(__name__)

st.set_page_config(page_title="PantsOnly: Рынок", layout="wide")

# --- Палитра (dataviz) в зависимости от темы Streamlit ---
_THEME = st.get_option("theme.base") or "light"
_PALETTE = DARK_PALETTE if _THEME == "dark" else LIGHT_PALETTE


def _news_for_ticker(db: Session, ticker: str, instrument_name: str = "",
                     limit: int = 10) -> list[RawNews]:
    """
    Новости по тикеру из raw_news.
    Ищем по тикеру И по названию компании (если задано).
    В raw_news нет колонки тикера — фильтруем по вхождению в заголовок/текст.
    """
    conditions = [
        RawNews.title.ilike(f"%{ticker}%"),
        RawNews.full_text.ilike(f"%{ticker}%"),
    ]
    # Добавляем поиск по названию компании (напр. "Роснефть" для ROSN)
    if instrument_name:
        # Берём первое слово из названия (обычно компания)
        name_part = instrument_name.split()[0] if instrument_name else ""
        if name_part and len(name_part) > 2:
            conditions.append(RawNews.title.ilike(f"%{name_part}%"))
            conditions.append(RawNews.full_text.ilike(f"%{name_part}%"))
    return (
        db.query(RawNews)
        .filter(or_(*conditions))
        .order_by(
            RawNews.published_at.is_(None),
            RawNews.published_at.desc(),
            RawNews.created_at.desc(),
        )
        .limit(limit)
        .all()
    )


# --- Данные для сайдбара ---

with get_db_context() as db:
    instruments = get_instruments(db)
    # Материализуем строки внутри сессии (иначе DetachedInstanceError)
    instrument_options = (
        [f"{i.ticker} — {i.name}" for i in instruments]
        if instruments
        else list(settings.TRACKED_TICKERS)
    )
    known_tickers = {i.ticker for i in instruments}
    last_update = get_last_update_time(db)

known_tfs = [tf for tf in settings.MARKET_TIMEFRAMES if tf in TIMEFRAME_LABELS]


@st.cache_data(ttl=3600, show_spinner=False)
def _search_cached(query: str) -> list[dict]:
    """Поиск инструментов в T-Invest с кэшированием (1 час)."""
    return search_instruments(query)


# --- Сайдбар ---

st.sidebar.title("PantsOnly: Рынок")

# 1) Выбор тикера: editable selectbox + ввод нового
ticker_label = st.sidebar.selectbox(
    "Тикер",
    instrument_options,
    accept_new_options=True,
    placeholder="Выберите или введите тикер...",
)
if " — " in ticker_label:
    ticker = ticker_label.split(" — ")[0].strip().upper()
else:
    ticker = ticker_label.strip().upper()

# Авто-резолв: если тикер новый (нет в кэше) — пытаемся найти в T-Invest
if ticker and ticker not in known_tickers:
    try:
        with get_db_context() as db:
            client = build_client()
            with client as sdk:
                data = resolve_instrument(sdk, db, ticker)
                upsert_instrument(db, ticker, data)
        st.sidebar.success(f"✅ {ticker} — {data.get('name', '')}")
        st.rerun()
    except ValueError:
        st.sidebar.warning(f"Тикер «{ticker}» не найден в T-Invest")
    except Exception as e:
        st.sidebar.warning(f"Ошибка поиска {ticker}: {e}")

# Удаление тикера (только для добавленных вручную, не из .env)
if ticker in known_tickers and ticker not in settings.TRACKED_TICKERS:
    if st.sidebar.button(f"🗑️ Удалить {ticker}", key="btn_delete_ticker"):
        with get_db_context() as db:
            delete_instrument(db, ticker)
        st.rerun()

# 2) Поиск инструментов (expander)
with st.sidebar.expander("🔍 Найти инструмент", expanded=False):
    search_q = st.text_input("Запрос", placeholder="Название или тикер...", key="search_q")
    if st.button("Найти", key="btn_search"):
        if search_q:
            results = _search_cached(search_q)
            if results:
                st.session_state["search_results"] = results
            else:
                st.session_state["search_results"] = []
                st.info("Ничего не найдено")
    if st.session_state.get("search_results"):
        for idx, item in enumerate(st.session_state["search_results"][:10]):
            col1, col2 = st.columns([5, 1])
            with col1:
                st.caption(f"**{item['ticker']}** — {item['name']}  (lot {item['lot']})")
            with col2:
                if st.button("➕", key=f"add_{idx}_{item['ticker']}_{item['figi']}"):
                    try:
                        with get_db_context() as db:
                            upsert_instrument(db, item["ticker"], item)
                        st.session_state.pop("search_results", None)
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

# 3) Таймфрейм и кол-во свечей
timeframe = st.sidebar.selectbox(
    "Таймфрейм",
    known_tfs if known_tfs else ["1d"],
    format_func=lambda tf: TIMEFRAME_LABELS.get(tf, tf),
)
candle_count = st.sidebar.slider("Последних свечей", 30, 500, 120)

# --- Кнопка обновления (разовый сбор через T-Invest) ---

if "update_msg" not in st.session_state:
    st.session_state.update_msg = None
    st.session_state.update_is_error = False

if st.sidebar.button("Обновить данные", type="primary", width="stretch"):
    with st.spinner(f"Загружаю свечи для {ticker}..."):
        try:
            with get_db_context() as db:
                # Собираем именно выбранный тикер (а не все TRACKED_TICKERS)
                stats = collect_market_data(db, tickers=[ticker])
            total = int(stats.get("total", 0))
            # Показываем детали по таймфреймам
            ticker_stats = stats.get(ticker, {})
            if isinstance(ticker_stats, dict) and "error" not in ticker_stats:
                parts = ", ".join(f"{tf}={n}" for tf, n in ticker_stats.items() if isinstance(n, int))
                st.session_state.update_msg = f"✅ {ticker}: {parts or '0 свечей'}"
            elif total > 0:
                st.session_state.update_msg = f"✅ {ticker}: {total} новых свечей"
            else:
                st.session_state.update_msg = f"⚠️ {ticker}: свечей не добавлено (возможно, нет данных в песочнице)"
            st.session_state.update_is_error = False
        except Exception as exc:
            logger.exception("Ошибка сбора рыночных данных")
            st.session_state.update_msg = f"❌ Ошибка сбора {ticker}: {exc}"
            st.session_state.update_is_error = True
    st.rerun()

if st.session_state.update_msg:
    if st.session_state.update_is_error:
        st.sidebar.error(st.session_state.update_msg)
    else:
        st.sidebar.success(st.session_state.update_msg)

if last_update is not None:
    st.sidebar.caption(
        f"Последнее обновление БД: {last_update:%d.%m.%Y %H:%M:%S} UTC"
    )
st.sidebar.caption("Фоновый сбор: каждые %d сек" % settings.COLLECT_INTERVAL_SECONDS)

# --- Основная часть ---

with get_db_context() as db:
    rows = get_candles_from_db(db, ticker, timeframe, candle_count)
    # Материализуем в DataFrame и плоские кортежи до закрытия сессии
    df_raw = candles_to_df(rows)
    df = add_indicators(df_raw) if not df_raw.empty else df_raw
    # Имя инструмента для поиска новостей (напр. "РОСНЕФТЬ" для ROSN)
    inst = db.get(Instrument, ticker)
    inst_name = inst.name if inst else ""
    news = [
        (n.source or "?", n.published_at, n.title)
        for n in _news_for_ticker(db, ticker, inst_name)
    ]

if df.empty:
    st.info(
        "Нет данных по этому тикеру/таймфрейму. "
        "Нажмите «Обновить данные» в боковой панели."
    )
    st.stop()

# Метрики
last_close = df["close"].iloc[-1]
first_open = df["open"].iloc[0]
period_change = (last_close - first_open) / first_open * 100 if first_open else 0.0
col1, col2, col3, col4 = st.columns(4)
col1.metric("Последняя цена", f"{last_close:,.2f} ₽",
            delta=f"{period_change:+.2f}% за период")
col2.metric("Максимум", f"{df['high'].max():,.2f} ₽")
col3.metric("Минимум", f"{df['low'].min():,.2f} ₽")
col4.metric("Суммарный объём", f"{df['volume'].sum():,.0f}")

# График
fig = build_chart(df, ticker, TIMEFRAME_LABELS.get(timeframe, timeframe), _PALETTE)
st.plotly_chart(fig, width="stretch")

# Новости по тикеру
st.subheader(f"Новости по {ticker}")
if news:
    for src, published_at, title in news:
        when = published_at.strftime("%d.%m.%Y %H:%M") if published_at else "—"
        st.markdown(f"- **[{src}]** ({when}) {title}")
else:
    st.caption("Новостей с этим тикером пока нет.")
