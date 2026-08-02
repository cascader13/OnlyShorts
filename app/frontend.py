"""
Streamlit-дашборд рыночных данных.

Запуск:
    streamlit run app/frontend.py

Вкладка «Рынок»: свечные графики по тикеру (SMA/объём/RSI) из таблицы candles и
лента последних новостей по тикеру из raw_news. Кнопка «Обновить данные»
вызывает разовый сбор свечей через T-Invest SDK.

Вкладка «Байесовские сети»: генерация сети по тикеру (технические агенты
RSI/SMA/волатильность + новости за 3 часа → LLM → JSON), построение модели
pgmpy из JSON и её визуализация (граф, CPD, апостериорная вероятность).

Построение графиков — в app/services/charting.py (чистая логика).
Построение байесовских сетей — в app/services/bayesian_network_viz.py.
"""

import os
import sys

# Корень проекта — родительская папка app/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json
import logging
from datetime import datetime, timezone, timedelta

import streamlit as st
from sqlalchemy import or_
from sqlalchemy.orm import Session

MSK = timezone(timedelta(hours=3))  # Московское время


def _to_msk(dt) -> str:
    """Конвертирует naive UTC datetime в строку МСК."""
    if dt is None:
        return "—"
    return dt.replace(tzinfo=timezone.utc).astimezone(MSK).strftime("%d.%m.%Y %H:%M")

from app.core.config import settings
from app.core.database import get_db_context
from app.models.market import Instrument
from app.models.news import RawNews, NewsArticle
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
from app.services.news_aggregator import aggregate_news
from app.services.bayesian_network import (
    get_saved_networks,
    run_bayesian_agent,
    save_network_result,
)
from app.services import bayesian_network_viz as bnv
from app.services.sandbox_account import (
    get_accounts,
    open_account,
    pay_in,
    get_positions,
    get_portfolio,
)
from app.models.decision import Decision

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


def _processed_news_for_ticker(db: Session, ticker: str, instrument_name: str = "",
                               limit: int = 15) -> list[NewsArticle]:
    """
    Обработанные новости (LLM) по тикеру с сентиментом.
    Ищет по primary_ticker И по заголовку/тексту.
    """
    conditions = [
        NewsArticle.primary_ticker.ilike(f"%{ticker}%"),
        NewsArticle.title.ilike(f"%{ticker}%"),
    ]
    if instrument_name:
        name_part = instrument_name.split()[0] if instrument_name else ""
        if name_part and len(name_part) > 2:
            conditions.append(NewsArticle.title.ilike(f"%{name_part}%"))
            conditions.append(NewsArticle.summary.ilike(f"%{name_part}%"))
    return (
        db.query(NewsArticle)
        .filter(or_(*conditions))
        .order_by(
            NewsArticle.published_at.is_(None),
            NewsArticle.published_at.desc(),
            NewsArticle.created_at.desc(),
        )
        .limit(limit)
        .all()
    )


def _show_network(structure, ticker: str, chart_key: str = "network"):
    """Отрисовывает сеть из JSON: граф (plotly), таблицы CPD, инференс, объяснение.

    chart_key — уникальный суффикс для plotly_chart. _show_network может
    вызываться на одном прогоне несколько раз (свежесгенерированная сеть +
    выбранная из сохранённых), и одинаковые графики без key дают
    Streamlit-ошибку «multiple plotly_chart elements».
    """
    if not structure:
        st.warning("Структура сети пуста.")
        return

    try:
        model, warnings = bnv.build_model_from_json(structure)
    except ValueError as exc:
        st.error(f"Не удалось собрать модель pgmpy: {exc}")
        # Граф структуры показываем даже если модель не собралась
        try:
            st.plotly_chart(bnv.build_figure(structure, _PALETTE), width="stretch",
                            key=f"bn_chart_{chart_key}_{ticker}")
        except Exception:
            st.caption("Граф не удалось построить.")
        return
    for w in warnings:
        st.warning(w)

    col1, col2 = st.columns([2, 1])
    with col1:
        try:
            fig = bnv.build_figure(structure, _PALETTE)
            st.plotly_chart(fig, width="stretch",
                            key=f"bn_chart_{chart_key}_{ticker}")
        except Exception as exc:
            st.error(f"Ошибка построения графа: {exc}")
    with col2:
        try:
            target = structure.get("target_variable", "Price_Change")
            _, probs = bnv.infer_target(model, target)
            st.markdown(f"**Апостериорная вероятность `{target}`**")
            for stt in ["Down", "Neutral", "Up"]:
                val = probs.get(stt, 0.0)
                emoji = "🔴" if stt == "Down" else ("🟢" if stt == "Up" else "⚪")
                st.markdown(f"{emoji} **{stt}**: {val:.1%}")
            action, best_state = bnv.action_from_probs(probs)
            hint = " — нет чёткого перевеса" if best_state == "Neutral" else ""
            st.caption(f"Действие по сети: **{action}**{hint}")
        except Exception as exc:
            st.caption(f"Инференс недоступен: {exc}")

    st.markdown(f"**Объяснение:** {structure.get('explanation', '—')}")

    with st.expander("Исходный JSON сети", expanded=False):
        st.code(json.dumps(structure, ensure_ascii=False, indent=2),
                language="json")

    with st.expander("Таблицы CPD", expanded=False):
        for var, df_cpd in bnv.cpd_tables(model):
            st.markdown(f"**{var}**")
            st.dataframe(df_cpd, width="stretch")


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
        f"Последнее обновление БД: {_to_msk(last_update)} МСК"
    )

# Текущее время МСК
now_msk = datetime.now(MSK)
st.sidebar.caption(f"Текущее время: {now_msk:%H:%M:%S} МСК")
st.sidebar.caption("Фоновый сбор: каждые %d сек" % settings.COLLECT_INTERVAL_SECONDS)

# --- Вкладки ---

tab_market, tab_bayes, tab_account = st.tabs(
    ["📈 Рынок", "🕸️ Байесовские сети", "💰 Счёт и портфель"]
)

with tab_market:
    with get_db_context() as db:
        rows = get_candles_from_db(db, ticker, timeframe, candle_count)
        # Материализуем в DataFrame и плоские кортежи до закрытия сессии
        df_raw = candles_to_df(rows)
        df = add_indicators(df_raw) if not df_raw.empty else df_raw
        # Имя инструмента для поиска новостей (напр. "РОСНЕФТЬ" для ROSN)
        inst = db.get(Instrument, ticker)
        inst_name = inst.name if inst else ""
        raw_news = [
            (n.source or "?", n.published_at, n.title)
            for n in _news_for_ticker(db, ticker, inst_name)
        ]
        news_agg = aggregate_news(db, ticker, hours=3)
        processed_news = [
            {
                "source": n.source or "?",
                "published_at": n.published_at,
                "title": n.title,
                "summary": n.summary or "",
                "sentiment_score": n.sentiment_score,
                "sentiment_label": n.sentiment_label or "neutral",
                "tickers": n.tickers or "",
                "tags": n.tags or "",
            }
            for n in _processed_news_for_ticker(db, ticker, inst_name)
        ]

    if df.empty:
        st.info(
            "Нет данных по этому тикеру/таймфрейму. "
            "Нажмите «Обновить данные» в боковой панели."
        )
    else:
        last_close = df["close"].iloc[-1]
        first_open = df["open"].iloc[0]
        period_change = (last_close - first_open) / first_open * 100 if first_open else 0.0
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Последняя цена", f"{last_close:,.2f} ₽",
                    delta=f"{period_change:+.2f}% за период")
        col2.metric("Максимум", f"{df['high'].max():,.2f} ₽")
        col3.metric("Минимум", f"{df['low'].min():,.2f} ₽")
        col4.metric("Суммарный объём", f"{df['volume'].sum():,.0f}")

        fig = build_chart(df, ticker, TIMEFRAME_LABELS.get(timeframe, timeframe), _PALETTE)
        st.plotly_chart(fig, width="stretch")

        st.subheader(f"Новости по {ticker}")

        if news_agg.total_news > 0:
            if news_agg.avg_sentiment > 0.3:
                agg_color = "#0ca30c"
                agg_emoji = "🟢"
            elif news_agg.avg_sentiment < -0.3:
                agg_color = "#d03b3b"
                agg_emoji = "🔴"
            else:
                agg_color = "#52514e"
                agg_emoji = "⚪"

            st.markdown(
                f"**Агрегация за {news_agg.period_hours}ч:** "
                f"{news_agg.total_news} новостей | "
                f"Сентимент: <span style='color:{agg_color}'>{news_agg.avg_sentiment:+.2f}</span> | "
                f"Достоверность: {news_agg.avg_confidence:.0%} | "
                f"Позитив: {news_agg.positive_count} | "
                f"Негатив: {news_agg.negative_count} | "
                f"Нейтрально: {news_agg.neutral_count}",
                unsafe_allow_html=True,
            )
            if news_agg.dominant_signal != "none":
                signal_color = "#d03b3b" if news_agg.dominant_signal == "short" else "#0ca30c"
                st.markdown(
                    f":arrow_lower_right: **Сигнал: "
                    f"<span style='color:{signal_color}'>{news_agg.dominant_signal.upper()}</span>** "
                    f"(сила: {news_agg.signal_strength:.0%})",
                    unsafe_allow_html=True,
                )
            st.divider()
        else:
            st.caption(f"Обработанных новостей за {news_agg.period_hours}ч нет.")

        if processed_news:
            st.markdown("**Обработанные новости (LLM):**")
            for item in processed_news:
                score = item["sentiment_score"]
                when = _to_msk(item["published_at"])

                if score is not None and score > 0.3:
                    color = "#0ca30c"  # зелёный (позитив)
                    emoji = "🟢"
                elif score is not None and score < -0.3:
                    color = "#d03b3b"  # красный (негатив)
                    emoji = "🔴"
                else:
                    color = "#52514e"  # серый (нейтральный)
                    emoji = "⚪"

                score_str = f"{score:+.2f}" if score is not None else "—"
                confidence_str = (f"({item['sentiment_score']:.0%})"
                                  if item.get("sentiment_confidence") else "")

                st.markdown(
                    f":{emoji[0]}- <span style='color:{color}'>**[{item['source']}]** "
                    f"({when} МСК) {item['title']} "
                    f"[{score_str} {confidence_str}]</span>",
                    unsafe_allow_html=True,
                )
                if item["summary"]:
                    st.caption(f"  {item['summary'][:150]}...")
        elif raw_news:
            # Фолбэк: сырые новости (если LLM ещё не обработал)
            st.caption("Обработанных новостей пока нет. Показываю сырые:")
            for src, published_at, title in raw_news:
                when = _to_msk(published_at)
                st.markdown(f"- **[{src}]** ({when} МСК) {title}")
        else:
            st.caption("Новостей с этим тикером пока нет.")

with tab_bayes:
    st.subheader("🕸️ Байесовские сети (pgmpy)")
    st.caption(
        "Генерация сети: технические агенты (RSI/SMA/волатильность) + новости "
        "за 3 часа → LLM → JSON → модель pgmpy."
    )

    bay_tickers = list(dict.fromkeys(list(known_tickers) + list(settings.TRACKED_TICKERS)))
    bay_ticker = st.selectbox(
        "Тикер",
        bay_tickers,
        key="bay_ticker",
        accept_new_options=True,
        placeholder="Выберите или введите тикер...",
    ).strip().upper()

    if st.button("🚀 Сгенерировать сеть", type="primary", key="btn_gen_bayes"):
        with st.spinner(f"Собираю данные и запрашиваю LLM для {bay_ticker}..."):
            with get_db_context() as db:
                result = run_bayesian_agent(db, bay_ticker)
                saved_id = save_network_result(db, result) if result.get("valid") else None

        if result.get("status") == "error":
            st.error(f"Ошибка LLM: {result.get('error')}")
        elif not result.get("valid"):
            st.error("JSON не валиден:")
            for e in result.get("errors", []):
                st.error(f"  • {e}")
        else:
            st.success(f"✅ Сеть сгенерирована и сохранена (id={saved_id})")
            for w in result.get("warnings", []):
                st.warning(w)
            _show_network(result.get("json"), bay_ticker, chart_key="generated")

    st.divider()
    st.markdown("**Сохранённые сети**")

    view_all = st.checkbox(
        "Показывать сети всех тикеров",
        value=False,
        key="bay_view_all",
        help="По умолчанию — только сети выбранного выше тикера.",
    )
    # Материализуем строки в dict ДО закрытия сессии (иначе DetachedInstanceError
    # при чтении атрибутов после выхода из with). Действие пересчитываем по
    # сохранённым вероятностям (вдруг сохранено со старой логикой Up-vs-Down).
    with get_db_context() as db:
        saved = []
        for d in get_saved_networks(
            db, ticker=None if view_all else bay_ticker, limit=30
        ):
            action = d.action
            infer = d.bayesian_inference_result or {}
            probs = infer.get("probabilities") if isinstance(infer, dict) else None
            if probs:
                action, _ = bnv.action_from_probs(probs)
            saved.append({
                "label": (
                    f"{d.created_at:%d.%m %H:%M} • {d.ticker} • {action} • "
                    f"conf {d.confidence:.2f} • {d.bayesian_visualization or ''}"
                ),
                "ticker": d.ticker,
                "structure": d.bayesian_network_structure,
            })

    if not saved:
        scope = "всех тикеров" if view_all else f"«{bay_ticker}»"
        st.caption(f"Сохранённых сетей {scope} нет. Сгенерируйте первую.")
    else:
        options = {row["label"]: row for row in saved}
        sel_label = st.selectbox("Выбрать сохранённую сеть", list(options.keys()), key="bay_saved")
        sel = options[sel_label]
        _show_network(sel["structure"], sel["ticker"], chart_key="saved")


with tab_account:
    st.subheader("💰 Счёт и портфель (песочница T-Invest)")
    st.caption(
        "Баланс и пополнение — счёт песочницы. Справа — текущие позиции "
        "(шорты выделяются) и прошедшие шорты со всеми деталями и ссылками "
        "на байесовские сети."
    )

    # --- Аккаунт песочницы ---
    try:
        accounts = get_accounts()
    except Exception as exc:
        st.error(f"Не удалось получить аккаунты песочницы: {exc}")
        accounts = []

    if not accounts:
        st.info("Аккаунт песочницы ещё не открыт.")
        if st.button("🔓 Открыть аккаунт", type="primary", key="btn_open_sandbox"):
            with st.spinner("Открываю аккаунт в песочнице..."):
                try:
                    open_account(name="PantsOnly")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Ошибка открытия аккаунта: {exc}")
    else:
        acc_options = {
            f"{a['name']} ({a['account_id'][:8]}…)": a["account_id"] for a in accounts
        }
        sel_acc = st.selectbox("Аккаунт песочницы", list(acc_options.keys()))
        account_id = acc_options[sel_acc]

        # Портфель и позиции запрашиваем один раз (песочница нестабильна: 6
        # ретраев на вызов), данные уже плоские dict — можно использовать ниже.
        try:
            pf = get_portfolio(account_id)
            ps = get_positions(account_id)
        except Exception as exc:
            st.error(f"Ошибка получения портфеля: {exc}")
            pf, ps = None, None

        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.markdown("**Баланс и пополнение**")
            if pf:
                total = pf["total_amount_portfolio"]
                yield_total = pf["expected_yield"]
                st.metric("Стоимость портфеля", f"{total:,.2f} ₽")
                st.metric("Доходность", f"{yield_total:+,.2f} ₽")
                if ps and ps["money"]:
                    free = {m["currency"]: m["value"] for m in ps["money"]}
                    st.caption("Доступно: " + ", ".join(
                        f"{v:,.2f} {c.upper()}" for c, v in free.items()
                    ))

            st.divider()
            st.markdown("**Пополнение**")
            amount = st.number_input("Сумма пополнения, ₽",
                                     min_value=0.0, value=100000.0, step=10000.0)
            if st.button("💳 Пополнить", type="primary", key="btn_pay_in"):
                with st.spinner("Пополняю счёт..."):
                    try:
                        res = pay_in(account_id, int(amount))
                        st.success(f"Баланс после пополнения: {res['balance']:,.2f} {res['currency'].upper()}")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Ошибка пополнения: {exc}")

        with col_right:
            st.markdown("**Портфель**")
            positions = pf["positions"] if pf else []
            if not positions:
                st.caption("Открытых позиций нет.")
            else:
                for p in positions:
                    qty = p["quantity"]
                    is_short = qty < 0
                    badge = "🔴 SHORT" if is_short else "🟢 LONG"
                    avg = p["average_position_price"]
                    cur = p["current_price"]
                    yld = p["expected_yield"]
                    st.markdown(
                        f"**{p['ticker']}** {badge} "
                        f"qty={abs(qty):g} avg={avg:,.2f} cur={cur:,.2f} "
                        f"PnL={yld:+,.2f} ₽"
                    )
                    # Ссылка на байесовскую сеть по этому тикеру.
                    # Материализуем поля в dict ДО закрытия сессии
                    # (иначе DetachedInstanceError при чтении атрибутов).
                    with get_db_context() as db:
                        dec = (
                            db.query(Decision)
                            .filter(Decision.ticker == p["ticker"])
                            .order_by(Decision.created_at.desc())
                            .first()
                        )
                        if dec:
                            dec_info = {
                                "created_at": dec.created_at,
                                "confidence": dec.confidence,
                                "structure": dec.bayesian_network_structure,
                            }
                        else:
                            dec_info = None
                    if dec_info and dec_info["structure"]:
                        with st.expander(
                            f"🕸️ Байесовская сеть {p['ticker']} "
                            f"({dec_info['created_at']:%d.%m %H:%M}, "
                            f"conf {dec_info['confidence']:.2f})",
                            expanded=False,
                        ):
                            _show_network(dec_info["structure"], p["ticker"],
                                          chart_key=f"pos_{p['ticker']}")

            st.divider()
            st.markdown("**Прошедшие шорты**")
            # Материализуем в dicts внутри сессии, поля читаются уже после неё
            with get_db_context() as db:
                past_shorts = [
                    {
                        "ticker": d.ticker,
                        "created_at": d.created_at,
                        "confidence": d.confidence,
                        "probability_down": d.probability_down,
                        "probability_up": d.probability_up,
                        "structure": d.bayesian_network_structure,
                        "id": d.id,
                    }
                    for d in (
                        db.query(Decision)
                        .filter(Decision.action == "SHORT")
                        .order_by(Decision.created_at.desc())
                        .limit(30)
                        .all()
                    )
                ]
            if not past_shorts:
                st.caption("Прошедших шортов пока нет.")
            else:
                for d in past_shorts:
                    pd = d["probability_down"]
                    pu = d["probability_up"]
                    pd_s = f"{pd:.2f}" if pd is not None else "—"
                    pu_s = f"{pu:.2f}" if pu is not None else "—"
                    st.markdown(
                        f"**{d['ticker']}** • {d['created_at']:%d.%m %H:%M} МСК • "
                        f"conf={d['confidence']:.2f} • P(down)={pd_s} "
                        f"P(up)={pu_s}"
                    )
                    if d["structure"]:
                        with st.expander(
                            f"🕸️ Сеть {d['ticker']} ({d['created_at']:%d.%m %H:%M})",
                            expanded=False,
                        ):
                            _show_network(d["structure"], d["ticker"],
                                          chart_key=f"past_{d['ticker']}_{d['id']}")
