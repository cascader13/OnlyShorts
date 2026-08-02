"""
Байесовский агент: технические индикаторы + новости → LLM → байесовская сеть.

Поток одного тикера:
1. Собирает данные: свечи из БД (1d, фолбэк 1h) и новости за последние 3 часа
   (средний сентимент и достоверность).
2. Запускает агентов технических индикаторов (RSI, SMA, волатильность)
   из app/services/agents.
3. Строит читаемый контекст для LLM: результаты агентов + новостной фон.
4. Отправляет системный промпт (app/prompts/bayesian_network_building) и контекст
   в LLM через app/services/llm.ask_llm.
5. Валидирует JSON-ответ: структура (variables/edges/target_variable/explanation),
   три состояния на переменную, полные CPD (1 строка без родителей, 3^k с k
   родителями), строки CPD суммируются в ~1.0. Полнота CPD нужна, чтобы pgmpy
   мог собрать модель (см. app/services/bayesian_network_viz.py).

Запуск вручную:
    python -m app.services.bayesian_network --ticker SBER [--show-context] [--show-json]
"""

import argparse
import json
import logging
import re
import sys
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db_context
from app.services.llm import ask_llm
from app.services.agents import RSIAgent, SMAAgent, VolatilityAgent
from app.services.charting import candles_to_df
from app.services.market_data import get_candles_from_db, get_tracked_tickers
from app.services.news_aggregator import aggregate_news

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "bayesian_network_building"

# Таймфреймы для свечей в порядке предпочтения (1d лучше всего описывает тренд).
FALLBACK_TIMEFRAMES = ("1d", "1h")

# Минимум свечей для осмысленной работы индикаторов (SMA50 + запас).
MIN_CANDLES = 30

# Период новостей, который отправляем в LLM (часы).
NEWS_PERIOD_HOURS = 3

MSK = timezone(timedelta(hours=3))  # Московское время для читаемых времен в промпте


def load_system_prompt() -> str:
    """Загружает системный промпт для построения байесовской сети."""
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


# --- Вспомогательные форматтеры ---

def _fmt(v, nd: int = 2) -> str:
    """Число -> строка; NaN/None -> '—' (чтобы не пороть NaN в читаемый промпт)."""
    if v is None:
        return "—"
    try:
        if v != v:  # NaN
            return "—"
    except TypeError:
        pass
    return f"{v:.{nd}f}"


def _fmt_pct(v, nd: int = 2) -> str:
    """Число со знаком как проценты: +5.73% / -5.73%."""
    s = _fmt(v, nd)
    if s == "—":
        return "—"
    return ("+" if v >= 0 else "") + s + "%"


def _plain(v):
    """numpy-значение -> JSON-совместимое: float, bool, None (для NaN)."""
    try:
        if v != v:  # NaN
            return None
    except TypeError:
        pass
    if isinstance(v, bool):
        return bool(v)
    if isinstance(v, (int, float)):
        return v
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def _yes_no(flag: bool) -> str:
    return "yes" if flag else "no"


def _to_msk(dt) -> str:
    """naive UTC datetime -> строка МСК."""
    if dt is None:
        return "—"
    return dt.replace(tzinfo=timezone.utc).astimezone(MSK).strftime("%d.%m %H:%M")


# --- Сбор данных ---

def _load_candles(db: Session, ticker: str, limit: int = 150):
    """
    Загружает свечи для индикаторов.

    Пробует таймфреймы из FALLBACK_TIMEFRAMES (1d → 1h), берёт первый, где
    набралось MIN_CANDLES свечей. Возвращает (DataFrame, timeframe) или
    (None, None), если данных недостаточно.
    """
    timeframes = list(dict.fromkeys(
        tf for tf in (settings.BAYESIAN_TIMEFRAME,) + FALLBACK_TIMEFRAMES
        if tf
    ))
    for tf in timeframes:
        rows = get_candles_from_db(db, ticker, tf, limit)
        if len(rows) < MIN_CANDLES:
            logger.info("%s: %s свечей по %s < %d, пробуем следующий таймфрейм",
                        ticker, len(rows), tf, MIN_CANDLES)
            continue
        df = candles_to_df(rows)
        return df, tf
    logger.warning("%s: недостаточно свечей (%s)", ticker, dict.fromkeys(timeframes))
    return None, None


def _agents_text(db: Session, ticker: str) -> tuple[str, dict]:
    """
    Запускает агентов технических индикаторов и возвращает
    (читаемый текст для промпта, структурированные данные).
    """
    df, timeframe = _load_candles(db, ticker)
    if df is None:
        note = (
            f"Not enough candle data (need >= {MIN_CANDLES} candles on "
            f"{settings.BAYESIAN_TIMEFRAME} or 1h)."
        )
        return note, {"timeframe": None, "note": note}

    rsi = RSIAgent().analyze(df)
    sma = SMAAgent().analyze(df)
    vol = VolatilityAgent().analyze(df)

    rsi_label = {
        "overbought": "overbought (>70)",
        "oversold": "oversold (<30)",
        "neutral": "neutral zone (30–70)",
    }.get(rsi.signal, rsi.signal)

    sma_trend = {"bullish": "bullish (SMA20 > SMA50)",
                 "bearish": "bearish (SMA20 < SMA50)",
                 "neutral": "neutral"}.get(sma.trend, sma.trend)
    sma_cross = {"golden_cross": "golden cross (bullish)",
                 "death_cross": "death cross (bearish)",
                 "none": "none"}.get(sma.cross, sma.cross)

    vol_level = {"high": "high", "medium": "medium", "low": "low"}.get(
        vol.volatility_level, vol.volatility_level)

    lines = [
        f"Candle timeframe: {timeframe}, candles in window: {len(df)}",
        "",
        "**RSI (14):**",
        f"- Value: {_fmt(rsi.value)}",
        f"- Zone: {rsi_label}",
        f"- Signal strength: {_fmt(rsi.strength)}",
        f"- Overbought (potential short): {_yes_no(rsi.is_short)}",
        f"- Oversold (potential long): {_yes_no(rsi.is_long)}",
        "",
        "**SMA (20/50):**",
        f"- SMA20: {_fmt(sma.sma20)}",
        f"- SMA50: {_fmt(sma.sma50)}",
        f"- Price deviation from SMA20: {_fmt_pct(sma.price_vs_sma20)}",
        f"- Price deviation from SMA50: {_fmt_pct(sma.price_vs_sma50)}",
        f"- Trend: {sma_trend}",
        f"- Crossover: {sma_cross}",
        f"- Signal strength: {_fmt(sma.strength)}",
        "",
        "**Volatility (ATR / Bollinger):**",
        f"- ATR: {_fmt(vol.atr)} ({_fmt(vol.atr_percent)}% of price)",
        f"- Bollinger Bands width: {_fmt(vol.bb_width)}%",
        f"- Price position in channel: {_fmt(vol.bb_position)} (0 = lower band, 1 = upper)",
        f"- Volatility level: {vol_level}",
        f"- Reversal risk: {_yes_no(vol.is_reversal_risk)}",
        f"- Signal strength: {_fmt(vol.strength)}",
    ]

    data = {
        "timeframe": timeframe,
        "rsi": {"value": _plain(rsi.value), "signal": rsi.signal,
                "strength": _plain(rsi.strength),
                "is_short": _plain(rsi.is_short), "is_long": _plain(rsi.is_long)},
        "sma": {"sma20": _plain(sma.sma20), "sma50": _plain(sma.sma50),
                "price_vs_sma20": _plain(sma.price_vs_sma20),
                "price_vs_sma50": _plain(sma.price_vs_sma50),
                "trend": sma.trend, "cross": sma.cross,
                "strength": _plain(sma.strength)},
        "volatility": {"atr": _plain(vol.atr), "atr_percent": _plain(vol.atr_percent),
                       "bb_width": _plain(vol.bb_width),
                       "bb_position": _plain(vol.bb_position),
                       "level": vol.volatility_level,
                       "is_reversal_risk": _plain(vol.is_reversal_risk),
                       "strength": _plain(vol.strength)},
    }
    return "\n".join(lines), data


def _news_text(db: Session, ticker: str) -> tuple[str, dict]:
    """
    Агрегирует новости за NEWS_PERIOD_HOURS часов и возвращает
    (читаемый текст для промпта, структурированные данные).
    """
    agg = aggregate_news(db, ticker, hours=NEWS_PERIOD_HOURS)

    if agg.total_news == 0:
        note = f"No processed news for {ticker} in the last {NEWS_PERIOD_HOURS}h."
        return note, {
            "period_hours": NEWS_PERIOD_HOURS,
            "total_news": 0,
            "avg_sentiment": 0.0,
            "avg_confidence": 0.0,
            "note": note,
        }

    label_en = {"positive": "positive", "negative": "negative",
                "neutral": "neutral"}.get(agg.sentiment_label, agg.sentiment_label)
    signal_en = {"long": "long (up)", "short": "short (down)",
                 "none": "none"}.get(agg.dominant_signal, agg.dominant_signal)

    lines = [
        f"Period: last {agg.period_hours}h | news count: {agg.total_news}",
        f"- Average sentiment: {agg.avg_sentiment:+.2f} ({label_en})",
        f"- Average confidence: {agg.avg_confidence:.0%}",
        f"- Distribution: positive {agg.positive_count} | negative {agg.negative_count} | neutral {agg.neutral_count}",
        f"- Dominant signal: {signal_en} (strength {_fmt(agg.signal_strength)})",
        "",
        "Recent news:",
    ]
    for i, item in enumerate(agg.news_items[:10], 1):
        lines.append(
            f"  {i}. [{item['source']}] ({_to_msk(item['published_at'])} MSK) "
            f"\"{item['title']}\" — sentiment {item['sentiment_score']:+.2f}, "
            f"confidence {item['confidence']:.0%}"
        )

    data = {
        "period_hours": agg.period_hours,
        "total_news": agg.total_news,
        "avg_sentiment": agg.avg_sentiment,
        "avg_confidence": agg.avg_confidence,
        "sentiment_label": agg.sentiment_label,
        "positive_count": agg.positive_count,
        "negative_count": agg.negative_count,
        "neutral_count": agg.neutral_count,
        "dominant_signal": agg.dominant_signal,
        "signal_strength": agg.signal_strength,
    }
    return "\n".join(lines), data


def build_user_prompt(ticker: str, agents_text: str, news_text: str) -> str:
    """Builds the user prompt from context (technical agents + news)."""
    return (
        f"Ticker: {ticker}\n"
        f"Task: build a Bayesian network to forecast the price move.\n\n"
        f"## Technical agents output\n{agents_text}\n\n"
        f"## News for the last {NEWS_PERIOD_HOURS} hours (sentiment and confidence)\n{news_text}\n\n"
        f"## Task\n"
        f"Build a Bayesian network with the target variable Price_Change "
        f"(states: Down/Neutral/Up). Use technical indicators (RSI, SMA trend, "
        f"volatility) and the news background (sentiment with confidence) as input "
        f"variables. Follow realistic market logic: overbought conditions and "
        f"negative sentiment lean toward Down; oversold conditions and positive "
        f"sentiment with high confidence lean toward Up. If data is scarce or "
        f"confidence is low, reduce confidence in the probabilities.\n"
        f"\n"
        f"Structural requirements (mandatory for the pgmpy model to build; "
        f"violating them makes the answer invalid):\n"
        f"- each variable has exactly 3 states;\n"
        f"- any variable has AT MOST 2 parents (including Price_Change); "
        f"fewer parents = more robust model;\n"
        f"- the number of CPD rows must be EXACTLY 3^K, where K is the number of "
        f"parents of the variable: 0 parents → exactly 1 row (prior probability); "
        f"1 parent → exactly 3 rows; 2 parents → exactly 9 rows, one row per "
        f"combination of parent states;\n"
        f"- DO NOT truncate the CPD, skip combinations, or repeat rows — "
        f"a partial table makes the model impossible to build in pgmpy;\n"
        f"- each CPD row is a distribution, probabilities must sum to exactly 1.0;\n"
        f"- the first parent in the list changes slowest, the last one fastest;\n"
        f"- the edges list includes ALL parent->child links from the parents "
        f"fields and nothing extra.\n"
        f"\n"
        f"Example of a valid JSON (format only; numbers are illustrative):\n"
        f"{{\n"
        f"  \"variables\": [\n"
        f"    {{\"name\": \"RSI\", \"states\": [\"Overbought\", \"Neutral\", \"Oversold\"], "
        f"\"parents\": [], \"cpd\": [[0.15, 0.65, 0.20]]}},\n"
        f"    {{\"name\": \"News_Sentiment\", \"states\": [\"Negative\", \"Neutral\", \"Positive\"], "
        f"\"parents\": [], \"cpd\": [[0.3, 0.5, 0.2]]}},\n"
        f"    {{\"name\": \"Price_Change\", \"states\": [\"Down\", \"Neutral\", \"Up\"], "
        f"\"parents\": [\"RSI\", \"News_Sentiment\"], \"cpd\": [\n"
        f"      [0.6, 0.25, 0.15], [0.5, 0.3, 0.2], [0.4, 0.35, 0.25],\n"
        f"      [0.45, 0.3, 0.25], [0.3, 0.4, 0.3], [0.2, 0.35, 0.45],\n"
        f"      [0.3, 0.35, 0.35], [0.2, 0.4, 0.4], [0.1, 0.3, 0.6]\n"
        f"    ]}}\n"
        f"  ],\n"
        f"  \"edges\": [[\"RSI\", \"Price_Change\"], [\"News_Sentiment\", \"Price_Change\"]],\n"
        f"  \"target_variable\": \"Price_Change\",\n"
        f"  \"explanation\": \"Brief justification of the structure\"\n"
        f"}}\n"
        f"\n"
        f"Return STRICTLY valid JSON matching the format from the system prompt, "
        f"with no text outside the JSON."
    )


# --- Валидация ответа LLM ---

def _extract_json(response: str):
    """
    Извлекает JSON-объект из ответа LLM.
    Ответ может быть голым JSON, в markdown-блоке ```json ... ``` или с текстом вокруг.
    """
    if not response:
        return None
    text = response.strip()

    # Markdown-блок ```json ... ```
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    # Голый JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Самый внешний { ... } в тексте
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None


def validate_bayesian_json(data) -> tuple[bool, list[str], list[str]]:
    """
    Проверяет JSON байесовской сети. Жёсткие ошибки (errors) делают сеть
    невалидной; мягкие (warnings) чинятся сборщиком pgmpy автоматически.

    Жёсткие (невалидный JSON):
    - 4 обязательных поля: variables, edges, target_variable, explanation;
    - каждая переменная: name, ровно 3 состояния, parents (список строк),
      cpd (непустой список);
    - target_variable == "Price_Change" и присутствует среди variables.

    Мягкие (repair в bayesian_network_viz):
    - неполный cpd (число строк != 1 / 3^k);
    - строка cpd не той длины или сумма != 1.0;
    - рёбра на неизвестные переменные (отбрасываются).

    Returns:
        (is_valid, errors, warnings)
    """
    errors: list[str] = []
    warnings: list[str] = []

    if data is None:
        return False, ["LLM не вернул JSON"], []
    if not isinstance(data, dict):
        return False, [f"Ожидался JSON-объект, получен {type(data).__name__}"], []

    required = ("variables", "edges", "target_variable", "explanation")
    for key in required:
        if key not in data:
            errors.append(f"Отсутствует обязательное поле '{key}'")

    variables = data.get("variables")
    edges = data.get("edges")
    target = data.get("target_variable")
    explanation = data.get("explanation")

    if not isinstance(variables, list) or not variables:
        errors.append("Поле 'variables' должно быть непустым массивом")
    if not isinstance(edges, list):
        errors.append("Поле 'edges' должно быть массивом")
    if not isinstance(target, str) or not target.strip():
        errors.append("Поле 'target_variable' должно быть непустой строкой")
    if not isinstance(explanation, str) or not explanation.strip():
        errors.append("Поле 'explanation' должно быть непустой строкой")

    names: set[str] = set()

    if isinstance(variables, list):
        for i, var in enumerate(variables):
            if not isinstance(var, dict):
                errors.append(f"variables[{i}]: элемент не является объектом")
                continue

            name = var.get("name")
            states = var.get("states")
            parents = var.get("parents", [])
            cpd = var.get("cpd")

            if not isinstance(name, str) or not name.strip():
                errors.append(f"variables[{i}]: отсутствует или пусто поле 'name'")
                continue
            names.add(name)

            if not isinstance(states, list) or len(states) != 3:
                errors.append(f"'{name}': 'states' должен содержать ровно 3 состояния")

            if not isinstance(parents, list) or not all(isinstance(p, str) for p in parents):
                errors.append(f"'{name}': 'parents' должен быть списком строк")

            if not isinstance(cpd, list) or not cpd:
                errors.append(f"'{name}': 'cpd' не задан или пуст")
                continue

            # Мягкие проверки CPD — сборщик чинит их автоматически.
            n_states = len(states) if isinstance(states, list) and len(states) == 3 else 3
            expected_rows = 1 if not parents else 3 ** len(parents)
            if len(cpd) != expected_rows:
                warnings.append(
                    f"'{name}': CPD неполный — {len(cpd)} из {expected_rows} строк "
                    f"(будет дополнен сборщиком)"
                )

            for j, row in enumerate(cpd):
                if not isinstance(row, list) or not all(
                    isinstance(v, (int, float)) and not isinstance(v, bool) for v in row
                ):
                    warnings.append(f"'{name}': строка cpd[{j}] не является числовым списком")
                    continue
                if len(row) != n_states:
                    warnings.append(
                        f"'{name}': строка cpd[{j}] содержит {len(row)} значений "
                        f"(нужно {n_states})"
                    )
                total = sum(row)
                if abs(total - 1.0) > 0.05:
                    warnings.append(
                        f"'{name}': строка cpd[{j}] суммируется в {total:.2f}, не 1.0"
                    )

    if not target or target != "Price_Change":
        errors.append(f"target_variable должен быть 'Price_Change', получено '{target}'")
    elif target not in names:
        errors.append(f"target_variable '{target}' отсутствует среди variables")

    if isinstance(edges, list):
        for e in edges:
            if not isinstance(e, (list, tuple)) or len(e) != 2:
                warnings.append(
                    f"Ребро {e} не является парой ['Parent', 'Child'] — будет отброшено"
                )
                continue
            parent, child = e[0], e[1]
            if parent not in names:
                warnings.append(f"Ребро {e}: родитель '{parent}' не найден — ребро отброшено")
            if child not in names:
                warnings.append(f"Ребро {e}: потомок '{child}' не найден — ребро отброшено")

    return (not errors), errors, warnings


# --- Оркестрация ---

def run_bayesian_agent(db: Session, ticker: str,
                       model: Optional[str] = None) -> dict:
    """
    Полный проход по одному тикеру: агенты → новости → LLM → валидация.

    Returns:
        dict: {ticker, status, agents, news, prompt, response,
               valid, errors, warnings, json}
    """
    system_prompt = load_system_prompt()
    agents_text, agents_data = _agents_text(db, ticker)
    news_text, news_data = _news_text(db, ticker)
    user_prompt = build_user_prompt(ticker, agents_text, news_text)

    result = {
        "ticker": ticker,
        "status": "ok",
        "agents": agents_data,
        "news": news_data,
        "prompt": user_prompt,
    }

    try:
        response = ask_llm(
            user_prompt,
            system=system_prompt,
            model=model,
            temperature=0.2,
            max_tokens=2048,
        )
    except Exception as exc:
        logger.exception("Байесовский агент %s: ошибка LLM", ticker)
        result["status"] = "error"
        result["error"] = str(exc)
        return result

    result["response"] = response

    parsed = _extract_json(response)
    valid, errors, warnings = validate_bayesian_json(parsed)
    result["valid"] = valid
    result["errors"] = errors
    result["warnings"] = warnings
    result["json"] = parsed
    result["status"] = "ok" if valid else "invalid_json"

    logger.info(
        "Байесовский агент %s: valid=%s, errors=%d, warnings=%d, переменных=%d",
        ticker, valid, len(errors), len(warnings),
        len(parsed.get("variables", [])) if isinstance(parsed, dict) else 0,
    )
    return result


def run_bayesian_for_tickers(db: Session,
                             tickers: Optional[list[str]] = None) -> dict:
    """
    Запускает байесовского агента по списку тикеров (по умолчанию — все
    отслеживаемые). Возвращает компактную сводку для статистики сбора.
    """
    tickers = tickers or get_tracked_tickers(db)
    summary: dict = {}
    for ticker in tickers:
        try:
            res = run_bayesian_agent(db, ticker)
        except Exception:
            logger.exception("Байесовский агент: внутренняя ошибка по %s", ticker)
            res = {"ticker": ticker, "status": "error", "error": "внутренняя ошибка"}

        entry = {
            "status": res.get("status"),
            "valid": res.get("valid"),
            "errors": res.get("errors", []),
            "warnings_count": len(res.get("warnings", [])),
        }
        parsed = res.get("json")
        if isinstance(parsed, dict):
            entry["variables_count"] = len(parsed.get("variables", []))
        summary[ticker] = entry
    return summary


# --- Сохранение / загрузка сетей (таблица decisions) ---

def _describe_network(parsed: dict, probs: Optional[dict]) -> str:
    """Короткое описание сети для поля bayesian_visualization."""
    n_vars = len(parsed.get("variables", []))
    n_edges = len(parsed.get("edges", []))
    text = f"{n_vars} переменных, {n_edges} рёбер"
    if probs:
        text += "; " + ", ".join(f"{k}={v:.2f}" for k, v in probs.items())
    return text


def save_network_result(db: Session, result: dict) -> Optional[int]:
    """
    Сохраняет валидный результат байесовского агента в таблицу decisions.

    Считает апостериорную вероятность целевой переменной (pgmpy) и выводит
    action/confidence. Поля-снапшоты агентов (rsi, volatility, sma_50) и новостей
    (sentiment_score, sentiment_news_count) заполняются для контекста решения.

    Returns:
        id записи Decision, или None если сеть невалидна.
    """
    if not result.get("valid"):
        return None
    from app.models.decision import Decision

    parsed = result.get("json") or {}
    agents = result.get("agents") or {}
    news = result.get("news") or {}
    target = parsed.get("target_variable", "Price_Change")

    states: list = []
    probs: dict = {}
    action = "NEUTRAL"
    confidence = 0.0
    try:
        from app.services.bayesian_network_viz import (
            build_model_from_json, infer_target, action_from_probs,
        )
        model, _ = build_model_from_json(parsed)
        states, probs = infer_target(model, target)
        confidence = max(probs.values(), default=0.0)
        action, _ = action_from_probs(probs)
    except Exception:
        logger.warning(
            "%s: инференс pgmpy не удался — сохраняю структуру без вероятностей",
            result["ticker"],
        )
        confidence = news.get("avg_confidence") or 0.0

    rsi = agents.get("rsi", {}).get("value")
    atr_pct = agents.get("volatility", {}).get("atr_percent")
    sma50 = agents.get("sma", {}).get("sma50")

    decision = Decision(
        decision_id=str(uuid.uuid4()),
        ticker=result["ticker"],
        action=action,
        confidence=confidence,
        probability_down=probs.get("Down"),
        probability_up=probs.get("Up"),
        sentiment_score=news.get("avg_sentiment"),
        sentiment_news_count=news.get("total_news"),
        rsi=rsi,
        volatility=atr_pct,
        sma_50=sma50,
        bayesian_network_structure=parsed,
        bayesian_network_cpds=parsed,
        bayesian_inference_result={"states": states, "probabilities": probs} if probs else None,
        bayesian_visualization=_describe_network(parsed, probs),
        llm_prompt_network=result.get("prompt"),
        llm_response_network=result.get("response"),
        evaluated_at=datetime.utcnow(),
    )
    db.add(decision)
    db.commit()
    return decision.id


def get_saved_networks(db: Session, ticker: Optional[str] = None,
                       limit: int = 20):
    """Сохранённые сети из decisions, свежие сверху. По тикеру — опционально."""
    from app.models.decision import Decision
    query = db.query(Decision).order_by(Decision.created_at.desc())
    if ticker:
        query = query.filter(Decision.ticker == ticker)
    return query.limit(limit).all()


# --- CLI: python -m app.services.bayesian_network --ticker SBER ---

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Windows-консоль (cp1251) не умеет кириллицу+спецсимволы в UTF-8;
    # принудительно переводим stdout в UTF-8, чтобы не падать на печати.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description="Байесовский агент по тикеру")
    parser.add_argument("--ticker", default=settings.TRACKED_TICKERS[0],
                        help="Тикер (по умолчанию первый из TRACKED_TICKERS)")
    parser.add_argument("--show-context", action="store_true",
                        help="Показать контекст (агенты + новости), отправляемый в LLM")
    parser.add_argument("--show-json", action="store_true",
                        help="Показать распарсенный JSON-ответ")
    args = parser.parse_args()

    print("=" * 70)
    print(f"Байесовский агент | тикер: {args.ticker} | LLM: {settings.LLM_BASE_URL}")
    print(f"Системный промпт: {PROMPT_PATH}")
    print("=" * 70)

    with get_db_context() as db:
        result = run_bayesian_agent(db, args.ticker)

    if args.show_context:
        print("\n--- КОНТЕКСТ ДЛЯ LLM ---")
        print(result["prompt"])
        print("------------------------")

    print(f"\nСтатус: {result['status']}")
    if result.get("error"):
        print(f"Ошибка: {result['error']}")

    if result.get("valid") is not None:
        print(f"Валидность JSON: {'[OK] валиден' if result['valid'] else '[FAIL] НЕ валиден'}")
        if result["errors"]:
            for e in result["errors"]:
                print(f"  - {e}")
        if result.get("warnings"):
            print("  Предупреждения (чинятся сборщиком pgmpy):")
            for w in result["warnings"]:
                print(f"    • {w}")

    if args.show_json and result.get("json") is not None:
        print("\n--- JSON ОТВЕТ LLM ---")
        print(json.dumps(result["json"], ensure_ascii=False, indent=2))
        print("-----------------------")
