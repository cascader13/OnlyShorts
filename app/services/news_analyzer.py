"""
Анализ новостей.

Реализация выбирается по settings.NEWS_ANALYZER:
  - "llm" (по умолчанию): загружает промпт из app/prompts/news_handler,
    отправляет пачки новостей в LLM, парсит JSON-ответ и создаёт NewsArticle;
  - "heuristic": лексиконный анализатор без LLM
    (app/services/news_analyzer_heuristic.py), тот же контракт NewsArticle.

Запуск:
    python -m app.services.news_analyzer          # обработать все необработанные
    python -m app.services.news_analyzer --once   # один проход и выйти
"""

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db_context
from app.models.news import RawNews, NewsArticle
from app.services.llm import ask_llm

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "news_handler"

# Размер пакета новостей для одного запроса к LLM (из настроек, fallback 10)
BATCH_SIZE = settings.NEWS_BATCH_SIZE

# Допустимые секторы (те же, что перечислены в промпте news_handler).
# Значение вне списка — признак, что модель не следует схеме: не сохраняем.
SECTORS = {
    "IT",
    "Metals & Mining",
    "Oil & Gas",
    "Banking & Finance",
    "Retail & Consumer Goods",
    "Telecommunications",
    "Transportation & Logistics",
    "Utilities",
    "Healthcare",
    "Chemicals",
}


def _clamp_float(value, lo: float, hi: float) -> Optional[float]:
    """Приводит значение из ответа LLM к float и ограничивает диапазоном.

    LLM иногда отдаёт score вне шкалы (1.5, -2) или строкой — валидируем,
    а не доверяем на слово.
    """
    try:
        value = float(value) if value is not None else None
    except (ValueError, TypeError):
        return None
    if value is None:
        return None
    return max(lo, min(hi, value))


def load_system_prompt() -> str:
    """Загружает системный промпт из файла."""
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def _parse_llm_json(response: str) -> list[dict]:
    """
    Извлекает JSON из ответа LLM.
    LLM может вернуть JSON в markdown-блоке ```json ... ``` или голый JSON.
    """
    # Пробуем извлечь из markdown-блока
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", response, re.DOTALL)
    if match:
        text = match.group(1)
    else:
        text = response.strip()

    # Пробуем распарсить как массив
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return [data]
        return data
    except json.JSONDecodeError:
        pass

    # Ищем первый [ ... ] в тексте
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    logger.warning("Не удалось распарсить JSON из ответа LLM: %s", response[:200])
    return []


def _build_news_prompt(raw_news_list: list[RawNews]) -> str:
    """Формирует user-промпт с нумерованным списком новостей."""
    news_text_parts = []
    for i, item in enumerate(raw_news_list, 1):
        news_text_parts.append(
            f"[{i}] Источник: {item.source}\n"
            f"Заголовок: {item.title}\n"
            f"Текст: {item.full_text[:1500]}\n"
        )
    return (
        "Обработай следующие новости. Верни JSON-массив.\n\n"
        + "\n---\n".join(news_text_parts)
    )


def _ask_llm_json(
    raw_news_list: list[RawNews],
    system_prompt: str,
    retries: Optional[int] = None,
) -> list[dict]:
    """Запрашивает LLM и возвращает распарсенный JSON-список.

    LLM иногда возвращает невалидный JSON (рвёт массив, оборачивает в
    markdown, добавляет пояснения) или падает по сети. Делаем до `retries`
    повторных попыток: на ретрае подсказываем модели, что прошлый ответ не
    распарсился и нужен строго JSON-массив по схеме. Сетевые ошибки также
    повторяются (обычно транзиентные).

    Returns:
        list[dict] — элементы ответа; [] если все попытки исчерпаны.
    """
    retries = retries if retries is not None else settings.NEWS_LLM_RETRIES
    base_prompt = _build_news_prompt(raw_news_list)
    attempts = retries + 1
    last_raw = ""

    for attempt in range(attempts):
        if attempt == 0:
            user_prompt = base_prompt
        else:
            user_prompt = (
                base_prompt
                + "\n\n⚠️ Твой предыдущий ответ не был валидным JSON:\n"
                + last_raw[:300]
                + "\nВерни ТОЛЬКО валидный JSON-массив по схеме из системного промпта, без пояснений."
            )

        try:
            response = ask_llm(
                user_prompt,
                system=system_prompt,
                max_tokens=settings.NEWS_MAX_TOKENS,
                reasoning_effort=settings.LLM_REASONING_EFFORT or None,
            )
        except Exception:
            logger.warning(
                "LLM-ошибка на попытке %d/%d (батч %d новостей)",
                attempt + 1, attempts, len(raw_news_list),
            )
            if attempt == attempts - 1:
                logger.exception("LLM не ответил за %d попыток", attempts)
                return []
            continue  # транзиентный сбой — пробуем ещё раз

        items = _parse_llm_json(response)
        if items:
            return items
        last_raw = response
        logger.warning(
            "Попытка %d/%d: LLM вернул невалидный JSON (%d символов)",
            attempt + 1, attempts, len(response),
        )

    return []


def _sentiment_label(score: Optional[float]) -> Optional[str]:
    """Числовой скор -> текстовая метка."""
    if score is None:
        return None
    if score < -0.3:
        return "negative"
    if score > 0.3:
        return "positive"
    return "neutral"


# Whitelist тикеров и regex тегов {$TICKER} берём из эвристического анализатора,
# чтобы не дублировать списки (единый источник правды). Импорт ленивый: модуль
# эвристики поднимает MorphAnalyzer (~0.8с) при первом импорте, а LLM-пути он
# не нужен.
_KNOWN_TICKERS: Optional[frozenset[str]] = None
_TAG_RE: Optional[re.Pattern] = None


def _ticker_whitelist() -> tuple[frozenset[str], re.Pattern]:
    """Возвращает (whitelist тикеров, regex тегов {$TICKER}) из эвристики."""
    global _KNOWN_TICKERS, _TAG_RE
    if _KNOWN_TICKERS is None:
        from app.services.news_analyzer_heuristic import KNOWN_TICKERS, _TAG_RE

        _KNOWN_TICKERS = frozenset(KNOWN_TICKERS)
        _TAG_RE = _TAG_RE
    return _KNOWN_TICKERS, _TAG_RE


def _normalize_tickers(tickers_list: list) -> list[str]:
    """Приводит «тикеры» из ответа LLM к списку строк ВЕРХНИМ РЕГИСТРОМ."""
    normalized = []
    for t in tickers_list:
        if isinstance(t, dict):
            # Схема допускает {"ticker": "SBER"}: берём первое строковое значение
            for v in t.values():
                if isinstance(v, str) and v:
                    normalized.append(v.upper())
                    break
        elif isinstance(t, str) and t:
            normalized.append(t.upper())
        elif t is not None:
            normalized.append(str(t).upper())
    return normalized


def _extract_tickers(item: dict, raw: Optional[RawNews] = None) -> tuple[str, str]:
    """Извлекает тикеры из ответа LLM и фильтрует их по whitelist.

    LLM может «галлюцинировать» тикеры: вернуть ISIN облигации
    (RU000A10FA72), код фьючерса (BRQ6), индекс (IMOEX) или иностранный
    тикер (T, YDEX). Такие значения не торгуются и теряют новость для
    агрегатора — ticker_matches матчит primary_ticker/tickers только против
    известных тикеров. Поэтому:
      1) выкидываем всё, чего нет в KNOWN_TICKERS;
      2) если после фильтра ничего не осталось — берём теги {$TICKER} из
         текста новости (их вставляет сама платформа, они надёжны и покрывают
         тикеры вне whitelist, например OZON).

    Returns:
        (tickers_str, primary_ticker).
    """
    known, tag_re = _ticker_whitelist()

    raw_tickers = item.get("tickers", "")
    if isinstance(raw_tickers, dict):
        # {"ticker": "SBER"} или {"ticker": ["SBER", "GAZP"]} и т.п. — берём все значения
        tickers_list = []
        for val in raw_tickers.values():
            if isinstance(val, str):
                tickers_list.append(val)
            elif isinstance(val, list):
                tickers_list.extend(val)
    elif isinstance(raw_tickers, str):
        tickers_list = [t.strip() for t in raw_tickers.split(",") if t.strip()]
    elif isinstance(raw_tickers, list):
        tickers_list = raw_tickers
    else:
        tickers_list = []

    filtered: list[str] = []
    for t in _normalize_tickers(tickers_list):
        if t in known and t not in filtered:
            filtered.append(t)

    # Фолбэк на теги платформы (без whitelist-фильтра — они уже настоящие)
    if not filtered and raw and raw.full_text:
        for m in tag_re.finditer(raw.full_text):
            t = m.group(1).upper()
            if t not in filtered:
                filtered.append(t)

    return ",".join(filtered), filtered[0] if filtered else ""


def _parse_news_index(item: dict) -> Optional[int]:
    """Извлекает news_index из ответа LLM (1-based, соответствует [N] во входе).

    Если модель слила несколько новостей в одну — берём наименьший индекс
    (первый источник слияния). Без индекса возвращаем None.
    """
    raw = item.get("news_index", item.get("index"))
    if raw is None:
        return None
    if isinstance(raw, list):
        vals = [int(v) for v in raw if isinstance(v, (int, str)) and str(v).isdigit()]
        return min(vals) if vals else None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _pair_items(
    items: list[dict], raw_news_list: list[RawNews]
) -> list[tuple[dict, Optional[RawNews]]]:
    """Сопоставляет ответы LLM с сырыми новостями.

    Приоритет — news_index из ответа: промпт требует возвращать его, поэтому
    он надёжен даже когда LLM сливает дубликаты (элементов меньше, чем на
    входе) или переставляет порядок.

    Политика:
      * валидный и неиспользованный news_index -> точная привязка;
      * нет news_index вовсе -> позиционный фолбэк на первую свободную
        новость (без дублей) — редкий случай, промпт требует индекс;
      * news_index невалиден (вне [1..N]) или уже использован -> БЕЗ привязки:
        статью создаём без raw_news_id, сырую новость не размечаем (лучше
        повторная обработка в следующем цикле, чем разметить не ту новость).
    """
    by_index = {i + 1: raw for i, raw in enumerate(raw_news_list)}
    used_raw_ids: set[int] = set()
    paired: list[tuple[dict, Optional[RawNews]]] = []
    no_index: list[dict] = []
    invalid_index: list[dict] = []

    for item in items:
        idx = _parse_news_index(item)
        if idx is None:
            no_index.append(item)
            continue
        raw = by_index.get(idx)
        if raw is not None and raw.id not in used_raw_ids:
            paired.append((item, raw))
            used_raw_ids.add(raw.id)
        else:
            invalid_index.append(item)

    free_raws = [r for r in raw_news_list if r.id not in used_raw_ids]
    for item in no_index:
        raw = free_raws.pop(0) if free_raws else None
        if raw is not None:
            used_raw_ids.add(raw.id)
        paired.append((item, raw))

    paired.extend((item, None) for item in invalid_index)
    return paired


def analyze_batch(db: Session, raw_news_list: list[RawNews], system_prompt: str) -> int:
    """
    Отправляет пакет новостей в LLM, создаёт NewsArticle.
    Возвращает число созданных статей.
    """
    if not raw_news_list:
        return 0

    # Запрос к LLM с ретраями на невалидный JSON (см. _ask_llm_json)
    items = _ask_llm_json(raw_news_list, system_prompt)
    if not items:
        logger.warning(
            "LLM не вернул валидный JSON за %d попыток (пакет из %d новостей)",
            settings.NEWS_LLM_RETRIES + 1, len(raw_news_list),
        )
        return 0

    created = 0
    # Сопоставляем ответы с сырыми новостями по news_index (см. _pair_items):
    # при большом батче модель может слить/переставить элементы, и позиционная
    # привязка разметила бы не ту новость.
    for item, raw in _pair_items(items, raw_news_list):
        tickers_str, primary_ticker = _extract_tickers(item, raw)
        score = _clamp_float(item.get("sentiment_score"), -1.0, 1.0)
        confidence = _clamp_float(item.get("sentiment_confidence"), 0.0, 1.0)
        industry = item.get("industry_tag", "").strip()
        if industry and industry not in SECTORS:
            logger.warning("Нестандартный industry_tag от LLM: %r", industry)
            industry = ""

        article = NewsArticle(
            title=item.get("title", raw.title if raw else ""),
            full_text=item.get("summary", raw.full_text[:500] if raw else ""),
            summary=item.get("summary"),
            sentiment_score=score,
            sentiment_label=_sentiment_label(score),
            sentiment_confidence=confidence,
            tickers=tickers_str,
            primary_ticker=primary_ticker,
            tags=industry,
            is_ai_generated=True,
            source=raw.source if raw else "unknown",
            published_at=raw.published_at if raw else None,
            raw_news_id=raw.id if raw else None,
        )
        db.add(article)
        created += 1

        # Помечаем сырую новость как обработанную (только реально
        # сопоставленную; без индекса raw не размечаем).
        if raw:
            raw.is_processed = True

    db.commit()
    logger.info("Создано %d NewsArticle из %d сырых новостей", created, len(raw_news_list))
    return created


def process_unprocessed(db: Session, limit: int = BATCH_SIZE) -> int:
    """
    Находит необработанные новости и анализирует их.
    Реализация выбирается по settings.NEWS_ANALYZER:
      - "llm" — LLM-анализ;
      - "heuristic" — лексиконный анализатор без LLM.
    Возвращает число созданных статей.
    """
    raw_list = (
        db.query(RawNews)
        .filter(RawNews.is_processed == False, RawNews.is_duplicate == False)
        .order_by(RawNews.created_at.asc())
        .limit(limit)
        .all()
    )
    if not raw_list:
        return 0

    logger.info("Обработка %d необработанных новостей", len(raw_list))

    if settings.NEWS_ANALYZER == "heuristic":
        from app.services.news_analyzer_heuristic import analyze_batch as heuristic_analyze_batch

        logger.info("Анализатор: эвристический (без LLM)")
        return heuristic_analyze_batch(db, raw_list)

    system_prompt = load_system_prompt()
    return analyze_batch(db, raw_list, system_prompt)


def process_many(
    db: Session,
    *,
    batch_size: int = BATCH_SIZE,
    max_news: Optional[int] = None,
    max_batches: Optional[int] = None,
    max_seconds: Optional[float] = None,
) -> int:
    """Обрабатывает необработанные новости несколькими батчами подряд.

    Один вызов = несколько LLM-запросов, чтобы за один цикл сбора охватить
    большую часть свежих новостей (раньше цикл обрабатывал один батч из 5).
    Останавливается по первому из условий:
      - очередь необработанных новостей пуста;
      - обработано max_news новостей;
      - сделано max_batches батчей;
      - истёк бюджет времени max_seconds (мягкий guard, чтобы не «съесть»
        весь интервал сбора).

    Реализация анализа (LLM/heuristic) выбирается внутри process_unprocessed.
    Возвращает общее число созданных статей.
    """
    total = 0
    batches = 0
    deadline = time.monotonic() + max_seconds if max_seconds else None

    while True:
        if max_news is not None and total >= max_news:
            logger.info("process_many: лимит новостей (%d) достигнут", max_news)
            break
        if max_batches is not None and batches >= max_batches:
            break
        if deadline is not None and time.monotonic() >= deadline:
            logger.info(
                "process_many: бюджет времени %.0fс исчерпан (обработано %d)",
                max_seconds, total,
            )
            break

        remaining = (max_news - total) if max_news is not None else batch_size
        size = min(batch_size, max(remaining, 1))
        created = process_unprocessed(db, limit=size)
        batches += 1
        total += created
        if created == 0:
            break  # очередь пуста или LLM-ошибка — не долбим дальше

    if total:
        logger.info("process_many: обработано %d новостей за %d батчей", total, batches)
    return total


def get_stats(db: Session) -> dict:
    """Статистика обработки новостей."""
    total = db.query(RawNews).filter(RawNews.is_duplicate == False).count()
    processed = db.query(RawNews).filter(
        RawNews.is_processed == True, RawNews.is_duplicate == False
    ).count()
    articles = db.query(NewsArticle).count()
    return {
        "raw_total": total,
        "raw_processed": processed,
        "raw_pending": total - processed,
        "articles": articles,
    }


# --- CLI: python -m app.services.news_analyzer --once ---

if __name__ == "__main__":
    import argparse
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Анализ новостей через LLM")
    parser.add_argument("--once", action="store_true", help="Один проход и выйти")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help="Размер пакета")
    args = parser.parse_args()

    with get_db_context() as db:
        stats = get_stats(db)
        print(f"До обработки: {stats}")

        if args.once:
            created = process_unprocessed(db, limit=args.batch)
            stats = get_stats(db)
            print(f"Создано: {created}, после: {stats}")
        else:
            while True:
                created = process_unprocessed(db, limit=args.batch)
                if created == 0:
                    print("Нет необработанных новостей")
                    break
                stats = get_stats(db)
                print(f"Обработано: {created}, статус: {stats}")
