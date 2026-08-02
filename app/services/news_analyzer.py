"""
Анализ новостей через LLM.

Загружает промпт из app/prompts/news_handler, отправляет пачки новостей
в LLM, парсит JSON-ответ и создаёт NewsArticle.

Запуск:
    python -m app.services.news_analyzer          # обработать все необработанные
    python -m app.services.news_analyzer --once   # один проход и выйти
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from app.core.database import get_db_context
from app.models.news import RawNews, NewsArticle
from app.services.llm import ask_llm

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "news_handler"

# Размер пакета новостей для одного запроса к LLM
BATCH_SIZE = 10


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


def _sentiment_label(score: Optional[float]) -> Optional[str]:
    """Числовой скор -> текстовая метка."""
    if score is None:
        return None
    if score < -0.3:
        return "negative"
    if score > 0.3:
        return "positive"
    return "neutral"


def _extract_tickers(item: dict) -> tuple[str, str]:
    """
    Извлекает тикеры из ответа LLM.
    Возвращает (tickers_str, primary_ticker).
    """
    raw = item.get("tickers", "")
    tickers_list = []

    if isinstance(raw, dict):
        # Формат {"ticker": "SBER"} или {"ticker": ["SBER", "GAZP"]}
        val = raw.get("ticker", [])
        if isinstance(val, str):
            tickers_list = [val]
        elif isinstance(val, list):
            tickers_list = val
        else:
            tickers_list = [str(val)] if val else []
    elif isinstance(raw, str):
        tickers_list = [t.strip() for t in raw.split(",") if t.strip()]
    elif isinstance(raw, list):
        tickers_list = raw
    else:
        tickers_list = []

    # Нормализуем: извлекаем строковые значения из любых вложенных объектов
    normalized = []
    for t in tickers_list:
        if isinstance(t, dict):
            # Если внутри dict — берём любое строковое значение
            for v in t.values():
                if isinstance(v, str) and v:
                    normalized.append(v.upper())
                    break
        elif isinstance(t, str) and t:
            normalized.append(t.upper())
        elif t is not None:
            normalized.append(str(t).upper())

    tickers_str = ",".join(normalized)
    primary = normalized[0] if normalized else ""
    return tickers_str, primary


def analyze_batch(db: Session, raw_news_list: list[RawNews], system_prompt: str) -> int:
    """
    Отправляет пакет новостей в LLM, создаёт NewsArticle.
    Возвращает число созданных статей.
    """
    if not raw_news_list:
        return 0

    # Формируем промпт с новостями
    news_text_parts = []
    for i, item in enumerate(raw_news_list, 1):
        news_text_parts.append(
            f"[{i}] Источник: {item.source}\n"
            f"Заголовок: {item.title}\n"
            f"Текст: {item.full_text[:1500]}\n"
        )
    user_prompt = (
        "Обработай следующие новости. Верни JSON-массив.\n\n"
        + "\n---\n".join(news_text_parts)
    )

    try:
        response = ask_llm(user_prompt, system=system_prompt, max_tokens=4096)
    except Exception:
        logger.exception("LLM вернул ошибку при обработке пакета из %d новостей", len(raw_news_list))
        return 0

    items = _parse_llm_json(response)
    if not items:
        logger.warning("LLM не вернул валидный JSON для пакета из %d новостей", len(raw_news_list))
        return 0

    created = 0
    for i, item in enumerate(items):
        raw = raw_news_list[i] if i < len(raw_news_list) else None
        tickers_str, primary_ticker = _extract_tickers(item)
        score = item.get("sentiment_score")
        try:
            score = float(score) if score is not None else None
        except (ValueError, TypeError):
            score = None

        confidence = item.get("sentiment_confidence")
        try:
            confidence = float(confidence) if confidence is not None else None
        except (ValueError, TypeError):
            confidence = None

        article = NewsArticle(
            title=item.get("title", raw.title if raw else ""),
            full_text=item.get("summary", raw.full_text[:500] if raw else ""),
            summary=item.get("summary"),
            sentiment_score=score,
            sentiment_label=_sentiment_label(score),
            sentiment_confidence=confidence,
            tickers=tickers_str,
            primary_ticker=primary_ticker,
            tags=item.get("industry_tag", ""),
            is_ai_generated=True,
            source=raw.source if raw else "unknown",
            published_at=raw.published_at if raw else None,
            raw_news_id=raw.id if raw else None,
        )
        db.add(article)
        created += 1

        # Помечаем сырую новость как обработанную
        if raw:
            raw.is_processed = True

    db.commit()
    logger.info("Создано %d NewsArticle из %d сырых новостей", created, len(raw_news_list))
    return created


def process_unprocessed(db: Session, limit: int = BATCH_SIZE) -> int:
    """
    Находит необработанные новости и отправляет их в LLM.
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
    system_prompt = load_system_prompt()
    return analyze_batch(db, raw_list, system_prompt)


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
