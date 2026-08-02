"""
Агрегатор обработанных новостей за период.

Собирает новости за N часов, считает средний сентимент и достоверность.
Используется для принятия торговых решений и отображения на дашборде.
"""

from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field

from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models.news import NewsArticle


@dataclass
class NewsAggregation:
    """Результат агрегации новостей за период."""
    ticker: str
    period_hours: int
    total_news: int                 # Количество новостей за период
    avg_sentiment: float            # Средний сентимент (-1..+1)
    avg_confidence: float           # Средняя достоверность (0..1)
    sentiment_label: str            # "positive", "negative", "neutral"
    positive_count: int             # Количество позитивных
    negative_count: int             # Количество негативных
    neutral_count: int              # Количество нейтральных
    dominant_signal: str            # "short", "long", "none"
    signal_strength: float          # Сила сигнала (0-1)
    news_items: list = field(default_factory=list)  # Список новостей


def aggregate_news(db: Session, ticker: str, hours: int = 3) -> NewsAggregation:
    """
    Агрегирует обработанные новости за последние N часов.

    Args:
        db: Сессия БД
        ticker: Тикер для поиска
        hours: Период в часах (по умолчанию 3)

    Returns:
        NewsAggregation с результатами
    """
    since = datetime.now(timezone.utc) - timedelta(hours=hours)

    # Ищем по primary_ticker и по заголовку
    from sqlalchemy import or_
    articles = (
        db.query(NewsArticle)
        .filter(
            or_(
                NewsArticle.primary_ticker.ilike(f"%{ticker}%"),
                NewsArticle.title.ilike(f"%{ticker}%"),
            ),
            NewsArticle.created_at >= since.replace(tzinfo=None),
            NewsArticle.sentiment_score.isnot(None),
        )
        .order_by(NewsArticle.created_at.desc())
        .all()
    )

    if not articles:
        return NewsAggregation(
            ticker=ticker,
            period_hours=hours,
            total_news=0,
            avg_sentiment=0.0,
            avg_confidence=0.0,
            sentiment_label="neutral",
            positive_count=0,
            negative_count=0,
            neutral_count=0,
            dominant_signal="none",
            signal_strength=0.0,
            news_items=[],
        )

    sentiments = [a.sentiment_score for a in articles if a.sentiment_score is not None]
    confidences = [a.sentiment_confidence for a in articles if a.sentiment_confidence is not None]

    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    positive_count = sum(1 for s in sentiments if s > 0.3)
    negative_count = sum(1 for s in sentiments if s < -0.3)
    neutral_count = len(sentiments) - positive_count - negative_count

    if avg_sentiment > 0.3:
        sentiment_label = "positive"
    elif avg_sentiment < -0.3:
        sentiment_label = "negative"
    else:
        sentiment_label = "neutral"

    if avg_sentiment < -0.3 and avg_confidence > 0.5:
        dominant_signal = "short"
        signal_strength = abs(avg_sentiment) * avg_confidence
    elif avg_sentiment > 0.3 and avg_confidence > 0.5:
        dominant_signal = "long"
        signal_strength = abs(avg_sentiment) * avg_confidence
    else:
        dominant_signal = "none"
        signal_strength = 0.0

    news_items = [
        {
            "title": a.title,
            "summary": a.summary or "",
            "sentiment_score": a.sentiment_score,
            "sentiment_label": a.sentiment_label or "neutral",
            "confidence": a.sentiment_confidence or 0.0,
            "published_at": a.published_at,
            "source": a.source,
        }
        for a in articles[:20]  # Ограничиваем 20 новостями
    ]

    return NewsAggregation(
        ticker=ticker,
        period_hours=hours,
        total_news=len(articles),
        avg_sentiment=round(avg_sentiment, 3),
        avg_confidence=round(avg_confidence, 3),
        sentiment_label=sentiment_label,
        positive_count=positive_count,
        negative_count=negative_count,
        neutral_count=neutral_count,
        dominant_signal=dominant_signal,
        signal_strength=round(signal_strength, 3),
        news_items=news_items,
    )


def get_news_sentiment_for_agent(db: Session, ticker: str, hours: int = 3) -> dict:
    """
    Возвращает агрегацию новостей в формате dict для агентов/байевской сети.

    Returns:
        dict: {
            "sentiment_score": float,
            "confidence": float,
            "news_count": int,
            "signal": "short"|"long"|"none",
            "signal_strength": float
        }
    """
    agg = aggregate_news(db, ticker, hours)
    return {
        "sentiment_score": agg.avg_sentiment,
        "confidence": agg.avg_confidence,
        "news_count": agg.total_news,
        "signal": agg.dominant_signal,
        "signal_strength": agg.signal_strength,
    }
