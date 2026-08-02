"""
FastAPI-приложение для инспекции собранных новостей.

Запуск:
    uvicorn app.api.t_news:app --reload --port 8000

При старте сервера автоматически запускается фоновый цикл сбора новостей
(каждые COLLECT_INTERVAL_SECONDS, по умолчанию 300 = 5 минут). Пока сервер
работает — данные пишутся в raw_news.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.models.news import RawNews
from app.services.scheduler import start_background_collector

logger = logging.getLogger(__name__)

# Делает активность фонового сбора видимой при запуске через uvicorn
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Фоновый сборщик данных (запускается вместе с API и работает, пока сервер активен)
_collector = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Запускает фоновый цикл сбора новостей при старте API и останавливает при выключении."""
    global _collector
    _collector = start_background_collector(settings.COLLECT_INTERVAL_SECONDS)
    logger.info(
        "Фоновый сбор данных запущен (интервал %d сек), данные будут обновляться автоматически",
        _collector.interval,
    )
    try:
        yield
    finally:
        _collector.stop()
        logger.info("Фоновый сбор данных остановлен")


app = FastAPI(
    title="T-News Collector API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/news/recent")
def recent_news(limit: int = 20, db: Session = Depends(get_db)):
    """Возвращает последние собранные новости из raw_news."""
    rows = (
        db.query(RawNews)
        .order_by(RawNews.created_at.desc())
        .limit(min(limit, 200))
        .all()
    )
    return [
        {
            "id": row.id,
            "source": row.source,
            "title": row.title,
            "published_at": row.published_at.isoformat() if row.published_at else None,
            "source_url": row.source_url,
            "external_id": row.external_id,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
        for row in rows
    ]


@app.get("/news/stats")
def news_stats(db: Session = Depends(get_db)):
    """Статистика по источникам: сколько записей собрано каждым."""
    from sqlalchemy import func

    rows = db.query(RawNews.source, func.count(RawNews.id)).group_by(
        RawNews.source
    ).all()
    return {"by_source": dict(rows), "total": sum(count for _, count in rows)}
