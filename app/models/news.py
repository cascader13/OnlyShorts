from datetime import datetime
from typing import Any

from sqlalchemy import String, DateTime, Text, Boolean, Float, Integer, ColumnElement
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base


class RawNews(Base):
    """
    Модель для хранения сырых новостей из парсеров.
    Используется для проверки дубликатов перед обработкой.
    """
    __tablename__ = "raw_news"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(512), comment="Оригинальный заголовок")
    full_text: Mapped[str] = mapped_column(Text, comment="Оригинальный текст")

    # Привязка к источнику и времени
    source: Mapped[str] = mapped_column(String(100), comment="Источник: pulse, rbc, moex, etc.")
    source_url: Mapped[str] = mapped_column(String(512), nullable=True, comment="Ссылка на оригинал")
    published_at: Mapped[datetime] = mapped_column(DateTime, nullable=True, comment="Время публикации источника")

    # Стабильный ID записи в источнике (UUID поста Пульса, числовой ID новости MOEX)
    external_id: Mapped[str] = mapped_column(
        String(128), nullable=True, index=True,
        comment="Стабильный ID записи в источнике для дедупликации"
    )

    # Флаг обработки LLM: False = не обработана, True = обработана
    is_processed: Mapped[bool] = mapped_column(
        Boolean, default=False, index=True,
        comment="Обработана ли новость LLM"
    )

    # Флаг для дедупликации
    is_duplicate: Mapped[bool] = mapped_column(Boolean, default=False, comment="Пометка о дубликате")
    hash_content: Mapped[str] = mapped_column(String(64), nullable=True,
                                              comment="Хеш текста для быстрого поиска дублей")

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, comment="Время сохранения в БД")

    def __repr__(self) -> str:
        return f"<RawNews(id={self.id}, source='{self.source}', title='{self.title[:50]}...')>"


class NewsArticle(Base):
    """
    Модель для хранения обработанных и обогащенных новостей (LLM).
    """
    __tablename__ = "news_articles"

    id: Mapped[int] = mapped_column(primary_key=True)

    # Основная информация
    title: Mapped[str] = mapped_column(String(512), comment="Обобщенный заголовок")
    full_text: Mapped[str] = mapped_column(Text, comment="Полный, объединенный текст")
    summary: Mapped[str] = mapped_column(Text, nullable=True, comment="Краткое содержание от LLM")

    #.sentiment как числовой скор
    sentiment_score: Mapped[float] = mapped_column(
        Float, nullable=True, comment="Тональность: -1 (негатив) .. +1 (позитив)"
    )
    sentiment_label: Mapped[str] = mapped_column(
        String(20), nullable=True, comment="Метка: negative, neutral, positive"
    )
    sentiment_confidence: Mapped[float] = mapped_column(
        Float, nullable=True, comment="Уверенность LLM в оценке 0..1"
    )

    # Привязка к тикеру
    tickers: Mapped[str] = mapped_column(
        String(500), nullable=True, comment="Список тикеров через запятую"
    )
    primary_ticker: Mapped[str] = mapped_column(
        String(20), nullable=True, index=True, comment="Основной тикер (первый в списке)"
    )

    tags: Mapped[str] = mapped_column(String(500), nullable=True, comment="Теги от LLM (industry_tag)")
    is_ai_generated: Mapped[bool] = mapped_column(Boolean, default=False, comment="AI-сгенерированный контент")

    # Источник и время публикации
    source: Mapped[str] = mapped_column(String(100), comment="Источник")
    published_at: Mapped[datetime] = mapped_column(DateTime, nullable=True, comment="Время публикации")

    # Ссылка на исходную новость (для просмотра оригинала)
    raw_news_id: Mapped[int] = mapped_column(Integer, nullable=True, comment="ID исходной новости в raw_news")

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, comment="Время сохранения")

    def __repr__(self) -> str:
        return f"<NewsArticle(id={self.id}, ticker='{self.primary_ticker}', sentiment={self.sentiment_score:.2f})>"

    @property
    def is_negative(self) -> ColumnElement[bool] | Any:
        """метод для проверки негатива"""
        return self.sentiment_score is not None and self.sentiment_score < -0.3

    @property
    def is_positive(self) -> ColumnElement[bool] | Any:
        """метод для проверки позитива"""
        return self.sentiment_score is not None and self.sentiment_score > 0.3
