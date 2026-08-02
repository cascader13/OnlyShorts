"""
PulseCollector — сбор постов Т-Пульса (tbank.ru) по отслеживаемым тикерам.

REST API: GET https://www.tbank.ru/api/invest-gw/social/v1/post/instrument/{TICKER}
Ответ: {"payload": {"items": [...], "nextCursor": "...", "hasNext": bool}}

Элемент items:
  id            - UUID поста (стабильный external_id для дедупликации)
  content.text  - текст поста (основной контент)
  text          - краткий текст (часто пустой)
  inserted      - время публикации, ISO8601
  instruments[] - привязанные инструменты с полем ticker
  likesCount, commentsCount - метрики активности
"""

import logging
from datetime import datetime
from typing import Optional

import requests

from app.collectors.base import BaseCollector, parse_time
from app.core.config import settings

logger = logging.getLogger(__name__)


class PulseCollector(BaseCollector):
    """Собирает посты из Т-Пульса по каждому тикеру из конфига."""

    source_name = "pulse"
    BASE_URL = "https://www.tbank.ru/api/invest-gw/social/v1/post/instrument/"

    def __init__(self, db):
        super().__init__(db)
        self.tickers = settings.TRACKED_TICKERS
        self.max_pages = settings.PULSE_MAX_PAGES
        self.page_size = settings.PULSE_PAGE_SIZE
        self.delay = settings.PULSE_DELAY
        self.timeout = 10

    def collect(self) -> int:
        """
        Собирает посты по всем тикерам.

        Returns:
            int: число новых записей, сохранённых в raw_news за этот проход.
        """
        for ticker in self.tickers:
            self._collect_ticker(ticker)
        saved = self.commit()
        logger.info("Пульс: сохранено %d новых постов", saved)
        return saved

    def _collect_ticker(self, ticker: str):
        """Собирает посты по одному тикеру с пагинацией по nextCursor."""
        cursor = None
        for _ in range(self.max_pages):
            try:
                params = {"page_size": self.page_size}
                if cursor:
                    params["cursor"] = cursor

                response = requests.get(
                    self.BASE_URL + ticker, params=params, timeout=self.timeout
                )
                if response.status_code != 200:
                    logger.warning(
                        "Пульс %s: статус %d", ticker, response.status_code
                    )
                    break

                payload = response.json().get("payload", {})
                items = payload.get("items", [])

                for item in items:
                    self._parse_item(item, ticker)

                if not payload.get("hasNext"):
                    break
                cursor = payload.get("nextCursor")
                if not cursor:
                    break

            except requests.RequestException as e:
                logger.error("Ошибка сбора Пульса для %s: %s", ticker, e)
                break
            except ValueError as e:
                logger.error("Некорректный JSON от Пульса для %s: %s", ticker, e)
                break

            self.throttle(self.delay)

    def _parse_item(self, item: dict, ticker: str):
        """Разбирает один пост и добавляет его в raw_news."""
        text = (item.get("content") or {}).get("text") or item.get("text") or ""
        if not text.strip():
            return

        published_at = parse_time(item.get("inserted"))

        self.save(
            title=text[:100],
            full_text=text,
            source_url=f"{self.BASE_URL}{ticker}",
            published_at=published_at,
            external_id=item.get("id"),
        )
