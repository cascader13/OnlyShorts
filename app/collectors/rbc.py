"""
RBCCollector — сбор новостей РБК.

Прямые RSS-ленты РБК (rbc.ru/rss/) закрыты защитой Qrator (401/403) для
некоторых сетей. Поэтому коллектор работает в два этапа:
1. Пробует официальные RSS-ленты РБК.
2. Если все заблокированы или пустые — фолбэк на Google News RSS по site:rbc.ru.

Записи в обоих случаях сохраняются с source="rbc".
"""

import logging
from datetime import datetime
from time import struct_time

import feedparser

from app.collectors.base import BaseCollector
from app.core.config import settings
from app.core.timeutil import naive_utc_to_msk, to_naive_msk

logger = logging.getLogger(__name__)


class RBCCollector(BaseCollector):
    """Собирает новости РБК с фолбэком на Google News."""

    source_name = "rbc"

    # Прямые RSS-ленты РБК в порядке предпочтения
    RSS_FEEDS = [
        "https://www.rbc.ru/rss/",
        "https://www.rbc.ru/rss/finance/",
        "https://www.rbc.ru/rss/main/",
    ]

    # Фолбэк: Google News по сайту rbc.ru
    GOOGLE_NEWS_URL = (
        "https://news.google.com/rss/search?q=site:rbc.ru&hl=ru&gl=RU&ceid=RU:ru"
    )

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ru-RU,ru;q=0.9",
    }

    def collect(self) -> int:
        """
        Собирает новости РБК, при необходимости переключаясь на фолбэк.

        Returns:
            int: число новых записей, сохранённых в raw_news за этот проход.
        """
        feed = self._try_direct_feeds()
        used_fallback = False

        if feed is None:
            fallback_feed = self._parse_rss(self.GOOGLE_NEWS_URL)
            if fallback_feed and fallback_feed.entries:
                feed = fallback_feed
                used_fallback = True
                logger.info("RBC: прямые RSS заблокированы, используем Google News")

        if feed is not None:
            for entry in feed.entries[:settings.RBC_MAX_ENTRIES]:
                self._parse_entry(entry, used_fallback)

        saved = self.commit()
        logger.info("RBC: сохранено %d новостей", saved)
        return saved

    def _try_direct_feeds(self):
        """Пробует прямые RSS-ленты РБК, возвращает первую рабочую."""
        for url in self.RSS_FEEDS:
            feed = self._parse_rss(url)
            if feed and feed.entries:
                logger.info("RBC: работает прямой RSS %s", url)
                return feed
        return None

    def _parse_rss(self, url):
        """Парсит RSS-ленту; при ошибке возвращает None."""
        try:
            return feedparser.parse(url, request_headers=self.HEADERS)
        except Exception as e:
            logger.warning("RBC: ошибка парсинга %s: %s", url, e)
            return None

    def _parse_entry(self, entry, used_fallback: bool):
        """Разбирает одну запись RSS и добавляет её в raw_news."""
        title = entry.get("title") or ""
        if used_fallback:
            # Google News добавляет суффикс " - rbc.ru"
            title = title.removesuffix(" - rbc.ru").strip()

        summary = entry.get("description") or entry.get("summary") or ""
        if not title and not summary:
            return

        full_text = summary or title
        published_at = self._parse_published(entry)

        self.save(
            title=title or full_text[:100],
            full_text=full_text,
            source_url=entry.get("link") or "",
            published_at=published_at,
            external_id=entry.get("id") or entry.get("guid"),
        )

    @staticmethod
    def _parse_published(entry) -> datetime:
        """Разбирает время публикации из RSS и нормализует в naive МСК.

        feedparser отдаёт `published_parsed` как struct_time в UTC — сдвигаем
        в МСК (naive_utc_to_msk). Строковый фолбэк (RFC822/ISO) приводим
        к МСК через to_naive_msk.
        """
        parsed = entry.get("published_parsed") or entry.get("updated_parsed")
        if parsed and isinstance(parsed, struct_time):
            return naive_utc_to_msk(datetime(*parsed[:6]))
        published = entry.get("published") or entry.get("updated") or ""
        if published:
            try:
                return to_naive_msk(
                    datetime.fromisoformat(published.replace("Z", "+00:00"))
                )
            except ValueError:
                return None
        return None
