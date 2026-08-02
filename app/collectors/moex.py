"""
MOEXCollector — сбор новостей Московской биржи через официальный API ISS.

Эндпоинт: GET https://iss.moex.com/iss/sitenews.json
Ответ:    {"sitenews": {"columns": ["id","tag","title","published_at","modified_at"],
                         "data": [[...]]},
            "sitenews.cursor": {"columns": ["INDEX","TOTAL","PAGESIZE"], "data": [...]}}

Пагинация через параметр start (по limit строк на страницу).
У новостей нет отдельного тела — заголовок и есть контент.
"""

import logging

import requests

from app.collectors.base import BaseCollector, parse_time
from app.core.config import settings

logger = logging.getLogger(__name__)


class MOEXCollector(BaseCollector):
    """Собирает новости с сайта Московской биржи."""

    source_name = "moex"
    BASE_URL = "https://iss.moex.com/iss/sitenews.json"
    HEADERS = {"User-Agent": "Mozilla/5.0"}

    def __init__(self, db):
        super().__init__(db)
        self.pages = settings.MOEX_NEWS_PAGES
        # Лимит строк на страницу (API поддерживает до 100)
        self.limit = min(settings.MOEX_NEWS_LIMIT, 100)
        self.timeout = 15

    def collect(self) -> int:
        """
        Собирает последние новости с нескольких страниц.

        Returns:
            int: число новых записей, сохранённых в raw_news за этот проход.
        """
        start = 0
        for _ in range(self.pages):
            try:
                params = {"start": start, "limit": self.limit}
                response = requests.get(
                    self.BASE_URL, params=params, headers=self.HEADERS, timeout=self.timeout
                )
                if response.status_code != 200:
                    logger.warning(
                        "MOEX: статус %d при start=%d", response.status_code, start
                    )
                    break

                data = response.json()
                rows = data.get("sitenews", {}).get("data", [])
                columns = data.get("sitenews", {}).get("columns", [])

                for row in rows:
                    self._parse_item(dict(zip(columns, row)))

                # Если получили меньше, чем просили — это последняя страница
                if len(rows) < self.limit:
                    break
                start += self.limit

            except requests.RequestException as e:
                logger.error("Ошибка сбора MOEX: %s", e)
                break
            except ValueError as e:
                logger.error("Некорректный JSON от MOEX: %s", e)
                break

        saved = self.commit()
        logger.info("MOEX: сохранено %d новостей", saved)
        return saved

    def _parse_item(self, item: dict):
        """Разбирает одну строку sitenews и добавляет её в raw_news."""
        title = item.get("title") or ""
        if not title.strip():
            return

        published_at = parse_time(item.get("published_at"))
        item_id = item.get("id")

        self.save(
            title=title,
            full_text=title,
            source_url=None,
            published_at=published_at,
            external_id=str(item_id) if item_id is not None else None,
        )
