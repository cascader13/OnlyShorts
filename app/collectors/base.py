"""
BaseCollector — общая логика сохранения сырых новостей в БД.

Отвечает за:
1. Нормализацию текста для хеширования
2. Дедупликацию: по external_id (стабильный ID источника), иначе по md5 текста
3. Пакетную запись в RawNews с одним коммитом за проход

Наследники задают `source_name` и реализуют `collect()`.
"""

import hashlib
import logging
import time
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from app.core.timeutil import msk_now, to_naive_msk
from app.models.news import RawNews

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Нормализует текст: схлопывает пробелы и приводит к нижнему регистру."""
    return " ".join(text.split()).strip().lower()


def parse_time(value: Optional[str]) -> Optional[datetime]:
    """
    Разбирает время публикации из строки ISO8601 и нормализует в naive МСК.

    Значение с offset (Пульс: "+03:00", "Z") приводится к московскому времени;
    naive-значение (MOEX ISS отдаёт уже по Москве) остаётся как есть.
    Возвращает None, если значение пустое или не удалось разобрать.
    """
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return to_naive_msk(dt)


class BaseCollector:
    """Базовый класс коллектора: собирает записи и сохраняет их в raw_news."""

    # Имя источника, под которым записи попадают в RawNews.source
    source_name: str = "base"

    def __init__(self, db: Session):
        self.db = db
        # Кэши уже сохранённых external_id и хешей (на время жизни процесса)
        self._known_external_ids: set = set()
        self._known_hashes: set = set()
        # Счётчик новых записей за текущий проход
        self._saved_count = 0
        self._load_known()

    def _load_known(self):
        """Загружает из БД уже сохранённые external_id и хеши по этому источнику."""
        try:
            rows = self.db.query(RawNews.external_id, RawNews.hash_content).filter(
                RawNews.source == self.source_name
            ).all()
            self._known_external_ids = {r[0] for r in rows if r[0]}
            self._known_hashes = {r[1] for r in rows if r[1]}
            logger.info(
                "Дедупликация %s: загружено %d external_id, %d хешей",
                self.source_name,
                len(self._known_external_ids),
                len(self._known_hashes),
            )
        except Exception as e:
            logger.warning(
                "Не удалось загрузить кэш дедупликации для %s: %s",
                self.source_name, e,
            )

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.md5(normalize_text(text).encode("utf-8")).hexdigest()

    def _db_has_duplicate(self, external_id: Optional[str], text_hash: str) -> bool:
        """Проверяет дубликат в БД (для случая, когда кэш устарел)."""
        query = self.db.query(RawNews.id).filter(RawNews.source == self.source_name)
        if external_id:
            if query.filter(RawNews.external_id == external_id).first():
                return True
        return query.filter(RawNews.hash_content == text_hash).first() is not None

    def save(
        self,
        title: str,
        full_text: str,
        source_url: Optional[str] = None,
        published_at: Optional[datetime] = None,
        external_id: Optional[str] = None,
    ) -> bool:
        """
        Добавляет сырую новость в сессию (без коммита).

        Returns:
            bool: True если запись новая и добавлена, False если дубликат или пустой текст.
        """
        text = (full_text or title or "").strip()
        if not text:
            return False

        text_hash = self._hash_text(text)

        # 1. Проверка по кэшу в памяти
        if external_id and external_id in self._known_external_ids:
            return False
        if text_hash in self._known_hashes:
            return False

        # 2. Проверка в БД (переживает перезапуск / другие процессы)
        if self._db_has_duplicate(external_id, text_hash):
            if external_id:
                self._known_external_ids.add(external_id)
            self._known_hashes.add(text_hash)
            return False

        raw = RawNews(
            title=(title or text)[:512],
            full_text=text,
            source=self.source_name,
            source_url=source_url,
            published_at=published_at,
            external_id=external_id,
            hash_content=text_hash,
            is_duplicate=False,
            created_at=msk_now(),
        )
        self.db.add(raw)

        if external_id:
            self._known_external_ids.add(external_id)
        self._known_hashes.add(text_hash)
        self._saved_count += 1
        return True

    def commit(self) -> int:
        """Коммитит накопленные записи и возвращает число новых записей за проход."""
        count = self._saved_count
        if count:
            self.db.commit()
        self._saved_count = 0
        return count

    def throttle(self, seconds: float = 1.0):
        """Пауза между запросами, чтобы не нагружать источник."""
        time.sleep(seconds)
