"""
Единый модуль времени проекта.

Конвенция: БД хранит НАИВНОЕ московское время (UTC+3). МСК — фиксированный
сдвиг без перехода на летнее время, поэтому UTC+3 не зависит от сезона.

- Все сервисы пишут время через msk_now() (аналог бывшего datetime.utcnow).
- Все timezone-aware значения от SDK/API приводятся к naive МСК через
  to_naive_msk().
- Отображение просто форматирует значение из БД, без конвертации.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

# Московское время: UTC+3, фиксированный сдвиг
MSK = timezone(timedelta(hours=3))


def msk_now() -> datetime:
    """Наивное МСК-время (аналог datetime.utcnow в старой UTC-конвенции)."""
    return datetime.now(MSK).replace(tzinfo=None)


def to_naive_msk(dt: Optional[datetime]) -> Optional[datetime]:
    """Переводит timezone-aware datetime в naive МСК; naive оставляет как есть.

    naive-значение трактуем как уже сохранённое в МСК (конвенция проекта):
    источник уже отдал время по Москве (например, MOEX ISS).
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(MSK).replace(tzinfo=None)


def naive_utc_to_msk(dt: Optional[datetime]) -> Optional[datetime]:
    """Naive datetime в UTC -> naive МСК (сдвиг +3ч).

    Для источников, отдающих время без offset в UTC (например, struct_time
    из feedparser — всегда UTC). Значение с tzinfo -> обычный to_naive_msk.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(MSK).replace(tzinfo=None)
