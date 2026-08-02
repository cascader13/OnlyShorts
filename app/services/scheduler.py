"""
CollectorScheduler — цикл реального времени для сбора новостей.

Запускает collect_all() с заданным интервалом, корректно завершается
по сигналу (Ctrl+C / SIGTERM).
"""

import logging
import signal
import threading
import time

from app.core.config import settings
from app.services.data_collector import DataCollector
from app.core.database import get_db_context

logger = logging.getLogger(__name__)


class CollectorScheduler:
    """Периодически запускает сбор новостей из всех источников."""

    def __init__(self, interval_seconds: int = None):
        self.interval = interval_seconds or settings.COLLECT_INTERVAL_SECONDS
        self._stopped = False

    def _install_signal_handlers(self):
        try:
            signal.signal(signal.SIGINT, self._on_stop)
            signal.signal(signal.SIGTERM, self._on_stop)
        except ValueError:
            # Сигналы доступны только в главном потоке
            logger.warning("Сигналы недоступны, остановка только через Ctrl+C")

    def _on_stop(self, *args):
        logger.info("Получен сигнал остановки")
        self._stopped = True

    def run_once(self) -> dict:
        """Один проход сбора. Возвращает статистику."""
        with get_db_context() as db:
            return DataCollector(db).collect_all()

    def run(self):
        """Запускает бесконечный цикл сбора."""
        self._install_signal_handlers()
        logger.info("Запуск цикла сбора, интервал: %d сек", self.interval)

        while not self._stopped:
            started = time.monotonic()
            try:
                stats = self.run_once()
                logger.info("Проход завершен: %s", stats)
            except Exception:
                logger.exception("Ошибка в проходе сбора")

            elapsed = time.monotonic() - started
            sleep_for = max(self.interval - elapsed, 1)

            # Спим небольшими порциями, чтобы быстро реагировать на остановку
            for _ in range(int(sleep_for)):
                if self._stopped:
                    break
                time.sleep(1)

        logger.info("Цикл сбора остановлен")

    def stop(self):
        """Запрашивает остановку цикла сбора (для фонового запуска)."""
        self._on_stop()
        thread = getattr(self, "_thread", None)
        if thread is not None:
            thread.join(timeout=5)


def start_background_collector(interval_seconds: int = None) -> CollectorScheduler:
    """
    Запускает цикл сбора новостей в фоновом daemon-потоке.

    Поток завершается вместе с основным процессом. Остановить досрочно
    можно через метод stop() возвращённого объекта.

    Пока процесс (API-сервер, скрипт) активен — сбор выполняется каждые
    interval_seconds секунд, новые данные пишутся в raw_news.
    """
    scheduler = CollectorScheduler(interval_seconds=interval_seconds)
    thread = threading.Thread(
        target=scheduler.run,
        name="collector-scheduler",
        daemon=True,
    )
    scheduler._thread = thread
    thread.start()
    logger.info(
        "Фоновый сбор запущен, интервал: %d сек",
        scheduler.interval,
    )
    return scheduler
