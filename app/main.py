"""
Точка входа агента сбора новостей + дашборд.

Запуск:
    python -m app.main                 # планировщик + Streamlit дашборд
    python -m app.main --once          # один проход сбора и выход
    python -m app.main --no-dashboard  # только планировщик (без дашборда)
    python -m app.main --scheduler-only # только цикл сбора (блокирующий)
"""

import argparse
import logging
import signal
import subprocess
import sys
import threading
import time

from app.services.scheduler import CollectorScheduler, start_background_collector


def setup_logging(debug: bool = False):
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def run_streamlit():
    """Запускает Streamlit как дочерний процесс."""
    logger = logging.getLogger(__name__)
    logger.info("Запуск дашборда: http://localhost:8501")
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "app/frontend.py",
             "--server.headless", "true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return proc
    except Exception:
        logger.exception("Не удалось запустить Streamlit")
        return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Агент сбора новостей + дашборд"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Один проход сбора и выход",
    )
    parser.add_argument(
        "--no-dashboard", action="store_true",
        help="Запустить только планировщик (без дашборда)",
    )
    parser.add_argument(
        "--scheduler-only", action="store_true",
        help="Блокирующий цикл планировщика (без дашборда)",
    )
    parser.add_argument(
        "--port", type=int, default=8501,
        help="Порт дашборда (по умолчанию 8501)",
    )
    args = parser.parse_args(argv)

    from app.core.config import settings

    setup_logging(settings.DEBUG)
    logger = logging.getLogger(__name__)

    # === Режим: один проход ===
    if args.once:
        scheduler = CollectorScheduler()
        stats = scheduler.run_once()
        print("Статистика сбора:", stats)
        return 0

    # === Режим: только планировщик (блокирующий) ===
    if args.scheduler_only:
        logger.info("Запуск планировщика (интервал %d сек)", settings.COLLECT_INTERVAL_SECONDS)
        scheduler = CollectorScheduler()
        scheduler.run()
        return 0

    # === Режим: планировщик + дашборд (по умолчанию) ===
    logger.info("="*60)
    logger.info("Запуск PantsOnly: планировщик + дашборд")
    logger.info("="*60)

    # Запускаем фоновый сбор
    scheduler = start_background_collector(settings.COLLECT_INTERVAL_SECONDS)
    logger.info("Фоновый сбор запущен (каждые %d сек)", scheduler.interval)

    # Запускаем Streamlit
    streamlit_proc = None
    if not args.no_dashboard:
        streamlit_proc = run_streamlit()

    # Ждём завершения (Ctrl+C)
    logger.info("Нажмите Ctrl+C для остановки")

    def _on_stop(sig, frame):
        logger.info("Получен сигнал остановки...")
        scheduler.stop()
        if streamlit_proc:
            streamlit_proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, _on_stop)
    signal.signal(signal.SIGTERM, _on_stop)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        _on_stop(None, None)

    return 0


if __name__ == "__main__":
    sys.exit(main())
