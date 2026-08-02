"""
Точка входа агента сбора новостей.

Запуск:
    python -m app.main            # бесконечный цикл сбора (реальное время)
    python -m app.main --once     # один проход сбора (для проверки/CI)
"""

import argparse
import logging
import sys

from app.services.scheduler import CollectorScheduler


def setup_logging(debug: bool = False):
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Сбор новостей для шорт-агента (Т-Пульс, РБК, MOEX)"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Один проход сбора и выход (для проверки)",
    )
    args = parser.parse_args(argv)

    from app.core.config import settings

    setup_logging(settings.DEBUG)

    scheduler = CollectorScheduler()
    if args.once:
        stats = scheduler.run_once()
        print("Статистика сбора:", stats)
        return 0

    scheduler.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
