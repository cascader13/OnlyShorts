#!/usr/bin/env python
"""
Скрипт для инициализации базы данных.
Запуск: python init_db.py
"""

from app.core.database import init_database, get_table_names

if __name__ == "__main__":
    print("Инициализация базы данных...")
    init_database()

    print("\nСозданные таблицы:")
    for table in get_table_names():
        print(f"  - {table}")

    print("\nГотово!")