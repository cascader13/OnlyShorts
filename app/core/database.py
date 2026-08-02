from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from typing import Generator
import os
from datetime import datetime
import json
from pathlib import Path

from app.core.config import settings

# === Базовый класс для всех моделей ===
Base = declarative_base()


# === Настройка движка ===

def get_database_url():
    """
    Возвращает URL для подключения к БД.
    Если DATABASE_URL не задан - создает SQLite в папке data/
    """
    if settings.DATABASE_URL:
        return settings.DATABASE_URL

    # Создаем папку data/, если её нет
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Используем SQLite по умолчанию
    db_path = data_dir / "trading.db"
    return f"sqlite:///{db_path}"


def create_engine_with_options():
    """
    Создает движок SQLAlchemy с правильными настройками.
    """
    database_url = get_database_url()

    # Базовые настройки для всех движков
    engine_kwargs = {
        "echo": settings.DEBUG,  # Логирование SQL-запросов в режиме отладки
        "pool_pre_ping": True,  # Проверка соединения перед использованием
    }

    # Дополнительные настройки для SQLite
    if database_url.startswith("sqlite"):
        engine_kwargs.update({
            "connect_args": {
                "check_same_thread": False,  # Разрешаем использование в нескольких потоках
                "timeout": 30,  # Таймаут ожидания блокировки
            },
            "poolclass": StaticPool,  # Для SQLite используем StaticPool
        })

    engine = create_engine(database_url, **engine_kwargs)

    # === Автоматическое создание таблиц ===
    # Это сработает только при первом импорте, если таблиц нет
    # Для production используйте Alembic для миграций

    return engine


# === Глобальный движок и фабрика сессий ===

engine = create_engine_with_options()
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


# === Функции для работы с сессиями ===

def get_db() -> Generator[Session, None, None]:
    """
    Зависимость для FastAPI: создает сессию на запрос и закрывает после.
    Использование:

    @app.get("/items")
    def get_items(db: Session = Depends(get_db)):
        return db.query(Item).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """
    Контекстный менеджер для использования вне FastAPI (в скриптах, агентах).

    Использование:
    with get_db_context() as db:
        news = db.query(News).all()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# === Утилиты для работы с БД ===

def init_database():
    """
    Инициализация БД: создает все таблицы, если их нет.
    Запускать при старте приложения или вручную.
    """
    from app.models import news, decision, trade, market  # Импорт моделей для регистрации

    Base.metadata.create_all(bind=engine)

    # Миграции: добавляем новые колонки если их нет
    _migrate_database()

    print("[OK] База данных инициализирована")
    print(f"   Путь: {get_database_url()}")


def _migrate_database():
    """Простые миграции: добавление/удаление колонок (SQLite raw)."""
    import sqlite3
    db_path = get_database_url().replace("sqlite:///", "")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = OFF")

    # Проверяем колонки raw_news
    columns = {row[1] for row in conn.execute("PRAGMA table_info(raw_news)").fetchall()}

    # SQLite не даёт DROP COLUMN с FK — пересоздаём таблицу
    if "processed_id" in columns or "processed" in columns:
        conn.execute("""
            CREATE TABLE raw_news_new AS
            SELECT id, title, full_text, source, source_url, published_at,
                   external_id, is_duplicate, hash_content, created_at
            FROM raw_news
        """)
        conn.execute("DROP TABLE raw_news")
        conn.execute("ALTER TABLE raw_news_new RENAME TO raw_news")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_raw_news_external_id ON raw_news(external_id)")
        print("  [миграция] Пересоздана таблица raw_news (без processed_id)")

    # Добавляем is_processed если нет
    columns = {row[1] for row in conn.execute("PRAGMA table_info(raw_news)").fetchall()}
    if "is_processed" not in columns:
        conn.execute("ALTER TABLE raw_news ADD COLUMN is_processed BOOLEAN DEFAULT 0")
        print("  [миграция] Добавлена колонка raw_news.is_processed")

    # Проверяем колонки news_articles
    art_columns = {row[1] for row in conn.execute("PRAGMA table_info(news_articles)").fetchall()}
    if "raw_news_id" not in art_columns:
        conn.execute("ALTER TABLE news_articles ADD COLUMN raw_news_id INTEGER")
        print("  [миграция] Добавлена колонка news_articles.raw_news_id")

    conn.execute("PRAGMA foreign_keys = ON")
    conn.commit()
    conn.close()


def drop_database():
    """
    Удаляет все таблицы (ОСТОРОЖНО!).
    Используйте только для тестов и разработки.
    """
    Base.metadata.drop_all(bind=engine)
    print("[!] Все таблицы удалены")


def reset_database():
    """
    Полный сброс БД (drop + create).
    Используйте только для тестов.
    """
    drop_database()
    init_database()
    print("[RESET] База данных сброшена и пересоздана")


# === Инструменты для работы с JSON в SQLite ===

def json_serializer(obj):
    """
    Сериализатор для объектов, которые не поддерживаются json.dumps.
    Используется для полей типа JSON.
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def json_deserializer(data):
    """
    Десериализатор для JSON-данных из БД.
    Можно добавить преобразование строк в datetime при загрузке.
    """
    return json.loads(data)


# === Настройка SQLite для работы с JSON ===

# Регистрируем адаптер для datetime в SQLite
# Это позволит автоматически преобразовывать datetime в строку при сохранении в JSON-поля
from sqlalchemy import TypeDecorator, TEXT
import json


class JSONType(TypeDecorator):
    """
    Кастомный тип для хранения JSON в SQLite.
    Автоматически сериализует/десериализует Python-объекты.
    """
    impl = TEXT

    def process_bind_param(self, value, dialect):
        if value is not None:
            value = json.dumps(value, default=json_serializer)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            return json.loads(value)
        return value


# === Вспомогательные функции ===

def get_table_names():
    """Возвращает список всех таблиц в БД"""
    return [table.name for table in Base.metadata.tables.values()]


def table_exists(table_name: str) -> bool:
    """Проверяет, существует ли таблица"""
    return table_name in get_table_names()


def get_record_count(table_name: str) -> int:
    """Возвращает количество записей в таблице"""
    from sqlalchemy import text
    with get_db_context() as db:
        result = db.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
        return result.scalar()


def vacuum_database():
    """Оптимизация SQLite БД (удаление удаленных записей)"""
    if get_database_url().startswith("sqlite"):
        with get_db_context() as db:
            db.execute(text("VACUUM"))
            print("[OK] База данных оптимизирована (VACUUM)")


# === Экспорт для удобства ===

__all__ = [
    "Base",
    "engine",
    "SessionLocal",
    "get_db",
    "get_db_context",
    "init_database",
    "drop_database",
    "reset_database",
    "JSONType",
    "get_table_names",
    "table_exists",
    "get_record_count",
    "vacuum_database",
]