import os
from dotenv import load_dotenv
from pathlib import Path

# Загружаем .env файл
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)


class Settings:
    """Настройки приложения"""

    # === База данных ===
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        ""  # Если пусто, будет создана SQLite в data/trading.db
    )

    # === Режим отладки ===
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    # === API ключи ===
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_MODEL: str = os.getenv(
        "OPENROUTER_MODEL",
        "google/gemini-2.0-flash-exp:free"
    )

    # === LLM (локальный шлюз OmniRoute или облачный OpenRouter) ===
    # OmniRoute: http://localhost:20128/ (без ключа)
    # OpenRouter: https://openrouter.ai/api/v1/ (с ключом sk-or-...)
    LLM_BASE_URL: str = os.getenv(
        "LLM_BASE_URL",
        "http://localhost:20128"
    )
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")  # пусто для локального шлюза
    LLM_MODEL: str = os.getenv("LLM_MODEL", "auto/best-reasoning")  # модель по умолчанию
    # Хинт для reasoning-моделей (DeepSeek и т.п.): ограничивает «размышления»,
    # иначе модель тратит весь max_tokens на reasoning_content и возвращает
    # пустой content. Пусто = параметр не отправляется.
    LLM_REASONING_EFFORT: str = os.getenv("LLM_REASONING_EFFORT", "low")

    # === T-Invest ===
    TINKOFF_TOKEN: str = os.getenv(
        "TINKOFF_TOKEN",
        "Bearer TBankSandboxToken"  # Публичный токен для песочницы
    )
    TINKOFF_SANDBOX: bool = os.getenv("TINKOFF_SANDBOX", "True").lower() == "true"
    TINKOFF_SANDBOX_ADDRESS: str = "sandbox-invest-public-api.tbank.ru:443"
    TINKOFF_LIVE_ADDRESS: str = "invest-public-api.tbank.ru:443"

    # gRPC-канал t_tech.invest читает SSL_TBANK_VERIFY в create_channel
    # (см. t_tech/invest/channels.py). Без "true" подключение падает с
    # CERTIFICATE_VERIFY_FAILED: SDK использует встроенный RussianTrustedRootCA.pem.
    # Выставляем в os.environ при импорте config — ДО конструирования любого
    # Client (канал создаётся в Client.__init__). setdefault уважает явный "false".
    SSL_TBANK_VERIFY: bool = os.getenv("SSL_TBANK_VERIFY", "true").lower() == "true"
    os.environ["SSL_TBANK_VERIFY"] = "true" if SSL_TBANK_VERIFY else "false"

    # === Рыночные данные (свечи) ===
    # Какие таймфреймы собирать (через запятую): 1d,1h,15m
    MARKET_TIMEFRAMES: list = os.getenv("MARKET_TIMEFRAMES", "1d,1h,15m").split(",")

    # === Настройки агента ===
    AGENT_INTERVAL_MINUTES: int = int(os.getenv("AGENT_INTERVAL_MINUTES", "5"))
    # Все тикеры словаря эвристики (BASE_TICKERS + алиасы) — для сбора данных
    # по максимальному числу инструментов. Некоторые могут быть недоступны в
    # песочнице (делистинг) — коллекторы пропускают их по одному.
    TRACKED_TICKERS: list = os.getenv(
        "TRACKED_TICKERS",
        "SBER,GAZP,VTBR,LKOH,YNDX,TATN,NVTK,ROSN,MGNT,CHMF,GMKN,PLZL,NLMK,"
        "AFLT,ALRS,PHOR,POLY,SNGS,SNGR,RUAL,MTSS,RTKM,MAGN,AKRN,RASP,MOEX,"
        "RENI,SBERP,TCSG,IRAO,HYDR,FEES,MTLR,MVID,FIVE,LENT,MGTX,VKCO,"
        "MDSB,GEMC,GLTR,SELG",
    ).split(",")

    # === Режим торговли ===
    # True — бумажный режим: ордера НЕ отправляются брокеру, все сделки
    # только пишутся в БД (симуляция исполнения по текущей цене).
    # False — реальная торговля на T-Invest (песочница или live).
    PAPER_TRADING: bool = os.getenv("PAPER_TRADING", "True").lower() == "true"

    # Мастер-переключатель торгового агента: если False, агент не создаётся
    # и позиции не открываются (сбор и анализ продолжаются как обычно).
    TRADING_ENABLED: bool = os.getenv("TRADING_ENABLED", "False").lower() == "true"

    # Счёт, на котором торгует агент. Пусто — берётся первый аккаунт брокера.
    ACCOUNT_ID: str = os.getenv("ACCOUNT_ID", "")

    # === Риск-менеджмент ===
    # Лимиты, которые агент проверяет перед открытием шорта.
    MAX_OPEN_POSITIONS: int = int(os.getenv("MAX_OPEN_POSITIONS", "3"))
    DAILY_LOSS_LIMIT_PERCENT: float = float(os.getenv("DAILY_LOSS_LIMIT_PERCENT", "5.0"))
    TRAILING_STOP_ACTIVATION_PERCENT: float = float(
        os.getenv("TRAILING_STOP_ACTIVATION_PERCENT", "2.0")
    )
    TRAILING_STOP_DISTANCE_PERCENT: float = float(
        os.getenv("TRAILING_STOP_DISTANCE_PERCENT", "1.5")
    )

    # === Настройки сбора новостей ===
    COLLECT_INTERVAL_SECONDS: int = int(os.getenv("COLLECT_INTERVAL_SECONDS", "300"))
    PULSE_MAX_PAGES: int = int(os.getenv("PULSE_MAX_PAGES", "2"))
    PULSE_PAGE_SIZE: int = int(os.getenv("PULSE_PAGE_SIZE", "30"))
    PULSE_DELAY: float = float(os.getenv("PULSE_DELAY", "1"))
    MOEX_NEWS_LIMIT: int = int(os.getenv("MOEX_NEWS_LIMIT", "100"))
    MOEX_NEWS_PAGES: int = int(os.getenv("MOEX_NEWS_PAGES", "3"))
    RBC_MAX_ENTRIES: int = int(os.getenv("RBC_MAX_ENTRIES", "30"))

    # === Датасет для обучения ===
    # Мастер-флаг live-захвата снапшотов в цикле сбора (capture_live + label_expired).
    SNAPSHOT_ENABLED: bool = os.getenv("SNAPSHOT_ENABLED", "True").lower() == "true"
    SNAPSHOT_TIMEFRAME: str = os.getenv("SNAPSHOT_TIMEFRAME", "15m")
    NEWS_WINDOW_HOURS: int = int(os.getenv("NEWS_WINDOW_HOURS", "2"))
    TARGET_HOURS: int = int(os.getenv("TARGET_HOURS", "2"))
    SNAPSHOT_STEP_MINUTES: int = int(os.getenv("SNAPSHOT_STEP_MINUTES", "60"))
    # Lookback для refresh_news_features: статьи, обработанные за последние N
    # часов, «догоняют» уже записанные снапшоты, чьё окно публикаций они
    # накрыли. Пересчёт идемпотентен, поэтому окно можно держать с запасом
    # (покрывает и пропущенные циклы после рестарта).
    SNAPSHOT_REFRESH_HOURS: int = int(os.getenv("SNAPSHOT_REFRESH_HOURS", "6"))

    # === Анализатор новостей ===
    # "llm" — анализ через LLM (текущий путь); "heuristic" — лексиконный анализатор без LLM.
    NEWS_ANALYZER: str = os.getenv("NEWS_ANALYZER", "llm").lower().strip()

    # === Анализ новостей через LLM: размер батча и покрытие за цикл ===
    # NEWS_BATCH_SIZE — сколько новостей в одном запросе к LLM.
    # NEWS_MAX_PER_CYCLE — максимум новостей за один цикл сбора (несколько
    #   батчей подряд; иначе очередь отстаёт от потока новостей).
    # NEWS_MAX_SECONDS_PER_CYCLE — бюджет времени на анализ в одном цикле
    #   (мягкий guard, чтобы не «съесть» весь интервал сбора).
    # NEWS_MAX_TOKENS — лимит токенов ответа на один батч (растёт с батчем).
    NEWS_BATCH_SIZE: int = int(os.getenv("NEWS_BATCH_SIZE", "20"))
    NEWS_MAX_PER_CYCLE: int = int(os.getenv("NEWS_MAX_PER_CYCLE", "100"))
    NEWS_MAX_SECONDS_PER_CYCLE: int = int(os.getenv("NEWS_MAX_SECONDS_PER_CYCLE", "60"))
    NEWS_MAX_TOKENS: int = int(os.getenv("NEWS_MAX_TOKENS", "16384"))
    # Повторные попытки анализа батча, если LLM вернул некорректный JSON
    # (всего запросов = NEWS_LLM_RETRIES + 1). Ретрай подсказывает модели,
    # что прошлый ответ не распарсился.
    NEWS_LLM_RETRIES: int = int(os.getenv("NEWS_LLM_RETRIES", "2"))

    # === Риск-менеджмент ===
    MAX_POSITION_SIZE_PERCENT: float = float(os.getenv("MAX_POSITION_SIZE_PERCENT", "5"))
    STOP_LOSS_PERCENT: float = float(os.getenv("STOP_LOSS_PERCENT", "3"))
    TAKE_PROFIT_PERCENT: float = float(os.getenv("TAKE_PROFIT_PERCENT", "5"))
    MAX_HOLD_HOURS: int = int(os.getenv("MAX_HOLD_HOURS", "4"))

    # === Пороги для сигналов ===
    SENTIMENT_THRESHOLD: float = float(os.getenv("SENTIMENT_THRESHOLD", "-0.4"))
    MIN_NEWS_COUNT: int = int(os.getenv("MIN_NEWS_COUNT", "3"))
    RSI_OVERBOUGHT: float = float(os.getenv("RSI_OVERBOUGHT", "70"))
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))


settings = Settings()