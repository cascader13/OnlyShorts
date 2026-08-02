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

    # === T-Invest ===
    TINKOFF_TOKEN: str = os.getenv(
        "TINKOFF_TOKEN",
        "Bearer TBankSandboxToken"  # Публичный токен для песочницы
    )
    TINKOFF_SANDBOX: bool = os.getenv("TINKOFF_SANDBOX", "True").lower() == "true"
    TINKOFF_SANDBOX_ADDRESS: str = "sandbox-invest-public-api.tbank.ru:443"

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
    TRACKED_TICKERS: list = os.getenv("TRACKED_TICKERS", "SBER,GAZP,VTBR,LKOH,YNDX").split(",")

    # === Настройки сбора новостей ===
    COLLECT_INTERVAL_SECONDS: int = int(os.getenv("COLLECT_INTERVAL_SECONDS", "300"))
    PULSE_MAX_PAGES: int = int(os.getenv("PULSE_MAX_PAGES", "2"))
    PULSE_PAGE_SIZE: int = int(os.getenv("PULSE_PAGE_SIZE", "30"))
    PULSE_DELAY: float = float(os.getenv("PULSE_DELAY", "1"))
    MOEX_NEWS_LIMIT: int = int(os.getenv("MOEX_NEWS_LIMIT", "100"))
    MOEX_NEWS_PAGES: int = int(os.getenv("MOEX_NEWS_PAGES", "3"))
    RBC_MAX_ENTRIES: int = int(os.getenv("RBC_MAX_ENTRIES", "30"))

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