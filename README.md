# T-News Collector

Агент для сбора новостей российского фондового рынка, рыночных данных
(T-Invest SDK), анализа через LLM и автоматического открытия SHORT-позиций.

- Новости: Т-Пульс, РБК, MOEX → `raw_news` → LLM → `news_articles` (сентимент, тикеры)
- Рыночные данные: свечи T-Invest → `candles` (инкрементально)
- Дашборд: графики (свечи + SMA + объём + RSI) + новости по тикеру

## Структура проекта

```
app/
  __init__.py
  main.py               # точка входа: python -m app.main [--once]
  frontend.py           # Streamlit-дашборд: графики + новости
  prompts/
    news_handler        # системный промпт для LLM-анализа новостей
  api/
    t_news.py           # FastAPI для инспекции новостей
  collectors/
    base.py             # сохранение + дедупликация в raw_news
    pulse.py            # Т-Пульс (REST API tbank.ru)
    rbc.py              # РБК (RSS + фолбэк на Google News)
    moex.py             # MOEX (официальный API iss.moex.com)
    tinvest.py          # T-Invest: рыночные данные (свечи)
  core/
    config.py           # настройки из .env
    database.py         # SQLAlchemy engine, сессии, миграции
  models/
    news.py             # RawNews, NewsArticle
    trade.py            # Trade
    decision.py         # Decision
    market.py           # Candle, Instrument (кэш метаданных)
  services/
    llm.py              # клиент LLM (OmniRoute / OpenRouter)
    news_analyzer.py    # анализ новостей через LLM → NewsArticle
    data_collector.py   # оркестратор: сбор + анализ
    market_data.py      # сбор свечей, инкремент, резолв инструментов
    charting.py         # построение графиков (plotly), чистая логика
    scheduler.py        # цикл сбора в реальном времени
data/
  trading.db            # SQLite (создаётся автоматически)
init_db.py              # создание таблиц + миграции
```

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Скопируйте `.env.example` в `.env` и заполните ключи:

```bash
copy .env.example .env
```

## Конфигурация (.env)

### T-Invest (рыночные данные)

```env
TINKOFF_TOKEN=ваш_токен              # Токен T-Invest API
TINKOFF_SANDBOX=True                  # Песочница
SSL_TBANK_VERIFY=true                 # gRPC корневой сертификат РФ (обязательно)
MARKET_TIMEFRAMES=1d,1h,15m          # Таймфреймы свечей
TRACKED_TICKERS=SBER,GAZP,VTBR,LKOH,YNDX  # Отслеживаемые тикеры
```

### LLM (анализ новостей)

```env
# Вариант 1: OmniRoute (локальный шлюз, без ключа)
LLM_BASE_URL=http://localhost:20128
LLM_API_KEY=
LLM_MODEL=auto/best-reasoning

# Вариант 2: OpenRouter (облачный)
# LLM_BASE_URL=https://openrouter.ai/api/v1
# LLM_API_KEY=sk-or-v1-ваш_ключ
# LLM_MODEL=google/gemini-2.0-flash-exp:free
```

## Запуск

```bash
# 1. Инициализация БД + миграции
python init_db.py

# 2. Полный цикл (сбор новостей + свечей + анализ LLM)
python -m app.main

# 3. Дашборд (отдельный терминал)
streamlit run app/frontend.py
```

