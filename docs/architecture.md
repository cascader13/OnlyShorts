# Архитектура: агент SHORT-позиций на T-Invest

> Ветка `develop`. Документ описывает текущую реализацию (сбор данных, анализ,
> решения, исполнение) и планируемый ML-слой (CatBoost) поверх датасета
> `training_snapshots`.

## 1. Цель

Автоматически открывать и закрывать **SHORT-позиции** на T-Invest по сигналам
из рыночных данных и новостного сентимента. Путь: сначала **paper-режим** и
**песочница** (отлаживаем логику и модель), потом **live**.

Три контура, которые уже работают:

| Контур | Что делает | Файл |
|---|---|---|
| Сбор данных | Новости (Т-Пульс/РБК/MOEX), свечи T-Invest, кэш инструментов | `collectors/*`, `market_data.py` |
| Анализ | LLM/эвристика → сентимент и тикеры | `news_analyzer*.py` |
| Торговля | Решение → проверка рисков → шорт → мониторинг → закрытие | `trading_agent.py`, `risk_manager.py`, `position_manager.py`, `execution.py` |

Поверх них строится **четвёртый контур** — ML-модель (CatBoost) на снимках
`training_snapshots`, которая даст сигнал `P(down)` для `Decision`.

## 2. Общая схема

```
                 ВНЕШНИЕ ИСТОЧНИКИ
 ┌──────────────┬───────────────┬──────────────┬─────────────────────┐
 │ Т-Пульс REST │ РБК RSS       │ MOEX API     │ T-Invest gRPC (свечи)│
 └──────┬───────┴───────┬───────┴──────┬───────┴──────────┬──────────┘
        ▼              ▼              ▼                   ▼
   ┌─────────┐   ┌─────────┐   ┌─────────┐         ┌────────────┐
   │ pulse.py│   │ rbc.py  │   │ moex.py │        │  tinvest.py│
   └────┬────┘   └────┬────┘   └────┬────┘         └─────┬──────┘
        └──────────────┴─────┬──────┘                    │
                             ▼                           ▼
                        raw_news ────────────┐     candles ──────────┐
                             │               │                       │
        Scheduler            │               ▼                       ▼
   (каждые ~5 мин) ──────────┼───────────────►  СНАПШОТЫ (обучение)  │
                             │                     TrainingSnapshot  │
                             ▼                                     ▼
                    news_analyzer                          training_data
                    (LLM | heuristic)                  (backfill/capture/label)
                             │                               │
                             ▼                               ▼
                     news_articles                    DATASET → CatBoost
                     (sentiment, tickers)              P(шорт в плюс)
                                                         │ (план)
                                                         ▼
                              ┌───────────────────────────────────────┐
                              │        Decision (action, confidence)  │
                              └──────────────────┬────────────────────┘
                                                 ▼
                    ┌────────────────────────────────────────────────────┐
                    │ TradingAgent.run_cycle()                           │
                    │  RiskManager    → «можно? сколько лотов?»          │
                    │  ExecutionService → open_short (SELL)              │
                    │  PositionManager → стоп/тейк/trailing/время → close│
                    └────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
                                     trades → БД + дашборд
```

## 3. Слои и компоненты

### 3.1 Сбор данных

- **`collectors/pulse.py`** — Т-Пульс (REST tbank.ru), основной источник. Текст
  с тегами `{$TICKER}` — первичный сигнал для эвристики.
- **`collectors/rbc.py`** — РБК (RSS + фолбэк на Google News). *На 2026-07 даёт
  0 записей (Qrator 404) — фактически отключён.*
- **`collectors/moex.py`** — официальный API `iss.moex.com`.
- **`collectors/tinvest.py`** + **`market_data.py`** — свечи T-Invest
  (`1d/1h/15m`), инкрементальная загрузка, кэш инструментов
  (`figi`, `lot`, `instrument_type`, `class_code`).
- Все коллекторы пишут в `raw_news` / `candles` с дедупликацией. Сбой одного
  источника не роняет проход.

### 3.2 Анализ новостей

- **`news_analyzer.py`** (LLM) — системный промпт `prompts/news_handler`,
  сентимент и тикеры → `news_articles`.
- **`news_analyzer_heuristic.py`** (эвристика, без LLM) — лексиконный анализатор
  с лемматизацией (pymorphy3). Тег `{$TICKER}` = primary-сигнал, контекстные
  правила сентимента. Переключается `NEWS_ANALYZER=llm|heuristic`.
- Результат: `NewsArticle.sentiment_score` ∈ [-1, 1] и `sentiment_confidence`.

### 3.3 Рыночные данные

- Свечи `Candle(ticker, timeframe, ts, open/high/low/close, volume)`.
- Из свечей считаются RSI14, SMA20/50, ATR-волатильность, объём
  (агенты `services/agents/rsi.py|sma.py|volatility.py`, `charting.candles_to_df`).
- **`market_data.py`** выбирает адрес API по `TINKOFF_SANDBOX`
  (песочница `sandbox-invest-public-api.tbank.ru` / live `invest-public-api.tbank.ru`).

### 3.4 Датасет для обучения

Единый конвейер **«снимок → отложенная разметка»** (`training_data.py` +
модель `TrainingSnapshot`):

- **Строка** = момент времени `T` + признаки, известные на `T`:
  - техника по свечам 15m as-of `T`: `rsi`, `price`, `sma_20`, `sma_50`,
    `price_vs_sma_20/50`, `volatility` (ATR/price), `volume`;
  - новости за окно `[T − window_hours, T)` по **времени публикации**
    (`published_at`, а не `created_at`): `news_sentiment_avg`,
    `news_count`, `news_confidence_avg`, `sentiment_change_2h`.
- **Атрибуция новостей — по `published_at`.** Использование `created_at`
  (момент обработки LLM) даёт два дефекта: потеря сигнала (новость, вышедшая
  в окне слепка, но обработанная позже, в него не попадает) и загрязнение
  (такая новость «оседает» в более позднем слепке — ложная связка
  «в T негатив → цена падала T..T+N» и look-ahead в данных).
- **`refresh_news_features()`** — «догоняет» слепки: когда LLM обработал
  новые новости, пересчитывает новостные колонки уже записанных снапшотов,
  чьё окно публикаций накрыло `published_at` этих новостей. Пересчёт
  идемпотентен (по всем статьям окна). Вызывается в `collect_all` после
  анализа новостей; `--refresh` — полный пересчёт для миграции старых строк.
- **Таргет — path-симуляция стратегии** (`target = 1`, если симуляция шорта
  закрылась в плюс). Путь свечей `(T, T + MAX_HOLD_HOURS]` прогоняется через
  РОВНО те же правила выхода, что исполняет `PositionManager` (стоп-лосс
  `STOP_LOSS_PERCENT`, тейк-профит `TAKE_PROFIT_PERCENT`, trailing stop,
  лимит времени `MAX_HOLD_HOURS`) — метка и исполнение живут в одной
  стратегии, модель обучается на том, что агент реально сделает. Допущения
  внутри-свечевой разметки — в docstring `simulate_short_exit`. Рядом с
  меткой сохраняются: реализованный P&L (`target_return_pct`), причина выхода
  (`exit_reason`), длительность (`sim_duration_hours`), MAE/MFE (`mae_pct`/
  `mfe_pct`), горизонт (`max_hold_hours`) и параметры правил выхода
  (`label_stop_loss_pct`/`label_take_profit_pct`/`label_trail_*`). Метка
  самодокументирующая: `label_version` кодирует «рецепт»
  (`path_sim_sl5_tp3_ta2_td1.5_h4`) — после смены процентов и `--relabel`
  по БД видно, по каким параметрам размечена каждая строка; NULL у старых
  строк со старой меткой «close(T+N) < price».
- `backfill()` — исторические снимки за N дней (шаг 60 мин);
  `capture_live()` — 1 снимок/час/тикер в текущем часе (target=NULL);
  `label_expired()` — заполняет target у строк, у которых прошёл горизонт
  симуляции; `relabel()` (`--relabel`) — переразмечает все строки по текущим
  правилам (миграция старого датасета).
- Уникальный ключ `(ticker, timestamp, window_hours, lookahead_hours)` —
  идемпотентность, дублей нет.
- Запуск:
  ```bash
  python -m app.services.training_data --backfill --days 30
  python -m app.services.training_data --label
  python -m app.services.training_data --refresh   # пересчёт новостных колонок
  python -m app.services.training_data --stats
  ```

### 3.5 Сигналы → Decision

- **Сейчас:** автоматическая генерация `Decision` отключена. Таблица
  `decisions` заполняется вручную либо будущим генератором сигналов;
  торговый агент читает готовые решения.
- **План:** CatBoost даёт `P(down)`. Если `P(down) ≥ порог` → строка
  `Decision(action="SHORT", confidence=P(down))`. Это тот же контракт, что ест
  `TradingAgent`, поэтому ML-модель подключается **без изменения торгового слоя**.
- `Decision.is_short_signal` = `action == "SHORT" and confidence > 0.6`
  (порог продублирован в `CONFIDENCE_THRESHOLD`).

### 3.6 Торговый слой

- **`TradingAgent.run_cycle()`** — дирижёр одного прохода:
  1. читает SHORT-решения за последние 10 минут;
  2. каждое прогоняет через `process_decision`;
  3. зовёт `position_manager.monitor_and_close()`.
- **`RiskManager.can_open_short()`** — «можно ли» и «сколько лотов»:
  - уверенность ≥ `CONFIDENCE_THRESHOLD`;
  - тикер шортуем (TQBR-share);
  - дневные лимиты: ≤ 10 сделок/день, реализованный убыток ≤
    `DAILY_LOSS_LIMIT_PERCENT` капитала, нет «зависших» позиций;
  - ≤ `MAX_OPEN_POSITIONS` открытых;
  - размер позиции = **минимум** из:
    * нельного лимита `MAX_POSITION_SIZE_PERCENT` капитала;
    * масштаба по волатильности: если стоп `2·ATR` шире фиксированного
      `STOP_LOSS_PERCENT` — позиция пропорционально уменьшается.
  - equity — из портфеля песочницы (фолбэк на деньги по валютам).
- **`ExecutionService`** (`execution.py`) — gRPC-мост к T-Invest:
  `resolve_figi` (TQBR + фолбэк `find_instrument`), `get_lot_size`,
  `post_market_order` (рыночный, uuid4 order_id = ключ идемпотентности),
  `open_short` (SELL) / `close_short` (BUY/cover), `cancel_order`,
  `get_order_state`. Ретраи с экспоненциальным backoff, но постоянные ошибки
  (`INVALID_ARGUMENT` и т.п.) не повторяются. В paper-режиме исполнение
  имитируется (status=`FILL`, `broker_order_id` = `PAPER-...`).
- **`PositionManager`** — мониторинг и закрытие:
  - `monitor_and_close()`: цена (live → фолбэк свеча из БД), экстремумы
    `highest/lowest`, триггеры выхода → закрытие через брокера;
  - триггеры: **стоп-лосс** (цена ≥ стопа, шорт убыточен при росте),
    **тейк-профит** (цена ≤ цели), **лимит времени** (`MAX_HOLD_HOURS`),
    **trailing stop** (активация после хода вниз ≥
    `TRAILING_STOP_ACTIVATION_PERCENT`, отскок ≥ `TRAILING_STOP_DISTANCE_PERCENT`);
  - `sync_open_trades()` — сверяет нашу таблицу `trades` с реальными позициями
    брокера (внешние закрытия, маржин-коллы);
  - `force_close_all()` — аварийное закрытие.
- **Ключевая конвенция:** в БД `quantity` хранится в **акциях**, брокеру уходит
  **число лотов** (акции → лоты через размер лота).

### 3.7 Оркестрация

- **`scheduler.py`** — цикл реального времени (каждые
  `COLLECT_INTERVAL_SECONDS` ≈ 5 мин): `DataCollector.collect_all()` →
  сбор новостей + анализ + снапшоты + (если включено) торговый цикл.
- Фоновый поток через `start_background_collector()` в lifespan FastAPI.
- **`data_collector.py`** — `run_trading_cycle()` создаёт `TradingAgent` со
  счётом из `ACCOUNT_ID` или первого аккаунта песочницы.

### 3.8 Дашборд / API

- **`frontend.py`** (Streamlit): вкладки «Счёт и портфель» (пополнение
  песочницы, портфель с шортами, прошедшие шорты), «Рынок» (свечи + новости).
- **`api/t_news.py`** (FastAPI): инспекция новостей, `/news/recent`.
- Время в БД хранится как **naive МСК (UTC+3)** — единая конвенция проекта
  (`app/core/timeutil.py`: `msk_now()`, `to_naive_msk()`). Отображение
  форматирует значения без конвертации.

## 4. Жизненный цикл сделки

```
Свежие Decision(SHORT)
   → can_open_short? ──НЕТ──> пропуск (причина в логах)
   │ДА
   → расчёт лотов (equity, ATR, лимиты)
   → стоп/тейк из настроек
   → ExecutionService.open_short(ticker, lots)   # SELL по рынку
   → status FILL? ──НЕТ──> пропуск
   │ДА
   → Trade(status=OPEN, broker_order_id) + decision.trade_id (идемпотентность)

Пока OPEN (каждый проход):
   → цена + update_price_monitoring (экстремумы)
   → стоп / тейк / MAX_HOLD_HOURS / trailing ── сработал?
        └─ДА──> close_short(BUY) → FILL? ──> Trade.close() → pnl, pnl_percent
   → sync_open_trades: позиции нет у брокера → закрываем в БД
```

## 5. ML-слой: CatBoost (план)

### 5.1 Контракт

| Часть | Значение |
|---|---|
| Вход | 12 признаков снимка (таблица `TrainingSnapshot`) |
| Таргет | `target ∈ {0,1}` — path-симуляция шорта (стоп/тейк/trailing/`MAX_HOLD_HOURS`) закрылась в плюс |
| Выход | `P(down)` → `Decision(action="SHORT", confidence=P(down))` |
| Точка интеграции | новый сервис `ml_model.py` + запись в `Decision` (торговый слой не трогаем) |

### 5.2 Сколько нужно данных

Признаков 12, таргет бинарный. Скорость накопления снимков:

- **1 снимок / час / тикер** в торговые часы (≈ 12 часов/день, когда свечи
  15m свежие);
- `TRACKED_TICKERS` по умолчанию ≈ 44, реально собирается ~30–40;
- **≈ 400–500 строк/день**, **≈ 8–10 тыс. строк/месяц**.

Рекомендации по объёму для CatBoost:

- **Минимум, чтобы модель вообще «поехала»:** ~2–3 тыс. размеченных строк
  (CatBoost хорошо регуляризуется на малых выборках, но 12 признаков на 1000
  строк — уже на грани переобучения).
- **Комфортно:** 8–15 тыс. строк. Стабильные split'ы, спокойная валидация.
- **С учётом ежемесячного переобучения:** одного месяца live-сбора (~10k)
  достаточно. Больше всего добавит **разнообразие тикеров** (кросс-секция) и
  **разные рыночные режимы** (несколько недель), а не просто число строк.
- Классовый баланс близок к 50/50 (цена за торговый день падает примерно так
  же часто, как растёт) — веса, скорее всего, не понадобятся; проверить
  через `--stats` (`target_1` vs `target_0`).

Первый прогон «завтра» ограничен тем, сколько истории свечей уже лежит в БД:
`backfill --days N` строит снимки из накопленных свечей. Если есть ~месяц
15m-свечей по крупным тикерам — это сразу несколько тысяч размеченных строк,
и модель можно тренировать хоть сейчас.

### 5.3 Как оценивать (важно)

Финансовый ряд — **только time-based split**, не случайный:
обучаемся на первых 80% по времени, валидируемся на последних 20%.
Случайное перемешивание даст красивую метрику и мертвую модель на живых данных
(соседние часы почти одинаковые → «подглядывание»).

- Метрика: **AUC-ROC** + precision@порог. Порог подбирается по валидации и
  попадает в `CONFIDENCE_THRESHOLD`.
- Baseline обязателен: сравнить с «всегда шорт в час X» / случайным — иначе
  не понять, даёт ли модель хоть что-то сверх тренда.

### 5.4 Переобучение / дообучение

- При 10–50 тыс. строк × 12 признаков CatBoost обучается **секунды на CPU** —
  GPU не нужен.
- Поэтому **ежемесячное полное переобучение на всех данных — правильный выбор**
  (проще и надёжнее, чем fine-tuning). Дообучение через `init_model`
  (продолжение обучения готовой модели) имеет смысл только если захочется
  обновлять модель внутри месяца — и то это скорее про экономию, чем про качество.
- Ритм: полное переобучение раз в месяц (после накопления нового месяца
  снапшотов) + подбор порога по свежей валидации. Если рынок сменил режим —
  можно переобучать чаще (еженедельно, это дёшево).

### 5.5 Фичи, которые стоит добавить при первой же итерации

- `ticker` как **категориальную** фичу — коронное преимущество CatBoost;
- горизонт фиксирован стратегией (`MAX_HOLD_HOURS` входит в правила выхода,
  поэтому менять его = менять и метку, и исполнение; эксперимент с
  горизонтом — это переразметка `--relabel` после смены `MAX_HOLD_HOURS`);
- лаги цены/RSI (доходность за 1/2/4 часа) — CatBoost сам разрулит нелинейности.

## 6. Конфигурация (.env) — торговый контур

```env
# Режим: True = paper (ордера не идут брокеру), False = реальные (сначала песочница)
PAPER_TRADING=True
# Мастер-переключатель торгового агента
TRADING_ENABLED=True
# Счёт (пусто → первый аккаунт песочницы)
ACCOUNT_ID=

# Риск
MAX_OPEN_POSITIONS=3
DAILY_LOSS_LIMIT_PERCENT=5.0
MAX_POSITION_SIZE_PERCENT=5
STOP_LOSS_PERCENT=3
TAKE_PROFIT_PERCENT=5
MAX_HOLD_HOURS=4
TRAILING_STOP_ACTIVATION_PERCENT=2.0
TRAILING_STOP_DISTANCE_PERCENT=1.5
CONFIDENCE_THRESHOLD=0.6

# Датасет
SNAPSHOT_ENABLED=True
SNAPSHOT_TIMEFRAME=15m
NEWS_WINDOW_HOURS=2
TARGET_HOURS=2
SNAPSHOT_STEP_MINUTES=60
```

## 7. Запуск

```bash
# 1. БД + миграции
python init_db.py

# 2. Сейчас — накопление датасета (сбор + снапшоты). В отдельном терминале:
python -m app.main                      # цикл сбора (в фоне — в lifespan)
# или один проход:
python -m app.main --once

# 3. Бэкфилл снапшотов из уже собранных свечей + разметка
python -m app.services.training_data --backfill --days 30
python -m app.services.training_data --label
python -m app.services.training_data --stats

# 4. Дашборд (отдельный терминал)
streamlit run app/frontend.py
```

Порядок проверки в песочнице:
1. `PAPER_TRADING=True` — логика решения→сделки, без ордеров брокеру;
2. `PAPER_TRADING=False` + `TINKOFF_SANDBOX=True` — реальные шорты на
   песочном счёте T-Invest (пополняется на дашборде / `sandbox_pay_in`);
3. только потом live (`TINKOFF_SANDBOX=False`, обязательный `ACCOUNT_ID`).

## 8. Дорожная карта

- [x] Сбор новостей + свечей, LLM/эвристический анализ
- [x] Торговый слой: RiskManager, ExecutionService (paper + sandbox), PositionManager
- [ ] **CatBoost: `ml_model.py` + обучение + запись Decision** ← следующий шаг
- [ ] Сравнение ML-сигнала с LLM/эвристикой (и baseline) на истории
- [ ] Live-режим с жёсткими лимитами и алертами
