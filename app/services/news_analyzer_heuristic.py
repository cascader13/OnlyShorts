"""
Эвристический анализатор новостей (без LLM).

Заменяет LLM-анализ классическими методами, сохраняя тот же контракт
(NewsArticle), поэтому агрегатор и торговый агент работают без изменений:

- тикеры:  теги {$TICKER} Т-Пульса (их вставляет сама платформа — самый
           надёжный сигнал, покрывает и тикеры вне whitelist) + whitelist
           известных MOEX-тикеров + словарь названий компаний
           + тикер из source_url коллектора (Пульс кладёт его в URL)
- сентимент: лексикон русского финансового домена + лемматизация pymorphy3
           (отрицания, усилители, близость к тикеру) + контекстные правила —
           многословные фразы, которые лексикон не ловит («слабее ожиданий»,
           «превзошёл прогноз», «неприятный сюрприз» и т.п.)
- confidence: покрытие лексиконом (суммарный вес подтверждений)
- сектор:  словарь ключевых слов по 10 секторам, фолбэк тикер -> сектор
- дедупликация: cross-source near-duplicate по Jaccard биграмм слов

Запуск:
    python -m app.services.news_analyzer_heuristic --once
"""

import functools
import logging
import re
from datetime import timedelta
from typing import Optional

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db_context
from app.core.timeutil import msk_now
from app.models.news import RawNews, NewsArticle

logger = logging.getLogger(__name__)

# Лемматизатор pymorphy3 — опционально: если библиотека не установлена,
# работаем в режиме точного совпадения словоформ (качество ниже).
try:
    from pymorphy3 import MorphAnalyzer

    _MORPH = MorphAnalyzer()
except Exception:  # pragma: no cover — зависит от окружения
    _MORPH = None
    logger.warning("pymorphy3 не установлен — лемматизация отключена")

BATCH_SIZE = 10

# Порог Jaccard для cross-source near-duplicate. Завышен осознанно: биграммы
# слов у разных отчётов одной компании (например, помесячных) могут совпадать
# на ~0.6, и схлопывать их в один артикль не нужно.
SIMILARITY_THRESHOLD = 0.7

# Окно (в токенах) для близости сентимент-слова к тикеру и его буст веса
PROXIMITY_WINDOW = 6
PROXIMITY_BOOST = 1.3

# Период (дней), за который ищем near-duplicate
DEDUP_WINDOW_DAYS = 7
DEDUP_MAX_ROWS = 200

# --- Тикеры ---------------------------------------------------------------

# Известные тикеры MOEX (whitelist против ложных срабатываний латинских слов)
BASE_TICKERS = {
    "SBER", "GAZP", "VTBR", "LKOH", "YNDX", "TATN", "NVTK", "ROSN", "MGNT",
    "CHMF", "GMKN", "PLZL", "NLMK", "AFLT", "ALRS", "PHOR", "POLY", "SNGS",
    "SNGR", "RUAL", "MTSS", "RTKM", "MAGN", "AKRN", "RASP", "MOEX", "RENI",
    "SBERP", "TCSG", "IRAO", "HYDR", "FEES", "MTLR", "MVID", "FIVE", "LENT",
    "MGTX", "VKCO", "MDSB", "GEMC", "GLTR", "SELG",
}

KNOWN_TICKERS: set[str] = BASE_TICKERS | {t.upper() for t in settings.TRACKED_TICKERS}

# Названия компаний -> тикер. Короткие алиасы (<5 символов) сопоставляются
# только как отдельное слово, чтобы "сбер" не срабатывал в "сбережениях".
TICKER_ALIASES: dict[str, list[str]] = {
    "SBER": ["сбербанк", "сбербанка", "сбербанку", "сбербанком", "сбербанке", "сберовский", "sber", "сбер"],
    "GAZP": ["газпром", "газпрома", "газпромом", "газпромбанк", "gazprom"],
    "VTBR": ["втб", "банк втб", "vtb"],
    "LKOH": ["лукойл", "лукойла", "лукойлом", "lukoil"],
    "YNDX": ["яндекс", "яндекса", "яндексом", "yandex"],
    "ROSN": ["роснефть", "роснефти", "роснефтью", "rosneft"],
    "NVTK": ["новатэк", "новатэка", "novatek"],
    "TATN": ["татнефть", "татнефти"],
    "MGNT": ["магнит", "магнита", "магнитом"],
    "CHMF": ["северсталь", "северстали"],
    "GMKN": ["норникель", "норильский никель", "норникеля"],
    "PLZL": ["полюс", "полюса"],
    "NLMK": ["нлмк"],
    "AFLT": ["аэрофлот", "аэрофлота"],
    "ALRS": ["алроса", "алросы"],
    "MTSS": ["мтс"],
    "RTKM": ["ростелеком"],
    "MOEX": ["мосбиржа", "московская биржа", "московской биржи"],
    "TCSG": ["т-банк", "тинькофф", "тинкофф", "т-технологии"],
    "IRAO": ["интер рао"],
    "HYDR": ["русгидро"],
    "MVID": ["м.видео", "мвидео", "эльдорадо"],
    "FIVE": ["x5", "х5", "пятерочка", "пятёрочка", "перекресток", "перекрёсток"],
    "VKCO": ["вконтакте", "vkontakte", "vk.com"],
}

# --- Лексикон сентимента --------------------------------------------------
# Ключи — нормальные формы слов (леммы). Значение — сила тональности.

NEGATIVE_LEXICON: dict[str, float] = {
    "убыток": 0.7, "убыточный": 0.7, "убыточность": 0.6,
    "падение": 0.6, "падать": 0.6, "упасть": 0.6,
    "снижение": 0.5, "снизиться": 0.5,
    "сокращение": 0.5, "сократить": 0.5, "сократиться": 0.5,
    "штраф": 0.8,
    "санкция": 0.9, "санкционный": 0.9,
    "банкротство": 0.9, "обанкротиться": 0.9, "разорение": 0.9,
    "дефолт": 0.9,
    "иск": 0.6, "суд": 0.5, "расследование": 0.7, "проверка": 0.4,
    "лишение": 0.6, "изъятие": 0.7,
    "потеря": 0.6, "потерять": 0.6,
    "увольнение": 0.5, "уволить": 0.5, "увольнять": 0.5,
    "закрытие": 0.5, "приостановка": 0.5, "приостановить": 0.5,
    "распродажа": 0.6,
    "кризис": 0.7, "рецессия": 0.8, "стагнация": 0.7, "застой": 0.6,
    "инфляция": 0.5, "девальвация": 0.7, "безработица": 0.6,
    "обвал": 0.8, "обрушиться": 0.8, "рухнуть": 0.8, "провал": 0.8,
    "провалить": 0.7, "спад": 0.6,
    "негативный": 0.6,
    "проблема": 0.5, "сложность": 0.5, "трудность": 0.4,
    "долг": 0.4, "задолженность": 0.5, "просрочка": 0.6, "дефицит": 0.5,
    "нехватка": 0.5, "отток": 0.6, "бегство": 0.6,
    "отставка": 0.5, "уход": 0.3,
    "ущерб": 0.7, "арест": 0.8, "конфискация": 0.8, "экспроприация": 0.8,
    "заморозка": 0.6, "заморозить": 0.6,
    "национализация": 0.7,
    "предупреждение": 0.4, "претензия": 0.5, "жалоба": 0.5,
    "остановка": 0.5, "остановить": 0.5, "остановиться": 0.5,
    "запрет": 0.7, "запретить": 0.7, "блокировка": 0.6, "ограничение": 0.4,
    "риск": 0.4, "угроза": 0.7, "угрожать": 0.6, "опасность": 0.6,
    "слабый": 0.4, "ослабление": 0.5, "ослабеть": 0.5,
    "протест": 0.5,
    # События и оценки, которых не хватало в первом срезе лексикона
    "столкновение": 0.5,
    "разочарование": 0.6, "разочаровать": 0.6,
    "просесть": 0.5, "просадка": 0.5,
    "ухудшение": 0.5, "ухудшить": 0.5, "ухудшиться": 0.5,
    "неопределенность": 0.45, "нестабильность": 0.45,
    "эскалация": 0.6, "забастовка": 0.6, "авария": 0.6,
    "критика": 0.5, "насторожить": 0.5, "встревожить": 0.6,
    "медвежий": 0.4, "плохой": 0.4,
}

POSITIVE_LEXICON: dict[str, float] = {
    "рост": 0.6, "расти": 0.6, "вырасти": 0.6,
    "увеличение": 0.5, "увеличиться": 0.5, "наращивание": 0.5, "нарастить": 0.5,
    "удвоение": 0.6, "удвоить": 0.6,
    "прибыль": 0.7, "прибыльность": 0.6, "доход": 0.5, "доходность": 0.5,
    "дивиденд": 0.7, "выручка": 0.4,
    "рекорд": 0.7, "рекордный": 0.7, "максимум": 0.5, "превысить": 0.5,
    "превышение": 0.5,
    "повышение": 0.5, "повысить": 0.5, "повыситься": 0.5, "подъем": 0.5,
    "запуск": 0.5, "запустить": 0.5,
    "расширение": 0.5, "расширить": 0.5,
    "одобрение": 0.6, "одобрить": 0.6, "согласование": 0.4, "разрешение": 0.3,
    "контракт": 0.4,
    "окупаемость": 0.5, "эффективность": 0.4,
    "укрепление": 0.5, "укрепить": 0.5, "укрепляться": 0.5,
    "улучшение": 0.5, "улучшить": 0.5, "восстановление": 0.5, "восстановить": 0.5,
    "отскок": 0.4,
    "выкуп": 0.5, "покупка": 0.3,
    "стабильность": 0.3, "стабильный": 0.3,
    "позитивный": 0.5, "успех": 0.6, "успешный": 0.6,
    "награда": 0.5, "премия": 0.4, "признание": 0.4, "бонус": 0.4,
    "инвестиция": 0.4, "инвестировать": 0.5, "капитализация": 0.4,
    "рентабельность": 0.5, "поддержка": 0.3, "субсидия": 0.4, "грант": 0.4,
    "модернизация": 0.4,
    "ралли": 0.5, "бычий": 0.4,
}

# Усилители и отрицания (сопоставляются по сырой словоформе, без лемматизации)
INTENSIFIERS: dict[str, float] = {
    "резко": 1.5, "значительно": 1.4, "существенно": 1.3, "сильно": 1.2,
    "заметно": 1.2, "вдвое": 1.4, "втрое": 1.4, "многократно": 1.5,
}
NEGATIONS: set[str] = {"не", "без", "ни", "нет", "вместо"}

# --- Контекстные правила сентимента -----------------------------------------
# Многословные фразы, которые однословный лексикон не покрывает (проверки
# против ожиданий, события). Сопоставляются по сырому тексту в нижнем регистре.
# Вес: отрицательный -> негатив, положительный -> позитив. Фразы подобраны так,
# чтобы ключевые слова не пересекались с лексиконом — иначе вес задваивается.

_CONTEXT_PATTERNS: list[tuple[str, float]] = [
    # Результаты/отчёты против ожиданий
    (r"(?:слабее|ниже|хуже)\s+(?:ожиданий|прогнозов|консенсуса)", -0.65),
    (r"не\s+(?:оправдал|оправдала|оправдали|оправдало)\s+(?:ожидания|ожиданий|прогноз|прогнозы)", -0.7),
    (r"(?:сильнее|выше|лучше)\s+(?:ожиданий|прогнозов|консенсуса)", 0.65),
    (r"(?:превзошел|превзошла|превзошли|превзошло|превзойд\w+)\s+(?:ожидания|ожиданий|прогноз|прогнозы)", 0.7),
    # События и оценки
    (r"неприятный\s+сюрприз", -0.6),
    (r"приятный\s+сюрприз", 0.6),
    (r"пересмотр\s+прогноза\s+вниз", -0.55),
    (r"не\s+(?:смог|смогла|смогли)\s+(?:удержать|поддержать)", -0.5),
    (r"(?:давление|давления)\s+на\s+(?:акции|котировки|бумаги)", -0.4),
    (r"устойчивый\s+спрос", 0.4),
]

_CONTEXT_PATTERNS_COMPILED: list[tuple[re.Pattern, float]] = [
    (re.compile(pattern), weight) for pattern, weight in _CONTEXT_PATTERNS
]

# --- Секторы --------------------------------------------------------------

SECTOR_BY_TICKER: dict[str, str] = {
    "SBER": "Banking & Finance", "VTBR": "Banking & Finance", "TCSG": "Banking & Finance",
    "MOEX": "Banking & Finance", "RENI": "Banking & Finance",
    "GAZP": "Oil & Gas", "LKOH": "Oil & Gas", "ROSN": "Oil & Gas", "NVTK": "Oil & Gas",
    "SNGS": "Oil & Gas", "SNGR": "Oil & Gas", "TATN": "Oil & Gas",
    "YNDX": "IT", "MGTX": "IT", "VKCO": "IT",
    "MGNT": "Retail & Consumer Goods", "MVID": "Retail & Consumer Goods",
    "FIVE": "Retail & Consumer Goods", "LENT": "Retail & Consumer Goods",
    "CHMF": "Metals & Mining", "NLMK": "Metals & Mining", "GMKN": "Metals & Mining",
    "PLZL": "Metals & Mining", "ALRS": "Metals & Mining", "RUAL": "Metals & Mining",
    "POLY": "Metals & Mining", "MTLR": "Metals & Mining", "MAGN": "Metals & Mining",
    "RASP": "Metals & Mining", "SELG": "Metals & Mining", "GLTR": "Metals & Mining",
    "AFLT": "Transportation & Logistics",
    "RTKM": "Telecommunications", "MTSS": "Telecommunications",
    "IRAO": "Utilities", "HYDR": "Utilities", "FEES": "Utilities",
    "PHOR": "Chemicals", "AKRN": "Chemicals",
    "MDSB": "Healthcare", "GEMC": "Healthcare",
}

# Ключевые слова секторов (подстрока в тексте в нижнем регистре).
# Латинские ключевые слова сопоставляются как отдельные слова.
SECTOR_KEYWORDS: dict[str, list[str]] = {
    "IT": [" it ", " программ", "цифров", "платформ", "интернет-", "облачн",
           "нейросет", "кибербезопасн", "технолог", "приложени", "стартап",
           "искусственн интеллект", " данн", " сервер", "чип", "микропроцессор"],
    "Metals & Mining": ["металл", "сталь", "никел", "медь", "медн", "алюмини",
                        "золот", "серебр", "добыч", "руды", "руду", "уголь",
                        "угля", "углем", "недра", "горнодоб", "обогащен",
                        "платин", "паллади"],
    "Oil & Gas": ["нефт", " газ", "газпром", "баррель", "углеводород",
                  "месторожден", "спг", "трубопровод", "газопровод", "топлив"],
    "Banking & Finance": ["банков", "банка ", " банк ", "банки", "банком",
                          "банке", "кредит", "ставк", "финанс", "ипотек",
                          "вклад", "депозит", "платеж", "платёж", "страхован",
                          "лизинг", "процент", "центробанк", "облигац",
                          "акци", "валют", " рубл", "инвестицион", "капитал",
                          "биржа"],
    "Retail & Consumer Goods": ["ритейл", "магазин", "потребительск", "товар",
                                "продаж", "маркетплейс", "покупател",
                                "e-commerce", "ecommerce", "продукт",
                                "сеть магазин"],
    "Telecommunications": ["телеком", "мобильн", "оператор связи", " связ",
                           "интернет-провайдер", " 5g", "абонент",
                           "телекоммуникац"],
    "Transportation & Logistics": ["перевозк", "логистик", "авиакомпани",
                                   "авиа", "железнодорожн", "судоходств",
                                   "грузоперевозк", " порт", "транспорт", "рейс"],
    "Utilities": ["электроэнерг", "энергетик", "электричеств", "тепло",
                  "генерац", "сбыт", "коммунальн", "мощность"],
    "Healthcare": ["медицин", "фармацевт", "лекарств", "клиник", "биотех",
                   "вакцин", "больниц", "здоровь"],
    "Chemicals": ["химическ", " химия", "удобрен", "полимер", "пластик",
                  "каучук", "нефтехим", "щелоч"],
}


def _build_pattern(keyword: str) -> re.Pattern:
    """Латинские слова сопоставляет как целые слова, кириллические — подстрокой."""
    if keyword.isascii() and keyword.isalpha():
        return re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE)
    return re.compile(re.escape(keyword), re.IGNORECASE)


SECTOR_PATTERNS: dict[str, list[re.Pattern]] = {
    sector: [_build_pattern(kw) for kw in kws]
    for sector, kws in SECTOR_KEYWORDS.items()
}


# --- Текст и лемматизация --------------------------------------------------

_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)

# Теги инструментов Т-Пульса вида {$SBER} — платформа вставляет их прямо в текст
# поста. Внутри тега ровно один токен (латиница/цифры), поэтому его смещение
# в тексте однозначно даёт индекс токена для близости сентимента.
_TAG_RE = re.compile(r"\{\$([A-Za-z0-9]+)\}")


@functools.lru_cache(maxsize=16384)
def _lemmatize(word: str) -> str:
    """Возвращает нормальную форму слова (лемму) через pymorphy3."""
    if _MORPH is None or not re.search(r"[а-яё]", word):
        return word
    try:
        # pymorphy3 возвращает «ё»; нормализуем до «е», чтобы ключи лексикона
        # («подъем») совпадали с текстом как с «подъём», так и с «подъем».
        return _MORPH.parse(word)[0].normal_form.replace("ё", "е")
    except Exception:
        return word


def _tokenize_with_offsets(text: str) -> list[tuple[int, str]]:
    """Токены с начальным смещением в тексте (для привязки тегов и фраз)."""
    return [(m.start(), m.group(0).lower()) for m in _TOKEN_RE.finditer(text.lower())]


def _tokenize(text: str) -> list[str]:
    """Разбивает текст на токены в нижнем регистре."""
    return [token for _, token in _tokenize_with_offsets(text)]


def _token_bigrams(text: str) -> set[tuple[str, str]]:
    """Множество биграмм слов для near-duplicate."""
    tokens = _tokenize(text)
    return set(zip(tokens, tokens[1:]))


def _jaccard(a: set, b: set) -> float:
    """Коэффициент Жаккара двух множеств."""
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _first_sentence(text: str, max_len: int = 300) -> str:
    """Выделяет первое предложение (lead) как краткое содержание."""
    if not text:
        return ""
    clean = " ".join(text.split())
    for m in re.finditer(r"[.!?…]+\s*", clean):
        sent = clean[: m.end()].strip()
        if len(sent) >= 15:  # пропускаем сокращения вида "г."
            return sent[:max_len]
    return clean[:max_len]


def _sentiment_label(score: Optional[float]) -> Optional[str]:
    """Числовой скор -> текстовая метка (те же пороги, что у LLM-пути)."""
    if score is None:
        return None
    if score < -0.3:
        return "negative"
    if score > 0.3:
        return "positive"
    return "neutral"


# --- Тикеры ---------------------------------------------------------------

def _alias_matches(alias: str, text_lower: str) -> bool:
    """Короткие алиасы — только целым словом, длинные — подстрокой."""
    if len(alias) >= 5:
        return alias in text_lower
    return (
        re.search(rf"(?<![а-яёa-z0-9]){re.escape(alias)}(?![а-яёa-z0-9])", text_lower)
        is not None
    )


def _ticker_from_url(source_url: Optional[str]) -> Optional[str]:
    """Тикер из source_url коллектора (Пульс кладёт его в последний сегмент URL)."""
    if not source_url:
        return None
    last = source_url.rstrip("/").split("/")[-1]
    last = last.split("?")[0].split("#")[0].upper()
    if re.fullmatch(r"[A-Z0-9]{1,6}", last) and last in KNOWN_TICKERS:
        return last
    return None


def extract_tickers(raw: RawNews) -> tuple[str, str, set[int], bool]:
    """
    Определяет тикеры новости.

    Приоритет primary-тикера: теги {$TICKER} Т-Пульса > source_url коллектора
    > алиас компании > латинский тикер в тексте (только из whitelist).

    Returns:
        (tickers_str, primary_ticker, ticker_positions, has_ticker)
        ticker_positions — индексы токенов с тикерами (для близости сентимента).
    """
    text = raw.full_text or ""
    text_lower = text.lower()
    token_pairs = _tokenize_with_offsets(text)
    tokens = [t for _, t in token_pairs]
    pos_by_offset = {off: i for i, (off, _) in enumerate(token_pairs)}

    found: list[str] = []
    ticker_positions: set[int] = set()

    # 1) Явные теги {$TICKER}: их вставляет сама платформа, поэтому они надёжнее
    #    любого текстового матчинга и покрывают тикеры вне whitelist (OZON и т.п.).
    for m in _TAG_RE.finditer(text):
        ticker = m.group(1).upper()
        if ticker not in found:
            found.append(ticker)
        idx = pos_by_offset.get(m.start() + 2)  # буквы тикера начинаются после "{$"
        if idx is not None:
            ticker_positions.add(idx)

    # 2) Тикер из source_url коллектора (Пульс кладёт его в последний сегмент URL)
    url_ticker = _ticker_from_url(raw.source_url)
    if url_ticker and url_ticker not in found:
        found.append(url_ticker)

    # 3) Названия компаний
    for ticker, aliases in TICKER_ALIASES.items():
        if ticker in found:
            continue
        if any(_alias_matches(alias, text_lower) for alias in aliases):
            found.append(ticker)

    # 4) Латинские тикеры в тексте (только из whitelist — против ложных
    #    срабатываний на латинских словах вроде "TATN" в случайных строках)
    for idx, token in enumerate(tokens):
        upper = token.upper()
        if upper in KNOWN_TICKERS:
            ticker_positions.add(idx)
            if upper not in found:
                found.append(upper)

    primary = found[0] if found else ""
    return ",".join(found), primary, ticker_positions, bool(found)


# --- Сентимент ------------------------------------------------------------

def _confidence(matched_weight: float, has_ticker: bool) -> float:
    """Уверенность по покрытию лексиконом: 1 слово ~0.4, насыщение ~0.85."""
    if matched_weight <= 0:
        return 0.0
    conf = min(0.85, 0.2 + 0.25 * matched_weight)
    if not has_ticker:
        conf *= 0.6  # без тикера доверие ниже
    return round(conf, 3)


def _match_context_phrases(
    text: str,
    tokens: list[str],
    pos_by_offset: dict[int, int],
    ticker_positions: set[int],
) -> tuple[float, float, float]:
    """
    Сопоставляет контекстные правила (многословные фразы).

    Отрицание в 2 токенах перед фразой разворачивает полярность («не слабее
    ожиданий» -> позитив). Близость к тикеру повышает вес, как у лексикона.

    Returns:
        (pos_total, neg_total, matched_weight).
    """
    if not text:
        return 0.0, 0.0, 0.0
    # Нормализуем ё->е, как _lemmatize для лексикона: «превзошёл» должно
    # совпадать с паттерном «превзошел». Длина строки не меняется, поэтому
    # смещения токенов из pos_by_offset остаются валидными.
    text_lower = text.lower().replace("ё", "е")

    pos_total = 0.0
    neg_total = 0.0
    matched_weight = 0.0

    for pattern, weight in _CONTEXT_PATTERNS_COMPILED:
        for m in pattern.finditer(text_lower):
            idx = pos_by_offset.get(m.start())
            if idx is None:
                continue  # фраза начинается не с токена (практически не бывает)
            polarity = 1 if weight > 0 else -1
            # отрицание перед фразой разворачивает полярность
            if any(wt in NEGATIONS for wt in tokens[max(0, idx - 2): idx]):
                polarity = -polarity
            proximity = (
                PROXIMITY_BOOST
                if any(abs(idx - tp) <= PROXIMITY_WINDOW for tp in ticker_positions)
                else 1.0
            )
            w = abs(weight) * proximity
            if polarity > 0:
                pos_total += w
            else:
                neg_total += w
            matched_weight += w

    return round(pos_total, 3), round(neg_total, 3), round(matched_weight, 3)


def analyze_sentiment(
    text: str, ticker_positions: set[int], has_ticker: bool
) -> tuple[float, float]:
    """
    Лексиконный анализ тональности + контекстные правила.

    Учитывает отрицания и усилители в окне 2 токена, близость сентимент-слова
    к тикеру (буст веса). Контекстные правила ловят многословные фразы, которые
    лексикон не покрывает («слабее ожиданий», «превзошёл прогноз» и т.п.).

    Returns:
        (score, confidence): score в [-1, 1], confidence в [0, 1].
    """
    text = text or ""
    token_pairs = _tokenize_with_offsets(text)
    tokens = [t for _, t in token_pairs]
    pos_by_offset = {off: i for i, (off, _) in enumerate(token_pairs)}

    pos_total = 0.0
    neg_total = 0.0
    matched_weight = 0.0

    for idx, token in enumerate(tokens):
        lemma = _lemmatize(token)
        polarity = 0
        weight = POSITIVE_LEXICON.get(lemma)
        if weight is not None:
            polarity = 1
        else:
            weight = NEGATIVE_LEXICON.get(lemma)
            if weight is not None:
                polarity = -1
        if polarity == 0:
            continue

        # отрицания и усилители в окне 2 токена до
        window = tokens[max(0, idx - 2): idx]
        intensity = 1.0
        for wt in window:
            if wt in INTENSIFIERS:
                intensity = max(intensity, INTENSIFIERS[wt])
            if wt in NEGATIONS:
                polarity = -polarity

        # близость к тикеру повышает вес
        proximity = (
            PROXIMITY_BOOST
            if any(abs(idx - tp) <= PROXIMITY_WINDOW for tp in ticker_positions)
            else 1.0
        )

        w = weight * intensity * proximity
        if polarity > 0:
            pos_total += w
        else:
            neg_total += w
        matched_weight += w

    # Контекстные фразы-правила (вес идёт в ту же шкалу, что и лексикон)
    c_pos, c_neg, c_weight = _match_context_phrases(
        text, tokens, pos_by_offset, ticker_positions
    )
    pos_total += c_pos
    neg_total += c_neg
    matched_weight += c_weight

    total = pos_total + neg_total
    if total <= 0:
        return 0.0, 0.0

    score = max(-1.0, min(1.0, (pos_total - neg_total) / total))
    return round(score, 3), _confidence(matched_weight, has_ticker)


# --- Сектор ---------------------------------------------------------------

def classify_sector(raw: RawNews, primary_ticker: str) -> str:
    """Определяет отрасль: тикер -> сектор, иначе по ключевым словам текста."""
    mapped = SECTOR_BY_TICKER.get(primary_ticker)
    if mapped:
        return mapped

    text = f"{raw.title or ''}\n{raw.full_text or ''}".lower()
    best, best_count = "", 0
    for sector, patterns in SECTOR_PATTERNS.items():
        count = sum(len(p.findall(text)) for p in patterns)
        if count > best_count:
            best, best_count = sector, count
    return best


# --- Сборка статьи --------------------------------------------------------

def _build_article(raw: RawNews) -> NewsArticle:
    """Превращает сырую новость в NewsArticle по эвристическим правилам."""
    tickers_str, primary, ticker_positions, has_ticker = extract_tickers(raw)
    score, confidence = analyze_sentiment(raw.full_text, ticker_positions, has_ticker)
    sector = classify_sector(raw, primary)

    return NewsArticle(
        title=raw.title,
        full_text=raw.full_text,
        summary=_first_sentence(raw.full_text),
        sentiment_score=score,
        sentiment_label=_sentiment_label(score),
        sentiment_confidence=confidence,
        tickers=tickers_str,
        primary_ticker=primary,
        tags=sector,
        is_ai_generated=False,
        source=raw.source,
        published_at=raw.published_at,
        raw_news_id=raw.id,
    )


# --- Дедупликация ---------------------------------------------------------

def _load_recent(db: Session) -> list[tuple[int, str]]:
    """Недавние недублированные новости для cross-source near-duplicate."""
    since = msk_now() - timedelta(days=DEDUP_WINDOW_DAYS)
    rows = (
        db.query(RawNews.id, RawNews.full_text)
        .filter(RawNews.is_duplicate == False, RawNews.created_at >= since)
        .order_by(RawNews.id.desc())
        .limit(DEDUP_MAX_ROWS)
        .all()
    )
    return [(rid, text or "") for rid, text in rows]


def _find_duplicate(
    raw_id: int, bigrams: set[tuple[str, str]], recent_bigrams: list[tuple[int, set]]
) -> Optional[int]:
    """Ищет near-duplicate среди уже известных новостей, возвращает его id."""
    for rid, rb in recent_bigrams:
        if rid == raw_id:
            continue
        if _jaccard(bigrams, rb) >= SIMILARITY_THRESHOLD:
            return rid
    return None


# --- Публичный API --------------------------------------------------------

def analyze_batch(db: Session, raw_news_list: list[RawNews]) -> int:
    """
    Анализирует пакет сырых новостей эвристическим анализатором.

    Near-дубликаты (одно событие из разных источников) помечаются
    is_duplicate=True и пропускаются. Возвращает число созданных статей.
    """
    if not raw_news_list:
        return 0

    recent_bigrams: list[tuple[int, set[tuple[str, str]]]] = []
    for rid, text in _load_recent(db):
        bigrams = _token_bigrams(text)
        if bigrams:
            recent_bigrams.append((rid, bigrams))

    created = 0
    skipped = 0
    for raw in raw_news_list:
        bigrams = _token_bigrams(raw.full_text)
        if bigrams:
            dup_id = _find_duplicate(raw.id, bigrams, recent_bigrams)
            if dup_id is not None:
                raw.is_duplicate = True
                skipped += 1
                logger.info(
                    "Near-duplicate для #%d (похоже на #%d): %s",
                    raw.id, dup_id, raw.title[:80],
                )
                continue
            recent_bigrams.append((raw.id, bigrams))

        db.add(_build_article(raw))
        raw.is_processed = True
        created += 1

    db.commit()
    logger.info(
        "Эвристический анализ: создано %d статей, пропущено дубликатов %d из %d",
        created, skipped, len(raw_news_list),
    )
    return created


def process_unprocessed(db: Session, limit: int = BATCH_SIZE) -> int:
    """Находит необработанные новости и анализирует их (без LLM)."""
    raw_list = (
        db.query(RawNews)
        .filter(RawNews.is_processed == False, RawNews.is_duplicate == False)
        .order_by(RawNews.created_at.asc())
        .limit(limit)
        .all()
    )
    if not raw_list:
        return 0
    logger.info("Эвристический анализ: %d необработанных новостей", len(raw_list))
    return analyze_batch(db, raw_list)


# --- CLI: python -m app.services.news_analyzer_heuristic --once -----------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Эвристический анализ новостей без LLM")
    parser.add_argument("--once", action="store_true", help="Один проход и выйти")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help="Размер пакета")
    args = parser.parse_args()

    with get_db_context() as db:
        from app.services.news_analyzer import get_stats

        stats = get_stats(db)
        print(f"До обработки: {stats}")

        if args.once:
            created = process_unprocessed(db, limit=args.batch)
            stats = get_stats(db)
            print(f"Создано: {created}, после: {stats}")
        else:
            while True:
                created = process_unprocessed(db, limit=args.batch)
                if created == 0:
                    print("Нет необработанных новостей")
                    break
                stats = get_stats(db)
                print(f"Обработано: {created}, статус: {stats}")
