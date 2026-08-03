"""
Клиент для LLM (совместимый с OpenAI Chat Completions API).

Поддерживает:
- OmniRoute (локальный шлюз): http://localhost:20128 — без ключа
- OpenRouter (облачный): https://openrouter.ai/api/v1 — с ключом sk-or-...

Настройки в .env:
    LLM_BASE_URL=http://localhost:20128    # адрес шлюза
    LLM_API_KEY=                           # пусто для локального шлюза
    LLM_MODEL=                             # пусто = модель по умолчанию шлюза

Использование:
    from app.services.llm import ask_llm
    answer = ask_llm("Что такое RSI?")
    answer = ask_llm("Проанализируй новости", system="Ты аналитик рынка")
"""

import json
import logging
from typing import Optional

import requests

from app.core.config import settings

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 180  # секунд (LLM может думать долго на больших промптах)


def _get_endpoint() -> tuple[str, dict]:
    """Возвращает (URL, headers) для запроса к LLM."""
    base = settings.LLM_BASE_URL.rstrip("/")
    api_key = settings.LLM_API_KEY

    # Определяем URL: если base уже содержит /v1, используем как есть
    if "/v1" in base:
        url = f"{base}/chat/completions"
    else:
        url = f"{base}/v1/chat/completions"

    headers = {"Content-Type": "application/json"}

    # Добавляем авторизацию только если ключ задан
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Для OpenRouter — дополнительные заголовки
    if "openrouter.ai" in base:
        headers["HTTP-Referer"] = "https://github.com/pantsonly"
        headers["X-Title"] = "PantsOnly"

    return url, headers


def ask_llm(
    prompt: str,
    *,
    system: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    reasoning_effort: Optional[str] = None,
) -> str:
    """
    Отправляет запрос к LLM и возвращает текст ответа.

    Args:
        prompt: Пользовательский промпт.
        system: Системный промпт (опционально).
        model: Модель (по умолчанию из config LLM_MODEL или модель шлюза).
        temperature: Креативность (0.0–1.0).
        max_tokens: Максимум токенов в ответе.

    Returns:
        str: Текст ответа модели.

    Raises:
        ValueError: Если ответ не содержит выборки.
        requests.RequestException: При сетевых ошибках.
    """
    url, headers = _get_endpoint()
    model = model or settings.LLM_MODEL  # пусто = шлюз выберет модель по умолчанию

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,  # OmniRoute стримит по умолчанию; отключаем для обычного JSON
    }
    if model:
        payload["model"] = model
    # Хинт для reasoning-моделей (DeepSeek и т.п.): без него модель тратит
    # весь max_tokens на reasoning_content и возвращает пустой content.
    # Модели/шлюзы без поддержки параметр игнорируют.
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    display_model = model or "(модель шлюза по умолчанию)"
    logger.info("LLM запрос: %s, модель=%s, длина промпта=%d", url, display_model, len(prompt))

    resp = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)

    if resp.status_code != 200:
        logger.error("LLM ошибка %d: %s", resp.status_code, resp.text[:500])
        raise ValueError(f"LLM вернул {resp.status_code}: {resp.text[:200]}")

    data = resp.json()

    # Извлекаем текст ответа (совместимо с OpenAI API)
    choices = data.get("choices", [])
    if not choices:
        raise ValueError(f"LLM не вернул choices: {json.dumps(data, ensure_ascii=False)[:300]}")

    content = choices[0].get("message", {}).get("content", "")
    if not content:
        raise ValueError("LLM вернул пустой content")

    # Логируем статистику использования
    usage = data.get("usage", {})
    if usage:
        logger.info(
            "LLM ответ: model=%s, prompt_tokens=%d, completion_tokens=%d",
            data.get("model", display_model),
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )

    return content.strip()


# --- Тестовый запуск: python -m app.services.llm ---

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    prompt = "Что такое RSI (индекс относительной силы)? Ответь на русском, 2 предложения."
    system = "Ты — аналитик фондового рынка. Отвечай кратко и по делу."

    print(f"Эндпоинт: {settings.LLM_BASE_URL}")
    print(f"Модель: {settings.LLM_MODEL or '(по умолчанию шлюза)'}")
    print(f"Промпт: {prompt}")
    print("---")

    try:
        answer = ask_llm(prompt, system=system)
        print(f"Ответ:\n{answer}")
    except Exception as e:
        print(f"ОШИБКА: {e}", file=sys.stderr)
        sys.exit(1)
