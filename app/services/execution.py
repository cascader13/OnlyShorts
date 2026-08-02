"""
Исполнение торговых операций на T-Invest (короткие позиции).

Продвинутый слой над SDK: резолвит тикеры в FIGI, выставляет рыночные
ордера (SELL для открытия шорта, BUY для закрытия) и следит за их
состоянием. Работает и в песочнице, и в боевом режиме — адрес и токен
выбирает build_client() из market_data по settings.TINKOFF_SANDBOX.

Ключевые решения:
- Каждый ордер получает uuid4 order_id — это ключ идемпотентности:
  повторная подача с тем же order_id не создаст второй ордер, поэтому
  _retry безопасен даже для post_order (сеть упала после приёма заявки —
  повторим с тем же order_id и получим ту же заявку).
- _retry с экспоненциальным backoff повторяет только транзиентные ошибки
  (INTERNAL 70001 песочницы, UNAVAILABLE и т.п.); постоянные ошибки
  (неверные параметры, отсутствие инструмента, нет прав) пробрасываются сразу.
- Все функции возвращают структурированный dict
  {order_id, status, executed_quantity, average_price, message, raw},
  где raw — исходный объект SDK для глубокой отладки (не сериализуется).
"""

import logging
import time
from uuid import uuid4

from grpc import StatusCode

from t_tech.invest.schemas import (
    InstrumentIdType,
    OrderDirection,
    OrderIdType,
    OrderType,
    PriceType,
)

from app.core.config import settings
from app.services.market_data import DEFAULT_CLASS_CODE, build_client

logger = logging.getLogger(__name__)

# Режим подключения (для логов): песочница или боевой API
_MODE = "sandbox" if settings.TINKOFF_SANDBOX else "live"

# --- Ретраи ---

_RETRIES = 5
_RETRY_BASE_DELAY = 1.0

# gRPC-коды, которые не стоит повторять: перезапрос не поможет
_NON_RETRYABLE_CODES = {
    StatusCode.INVALID_ARGUMENT,
    StatusCode.NOT_FOUND,
    StatusCode.ALREADY_EXISTS,
    StatusCode.PERMISSION_DENIED,
    StatusCode.UNAUTHENTICATED,
    StatusCode.FAILED_PRECONDITION,
    StatusCode.OUT_OF_RANGE,
    StatusCode.UNIMPLEMENTED,
}

# Человекочитаемые описания статусов исполнения (ключ — короткое имя статуса)
_STATUS_MESSAGES = {
    "FILL": "ордер исполнен полностью",
    "PARTIALLYFILL": "ордер исполнен частично",
    "NEW": "ордер выставлен, ждёт исполнения",
    "REJECTED": "ордер отклонён",
    "CANCELLED": "ордер отменён",
    "UNSPECIFIED": "статус не указан",
}


def _retry(fn, tries: int = _RETRIES, base_delay: float = _RETRY_BASE_DELAY):
    """Повторяет вызов с экспоненциальным backoff при транзиентных ошибках.

    Постоянные ошибки (неверные параметры, отсутствие инструмента, нет прав)
    пробрасываются сразу — ретраи лишь растянули бы неудачу.
    """
    last = None
    for attempt in range(tries):
        try:
            return fn()
        except Exception as exc:  # ловим всё, как в sandbox_account._retry
            last = exc
            code = getattr(exc, "code", None)
            if code in _NON_RETRYABLE_CODES:
                raise
            logger.warning(
                "execution (%s) retry %d/%d: %s: %s",
                _MODE, attempt + 1, tries, type(exc).__name__, exc,
            )
            time.sleep(base_delay * (2 ** attempt))
    raise last


def _money_to_float(v) -> float:
    """MoneyValue/Quotation -> float (units + nano/1e9)."""
    if v is None:
        return 0.0
    return float(v.units) + float(v.nano) / 1e9


def _status_name(status) -> str:
    """EXECUTION_REPORT_STATUS_FILL -> 'FILL' (короткое имя статуса)."""
    name = getattr(status, "name", None) or str(status)
    return name.rsplit("_", 1)[-1] if "_" in name else name


def _order_message(status) -> str:
    """Человекочитаемое описание статуса исполнения."""
    short = _status_name(status)
    return _STATUS_MESSAGES.get(short, short)


def _order_result(order_id: str, status: str, executed_quantity: int = 0,
                  average_price: float = 0.0, message: str = "",
                  raw=None) -> dict:
    """Нормализованный результат ордера для агента."""
    return {
        "order_id": order_id,
        "status": status,
        "executed_quantity": executed_quantity,
        "average_price": average_price,
        "message": message or "",
        "raw": raw,
    }


# --- Резолвинг инструментов ---

def resolve_figi(ticker: str) -> str:
    """Тикер -> FIGI. Точный запрос по площадке TQBR, фолбэк — find_instrument."""
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("ticker не должен быть пустым")

    def _call(sdk):
        try:
            resp = sdk.instruments.get_instrument_by(
                id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_TICKER,
                class_code=DEFAULT_CLASS_CODE,
                id=ticker,
            )
            return resp.instrument.figi
        except Exception:
            # Фолбэк — поиск по имени/тикеру
            resp = sdk.instruments.find_instrument(query=ticker)
            for item in resp.instruments:
                if item.ticker.upper() == ticker:
                    return item.figi
            return None

    with build_client() as sdk:
        figi = _retry(lambda: _call(sdk))
    if not figi:
        raise ValueError(f"FIGI для {ticker} не найден в T-Invest")
    logger.info("resolve_figi (%s): %s -> %s", _MODE, ticker, figi)
    return figi


def get_lot_size(figi: str) -> int:
    """Размер лота инструмента по FIGI (целое число акций в лоте)."""
    def _call(sdk):
        resp = sdk.instruments.get_instrument_by(
            id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_FIGI,
            id=figi,
        )
        return int(resp.instrument.lot or 1)

    with build_client() as sdk:
        lot = _retry(lambda: _call(sdk))
    logger.info("get_lot_size (%s): figi=%s lot=%s", _MODE, figi, lot)
    return lot


# --- Ордера ---

def post_market_order(
    account_id: str,
    figi: str,
    quantity_lots: int,
    direction: OrderDirection,
    order_id: str | None = None,
) -> dict:
    """Рыночный ордер по инструменту (исполнение по лучшей доступной цене).

    quantity_lots — число лотов, а не акций. order_id — ключ идемпотентности;
    если не передан, генерируется uuid4. Повторный вызов с тем же order_id
    не создаст второй ордер.
    """
    order_id = order_id or str(uuid4())
    if quantity_lots <= 0:
        raise ValueError("quantity_lots должен быть положительным")

    logger.info(
        "post_market_order (%s): account=%s figi=%s lots=%s direction=%s order_id=%s",
        _MODE, account_id, figi, quantity_lots, direction.name, order_id,
    )

    def _call(sdk):
        return sdk.orders.post_order(
            figi=figi,
            quantity=int(quantity_lots),
            direction=direction,
            account_id=account_id,
            order_type=OrderType.ORDER_TYPE_MARKET,
            order_id=order_id,
        )

    with build_client() as sdk:
        resp = _retry(lambda: _call(sdk))

    result = _order_result(
        order_id=resp.order_id or order_id,
        status=_status_name(resp.execution_report_status),
        executed_quantity=int(resp.lots_executed),
        average_price=_money_to_float(resp.executed_order_price),
        message=resp.message or _order_message(resp.execution_report_status),
        raw=resp,
    )
    logger.info(
        "post_market_order: order_id=%s status=%s executed=%s avg_price=%.4f",
        result["order_id"], result["status"], result["executed_quantity"],
        result["average_price"],
    )
    return result


def open_short(account_id: str, ticker: str, quantity_lots: int) -> dict:
    """Открывает шорт: продажа quantity_lots лотов по рынку."""
    figi = resolve_figi(ticker)
    logger.info("open_short (%s): %s lots=%s figi=%s", _MODE, ticker, quantity_lots, figi)
    return post_market_order(
        account_id=account_id,
        figi=figi,
        quantity_lots=quantity_lots,
        direction=OrderDirection.ORDER_DIRECTION_SELL,
    )


def close_short(account_id: str, ticker: str, quantity_lots: int) -> dict:
    """Закрывает шорт: выкуп (cover) quantity_lots лотов по рынку."""
    figi = resolve_figi(ticker)
    logger.info("close_short (%s): %s lots=%s figi=%s", _MODE, ticker, quantity_lots, figi)
    return post_market_order(
        account_id=account_id,
        figi=figi,
        quantity_lots=quantity_lots,
        direction=OrderDirection.ORDER_DIRECTION_BUY,
    )


def cancel_order(account_id: str, order_id: str) -> dict:
    """Отменяет активный ордер и возвращает его итоговое состояние."""
    logger.info("cancel_order (%s): account=%s order_id=%s", _MODE, account_id, order_id)

    def _call(sdk):
        # Все наши order_id — uuid4 (ключ запроса), поэтому явно указываем
        # ORDER_ID_TYPE_REQUEST: с UNSPECIFIED песочница не находит ордер (50005)
        sdk.orders.cancel_order(
            account_id=account_id,
            order_id=order_id,
            order_id_type=OrderIdType.ORDER_ID_TYPE_REQUEST,
        )

    with build_client() as sdk:
        _retry(lambda: _call(sdk))

    # После отмены запрашиваем финальное состояние: ордер мог успеть
    # исполниться частично до отмены
    try:
        return get_order_state(account_id, order_id)
    except Exception:
        logger.exception("cancel_order: не удалось получить состояние %s", order_id)
        return _order_result(order_id=order_id, status="CANCELLED",
                             message="ордер отменён")


def get_order_state(account_id: str, order_id: str) -> dict:
    """Текущее состояние ордера: статус, исполненный объём, средняя цена."""
    logger.info("get_order_state (%s): account=%s order_id=%s", _MODE, account_id, order_id)

    def _call(sdk):
        return sdk.orders.get_order_state(
            account_id=account_id,
            order_id=order_id,
            price_type=PriceType.PRICE_TYPE_CURRENCY,
            order_id_type=OrderIdType.ORDER_ID_TYPE_REQUEST,
        )

    with build_client() as sdk:
        state = _retry(lambda: _call(sdk))

    result = _order_result(
        order_id=state.order_id or order_id,
        status=_status_name(state.execution_report_status),
        executed_quantity=int(state.lots_executed),
        average_price=_money_to_float(state.executed_order_price),
        message=_order_message(state.execution_report_status),
        raw=state,
    )
    logger.info(
        "get_order_state: order_id=%s status=%s executed=%s",
        result["order_id"], result["status"], result["executed_quantity"],
    )
    return result
