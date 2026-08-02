"""
Счёт и портфель в песочнице T-Invest.

Песочница периодически отвечает INTERNAL 70001 (нестабильный публичный API),
поэтому все обращения обёрнуты в _retry с backoff. SDK помечает методы
песочницы deprecated, но они единственные, кто ходит в песочницу
(sandbox_token), поэтому используем именно их.

Конвертация MoneyValue/Quotation -> float в _money_to_float (units + nano/1e9).
"""

import logging
import time
from typing import Optional

from t_tech.invest.schemas import MoneyValue

from app.services.market_data import build_client

logger = logging.getLogger(__name__)

# Сколько раз повторяем вызов при транзиентной ошибке песочницы
_RETRIES = 6
# Базовая пауза между повторами (сек), плюс случайный разброс
_RETRY_DELAY = 2.0


def _money_to_float(v) -> float:
    """MoneyValue/Quotation -> float (units + nano/1e9)."""
    if v is None:
        return 0.0
    return float(v.units) + float(v.nano) / 1e9


def _retry(fn, tries: int = _RETRIES, delay: float = _RETRY_DELAY):
    """Повторяет вызов при транзиентных ошибках песочницы (INTERNAL 70001)."""
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as exc:
            last = exc
            logger.warning("sandbox retry %d/%d: %s", i + 1, tries, exc)
            time.sleep(delay + (i * 0.5))
    raise last


# --- Аккаунты ---

def get_accounts() -> list[dict]:
    """Список аккаунтов песочницы: [{account_id, name, status}]."""
    with build_client() as sdk:
        resp = _retry(lambda: sdk.sandbox.get_sandbox_accounts())
    return [
        {
            "account_id": a.id,
            "name": a.name or "(без имени)",
            "status": str(a.status),
        }
        for a in resp.accounts
    ]


def open_account(name: str = "PantsOnly") -> str:
    """Открывает новый аккаунт песочницы, возвращает account_id."""
    with build_client() as sdk:
        resp = _retry(lambda: sdk.sandbox.open_sandbox_account(name=name))
    return resp.account_id


def pay_in(account_id: str, amount_units: int, currency: str = "rub") -> dict:
    """Пополняет счёт песочницы, возвращает {balance, currency}."""
    with build_client() as sdk:
        resp = _retry(lambda: sdk.sandbox.sandbox_pay_in(
            account_id=account_id,
            amount=MoneyValue(currency=currency, units=int(amount_units), nano=0),
        ))
    return {
        "balance": _money_to_float(resp.balance),
        "currency": resp.balance.currency,
    }


# --- Позиции и портфель ---

def get_positions(account_id: str) -> dict:
    """Деньги и ценные бумаги по аккаунту: {money: [...], securities: [...]}.

    balance < 0 у бумаги означает шорт. money — доступные средства по валютам.
    """
    with build_client() as sdk:
        resp = _retry(lambda: sdk.operations.get_positions(account_id=account_id))
    money = [
        {"currency": m.currency, "value": _money_to_float(m)}
        for m in resp.money if _money_to_float(m) != 0
    ]
    securities = [
        {
            "ticker": s.ticker or s.figi,
            "figi": s.figi,
            "balance": int(s.balance),
            "blocked": int(s.blocked),
            "instrument_type": s.instrument_type,
        }
        for s in resp.securities
    ]
    return {"money": money, "securities": securities}


def get_portfolio(account_id: str) -> dict:
    """Портфель по аккаунту: итоги + позиции с ценами и доходностью."""
    with build_client() as sdk:
        resp = _retry(lambda: sdk.operations.get_portfolio(account_id=account_id))
    return {
        "total_amount_portfolio": _money_to_float(resp.total_amount_portfolio),
        "total_amount_currencies": _money_to_float(resp.total_amount_currencies),
        "total_amount_shares": _money_to_float(resp.total_amount_shares),
        "expected_yield": _money_to_float(resp.expected_yield),
        "daily_yield": _money_to_float(resp.daily_yield),
        "positions": [
            {
                "ticker": p.ticker or p.figi,
                "figi": p.figi,
                "instrument_type": p.instrument_type,
                "quantity": _money_to_float(p.quantity),
                "quantity_lots": _money_to_float(p.quantity_lots),
                "average_position_price": _money_to_float(p.average_position_price),
                "current_price": _money_to_float(p.current_price),
                "expected_yield": _money_to_float(p.expected_yield),
                "daily_yield": _money_to_float(p.daily_yield),
                "blocked": bool(p.blocked),
            }
            for p in resp.positions
        ],
    }
