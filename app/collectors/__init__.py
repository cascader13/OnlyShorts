"""Пакет коллекторов: по одному классу на источник новостей.

Каждый коллектор наследует BaseCollector (сохранение + дедупликация)
и реализует collect(), возвращающий число новых записей.
"""

from app.collectors.base import BaseCollector
from app.collectors.pulse import PulseCollector
from app.collectors.rbc import RBCCollector
from app.collectors.moex import MOEXCollector
from app.collectors.tinvest import TInvestMarketCollector

# Порядок важен для читаемой статистики: pulse, rbc, moex, tinvest
ALL_COLLECTORS = [PulseCollector, RBCCollector, MOEXCollector, TInvestMarketCollector]

__all__ = [
    "BaseCollector",
    "PulseCollector",
    "RBCCollector",
    "MOEXCollector",
    "TInvestMarketCollector",
    "ALL_COLLECTORS",
]
