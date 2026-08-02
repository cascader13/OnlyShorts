"""
Агент волатильности.

Измеряет амплитуду колебаний цены:
- ATR (Average True Range): средний диапазон свечи
- Bollinger Bands: ценовой канал на основе стандартного отклонения
- Высокая волатильность = потенциальный разворот
- Низкая волатильность = затишье перед движением
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class VolatilitySignal:
    """Результат анализа волатильности."""
    atr: float              # Average True Range (абсолютное значение)
    atr_percent: float      # ATR как % от цены
    bb_width: float         # Ширина Bollinger Bands (%)
    bb_position: float      # Позиция цены в канале (0-1, 0=нижняя граница, 1=верхняя)
    volatility_level: str   # "high", "medium", "low"
    is_reversal_risk: bool  # Высокая волатильность = риск разворота
    strength: float         # Сила сигнала (0-1)


class VolatilityAgent:
    """Агент для расчёта и анализа волатильности."""

    def __init__(self, atr_period: int = 14, bb_period: int = 20, bb_std: float = 2.0):
        self.atr_period = atr_period
        self.bb_period = bb_period
        self.bb_std = bb_std

    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Рассчитывает ATR (Average True Range)."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_period).mean()
        return atr

    def calculate_bollinger(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Рассчитывает Bollinger Bands (upper, middle, lower)."""
        middle = df["close"].rolling(self.bb_period).mean()
        std = df["close"].rolling(self.bb_period).std()
        upper = middle + self.bb_std * std
        lower = middle - self.bb_std * std
        return upper, middle, lower

    def analyze(self, df: pd.DataFrame) -> VolatilitySignal:
        """Анализирует текущую волатильность и возвращает сигнал."""
        atr = self.calculate_atr(df)
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger(df)

        current_price = df["close"].iloc[-1]
        current_atr = atr.iloc[-1] if len(atr) > 0 else 0
        current_upper = bb_upper.iloc[-1] if len(bb_upper) > 0 else current_price
        current_lower = bb_lower.iloc[-1] if len(bb_lower) > 0 else current_price

        # ATR как % от цены
        atr_percent = (current_atr / current_price * 100) if current_price else 0

        # Ширина Bollinger Bands (%)
        bb_width = ((current_upper - current_lower) / current_price * 100) if current_price else 0

        # Позиция цены в канале (0-1)
        bb_range = current_upper - current_lower
        if bb_range > 0:
            bb_position = (current_price - current_lower) / bb_range
        else:
            bb_position = 0.5

        # Определяем уровень волатильности
        if atr_percent > 3.0:
            volatility_level = "high"
            strength = min(atr_percent / 5.0, 1.0)
        elif atr_percent > 1.5:
            volatility_level = "medium"
            strength = atr_percent / 3.0
        else:
            volatility_level = "low"
            strength = 0.0

        # Риск разворота: высокая волатильность + цена у границы канала
        is_reversal_risk = (
            volatility_level == "high" and
            (bb_position > 0.9 or bb_position < 0.1)
        )

        return VolatilitySignal(
            atr=round(current_atr, 2),
            atr_percent=round(atr_percent, 2),
            bb_width=round(bb_width, 2),
            bb_position=round(bb_position, 3),
            volatility_level=volatility_level,
            is_reversal_risk=is_reversal_risk,
            strength=round(min(strength, 1.0), 3),
        )
