"""
Агент SMA (Simple Moving Average).

Скользящие средние определяют тренд:
- SMA20 > SMA50: бычий тренд (рост)
- SMA20 < SMA50: медвежий тренд (падение)
- Пересечение (кроссовок): смена тренда
- Цена выше/ниже SMA: направление относительно тренда
"""

import pandas as pd
from dataclasses import dataclass


@dataclass
class SMASignal:
    """Результат анализа SMA."""
    sma20: float           # Текущее значение SMA20
    sma50: float           # Текущее значение SMA50
    price_vs_sma20: float  # Отклонение цены от SMA20 (%)
    price_vs_sma50: float  # Отклонение цены от SMA50 (%)
    trend: str             # "bullish", "bearish", "neutral"
    cross: str             # "golden_cross", "death_cross", "none"
    strength: float        # Сила сигнала (0-1)
    is_short: bool         # True если медвежий тренд
    is_long: bool          # True если бычий тренд


class SMAAgent:
    """Агент для расчёта и анализа SMA."""

    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        self.fast_period = fast_period
        self.slow_period = slow_period

    def calculate(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Рассчитывает SMA20 и SMA50."""
        sma_fast = df["close"].rolling(self.fast_period).mean()
        sma_slow = df["close"].rolling(self.slow_period).mean()
        return sma_fast, sma_slow

    def analyze(self, df: pd.DataFrame) -> SMASignal:
        """Анализирует текущий тренд и возвращает сигнал."""
        sma_fast, sma_slow = self.calculate(df)

        current_price = df["close"].iloc[-1]
        current_fast = sma_fast.iloc[-1] if len(sma_fast) > 0 else current_price
        current_slow = sma_slow.iloc[-1] if len(sma_slow) > 0 else current_price

        # Отклонение цены от SMA (%)
        price_vs_fast = ((current_price - current_fast) / current_fast * 100) if current_fast else 0
        price_vs_slow = ((current_price - current_slow) / current_slow * 100) if current_slow else 0

        # Определяем тренд
        if current_fast > current_slow:
            trend = "bullish"
            strength = min((current_fast - current_slow) / current_slow * 100, 1.0) if current_slow else 0
        elif current_fast < current_slow:
            trend = "bearish"
            strength = min((current_slow - current_fast) / current_slow * 100, 1.0) if current_slow else 0
        else:
            trend = "neutral"
            strength = 0.0

        # Определяем кроссовок (сравниваем с предыдущим периодом)
        cross = "none"
        if len(sma_fast) >= 2 and len(sma_slow) >= 2:
            prev_fast = sma_fast.iloc[-2]
            prev_slow = sma_slow.iloc[-2]
            if prev_fast <= prev_slow and current_fast > current_slow:
                cross = "golden_cross"  # Бычий кроссовок
            elif prev_fast >= prev_slow and current_fast < current_slow:
                cross = "death_cross"   # Медвежий кроссовок

        return SMASignal(
            sma20=round(current_fast, 2),
            sma50=round(current_slow, 2),
            price_vs_sma20=round(price_vs_fast, 2),
            price_vs_sma50=round(price_vs_slow, 2),
            trend=trend,
            cross=cross,
            strength=round(min(strength, 1.0), 3),
            is_short=trend == "bearish",
            is_long=trend == "bullish",
        )
