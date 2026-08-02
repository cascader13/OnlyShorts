"""
Агент RSI (Relative Strength Index).

Измеряет скорость и величину изменения цены.
Диапазон: 0-100
- >70: перекупленность (потенциальный шорт)
- <30: перепроданность (потенциальный лонг)
- 30-70: нейтральная зона
"""

import pandas as pd
from dataclasses import dataclass


@dataclass
class RSISignal:
    """Результат анализа RSI."""
    value: float          # Текущее значение RSI (0-100)
    signal: str           # "overbought", "oversold", "neutral"
    strength: float       # Сила сигнала (0-1), чем дальше от 50, тем сильнее
    is_short: bool        # True если RSI > 70 (перекупленность)
    is_long: bool         # True если RSI < 30 (перепроданность)


class RSIAgent:
    """Агент для расчёта и анализа RSI."""

    def __init__(self, period: int = 14, overbought: float = 70, oversold: float = 30):
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Рассчитывает RSI по DataFrame с колонкой 'close'."""
        delta = df["close"].diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)

        avg_gain = gain.ewm(alpha=1 / self.period, min_periods=self.period).mean()
        avg_loss = loss.ewm(alpha=1 / self.period, min_periods=self.period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Крайние случаи
        rsi = rsi.mask(avg_loss == 0, 100.0)
        rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50.0)

        return rsi

    def analyze(self, df: pd.DataFrame) -> RSISignal:
        """Анализирует текущий RSI и возвращает сигнал."""
        rsi = self.calculate(df)
        current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50.0

        # Определяем сигнал
        if current_rsi >= self.overbought:
            signal = "overbought"
            strength = (current_rsi - self.overbought) / (100 - self.overbought)
        elif current_rsi <= self.oversold:
            signal = "oversold"
            strength = (self.oversold - current_rsi) / self.oversold
        else:
            signal = "neutral"
            strength = 0.0

        return RSISignal(
            value=round(current_rsi, 2),
            signal=signal,
            strength=round(min(strength, 1.0), 3),
            is_short=current_rsi >= self.overbought,
            is_long=current_rsi <= self.oversold,
        )
