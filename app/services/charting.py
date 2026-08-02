"""
Построение графиков рыночных данных (plotly) — чистая логика без Streamlit.

Используется дашбордом app/frontend.py; функции можно тестировать напрямую.
Цвета — из палитры dataviz (validate_palette.js: светлая и тёмная прошли
проверки; серийные цвета — categorical-слоты, up/down — статусные good/critical).
"""

from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Палитры по темам. Ключи: surface, grid, axis, ink, sma20, sma50, rsi,
# up, down, muted.
LIGHT_PALETTE: dict[str, str] = {
    "surface": "#ffffff",
    "grid": "#e1e0d9",
    "axis": "#898781",
    "ink": "#52514e",
    "sma20": "#2a78d6",
    "sma50": "#eb6834",
    "rsi": "#1baf7a",
    "up": "#0ca30c",    # статус good
    "down": "#d03b3b",  # статус critical
    "muted": "#c3c2b7",
}

DARK_PALETTE: dict[str, str] = {
    "surface": "#1a1a19",
    "grid": "#2c2c2a",
    "axis": "#898781",
    "ink": "#c3c2b7",
    "sma20": "#3987e5",
    "sma50": "#d95926",
    "rsi": "#199e70",
    "up": "#0ca30c",
    "down": "#d03b3b",
    "muted": "#383835",
}


def candles_to_df(rows) -> pd.DataFrame:
    """Candle-объекты -> DataFrame с tz-aware UTC индексом (для plotly)."""
    df = pd.DataFrame([
        {
            "ts": r.ts,
            "open": r.open,
            "high": r.high,
            "low": r.low,
            "close": r.close,
            "volume": r.volume,
        }
        for r in rows
    ])
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.set_index("ts").sort_index()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """SMA20/SMA50 и RSI(14, Wilder) поверх DataFrame."""
    df = df.copy()
    close = df["close"]
    df["sma20"] = close.rolling(20).mean()
    df["sma50"] = close.rolling(50).mean()

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    # Крайние случаи: совсем нет убытков -> RSI 100; плоский участок -> 50
    rsi = rsi.mask(avg_loss == 0, 100.0)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
    df["rsi"] = rsi
    return df


def build_chart(df: pd.DataFrame, ticker: str, tf_label: str,
                palette: dict[str, str]) -> go.Figure:
    """Свечи + SMA + объём + RSI в трёх панелях с общей осью времени."""
    p = palette
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.18, 0.27],
        vertical_spacing=0.04,
        subplot_titles=("Цена", "Объём", "RSI (14)"),
    )

    # Панель 1: свечи + скользящие средние
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Свечи",
        increasing_line_color=p["up"], increasing_fillcolor=p["up"],
        decreasing_line_color=p["down"], decreasing_fillcolor=p["down"],
        line_width=1,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["sma20"], name="SMA 20",
        line=dict(color=p["sma20"], width=2),
        hovertemplate="SMA20 %{y:.2f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["sma50"], name="SMA 50",
        line=dict(color=p["sma50"], width=2),
        hovertemplate="SMA50 %{y:.2f}<extra></extra>",
    ), row=1, col=1)

    # Панель 2: объём, цвет по направлению свечи
    bar_colors = [
        p["up"] if c >= o else p["down"]
        for o, c in zip(df["open"], df["close"])
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=df["volume"],
        name="Объём", marker_color=bar_colors, opacity=0.55,
        hovertemplate="Объём %{y:,.0f}<extra></extra>",
    ), row=2, col=1)

    # Панель 3: RSI + референс-уровни 30/70
    fig.add_trace(go.Scatter(
        x=df.index, y=df["rsi"], name="RSI 14",
        line=dict(color=p["rsi"], width=2),
        hovertemplate="RSI %{y:.1f}<extra></extra>",
    ), row=3, col=1)
    fig.add_hline(y=70, line=dict(color=p["muted"], dash="dot", width=1), row=3, col=1)
    fig.add_hline(y=30, line=dict(color=p["muted"], dash="dot", width=1), row=3, col=1)
    fig.add_hline(y=50, line=dict(color=p["muted"], dash="dash", width=1), row=3, col=1)

    # Общие стили: рецессивные сетки, скромные подписи, кроссхер
    fig.update_layout(
        title=dict(
            text=f"{ticker} — {tf_label}",
            font=dict(color=p["ink"]),
        ),
        height=680,
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(color=p["ink"]),
        ),
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=p["ink"]),
    )
    fig.update_xaxes(
        showgrid=True, gridcolor=p["grid"], linecolor=p["muted"],
        tickfont=dict(color=p["axis"]), rangeslider_visible=False,
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=p["grid"], linecolor=p["muted"],
        tickfont=dict(color=p["axis"]),
    )
    # Подписи панелей — в цвет осей (рецессивные)
    for annotation in fig["layout"]["annotations"]:
        annotation["font"]["color"] = p["axis"]
    return fig
