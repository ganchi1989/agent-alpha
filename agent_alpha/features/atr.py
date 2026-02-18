from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

ATRMethod = Literal["wilder", "ema", "sma"]


def _validate_period(period: int) -> int:
    if period <= 0:
        raise ValueError("period must be >= 1")
    return int(period)


def _validate_method(method: str) -> ATRMethod:
    m = method.strip().lower()
    if m not in {"wilder", "ema", "sma"}:
        raise ValueError("method must be one of: 'wilder', 'ema', 'sma'")
    return m  # type: ignore[return-value]


def _rma(values: np.ndarray, period: int) -> np.ndarray:
    period = _validate_period(period)
    n = int(values.size)
    out = np.full(n, np.nan, dtype="float64")
    if n == 0:
        return out

    x = np.asarray(values, dtype="float64")
    x = np.nan_to_num(x, nan=0.0)

    if n <= period:
        return out

    seed = float(np.mean(x[1 : period + 1]))
    out[period] = seed
    for i in range(period + 1, n):
        out[i] = (out[i - 1] * (period - 1) + x[i]) / period
    return out


def _compute_true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    high = high.astype("float64")
    low = low.astype("float64")
    close = close.astype("float64")

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr = tr.fillna(tr1)  # first row (no prev_close)
    return tr.astype("float64")


def _compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    period: int,
    method: ATRMethod,
) -> pd.Series:
    period = _validate_period(period)
    tr = _compute_true_range(high, low, close)

    if method == "sma":
        return tr.rolling(period, min_periods=period).mean().astype("float64")
    if method == "ema":
        alpha = 2.0 / (period + 1.0)
        return tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean().astype("float64")

    atr = pd.Series(_rma(tr.to_numpy(), period), index=tr.index, name="atr")
    return atr.astype("float64")


def atr(
    df: pd.DataFrame | pd.Series,
    period: int = 14,
    method: ATRMethod = "wilder",
    *,
    high: str = "high",
    low: str = "low",
    close: str = "close",
    shift: int = 1,
) -> pd.Series:
    """Compute the Average True Range (ATR) for a single instrument.

    ATR measures market volatility as a rolling average of the True Range,
    where True Range = max(high-low, |high-prev_close|, |low-prev_close|).

    Args:
        df: DataFrame with OHLC columns.  A bare Series is rejected.
        period: Look-back window in bars.  Must be >= 1.  Default: 14.
        method: Smoothing method for averaging the True Range.

            - ``"wilder"`` — Wilder's RMA (default; matches TradingView ATR).
            - ``"ema"`` — Exponential moving average (alpha = 2/(period+1)).
            - ``"sma"`` — Simple rolling mean.

        high: Column name for high prices.  Default: ``"high"``.
        low: Column name for low prices.  Default: ``"low"``.
        close: Column name for close prices.  Default: ``"close"``.
        shift: Number of bars to shift the result forward to avoid look-ahead
            bias.  ``shift=1`` (default) aligns the ATR value with the *next*
            bar's open.  Set to ``0`` to disable.

    Returns:
        Float64 Series named ``"atr_{period}_{method}"``, aligned to
        ``df.index``.  The first ``period`` bars are ``NaN`` due to warm-up.

    Raises:
        TypeError: If *df* is a Series rather than a DataFrame.
        ValueError: If *period* <= 0, *shift* < 0, or *method* is invalid.
    """
    method = _validate_method(method)
    period = _validate_period(period)
    shift = int(shift)
    if shift < 0:
        raise ValueError("shift must be >= 0")

    if isinstance(df, pd.Series):
        raise TypeError("atr() expects a DataFrame with OHLC columns")

    out = _compute_atr(df[high], df[low], df[close], period=period, method=method).astype("float64")
    if shift:
        out = out.shift(shift)
    out.name = f"atr_{period}_{method}"
    return out
