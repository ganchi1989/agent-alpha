from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

RSIMethod = Literal["wilder", "ema", "sma"]


def _validate_period(period: int) -> int:
    if period <= 0:
        raise ValueError("period must be >= 1")
    return int(period)


def _validate_method(method: str) -> RSIMethod:
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


def _compute_rsi(
    close: pd.Series,
    *,
    period: int,
    method: RSIMethod,
) -> pd.Series:
    period = _validate_period(period)

    delta = close.astype("float64").diff()
    delta = delta.fillna(0.0)

    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    if method == "sma":
        avg_gain = gain.rolling(period, min_periods=period).mean()
        avg_loss = loss.rolling(period, min_periods=period).mean()
    elif method == "ema":
        alpha = 2.0 / (period + 1.0)
        avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    else:  # wilder
        avg_gain = pd.Series(_rma(gain.to_numpy(), period), index=gain.index, name=gain.name)
        avg_loss = pd.Series(_rma(loss.to_numpy(), period), index=loss.index, name=loss.name)

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle 0/0 and x/0 edges explicitly.
    rsi = rsi.astype("float64")
    gain0 = avg_gain.to_numpy(dtype="float64")
    loss0 = avg_loss.to_numpy(dtype="float64")
    rsi0 = rsi.to_numpy(dtype="float64")
    both0 = (gain0 == 0.0) & (loss0 == 0.0)
    rsi0 = np.where(both0, 50.0, rsi0)
    rsi0 = np.where((loss0 == 0.0) & ~both0, 100.0, rsi0)
    rsi0 = np.where((gain0 == 0.0) & (loss0 != 0.0), 0.0, rsi0)

    return pd.Series(rsi0, index=close.index, name="rsi")


def rsi(
    df: pd.DataFrame | pd.Series,
    period: int = 14,
    method: RSIMethod = "wilder",
    *,
    value: str = "close",
    shift: int = 1,
) -> pd.Series:
    method = _validate_method(method)
    period = _validate_period(period)
    shift = int(shift)
    if shift < 0:
        raise ValueError("shift must be >= 0")

    close = df if isinstance(df, pd.Series) else df[value]
    out = _compute_rsi(close, period=period, method=method).astype("float64")
    if shift:
        out = out.shift(shift)
    out.name = f"rsi_{period}_{method}"
    return out
