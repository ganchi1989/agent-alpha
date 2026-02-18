from __future__ import annotations

import pandas as pd


def _validate_period(period: int) -> int:
    if period <= 0:
        raise ValueError("period must be >= 1")
    return int(period)


def _alpha_from_period(period: int) -> float:
    period = _validate_period(period)
    return 2.0 / (period + 1.0)


def sma(
    df: pd.DataFrame | pd.Series,
    period: int = 20,
    *,
    value: str = "close",
    shift: int = 1,
) -> pd.Series:
    period = _validate_period(period)
    shift = int(shift)
    if shift < 0:
        raise ValueError("shift must be >= 0")

    s = df if isinstance(df, pd.Series) else df[value]
    out = s.astype("float64").rolling(window=period, min_periods=period).mean().astype("float64")
    if shift:
        out = out.shift(shift)
    out.name = f"sma_{period}"
    return out


def ema(
    df: pd.DataFrame | pd.Series,
    period: int = 20,
    *,
    value: str = "close",
    shift: int = 1,
) -> pd.Series:
    period = _validate_period(period)
    shift = int(shift)
    if shift < 0:
        raise ValueError("shift must be >= 0")

    alpha = _alpha_from_period(period)
    s = df if isinstance(df, pd.Series) else df[value]
    out = s.astype("float64").ewm(alpha=alpha, adjust=False, min_periods=period).mean().astype("float64")
    if shift:
        out = out.shift(shift)
    out.name = f"ema_{period}"
    return out
