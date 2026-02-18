from __future__ import annotations

from typing import Literal

import pandas as pd

MACDMethod = Literal["ema", "sma"]


def _validate_period(period: int) -> int:
    if period <= 0:
        raise ValueError("period must be >= 1")
    return int(period)


def _validate_method(method: str) -> MACDMethod:
    m = method.strip().lower()
    if m not in {"ema", "sma"}:
        raise ValueError("method must be one of: 'ema', 'sma'")
    return m  # type: ignore[return-value]


def _compute_ma(values: pd.Series, *, period: int, method: MACDMethod) -> pd.Series:
    period = _validate_period(period)
    values = values.astype("float64")

    if method == "sma":
        return values.rolling(period, min_periods=period).mean().astype("float64")

    alpha = 2.0 / (period + 1.0)
    return values.ewm(alpha=alpha, adjust=False, min_periods=period).mean().astype("float64")


def _compute_macd(
    values: pd.Series,
    *,
    fast_period: int,
    slow_period: int,
    signal_period: int,
    ma_method: MACDMethod,
    signal_method: MACDMethod,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast_period = _validate_period(fast_period)
    slow_period = _validate_period(slow_period)
    signal_period = _validate_period(signal_period)
    if fast_period >= slow_period:
        raise ValueError("fast_period must be < slow_period")

    fast = _compute_ma(values, period=fast_period, method=ma_method)
    slow = _compute_ma(values, period=slow_period, method=ma_method)
    macd = (fast - slow).astype("float64")
    signal = _compute_ma(macd, period=signal_period, method=signal_method)
    hist = (macd - signal).astype("float64")
    return macd, signal, hist


def _validate_shift(shift: int) -> int:
    shift = int(shift)
    if shift < 0:
        raise ValueError("shift must be >= 0")
    return shift


def macd_line(
    df: pd.DataFrame | pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    ma_method: MACDMethod = "ema",
    signal_method: MACDMethod | None = None,
    *,
    value: str = "close",
    shift: int = 1,
) -> pd.Series:
    ma_method_v = _validate_method(ma_method)
    signal_method_v = ma_method_v if signal_method is None else _validate_method(signal_method)
    shift = _validate_shift(shift)

    values = df if isinstance(df, pd.Series) else df[value]
    macd, _, _ = _compute_macd(
        values.astype("float64"),
        fast_period=fast_period,
        slow_period=slow_period,
        signal_period=signal_period,
        ma_method=ma_method_v,
        signal_method=signal_method_v,
    )
    out = macd.astype("float64")
    if shift:
        out = out.shift(shift)
    out.name = "macd"
    return out


def macd_signal(
    df: pd.DataFrame | pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    ma_method: MACDMethod = "ema",
    signal_method: MACDMethod | None = None,
    *,
    value: str = "close",
    shift: int = 1,
) -> pd.Series:
    ma_method_v = _validate_method(ma_method)
    signal_method_v = ma_method_v if signal_method is None else _validate_method(signal_method)
    shift = _validate_shift(shift)

    values = df if isinstance(df, pd.Series) else df[value]
    _, sig, _ = _compute_macd(
        values.astype("float64"),
        fast_period=fast_period,
        slow_period=slow_period,
        signal_period=signal_period,
        ma_method=ma_method_v,
        signal_method=signal_method_v,
    )
    out = sig.astype("float64")
    if shift:
        out = out.shift(shift)
    out.name = "macd_signal"
    return out


def macd_hist(
    df: pd.DataFrame | pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    ma_method: MACDMethod = "ema",
    signal_method: MACDMethod | None = None,
    *,
    value: str = "close",
    shift: int = 1,
) -> pd.Series:
    ma_method_v = _validate_method(ma_method)
    signal_method_v = ma_method_v if signal_method is None else _validate_method(signal_method)
    shift = _validate_shift(shift)

    values = df if isinstance(df, pd.Series) else df[value]
    _, _, hist = _compute_macd(
        values.astype("float64"),
        fast_period=fast_period,
        slow_period=slow_period,
        signal_period=signal_period,
        ma_method=ma_method_v,
        signal_method=signal_method_v,
    )
    out = hist.astype("float64")
    if shift:
        out = out.shift(shift)
    out.name = "macd_hist"
    return out
