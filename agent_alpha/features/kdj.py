from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_period(period: int) -> int:
    if period <= 0:
        raise ValueError("period must be >= 1")
    return int(period)


def _cn_sma(values: np.ndarray, period: int, *, seed: float = 50.0) -> np.ndarray:
    period = _validate_period(period)
    n = int(values.size)
    out = np.full(n, np.nan, dtype="float64")
    if n == 0:
        return out

    x = np.asarray(values, dtype="float64")
    first_valid = int(np.argmax(np.isfinite(x)))
    if not np.isfinite(x[first_valid]):
        return out

    prev = float(seed)
    for i in range(first_valid, n):
        xi = x[i]
        if not np.isfinite(xi):
            continue
        prev = (prev * (period - 1) + xi) / period
        out[i] = prev
    return out


def _compute_kdj(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    period: int,
    k_smooth: int,
    d_smooth: int,
    seed: float = 50.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    period = _validate_period(period)
    k_smooth = _validate_period(k_smooth)
    d_smooth = _validate_period(d_smooth)

    high_n = high.astype("float64").rolling(period, min_periods=period).max()
    low_n = low.astype("float64").rolling(period, min_periods=period).min()
    denom = high_n - low_n

    rsv = 100.0 * (close.astype("float64") - low_n) / denom
    rsv = rsv.astype("float64")
    rsv = rsv.where(denom != 0.0, 50.0)
    rsv = rsv.clip(lower=0.0, upper=100.0)

    k = pd.Series(_cn_sma(rsv.to_numpy(), k_smooth, seed=seed), index=rsv.index, name="k")
    d = pd.Series(_cn_sma(k.to_numpy(), d_smooth, seed=seed), index=rsv.index, name="d")
    j = (3.0 * k - 2.0 * d).astype("float64")
    j.name = "j"
    return k.astype("float64"), d.astype("float64"), j


def _validate_shift(shift: int) -> int:
    shift = int(shift)
    if shift < 0:
        raise ValueError("shift must be >= 0")
    return shift


def kdj_k(
    df: pd.DataFrame,
    period: int = 9,
    k_smooth: int = 3,
    d_smooth: int = 3,
    *,
    high: str = "high",
    low: str = "low",
    close: str = "close",
    seed: float = 50.0,
    shift: int = 1,
) -> pd.Series:
    shift = _validate_shift(shift)

    k, _, _ = _compute_kdj(
        df[high],
        df[low],
        df[close],
        period=period,
        k_smooth=k_smooth,
        d_smooth=d_smooth,
        seed=seed,
    )
    out = k.astype("float64")
    if shift:
        out = out.shift(shift)
    out.name = "kdj_k"
    return out


def kdj_d(
    df: pd.DataFrame,
    period: int = 9,
    k_smooth: int = 3,
    d_smooth: int = 3,
    *,
    high: str = "high",
    low: str = "low",
    close: str = "close",
    seed: float = 50.0,
    shift: int = 1,
) -> pd.Series:
    shift = _validate_shift(shift)

    _, d, _ = _compute_kdj(
        df[high],
        df[low],
        df[close],
        period=period,
        k_smooth=k_smooth,
        d_smooth=d_smooth,
        seed=seed,
    )
    out = d.astype("float64")
    if shift:
        out = out.shift(shift)
    out.name = "kdj_d"
    return out


def kdj_j(
    df: pd.DataFrame,
    period: int = 9,
    k_smooth: int = 3,
    d_smooth: int = 3,
    *,
    high: str = "high",
    low: str = "low",
    close: str = "close",
    seed: float = 50.0,
    shift: int = 1,
) -> pd.Series:
    shift = _validate_shift(shift)

    _, _, j = _compute_kdj(
        df[high],
        df[low],
        df[close],
        period=period,
        k_smooth=k_smooth,
        d_smooth=d_smooth,
        seed=seed,
    )
    out = j.astype("float64")
    if shift:
        out = out.shift(shift)
    out.name = "kdj_j"
    return out
