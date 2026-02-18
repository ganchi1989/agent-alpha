from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_eps(eps: float) -> float:
    eps = float(eps)
    if eps <= 0.0:
        raise ValueError("eps must be > 0")
    return eps


def _validate_shift(shift: int) -> int:
    shift = int(shift)
    if shift < 0:
        raise ValueError("shift must be >= 0")
    return shift


def candle_range(
    df: pd.DataFrame,
    *,
    high: str = "high",
    low: str = "low",
    shift: int = 1,
) -> pd.Series:
    shift = _validate_shift(shift)

    out = (df[high].astype("float64") - df[low].astype("float64")).astype("float64")
    if shift:
        out = out.shift(shift)
    out.name = "candle_range"
    return out


def wick_ratio(
    df: pd.DataFrame,
    eps: float = 1e-12,
    *,
    open: str = "open",
    high: str = "high",
    low: str = "low",
    close: str = "close",
    shift: int = 1,
) -> pd.Series:
    eps = _validate_eps(eps)
    shift = _validate_shift(shift)

    op = df[open].astype("float64")
    hi = df[high].astype("float64")
    lo = df[low].astype("float64")
    cl = df[close].astype("float64")

    rng = (hi - lo).astype("float64")
    top = (hi - np.maximum(op, cl)).clip(lower=0.0)
    bot = (np.minimum(op, cl) - lo).clip(lower=0.0)

    out = ((top + bot) / (rng + eps)).astype("float64")
    if shift:
        out = out.shift(shift)
    out.name = "wick_ratio"
    return out
