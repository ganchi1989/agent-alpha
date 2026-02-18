from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd

RVEstimator = Literal["cc", "parkinson", "garman_klass", "rogers_satchell"]
RVSmoother = Literal["ewm", "rolling"]


def _validate_period(period: int) -> int:
    if period <= 0:
        raise ValueError("period must be >= 1")
    return int(period)


def _validate_estimator(estimator: str) -> RVEstimator:
    e = estimator.strip().lower()
    if e == "gk":
        e = "garman_klass"
    if e == "rs":
        e = "rogers_satchell"
    if e not in {"cc", "parkinson", "garman_klass", "rogers_satchell"}:
        raise ValueError(
            "estimator must be one of: 'cc', 'parkinson', 'garman_klass', 'rogers_satchell'"
        )
    return e  # type: ignore[return-value]


def _validate_smoother(smoother: str) -> RVSmoother:
    s = smoother.strip().lower()
    if s not in {"ewm", "rolling"}:
        raise ValueError("smoother must be one of: 'ewm', 'rolling'")
    return s  # type: ignore[return-value]


def _validate_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]")
    return alpha


def _alpha_from_period(period: int) -> float:
    period = _validate_period(period)
    return 2.0 / (period + 1.0)


def _safe_log(x: pd.Series) -> pd.Series:
    x = x.astype("float64")
    return np.log(x.where(x > 0.0))


def _var_proxy(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    estimator: RVEstimator,
) -> pd.Series:
    if estimator == "cc":
        r = _safe_log(close).diff()
        return (r * r).astype("float64")

    if estimator == "parkinson":
        hl = _safe_log(high) - _safe_log(low)
        return ((hl * hl) / (4.0 * math.log(2.0))).astype("float64")

    if estimator == "garman_klass":
        hl = _safe_log(high) - _safe_log(low)
        co = _safe_log(close) - _safe_log(open_)
        v = 0.5 * (hl * hl) - (2.0 * math.log(2.0) - 1.0) * (co * co)
        return v.clip(lower=0.0).astype("float64")

    # rogers_satchell
    ho = _safe_log(high) - _safe_log(open_)
    hc = _safe_log(high) - _safe_log(close)
    lo = _safe_log(low) - _safe_log(open_)
    lc = _safe_log(low) - _safe_log(close)
    v = ho * hc + lo * lc
    return v.clip(lower=0.0).astype("float64")


def _rv_from_var_proxy(
    proxy: pd.Series,
    *,
    period: int,
    smoother: RVSmoother,
    alpha: float,
) -> pd.Series:
    period = _validate_period(period)
    if smoother == "rolling":
        var = proxy.rolling(period, min_periods=period).mean()
    else:
        var = proxy.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    return np.sqrt(var.astype("float64"))


def _validate_shift(shift: int) -> int:
    shift = int(shift)
    if shift < 0:
        raise ValueError("shift must be >= 0")
    return shift


def rv(
    df: pd.DataFrame,
    period: int = 20,
    estimator: RVEstimator = "cc",
    smoother: RVSmoother = "ewm",
    alpha: float | None = None,
    *,
    open: str = "open",
    high: str = "high",
    low: str = "low",
    close: str = "close",
    shift: int = 1,
) -> pd.Series:
    period = _validate_period(period)
    estimator = _validate_estimator(estimator)
    smoother = _validate_smoother(smoother)
    if alpha is None:
        alpha = _alpha_from_period(period)
    alpha = _validate_alpha(alpha)
    shift = _validate_shift(shift)

    proxy = _var_proxy(df[open], df[high], df[low], df[close], estimator=estimator)
    out = _rv_from_var_proxy(proxy, period=period, smoother=smoother, alpha=alpha).astype("float64")
    if shift:
        out = out.shift(shift)
    out.name = "rv"
    return out
