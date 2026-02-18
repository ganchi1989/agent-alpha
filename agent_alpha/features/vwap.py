from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

VWAPPrice = Literal["hl2", "hlc3", "close"]


def _validate_price(price: str) -> VWAPPrice:
    p = price.strip().lower()
    if p == "median":
        p = "hl2"
    if p not in {"hl2", "hlc3", "close"}:
        raise ValueError("price must be one of: 'hl2', 'hlc3', 'close'")
    return p  # type: ignore[return-value]


def _price_from_ohlc(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    price: VWAPPrice,
) -> pd.Series:
    if price == "close":
        return close.astype("float64")
    if price == "hlc3":
        return (
            (high.astype("float64") + low.astype("float64") + close.astype("float64")) / 3.0
        ).astype("float64")
    return ((high.astype("float64") + low.astype("float64")) / 2.0).astype("float64")


def _validate_shift(shift: int) -> int:
    shift = int(shift)
    if shift < 0:
        raise ValueError("shift must be >= 0")
    return shift


def _ts_values(df: pd.DataFrame) -> pd.Series | pd.DatetimeIndex:
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index
    if "ts" in df.columns:
        return pd.to_datetime(df["ts"], utc=True)
    raise ValueError("df must have a DatetimeIndex or a 'ts' column")


def vwap_day(
    df: pd.DataFrame,
    price: VWAPPrice = "hl2",
    *,
    high: str = "high",
    low: str = "low",
    close: str = "close",
    volume: str = "volume",
    shift: int = 1,
) -> pd.Series:
    price = _validate_price(price)
    shift = _validate_shift(shift)

    hi = df[high].astype("float64")
    lo = df[low].astype("float64")
    cl = df[close].astype("float64")
    vol = df[volume].astype("float64")

    px = _price_from_ohlc(hi, lo, cl, price=price)
    pv = (px.to_numpy(dtype="float64") * vol.to_numpy(dtype="float64")).astype("float64")
    vol0 = vol.to_numpy(dtype="float64")

    ts = _ts_values(df)
    day_key = ts.normalize() if isinstance(ts, pd.DatetimeIndex) else ts.dt.normalize()

    cum_pv = pd.Series(pv, index=df.index).groupby(day_key).cumsum().to_numpy(dtype="float64")
    cum_vol = pd.Series(vol0, index=df.index).groupby(day_key).cumsum().to_numpy(dtype="float64")
    vwap = np.divide(cum_pv, cum_vol, out=np.full_like(cum_pv, np.nan), where=cum_vol > 0.0)

    out = pd.Series(vwap, index=df.index, name="vwap_day", dtype="float64")
    if shift:
        out = out.shift(shift)
    return out


def dist_to_vwap(
    df: pd.DataFrame,
    price: VWAPPrice = "hl2",
    *,
    high: str = "high",
    low: str = "low",
    close: str = "close",
    volume: str = "volume",
    shift: int = 1,
) -> pd.Series:
    vwap = vwap_day(
        df,
        price=price,
        high=high,
        low=low,
        close=close,
        volume=volume,
        shift=0,
    )
    out = (df[close].astype("float64") - vwap) / vwap
    shift = _validate_shift(shift)
    if shift:
        out = out.shift(shift)
    out.name = "dist_to_vwap"
    return out.astype("float64")
