"""Feature functions and enums used by the feature-engine catalog."""

from __future__ import annotations

from .atr import ATRMethod, atr
from .core import FeatureQuery
from .cvd import cvd
from .kdj import kdj_d, kdj_j, kdj_k
from .macd import MACDMethod, macd_hist, macd_line, macd_signal
from .moving_averages import ema, sma
from .range_wicks import candle_range, wick_ratio
from .rsi import RSIMethod, rsi
from .rv import RVEstimator, RVSmoother, rv
from .vwap import VWAPPrice, dist_to_vwap, vwap_day

__all__ = [
    "ATRMethod",
    "FeatureQuery",
    "MACDMethod",
    "RSIMethod",
    "RVEstimator",
    "RVSmoother",
    "VWAPPrice",
    "atr",
    "candle_range",
    "cvd",
    "dist_to_vwap",
    "ema",
    "kdj_d",
    "kdj_j",
    "kdj_k",
    "macd_hist",
    "macd_line",
    "macd_signal",
    "rsi",
    "rv",
    "sma",
    "vwap_day",
    "wick_ratio",
]
