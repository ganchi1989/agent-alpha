"""Feature catalog and component computation for blueprint execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from .features import (
    atr,
    candle_range,
    cvd,
    dist_to_vwap,
    ema,
    kdj_d,
    kdj_k,
    kdj_j,
    macd_hist,
    rsi,
    rv,
    sma,
    wick_ratio,
)

from .models import FeatureComponentSpec

FeatureFn = Callable[..., pd.Series]


@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    """Catalog entry describing one feature function.

    Attributes:
        name: Stable feature key used in blueprint component specs.
        fn: Callable that computes the feature over one-instrument OHLCV data.
        description: Short human-readable explanation.
        default_params: Default keyword arguments merged with user overrides.
    """

    name: str
    fn: FeatureFn
    description: str
    default_params: dict[str, Any]


FEATURE_DEFINITIONS: dict[str, FeatureDefinition] = {
    "rsi": FeatureDefinition(
        name="rsi",
        fn=rsi,
        description="Relative Strength Index oscillator.",
        default_params={"period": 14, "method": "wilder", "shift": 1},
    ),
    "macd_hist": FeatureDefinition(
        name="macd_hist",
        fn=macd_hist,
        description="MACD histogram (momentum trend spread).",
        default_params={
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9,
            "ma_method": "ema",
            "shift": 1,
        },
    ),
    "atr": FeatureDefinition(
        name="atr",
        fn=atr,
        description="Average True Range (volatility proxy).",
        default_params={"period": 14, "method": "wilder", "shift": 1},
    ),
    "rv": FeatureDefinition(
        name="rv",
        fn=rv,
        description="Realized volatility from OHLC estimators.",
        default_params={"period": 20, "estimator": "cc", "smoother": "ewm", "shift": 1},
    ),
    "vwap_dist": FeatureDefinition(
        name="vwap_dist",
        fn=dist_to_vwap,
        description="Distance of close to intraday cumulative VWAP.",
        default_params={"price": "hl2", "shift": 1},
    ),
    "wick_ratio": FeatureDefinition(
        name="wick_ratio",
        fn=wick_ratio,
        description="Wick length ratio to total candle range.",
        default_params={"eps": 1e-12, "shift": 1},
    ),
    "candle_range": FeatureDefinition(
        name="candle_range",
        fn=candle_range,
        description="High-low candle range.",
        default_params={"shift": 1},
    ),
    "cvd": FeatureDefinition(
        name="cvd",
        fn=cvd,
        description="Cumulative volume delta proxy based on candle direction.",
        default_params={"zero_base": True, "shift": 1},
    ),
    "ema": FeatureDefinition(
        name="ema",
        fn=ema,
        description="Exponential moving average of close.",
        default_params={"period": 20, "value": "close", "shift": 1},
    ),
    "sma": FeatureDefinition(
        name="sma",
        fn=sma,
        description="Simple moving average of close.",
        default_params={"period": 20, "value": "close", "shift": 1},
    ),
    "kdj_j": FeatureDefinition(
        name="kdj_j",
        fn=kdj_j,
        description="KDJ J-line oscillator.",
        default_params={"period": 9, "k_smooth": 3, "d_smooth": 3, "shift": 1},
    ),
    "kdj_d": FeatureDefinition(
        name="kdj_d",
        fn=kdj_d,
        description="KDJ D-line oscillator.",
        default_params={"period": 9, "k_smooth": 3, "d_smooth": 3, "shift": 1},
    ),
    "kdj_k": FeatureDefinition(
        name="kdj_k",
        fn=kdj_k,
        description="KDJ K-line oscillator.",
        default_params={"period": 9, "k_smooth": 3, "d_smooth": 3, "shift": 1},
    ),
}


def component_column_name(component_id: str) -> str:
    """Return the panel column name used for one computed component."""

    return f"$cmp_{component_id}"


def feature_catalog_for_prompt() -> dict[str, dict[str, Any]]:
    """Return compact feature metadata consumed by the blueprint-generation prompt."""

    return {
        key: {
            "description": value.description,
            "default_params": value.default_params,
        }
        for key, value in FEATURE_DEFINITIONS.items()
    }


def _require_panel_shape(panel: pd.DataFrame) -> None:
    if not isinstance(panel.index, pd.MultiIndex):
        raise ValueError("panel must use MultiIndex(datetime, instrument)")
    if list(panel.index.names) != ["datetime", "instrument"]:
        panel.index = panel.index.set_names(["datetime", "instrument"])
    required = {"$open", "$high", "$low", "$close", "$volume"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel is missing required columns: {sorted(missing)}")


def _prepare_single_instrument_frame(group: pd.DataFrame) -> pd.DataFrame:
    frame = group.droplevel("instrument").copy()
    return frame.rename(
        columns={
            "$open": "open",
            "$high": "high",
            "$low": "low",
            "$close": "close",
            "$volume": "volume",
        }
    )


def _as_series(value: Any, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.DataFrame):
        if value.shape[1] != 1:
            raise ValueError("Feature returned multi-column DataFrame.")
        value = value.iloc[:, 0]
    if isinstance(value, pd.Series):
        out = value
    else:
        out = pd.Series(value, index=index)
    return out.reindex(index).astype("float64")


def compute_feature_component(panel: pd.DataFrame, component: FeatureComponentSpec) -> pd.Series:
    """Compute one blueprint component across all instruments in the panel.

    Args:
        panel: OHLCV panel indexed by `(datetime, instrument)`.
        component: Blueprint component descriptor.

    Returns:
        A float series aligned to `panel.index` and named with `$cmp_<id>`.
    """

    _require_panel_shape(panel)
    feature_key = component.feature.strip().lower()
    if feature_key not in FEATURE_DEFINITIONS:
        raise KeyError(f"Unknown feature component: {component.feature!r}")

    definition = FEATURE_DEFINITIONS[feature_key]
    params = {**definition.default_params, **component.params}
    pieces: list[pd.Series] = []

    for instrument, group in panel.groupby(level="instrument", sort=False):
        local = _prepare_single_instrument_frame(group)
        try:
            out = definition.fn(local, **params)
        except KeyError as exc:
            missing = exc.args[0] if exc.args else "<unknown>"
            raise ValueError(
                f"Feature {feature_key!r} referenced missing input column {missing!r}. "
                f"Valid input columns: {sorted(local.columns)}. "
                f"Component id={component.id!r}, params={params!r}"
            ) from exc
        series = _as_series(out, local.index)
        idx = pd.MultiIndex.from_arrays(
            [series.index, np.repeat(instrument, len(series))],
            names=["datetime", "instrument"],
        )
        series.index = idx
        pieces.append(series)

    merged = pd.concat(pieces).sort_index()
    merged = merged.reindex(panel.index)
    merged.name = component_column_name(component.id)
    return merged


def compute_blueprint_components(
    panel: pd.DataFrame,
    components: list[FeatureComponentSpec],
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Compute all blueprint components and append them to a copy of `panel`.

    Returns:
        Tuple `(augmented_panel, mapping)` where mapping resolves component IDs
        to the generated component column names.
    """

    _require_panel_shape(panel)
    augmented = panel.copy()
    mapping: dict[str, str] = {}
    for component in components:
        column = component_column_name(component.id)
        augmented[column] = compute_feature_component(panel, component)
        mapping[component.id] = column
    return augmented, mapping
