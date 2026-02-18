from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PANEL_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class SyntheticSpec:
    seed: int
    start_date: str
    n_days: int
    n_tickers: int
    n_sectors: int

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "SyntheticSpec":
        return cls(
            seed=int(cfg.get("seed", 7)),
            start_date=str(cfg.get("start_date", "2018-01-01")),
            n_days=int(cfg.get("n_days", 756)),
            n_tickers=int(cfg.get("n_tickers", 600)),
            n_sectors=int(cfg.get("n_sectors", 10)),
        )


def _load_panel_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported panel file extension: {path.suffix}")


def validate_panel_df(panel_df: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(PANEL_COLUMNS) - set(panel_df.columns))
    if missing:
        raise ValueError(f"panel_df is missing required columns: {missing}")

    panel = panel_df[PANEL_COLUMNS].copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel["ticker"] = panel["ticker"].astype(str)

    key_dupes = panel.duplicated(subset=["date", "ticker"]).sum()
    if key_dupes:
        raise ValueError(f"panel_df has {key_dupes} duplicate (date,ticker) keys")

    if not (panel["high"] >= panel["low"]).all():
        raise ValueError("Detected rows where high < low")
    if (panel["volume"] < 0).any():
        raise ValueError("Detected rows where volume < 0")

    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    return panel


def generate_synthetic_panel(spec: SyntheticSpec) -> pd.DataFrame:
    rng = np.random.default_rng(spec.seed)
    dates = pd.bdate_range(spec.start_date, periods=spec.n_days)
    tickers = np.array([f"T{idx:04d}" for idx in range(spec.n_tickers)], dtype=object)
    sector_by_ticker = rng.integers(0, spec.n_sectors, size=spec.n_tickers)

    market = np.zeros(spec.n_days, dtype=float)
    market_shock = rng.normal(0.0, 0.006, size=spec.n_days)
    for t in range(1, spec.n_days):
        market[t] = 0.08 * market[t - 1] + market_shock[t]

    sectors = np.zeros((spec.n_days, spec.n_sectors), dtype=float)
    sector_shock = rng.normal(0.0, 0.008, size=(spec.n_days, spec.n_sectors))
    for t in range(1, spec.n_days):
        sectors[t] = 0.06 * sectors[t - 1] + sector_shock[t]

    beta_m = rng.normal(1.0, 0.18, size=spec.n_tickers)
    beta_s = rng.normal(1.0, 0.25, size=spec.n_tickers)
    idio_scale = rng.lognormal(mean=-4.6, sigma=0.25, size=spec.n_tickers)

    returns = np.zeros((spec.n_days, spec.n_tickers), dtype=float)
    idio_noise = rng.normal(0.0, 1.0, size=(spec.n_days, spec.n_tickers))
    for t in range(1, spec.n_days):
        sector_component = sectors[t, sector_by_ticker]
        returns[t] = (
            0.05 * returns[t - 1]
            + beta_m * market[t]
            + beta_s * sector_component
            + idio_noise[t] * idio_scale
        )
    returns = np.clip(returns, -0.20, 0.20)

    init_close = rng.uniform(15.0, 150.0, size=spec.n_tickers)
    close = np.empty_like(returns)
    close[0] = np.maximum(0.5, init_close * (1.0 + returns[0]))
    for t in range(1, spec.n_days):
        close[t] = np.maximum(0.5, close[t - 1] * (1.0 + returns[t]))

    overnight = rng.normal(0.0, 0.002, size=(spec.n_days, spec.n_tickers))
    open_ = np.empty_like(close)
    open_[0] = np.maximum(0.5, close[0] * (1.0 + overnight[0]))
    open_[1:] = np.maximum(0.5, close[:-1] * (1.0 + overnight[1:]))

    intraday_span = np.abs(returns) * 0.60 + np.abs(rng.normal(0.006, 0.003, size=returns.shape))
    intraday_span = np.clip(intraday_span, 0.001, 0.15)
    base_high = np.maximum(open_, close)
    base_low = np.minimum(open_, close)
    high = base_high * (1.0 + intraday_span)
    low = np.maximum(0.01, base_low * (1.0 - 0.9 * intraday_span))

    liquidity_base = rng.lognormal(mean=12.7, sigma=0.45, size=spec.n_tickers)
    volume_noise = rng.lognormal(mean=0.0, sigma=0.28, size=(spec.n_days, spec.n_tickers))
    volume = liquidity_base * (1.0 + 18.0 * np.abs(returns)) * volume_noise
    volume = np.maximum(0.0, np.round(volume))

    panel_df = pd.DataFrame(
        {
            "date": np.repeat(dates.values, spec.n_tickers),
            "ticker": np.tile(tickers, spec.n_days),
            "open": open_.reshape(-1),
            "high": high.reshape(-1),
            "low": low.reshape(-1),
            "close": close.reshape(-1),
            "volume": volume.reshape(-1),
        }
    )
    return validate_panel_df(panel_df)


def load_panel(config: dict[str, Any]) -> pd.DataFrame:
    data_cfg = config.get("data", {})
    source = str(data_cfg.get("source", "synthetic")).lower()

    if source == "synthetic":
        panel_df = generate_synthetic_panel(SyntheticSpec.from_config(data_cfg.get("synthetic", {})))
    elif source == "file":
        panel_path = data_cfg.get("panel_path")
        if not panel_path:
            raise ValueError("data.panel_path must be set when data.source=file")
        panel_df = _load_panel_file(Path(panel_path))
    else:
        raise ValueError(f"Unsupported data source: {source}")

    return validate_panel_df(panel_df)
