from __future__ import annotations

import numpy as np
import pandas as pd


def load_synthetic_panel(
    n_days: int = 220,
    n_tickers: int = 50,
    seed: int = 7,
    start_date: str = "2018-01-01",
    n_sectors: int = 8,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    n_days = int(n_days)
    n_tickers = int(n_tickers)
    n_sectors = int(n_sectors)

    dates = pd.bdate_range(start_date, periods=n_days)
    tickers = np.array([f"T{idx:04d}" for idx in range(n_tickers)], dtype=object)
    sector_by_ticker = rng.integers(0, n_sectors, size=n_tickers)

    market = np.zeros(n_days, dtype=float)
    market_shock = rng.normal(0.0, 0.006, size=n_days)
    for t in range(1, n_days):
        market[t] = 0.08 * market[t - 1] + market_shock[t]

    sectors = np.zeros((n_days, n_sectors), dtype=float)
    sector_shock = rng.normal(0.0, 0.008, size=(n_days, n_sectors))
    for t in range(1, n_days):
        sectors[t] = 0.06 * sectors[t - 1] + sector_shock[t]

    beta_m = rng.normal(1.0, 0.18, size=n_tickers)
    beta_s = rng.normal(1.0, 0.25, size=n_tickers)
    idio_scale = rng.lognormal(mean=-4.6, sigma=0.25, size=n_tickers)

    returns = np.zeros((n_days, n_tickers), dtype=float)
    idio_noise = rng.normal(0.0, 1.0, size=(n_days, n_tickers))
    for t in range(1, n_days):
        sector_component = sectors[t, sector_by_ticker]
        returns[t] = (
            0.05 * returns[t - 1]
            + beta_m * market[t]
            + beta_s * sector_component
            + idio_noise[t] * idio_scale
        )
    returns = np.clip(returns, -0.20, 0.20)

    init_close = rng.uniform(15.0, 150.0, size=n_tickers)
    close = np.empty_like(returns)
    close[0] = np.maximum(0.5, init_close * (1.0 + returns[0]))
    for t in range(1, n_days):
        close[t] = np.maximum(0.5, close[t - 1] * (1.0 + returns[t]))

    overnight = rng.normal(0.0, 0.002, size=(n_days, n_tickers))
    open_ = np.empty_like(close)
    open_[0] = np.maximum(0.5, close[0] * (1.0 + overnight[0]))
    open_[1:] = np.maximum(0.5, close[:-1] * (1.0 + overnight[1:]))

    intraday_span = np.abs(returns) * 0.60 + np.abs(rng.normal(0.006, 0.003, size=returns.shape))
    intraday_span = np.clip(intraday_span, 0.001, 0.15)
    base_high = np.maximum(open_, close)
    base_low = np.minimum(open_, close)
    high = base_high * (1.0 + intraday_span)
    low = np.maximum(0.01, base_low * (1.0 - 0.9 * intraday_span))

    liquidity_base = rng.lognormal(mean=12.7, sigma=0.45, size=n_tickers)
    volume_noise = rng.lognormal(mean=0.0, sigma=0.28, size=(n_days, n_tickers))
    volume = liquidity_base * (1.0 + 18.0 * np.abs(returns)) * volume_noise
    volume = np.maximum(0.0, np.round(volume))

    panel = pd.DataFrame(
        {
            "datetime": np.repeat(dates.values, n_tickers),
            "instrument": np.tile(tickers, n_days),
            "$open": open_.reshape(-1),
            "$high": high.reshape(-1),
            "$low": low.reshape(-1),
            "$close": close.reshape(-1),
            "$volume": volume.reshape(-1),
        }
    )
    panel = panel.set_index(["datetime", "instrument"]).sort_index()
    panel.index = panel.index.set_names(["datetime", "instrument"])
    return panel
