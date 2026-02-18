"""Universe mask construction, validation, and loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

UNIVERSE_COLUMNS = ["date", "ticker", "in_universe"]


@dataclass(frozen=True)
class UniverseSpec:
    """Configuration for synthetic universe construction.

    Attributes:
        size: Target number of instruments in each daily universe snapshot.
        seed: Seed used for tie-breaking noise in liquidity ranking.
        approx_tolerance: Allowed relative tolerance for expected universe size.
    """

    size: int
    seed: int
    approx_tolerance: float = 0.15

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "UniverseSpec":
        """Create `UniverseSpec` from a repository configuration dictionary."""

        synthetic_cfg = cfg.get("synthetic", {})
        return cls(
            size=int(synthetic_cfg.get("universe_size", 300)),
            seed=int(synthetic_cfg.get("seed", 7)),
            approx_tolerance=float(cfg.get("universe_tolerance", 0.15)),
        )


def _load_mask_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported universe file extension: {path.suffix}")


def build_dynamic_universe(panel_df: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    """Build a liquidity-ranked dynamic universe from panel data."""

    if size <= 0:
        raise ValueError("Universe size must be > 0")

    panel = panel_df[["date", "ticker", "close", "volume"]].copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel["liquidity"] = panel["close"] * panel["volume"]

    liquidity = panel.pivot(index="date", columns="ticker", values="liquidity").sort_index()
    score = liquidity.rolling(window=20, min_periods=1).mean()
    score_values = score.to_numpy(dtype=float, copy=True)

    rng = np.random.default_rng(seed + 101)
    row_scale = np.nanstd(score_values, axis=1)
    row_scale = np.where(np.isfinite(row_scale) & (row_scale > 0), row_scale, 1.0)
    noise = rng.normal(0.0, 1.0, size=score_values.shape) * row_scale[:, None] * 0.01
    noisy_score = score_values + noise

    tickers = score.columns.to_numpy(dtype=object)
    records: list[tuple[pd.Timestamp, str, int]] = []
    for idx, date in enumerate(score.index):
        row = noisy_score[idx]
        valid_idx = np.where(np.isfinite(row))[0]
        if len(valid_idx) == 0:
            continue
        k = min(size, len(valid_idx))
        top_local = valid_idx[np.argpartition(row[valid_idx], -k)[-k:]]
        top_sorted = top_local[np.argsort(row[top_local])[::-1]]
        records.extend((date, str(tickers[col]), 1) for col in top_sorted)

    mask_df = pd.DataFrame(records, columns=UNIVERSE_COLUMNS)
    return mask_df.sort_values(["date", "ticker"]).reset_index(drop=True)


def validate_universe(
    mask_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    expected_size: int | None = None,
    strict_tickers: bool = False,
) -> pd.DataFrame:
    """Validate and normalize a universe membership data frame.

    Args:
        mask_df: Input universe rows.
        panel_df: Panel data used to validate ticker availability.
        expected_size: Optional minimum expected daily member count.
        strict_tickers: Whether unknown tickers should raise instead of drop.
    """

    if {"date", "ticker"} - set(mask_df.columns):
        raise ValueError("Universe mask must contain date and ticker columns")

    mask = mask_df.copy()
    mask["date"] = pd.to_datetime(mask["date"])
    mask["ticker"] = mask["ticker"].astype(str)
    if "in_universe" not in mask.columns:
        mask["in_universe"] = 1
    mask = mask[mask["in_universe"] == 1][UNIVERSE_COLUMNS].copy()

    key_dupes = mask.duplicated(subset=["date", "ticker"]).sum()
    if key_dupes:
        raise ValueError(f"Universe mask has {key_dupes} duplicate (date,ticker) rows")

    panel_tickers = set(panel_df["ticker"].astype(str).unique())
    unknown_tickers = sorted(set(mask["ticker"].unique()) - panel_tickers)
    if unknown_tickers:
        if strict_tickers:
            raise ValueError(
                f"Universe has tickers not present in panel. Example: {unknown_tickers[:5]}"
            )
        mask = mask[mask["ticker"].isin(panel_tickers)].copy()
        if mask.empty:
            raise ValueError(
                "Universe is empty after dropping tickers not present in panel. "
                f"Example unknown tickers: {unknown_tickers[:5]}"
            )

    if expected_size is not None and expected_size > 0:
        counts = mask.groupby("date")["ticker"].nunique()
        min_allowed = int(np.floor(expected_size * 0.85))
        if (counts < min_allowed).any():
            bad_date = counts[counts < min_allowed].index[0]
            bad_count = int(counts.loc[bad_date])
            raise ValueError(
                f"Universe size too small on {bad_date.date()}: {bad_count}, expected around {expected_size}"
            )

    return mask.sort_values(["date", "ticker"]).reset_index(drop=True)


def load_universe(panel_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Load a universe mask from file or generate one from panel liquidity."""

    data_cfg = config.get("data", {})
    spec = UniverseSpec.from_config(data_cfg)
    universe_path = data_cfg.get("universe_path")
    enforce_universe_size = bool(data_cfg.get("enforce_universe_size", False))
    strict_tickers = bool(data_cfg.get("strict_universe_tickers", False))
    max_available = int(panel_df["ticker"].astype(str).nunique())
    effective_expected_size = min(spec.size, max_available) if max_available > 0 else spec.size

    if universe_path:
        mask_df = _load_mask_file(Path(universe_path))
        expected_size = effective_expected_size if enforce_universe_size else None
    else:
        mask_df = build_dynamic_universe(panel_df, size=spec.size, seed=spec.seed)
        expected_size = effective_expected_size

    return validate_universe(
        mask_df,
        panel_df,
        expected_size=expected_size,
        strict_tickers=strict_tickers,
    )
