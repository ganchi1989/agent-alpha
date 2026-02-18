"""Download and normalize S&P 500 universe and Yahoo OHLCV panel data."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from bs4.element import Tag

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
UNIVERSE_COLUMNS = ["date", "ticker", "in_universe"]
PANEL_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]


def _clean_text(text: str) -> str:
    text = re.sub(r"\[[^\]]*\]", "", str(text))
    text = text.replace("\xa0", " ")
    return " ".join(text.split()).strip()


def _normalize_ticker(ticker: str) -> str:
    return _clean_text(ticker).upper().replace(".", "-")


def _parse_current_constituents(table: Tag) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 2:
            continue
        ticker = _normalize_ticker(cells[0].get_text(" ", strip=True))
        name = _clean_text(cells[1].get_text(" ", strip=True))
        if ticker:
            rows.append({"ticker": ticker, "name": name})
    if not rows:
        raise ValueError("Failed to parse current S&P 500 constituents from Wikipedia")
    return (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["ticker"])
        .sort_values("ticker")
        .reset_index(drop=True)
    )


def _parse_changes(table: Tag) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    last_date = pd.NaT
    for tr in table.find_all("tr"):
        cells = [_clean_text(td.get_text(" ", strip=True)) for td in tr.find_all("td")]
        if not cells:
            continue

        maybe_date = pd.to_datetime(cells[0], errors="coerce")
        if pd.notna(maybe_date):
            row_date = pd.Timestamp(maybe_date).normalize()
            last_date = row_date
            payload = cells[1:]
        else:
            if pd.isna(last_date):
                continue
            row_date = pd.Timestamp(last_date)
            payload = cells

        payload = (payload + ["", "", "", "", ""])[:5]
        add_ticker, add_name, removed_ticker, removed_name, reason = payload
        rows.append(
            {
                "date": row_date,
                "add_ticker": _normalize_ticker(add_ticker) if add_ticker else "",
                "add_name": add_name,
                "removed_ticker": _normalize_ticker(removed_ticker) if removed_ticker else "",
                "removed_name": removed_name,
                "reason": reason,
            }
        )
    if not rows:
        raise ValueError("Failed to parse S&P 500 changes table from Wikipedia")
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def download_wikipedia_tables(
    url: str = WIKI_URL, timeout_seconds: float = 30.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch S&P 500 constituents and membership changes from Wikipedia.

    Args:
        url: Wikipedia page URL containing constituents and changes tables.
        timeout_seconds: HTTP request timeout in seconds.

    Returns:
        A tuple of two data frames: `(current_constituents, changes)`.

    Raises:
        ImportError: If `requests` or `beautifulsoup4` is not installed.
        ValueError: If expected HTML tables cannot be parsed.
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Wikipedia universe fetch requires requests and beautifulsoup4. "
            "Install with: pip install requests beautifulsoup4"
        ) from exc

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=timeout_seconds)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    current_table = soup.find("table", {"id": "constituents"})
    changes_table = soup.find("table", {"id": "changes"})
    if current_table is None or changes_table is None:
        tables = soup.select("table.wikitable")
        if len(tables) < 2:
            raise ValueError("Unable to locate constituents/changes tables on Wikipedia page")
        current_table = current_table or tables[0]
        changes_table = changes_table or tables[1]

    current = _parse_current_constituents(current_table)
    changes = _parse_changes(changes_table)
    return current, changes


def build_daily_universe(
    current_constituents: pd.DataFrame,
    changes_df: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build a business-day universe membership panel by replaying index changes.

    The algorithm starts from the current membership set and walks backward
    through the changes table to reconstruct membership on each date.

    Args:
        current_constituents: Data frame with at least a `ticker` column.
        changes_df: Data frame with `date`, `add_ticker`, `removed_ticker`.
        start_date: Inclusive lower bound for the output date range.
        end_date: Inclusive upper bound for the output date range. Defaults to today.

    Returns:
        Data frame with `date`, `ticker`, and `in_universe` columns.
    """
    start = pd.Timestamp(start_date).normalize()
    end = (
        pd.Timestamp(end_date).normalize()
        if end_date is not None
        else pd.Timestamp.today().normalize()
    )
    if end < start:
        raise ValueError(f"end_date {end.date()} is earlier than start_date {start.date()}")
    dates = pd.bdate_range(start=start, end=end)
    if len(dates) == 0:
        raise ValueError("No business days in requested range")

    state: set[str] = set(current_constituents["ticker"].astype(str))
    changes = changes_df.copy()
    changes["date"] = pd.to_datetime(changes["date"]).dt.normalize()
    changes = changes.sort_values("date", ascending=False).reset_index(drop=True)

    ptr = 0
    records: list[tuple[pd.Timestamp, str, int]] = []
    for date in reversed(dates):
        while ptr < len(changes) and changes.loc[ptr, "date"] > date:
            add_ticker = str(changes.loc[ptr, "add_ticker"] or "").strip()
            remove_ticker = str(changes.loc[ptr, "removed_ticker"] or "").strip()
            if add_ticker:
                state.discard(add_ticker)
            if remove_ticker:
                state.add(remove_ticker)
            ptr += 1
        records.extend((date, ticker, 1) for ticker in sorted(state))

    return (
        pd.DataFrame(records, columns=UNIVERSE_COLUMNS)
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )


def to_monthly_universe(daily_universe: pd.DataFrame) -> pd.DataFrame:
    """Downsample daily universe membership to one snapshot per month.

    The monthly snapshot is taken from each calendar month's first observed
    business day in the input data.
    """
    universe = daily_universe.copy()
    universe["month"] = universe["date"].dt.to_period("M")
    first_dates = universe.groupby("month")["date"].min().rename("month_first_date")
    monthly = universe.merge(first_dates, left_on="month", right_index=True, how="inner")
    monthly = monthly[monthly["date"] == monthly["month_first_date"]][UNIVERSE_COLUMNS]
    return monthly.sort_values(["date", "ticker"]).reset_index(drop=True)


def _chunked(values: list[str], batch_size: int) -> list[list[str]]:
    return [values[i : i + batch_size] for i in range(0, len(values), batch_size)]


def _normalize_yahoo_batch(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=PANEL_COLUMNS)

    frames: list[pd.DataFrame] = []
    required_fields = {"Open", "High", "Low", "Close", "Volume"}
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = set(raw.columns.get_level_values(0))
        level1 = set(raw.columns.get_level_values(1))
        ticker_first = any(t in level0 for t in tickers) and bool(required_fields & level1)
        field_first = bool(required_fields & level0) and any(t in level1 for t in tickers)

        if ticker_first:
            for ticker in tickers:
                if ticker in level0:
                    part = raw[ticker].copy().reset_index()
                    part["ticker"] = ticker
                    frames.append(part)
        elif field_first:
            for ticker in tickers:
                cols = [field for field in required_fields if (field, ticker) in raw.columns]
                if not cols:
                    continue
                part = pd.DataFrame({field: raw[(field, ticker)] for field in cols}).reset_index()
                part["ticker"] = ticker
                frames.append(part)
        else:
            raise ValueError("Unexpected yfinance column format")
    else:
        part = raw.copy().reset_index()
        part["ticker"] = tickers[0] if tickers else ""
        frames.append(part)

    if not frames:
        return pd.DataFrame(columns=PANEL_COLUMNS)

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.rename(
        columns={
            "Date": "date",
            "Datetime": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    missing = sorted(set(PANEL_COLUMNS) - set(panel.columns))
    if missing:
        raise ValueError(f"Downloaded panel missing columns: {missing}")

    panel = panel[PANEL_COLUMNS].copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel["ticker"] = panel["ticker"].astype(str).str.upper().str.replace(".", "-", regex=False)
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel["volume"] = pd.to_numeric(panel["volume"], errors="coerce").fillna(0.0)
    panel = panel.dropna(subset=["open", "high", "low", "close"])
    panel = panel[panel["volume"] >= 0]
    return (
        panel.sort_values(["date", "ticker"])
        .drop_duplicates(["date", "ticker"])
        .reset_index(drop=True)
    )


def download_yahoo_panel(
    tickers: list[str],
    start_date: str,
    end_date: str,
    batch_size: int = 100,
    auto_adjust: bool = True,
    pause_seconds: float = 0.2,
) -> pd.DataFrame:
    """Download and normalize daily OHLCV data from Yahoo Finance.

    Args:
        tickers: Symbols to download.
        start_date: Inclusive download start date (`YYYY-MM-DD`).
        end_date: Exclusive download end date (`YYYY-MM-DD`).
        batch_size: Number of tickers per request batch.
        auto_adjust: Whether to request adjusted prices from Yahoo.
        pause_seconds: Sleep interval between batches to reduce throttling risk.

    Returns:
        Normalized panel with columns in `PANEL_COLUMNS`.

    Raises:
        ImportError: If `yfinance` is not installed.
        ValueError: For invalid inputs or malformed downloaded data.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Yahoo download requires yfinance. Install with: pip install yfinance"
        ) from exc

    if not tickers:
        raise ValueError("No tickers provided for Yahoo download")

    frames: list[pd.DataFrame] = []
    for i, batch in enumerate(_chunked(tickers, batch_size), start=1):
        raw = yf.download(
            tickers=batch,
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=auto_adjust,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        part = _normalize_yahoo_batch(raw, batch)
        if not part.empty:
            frames.append(part)
        print(f"Batch {i}: tickers={len(batch)} rows={len(part):,}")
        if pause_seconds > 0:
            time.sleep(pause_seconds)

    if not frames:
        raise ValueError("No price data downloaded from Yahoo")

    panel = pd.concat(frames, ignore_index=True)
    panel = (
        panel.sort_values(["date", "ticker"])
        .drop_duplicates(["date", "ticker"])
        .reset_index(drop=True)
    )
    if (panel["high"] < panel["low"]).any():
        raise ValueError("Detected rows where high < low")
    if (panel["volume"] < 0).any():
        raise ValueError("Detected rows where volume < 0")
    return panel


def _save_frame(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        try:
            df.to_parquet(output_path, index=False)
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Parquet output requires pyarrow/fastparquet. Install one or use .csv output."
            ) from exc
    elif suffix in {".csv", ".txt"}:
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output extension: {output_path.suffix}")


@dataclass
class SP500UniverseBuilder:
    """Stateful builder for historical S&P 500 membership data.

    Attributes:
        start_date: Inclusive lower bound for generated membership snapshots.
        end_date: Inclusive upper bound. When `None`, uses current date.
        timeout_seconds: Timeout used for Wikipedia fetch requests.
        current_constituents: Cached constituents table from `fetch`.
        changes_df: Cached index change log from `fetch`.

    Invariants:
        - `current_constituents` and `changes_df` are either both populated
          after a successful fetch or both `None`.

    Example:
        >>> builder = SP500UniverseBuilder(start_date="2018-01-01")
        >>> monthly = builder.build_monthly()
        >>> SP500UniverseBuilder.save(monthly, "spx_universe.parquet")
    """

    start_date: str = "1990-01-01"
    end_date: str | None = None
    timeout_seconds: float = 30.0

    current_constituents: pd.DataFrame | None = None
    changes_df: pd.DataFrame | None = None

    def fetch(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch and cache the current constituents and historical changes."""
        current, changes = download_wikipedia_tables(timeout_seconds=self.timeout_seconds)
        self.current_constituents = current
        self.changes_df = changes
        return current, changes

    def _ensure_fetched(self) -> None:
        if self.current_constituents is None or self.changes_df is None:
            self.fetch()

    def build_daily(self) -> pd.DataFrame:
        """Build a daily universe panel from cached or freshly fetched tables."""
        self._ensure_fetched()
        return build_daily_universe(
            current_constituents=self.current_constituents,  # type: ignore[arg-type]
            changes_df=self.changes_df,  # type: ignore[arg-type]
            start_date=self.start_date,
            end_date=self.end_date,
        )

    def build_monthly(self) -> pd.DataFrame:
        """Build a first-business-day-per-month universe panel."""
        return to_monthly_universe(self.build_daily())

    @staticmethod
    def save(df: pd.DataFrame, output_path: str | Path) -> None:
        """Persist a universe data frame to CSV/TXT/Parquet."""
        _save_frame(df, Path(output_path))


@dataclass
class YahooPanelBuilder:
    """Builder that downloads OHLCV panel data for a given universe snapshot table.

    Attributes:
        batch_size: Number of tickers downloaded per Yahoo request.
        pause_seconds: Delay between batches to reduce request throttling.
        auto_adjust: Whether Yahoo returns adjusted prices.

    Invariants:
        - Output panel rows are unique by `(date, ticker)`.
        - Universe output contains only tickers and dates present in the panel.

    Example:
        >>> builder = YahooPanelBuilder(batch_size=80)
        >>> panel_df, universe_df = builder.download(universe_df)
        >>> builder.save_panel(panel_df, "spx_panel.parquet")
    """

    batch_size: int = 100
    pause_seconds: float = 0.2
    auto_adjust: bool = True

    def download(
        self,
        universe_df: pd.DataFrame,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Download panel data and align the universe to available market data."""
        required = {"date", "ticker"}
        missing = required - set(universe_df.columns)
        if missing:
            raise ValueError(f"universe_df missing required columns: {sorted(missing)}")

        u = universe_df.copy()
        u["date"] = pd.to_datetime(u["date"]).dt.normalize()
        u["ticker"] = u["ticker"].astype(str).str.upper().str.replace(".", "-", regex=False)
        if "in_universe" not in u.columns:
            u["in_universe"] = 1
        u = u[u["in_universe"] == 1][["date", "ticker", "in_universe"]].drop_duplicates()

        start = pd.Timestamp(start_date).normalize() if start_date else u["date"].min()
        end = (
            pd.Timestamp(end_date).normalize()
            if end_date
            else (u["date"].max() + pd.Timedelta(days=1))
        )
        if end <= start:
            raise ValueError("end_date must be later than start_date")

        tickers = sorted(u["ticker"].unique().tolist())
        panel = download_yahoo_panel(
            tickers=tickers,
            start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
            batch_size=self.batch_size,
            auto_adjust=self.auto_adjust,
            pause_seconds=self.pause_seconds,
        )

        available_tickers = set(panel["ticker"].unique())
        filtered_universe = u[u["ticker"].isin(available_tickers)].copy()
        panel_dates = set(panel["date"].unique())
        filtered_universe = filtered_universe[filtered_universe["date"].isin(panel_dates)].copy()

        panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
        filtered_universe = filtered_universe.sort_values(["date", "ticker"]).reset_index(drop=True)
        return panel, filtered_universe

    @staticmethod
    def save_panel(panel_df: pd.DataFrame, output_path: str | Path) -> None:
        """Persist a normalized panel data frame to disk."""
        _save_frame(panel_df, Path(output_path))

    @staticmethod
    def save_universe(universe_df: pd.DataFrame, output_path: str | Path) -> None:
        """Persist a normalized universe data frame to disk."""
        _save_frame(universe_df, Path(output_path))
