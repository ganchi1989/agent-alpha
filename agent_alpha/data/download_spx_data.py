from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .market_data import SP500UniverseBuilder, YahooPanelBuilder


def build_spx_data(
    *,
    start_date: str,
    end_date: str,
    output_dir: Path,
    batch_size: int = 100,
    pause_seconds: float = 0.2,
    auto_adjust: bool = True,
) -> dict[str, str | int]:
    output_dir.mkdir(parents=True, exist_ok=True)

    universe_builder = SP500UniverseBuilder(start_date=start_date, end_date=end_date)
    universe_daily = universe_builder.build_daily()

    universe_path = output_dir / "spx_universe.csv"
    universe_builder.save(universe_daily, universe_path)

    panel_builder = YahooPanelBuilder(
        batch_size=int(batch_size),
        pause_seconds=float(pause_seconds),
        auto_adjust=bool(auto_adjust),
    )
    prices_df, universe_filtered = panel_builder.download(
        universe_daily,
        start_date=start_date,
        end_date=end_date,
    )

    prices_path = output_dir / "spx_prices.csv"
    filtered_universe_path = output_dir / "spx_universe_filtered.csv"
    panel_builder.save_panel(prices_df, prices_path)
    panel_builder.save_universe(universe_filtered, filtered_universe_path)

    keys = universe_filtered[["date", "ticker"]].drop_duplicates()
    merged = prices_df.merge(keys, on=["date", "ticker"], how="inner")
    agent_panel = merged.rename(
        columns={
            "date": "datetime",
            "ticker": "instrument",
            "open": "$open",
            "high": "$high",
            "low": "$low",
            "close": "$close",
            "volume": "$volume",
        }
    )
    agent_panel = agent_panel[
        ["datetime", "instrument", "$open", "$high", "$low", "$close", "$volume"]
    ]
    agent_panel = agent_panel.sort_values(["datetime", "instrument"]).drop_duplicates(
        ["datetime", "instrument"]
    )

    panel_path = output_dir / "spx_agent_panel.csv"
    agent_panel.to_csv(panel_path, index=False)

    return {
        "universe_path": str(universe_path),
        "prices_path": str(prices_path),
        "filtered_universe_path": str(filtered_universe_path),
        "agent_panel_path": str(panel_path),
        "universe_rows": int(len(universe_daily)),
        "prices_rows": int(len(prices_df)),
        "panel_rows": int(len(agent_panel)),
        "n_universe_tickers": int(universe_daily["ticker"].nunique()),
        "n_price_tickers": int(prices_df["ticker"].nunique()),
        "n_panel_tickers": int(agent_panel["instrument"].nunique()),
        "n_panel_dates": int(pd.to_datetime(agent_panel["datetime"]).nunique()),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and prepare SPX data for agent-alpha workflow."
    )
    parser.add_argument(
        "--start-date", default="2018-01-01", help="Inclusive start date (YYYY-MM-DD)."
    )
    parser.add_argument("--end-date", default="2026-02-18", help="Inclusive end date (YYYY-MM-DD).")
    parser.add_argument(
        "--output-dir",
        default="agent_alpha/data",
        help="Directory for universe/prices/panel output files.",
    )
    parser.add_argument("--batch-size", type=int, default=100, help="Yahoo download batch size.")
    parser.add_argument(
        "--pause-seconds", type=float, default=0.2, help="Pause between Yahoo batches."
    )
    parser.add_argument("--no-auto-adjust", action="store_true", help="Disable Yahoo auto-adjust.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    summary = build_spx_data(
        start_date=str(args.start_date),
        end_date=str(args.end_date),
        output_dir=Path(args.output_dir),
        batch_size=int(args.batch_size),
        pause_seconds=float(args.pause_seconds),
        auto_adjust=not bool(args.no_auto_adjust),
    )

    print("SPX data build complete.")
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
