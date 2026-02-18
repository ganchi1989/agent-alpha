"""Data loaders and market-data builders used by agent-alpha."""

from .market_data import SP500UniverseBuilder, YahooPanelBuilder
from .panel import SyntheticSpec, generate_synthetic_panel, load_panel, validate_panel_df
from .universe import UniverseSpec, load_universe, validate_universe


def load_synthetic_panel(
    n_days: int = 220,
    n_tickers: int = 50,
    seed: int = 7,
    start_date: str = "2018-01-01",
    n_sectors: int = 8,
):
    """Build a synthetic panel in workflow-compatible index/column format."""
    spec = SyntheticSpec(
        seed=int(seed),
        start_date=str(start_date),
        n_days=int(n_days),
        n_tickers=int(n_tickers),
        n_sectors=int(n_sectors),
    )
    panel_df = generate_synthetic_panel(spec)
    return (
        panel_df.rename(
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
        .set_index(["datetime", "instrument"])
        .sort_index()
    )


__all__ = [
    "SP500UniverseBuilder",
    "YahooPanelBuilder",
    "SyntheticSpec",
    "load_synthetic_panel",
    "load_panel",
    "validate_panel_df",
    "UniverseSpec",
    "load_universe",
    "validate_universe",
]
