"""Deterministic factor evaluation utilities for AST-based expressions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .ast.codec import decode_factor_ast
from .ast.interpreter import ASTInterpreter
from .ast.nodes import CallNode, ConstNode, Node, VarNode
from .ast.registry import OperatorRegistry, build_default_registry
from .ast.types import infer_node_type
from .ast.validate import ValidationLimits, normalize_and_validate_ast
from .config import FactorEngineConfig


@dataclass
class FactorEvaluator:
    """Evaluate compiled factor expressions and summarize predictive metrics.

    Attributes:
        periods: Forward-return horizons (business days) used for scoring.
        min_cross_section: Minimum instruments per date for RankIC computation.
        engine_config: Validation limits and AST version expectations.
        registry: Operator registry used by the AST interpreter.
        interpreter: Runtime interpreter initialized from `registry`.

    Invariants:
        - Input panels use MultiIndex `("datetime", "instrument")`.
        - Returned factor series is aligned to the input panel index.
        - `calculate_ex_ante_ir` metrics are computed on either full data or
          the optional scoped universe mask.

    Typical usage:
        >>> evaluator = FactorEvaluator(periods=[1, 5, 10])
        >>> factor = evaluator.calculate_factor(panel, factor_ast)
        >>> fwd = evaluator.calculate_forward_returns(panel)
        >>> metrics = evaluator.calculate_ex_ante_ir(factor, fwd)
    """

    periods: list[int]
    min_cross_section: int = 5
    engine_config: FactorEngineConfig = field(default_factory=FactorEngineConfig)
    registry: OperatorRegistry = field(default_factory=build_default_registry)
    interpreter: ASTInterpreter = field(init=False)

    def __post_init__(self) -> None:
        self.interpreter = ASTInterpreter(self.registry)

    @staticmethod
    def _normalize_name(name: str) -> str:
        return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())

    @classmethod
    def _pick_column(
        cls, frame: pd.DataFrame, aliases: tuple[str, ...], *, required: bool
    ) -> str | None:
        lookup = {cls._normalize_name(col): col for col in frame.columns}
        for alias in aliases:
            key = cls._normalize_name(alias)
            if key in lookup:
                return lookup[key]
        if required:
            raise ValueError(f"Missing required universe column. Tried aliases: {list(aliases)}")
        return None

    @staticmethod
    def _parse_membership(values: pd.Series) -> pd.Series:
        if values.dtype == bool:
            return values.astype(bool)

        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.notna().any():
            return (numeric.fillna(0.0) != 0.0).astype(bool)

        text = values.astype(str).str.strip().str.lower()
        return text.isin({"1", "true", "t", "yes", "y", "in", "member"}).astype(bool)

    @classmethod
    def _build_evaluation_mask(
        cls,
        index: pd.MultiIndex,
        universe_mask: pd.DataFrame | pd.Series | None,
    ) -> pd.Series | None:
        if universe_mask is None:
            return None
        if not isinstance(index, pd.MultiIndex):
            raise ValueError(
                "Universe filtering requires factor index to be MultiIndex(datetime, instrument)"
            )
        if list(index.names) != ["datetime", "instrument"]:
            index = index.set_names(["datetime", "instrument"])

        if isinstance(universe_mask, pd.Series):
            if isinstance(universe_mask.index, pd.MultiIndex):
                series = cls._parse_membership(universe_mask)
                if list(series.index.names) != ["datetime", "instrument"]:
                    series.index = series.index.set_names(["datetime", "instrument"])
                return series.reindex(index).fillna(False).astype(bool)

            parsed = cls._parse_membership(universe_mask)
            active = parsed[parsed].index.astype(str)
            active_tickers = set(pd.Index(active).str.upper().str.replace(".", "-", regex=False))
            flags = (
                index.get_level_values("instrument")
                .astype(str)
                .str.upper()
                .str.replace(".", "-", regex=False)
            )
            return pd.Series(
                np.asarray(flags.isin(active_tickers), dtype=bool),
                index=index,
                name="in_universe",
                dtype=bool,
            )

        if not isinstance(universe_mask, pd.DataFrame):
            raise TypeError("universe_mask must be a pandas DataFrame/Series or None")
        if universe_mask.empty:
            raise ValueError("universe_mask is empty")

        date_col = cls._pick_column(
            universe_mask,
            aliases=("date", "datetime", "timestamp", "trading_date"),
            required=False,
        )
        ticker_col = cls._pick_column(
            universe_mask,
            aliases=("ticker", "instrument", "symbol", "asset", "ric"),
            required=True,
        )
        flag_col = cls._pick_column(
            universe_mask,
            aliases=("in_universe", "is_member", "in_index", "member", "active", "weight"),
            required=False,
        )

        normalized = pd.DataFrame(index=universe_mask.index)
        normalized["instrument"] = (
            universe_mask[ticker_col]
            .astype(str)
            .str.upper()
            .str.replace(".", "-", regex=False)
            .str.strip()
        )
        if flag_col is None:
            normalized["in_universe"] = True
        else:
            parsed_flag = cls._parse_membership(universe_mask[flag_col])
            if cls._normalize_name(flag_col) == "weight":
                parsed_flag = (
                    pd.to_numeric(universe_mask[flag_col], errors="coerce").fillna(0.0) > 0.0
                ).astype(bool)
            normalized["in_universe"] = parsed_flag.astype(bool)

        normalized = normalized[normalized["instrument"] != ""].copy()
        if date_col is not None:
            dates = pd.to_datetime(universe_mask[date_col], errors="coerce")
            if getattr(dates.dt, "tz", None) is not None:
                dates = dates.dt.tz_convert(None)
            normalized["datetime"] = dates.dt.normalize()
        else:
            normalized["datetime"] = pd.NaT

        if normalized["datetime"].notna().sum() == 0:
            active_tickers = set(normalized.loc[normalized["in_universe"], "instrument"])
            flags = (
                index.get_level_values("instrument")
                .astype(str)
                .str.upper()
                .str.replace(".", "-", regex=False)
            )
            return pd.Series(
                np.asarray(flags.isin(active_tickers), dtype=bool),
                index=index,
                name="in_universe",
                dtype=bool,
            )

        normalized = normalized.dropna(subset=["datetime"])
        if normalized.empty:
            raise ValueError("universe_mask has no valid datetime rows after parsing")

        snapshots = (
            normalized.assign(in_universe=normalized["in_universe"].astype(int))
            .groupby(["datetime", "instrument"], sort=True, as_index=False)["in_universe"]
            .max()
        )

        panel_dates = pd.Index(
            index.get_level_values("datetime").unique(), name="datetime"
        ).sort_values()
        panel_tickers = (
            pd.Index(index.get_level_values("instrument").astype(str).unique(), name="instrument")
            .str.upper()
            .str.replace(".", "-", regex=False)
            .sort_values()
        )

        matrix = snapshots.pivot(
            index="datetime", columns="instrument", values="in_universe"
        ).sort_index()
        matrix = matrix.reindex(columns=panel_tickers, fill_value=0.0)
        matrix = matrix.reindex(panel_dates).ffill().fillna(0.0)

        dense = matrix.stack(future_stack=True).astype(bool)
        dense.index = dense.index.set_names(["datetime", "instrument"])
        return dense.reindex(index, fill_value=False).astype(bool)

    def _validate_panel(self, panel: pd.DataFrame) -> None:
        if not isinstance(panel, pd.DataFrame):
            raise TypeError("panel must be a pandas DataFrame")
        if not isinstance(panel.index, pd.MultiIndex):
            raise ValueError("panel must use MultiIndex(datetime, instrument)")
        if list(panel.index.names) != ["datetime", "instrument"]:
            panel.index = panel.index.set_names(["datetime", "instrument"])

        required = {"$open", "$high", "$low", "$close", "$volume"}
        missing = required - set(panel.columns)
        if missing:
            raise ValueError(f"panel is missing required columns: {sorted(missing)}")

    @staticmethod
    def _normalize_result(result: Any, panel: pd.DataFrame) -> pd.Series:
        if isinstance(result, pd.DataFrame):
            if result.shape[1] != 1:
                raise ValueError(
                    "Factor expression returned multi-column DataFrame; expected one series"
                )
            series = result.iloc[:, 0]
        elif isinstance(result, pd.Series):
            series = result
        elif np.isscalar(result):
            series = pd.Series(float(result), index=panel.index)
        elif isinstance(result, np.ndarray):
            if len(result) != len(panel.index):
                raise ValueError("Factor expression returned ndarray with unexpected length")
            series = pd.Series(result, index=panel.index)
        else:
            raise TypeError(f"Unsupported factor result type: {type(result)!r}")

        series = series.reindex(panel.index)
        series.name = "factor"
        return series.astype(float)

    @staticmethod
    def _panel_context(panel: pd.DataFrame) -> dict[str, Any]:
        context: dict[str, Any] = {"np": np}
        for col in panel.columns:
            plain = col.replace("$", "")
            context[col] = panel[col]
            context[plain] = panel[col]
            context[plain.upper()] = panel[col]
            context[plain.lower()] = panel[col]
            context[plain.capitalize()] = panel[col]
        return context

    def _calculate_factor_ast(
        self, panel: pd.DataFrame, expression: dict[str, Any] | Node
    ) -> pd.Series:
        version, node = decode_factor_ast(expression)
        expected_version = str(self.engine_config.ast_version)
        if expected_version and version != expected_version:
            raise ValueError(f"Unsupported AST version: {version!r}, expected {expected_version!r}")

        limits = ValidationLimits(
            max_depth=int(self.engine_config.max_ast_depth),
            max_nodes=int(self.engine_config.max_ast_nodes),
            allowed_columns=set(panel.columns),
            allowed_windows=set(self.engine_config.allowed_windows)
            if self.engine_config.allowed_windows
            else None,
        )
        normalized = normalize_and_validate_ast(node, self.registry, limits=limits)
        infer_node_type(normalized, self.registry)

        context = self._panel_context(panel)
        try:
            result = self.interpreter.evaluate(normalized, context)
        except Exception as exc:
            raise RuntimeError(f"AST factor evaluation failed: {exc}") from exc

        return self._normalize_result(result, panel)

    def calculate_factor(self, panel: pd.DataFrame, expression: dict[str, Any] | Node) -> pd.Series:
        """Evaluate one AST expression and return a numeric factor series."""

        self._validate_panel(panel)
        if isinstance(expression, (dict, VarNode, ConstNode, CallNode)):
            return self._calculate_factor_ast(panel, expression)
        raise TypeError(
            "Only AST payloads are supported. Expected dict envelope/node or Node instance, "
            f"got {type(expression)!r}"
        )

    def calculate_forward_returns(
        self, panel: pd.DataFrame, periods: list[int] | None = None
    ) -> pd.DataFrame:
        """Compute per-instrument forward returns for configured horizons."""

        self._validate_panel(panel)
        use_periods = periods or self.periods

        close = panel["$close"].astype(float)
        out: dict[str, pd.Series] = {}
        grouped = close.groupby(level="instrument", sort=False)

        for p in use_periods:
            p = int(p)
            if p <= 0:
                raise ValueError(f"Forward return period must be > 0, got {p}")
            out[f"ret_{p}"] = grouped.pct_change(periods=p).shift(-p)

        return pd.DataFrame(out, index=panel.index)

    def _daily_rank_ic(self, factor: pd.Series, returns: pd.Series) -> pd.Series:
        joined = pd.concat([factor.rename("factor"), returns.rename("ret")], axis=1).dropna()
        if joined.empty:
            return pd.Series(dtype=float)

        rank_ics: list[tuple[pd.Timestamp, float]] = []
        for dt, group in joined.groupby(level="datetime", sort=False):
            if len(group) < self.min_cross_section:
                continue
            fac_rank = group["factor"].rank(method="average")
            ret_rank = group["ret"].rank(method="average")
            if fac_rank.nunique(dropna=True) < 2 or ret_rank.nunique(dropna=True) < 2:
                continue
            ic = fac_rank.corr(ret_rank, method="pearson")
            if ic is not None and np.isfinite(ic):
                rank_ics.append((dt, float(ic)))

        if not rank_ics:
            return pd.Series(dtype=float)

        index = pd.Index([d for d, _ in rank_ics], name="datetime")
        values = [v for _, v in rank_ics]
        return pd.Series(values, index=index, name="rank_ic")

    @staticmethod
    def _ir_from_rank_ic(rank_ic: pd.Series) -> tuple[float, float, float, int]:
        if rank_ic.empty:
            return 0.0, 0.0, 0.0, 0
        mean_ic = float(rank_ic.mean())
        std_ic = float(rank_ic.std(ddof=1))
        if not np.isfinite(std_ic) or std_ic <= 1e-12:
            return 0.0, mean_ic, std_ic, int(rank_ic.shape[0])
        ir = mean_ic / std_ic
        return float(ir), mean_ic, std_ic, int(rank_ic.shape[0])

    def calculate_ex_ante_ir(
        self,
        factor: pd.Series,
        forward_returns: pd.DataFrame,
        universe_mask: pd.DataFrame | pd.Series | None = None,
    ) -> dict[str, Any]:
        """Calculate RankIC and RankIC-IR aggregates over forward-return periods.

        Args:
            factor: Factor values aligned by `(datetime, instrument)`.
            forward_returns: Data frame of forward return columns.
            universe_mask: Optional membership mask used to scope evaluation.

        Returns:
            Dictionary with overall metrics, per-period metrics, and scope stats.
        """

        if not isinstance(factor, pd.Series):
            raise TypeError("factor must be a pandas Series")
        if not isinstance(forward_returns, pd.DataFrame):
            raise TypeError("forward_returns must be a pandas DataFrame")
        if not factor.index.equals(forward_returns.index):
            forward_returns = forward_returns.reindex(factor.index)

        mask = self._build_evaluation_mask(factor.index, universe_mask)
        if mask is None:
            scoped_factor = factor
            scoped_returns = forward_returns
        else:
            scoped_factor = factor.where(mask)
            scoped_returns = forward_returns.where(mask, np.nan, axis=0)

        period_metrics: dict[str, dict[str, Any]] = {}
        period_rank_ics: list[float] = []
        period_irs: list[float] = []

        for col in scoped_returns.columns:
            rank_ic = self._daily_rank_ic(scoped_factor, scoped_returns[col])
            ir, mean_ic, std_ic, n_days = self._ir_from_rank_ic(rank_ic)
            period_metrics[col] = {
                "rank_ic": mean_ic,
                "rank_ic_std": std_ic,
                "rank_ic_ir": ir,
                "n_days": n_days,
            }
            if n_days > 0:
                period_rank_ics.append(mean_ic)
                period_irs.append(ir)

        overall_rank_ic = float(np.mean(period_rank_ics)) if period_rank_ics else 0.0
        overall_ir = float(np.mean(period_irs)) if period_irs else 0.0
        total_rows = int(len(factor))
        scoped_rows = int(mask.sum()) if mask is not None else total_rows
        total_tickers = int(factor.index.get_level_values("instrument").nunique())
        scoped_tickers = total_tickers
        if mask is not None:
            scoped_index = factor.index[mask.to_numpy()]
            scoped_tickers = (
                int(scoped_index.get_level_values("instrument").nunique())
                if len(scoped_index)
                else 0
            )

        return {
            "rank_ic": overall_rank_ic,
            "rank_ic_ir": overall_ir,
            "period_metrics": period_metrics,
            "primary_metric": "rank_ic",
            "evaluation_scope": {
                "universe_filter_applied": mask is not None,
                "rows_total": total_rows,
                "rows_in_scope": scoped_rows,
                "n_tickers_total": total_tickers,
                "n_tickers_in_scope": scoped_tickers,
            },
        }
