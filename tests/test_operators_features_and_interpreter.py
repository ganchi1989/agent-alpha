"""Coverage tests for operators, feature indicators, and the AST interpreter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _panel(dates: int = 5, tickers: int = 4, seed: int = 0) -> pd.Series:
    """Synthetic MultiIndex(datetime, instrument) Series, values in [1, 10]."""
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=dates), [f"T{i}" for i in range(tickers)]],
        names=["datetime", "instrument"],
    )
    return pd.Series(rng.uniform(1.0, 10.0, len(idx)), index=idx, name="x")


def _pos_panel(dates: int = 10, tickers: int = 3, seed: int = 1) -> pd.Series:
    """Strictly positive panel series."""
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=dates), [f"T{i}" for i in range(tickers)]],
        names=["datetime", "instrument"],
    )
    return pd.Series(rng.uniform(0.1, 5.0, len(idx)), index=idx, name="x")


def _ohlcv(n: int = 40, seed: int = 42) -> pd.DataFrame:
    """Single-instrument OHLCV DataFrame with plain column names."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.uniform(1e6, 5e6, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


# ---------------------------------------------------------------------------
# Arithmetic operators
# ---------------------------------------------------------------------------


class TestArithmetic:
    def test_add_two_series(self):
        from agent_alpha.operators.function_lib import ADD

        s = _panel()
        pd.testing.assert_series_equal(ADD(s, s), s + s)

    def test_add_series_scalar(self):
        from agent_alpha.operators.function_lib import ADD

        s = _panel()
        assert (ADD(s, 1.0) == s + 1.0).all()

    def test_subtract_two(self):
        from agent_alpha.operators.function_lib import SUBTRACT

        s = _panel()
        assert (SUBTRACT(s, s) == 0).all()

    def test_subtract_single_arg_negates(self):
        from agent_alpha.operators.function_lib import NEG, SUBTRACT

        s = _panel()
        pd.testing.assert_series_equal(SUBTRACT(s), NEG(s))

    def test_multiply(self):
        from agent_alpha.operators.function_lib import MULTIPLY

        s = _panel()
        pd.testing.assert_series_equal(MULTIPLY(s, 2.0), s * 2.0)

    def test_multiply_variadic(self):
        from agent_alpha.operators.function_lib import MULTIPLY

        s = _panel()
        pd.testing.assert_series_equal(MULTIPLY(s, 2.0, 0.5), s)

    def test_divide(self):
        from agent_alpha.operators.function_lib import DIVIDE

        s = _panel()
        pd.testing.assert_series_equal(DIVIDE(s, 2.0), s / 2.0)

    def test_divide_single_arg(self):
        from agent_alpha.operators.function_lib import DIVIDE

        s = _panel()
        pd.testing.assert_series_equal(DIVIDE(s), s)

    def test_neg(self):
        from agent_alpha.operators.function_lib import NEG

        s = _panel()
        pd.testing.assert_series_equal(NEG(s), -s)

    def test_negative_alias(self):
        from agent_alpha.operators.function_lib import NEG, NEGATIVE

        s = _panel()
        pd.testing.assert_series_equal(NEGATIVE(s), NEG(s))

    def test_minus_two_args(self):
        from agent_alpha.operators.function_lib import MINUS, SUBTRACT

        s, t = _panel(), _panel(seed=7)
        pd.testing.assert_series_equal(MINUS(s, t), SUBTRACT(s, t))

    def test_minus_one_arg(self):
        from agent_alpha.operators.function_lib import MINUS, NEG

        s = _panel()
        pd.testing.assert_series_equal(MINUS(s), NEG(s))

    def test_abs(self):
        from agent_alpha.operators.function_lib import ABS

        s = _panel() - 5.0
        assert (ABS(s) >= 0).all()

    def test_sign(self):
        from agent_alpha.operators.function_lib import SIGN

        s = _panel() - 5.0
        result = SIGN(s)
        assert set(result.dropna().unique()).issubset({-1.0, 0.0, 1.0})

    def test_add_raises_on_empty(self):
        from agent_alpha.operators.function_lib import ADD

        with pytest.raises(ValueError):
            ADD()

    def test_subtract_raises_on_empty(self):
        from agent_alpha.operators.function_lib import SUBTRACT

        with pytest.raises(ValueError):
            SUBTRACT()

    def test_multiply_raises_on_empty(self):
        from agent_alpha.operators.function_lib import MULTIPLY

        with pytest.raises(ValueError):
            MULTIPLY()

    def test_divide_raises_on_empty(self):
        from agent_alpha.operators.function_lib import DIVIDE

        with pytest.raises(ValueError):
            DIVIDE()


# ---------------------------------------------------------------------------
# Comparison operators
# ---------------------------------------------------------------------------


class TestComparisons:
    def test_gt(self):
        from agent_alpha.operators.function_lib import GT

        s = _panel()
        pd.testing.assert_series_equal(GT(s, 5.0).astype(bool), s > 5.0)

    def test_lt(self):
        from agent_alpha.operators.function_lib import LT

        s = _panel()
        pd.testing.assert_series_equal(LT(s, 5.0).astype(bool), s < 5.0)

    def test_ge_self(self):
        from agent_alpha.operators.function_lib import GE

        s = _panel()
        assert GE(s, s).all()

    def test_le_self(self):
        from agent_alpha.operators.function_lib import LE

        s = _panel()
        assert LE(s, s).all()

    def test_eq_self(self):
        from agent_alpha.operators.function_lib import EQ

        s = _panel()
        assert EQ(s, s).all()

    def test_ne_self_is_false(self):
        from agent_alpha.operators.function_lib import NE

        s = _panel()
        assert not NE(s, s).any()


# ---------------------------------------------------------------------------
# Logical operators
# ---------------------------------------------------------------------------


class TestLogical:
    def test_and_all_true(self):
        from agent_alpha.operators.function_lib import AND, GT

        s = _panel()
        cond = GT(s, 0)
        assert AND(cond, cond).all()

    def test_and_single_arg(self):
        from agent_alpha.operators.function_lib import AND, GT

        s = _panel()
        cond = GT(s, 0)
        assert AND(cond).all()

    def test_and_three_args(self):
        from agent_alpha.operators.function_lib import AND, GT

        s = _panel()
        cond = GT(s, 0)
        assert AND(cond, cond, cond).all()

    def test_or_one_true(self):
        from agent_alpha.operators.function_lib import GT, LT, OR

        s = _panel()
        assert OR(GT(s, 0), LT(s, 0)).all()

    def test_or_single_arg(self):
        from agent_alpha.operators.function_lib import GT, OR

        s = _panel()
        assert OR(GT(s, 0)).all()

    def test_where_selects_branches(self):
        from agent_alpha.operators.function_lib import GT, WHERE

        s = _panel()
        cond = GT(s, 5.0)
        result = WHERE(cond, s, 0.0)
        assert (result[cond] == s[cond]).all()
        assert (result[~cond] == 0.0).all()

    def test_and_raises_on_empty(self):
        from agent_alpha.operators.function_lib import AND

        with pytest.raises(ValueError):
            AND()

    def test_or_raises_on_empty(self):
        from agent_alpha.operators.function_lib import OR

        with pytest.raises(ValueError):
            OR()


# ---------------------------------------------------------------------------
# Cross-sectional operators
# ---------------------------------------------------------------------------


class TestCrossSectional:
    def test_rank_in_unit_interval(self):
        from agent_alpha.operators.function_lib import RANK

        s = _panel()
        result = RANK(s)
        assert (result >= 0).all() and (result <= 1).all()

    def test_rank_with_period_delegates(self):
        from agent_alpha.operators.function_lib import RANK, TS_RANK

        s = _panel(dates=10)
        pd.testing.assert_series_equal(RANK(s, 3), TS_RANK(s, 3))

    def test_zscore_mean_near_zero(self):
        from agent_alpha.operators.function_lib import ZSCORE

        s = _panel(dates=20, tickers=10)
        result = ZSCORE(s)
        assert (result.groupby(level="datetime").mean().abs() < 1e-10).all()

    def test_zscore_with_period_delegates(self):
        from agent_alpha.operators.function_lib import TS_ZSCORE, ZSCORE

        s = _panel(dates=10)
        pd.testing.assert_series_equal(ZSCORE(s, 5), TS_ZSCORE(s, 5))

    def test_mean_cross_sectional(self):
        from agent_alpha.operators.function_lib import MEAN

        s = _panel()
        pd.testing.assert_series_equal(MEAN(s), s.groupby(level="datetime").transform("mean"))

    def test_mean_with_period_delegates(self):
        from agent_alpha.operators.function_lib import MEAN, TS_MEAN

        s = _panel(dates=10)
        pd.testing.assert_series_equal(MEAN(s, 3), TS_MEAN(s, 3))

    def test_std_cross_sectional(self):
        from agent_alpha.operators.function_lib import STD

        assert isinstance(STD(_panel(tickers=5)), pd.Series)

    def test_std_with_period_delegates(self):
        from agent_alpha.operators.function_lib import STD, TS_STD

        s = _panel(dates=10)
        pd.testing.assert_series_equal(STD(s, 3), TS_STD(s, 3))

    def test_median_cross_sectional(self):
        from agent_alpha.operators.function_lib import MEDIAN

        assert isinstance(MEDIAN(_panel()), pd.Series)

    def test_median_with_period(self):
        from agent_alpha.operators.function_lib import MEDIAN

        assert isinstance(MEDIAN(_panel(dates=10), 3), pd.Series)

    def test_max_cross_sectional(self):
        from agent_alpha.operators.function_lib import MAX

        s = _panel()
        pd.testing.assert_series_equal(MAX(s), s.groupby(level="datetime").transform("max"))

    def test_max_elementwise(self):
        from agent_alpha.operators.function_lib import MAX

        s, t = _panel(), _panel(seed=99)
        np.testing.assert_array_almost_equal(MAX(s, t).values, np.maximum(s.values, t.values))

    def test_max_raises_on_empty(self):
        from agent_alpha.operators.function_lib import MAX

        with pytest.raises(ValueError):
            MAX()

    def test_min_cross_sectional(self):
        from agent_alpha.operators.function_lib import MIN

        s = _panel()
        pd.testing.assert_series_equal(MIN(s), s.groupby(level="datetime").transform("min"))

    def test_min_elementwise(self):
        from agent_alpha.operators.function_lib import MIN

        s, t = _panel(), _panel(seed=99)
        np.testing.assert_array_almost_equal(MIN(s, t).values, np.minimum(s.values, t.values))

    def test_min_raises_on_empty(self):
        from agent_alpha.operators.function_lib import MIN

        with pytest.raises(ValueError):
            MIN()

    def test_percentile(self):
        from agent_alpha.operators.function_lib import PERCENTILE

        assert isinstance(PERCENTILE(_panel(), 0.5), pd.Series)

    def test_scale(self):
        from agent_alpha.operators.function_lib import SCALE

        assert isinstance(SCALE(_panel()), pd.Series)

    def test_sum_cross_sectional(self):
        from agent_alpha.operators.function_lib import SUM

        s = _panel()
        pd.testing.assert_series_equal(SUM(s), s.groupby(level="datetime").transform("sum"))

    def test_sum_with_series_args_delegates_add(self):
        from agent_alpha.operators.function_lib import ADD, SUM

        s, t = _panel(), _panel(seed=7)
        pd.testing.assert_series_equal(SUM(s, t), ADD(s, t))

    def test_sum_with_window_pair(self):
        from agent_alpha.operators.function_lib import SUM, TS_SUM

        s = _panel(dates=10)
        # SUM(x, 0, p) should call TS_SUM(x, p)
        result = SUM(s, 0, 3)
        pd.testing.assert_series_equal(result, TS_SUM(s, 3))

    def test_nullif_equal_all_nan(self):
        from agent_alpha.operators.function_lib import NULLIF

        s = _panel()
        assert NULLIF(s, s).isna().all()


# ---------------------------------------------------------------------------
# Time-series operators
# ---------------------------------------------------------------------------


class TestTimeSeries:
    def test_delta(self):
        from agent_alpha.operators.function_lib import DELTA

        assert isinstance(DELTA(_panel(dates=10), 1), pd.Series)

    def test_delta_series_period(self):
        from agent_alpha.operators.function_lib import DELTA

        s = _panel(dates=10)
        # When p is a Series, DELTA delegates to SUBTRACT
        p_as_series = pd.Series(1, index=s.index)
        result = DELTA(s, p_as_series)
        assert isinstance(result, pd.Series)

    def test_delay(self):
        from agent_alpha.operators.function_lib import DELAY

        assert isinstance(DELAY(_panel(dates=10), 2), pd.Series)

    def test_ts_mean(self):
        from agent_alpha.operators.function_lib import TS_MEAN

        assert isinstance(TS_MEAN(_panel(dates=10), 3), pd.Series)

    def test_ts_std(self):
        from agent_alpha.operators.function_lib import TS_STD

        assert isinstance(TS_STD(_panel(dates=10), 3), pd.Series)

    def test_ts_sum(self):
        from agent_alpha.operators.function_lib import TS_SUM

        assert isinstance(TS_SUM(_panel(dates=10), 3), pd.Series)

    def test_ts_max(self):
        from agent_alpha.operators.function_lib import TS_MAX

        assert isinstance(TS_MAX(_panel(dates=10), 3), pd.Series)

    def test_ts_min(self):
        from agent_alpha.operators.function_lib import TS_MIN

        assert isinstance(TS_MIN(_panel(dates=10), 3), pd.Series)

    def test_ts_rank_in_unit_interval(self):
        from agent_alpha.operators.function_lib import TS_RANK

        s = _panel(dates=10)
        valid = TS_RANK(s, 3).dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_ts_zscore(self):
        from agent_alpha.operators.function_lib import TS_ZSCORE

        assert isinstance(TS_ZSCORE(_panel(dates=10), 5), pd.Series)

    def test_ts_corr(self):
        from agent_alpha.operators.function_lib import TS_CORR

        s, t = _panel(dates=20), _panel(dates=20, seed=99)
        assert isinstance(TS_CORR(s, t, 5), pd.Series)

    def test_ema(self):
        from agent_alpha.operators.function_lib import EMA

        assert isinstance(EMA(_panel(dates=10), 3), pd.Series)

    def test_sma_simple(self):
        from agent_alpha.operators.function_lib import SMA

        assert isinstance(SMA(_panel(dates=10), 5), pd.Series)

    def test_sma_with_n(self):
        from agent_alpha.operators.function_lib import SMA

        assert isinstance(SMA(_panel(dates=10), 5, 2), pd.Series)

    def test_wma(self):
        from agent_alpha.operators.function_lib import WMA

        assert isinstance(WMA(_panel(dates=10), 3), pd.Series)

    def test_ts_pctchange(self):
        from agent_alpha.operators.function_lib import TS_PCTCHANGE

        assert isinstance(TS_PCTCHANGE(_panel(dates=10), 1), pd.Series)

    def test_cum_equals_ts_sum(self):
        from agent_alpha.operators.function_lib import CUM, TS_SUM

        s = _panel(dates=10)
        pd.testing.assert_series_equal(CUM(s, 3), TS_SUM(s, 3))


# ---------------------------------------------------------------------------
# Math operators
# ---------------------------------------------------------------------------


class TestMath:
    def test_log_positive(self):
        from agent_alpha.operators.function_lib import LOG

        assert not LOG(_pos_panel()).isna().all()

    def test_log_negative_gives_nan(self):
        from agent_alpha.operators.function_lib import LOG

        s = _panel() - 100.0  # all negative
        assert LOG(s).isna().all()

    def test_log_scalar_positive(self):
        from agent_alpha.operators.function_lib import LOG

        result = LOG(2.0)
        assert abs(result - np.log(2.0)) < 1e-12

    def test_log_scalar_nonpositive(self):
        import math

        from agent_alpha.operators.function_lib import LOG

        assert math.isnan(LOG(-1.0))

    def test_sqrt(self):
        from agent_alpha.operators.function_lib import SQRT

        assert isinstance(SQRT(_pos_panel()), pd.Series)

    def test_exp(self):
        from agent_alpha.operators.function_lib import EXP

        assert isinstance(EXP(_panel()), pd.Series)

    def test_inv(self):
        from agent_alpha.operators.function_lib import INV

        s = _pos_panel()
        pd.testing.assert_series_equal(INV(s), 1.0 / s)

    def test_pow(self):
        from agent_alpha.operators.function_lib import POW

        assert isinstance(POW(_pos_panel(), 2.0), pd.Series)


# ---------------------------------------------------------------------------
# Feature indicator functions
# ---------------------------------------------------------------------------


class TestFeatureIndicators:
    def test_rsi_range(self):
        from agent_alpha.features.rsi import rsi

        df = _ohlcv()
        valid = rsi(df, period=14, shift=0).dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rsi_series_input(self):
        from agent_alpha.features.rsi import rsi

        assert isinstance(rsi(_ohlcv()["close"], period=14, shift=0), pd.Series)

    def test_rsi_all_methods(self):
        from agent_alpha.features.rsi import rsi

        df = _ohlcv()
        for method in ("wilder", "ema", "sma"):
            assert not rsi(df, period=14, method=method, shift=0).dropna().empty

    def test_rsi_shift(self):
        from agent_alpha.features.rsi import rsi

        df = _ohlcv()
        r0 = rsi(df, shift=0)
        r1 = rsi(df, shift=1)
        pd.testing.assert_series_equal(
            r1.iloc[1:].reset_index(drop=True),
            r0.iloc[:-1].reset_index(drop=True),
        )

    def test_rsi_invalid_period(self):
        from agent_alpha.features.rsi import rsi

        with pytest.raises(ValueError):
            rsi(_ohlcv(), period=0)

    def test_rsi_invalid_shift(self):
        from agent_alpha.features.rsi import rsi

        with pytest.raises(ValueError):
            rsi(_ohlcv(), shift=-1)

    def test_atr_positive(self):
        from agent_alpha.features.atr import atr

        assert (atr(_ohlcv(), period=14, shift=0).dropna() > 0).all()

    def test_atr_series_raises(self):
        from agent_alpha.features.atr import atr

        with pytest.raises(TypeError):
            atr(_ohlcv()["close"])

    def test_atr_all_methods(self):
        from agent_alpha.features.atr import atr

        df = _ohlcv()
        for method in ("wilder", "ema", "sma"):
            assert not atr(df, period=14, method=method, shift=0).dropna().empty

    def test_atr_invalid_period(self):
        from agent_alpha.features.atr import atr

        with pytest.raises(ValueError):
            atr(_ohlcv(), period=0)

    def test_ema_basic(self):
        from agent_alpha.features.moving_averages import ema

        assert isinstance(ema(_ohlcv(), period=10, shift=0), pd.Series)

    def test_ema_series_input(self):
        from agent_alpha.features.moving_averages import ema

        assert isinstance(ema(_ohlcv()["close"], period=10, shift=0), pd.Series)

    def test_sma_basic(self):
        from agent_alpha.features.moving_averages import sma

        assert isinstance(sma(_ohlcv(), period=10, shift=0), pd.Series)

    def test_sma_series_input(self):
        from agent_alpha.features.moving_averages import sma

        assert isinstance(sma(_ohlcv()["close"], period=10, shift=0), pd.Series)

    def test_macd_hist(self):
        from agent_alpha.features.macd import macd_hist

        assert isinstance(macd_hist(_ohlcv(n=60), shift=0), pd.Series)

    def test_macd_hist_sma_method(self):
        from agent_alpha.features.macd import macd_hist

        assert isinstance(macd_hist(_ohlcv(n=60), ma_method="sma", shift=0), pd.Series)

    def test_macd_hist_series_input(self):
        from agent_alpha.features.macd import macd_hist

        assert isinstance(macd_hist(_ohlcv(n=60)["close"], shift=0), pd.Series)

    def test_kdj_k(self):
        from agent_alpha.features.kdj import kdj_k

        assert isinstance(kdj_k(_ohlcv(), shift=0), pd.Series)

    def test_kdj_d(self):
        from agent_alpha.features.kdj import kdj_d

        assert isinstance(kdj_d(_ohlcv(), shift=0), pd.Series)

    def test_kdj_j(self):
        from agent_alpha.features.kdj import kdj_j

        assert isinstance(kdj_j(_ohlcv(), shift=0), pd.Series)

    def test_candle_range_nonnegative(self):
        from agent_alpha.features.range_wicks import candle_range

        result = candle_range(_ohlcv(), shift=0)
        assert (result.dropna() >= 0).all()

    def test_wick_ratio(self):
        from agent_alpha.features.range_wicks import wick_ratio

        result = wick_ratio(_ohlcv(), shift=0)
        assert isinstance(result, pd.Series)

    def test_cvd(self):
        from agent_alpha.features.cvd import cvd

        assert isinstance(cvd(_ohlcv(), shift=0), pd.Series)

    def test_cvd_zero_base_false(self):
        from agent_alpha.features.cvd import cvd

        assert isinstance(cvd(_ohlcv(), zero_base=False, shift=0), pd.Series)


# ---------------------------------------------------------------------------
# AST interpreter
# ---------------------------------------------------------------------------


class TestASTInterpreter:
    def _variables(self, dates: int = 5, tickers: int = 3) -> dict:
        rng = np.random.default_rng(42)
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2024-01-01", periods=dates), [f"T{i}" for i in range(tickers)]],
            names=["datetime", "instrument"],
        )
        return {
            "$close": pd.Series(rng.uniform(90, 110, len(idx)), index=idx, name="$close"),
            "$volume": pd.Series(rng.uniform(1e6, 5e6, len(idx)), index=idx, name="$volume"),
        }

    def _interp(self):
        from agent_alpha.ast.interpreter import ASTInterpreter
        from agent_alpha.ast.registry import build_default_registry

        return ASTInterpreter(build_default_registry())

    def test_var_node(self):
        from agent_alpha.ast.nodes import VarNode

        interp = self._interp()
        variables = self._variables()
        result = interp.evaluate(VarNode("$close"), variables)
        pd.testing.assert_series_equal(result, variables["$close"])

    def test_var_node_plain_name_resolves(self):
        from agent_alpha.ast.nodes import VarNode

        interp = self._interp()
        variables = self._variables()
        # "close" should resolve to "$close" via the fallback lookup
        result = interp.evaluate(VarNode("close"), variables)
        pd.testing.assert_series_equal(result, variables["$close"])

    def test_var_node_unknown_raises(self):
        from agent_alpha.ast.interpreter import ASTInterpretationError
        from agent_alpha.ast.nodes import VarNode

        interp = self._interp()
        with pytest.raises(ASTInterpretationError):
            interp.evaluate(VarNode("$nonexistent"), {})

    def test_const_node(self):
        from agent_alpha.ast.nodes import ConstNode

        interp = self._interp()
        assert interp.evaluate(ConstNode(42.0), {}) == 42.0

    def test_call_node_add(self):
        from agent_alpha.ast.nodes import CallNode, ConstNode, VarNode

        interp = self._interp()
        variables = self._variables()
        node = CallNode(op="ADD", args=(VarNode("$close"), ConstNode(1.0)))
        result = interp.evaluate(node, variables)
        pd.testing.assert_series_equal(result, variables["$close"] + 1.0)

    def test_call_node_rank(self):
        from agent_alpha.ast.nodes import CallNode, VarNode

        interp = self._interp()
        variables = self._variables()
        node = CallNode(op="RANK", args=(VarNode("$close"),))
        result = interp.evaluate(node, variables)
        assert isinstance(result, pd.Series)
        assert (result >= 0).all() and (result <= 1).all()

    def test_call_node_where(self):
        from agent_alpha.ast.nodes import CallNode, ConstNode, VarNode

        interp = self._interp()
        variables = self._variables()
        node = CallNode(
            op="WHERE",
            args=(
                CallNode(op="GT", args=(VarNode("$close"), ConstNode(100.0))),
                VarNode("$close"),
                ConstNode(0.0),
            ),
        )
        result = interp.evaluate(node, variables)
        assert isinstance(result, pd.Series)

    def test_unsupported_node_raises(self):
        from agent_alpha.ast.interpreter import ASTInterpretationError

        interp = self._interp()
        with pytest.raises(ASTInterpretationError):
            interp.evaluate("not_a_node", {})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Feature engine
# ---------------------------------------------------------------------------


class TestFeatureEngine:
    def test_compute_rsi_component(self):
        from agent_alpha.data import load_synthetic_panel
        from agent_alpha.feature_engine import compute_feature_component
        from agent_alpha.models import FeatureComponentSpec

        panel = load_synthetic_panel(n_days=30, n_tickers=5, seed=42)
        comp = FeatureComponentSpec(id="c1", feature="rsi", params={"period": 14})
        result = compute_feature_component(panel, comp)
        assert isinstance(result, pd.Series)
        assert result.index.equals(panel.index)

    def test_compute_ema_component(self):
        from agent_alpha.data import load_synthetic_panel
        from agent_alpha.feature_engine import compute_feature_component
        from agent_alpha.models import FeatureComponentSpec

        panel = load_synthetic_panel(n_days=30, n_tickers=5, seed=42)
        comp = FeatureComponentSpec(id="c1", feature="ema", params={"period": 10})
        result = compute_feature_component(panel, comp)
        assert isinstance(result, pd.Series)

    def test_compute_atr_component(self):
        from agent_alpha.data import load_synthetic_panel
        from agent_alpha.feature_engine import compute_feature_component
        from agent_alpha.models import FeatureComponentSpec

        panel = load_synthetic_panel(n_days=30, n_tickers=5, seed=42)
        comp = FeatureComponentSpec(id="c1", feature="atr", params={"period": 14})
        result = compute_feature_component(panel, comp)
        assert isinstance(result, pd.Series)

    def test_compute_blueprint_components(self):
        from agent_alpha.data import load_synthetic_panel
        from agent_alpha.feature_engine import compute_blueprint_components
        from agent_alpha.models import FeatureComponentSpec

        panel = load_synthetic_panel(n_days=30, n_tickers=5, seed=42)
        components = [
            FeatureComponentSpec(id="c1", feature="rsi", params={"period": 14}),
            FeatureComponentSpec(id="c2", feature="ema", params={"period": 10}),
        ]
        # Returns (augmented_panel_df, column_mapping_dict)
        augmented, col_map = compute_blueprint_components(panel, components)
        assert set(col_map.keys()) == {"c1", "c2"}
        assert isinstance(augmented, pd.DataFrame)
        assert augmented.index.equals(panel.index)
