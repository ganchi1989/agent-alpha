from __future__ import annotations

import operator
import warnings
from typing import Any, Callable

import numpy as np
import pandas as pd


def _as_series(value: Any) -> pd.Series | float | int | bool:
    if isinstance(value, pd.DataFrame):
        if value.shape[1] != 1:
            raise ValueError("Expected a single-column DataFrame")
        return value.iloc[:, 0]
    if isinstance(value, pd.Series):
        return value
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _coerce_window(value: Any, default: int = 1, minimum: int = 1) -> int:
    raw = _as_series(value)
    if isinstance(raw, pd.Series):
        non_null = raw.dropna()
        if non_null.empty:
            raw = default
        else:
            raw = non_null.iloc[0]
    try:
        out = int(float(raw))
    except Exception:
        out = int(default)
    if out < int(minimum):
        out = int(minimum)
    return out


def _ensure_index_names(series: pd.Series) -> pd.Series:
    if isinstance(series.index, pd.MultiIndex) and list(series.index.names) != [
        "datetime",
        "instrument",
    ]:
        series = series.copy()
        series.index = series.index.set_names(["datetime", "instrument"])
    return series


def _target_index(lhs: Any, rhs: Any) -> pd.Index | pd.MultiIndex | None:
    for value in (lhs, rhs):
        value = _as_series(value)
        if isinstance(value, pd.Series):
            return value.index
    return None


def _broadcast_to_index(value: Any, index: pd.Index | pd.MultiIndex | None) -> Any:
    value = _as_series(value)
    if index is None:
        return value
    if not isinstance(value, pd.Series):
        if isinstance(value, np.ndarray):
            if len(value) != len(index):
                raise ValueError("Cannot align ndarray to target index with different length")
            return pd.Series(value, index=index)
        return value

    series = _ensure_index_names(value)
    if series.index.equals(index):
        return series

    if isinstance(index, pd.MultiIndex) and not isinstance(series.index, pd.MultiIndex):
        datetime_index = index.get_level_values("datetime")
        aligned = series.reindex(datetime_index)
        aligned.index = index
        return aligned

    if not isinstance(index, pd.MultiIndex) and isinstance(series.index, pd.MultiIndex):
        datetime_index = series.index.get_level_values("datetime")
        aligned = series.copy()
        aligned.index = datetime_index
        return aligned.reindex(index)

    return series.reindex(index)


def _binary_op(lhs: Any, rhs: Any, op: Callable[[Any, Any], Any]) -> Any:
    target = _target_index(lhs, rhs)
    left = _broadcast_to_index(lhs, target)
    right = _broadcast_to_index(rhs, target)
    return op(left, right)


def _groupby_instrument(series: pd.Series):
    series = _ensure_index_names(series)
    if not isinstance(series.index, pd.MultiIndex):
        raise ValueError("Expected MultiIndex(datetime, instrument)")
    return series.groupby(level="instrument", sort=False)


def _groupby_datetime(series: pd.Series):
    series = _ensure_index_names(series)
    if not isinstance(series.index, pd.MultiIndex):
        raise ValueError("Expected MultiIndex(datetime, instrument)")
    return series.groupby(level="datetime", sort=False)


def DELTA(x: Any, p: int = 1) -> pd.Series:
    s = _as_series(x)
    if not isinstance(s, pd.Series):
        return pd.Series([np.nan])
    p_val = _as_series(p)
    if isinstance(p_val, pd.Series):
        return SUBTRACT(s, p_val)
    try:
        period = int(p_val)
    except Exception:
        period = 1
    return _groupby_instrument(s).transform(lambda v: v.diff(periods=period))


def DELAY(x: Any, p: int = 1) -> pd.Series:
    p = _coerce_window(p, default=1, minimum=0)
    s = _as_series(x)
    if not isinstance(s, pd.Series):
        return pd.Series([np.nan])
    return _groupby_instrument(s).transform(lambda v: v.shift(p))


def RANK(x: Any, p: int | None = None) -> pd.Series:
    if p is not None:
        return TS_RANK(x, p)
    s = _as_series(x)
    if not isinstance(s, pd.Series):
        return pd.Series([1.0])
    return _groupby_datetime(s).rank(pct=True)


def ZSCORE(x: Any, p: int | None = None) -> pd.Series:
    if p is not None:
        return TS_ZSCORE(x, p)
    s = _as_series(x)
    if not isinstance(s, pd.Series):
        return pd.Series([0.0])
    mean = _groupby_datetime(s).transform("mean")
    std = _groupby_datetime(s).transform("std")
    std = std.replace(0, np.nan)
    return (s - mean) / std


def MEAN(x: Any, p: int | None = None) -> pd.Series:
    if p is not None:
        return TS_MEAN(x, p)
    s = _as_series(x)
    if not isinstance(s, pd.Series):
        return pd.Series([float(s)])
    return _groupby_datetime(s).transform("mean")


def STD(x: Any, p: int | None = None) -> pd.Series:
    if p is not None:
        return TS_STD(x, p)
    s = _as_series(x)
    if not isinstance(s, pd.Series):
        return pd.Series([0.0])
    return _groupby_datetime(s).transform("std")


def MEDIAN(x: Any, p: int | None = None) -> pd.Series:
    if p is not None:
        s = _as_series(x)
        w = _coerce_window(p, default=5, minimum=1)
        return _groupby_instrument(s).transform(lambda v: v.rolling(w, min_periods=1).median())
    s = _as_series(x)
    if not isinstance(s, pd.Series):
        return pd.Series([float(s)])
    return _groupby_datetime(s).transform("median")


def _rolling_apply(s: pd.Series, p: int, fn: Callable[[pd.Series], float]) -> pd.Series:
    p = _coerce_window(p, default=1, minimum=1)
    return _groupby_instrument(s).transform(
        lambda v: v.rolling(p, min_periods=1).apply(fn, raw=False)
    )


def TS_SUM(x: Any, p: int = 5) -> pd.Series:
    s = _as_series(x)
    w = _coerce_window(p, default=5, minimum=1)
    return _groupby_instrument(s).transform(lambda v: v.rolling(w, min_periods=1).sum())


def TS_MEAN(x: Any, p: int = 5) -> pd.Series:
    s = _as_series(x)
    w = _coerce_window(p, default=5, minimum=1)
    return _groupby_instrument(s).transform(lambda v: v.rolling(w, min_periods=1).mean())


def TS_STD(x: Any, p: int = 20) -> pd.Series:
    s = _as_series(x)
    w = _coerce_window(p, default=20, minimum=1)
    return _groupby_instrument(s).transform(lambda v: v.rolling(w, min_periods=1).std())


def TS_MAX(x: Any, p: int = 5) -> pd.Series:
    s = _as_series(x)
    w = _coerce_window(p, default=5, minimum=1)
    return _groupby_instrument(s).transform(lambda v: v.rolling(w, min_periods=1).max())


def TS_MIN(x: Any, p: int = 5) -> pd.Series:
    s = _as_series(x)
    w = _coerce_window(p, default=5, minimum=1)
    return _groupby_instrument(s).transform(lambda v: v.rolling(w, min_periods=1).min())


def TS_RANK(x: Any, p: int = 5) -> pd.Series:
    s = _as_series(x)

    def _rank_last(window: pd.Series) -> float:
        return float(window.rank(pct=True).iloc[-1])

    return _rolling_apply(s, p, _rank_last)


def TS_ZSCORE(x: Any, p: int = 5) -> pd.Series:
    s = _as_series(x)
    rolling_mean = TS_MEAN(s, p)
    rolling_std = TS_STD(s, p).replace(0, np.nan)
    return (s - rolling_mean) / rolling_std


def TS_CORR(x: Any, y: Any, p: int = 20) -> pd.Series:
    p = _coerce_window(p, default=20, minimum=2)

    target = _target_index(x, y)
    sx = _broadcast_to_index(x, target)
    sy = _broadcast_to_index(y, target)
    sx = _as_series(sx)
    sy = _as_series(sy)
    if not isinstance(sx, pd.Series) or not isinstance(sy, pd.Series):
        return pd.Series([np.nan])

    sx = _ensure_index_names(sx)
    sy = _ensure_index_names(sy)
    frame = pd.concat({"x": sx, "y": sy}, axis=1)

    def _corr_group(group: pd.DataFrame) -> pd.Series:
        # Rolling corr can emit many RuntimeWarnings (divide by zero on flat windows).
        # Keep those as NaN without flooding notebook output.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with np.errstate(all="ignore"):
                corr = group["x"].rolling(p, min_periods=2).corr(group["y"])
        return corr.replace([np.inf, -np.inf], np.nan)

    out = frame.groupby(level="instrument", sort=False, group_keys=False).apply(_corr_group)
    out.index = sx.index
    return out


def EMA(x: Any, p: int = 12) -> pd.Series:
    s = _as_series(x)
    w = _coerce_window(p, default=12, minimum=1)
    return _groupby_instrument(s).transform(
        lambda v: v.ewm(span=w, adjust=False, min_periods=1).mean()
    )


def SMA(x: Any, m: int = 5, n: int | None = None) -> pd.Series:
    s = _as_series(x)
    if n is None:
        w = _coerce_window(m, default=5, minimum=1)
        return _groupby_instrument(s).transform(lambda v: v.rolling(w, min_periods=1).mean())
    m_val = _coerce_window(m, default=5, minimum=1)
    n_val = _coerce_window(n, default=1, minimum=1)
    alpha = float(n_val) / float(m_val)
    return _groupby_instrument(s).transform(
        lambda v: v.ewm(alpha=alpha, adjust=False, min_periods=1).mean()
    )


def ABS(x: Any) -> Any:
    return _binary_op(x, 0, lambda a, _b: np.abs(a))


def WMA(x: Any, p: int = 5) -> pd.Series:
    p = _coerce_window(p, default=5, minimum=1)
    s = _as_series(x)
    if not isinstance(s, pd.Series):
        return pd.Series([float(s)])

    weights = np.arange(1, p + 1, dtype=float)

    def _wma_window(values: np.ndarray) -> float:
        w = weights[-len(values) :]
        denom = float(w.sum())
        if denom <= 0:
            return float(np.nan)
        return float(np.dot(values, w) / denom)

    return _groupby_instrument(s).transform(
        lambda v: v.rolling(p, min_periods=1).apply(_wma_window, raw=True)
    )


def LOG(x: Any) -> Any:
    s = _as_series(x)
    if isinstance(s, pd.Series):
        out = s.copy()
        out = out.where(out > 0, np.nan)
        return np.log(out)
    if s <= 0:
        return np.nan
    return np.log(s)


def SQRT(x: Any) -> Any:
    return _binary_op(x, 0, lambda a, _b: np.sqrt(a))


def EXP(x: Any) -> Any:
    return _binary_op(x, 0, lambda a, _b: np.exp(a))


def INV(x: Any) -> Any:
    return _binary_op(1.0, x, lambda a, b: a / b)


def POW(x: Any, n: float) -> Any:
    return _binary_op(x, n, lambda a, b: np.power(a, b))


def SIGN(x: Any) -> Any:
    return _binary_op(x, 0, lambda a, _b: np.sign(a))


def SCALE(x: Any, target_sum: float = 1.0) -> pd.Series:
    s = _as_series(x)
    abs_sum = ABS(s).groupby(level="datetime").transform("sum")
    abs_sum = abs_sum.replace(0, np.nan)
    return s * float(target_sum) / abs_sum


def CUM(x: Any, p: int = 5) -> pd.Series:
    return TS_SUM(x, p)


def PERCENTILE(x: Any, q: float = 0.5) -> Any:
    s = _as_series(x)
    qf = float(q)
    if isinstance(s, pd.Series):
        return _groupby_datetime(s).transform(lambda v: v.quantile(qf))
    return s


def TS_PCTCHANGE(x: Any, p: int = 1) -> pd.Series:
    s = _as_series(x)
    w = _coerce_window(p, default=1, minimum=1)
    return _groupby_instrument(s).transform(lambda v: v.pct_change(periods=w, fill_method=None))


def SUM(x: Any, *args: Any) -> Any:
    if args:
        if any(isinstance(_as_series(arg), pd.Series) for arg in args):
            return ADD(x, *args)

        numeric: list[int] = []
        for arg in args:
            try:
                numeric.append(int(float(_as_series(arg))))
            except Exception:
                continue
        if numeric:
            if len(numeric) >= 2 and numeric[0] == 0:
                return TS_SUM(x, numeric[1])
            return TS_SUM(x, numeric[0])

    s = _as_series(x)
    if not isinstance(s, pd.Series):
        return s
    return _groupby_datetime(s).transform("sum")


def ADD(*args: Any) -> Any:
    if not args:
        raise ValueError("ADD requires at least one argument")
    out = args[0]
    for value in args[1:]:
        out = _binary_op(out, value, operator.add)
    return out


def SUBTRACT(*args: Any) -> Any:
    if not args:
        raise ValueError("SUBTRACT requires at least one argument")
    if len(args) == 1:
        return NEG(args[0])
    out = args[0]
    for value in args[1:]:
        out = _binary_op(out, value, operator.sub)
    return out


def MULTIPLY(*args: Any) -> Any:
    if not args:
        raise ValueError("MULTIPLY requires at least one argument")
    out = args[0]
    for value in args[1:]:
        out = _binary_op(out, value, operator.mul)
    return out


def DIVIDE(*args: Any) -> Any:
    if not args:
        raise ValueError("DIVIDE requires at least one argument")
    if len(args) == 1:
        return args[0]
    out = args[0]
    for value in args[1:]:
        out = _binary_op(out, value, operator.truediv)
    return out


def NEG(x: Any) -> Any:
    return MULTIPLY(-1.0, x)


def NEGATIVE(x: Any) -> Any:
    return NEG(x)


def MINUS(x: Any, y: Any | None = None) -> Any:
    if y is None:
        return NEG(x)
    return SUBTRACT(x, y)


def MAX(*args: Any) -> Any:
    """MAX(x) -> cross-sectional max. MAX(x, y, ...) -> element-wise max."""
    if len(args) == 0:
        raise ValueError("MAX requires at least one argument")
    if len(args) == 1:
        s = _as_series(args[0])
        if not isinstance(s, pd.Series):
            return s
        return _groupby_datetime(s).transform("max")

    out = args[0]
    for value in args[1:]:
        out = _binary_op(out, value, np.maximum)
    return out


def MIN(*args: Any) -> Any:
    """MIN(x) -> cross-sectional min. MIN(x, y, ...) -> element-wise min."""
    if len(args) == 0:
        raise ValueError("MIN requires at least one argument")
    if len(args) == 1:
        s = _as_series(args[0])
        if not isinstance(s, pd.Series):
            return s
        return _groupby_datetime(s).transform("min")

    out = args[0]
    for value in args[1:]:
        out = _binary_op(out, value, np.minimum)
    return out


def GT(x: Any, y: Any = 0) -> Any:
    return _binary_op(x, y, operator.gt)


def LT(x: Any, y: Any = 0) -> Any:
    return _binary_op(x, y, operator.lt)


def GE(x: Any, y: Any = 0) -> Any:
    return _binary_op(x, y, operator.ge)


def LE(x: Any, y: Any = 0) -> Any:
    return _binary_op(x, y, operator.le)


def EQ(x: Any, y: Any = 0) -> Any:
    return _binary_op(x, y, operator.eq)


def NE(x: Any, y: Any = 0) -> Any:
    return _binary_op(x, y, operator.ne)


def _bool_view(x: Any) -> Any:
    if isinstance(x, pd.Series):
        return x.astype(bool)
    return np.asarray(x).astype(bool)


def AND(*args: Any) -> Any:
    if not args:
        raise ValueError("AND requires at least one argument")
    if len(args) == 1:
        return _binary_op(args[0], True, lambda a, b: np.logical_and(_bool_view(a), _bool_view(b)))
    out = _binary_op(args[0], args[1], lambda a, b: np.logical_and(_bool_view(a), _bool_view(b)))
    for value in args[2:]:
        out = _binary_op(out, value, lambda a, b: np.logical_and(_bool_view(a), _bool_view(b)))
    return out


def OR(*args: Any) -> Any:
    if not args:
        raise ValueError("OR requires at least one argument")
    if len(args) == 1:
        return _binary_op(args[0], False, lambda a, b: np.logical_or(_bool_view(a), _bool_view(b)))
    out = _binary_op(args[0], args[1], lambda a, b: np.logical_or(_bool_view(a), _bool_view(b)))
    for value in args[2:]:
        out = _binary_op(out, value, lambda a, b: np.logical_or(_bool_view(a), _bool_view(b)))
    return out


def WHERE(condition: Any, true_value: Any, false_value: Any = np.nan) -> Any:
    target = _target_index(condition, true_value)
    if target is None:
        target = _target_index(condition, false_value)
    cond = _broadcast_to_index(condition, target)
    t_val = _broadcast_to_index(true_value, target)
    f_val = _broadcast_to_index(false_value, target)
    out = np.where(cond, t_val, f_val)
    if target is not None:
        return pd.Series(out, index=target)
    return out


def NULLIF(x: Any, y: Any = 0) -> Any:
    target = _target_index(x, y)
    left = _broadcast_to_index(x, target)
    right = _broadcast_to_index(y, target)

    if isinstance(left, pd.Series):
        eq = _binary_op(left, right, operator.eq)
        out = np.where(eq, np.nan, left)
        return pd.Series(out, index=left.index)

    try:
        return np.nan if left == right else left
    except Exception:
        return left


SAFE_FUNCTIONS = {name: obj for name, obj in globals().items() if name.isupper() and callable(obj)}

# Convenience aliases for common lowercase / alternate DSL names from LLM outputs.
SAFE_FUNCTIONS.update({name.lower(): fn for name, fn in SAFE_FUNCTIONS.items()})
SAFE_FUNCTIONS.update(
    {
        "ln": LOG,
        "log": LOG,
        "ref": DELAY,
        "delay": DELAY,
        "delta": DELTA,
        "rank": RANK,
        "zscore": ZSCORE,
        "mean": MEAN,
        "std": STD,
        "median": MEDIAN,
        "ts_sum": TS_SUM,
        "ts_mean": TS_MEAN,
        "ts_std": TS_STD,
        "ts_rank": TS_RANK,
        "ts_zscore": TS_ZSCORE,
        "ts_corr": TS_CORR,
        "rolling_sum": TS_SUM,
        "rolling_mean": TS_MEAN,
        "rolling_std": TS_STD,
        "ema": EMA,
        "sma": SMA,
        "wma": WMA,
        "abs": ABS,
        "sqrt": SQRT,
        "exp": EXP,
        "inv": INV,
        "pow": POW,
        "sign": SIGN,
        "scale": SCALE,
        "pctchange": TS_PCTCHANGE,
        "max": MAX,
        "min": MIN,
        "gt": GT,
        "lt": LT,
        "ge": GE,
        "le": LE,
        "eq": EQ,
        "ne": NE,
        "and": AND,
        "or": OR,
        "where": WHERE,
        "cum": CUM,
        "percentile": PERCENTILE,
        "corr": TS_CORR,
        "sum": SUM,
        "neg": NEG,
        "negative": NEGATIVE,
        "minus": MINUS,
        "nullif": NULLIF,
    }
)


__all__ = sorted(SAFE_FUNCTIONS)
