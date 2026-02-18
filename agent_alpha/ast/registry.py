from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import inspect
from typing import Any, Callable

from ..operators.function_lib import SAFE_FUNCTIONS


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    name: str
    fn: Callable[..., Any]
    min_arity: int
    max_arity: int | None
    variadic: bool
    signature: str
    arg_kinds: tuple[str, ...] = ()
    return_kind: str = "series_like"
    window_arg_positions: tuple[int, ...] = ()
    min_window: int = 1
    cost: float = 1.0


class OperatorRegistry:
    def __init__(self, specs: dict[str, OperatorSpec], aliases: dict[str, str]):
        self._specs = specs
        self._aliases = aliases

    @property
    def aliases(self) -> dict[str, str]:
        return dict(self._aliases)

    def list_canonical(self) -> list[str]:
        return sorted(self._specs.keys())

    def resolve(self, name: str) -> str:
        key = str(name or "").strip()
        if key in self._specs:
            return key
        alias_key = key.lower()
        if alias_key in self._aliases:
            return self._aliases[alias_key]
        raise KeyError(f"Unknown operator: {name!r}")

    def get(self, name: str) -> OperatorSpec:
        canonical = self.resolve(name)
        return self._specs[canonical]

    def prompt_operator_list(self) -> str:
        return ", ".join(self.list_canonical())

    def operator_cost(self, name: str) -> float:
        return float(self.get(name).cost)


def _introspect_arity(fn: Callable[..., Any]) -> tuple[int, int | None, bool, str]:
    try:
        sig = inspect.signature(fn)
    except Exception:
        return 1, None, True, "(...)"

    min_arity = 0
    max_arity = 0
    variadic = False
    for param in sig.parameters.values():
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            if param.default is inspect.Parameter.empty:
                min_arity += 1
            max_arity += 1
            continue
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            variadic = True

    if variadic:
        max_out: int | None = None
    else:
        max_out = max_arity

    return min_arity, max_out, variadic, str(sig)


_OPERATOR_OVERRIDES: dict[str, dict[str, Any]] = {
    "ADD": {"return_kind": "series_like", "arg_kinds": ("any",), "cost": 1.0},
    "SUBTRACT": {"return_kind": "series_like", "arg_kinds": ("any",), "cost": 1.0},
    "MULTIPLY": {"return_kind": "series_like", "arg_kinds": ("any",), "cost": 1.0},
    "DIVIDE": {"return_kind": "series_like", "arg_kinds": ("any",), "cost": 1.2},
    "NEG": {"return_kind": "series_like", "arg_kinds": ("any",)},
    "NEGATIVE": {"return_kind": "series_like", "arg_kinds": ("any",)},
    "MINUS": {"return_kind": "series_like", "arg_kinds": ("any",)},
    "GT": {
        "min_arity": 2,
        "max_arity": 2,
        "arg_kinds": ("any", "any"),
        "return_kind": "bool_like",
        "cost": 0.8,
    },
    "LT": {
        "min_arity": 2,
        "max_arity": 2,
        "arg_kinds": ("any", "any"),
        "return_kind": "bool_like",
        "cost": 0.8,
    },
    "GE": {
        "min_arity": 2,
        "max_arity": 2,
        "arg_kinds": ("any", "any"),
        "return_kind": "bool_like",
        "cost": 0.8,
    },
    "LE": {
        "min_arity": 2,
        "max_arity": 2,
        "arg_kinds": ("any", "any"),
        "return_kind": "bool_like",
        "cost": 0.8,
    },
    "EQ": {
        "min_arity": 2,
        "max_arity": 2,
        "arg_kinds": ("any", "any"),
        "return_kind": "bool_like",
        "cost": 0.8,
    },
    "NE": {
        "min_arity": 2,
        "max_arity": 2,
        "arg_kinds": ("any", "any"),
        "return_kind": "bool_like",
        "cost": 0.8,
    },
    "AND": {
        "min_arity": 1,
        "max_arity": None,
        "variadic": True,
        "arg_kinds": ("bool_like",),
        "return_kind": "bool_like",
        "cost": 1.0,
    },
    "OR": {
        "min_arity": 1,
        "max_arity": None,
        "variadic": True,
        "arg_kinds": ("bool_like",),
        "return_kind": "bool_like",
        "cost": 1.0,
    },
    "WHERE": {
        "min_arity": 3,
        "max_arity": 3,
        "arg_kinds": ("bool_like", "any", "any"),
        "return_kind": "branch_value",
        "cost": 1.5,
    },
    "NULLIF": {
        "min_arity": 1,
        "max_arity": 2,
        "arg_kinds": ("any", "any"),
        "return_kind": "same_as_first",
    },
    "DELTA": {
        "min_arity": 1,
        "max_arity": 2,
        "arg_kinds": ("series_like", "scalar"),
        "window_arg_positions": (1,),
        "min_window": 1,
        "cost": 1.2,
    },
    "DELAY": {
        "min_arity": 1,
        "max_arity": 2,
        "arg_kinds": ("series_like", "scalar"),
        "window_arg_positions": (1,),
        "min_window": 0,
        "cost": 1.1,
    },
    "TS_SUM": {
        "min_arity": 1,
        "max_arity": 2,
        "arg_kinds": ("series_like", "scalar"),
        "window_arg_positions": (1,),
        "min_window": 1,
        "cost": 1.2,
    },
    "TS_MEAN": {
        "min_arity": 1,
        "max_arity": 2,
        "arg_kinds": ("series_like", "scalar"),
        "window_arg_positions": (1,),
        "min_window": 1,
        "cost": 1.2,
    },
    "TS_STD": {
        "min_arity": 1,
        "max_arity": 2,
        "arg_kinds": ("series_like", "scalar"),
        "window_arg_positions": (1,),
        "min_window": 1,
        "cost": 1.2,
    },
    "TS_MAX": {
        "min_arity": 1,
        "max_arity": 2,
        "arg_kinds": ("series_like", "scalar"),
        "window_arg_positions": (1,),
        "min_window": 1,
        "cost": 1.2,
    },
    "TS_MIN": {
        "min_arity": 1,
        "max_arity": 2,
        "arg_kinds": ("series_like", "scalar"),
        "window_arg_positions": (1,),
        "min_window": 1,
        "cost": 1.2,
    },
    "TS_RANK": {
        "min_arity": 1,
        "max_arity": 2,
        "arg_kinds": ("series_like", "scalar"),
        "window_arg_positions": (1,),
        "min_window": 1,
        "cost": 1.5,
    },
    "TS_ZSCORE": {
        "min_arity": 1,
        "max_arity": 2,
        "arg_kinds": ("series_like", "scalar"),
        "window_arg_positions": (1,),
        "min_window": 1,
        "cost": 1.5,
    },
    "TS_CORR": {
        "min_arity": 2,
        "max_arity": 3,
        "arg_kinds": ("series_like", "series_like", "scalar"),
        "window_arg_positions": (2,),
        "min_window": 2,
        "cost": 2.2,
    },
    "EMA": {
        "min_arity": 1,
        "max_arity": 2,
        "arg_kinds": ("series_like", "scalar"),
        "window_arg_positions": (1,),
        "min_window": 1,
        "cost": 1.4,
    },
    "SMA": {
        "min_arity": 1,
        "max_arity": 3,
        "arg_kinds": ("series_like", "scalar", "scalar"),
        "window_arg_positions": (1, 2),
        "min_window": 1,
        "cost": 1.4,
    },
    "WMA": {
        "min_arity": 1,
        "max_arity": 2,
        "arg_kinds": ("series_like", "scalar"),
        "window_arg_positions": (1,),
        "min_window": 1,
        "cost": 1.4,
    },
    "TS_PCTCHANGE": {
        "min_arity": 1,
        "max_arity": 2,
        "arg_kinds": ("series_like", "scalar"),
        "window_arg_positions": (1,),
        "min_window": 1,
        "cost": 1.3,
    },
    "CUM": {
        "min_arity": 1,
        "max_arity": 2,
        "arg_kinds": ("series_like", "scalar"),
        "window_arg_positions": (1,),
        "min_window": 1,
        "cost": 1.3,
    },
    "RANK": {
        "arg_kinds": ("series_like", "scalar"),
        "window_arg_positions": (1,),
        "min_window": 1,
        "return_kind": "series_like",
        "cost": 1.3,
    },
    "ZSCORE": {
        "arg_kinds": ("series_like", "scalar"),
        "window_arg_positions": (1,),
        "min_window": 1,
        "return_kind": "series_like",
        "cost": 1.3,
    },
    "MEAN": {
        "arg_kinds": ("series_like", "scalar"),
        "window_arg_positions": (1,),
        "min_window": 1,
        "return_kind": "series_like",
        "cost": 1.2,
    },
    "STD": {
        "arg_kinds": ("series_like", "scalar"),
        "window_arg_positions": (1,),
        "min_window": 1,
        "return_kind": "series_like",
        "cost": 1.2,
    },
    "MEDIAN": {
        "arg_kinds": ("series_like", "scalar"),
        "window_arg_positions": (1,),
        "min_window": 1,
        "return_kind": "series_like",
        "cost": 1.2,
    },
}


def _build_specs() -> tuple[dict[str, OperatorSpec], dict[str, str]]:
    canonical: dict[str, Callable[..., Any]] = {
        name: fn for name, fn in SAFE_FUNCTIONS.items() if name.isupper()
    }
    specs: dict[str, OperatorSpec] = {}

    for name, fn in canonical.items():
        min_arity, max_arity, variadic, signature = _introspect_arity(fn)
        base = OperatorSpec(
            name=name,
            fn=fn,
            min_arity=min_arity,
            max_arity=max_arity,
            variadic=variadic,
            signature=signature,
        )

        override = _OPERATOR_OVERRIDES.get(name, {})
        specs[name] = OperatorSpec(
            name=name,
            fn=fn,
            min_arity=int(override.get("min_arity", base.min_arity)),
            max_arity=override.get("max_arity", base.max_arity),
            variadic=bool(override.get("variadic", base.variadic)),
            signature=base.signature,
            arg_kinds=tuple(override.get("arg_kinds", base.arg_kinds)),
            return_kind=str(override.get("return_kind", base.return_kind)),
            window_arg_positions=tuple(
                override.get("window_arg_positions", base.window_arg_positions)
            ),
            min_window=int(override.get("min_window", base.min_window)),
            cost=float(override.get("cost", base.cost)),
        )

    aliases: dict[str, str] = {}
    fn_to_canonical: dict[int, str] = {}
    for name, fn in canonical.items():
        fn_to_canonical.setdefault(id(fn), name)

    for alias, fn in SAFE_FUNCTIONS.items():
        key = str(alias).strip().lower()
        if not key:
            continue
        if alias in canonical:
            aliases[key] = alias
            continue
        canonical_name = fn_to_canonical.get(id(fn))
        if canonical_name:
            aliases[key] = canonical_name

    return specs, aliases


@lru_cache(maxsize=1)
def build_default_registry() -> OperatorRegistry:
    specs, aliases = _build_specs()
    return OperatorRegistry(specs=specs, aliases=aliases)
