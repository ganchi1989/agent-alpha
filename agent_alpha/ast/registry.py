from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from ..operators.function_lib import SAFE_FUNCTIONS


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    """Immutable descriptor for a single AST operator.

    Instances are created by :func:`build_default_registry` via introspection
    of :data:`~agent_alpha.operators.function_lib.SAFE_FUNCTIONS` and the
    ``_OPERATOR_OVERRIDES`` table.  They are frozen dataclasses so they can be
    safely cached and shared.

    Attributes:
        name: Canonical uppercase operator name (e.g. ``"RANK"``).
        fn: The callable that implements the operator.
        min_arity: Minimum number of positional arguments the operator accepts.
        max_arity: Maximum number of positional arguments, or ``None`` for
            variadic operators.
        variadic: ``True`` when the operator accepts ``*args``.
        signature: String representation of the function signature (from
            :func:`inspect.signature`).
        arg_kinds: Tuple of per-argument kind hints (``"series_like"``,
            ``"bool_like"``, ``"scalar"``, or ``"any"``).
        return_kind: Kind hint for the operator's return value
            (``"series_like"``, ``"bool_like"``, ``"branch_value"``, etc.).
        window_arg_positions: Zero-based positions of arguments that are
            interpreted as rolling-window sizes and must satisfy
            *min_window* and the allowed-window set.
        min_window: Smallest valid window value for this operator.
        cost: Relative computational cost used for expression complexity
            budgeting (arbitrary units; 1.0 = baseline).
    """

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
    """Registry mapping operator names to their :class:`OperatorSpec` descriptors.

    Maintains a canonical set of uppercase operator names (e.g. ``"ADD"``,
    ``"RANK"``) and a case-insensitive alias map for fuzzy name resolution.
    Registry instances are effectively immutable after construction; use
    :func:`build_default_registry` to obtain the shared singleton.

    Typical usage::

        registry = build_default_registry()
        spec = registry.get("RANK")          # raises KeyError if unknown
        canonical = registry.resolve("rank") # → "RANK"
        print(registry.prompt_operator_list())

    Attributes:
        _specs: Internal mapping from canonical operator name to
            :class:`OperatorSpec`.
        _aliases: Internal mapping from lowercased alias/canonical name to the
            canonical uppercase name.
    """

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
    except (ValueError, TypeError):
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
