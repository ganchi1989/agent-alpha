from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .nodes import ConstNode, Node, VarNode
from .registry import OperatorRegistry


class ValueKind(str, Enum):
    SERIES = "series"
    SCALAR = "scalar"
    BOOL_SERIES = "bool_series"
    BOOL_SCALAR = "bool_scalar"
    ANY = "any"


@dataclass(slots=True)
class TypeInferenceError(ValueError):
    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


def _is_series_like(kind: ValueKind) -> bool:
    return kind in {ValueKind.SERIES, ValueKind.BOOL_SERIES}


def _is_bool_like(kind: ValueKind) -> bool:
    return kind in {ValueKind.BOOL_SERIES, ValueKind.BOOL_SCALAR}


def _kind_compatible(expected: str, actual: ValueKind) -> bool:
    expected_key = expected.strip().lower()
    if expected_key in {"", "any"}:
        return True
    if expected_key == "series_like":
        return _is_series_like(actual) or actual in {ValueKind.SCALAR, ValueKind.BOOL_SCALAR}
    if expected_key == "bool_like":
        return _is_bool_like(actual)
    if expected_key == "scalar":
        return actual in {ValueKind.SCALAR, ValueKind.BOOL_SCALAR}
    if expected_key == "series":
        return actual == ValueKind.SERIES
    if expected_key == "bool_series":
        return actual == ValueKind.BOOL_SERIES
    if expected_key == "bool_scalar":
        return actual == ValueKind.BOOL_SCALAR
    return True


def _resolve_return_kind(return_kind: str, arg_kinds: list[ValueKind]) -> ValueKind:
    key = return_kind.strip().lower()
    if key == ValueKind.SERIES.value:
        return ValueKind.SERIES
    if key == ValueKind.SCALAR.value:
        return ValueKind.SCALAR
    if key == ValueKind.BOOL_SERIES.value:
        return ValueKind.BOOL_SERIES
    if key == ValueKind.BOOL_SCALAR.value:
        return ValueKind.BOOL_SCALAR
    if key == "series_like":
        return ValueKind.SERIES if any(_is_series_like(v) for v in arg_kinds) else ValueKind.SCALAR
    if key == "bool_like":
        return (
            ValueKind.BOOL_SERIES
            if any(_is_series_like(v) for v in arg_kinds)
            else ValueKind.BOOL_SCALAR
        )
    if key == "same_as_first":
        return arg_kinds[0] if arg_kinds else ValueKind.SCALAR
    if key == "branch_value":
        branch = arg_kinds[1:3]
        if any(_is_series_like(v) for v in branch):
            return (
                ValueKind.BOOL_SERIES if all(_is_bool_like(v) for v in branch) else ValueKind.SERIES
            )
        return (
            ValueKind.BOOL_SCALAR
            if branch and all(_is_bool_like(v) for v in branch)
            else ValueKind.SCALAR
        )
    return ValueKind.SERIES if any(_is_series_like(v) for v in arg_kinds) else ValueKind.SCALAR


def infer_node_type(node: Node, registry: OperatorRegistry, path: str = "root") -> ValueKind:
    if isinstance(node, VarNode):
        return ValueKind.SERIES

    if isinstance(node, ConstNode):
        if isinstance(node.value, bool):
            return ValueKind.BOOL_SCALAR
        return ValueKind.SCALAR

    spec = registry.get(node.op)
    arg_kinds = [
        infer_node_type(arg, registry, f"{path}.args[{idx}]") for idx, arg in enumerate(node.args)
    ]
    effective_arg_kinds = list(arg_kinds)
    if spec.arg_kinds:
        for idx, actual in enumerate(arg_kinds):
            expected = spec.arg_kinds[idx] if idx < len(spec.arg_kinds) else spec.arg_kinds[-1]
            if not spec.variadic and idx >= len(spec.arg_kinds):
                expected = "any"
            if not _kind_compatible(expected, actual):
                raise TypeInferenceError(
                    path=f"{path}.args[{idx}]",
                    message=f"Operator {spec.name} expects {expected}, got {actual.value}",
                )
            expected_key = expected.strip().lower()
            if expected_key == "series_like" and actual == ValueKind.SCALAR:
                effective_arg_kinds[idx] = ValueKind.SERIES
            if expected_key == "series_like" and actual == ValueKind.BOOL_SCALAR:
                effective_arg_kinds[idx] = ValueKind.BOOL_SERIES

    return _resolve_return_kind(spec.return_kind, effective_arg_kinds)
