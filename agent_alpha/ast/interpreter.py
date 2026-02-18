from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from .nodes import CallNode, ConstNode, Node, VarNode
from .registry import OperatorRegistry


@dataclass(slots=True)
class ASTInterpretationError(RuntimeError):
    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


class ASTInterpreter:
    def __init__(self, registry: OperatorRegistry):
        self.registry = registry

    def _resolve_var(self, name: str, variables: Mapping[str, Any], path: str) -> Any:
        plain = name.lstrip("$")
        for key in (
            name,
            plain,
            plain.lower(),
            plain.upper(),
            plain.capitalize(),
            f"${plain}",
            f"${plain.lower()}",
            f"${plain.upper()}",
        ):
            if key in variables:
                return variables[key]
        raise ASTInterpretationError(path, f"Unknown variable: {name!r}")

    @staticmethod
    def _context_index(variables: Mapping[str, Any]) -> pd.Index | pd.MultiIndex | None:
        for value in variables.values():
            if isinstance(value, pd.Series):
                return value.index
            if isinstance(value, pd.DataFrame) and value.shape[1] == 1:
                return value.iloc[:, 0].index
        return None

    @staticmethod
    def _arg_target_index(
        values: list[Any], fallback: pd.Index | pd.MultiIndex | None
    ) -> pd.Index | pd.MultiIndex | None:
        for value in values:
            if isinstance(value, pd.Series):
                return value.index
            if isinstance(value, pd.DataFrame) and value.shape[1] == 1:
                return value.iloc[:, 0].index
        return fallback

    @staticmethod
    def _broadcast_scalar(value: Any, index: pd.Index | pd.MultiIndex | None) -> Any:
        if index is None:
            return value
        if isinstance(value, pd.DataFrame):
            if value.shape[1] != 1:
                return value
            value = value.iloc[:, 0]
        if isinstance(value, pd.Series):
            if value.index.equals(index):
                return value
            return value.reindex(index)
        if value is None or pd.api.types.is_scalar(value):
            return pd.Series(value, index=index)
        return value

    @staticmethod
    def _coerce_value_for_expected_kind(
        value: Any,
        expected_kind: str,
        index: pd.Index | pd.MultiIndex | None,
    ) -> Any:
        key = expected_kind.strip().lower()
        if key in {"series", "series_like", "bool_series"}:
            return ASTInterpreter._broadcast_scalar(value, index)
        return value

    def evaluate(self, node: Node, variables: Mapping[str, Any], path: str = "root") -> Any:
        if isinstance(node, VarNode):
            return self._resolve_var(node.name, variables, path)
        if isinstance(node, ConstNode):
            return node.value
        if not isinstance(node, CallNode):
            raise ASTInterpretationError(path, f"Unsupported node: {type(node)!r}")

        spec = self.registry.get(node.op)
        values = [
            self.evaluate(arg, variables, f"{path}.args[{idx}]")
            for idx, arg in enumerate(node.args)
        ]
        context_index = self._context_index(variables)
        target_index = self._arg_target_index(values, context_index)
        if spec.arg_kinds:
            coerced: list[Any] = []
            for idx, value in enumerate(values):
                if idx < len(spec.arg_kinds):
                    expected = spec.arg_kinds[idx]
                elif spec.variadic and len(spec.arg_kinds) > 0:
                    expected = spec.arg_kinds[-1]
                else:
                    expected = "any"
                coerced.append(self._coerce_value_for_expected_kind(value, expected, target_index))
            values = coerced

        try:
            return spec.fn(*values)
        except Exception as exc:
            raise ASTInterpretationError(path, f"{node.op} failed: {exc}") from exc
