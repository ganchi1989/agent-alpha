from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .ast.codec import ast_summary, encode_factor_ast, render_expression
from .ast.nodes import CallNode, ConstNode, Node, VarNode
from .ast.registry import build_default_registry
from .ast.types import infer_node_type
from .ast.validate import ValidationLimits, normalize_and_validate_ast

from .models import FactorBlueprint

ALLOWED_COMBINE_OPERATORS = {
    "ADD",
    "SUBTRACT",
    "MULTIPLY",
    "DIVIDE",
    "RANK",
    "ZSCORE",
    "DELTA",
    "DELAY",
    "WHERE",
}

_BOOL_RETURN_OPS = {"GT", "LT", "GE", "LE", "EQ", "NE", "AND", "OR"}


@dataclass(frozen=True, slots=True)
class CompileResult:
    factor_ast: dict[str, Any]
    expression: str
    summary: str


def _normalize_op(op: str) -> str:
    return str(op).strip().upper()


def _validate_arity(op: str, argc: int, path: str) -> None:
    if op in {"ADD", "SUBTRACT", "MULTIPLY", "DIVIDE"}:
        if argc < 2:
            raise ValueError(f"{path}: {op} requires at least 2 args, got {argc}")
        return
    if op in {"RANK", "ZSCORE", "DELTA", "DELAY"}:
        if argc not in {1, 2}:
            raise ValueError(f"{path}: {op} requires 1 or 2 args, got {argc}")
        return
    if op == "WHERE":
        if argc != 3:
            raise ValueError(f"{path}: WHERE requires exactly 3 args, got {argc}")
        return
    raise ValueError(f"{path}: Unsupported operator {op!r}")


def _looks_bool_node(node: Node) -> bool:
    if isinstance(node, ConstNode):
        return isinstance(node.value, bool)
    if isinstance(node, CallNode):
        return _normalize_op(node.op) in _BOOL_RETURN_OPS
    return False


def _compile_expr_node(
    node: Mapping[str, Any],
    component_columns: Mapping[str, str],
    path: str,
) -> Node:
    node_type = str(node.get("type", "")).strip().lower()
    if node_type == "component":
        component_id = str(node.get("id", "")).strip()
        if component_id not in component_columns:
            raise ValueError(f"{path}: Unknown component id {component_id!r}")
        return VarNode(component_columns[component_id])

    if node_type == "var":
        name = str(node.get("name", "")).strip()
        if not name:
            raise ValueError(f"{path}: var node requires non-empty 'name'")
        return VarNode(name)

    if node_type == "const":
        value = node.get("value")
        if not isinstance(value, (bool, int, float)) and value is not None:
            raise ValueError(f"{path}: const node value must be bool/int/float/null")
        return ConstNode(value)

    if node_type == "call":
        op = _normalize_op(str(node.get("op", "")))
        if op not in ALLOWED_COMBINE_OPERATORS:
            raise ValueError(f"{path}: Operator {op!r} is not allowed")
        raw_args = node.get("args")
        if not isinstance(raw_args, list):
            raise ValueError(f"{path}: call.args must be a list")
        _validate_arity(op, len(raw_args), path)
        args = tuple(
            _compile_expr_node(arg, component_columns, f"{path}.args[{idx}]")
            for idx, arg in enumerate(raw_args)
        )
        if op == "WHERE" and args:
            cond = args[0]
            # Allow compact numeric threshold-style conditions in blueprints.
            # Convention: numeric condition is interpreted as cond > 0.
            if not _looks_bool_node(cond):
                args = (CallNode(op="GT", args=(cond, ConstNode(0))), args[1], args[2])
        return CallNode(op=op, args=args)

    raise ValueError(f"{path}: unsupported node type {node_type!r}")


def compile_blueprint_to_ast(
    blueprint: FactorBlueprint,
    component_columns: Mapping[str, str],
    *,
    allowed_windows: set[int] | None = None,
    max_depth: int = 64,
    max_nodes: int = 512,
) -> CompileResult:
    if not isinstance(blueprint.combine, Mapping):
        raise ValueError("blueprint.combine must be an object")

    root = _compile_expr_node(blueprint.combine, component_columns, path="combine")
    registry = build_default_registry()

    normalized = normalize_and_validate_ast(
        root,
        registry,
        limits=ValidationLimits(
            max_depth=max_depth,
            max_nodes=max_nodes,
            allowed_columns=set(component_columns.values()),
            allowed_windows=allowed_windows,
        ),
    )
    infer_node_type(normalized, registry)

    factor_ast = encode_factor_ast(normalized)
    expression = render_expression(factor_ast)
    summary = ast_summary(factor_ast)
    return CompileResult(factor_ast=factor_ast, expression=expression, summary=summary)
