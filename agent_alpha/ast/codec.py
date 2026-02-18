from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

from .nodes import CallNode, ConstNode, Node, VarNode, ensure_node, node_depth, node_size, node_to_dict
from .schema import AST_VERSION, validate_factor_ast_payload


def encode_factor_ast(root: Node, version: str = AST_VERSION) -> dict[str, Any]:
    return {"version": str(version), "root": node_to_dict(root)}


def decode_factor_ast(payload: Node | Mapping[str, Any]) -> tuple[str, Node]:
    if isinstance(payload, (VarNode, ConstNode, CallNode)):
        return AST_VERSION, payload

    if not isinstance(payload, Mapping):
        raise TypeError(f"AST payload must be mapping or Node, got {type(payload)!r}")

    if "type" in payload:
        node = ensure_node(payload)
        return AST_VERSION, node

    envelope = validate_factor_ast_payload(payload)
    return str(envelope["version"]), ensure_node(envelope["root"])


def canonical_ast_json(payload: Node | Mapping[str, Any]) -> str:
    version, node = decode_factor_ast(payload)
    envelope = encode_factor_ast(node, version=version)
    return json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_ast_hash(payload: Node | Mapping[str, Any]) -> str:
    canonical = canonical_ast_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _render_const(value: bool | int | float | None) -> str:
    if value is None:
        return "NAN"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "NAN"
        if math.isinf(value):
            return "NAN"
        return f"{value:.12g}"
    return str(value)


def render_expression(root: Node | Mapping[str, Any]) -> str:
    _, node = decode_factor_ast(root)

    def _render(node_: Node) -> str:
        if isinstance(node_, VarNode):
            return node_.name
        if isinstance(node_, ConstNode):
            return _render_const(node_.value)
        args = ", ".join(_render(arg) for arg in node_.args)
        return f"{node_.op}({args})"

    return _render(node)


def ast_summary(root: Node | Mapping[str, Any], max_len: int = 180) -> str:
    _, node = decode_factor_ast(root)
    expr = render_expression(node)
    if len(expr) > max_len:
        expr = expr[: max_len - 3] + "..."
    return f"{expr} [nodes={node_size(node)}, depth={node_depth(node)}]"
