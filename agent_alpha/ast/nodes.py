from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class VarNode:
    name: str


@dataclass(frozen=True, slots=True)
class ConstNode:
    value: bool | int | float | None


@dataclass(frozen=True, slots=True)
class CallNode:
    op: str
    args: tuple[Node, ...]


Node = VarNode | ConstNode | CallNode


def ensure_node(payload: Node | Mapping[str, Any]) -> Node:
    if isinstance(payload, (VarNode, ConstNode, CallNode)):
        return payload
    if not isinstance(payload, Mapping):
        raise TypeError(f"AST node must be mapping or Node, got {type(payload)!r}")

    node_type = str(payload.get("type", "")).strip().lower()
    if node_type == "var":
        name = payload.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("var node requires non-empty string field 'name'")
        return VarNode(name=name.strip())
    if node_type == "const":
        value = payload.get("value")
        if not isinstance(value, (bool, int, float)) and value is not None:
            raise ValueError("const node field 'value' must be bool/int/float/null")
        return ConstNode(value=value)
    if node_type == "call":
        op = payload.get("op")
        args = payload.get("args")
        if not isinstance(op, str) or not op.strip():
            raise ValueError("call node requires non-empty string field 'op'")
        if not isinstance(args, list):
            raise ValueError("call node requires list field 'args'")
        return CallNode(op=op.strip(), args=tuple(ensure_node(arg) for arg in args))

    raise ValueError(f"Unknown AST node type: {node_type!r}")


def node_to_dict(node: Node) -> dict[str, Any]:
    if isinstance(node, VarNode):
        return {"type": "var", "name": node.name}
    if isinstance(node, ConstNode):
        return {"type": "const", "value": node.value}
    return {
        "type": "call",
        "op": node.op,
        "args": [node_to_dict(arg) for arg in node.args],
    }


def node_size(node: Node) -> int:
    if isinstance(node, CallNode):
        return 1 + sum(node_size(arg) for arg in node.args)
    return 1


def node_depth(node: Node) -> int:
    if isinstance(node, CallNode) and node.args:
        return 1 + max(node_depth(arg) for arg in node.args)
    return 1
