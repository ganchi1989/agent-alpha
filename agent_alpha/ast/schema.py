from __future__ import annotations

from collections.abc import Mapping
from typing import Any

AST_VERSION = "1"

FACTOR_AST_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "FactorASTEnvelope",
    "type": "object",
    "additionalProperties": False,
    "required": ["version", "root"],
    "properties": {
        "version": {"const": AST_VERSION},
        "root": {"$ref": "#/$defs/node"},
    },
    "$defs": {
        "node": {
            "oneOf": [
                {"$ref": "#/$defs/var"},
                {"$ref": "#/$defs/const"},
                {"$ref": "#/$defs/call"},
            ]
        },
        "var": {
            "type": "object",
            "additionalProperties": False,
            "required": ["type", "name"],
            "properties": {
                "type": {"const": "var"},
                "name": {"type": "string", "minLength": 1},
            },
        },
        "const": {
            "type": "object",
            "additionalProperties": False,
            "required": ["type", "value"],
            "properties": {
                "type": {"const": "const"},
                "value": {"type": ["number", "boolean", "null"]},
            },
        },
        "call": {
            "type": "object",
            "additionalProperties": False,
            "required": ["type", "op", "args"],
            "properties": {
                "type": {"const": "call"},
                "op": {"type": "string", "minLength": 1},
                "args": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/node"},
                },
            },
        },
    },
}


def _parse_scalar_literal(value: str) -> bool | int | float | None | str:
    text = value.strip()
    lower = text.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"null", "none", "nan"}:
        return None
    try:
        if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
            return int(text)
        return float(text)
    except Exception:
        return text


def _coerce_node_payload(node: Any) -> dict[str, Any]:
    if isinstance(node, Mapping):
        payload = dict(node)
        node_type = str(payload.get("type", "")).strip().lower()
        if not node_type:
            if "op" in payload or "args" in payload:
                node_type = "call"
            elif "name" in payload or "var" in payload or "column" in payload:
                node_type = "var"
            elif "value" in payload:
                node_type = "const"

        if node_type == "call":
            op = payload.get(
                "op", payload.get("name", payload.get("fn", payload.get("function", "")))
            )
            raw_args = payload.get("args", payload.get("arguments", payload.get("params", [])))
            if isinstance(raw_args, tuple):
                raw_args = list(raw_args)
            if not isinstance(raw_args, list):
                raw_args = [raw_args]
            out = {
                "type": "call",
                "op": str(op).strip(),
                "args": [_coerce_node_payload(arg) for arg in raw_args],
            }
            used = {"type", "op", "args", "name", "fn", "function", "arguments", "params"}
            for key, value in payload.items():
                if key not in used:
                    out[key] = value
            return out

        if node_type == "var":
            name = payload.get("name", payload.get("var", payload.get("column", "")))
            out = {"type": "var", "name": str(name).strip()}
            used = {"type", "name", "var", "column"}
            for key, value in payload.items():
                if key not in used:
                    out[key] = value
            return out

        if node_type == "const":
            value = payload.get("value")
            if isinstance(value, str):
                parsed = _parse_scalar_literal(value)
                value = parsed
            out = {"type": "const", "value": value}
            used = {"type", "value"}
            for key, v in payload.items():
                if key not in used:
                    out[key] = v
            return out

        return payload

    if isinstance(node, (bool, int, float)) or node is None:
        return {"type": "const", "value": node}

    if isinstance(node, str):
        parsed = _parse_scalar_literal(node)
        if isinstance(parsed, str):
            text = parsed.strip()
            return {"type": "var", "name": text}
        return {"type": "const", "value": parsed}

    raise ValueError(f"Unsupported AST node payload type: {type(node)!r}")


def _validate_node(node: Any, path: str) -> dict[str, Any]:
    node = _coerce_node_payload(node)
    if not isinstance(node, Mapping):
        raise ValueError(f"{path} must be an object")

    node_type = str(node.get("type", "")).strip().lower()
    if node_type == "var":
        allowed = {"type", "name"}
        extra = set(node.keys()) - allowed
        if extra:
            raise ValueError(f"{path} has unknown fields: {sorted(extra)}")
        name = node.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{path}.name must be a non-empty string")
        return {"type": "var", "name": name.strip()}

    if node_type == "const":
        allowed = {"type", "value"}
        extra = set(node.keys()) - allowed
        if extra:
            raise ValueError(f"{path} has unknown fields: {sorted(extra)}")
        value = node.get("value")
        if not isinstance(value, (bool, int, float)) and value is not None:
            raise ValueError(f"{path}.value must be bool/int/float/null")
        return {"type": "const", "value": value}

    if node_type == "call":
        allowed = {"type", "op", "args"}
        extra = set(node.keys()) - allowed
        if extra:
            raise ValueError(f"{path} has unknown fields: {sorted(extra)}")
        op = node.get("op")
        args = node.get("args")
        if not isinstance(op, str) or not op.strip():
            raise ValueError(f"{path}.op must be a non-empty string")
        if not isinstance(args, list):
            raise ValueError(f"{path}.args must be an array")
        return {
            "type": "call",
            "op": op.strip(),
            "args": [_validate_node(arg, f"{path}.args[{idx}]") for idx, arg in enumerate(args)],
        }

    raise ValueError(f"{path}.type must be one of var|const|call")


def _looks_like_node(payload: Mapping[str, Any]) -> bool:
    keys = set(payload.keys())
    if "type" in keys:
        return True
    if "op" in keys or "args" in keys:
        return True
    if "name" in keys or "var" in keys or "column" in keys:
        return True
    if "value" in keys:
        return True
    return False


def validate_factor_ast_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"AST payload must be an object, got {type(payload)!r}")

    root_payload: Mapping[str, Any] = payload
    if "factor_ast" in payload and isinstance(payload.get("factor_ast"), Mapping):
        root_payload = payload["factor_ast"]

    envelope = dict(root_payload)
    if "root" not in envelope and _looks_like_node(envelope):
        version = envelope.pop("version", AST_VERSION)
        envelope = {"version": version, "root": envelope}

    allowed = {"version", "root"}
    extra = set(envelope.keys()) - allowed
    if extra:
        raise ValueError(f"AST envelope has unknown fields: {sorted(extra)}")

    version = envelope.get("version")
    if str(version) != AST_VERSION:
        raise ValueError(f"AST envelope version must be {AST_VERSION!r}, got {version!r}")

    if "root" not in envelope:
        raise ValueError("AST envelope requires field 'root'")

    root = _validate_node(envelope["root"], "root")
    return {"version": AST_VERSION, "root": root}
