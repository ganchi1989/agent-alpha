from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .nodes import CallNode, ConstNode, Node, VarNode, ensure_node
from .registry import OperatorRegistry


@dataclass(slots=True)
class ValidationError(ValueError):
    """Raised when AST validation fails.

    Attributes:
        code: Short machine-readable error code (e.g. ``"unknown_operator"``,
            ``"max_depth"``, ``"arity"``).
        path: Dot-separated path to the offending node within the AST tree
            (e.g. ``"root.args[0].args[1]"``).
        message: Human-readable description of the violation.
    """

    code: str
    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.code} at {self.path}: {self.message}"


@dataclass(slots=True)
class ValidationLimits:
    """Structural constraints enforced during AST normalization and validation.

    All limits are checked during a single depth-first walk of the AST by
    :func:`normalize_and_validate_ast`.  Violating any limit raises
    :class:`ValidationError`.

    Attributes:
        max_depth: Maximum allowed nesting depth (root counts as depth 1).
            Default: 32.
        max_nodes: Maximum total node count across the entire tree.
            Default: 256.
        allowed_columns: If set, only variable references whose names appear in
            this set are permitted.  Names are normalized to their canonical
            ``$``-prefixed form before comparison.
        allowed_windows: If set, rolling-window arguments must belong to this
            set (or be coerced to the nearest member when
            *coerce_windows_to_allowed* is ``True``).
        coerce_windows_to_allowed: When ``True`` and *allowed_windows* is
            provided, out-of-set window values are silently rounded to the
            nearest allowed value instead of raising.  Default: ``True``.
    """

    max_depth: int = 32
    max_nodes: int = 256
    allowed_columns: set[str] | None = None
    allowed_windows: set[int] | None = None
    coerce_windows_to_allowed: bool = True


def _normalize_columns(columns: set[str] | None) -> dict[str, str]:
    if not columns:
        return {}
    out: dict[str, str] = {}
    for col in columns:
        canonical = str(col)
        plain = canonical.lstrip("$")
        for key in {
            canonical,
            canonical.lower(),
            canonical.upper(),
            plain,
            plain.lower(),
            plain.upper(),
            f"${plain}",
            f"${plain.lower()}",
            f"${plain.upper()}",
        }:
            out[key] = canonical
    return out


def _coerce_window(node: Node, path: str) -> int:
    if not isinstance(node, ConstNode):
        raise ValidationError("invalid_window", path, "window argument must be a const node")
    value = node.value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError("invalid_window", path, "window value must be numeric")
    coerced = int(float(value))
    if float(coerced) != float(value):
        raise ValidationError("invalid_window", path, f"window must be an integer, got {value!r}")
    return coerced


def _nearest_allowed_window(window: int, allowed_windows: set[int]) -> int:
    if not allowed_windows:
        return window
    return min(
        allowed_windows, key=lambda candidate: (abs(int(candidate) - int(window)), int(candidate))
    )


def normalize_and_validate_ast(
    payload: Node | Mapping[str, object],
    registry: OperatorRegistry,
    limits: ValidationLimits | None = None,
) -> Node:
    lim = limits or ValidationLimits()
    column_lookup = _normalize_columns(lim.allowed_columns)
    root = ensure_node(payload)

    counter = {"n": 0}

    def _walk(node: Node, path: str, depth: int) -> Node:
        if depth > lim.max_depth:
            raise ValidationError("max_depth", path, f"depth {depth} exceeds {lim.max_depth}")

        counter["n"] += 1
        if counter["n"] > lim.max_nodes:
            raise ValidationError("max_nodes", path, f"node count exceeds {lim.max_nodes}")

        if isinstance(node, VarNode):
            if not column_lookup:
                return node
            key = node.name if node.name in column_lookup else node.name.lower()
            canonical = column_lookup.get(key)
            if canonical is None:
                raise ValidationError(
                    "unknown_column", path, f"column {node.name!r} is not allowed"
                )
            return VarNode(canonical)

        if isinstance(node, ConstNode):
            return node

        try:
            canonical_op = registry.resolve(node.op)
        except KeyError as exc:
            raise ValidationError("unknown_operator", path, str(exc)) from exc

        spec = registry.get(canonical_op)
        args = [_walk(arg, f"{path}.args[{idx}]", depth + 1) for idx, arg in enumerate(node.args)]
        argc = len(args)
        if argc < spec.min_arity:
            raise ValidationError(
                "arity", path, f"{canonical_op} expects at least {spec.min_arity} args, got {argc}"
            )
        if spec.max_arity is not None and argc > spec.max_arity:
            raise ValidationError(
                "arity", path, f"{canonical_op} expects at most {spec.max_arity} args, got {argc}"
            )

        for arg_pos in spec.window_arg_positions:
            if arg_pos >= argc:
                continue
            win_path = f"{path}.args[{arg_pos}]"
            window = _coerce_window(args[arg_pos], win_path)
            if window < spec.min_window:
                raise ValidationError(
                    "window_range",
                    win_path,
                    f"{canonical_op} requires window >= {spec.min_window}, got {window}",
                )
            if lim.allowed_windows is not None and window not in lim.allowed_windows:
                if lim.coerce_windows_to_allowed and lim.allowed_windows:
                    window = _nearest_allowed_window(window, lim.allowed_windows)
                else:
                    raise ValidationError(
                        "window_allowed",
                        win_path,
                        f"window {window} is not in allowed set {sorted(lim.allowed_windows)}",
                    )
            args[arg_pos] = ConstNode(window)

        return CallNode(op=canonical_op, args=tuple(args))

    return _walk(root, "root", 1)
