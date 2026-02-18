"""Configuration objects for AST validation and evaluation limits."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FactorEngineConfig:
    """Runtime safeguards for factor AST evaluation.

    Attributes:
        max_ast_depth: Maximum allowed expression nesting depth.
        max_ast_nodes: Maximum number of nodes in a normalized AST.
        allowed_windows: Allowed rolling window sizes for time-series operators.
        ast_version: Expected AST envelope version.
    """

    max_ast_depth: int = 32
    max_ast_nodes: int = 256
    allowed_windows: list[int] = field(
        default_factory=lambda: [1, 2, 3, 5, 10, 20, 30, 60, 120, 252]
    )
    ast_version: str = "1"
