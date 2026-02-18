from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FactorEngineConfig:
    max_ast_depth: int = 32
    max_ast_nodes: int = 256
    allowed_windows: list[int] = field(default_factory=lambda: [1, 2, 3, 5, 10, 20, 30, 60, 120, 252])
    ast_version: str = "1"
