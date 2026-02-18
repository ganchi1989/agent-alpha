from __future__ import annotations

import argparse
import json
import os
from typing import Any

import pandas as pd

from .data import load_synthetic_panel
from .workflow import AgentAlphaWorkflow


def run_agent_alpha(
    user_goal: str,
    *,
    api_key: str | None = None,
    model_name: str = "gpt-5-mini",
    n_days: int = 220,
    n_tickers: int = 50,
    max_attempts: int = 2,
    universe_mask: pd.DataFrame | pd.Series | None = None,
) -> dict[str, Any]:
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    panel = load_synthetic_panel(n_days=n_days, n_tickers=n_tickers, seed=7)
    workflow = AgentAlphaWorkflow(model_name=model_name)
    state = workflow.run(
        user_goal=user_goal,
        panel=panel,
        max_attempts=max_attempts,
        universe_mask=universe_mask,
    )
    return dict(state)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run agent-alpha blueprint to AST workflow.")
    parser.add_argument("--goal", required=True, help="User goal / research objective")
    parser.add_argument("--model", default="gpt-5-mini", help="Model name (default: gpt-5-mini)")
    parser.add_argument("--max-attempts", type=int, default=2, help="Blueprint repair retries")
    parser.add_argument("--n-days", type=int, default=220, help="Synthetic panel business days")
    parser.add_argument("--n-tickers", type=int, default=50, help="Synthetic panel ticker count")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    output = run_agent_alpha(
        user_goal=args.goal,
        model_name=args.model,
        max_attempts=args.max_attempts,
        n_days=args.n_days,
        n_tickers=args.n_tickers,
    )

    printable = {
        "hypothesis": output.get("hypothesis"),
        "rationale": output.get("rationale"),
        "blueprint_json": output.get("blueprint_json"),
        "ast_expression": output.get("ast_expression"),
        "ast_summary": output.get("ast_summary"),
        "metrics": output.get("metrics"),
        "error": output.get("error"),
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
