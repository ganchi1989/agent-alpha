"""Programmatic and CLI entry points for running the full workflow."""

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
    temperature: float | None = None,
    n_days: int = 220,
    n_tickers: int = 50,
    max_attempts: int = 2,
    universe_mask: pd.DataFrame | pd.Series | None = None,
) -> dict[str, Any]:
    """Run the end-to-end workflow on a generated synthetic panel.

    Args:
        user_goal: Natural-language research objective.
        api_key: Optional OpenAI API key injected into `OPENAI_API_KEY`.
        model_name: Chat model name used by the workflow agents.
        temperature: Optional sampling temperature for models that support it.
        n_days: Number of business days in synthetic panel generation.
        n_tickers: Number of synthetic instruments.
        max_attempts: Maximum blueprint repair attempts.
        universe_mask: Optional evaluation scope mask for metrics.

    Returns:
        Final workflow state as a plain dictionary.
    """

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    panel = load_synthetic_panel(n_days=n_days, n_tickers=n_tickers, seed=7)
    workflow = AgentAlphaWorkflow(model_name=model_name, temperature=temperature)
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
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional sampling temperature (ignored by gpt-5 models).",
    )
    parser.add_argument("--max-attempts", type=int, default=2, help="Blueprint repair retries")
    parser.add_argument("--n-days", type=int, default=220, help="Synthetic panel business days")
    parser.add_argument("--n-tickers", type=int, default=50, help="Synthetic panel ticker count")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint that executes the workflow and prints JSON output."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    output = run_agent_alpha(
        user_goal=args.goal,
        model_name=args.model,
        temperature=args.temperature,
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
