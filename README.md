# agent-alpha

`agent-alpha` is a concise LangGraph workflow for robust factor ideation:

1. Generate a hypothesis with an LLM agent.
2. Generate a structured blueprint (well-known feature components + combine tree).
3. Compile blueprint to AST deterministically with a restricted operator set:
   - `ADD`, `SUBTRACT`, `MULTIPLY`, `DIVIDE`
   - `RANK`, `ZSCORE`, `DELTA`, `DELAY`
   - optional `WHERE`
4. Evaluate with agent-alpha's deterministic local AST evaluator.

## Requirements

- Python `>=3.10`
- OpenAI API key (for `gpt-5-mini`)

## Install

From PyPI:

```bash
pip install agent-alpha
```

From source:

```bash
pip install -e .
```

## Quick Start

```python
from agent_alpha.runner import load_synthetic_panel
from agent_alpha.workflow import AgentAlphaWorkflow

panel = load_synthetic_panel(n_days=220, n_tickers=50)

workflow = AgentAlphaWorkflow(
    model_name="gpt-5-mini",
    periods=(1, 5, 10),
)

result = workflow.run(
    user_goal="Generate a robust mean-reversion alpha hypothesis from OHLCV",
    panel=panel,
    max_attempts=2,
    # Optional: apply universe filtering only for RankIC/ICIR metrics.
    # universe_mask=universe_df,  # columns: date,ticker,in_universe (or static ticker list)
)

print(result["hypothesis"])
print(result["ast_expression"])
print(result["metrics"])
```

`panel` should contain all available price rows for all tickers you want to score.
If you pass `universe_mask`, factor/feature computation still runs on the full panel,
while RankIC/ICIR/ex-ante IR are computed only on the masked universe rows.

## CLI

```bash
agent-alpha-run --goal "Generate a robust mean-reversion alpha hypothesis from OHLCV"
```

Common options:

- `--model gpt-5-mini`
- `--temperature 0.1` (for models that support temperature)
- `--max-attempts 2`
- `--n-days 220`
- `--n-tickers 50`

## Development

```bash
pip install -e ".[dev]"
python -m ruff check .
python -m ruff format --check .
python -m mypy agent_alpha
python -m pytest
python -m build
python -m twine check dist/*
```

## Notebook

See `notebooks/01_agent_alpha_walkthrough.ipynb`.

## License

This project is licensed under the MIT License. See `LICENSE`.
