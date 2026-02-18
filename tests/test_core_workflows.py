"""Regression tests for core deterministic package paths."""

from __future__ import annotations

import pandas as pd

from agent_alpha.compiler import compile_blueprint_to_ast
from agent_alpha.data import load_synthetic_panel
from agent_alpha.evaluator import FactorEvaluator
from agent_alpha.models import FactorBlueprint, FeatureComponentSpec
from agent_alpha.workflow import AgentAlphaWorkflow


def test_load_synthetic_panel_shape_and_schema() -> None:
    panel = load_synthetic_panel(n_days=20, n_tickers=7, seed=42)
    assert isinstance(panel.index, pd.MultiIndex)
    assert list(panel.index.names) == ["datetime", "instrument"]
    assert panel.shape[0] == 20 * 7
    assert {"$open", "$high", "$low", "$close", "$volume"} <= set(panel.columns)


def test_compile_blueprint_to_ast_renders_expression() -> None:
    blueprint = FactorBlueprint(
        hypothesis="Mean-reversion signal from two robust components.",
        components=[
            FeatureComponentSpec(id="c1", feature="rsi", params={"period": 14}),
            FeatureComponentSpec(id="c2", feature="ema", params={"period": 20}),
        ],
        combine={
            "type": "call",
            "op": "ADD",
            "args": [
                {"type": "component", "id": "c1"},
                {"type": "component", "id": "c2"},
            ],
        },
    )
    compiled = compile_blueprint_to_ast(
        blueprint,
        component_columns={"c1": "$cmp_c1", "c2": "$cmp_c2"},
        allowed_windows={1, 5, 10, 20},
    )
    assert compiled.factor_ast["version"] == "1"
    assert "ADD(" in compiled.expression
    assert "$cmp_c1" in compiled.expression
    assert "$cmp_c2" in compiled.expression


def test_compile_blueprint_to_ast_allows_boolean_conditions() -> None:
    blueprint = FactorBlueprint(
        hypothesis="KDJ bullish condition with rising J momentum.",
        components=[
            FeatureComponentSpec(id="k", feature="kdj_k", params={}),
            FeatureComponentSpec(id="d", feature="kdj_d", params={}),
            FeatureComponentSpec(id="j", feature="kdj_j", params={}),
        ],
        combine={
            "type": "call",
            "op": "WHERE",
            "args": [
                {
                    "type": "call",
                    "op": "AND",
                    "args": [
                        {
                            "type": "call",
                            "op": "GT",
                            "args": [
                                {"type": "component", "id": "k"},
                                {"type": "component", "id": "d"},
                            ],
                        },
                        {
                            "type": "call",
                            "op": "GT",
                            "args": [
                                {
                                    "type": "call",
                                    "op": "DELTA",
                                    "args": [
                                        {"type": "component", "id": "j"},
                                        {"type": "const", "value": 1},
                                    ],
                                },
                                {"type": "const", "value": 0},
                            ],
                        },
                    ],
                },
                {"type": "component", "id": "j"},
                {"type": "const", "value": 0},
            ],
        },
    )
    compiled = compile_blueprint_to_ast(
        blueprint,
        component_columns={"k": "$cmp_k", "d": "$cmp_d", "j": "$cmp_j"},
        allowed_windows={1, 5, 10, 20},
    )
    assert "AND(" in compiled.expression
    assert "GT(" in compiled.expression


def test_compile_blueprint_to_ast_allows_whitelisted_price_vars() -> None:
    blueprint = FactorBlueprint(
        hypothesis="Blend close price with RSI component.",
        components=[FeatureComponentSpec(id="r", feature="rsi", params={"period": 14})],
        combine={
            "type": "call",
            "op": "ADD",
            "args": [
                {"type": "var", "name": "$close"},
                {"type": "component", "id": "r"},
            ],
        },
    )
    compiled = compile_blueprint_to_ast(
        blueprint,
        component_columns={"r": "$cmp_r"},
        allowed_var_columns={"$close"},
        allowed_windows={1, 5, 10, 20},
    )
    assert "ADD(" in compiled.expression
    assert "$close" in compiled.expression
    assert "$cmp_r" in compiled.expression


def test_factor_evaluator_returns_metrics_with_and_without_scope() -> None:
    panel = load_synthetic_panel(n_days=30, n_tickers=8, seed=7)
    evaluator = FactorEvaluator(periods=[1, 5])

    factor = evaluator.calculate_factor(panel, {"type": "var", "name": "$close"})
    forward_returns = evaluator.calculate_forward_returns(panel)
    metrics_all = evaluator.calculate_ex_ante_ir(factor, forward_returns)

    assert factor.index.equals(panel.index)
    assert metrics_all["evaluation_scope"]["rows_total"] == len(panel)
    assert metrics_all["evaluation_scope"]["rows_in_scope"] == len(panel)
    assert {"ret_1", "ret_5"} <= set(metrics_all["period_metrics"])

    mask = pd.Series(
        panel.index.get_level_values("instrument").isin({"T0000", "T0001"}),
        index=panel.index,
        name="in_universe",
    )
    metrics_scoped = evaluator.calculate_ex_ante_ir(factor, forward_returns, universe_mask=mask)
    assert (
        metrics_scoped["evaluation_scope"]["rows_in_scope"]
        < metrics_scoped["evaluation_scope"]["rows_total"]
    )


def test_chat_model_kwargs_omit_temperature_for_gpt5_models() -> None:
    kwargs = AgentAlphaWorkflow._chat_model_kwargs("gpt-5-mini", 0.1)
    assert kwargs == {"model": "gpt-5-mini"}


def test_chat_model_kwargs_keep_temperature_for_supported_models() -> None:
    kwargs = AgentAlphaWorkflow._chat_model_kwargs("gpt-4.1-mini", 0.1)
    assert kwargs == {"model": "gpt-4.1-mini", "temperature": 0.1}
