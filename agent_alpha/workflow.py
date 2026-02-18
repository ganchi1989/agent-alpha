"""LangGraph workflow that turns a research goal into an evaluated factor AST."""

from __future__ import annotations

import json
from typing import Any, TypedDict

import pandas as pd
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from .compiler import ALLOWED_COMBINE_OPERATORS, compile_blueprint_to_ast
from .config import FactorEngineConfig
from .evaluator import FactorEvaluator
from .feature_engine import compute_blueprint_components, feature_catalog_for_prompt
from .models import FactorBlueprint, HypothesisOutput


@tool
def list_feature_catalog() -> str:
    """List available robust feature components and default params."""
    return json.dumps(feature_catalog_for_prompt(), ensure_ascii=False)


@tool
def list_combine_operators() -> str:
    """List allowed combine operators for AST compilation."""
    return json.dumps(sorted(ALLOWED_COMBINE_OPERATORS), ensure_ascii=False)


class AgentAlphaState(TypedDict, total=False):
    """Mutable state container passed between LangGraph nodes.

    The workflow progressively fills this mapping with hypothesis text,
    blueprint output, compiled AST artifacts, and evaluation metrics.
    """

    user_goal: str
    panel: pd.DataFrame
    universe_mask: pd.DataFrame | pd.Series | None
    hypothesis: str
    rationale: str
    blueprint: FactorBlueprint
    blueprint_json: dict[str, Any]
    factor_ast: dict[str, Any]
    ast_expression: str
    ast_summary: str
    metrics: dict[str, Any]
    factor: pd.Series
    component_columns: dict[str, str]
    attempts: int
    max_attempts: int
    error: str | None


class AgentAlphaWorkflow:
    """High-level orchestration for the agent-alpha factor ideation pipeline.

    Purpose:
        Coordinate three stages: hypothesis generation, blueprint generation,
        and deterministic compile/evaluate execution.

    Key attributes:
        model_name: Chat model name used for both agent nodes.
        temperature: Sampling temperature passed to the chat model.
        periods: Forward-return horizons used by the evaluator.
        allowed_windows: Rolling windows accepted by AST validation.
        max_attempts: Retry budget for blueprint regeneration after failures.
        graph: Compiled LangGraph state machine used in `run`.

    Invariants:
        - Deterministic compile/evaluate stage never executes without a
          structured blueprint in state.
        - Retries are bounded by `max_attempts`.

    Example:
        >>> workflow = AgentAlphaWorkflow(model_name="gpt-5-mini")
        >>> state = workflow.run(user_goal="Mean reversion from OHLCV", panel=panel)
        >>> state["metrics"]["rank_ic"]
    """

    def __init__(
        self,
        model_name: str = "gpt-5-mini",
        temperature: float = 0.1,
        periods: tuple[int, ...] = (1, 5, 10),
        allowed_windows: tuple[int, ...] = (1, 2, 3, 5, 10, 20, 30, 60, 120, 252),
        max_attempts: int = 2,
    ):
        self.model_name = model_name
        self.temperature = float(temperature)
        self.periods = tuple(int(p) for p in periods)
        self.allowed_windows = tuple(sorted({int(w) for w in allowed_windows}))
        self.max_attempts = int(max_attempts)

        self.model = ChatOpenAI(model=self.model_name, temperature=self.temperature)
        self.hypothesis_agent = create_agent(
            model=self.model,
            tools=[],
            response_format=HypothesisOutput,
            system_prompt=(
                "You are a quantitative researcher. "
                "Propose concise, testable factor hypotheses from OHLCV data."
            ),
        )
        self.blueprint_agent = create_agent(
            model=self.model,
            tools=[list_feature_catalog, list_combine_operators],
            response_format=FactorBlueprint,
            system_prompt=(
                "You design robust factor blueprints. "
                "Use only feature components from the catalog and combine them with the allowed operators. "
                "Keep expressions compact and readable."
            ),
        )
        self.graph = self._build_graph()

    @staticmethod
    def _extract_structured(result: dict[str, Any], schema: type[BaseModel]) -> BaseModel:
        structured = result.get("structured_response")
        if structured is None:
            raise ValueError("Agent response missing structured_response")
        if isinstance(structured, schema):
            return structured
        return schema.model_validate(structured)

    def _node_hypothesis(self, state: AgentAlphaState) -> dict[str, Any]:
        prompt = (
            "User goal:\n"
            f"{state['user_goal']}\n\n"
            "Return one concise alpha hypothesis and rationale."
        )
        try:
            result = self.hypothesis_agent.invoke(
                {"messages": [{"role": "user", "content": prompt}]}
            )
            output = self._extract_structured(result, HypothesisOutput)
            return {
                "hypothesis": output.hypothesis,
                "rationale": output.rationale,
                "error": None,
            }
        except Exception as exc:
            fallback = str(
                state.get("user_goal", "Robust cross-sectional alpha from OHLCV")
            ).strip()
            fallback = " ".join(fallback.split())
            if len(fallback) < 8:
                fallback = "Robust cross-sectional alpha factor from OHLCV."
            return {
                "hypothesis": fallback,
                "rationale": "Fallback hypothesis generated after structured-output validation failure.",
                "error": f"hypothesis_generation_failed: {exc}",
            }

    def _node_blueprint(self, state: AgentAlphaState) -> dict[str, Any]:
        if "hypothesis" not in state:
            attempts = int(state.get("attempts", 0)) + 1
            return {"attempts": attempts, "error": state.get("error") or "missing_hypothesis"}
        repair = state.get("error") or "none"
        prompt = (
            "Hypothesis:\n"
            f"{state['hypothesis']}\n\n"
            f"Rationale:\n{state.get('rationale', 'none')}\n\n"
            "Blueprint requirements:\n"
            "1) Choose 2-5 robust components from the feature catalog.\n"
            "2) Use this combine expression grammar:\n"
            "   - {'type':'component','id':'<component_id>'}\n"
            "   - {'type':'const','value': number|bool|null}\n"
            "   - {'type':'call','op':'OP','args':[...]} where OP is in list_combine_operators.\n"
            "   - For WHERE, the first arg may be a numeric score; compiler interprets it as (score > 0).\n"
            "3) Keep trees shallow and avoid unnecessary nesting.\n"
            "4) Use only deterministic numeric logic.\n\n"
            "Previous compile/eval error (if retry):\n"
            f"{repair}\n"
        )
        try:
            result = self.blueprint_agent.invoke(
                {"messages": [{"role": "user", "content": prompt}]}
            )
            blueprint = self._extract_structured(result, FactorBlueprint)
            return {
                "blueprint": blueprint,
                "blueprint_json": blueprint.model_dump(),
                "error": None,
            }
        except Exception as exc:
            attempts = int(state.get("attempts", 0)) + 1
            return {"attempts": attempts, "error": f"blueprint_generation_failed: {exc}"}

    def _node_compile_and_evaluate(self, state: AgentAlphaState) -> dict[str, Any]:
        if "blueprint" not in state:
            return {
                "attempts": int(state.get("attempts", 0)),
                "error": state.get("error") or "missing_blueprint",
            }

        panel = state["panel"]
        universe_mask = state.get("universe_mask")
        blueprint = state["blueprint"]
        try:
            augmented_panel, component_columns = compute_blueprint_components(
                panel, blueprint.components
            )
            compiled = compile_blueprint_to_ast(
                blueprint,
                component_columns,
                allowed_windows=set(self.allowed_windows),
            )

            evaluator = FactorEvaluator(
                periods=list(self.periods),
                min_cross_section=5,
                engine_config=FactorEngineConfig(
                    max_ast_depth=64,
                    max_ast_nodes=512,
                    allowed_windows=list(self.allowed_windows),
                ),
            )
            factor = evaluator.calculate_factor(augmented_panel, compiled.factor_ast)
            forward_returns = evaluator.calculate_forward_returns(panel, periods=list(self.periods))
            metrics = evaluator.calculate_ex_ante_ir(
                factor,
                forward_returns,
                universe_mask=universe_mask,
            )

            return {
                "component_columns": component_columns,
                "factor_ast": compiled.factor_ast,
                "ast_expression": compiled.expression,
                "ast_summary": compiled.summary,
                "factor": factor,
                "metrics": metrics,
                "error": None,
            }
        except Exception as exc:
            attempts = int(state.get("attempts", 0)) + 1
            return {"attempts": attempts, "error": str(exc)}

    def _route_after_compile(self, state: AgentAlphaState) -> str:
        if state.get("error"):
            if int(state.get("attempts", 0)) < int(state.get("max_attempts", self.max_attempts)):
                return "retry"
        return "done"

    def _build_graph(self):
        graph = StateGraph(AgentAlphaState)
        graph.add_node("hypothesis", self._node_hypothesis)
        graph.add_node("blueprint", self._node_blueprint)
        graph.add_node("compile_eval", self._node_compile_and_evaluate)

        graph.add_edge(START, "hypothesis")
        graph.add_edge("hypothesis", "blueprint")
        graph.add_edge("blueprint", "compile_eval")
        graph.add_conditional_edges(
            "compile_eval",
            self._route_after_compile,
            {"retry": "blueprint", "done": END},
        )
        return graph.compile()

    def run(
        self,
        user_goal: str,
        panel: pd.DataFrame,
        max_attempts: int | None = None,
        universe_mask: pd.DataFrame | pd.Series | None = None,
    ) -> AgentAlphaState:
        """Execute the full workflow and return the final state.

        Args:
            user_goal: Research objective that seeds hypothesis generation.
            panel: Input OHLCV panel indexed by `(datetime, instrument)`.
            max_attempts: Optional override for blueprint retry limit.
            universe_mask: Optional evaluation-scope mask for metric reporting.

        Returns:
            Final `AgentAlphaState` with hypothesis, blueprint, AST, and metrics.
        """

        initial_state: AgentAlphaState = {
            "user_goal": user_goal,
            "panel": panel,
            "universe_mask": universe_mask,
            "attempts": 0,
            "max_attempts": int(max_attempts or self.max_attempts),
            "error": None,
        }
        final_state = self.graph.invoke(initial_state)
        return final_state
