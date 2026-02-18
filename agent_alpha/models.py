"""Pydantic schemas shared across workflow, compiler, and evaluation steps."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class HypothesisOutput(BaseModel):
    """Structured output returned by the hypothesis-generation agent.

    Attributes:
        hypothesis: Concise alpha hypothesis statement.
        rationale: Short explanation of why the hypothesis may work.

    Invariants:
        - Both fields must contain at least 8 characters.
    """

    hypothesis: str = Field(min_length=8)
    rationale: str = Field(min_length=8)


class FeatureComponentSpec(BaseModel):
    """Blueprint component descriptor bound to one feature definition.

    Attributes:
        id: Stable component identifier referenced from combine-expression nodes.
        feature: Catalog feature name (normalized to lowercase).
        params: Feature-specific parameter overrides.

    Invariants:
        - `id` matches `^[A-Za-z_][A-Za-z0-9_]*$`.
        - `feature` is normalized to lowercase for registry lookup.

    Example:
        >>> FeatureComponentSpec(id="mom_1", feature="RSI", params={"period": 14})
    """

    id: str = Field(
        min_length=1,
        max_length=48,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
        description="Unique component id used by combine expression nodes.",
    )
    feature: str = Field(
        min_length=1, max_length=64, description="Feature function name from catalog."
    )
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("feature")
    @classmethod
    def normalize_feature(cls, value: str) -> str:
        """Normalize catalog feature names to lowercase."""

        return value.strip().lower()


class FactorBlueprint(BaseModel):
    """LLM-produced factor blueprint consumed by the compiler.

    Attributes:
        hypothesis: Natural-language hypothesis linked to the blueprint.
        components: Feature components available to the combine expression.
        combine: Recursive expression tree using `component`, `const`, `var`,
            and `call` node variants.

    Invariants:
        - At least one component must be present and at most eight are allowed.
        - Component IDs are unique within one blueprint.

    Typical usage:
        Build a `FactorBlueprint` from structured LLM output, then pass it to
        `compile_blueprint_to_ast` together with resolved component columns.
    """

    hypothesis: str = Field(min_length=8)
    components: list[FeatureComponentSpec] = Field(min_length=1, max_length=8)
    combine: dict[str, Any] = Field(
        description=(
            "Recursive expression tree. Node types: "
            "component(id), const(value), var(name), call(op,args)."
        )
    )

    @model_validator(mode="after")
    def validate_component_ids_unique(self) -> "FactorBlueprint":
        """Reject blueprints with duplicate component IDs."""

        seen: set[str] = set()
        for component in self.components:
            if component.id in seen:
                raise ValueError(f"Duplicate component id: {component.id!r}")
            seen.add(component.id)
        return self
