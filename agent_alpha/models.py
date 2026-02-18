from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class HypothesisOutput(BaseModel):
    hypothesis: str = Field(min_length=8)
    rationale: str = Field(min_length=8)


class FeatureComponentSpec(BaseModel):
    id: str = Field(
        min_length=1,
        max_length=48,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
        description="Unique component id used by combine expression nodes.",
    )
    feature: str = Field(min_length=1, max_length=64, description="Feature function name from catalog.")
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("feature")
    @classmethod
    def normalize_feature(cls, value: str) -> str:
        return value.strip().lower()


class FactorBlueprint(BaseModel):
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
        seen: set[str] = set()
        for component in self.components:
            if component.id in seen:
                raise ValueError(f"Duplicate component id: {component.id!r}")
            seen.add(component.id)
        return self
