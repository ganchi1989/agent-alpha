"""Public package API for agent-alpha."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("agent-alpha")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

from .models import FactorBlueprint, FeatureComponentSpec, HypothesisOutput
from .runner import load_synthetic_panel, run_agent_alpha
from .workflow import AgentAlphaWorkflow

__all__ = [
    "__version__",
    "AgentAlphaWorkflow",
    "FactorBlueprint",
    "FeatureComponentSpec",
    "HypothesisOutput",
    "load_synthetic_panel",
    "run_agent_alpha",
]
