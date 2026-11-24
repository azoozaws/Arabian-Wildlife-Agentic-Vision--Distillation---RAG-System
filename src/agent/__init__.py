"""Agent state, evaluation gate, and lazy LangGraph exports (PEP 562)."""

from __future__ import annotations

from typing import Any

from app.src.agent.agent_state import (
    MetricsSnapshot,
    RunMode,
    WildlifeAgentState,
    load_yolo_cached,
)
from app.src.agent.eval_gate import eval_gate_node

_LAZY_GRAPH_EXPORTS = {
    "WildlifeAgentRunner",
    "build_wildlife_graph",
    "get_graph_mermaid",
    "hitl_review_node",
    "route_after_eval_gate",
}

__all__ = [
    "RunMode",
    "MetricsSnapshot",
    "WildlifeAgentState",
    "load_yolo_cached",
    "eval_gate_node",
    "WildlifeAgentRunner",
    "build_wildlife_graph",
    "get_graph_mermaid",
    "hitl_review_node",
    "route_after_eval_gate",
]


def __getattr__(name: str) -> Any:
    """Lazily expose LangGraph graph builders without creating circular imports on package init."""
    if name in _LAZY_GRAPH_EXPORTS:
        from app.src.agent import graph

        return getattr(graph, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
