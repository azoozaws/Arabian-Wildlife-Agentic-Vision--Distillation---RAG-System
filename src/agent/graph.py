"""LangGraph Wildlife Agent orchestrating video ingestion, RAG, training, evaluation gate, HITL, and ModelRegistry."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from app.data.rag_query_engine import search_wildlife_rag
from app.src.agent.agent_state import WildlifeAgentState
from app.src.agent.eval_gate import eval_gate_node
from app.src.agent.nodes import (
    evaluate_node,
    ingest_video_node,
    promote_or_reject_node,
    route_after_ingest,
    train_node,
)
from app.src.training.model_registry import (
    DEFAULT_MODELS_DIR,
    DEFAULT_REGISTRY_PATH,
    ModelRegistry,
)
from app.src.utils.logging_utils import get_logger

logger = get_logger("WildlifeLangGraphAgent")


def hitl_review_node(state: WildlifeAgentState) -> Dict[str, Any]:
    """Human-in-the-Loop node reached when eval_gate_node returns 'needs_hitl'.

    LangGraph interrupts BEFORE executing this node (`interrupt_before=["hitl_review"]`),
    allowing an engineer (via Gradio UI or API) to supply `engineer_override`:
    'approve_challenger' or 'reject_challenger' before resuming into `promote_or_reject_node`.
    """
    override = state.get("engineer_override")
    if override == "approve_challenger":
        return {
            "gate_decision": "promote",
            "gate_reason": f"{state.get('gate_reason', '')} [Engineer Override: APPROVED]".strip(),
        }
    if override == "reject_challenger":
        return {
            "gate_decision": "keep_champion",
            "gate_reason": f"{state.get('gate_reason', '')} [Engineer Override: REJECTED]".strip(),
        }
    return {
        "gate_decision": "needs_hitl",
    }


def route_after_eval_gate(state: WildlifeAgentState) -> Literal["hitl_review", "commit"]:
    """Route Champion-vs-Challenger gate outcome either to HITL review or directly to registry commit."""
    decision = state.get("gate_decision", "pending")
    override = state.get("engineer_override")
    if decision == "needs_hitl" and override is None:
        return "hitl_review"
    return "commit"


def build_wildlife_graph(
    checkpointer: Any = None,
    interrupt_on_hitl: bool = True,
) -> Any:
    """Construct and compile the LangGraph StateGraph for the Arabian Wildlife Agent.

    Graph flow:
        START -> ingest_video -> [route_after_ingest]
            - "end"   -> END  (display_only, summary_only, summary_with_rag, summary_with_qwen)
            - "train" -> train -> evaluate -> eval_gate -> [route_after_eval_gate]
                - "hitl_review" -> hitl_review (interrupted for human approval) -> commit -> END
                - "commit"      -> promote_or_reject -> END
    """
    try:
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise ImportError(
            "langgraph is required to compile the StateGraph. Install it via: pip install langgraph"
        ) from exc

    builder = StateGraph(WildlifeAgentState)

    builder.add_node("ingest_video", ingest_video_node)
    builder.add_node("train", train_node)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("eval_gate", eval_gate_node)
    builder.add_node("hitl_review", hitl_review_node)
    builder.add_node("commit", promote_or_reject_node)

    builder.add_edge(START, "ingest_video")
    builder.add_conditional_edges(
        "ingest_video",
        route_after_ingest,
        {
            "train": "train",
            "end": END,
        },
    )
    builder.add_edge("train", "evaluate")
    builder.add_edge("evaluate", "eval_gate")
    builder.add_conditional_edges(
        "eval_gate",
        route_after_eval_gate,
        {
            "hitl_review": "hitl_review",
            "commit": "commit",
        },
    )
    builder.add_edge("hitl_review", "commit")
    builder.add_edge("commit", END)

    cp = checkpointer if checkpointer is not None else MemorySaver()
    interrupt_nodes = ["hitl_review"] if interrupt_on_hitl else []
    return builder.compile(checkpointer=cp, interrupt_before=interrupt_nodes)


def get_graph_mermaid() -> str:
    """Return Mermaid flowchart markup representing the LangGraph workflow."""
    return (
        "flowchart TD\n"
        "    START([START]) --> ingest_video[ingest_video_node]\n"
        "    ingest_video -->|display_only / summary_*| END_NODE([END + Summary])\n"
        "    ingest_video -->|train_only / display_and_train| train[train_node]\n"
        "    train --> evaluate[evaluate_node]\n"
        "    evaluate --> eval_gate[eval_gate_node]\n"
        "    eval_gate -->|promote / keep_champion| commit[promote_or_reject_node]\n"
        "    eval_gate -->|needs_hitl| hitl[hitl_review_node - HITL Interrupt]\n"
        "    hitl --> commit\n"
        "    commit --> END_NODE\n"
    )


class _InProcessGraphFallback:
    """Deterministic local state-machine fallback matching LangGraph's exact edges & HITL pause semantics.

    Used ONLY when `langgraph` is not installed yet so the Gradio UI and tests remain runnable.
    """

    def __init__(self, interrupt_on_hitl: bool = True) -> None:
        self.interrupt_on_hitl = interrupt_on_hitl
        self._threads: Dict[str, Dict[str, Any]] = {}

    def run_thread(self, initial_state: WildlifeAgentState, thread_id: str) -> Dict[str, Any]:
        state: Dict[str, Any] = dict(initial_state)
        state.update(ingest_video_node(state))

        if route_after_ingest(state) == "end":
            self._threads[thread_id] = {"state": state, "next": ()}
            return state

        state.update(train_node(state))
        state.update(evaluate_node(state))
        state.update(eval_gate_node(state))

        next_hop = route_after_eval_gate(state)
        if next_hop == "hitl_review":
            if self.interrupt_on_hitl:
                self._threads[thread_id] = {"state": state, "next": ("hitl_review",)}
                return state
            state.update(hitl_review_node(state))

        state.update(promote_or_reject_node(state))
        self._threads[thread_id] = {"state": state, "next": ()}
        return state

    def resume_hitl(
        self,
        thread_id: str,
        engineer_override: Literal["approve_challenger", "reject_challenger"],
    ) -> Dict[str, Any]:
        if thread_id not in self._threads:
            raise KeyError(f"Unknown thread_id '{thread_id}'.")
        state = dict(self._threads[thread_id]["state"])
        state["engineer_override"] = engineer_override
        state.update(hitl_review_node(state))
        state.update(promote_or_reject_node(state))
        self._threads[thread_id] = {"state": state, "next": ()}
        return state

    def get_snapshot(self, thread_id: str) -> Dict[str, Any]:
        return self._threads.get(thread_id, {"state": {}, "next": ()})


class WildlifeAgentRunner:
    """High-level facade wrapping the compiled LangGraph agent, HITL resumption, RAG query, and rollback."""

    def __init__(
        self,
        registry_path: Path | str = DEFAULT_REGISTRY_PATH,
        models_dir: Path | str = DEFAULT_MODELS_DIR,
        interrupt_on_hitl: bool = True,
        allow_fallback: bool = True,
    ) -> None:
        self.registry_path = Path(registry_path)
        self.models_dir = Path(models_dir)
        self.interrupt_on_hitl = interrupt_on_hitl
        self.using_langgraph = False
        self._graph: Any = None
        self._fallback: Optional[_InProcessGraphFallback] = None

        try:
            self._graph = build_wildlife_graph(interrupt_on_hitl=interrupt_on_hitl)
            self.using_langgraph = True
            logger.info("Initialized WildlifeAgentRunner with native LangGraph StateGraph.")
        except ImportError:
            if not allow_fallback:
                raise
            self._fallback = _InProcessGraphFallback(interrupt_on_hitl=interrupt_on_hitl)
            logger.warning(
                "langgraph is not installed; using in-process StateGraph fallback. "
                "Install LangGraph with: pip install langgraph"
            )

    def get_registry(self) -> ModelRegistry:
        return ModelRegistry(registry_path=self.registry_path, models_dir=self.models_dir)

    def _prepare_initial_state(self, state: WildlifeAgentState) -> WildlifeAgentState:
        prepared: Dict[str, Any] = dict(state)
        prepared.setdefault("registry_path", str(self.registry_path))
        prepared.setdefault("models_dir", str(self.models_dir))
        prepared.setdefault("run_mode", "summary_only")
        prepared.setdefault("gate_decision", "pending")
        prepared.setdefault("gate_reason", "")
        prepared.setdefault("champion_points", 0)
        prepared.setdefault("challenger_points", 0)
        prepared.setdefault("engineer_override", None)

        if not prepared.get("champion_weights_path"):
            active = self.get_registry().get_active_champion()
            if active:
                prepared["champion_weights_path"] = active["weights_path"]
                prepared.setdefault("active_champion_model_id", active["model_id"])

        return prepared  # type: ignore[return-value]

    def run(
        self,
        state: WildlifeAgentState,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the wildlife agent graph and return final (or HITL-paused) state plus thread_id."""
        tid = thread_id or f"wildlife-{uuid.uuid4().hex[:8]}"
        init_state = self._prepare_initial_state(state)

        if self.using_langgraph:
            config = {"configurable": {"thread_id": tid}}
            final_state = self._graph.invoke(init_state, config=config)
            snap = self._graph.get_state(config)
            awaiting_hitl = "hitl_review" in tuple(getattr(snap, "next", ()) or ())
            return {
                "thread_id": tid,
                "engine": "langgraph",
                "awaiting_hitl": awaiting_hitl,
                "state": dict(final_state),
            }

        assert self._fallback is not None
        final_state = self._fallback.run_thread(init_state, thread_id=tid)
        snap = self._fallback.get_snapshot(tid)
        return {
            "thread_id": tid,
            "engine": "local_state_machine_fallback",
            "awaiting_hitl": "hitl_review" in snap.get("next", ()),
            "state": dict(final_state),
        }

    def resume_hitl(
        self,
        thread_id: str,
        engineer_override: Literal["approve_challenger", "reject_challenger"],
    ) -> Dict[str, Any]:
        """Resume a paused HITL thread after an engineer approves or rejects the Challenger."""
        if self.using_langgraph:
            config = {"configurable": {"thread_id": thread_id}}
            self._graph.update_state(config, {"engineer_override": engineer_override})
            final_state = self._graph.invoke(None, config=config)
            return {
                "thread_id": thread_id,
                "engine": "langgraph",
                "awaiting_hitl": False,
                "state": dict(final_state),
            }

        assert self._fallback is not None
        final_state = self._fallback.resume_hitl(thread_id, engineer_override=engineer_override)
        return {
            "thread_id": thread_id,
            "engine": "local_state_machine_fallback",
            "awaiting_hitl": False,
            "state": dict(final_state),
        }

    def query_rag(self, query_text: str, **kwargs: Any) -> Dict[str, Any]:
        """Run a natural-language wildlife RAG query (Arabic or English)."""
        return search_wildlife_rag(query_text=query_text, **kwargs)

    def rollback_champion(
        self,
        target_model_id: Optional[str] = None,
        reason: str = "Rollback requested via WildlifeAgentRunner",
    ) -> Dict[str, Any]:
        """Roll back the active Champion in registry.json to the previous stable version."""
        return self.get_registry().rollback(target_model_id=target_model_id, reason=reason)
