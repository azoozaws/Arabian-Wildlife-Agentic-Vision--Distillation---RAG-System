"""LangGraph-compatible agent state definitions and cached model loader."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, TypedDict

if TYPE_CHECKING:
    from ultralytics import YOLO

# Execution mode for video ingestion and downstream workflow:
# - display_only: live annotated video + summary only (no disk buckets, no Qwen, no YOLO training)
# - summary_only: scan + dual buckets + full summary only (skip Qwen and YOLO training)
# - summary_with_rag: scan + dual buckets + Qwen 3-frame RAG (-> ChromaDB) + summary (skip Qwen verify & YOLO training)
# - summary_with_qwen: scan + dual buckets + Qwen verify + Qwen 3-frame RAG (-> ChromaDB) + summary (skip YOLO training)
# - train_only: headless full workflow (scan + Qwen verify + RAG + summary + YOLO train/eval/gate)
# - display_and_train: live annotated video + full workflow (scan + Qwen verify + RAG + summary + YOLO train/eval/gate)
RunMode = Literal[
    "display_only",
    "summary_only",
    "summary_with_rag",
    "summary_with_qwen",
    "train_only",
    "display_and_train",
]


class MetricsSnapshot(TypedDict):
    precision: float
    recall: float
    mAP50: float
    fp_per_hour: float


class WildlifeAgentState(TypedDict, total=False):
    # Execution / display mode & summary outputs
    run_mode: RunMode
    display_clean_demo_mode: bool
    should_run_pipeline: bool
    video_summary: Optional[Dict[str, Any]]
    video_summary_markdown: str
    video_summary_path: str

    # Input videos and workspace directories
    video_paths: List[str]
    out_workspace_dir: str
    qwen_review_dir: str        # Bucket (< 0.95): images/, labels/, debug/ -> Qwen GRPO review -> YOLO training
    trusted_rag_dir: str        # Bucket (>= 0.75): images/, labels/, debug/ -> Qwen description -> ChromaDB
    qwen_distilled_dir: str     # Verified hard pos/neg dataset after Qwen GRPO audit
    rag_jsonl_path: str         # Generated RAG JSONL file path
    chromadb_dir: str           # Persistent ChromaDB directory
    chromadb_collection: str    # ChromaDB collection name

    # Dataset, training & model registry paths
    orig_dataset_dir: str
    merged_dataset_dir: str
    train_project_dir: str
    train_run_dir: Optional[str]
    qwen_lora_dir: Optional[str]
    version_tag: str
    registry_path: str
    models_dir: str
    active_champion_model_id: str
    challenger_model_id: str

    # Model paths (serializable strings — NEVER raw YOLO objects)
    champion_weights_path: str
    challenger_weights_path: str
    eval_yaml_path: str
    held_out_video_path: str

    # Evaluation metrics
    champion_metrics: Optional[MetricsSnapshot]
    challenger_metrics: Optional[MetricsSnapshot]

    # Gate scoring & HITL decision
    champion_points: int
    challenger_points: int
    gate_decision: Literal["pending", "promote", "keep_champion", "needs_hitl"]
    gate_reason: str
    engineer_override: Optional[Literal["approve_challenger", "reject_challenger"]]


@lru_cache(maxsize=2)
def load_yolo_cached(weights_path: str) -> "YOLO":
    """Keep YOLO weights cached in memory outside LangGraph's serializable state."""
    from ultralytics import YOLO

    return YOLO(weights_path)
