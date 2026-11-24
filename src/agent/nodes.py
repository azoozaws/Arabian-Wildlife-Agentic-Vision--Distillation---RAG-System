"""LangGraph node functions adhering strictly to WildlifeAgentState, ModelRegistry, and eval_gate_node."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal

from app.src.agent.agent_state import WildlifeAgentState
from app.src.config import (
    PipelineConfig,
    QwenVLMConfig,
    RAGDescriptionConfig,
    TrainingConfig,
    VideoScanConfig,
    resolve_mode_flags,
)
from app.src.pipelines.distillation_pipeline import run_distillation_and_rag_pipeline
from app.src.pipelines.training_pipeline import run_evaluation_stage, run_training_stage
from app.src.training.model_registry import (
    DEFAULT_MODELS_DIR,
    DEFAULT_REGISTRY_PATH,
    ModelRegistry,
)


def _get_registry(state: WildlifeAgentState) -> ModelRegistry:
    registry_path = Path(state["registry_path"]) if state.get("registry_path") else DEFAULT_REGISTRY_PATH
    models_dir = Path(state["models_dir"]) if state.get("models_dir") else DEFAULT_MODELS_DIR
    return ModelRegistry(registry_path=registry_path, models_dir=models_dir)


def ingest_video_node(state: WildlifeAgentState) -> Dict[str, Any]:
    """Node 1: Scan/display videos per state['run_mode'], run optional Qwen verify/RAG, and build full summary."""
    run_mode = state.get("run_mode", "train_only")
    mode_flags = resolve_mode_flags(run_mode)
    out_dir = Path(state["out_workspace_dir"])
    chroma_dir = Path(state["chromadb_dir"]) if state.get("chromadb_dir") else None

    cfg = PipelineConfig(
        out_dataset_dir=out_dir,
        video_scan=VideoScanConfig(
            yolo_weights=state["champion_weights_path"],
            run_mode=run_mode,
            display_clean_demo_mode=bool(state.get("display_clean_demo_mode", False)),
        ),
        qwen=QwenVLMConfig(
            lora_dir=state.get("qwen_lora_dir"),
        ),
        rag=RAGDescriptionConfig(
            enabled=mode_flags["run_qwen_rag"],
            trusted_only=True,
            temporal_frames_per_track=3,
            max_image_dim=512,
            chromadb_dir=chroma_dir,
            chromadb_collection=state.get("chromadb_collection", "arabian_wildlife_rag"),
        ),
        run_stage2_distill=mode_flags["run_qwen_verify"],
        save_debug=True,
    )

    report = run_distillation_and_rag_pipeline(
        video_paths=state["video_paths"],
        config=cfg,
    )

    video_summary = report.get("video_summary") or {}
    summary_md = video_summary.get("summary_markdown", "")
    summary_path = str(report.get("video_summary_path", ""))

    if not mode_flags["run_yolo_training"]:
        return {
            "should_run_pipeline": False,
            "qwen_review_dir": str(cfg.qwen_review_bucket_dir) if mode_flags["export_buckets"] else "",
            "trusted_rag_dir": str(cfg.trusted_rag_bucket_dir) if mode_flags["export_buckets"] else "",
            "rag_jsonl_path": str((report.get("stage3_qwen_rag") or {}).get("rag_jsonl_path", "")),
            "video_summary": video_summary,
            "video_summary_markdown": summary_md,
            "video_summary_path": summary_path,
            "gate_decision": "keep_champion",
            "gate_reason": f"run_mode='{run_mode}' skips YOLO training; summary generated.",
        }

    rag_info = report.get("stage3_qwen_rag") or {}
    return {
        "should_run_pipeline": True,
        "qwen_review_dir": str(cfg.qwen_review_bucket_dir),
        "trusted_rag_dir": str(cfg.trusted_rag_bucket_dir),
        "qwen_distilled_dir": str(out_dir),
        "rag_jsonl_path": str(rag_info.get("rag_jsonl_path", "")),
        "video_summary": video_summary,
        "video_summary_markdown": summary_md,
        "video_summary_path": summary_path,
    }


def route_after_ingest(state: WildlifeAgentState) -> Literal["train", "end"]:
    """Conditional edge router: proceed to train_node ONLY for modes that train YOLO."""
    run_mode = state.get("run_mode", "train_only")
    mode_flags = resolve_mode_flags(run_mode)
    if not mode_flags["run_yolo_training"] or not state.get("should_run_pipeline", True):
        return "end"
    return "train"


def train_node(state: WildlifeAgentState) -> Dict[str, Any]:
    """Node 2 (Train): Merge Qwen-verified samples with replay buffer, fine-tune, and register in registry.json."""
    version_tag = state.get("version_tag", "v4")
    qwen_dataset_dir = Path(state.get("qwen_distilled_dir") or state["qwen_review_dir"])
    registry_path = Path(state["registry_path"]) if state.get("registry_path") else DEFAULT_REGISTRY_PATH
    models_dir = Path(state["models_dir"]) if state.get("models_dir") else DEFAULT_MODELS_DIR

    cfg = TrainingConfig(
        orig_dataset=Path(state["orig_dataset_dir"]),
        qwen_dataset=qwen_dataset_dir,
        merged_dataset=Path(state["merged_dataset_dir"]),
        baseline_weights=Path(state["champion_weights_path"]),
        project_dir=Path(state["train_project_dir"]),
        version_tag=version_tag,
        run_name=f"yolo_{version_tag}_rl_distilled",
        model_id=state.get("challenger_model_id") or f"yolo_{version_tag}_best",
        parent_model_id=state.get("active_champion_model_id") or Path(state["champion_weights_path"]).stem,
        registry_path=registry_path,
        models_dir=models_dir,
    )

    train_out = run_training_stage(cfg)
    return {
        "challenger_model_id": train_out["challenger_model_id"],
        "challenger_weights_path": train_out["challenger_weights_path"],
        "eval_yaml_path": train_out["eval_yaml_path"],
        "train_run_dir": train_out["run_dir"],
        "registry_path": train_out["registry_path"],
        "models_dir": train_out["models_dir"],
        "gate_decision": "pending",
    }


def evaluate_node(state: WildlifeAgentState) -> Dict[str, Any]:
    """Node 3 (Evaluate): Compute Champion and Challenger MetricsSnapshots and log them to registry.json."""
    version_tag = state.get("version_tag", "v4")
    out_dir = Path(state["train_run_dir"]) if state.get("train_run_dir") else None
    merged_dir = Path(state["merged_dataset_dir"]) if state.get("merged_dataset_dir") else None
    test_images_dir = (merged_dir / "test" / "images") if merged_dir is not None else None

    eval_out = run_evaluation_stage(
        champion_weights_path=state["champion_weights_path"],
        challenger_weights_path=state["challenger_weights_path"],
        eval_yaml_path=state["eval_yaml_path"],
        held_out_video_path=state.get("held_out_video_path", ""),
        out_dir=out_dir,
        test_images_dir=test_images_dir,
        version_tag=version_tag,
    )

    challenger_id = state.get("challenger_model_id") or Path(state["challenger_weights_path"]).stem
    registry = _get_registry(state)
    if registry.get_model(challenger_id) is not None:
        registry.record_evaluation(
            model_id=challenger_id,
            metrics=eval_out["challenger_metrics"],
            gate_decision="pending",
        )

    return {
        "champion_metrics": eval_out["champion_metrics"],
        "challenger_metrics": eval_out["challenger_metrics"],
    }


def promote_or_reject_node(state: WildlifeAgentState) -> Dict[str, Any]:
    """Node 5 (Registry Commit): Persist eval_gate_node / HITL decision into registry.json.

    - Promotes Challenger to active Champion (archiving previous Champion for easy rollback)
      when gate_decision == 'promote' or engineer_override == 'approve_challenger'.
    - Rejects Challenger when gate_decision == 'keep_champion' or engineer_override == 'reject_challenger'.
    """
    registry = _get_registry(state)
    challenger_id = state.get("challenger_model_id") or Path(state["challenger_weights_path"]).stem
    decision = state.get("gate_decision", "pending")
    override = state.get("engineer_override")
    reason = state.get("gate_reason", "")
    chall_metrics = state.get("challenger_metrics")

    # Record scorecard first if model exists in registry
    if registry.get_model(challenger_id) is not None and chall_metrics is not None:
        registry.record_evaluation(
            model_id=challenger_id,
            metrics=chall_metrics,
            gate_decision=decision,
            gate_reason=reason,
            champion_points=state.get("champion_points"),
            challenger_points=state.get("challenger_points"),
        )

    should_promote = (decision == "promote") or (override == "approve_challenger")
    should_reject = (decision == "keep_champion") or (override == "reject_challenger")

    if should_promote:
        entry = registry.promote_model(
            model_id=challenger_id,
            metrics=chall_metrics,
            gate_reason=reason,
            engineer_override=override,
        )
        return {
            "active_champion_model_id": entry["model_id"],
            "champion_weights_path": entry["weights_path"],
            "gate_decision": "promote",
        }

    if should_reject:
        registry.reject_model(
            model_id=challenger_id,
            gate_reason=reason,
            engineer_override=override,
        )
        return {"gate_decision": "keep_champion"}

    return {}
