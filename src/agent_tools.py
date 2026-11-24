"""Agent-ready tool facade with clean JSON-serializable inputs and outputs.

Each function in this module is designed to be registered directly as a tool
in an Agent framework (e.g., LangGraph, Google ADK, CrewAI, OpenAI Agents SDK, or MCP).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from app.data.rag_store import (
    filter_rag_documents,
    load_rag_documents_jsonl,
    query_chromadb_rag,
    rebuild_chromadb_from_jsonl,
    summarize_rag_corpus,
    upsert_rag_documents_to_chromadb,
)
from app.src.agent.agent_state import RunMode
from app.src.config import (
    DEFAULT_EMBEDDING_MODEL_ID,
    DEFAULT_QWEN_MODEL_ID,
    PipelineConfig,
    QwenVLMConfig,
    RAGDescriptionConfig,
    TrainingConfig,
    VideoScanConfig,
    WandbConfig,
)
from app.src.pipelines.distillation_pipeline import run_distillation_and_rag_pipeline
from app.src.pipelines.training_pipeline import (
    run_evaluation_stage,
    run_training_pipeline,
    run_training_stage,
)
from app.src.schemas import RAGQueryFilter
from app.src.training.model_registry import (
    DEFAULT_MODELS_DIR,
    DEFAULT_REGISTRY_PATH,
    ModelRegistry,
)
from app.src.vision.qwen_vlm import QwenWildlifeVLM
from app.src.vision.video_scanner import VideoScanner


def tool_scan_wildlife_videos(
    video_paths: List[str],
    yolo_weights: str,
    staging_dir: str,
    run_mode: RunMode = "train_only",
    conf_min: float = 0.15,
    conf_review_thresh: float = 0.95,
    trusted_rag_min_conf: float = 0.75,
    idle_stride: int = 30,
    active_stride: int = 5,
    audit_all: bool = False,
) -> Dict[str, Any]:
    """Scan/display wildlife videos and export <0.95 review and >=0.75 trusted RAG buckets."""
    scan_cfg = VideoScanConfig(
        yolo_weights=yolo_weights,
        run_mode=run_mode,
        conf_min=conf_min,
        conf_review_thresh=conf_review_thresh,
        trusted_rag_min_conf=trusted_rag_min_conf,
        idle_stride=idle_stride,
        active_stride=active_stride,
        audit_all=audit_all,
    )
    scanner = VideoScanner(scan_cfg)
    manifest, stats = scanner.scan_videos(
        video_paths=video_paths,
        staging_dir=Path(staging_dir),
        unload_after=True,
    )
    return {
        "run_mode": run_mode,
        "video_stats": stats,
        "total_tracks": len(manifest),
        "tracks": [m.to_dict() for m in manifest],
    }


def tool_run_distillation_and_rag(
    video_paths: List[str],
    yolo_weights: str,
    out_dataset_dir: str,
    run_mode: RunMode = "train_only",
    lora_dir: Optional[str] = None,
    qwen_model_id: str = DEFAULT_QWEN_MODEL_ID,
    conf_min: float = 0.15,
    conf_review_thresh: float = 0.95,
    trusted_rag_min_conf: float = 0.75,
    audit_all: bool = False,
    qwen_batch: int = 8,
    run_stage2_distill: bool = True,
    generate_rag: bool = True,
    rag_language: Literal["ar", "en", "bilingual"] = "ar",
    chromadb_dir: Optional[str] = None,
    chromadb_collection: str = "arabian_wildlife_rag",
    save_debug: bool = True,
) -> Dict[str, Any]:
    """Run Stage 1 (scan/display + dual buckets), Stage 2 (Qwen GRPO), and Stage 3 (Qwen RAG -> ChromaDB)."""
    cfg = PipelineConfig(
        out_dataset_dir=Path(out_dataset_dir),
        video_scan=VideoScanConfig(
            yolo_weights=yolo_weights,
            run_mode=run_mode,
            conf_min=conf_min,
            conf_review_thresh=conf_review_thresh,
            trusted_rag_min_conf=trusted_rag_min_conf,
            audit_all=audit_all,
        ),
        qwen=QwenVLMConfig(
            model_id=qwen_model_id,
            lora_dir=lora_dir,
            batch_size=qwen_batch,
        ),
        rag=RAGDescriptionConfig(
            enabled=generate_rag,
            language=rag_language,
            trusted_only=True,
            chromadb_dir=Path(chromadb_dir) if chromadb_dir else None,
            chromadb_collection=chromadb_collection,
        ),
        run_stage2_distill=run_stage2_distill,
        save_debug=save_debug,
    )
    return run_distillation_and_rag_pipeline(video_paths=video_paths, config=cfg)


def tool_describe_track_frames(
    frame_paths: List[str],
    species: str = "wildlife",
    video_source: Optional[str] = None,
    temporal_frame_indices: Optional[List[int]] = None,
    start_hms: Optional[str] = None,
    end_hms: Optional[str] = None,
    species_counts: Optional[Dict[str, int]] = None,
    avg_conf: Optional[float] = None,
    qwen_model_id: str = DEFAULT_QWEN_MODEL_ID,
    lora_dir: Optional[str] = None,
    max_dim: int = 512,
    use_adapter_for_description: bool = False,
) -> Dict[str, Any]:
    """Use Base Qwen3-VL (adapter disabled) on up to 3 chronological frames to describe a wildlife track for RAG."""
    context: Dict[str, Any] = {
        "species": species,
        "video_source": video_source or (Path(frame_paths[0]).name if frame_paths else "unknown"),
        "temporal_frame_indices": temporal_frame_indices or [],
        "start_hms": start_hms or "00:00:00",
        "end_hms": end_hms or start_hms or "00:00:00",
        "best_hms": start_hms or "00:00:00",
        "species_counts": species_counts or {},
        "avg_conf": avg_conf,
    }
    vlm_cfg = QwenVLMConfig(
        model_id=qwen_model_id,
        lora_dir=lora_dir,
        use_adapter_for_description=use_adapter_for_description,
    )
    with QwenWildlifeVLM(vlm_cfg) as vlm:
        result = vlm.describe_track_for_rag(
            frame_paths=frame_paths[:3],
            species=species,
            context=context,
            max_dim=max_dim,
        )
    return result.to_dict()


def tool_describe_wildlife_image(
    image_path: str,
    video_source: Optional[str] = None,
    best_frame: Optional[int] = None,
    start_hms: Optional[str] = None,
    end_hms: Optional[str] = None,
    species_counts: Optional[Dict[str, int]] = None,
    avg_conf: Optional[float] = None,
    qwen_model_id: str = DEFAULT_QWEN_MODEL_ID,
    lora_dir: Optional[str] = None,
    use_adapter_for_description: bool = False,
) -> Dict[str, Any]:
    """Call Qwen3-VL to generate a rich RAG-ready Arabic/English description for a single wildlife image."""
    context: Optional[Dict[str, Any]] = None
    if any(v is not None for v in (video_source, best_frame, start_hms, end_hms, species_counts, avg_conf)):
        context = {
            "video_source": video_source or Path(image_path).name,
            "best_frame": best_frame if best_frame is not None else "N/A",
            "start_hms": start_hms or "00:00:00",
            "end_hms": end_hms or start_hms or "00:00:00",
            "best_hms": start_hms or "00:00:00",
            "species_counts": species_counts or {},
            "avg_conf": avg_conf,
        }

    vlm_cfg = QwenVLMConfig(
        model_id=qwen_model_id,
        lora_dir=lora_dir,
        use_adapter_for_description=use_adapter_for_description,
    )
    with QwenWildlifeVLM(vlm_cfg) as vlm:
        result = vlm.describe_single_image(image_path=image_path, context=context)
    return result.to_dict()


def tool_query_rag_documents(
    rag_jsonl_path: str,
    video_source: Optional[str] = None,
    species: Optional[List[str]] = None,
    min_oryx_count: Optional[int] = None,
    max_oryx_count: Optional[int] = None,
    min_gazelle_count: Optional[int] = None,
    max_gazelle_count: Optional[int] = None,
    min_total_animals: Optional[int] = None,
    min_conf: Optional[float] = None,
    max_conf: Optional[float] = None,
    needs_review: Optional[bool] = None,
    qwen_verified: Optional[bool] = None,
    time_of_day: Optional[List[str]] = None,
    environment_type: Optional[str] = None,
    animal_states: Optional[List[str]] = None,
    frame_start: Optional[int] = None,
    frame_end: Optional[int] = None,
    text_contains: Optional[str] = None,
    limit: int = 25,
) -> Dict[str, Any]:
    """Query and filter wildlife RAG JSONL documents by species counts, confidence, review flags, and scene tags."""
    docs = load_rag_documents_jsonl(Path(rag_jsonl_path))
    query_filter = RAGQueryFilter(
        video_source=video_source,
        species=species,
        min_oryx_count=min_oryx_count,
        max_oryx_count=max_oryx_count,
        min_gazelle_count=min_gazelle_count,
        max_gazelle_count=max_gazelle_count,
        min_total_animals=min_total_animals,
        min_conf=min_conf,
        max_conf=max_conf,
        needs_review=needs_review,
        qwen_verified=qwen_verified,
        time_of_day=time_of_day,
        environment_type=environment_type,
        animal_states=animal_states,
        frame_start=frame_start,
        frame_end=frame_end,
        text_contains=text_contains,
    )
    matched = filter_rag_documents(docs, query_filter)
    limited = matched[: max(1, limit)] if limit > 0 else matched
    return {
        "total_in_store": len(docs),
        "total_matched": len(matched),
        "returned": len(limited),
        "summary": summarize_rag_corpus(matched),
        "results": [d.to_dict() for d in limited],
    }


def tool_index_rag_jsonl_to_chromadb(
    rag_jsonl_path: str,
    chromadb_dir: str,
    collection_name: str = "arabian_wildlife_rag",
    embedding_model_id: str = DEFAULT_EMBEDDING_MODEL_ID,
    reset_collection: bool = False,
) -> Dict[str, Any]:
    """Load RAG documents from JSONL and upsert them into ChromaDB using Qwen3-Embedding-0.6B."""
    docs = load_rag_documents_jsonl(Path(rag_jsonl_path))
    return upsert_rag_documents_to_chromadb(
        docs=docs,
        persist_dir=Path(chromadb_dir),
        collection_name=collection_name,
        embedding_model_id=embedding_model_id,
        reset_collection=reset_collection,
    )


def tool_rebuild_chromadb(
    rag_jsonl_paths: List[str],
    chromadb_dir: str,
    collection_name: str = "arabian_wildlife_rag",
    embedding_model_id: str = DEFAULT_EMBEDDING_MODEL_ID,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """Delete and rebuild a ChromaDB collection from JSONL file(s) using Qwen/Qwen3-Embedding-0.6B."""
    return rebuild_chromadb_from_jsonl(
        rag_jsonl_paths=rag_jsonl_paths,
        persist_dir=Path(chromadb_dir),
        collection_name=collection_name,
        embedding_model_id=embedding_model_id,
        batch_size=batch_size,
    )


def tool_query_chromadb_rag(
    chromadb_dir: str,
    query_text: str,
    collection_name: str = "arabian_wildlife_rag",
    embedding_model_id: str = DEFAULT_EMBEDDING_MODEL_ID,
    where_filter: Optional[Dict[str, Any]] = None,
    n_results: int = 10,
) -> Dict[str, Any]:
    """Run semantic + metadata-filtered query against ChromaDB wildlife descriptions."""
    return query_chromadb_rag(
        persist_dir=Path(chromadb_dir),
        query_text=query_text,
        collection_name=collection_name,
        embedding_model_id=embedding_model_id,
        where_filter=where_filter,
        n_results=n_results,
    )


def tool_get_rag_corpus_summary(rag_jsonl_path: str) -> Dict[str, Any]:
    """Return high-level wildlife sighting and review statistics from a RAG JSONL file."""
    docs = load_rag_documents_jsonl(Path(rag_jsonl_path))
    return summarize_rag_corpus(docs)


def tool_train_yolo(
    orig_dataset: str,
    qwen_dataset: str,
    merged_dataset: str,
    champion_weights: str,
    project_dir: str,
    version_tag: str = "v3",
    replay_samples: int = 400,
    epochs: int = 25,
    enable_wandb: bool = True,
) -> Dict[str, Any]:
    """Stage A (Train only): Merge Qwen + replay dataset and fine-tune Challenger YOLO weights."""
    cfg = TrainingConfig(
        orig_dataset=Path(orig_dataset),
        qwen_dataset=Path(qwen_dataset),
        merged_dataset=Path(merged_dataset),
        baseline_weights=Path(champion_weights),
        project_dir=Path(project_dir),
        version_tag=version_tag,
        run_name=f"yolo_{version_tag}_rl_distilled",
        replay_samples=replay_samples,
        epochs=epochs,
        wandb=WandbConfig(enabled=enable_wandb),
    )
    return run_training_stage(cfg)


def tool_evaluate_champion_vs_challenger(
    champion_weights_path: str,
    challenger_weights_path: str,
    eval_yaml_path: str,
    held_out_video_path: str = "",
    split: str = "test",
    out_dir: Optional[str] = None,
    test_images_dir: Optional[str] = None,
    version_tag: str = "v3",
) -> Dict[str, Any]:
    """Stage B (Evaluate only): Compute Champion and Challenger MetricsSnapshots for eval_gate_node."""
    return run_evaluation_stage(
        champion_weights_path=champion_weights_path,
        challenger_weights_path=challenger_weights_path,
        eval_yaml_path=eval_yaml_path,
        held_out_video_path=held_out_video_path,
        split=split,
        out_dir=Path(out_dir) if out_dir else None,
        test_images_dir=Path(test_images_dir) if test_images_dir else None,
        version_tag=version_tag,
    )


def tool_train_incremental_yolo(
    orig_dataset: str,
    qwen_dataset: str,
    merged_dataset: str,
    baseline_weights: str,
    project_dir: str,
    held_out_video_path: str = "",
    version_tag: str = "v3",
    replay_samples: int = 400,
    epochs: int = 25,
    enable_wandb: bool = True,
) -> Dict[str, Any]:
    """Convenience wrapper running both training and Champion-vs-Challenger evaluation."""
    cfg = TrainingConfig(
        orig_dataset=Path(orig_dataset),
        qwen_dataset=Path(qwen_dataset),
        merged_dataset=Path(merged_dataset),
        baseline_weights=Path(baseline_weights),
        project_dir=Path(project_dir),
        version_tag=version_tag,
        run_name=f"yolo_{version_tag}_rl_distilled",
        replay_samples=replay_samples,
        epochs=epochs,
        wandb=WandbConfig(enabled=enable_wandb),
    )
    return run_training_pipeline(cfg, held_out_video_path=held_out_video_path)


def tool_get_active_champion(
    registry_path: Optional[str] = None,
    models_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the currently active Champion model entry from registry.json."""
    reg = ModelRegistry(
        registry_path=Path(registry_path) if registry_path else DEFAULT_REGISTRY_PATH,
        models_dir=Path(models_dir) if models_dir else DEFAULT_MODELS_DIR,
    )
    return {
        "active_champion": reg.get_active_champion(),
        "previous_champion": reg.get_previous_champion(),
    }


def tool_list_registered_models(
    status: Optional[str] = None,
    registry_path: Optional[str] = None,
    models_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """List all registered YOLO models and their evaluation metrics from registry.json."""
    reg = ModelRegistry(
        registry_path=Path(registry_path) if registry_path else DEFAULT_REGISTRY_PATH,
        models_dir=Path(models_dir) if models_dir else DEFAULT_MODELS_DIR,
    )
    models = reg.list_models(status=status)
    return {
        "total": len(models),
        "models": models,
    }


def tool_promote_model(
    model_id: str,
    gate_reason: str = "",
    engineer_override: Optional[str] = None,
    registry_path: Optional[str] = None,
    models_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Promote a registered candidate model to active Champion in registry.json."""
    reg = ModelRegistry(
        registry_path=Path(registry_path) if registry_path else DEFAULT_REGISTRY_PATH,
        models_dir=Path(models_dir) if models_dir else DEFAULT_MODELS_DIR,
    )
    return reg.promote_model(
        model_id=model_id,
        gate_reason=gate_reason,
        engineer_override=engineer_override,
    )


def tool_rollback_model(
    target_model_id: Optional[str] = None,
    reason: str = "Rollback to previous stable champion",
    registry_path: Optional[str] = None,
    models_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Roll back active Champion in registry.json to previous_champion (or a specific archived model_id)."""
    reg = ModelRegistry(
        registry_path=Path(registry_path) if registry_path else DEFAULT_REGISTRY_PATH,
        models_dir=Path(models_dir) if models_dir else DEFAULT_MODELS_DIR,
    )
    return reg.rollback(target_model_id=target_model_id, reason=reason)
