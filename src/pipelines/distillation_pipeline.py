"""End-to-end orchestration: YOLO scan (dual buckets + display modes) -> Qwen GRPO -> Qwen RAG -> ChromaDB."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
from tqdm import tqdm

from app.data.rag_store import (
    build_rag_document,
    save_rag_documents_jsonl,
    summarize_rag_corpus,
    upsert_rag_documents_to_chromadb,
)
from app.src.config import CLASS_ID_TO_NAME, CLASS_NAME_TO_ID, PipelineConfig, resolve_mode_flags
from app.src.schemas import RAGDocument, TrackManifestItem
from app.src.utils.geometry import (
    build_zoomed_visual_prompt,
    norm_species,
    prepare_full_frame_pil,
    xyxy_to_yolo,
)
from app.src.utils.logging_utils import get_logger
from app.src.utils.summary_builder import build_video_summary
from app.src.utils.visualization import save_side_by_side_qwen_debug
from app.src.vision.qwen_vlm import QwenWildlifeVLM
from app.src.vision.video_scanner import VideoScanner

logger = get_logger("DistillationPipeline")


def _resolve_vram_safe_batch(requested: int, is_multiframe: bool = False) -> Tuple[int, bool]:
    """Auto-tune Qwen batch size based on available GPU VRAM (e.g., 15GB T4 vs 24GB L4 vs 40GB A100).

    Returns (effective_batch_size, is_low_vram_gpu).
    """
    cap = 4 if is_multiframe else requested
    low_vram = False
    try:
        import torch

        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram_gb < 18.0:
                # 15GB T4 GPU: keep peak VRAM safely under 12GB for FP16 Qwen3-VL
                low_vram = True
                cap = 2 if is_multiframe else min(requested, 4)
    except Exception:
        pass
    return max(1, min(requested, cap)), low_vram


def _clear_cuda_cache_if_needed(enabled: bool) -> None:
    if not enabled:
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_qwen_distillation(
    manifest: List[TrackManifestItem],
    vlm: QwenWildlifeVLM,
    out_dataset_dir: Path,
    batch_size: int = 8,
    zoom_scale: float = 3.0,
    save_debug: bool = True,
) -> Dict[str, int]:
    """Stage 2: Batched Qwen3-VL GRPO verification on tracks below conf_review_thresh (<0.95)."""
    to_audit = [m for m in manifest if m.needs_rl_review]
    stats = {"audited": 0, "hard_negatives": 0, "hard_positives": 0, "skipped": 0}
    if not to_audit:
        logger.info("No tracks flagged for Qwen classification review.")
        return stats

    out_images = out_dataset_dir / "images"
    out_labels = out_dataset_dir / "labels"
    out_debug = out_dataset_dir / "debug_compare"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    if save_debug:
        out_debug.mkdir(parents=True, exist_ok=True)

    eff_batch, is_low_vram = _resolve_vram_safe_batch(batch_size, is_multiframe=False)
    for i in tqdm(range(0, len(to_audit), eff_batch), desc=f"Stage 2 [Qwen Classify B={eff_batch}]"):
        chunk = to_audit[i : i + eff_batch]
        valid_items = []
        zoomed_imgs = []

        for ev in chunk:
            img_path = Path(ev.review_bucket_image_path or ev.raw_image_path)
            raw_bgr = cv2.imread(str(img_path))
            if raw_bgr is None:
                stats["skipped"] += 1
                continue
            zoomed_pil = build_zoomed_visual_prompt(raw_bgr, ev.yolo_bbox_xyxy, zoom_scale=zoom_scale)
            zoomed_imgs.append(zoomed_pil)
            valid_items.append((ev, img_path, raw_bgr, zoomed_pil))

        if not valid_items:
            continue

        results = vlm.classify_crops_batch(zoomed_imgs)

        for (ev, img_path, raw_bgr, zoomed_pil), cls_res in zip(valid_items, results):
            stats["audited"] += 1
            if not cls_res.parsed_ok:
                stats["skipped"] += 1
                continue

            pred_sp = norm_species(cls_res.species)
            img_h, img_w = raw_bgr.shape[:2]
            stem = f"{ev.video_tag}__trk_{ev.track_id:05d}"
            dst_img = out_images / f"{stem}.jpg"
            dst_lbl = out_labels / f"{stem}.txt"

            updated_lines: List[str] = []
            verified_counts: Dict[str, int] = {}

            for box_info in ev.all_frame_boxes:
                is_target = box_info["track_id"] == ev.track_id
                if is_target:
                    if pred_sp == "none" or pred_sp not in CLASS_NAME_TO_ID:
                        continue  # Drop rejected false positive
                    cls_id = CLASS_NAME_TO_ID[pred_sp]
                else:
                    cls_id = int(box_info["class_id"])

                sp_key = CLASS_ID_TO_NAME.get(cls_id, norm_species(box_info.get("species", f"class_{cls_id}")))
                verified_counts[sp_key] = verified_counts.get(sp_key, 0) + 1

                cx, cy, bw, bh = xyxy_to_yolo(box_info["xyxy"], img_w, img_h)
                updated_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            # Always save raw unmodified full frame alongside clean YOLO .txt labels
            shutil.copy2(img_path, dst_img)
            dst_lbl.write_text("\n".join(updated_lines) + ("\n" if updated_lines else ""), encoding="utf-8")

            # Enrich manifest item in-place for downstream RAG scene description
            ev.qwen_verified = True
            ev.qwen_species = pred_sp
            ev.verified_species_counts = verified_counts
            ev.distilled_image_path = str(dst_img)
            ev.distilled_label_path = str(dst_lbl)

            if pred_sp == "none":
                stats["hard_negatives"] += 1
            elif pred_sp in CLASS_NAME_TO_ID:
                stats["hard_positives"] += 1
            else:
                stats["skipped"] += 1

            if save_debug:
                tag = "POS" if pred_sp in CLASS_NAME_TO_ID else "NEG"
                save_side_by_side_qwen_debug(
                    raw_bgr=raw_bgr,
                    zoomed_prompt_pil=zoomed_pil,
                    out_path=out_debug / f"{stem}_{tag}.jpg",
                    yolo_species=ev.species_yolo,
                    yolo_conf=ev.avg_conf,
                    yolo_xyxy=ev.yolo_bbox_xyxy,
                    qwen_species=pred_sp,
                )
        _clear_cuda_cache_if_needed(is_low_vram)

    return stats


def run_qwen_rag_description(
    manifest: List[TrackManifestItem],
    vlm: QwenWildlifeVLM,
    config: PipelineConfig,
) -> Tuple[List[RAGDocument], Dict[str, Any]]:
    """Stage 3: Describe trusted (>=0.75) tracks via Qwen3-VL, save JSONL, and optionally index in ChromaDB."""
    rag_cfg = config.rag
    targets: List[TrackManifestItem] = []

    for ev in manifest:
        if rag_cfg.trusted_only and not ev.is_trusted_for_rag:
            continue
        if (
            not rag_cfg.include_empty_or_rejected
            and ev.qwen_verified
            and ev.qwen_species == "none"
            and not ev.verified_species_counts
        ):
            continue
        targets.append(ev)

    if not targets:
        logger.info("No eligible tracks for RAG scene description.")
        return [], {"described": 0, "skipped": 0, "rag_jsonl_path": "", "chromadb": None}

    rag_docs: List[RAGDocument] = []
    skipped = 0
    # Auto-cap multi-frame RAG batch size by GPU VRAM (T4 15GB -> B=2, L4/A100 -> B=4)
    batch_size, is_low_vram = _resolve_vram_safe_batch(config.qwen.batch_size, is_multiframe=True)
    _clear_cuda_cache_if_needed(is_low_vram)

    for i in tqdm(range(0, len(targets), batch_size), desc=f"Stage 3 [Qwen 3-Frame RAG B={batch_size}]"):
        chunk = targets[i : i + batch_size]
        valid_tracks: List[TrackManifestItem] = []
        frame_groups: List[List[Any]] = []
        contexts: List[Dict[str, Any]] = []

        for ev in chunk:
            candidate_paths = (
                ev.trusted_temporal_frame_paths
                or ev.temporal_frame_paths
                or [ev.trusted_bucket_image_path or ev.distilled_image_path or ev.raw_image_path]
            )
            loaded_pils = []
            for p_str in candidate_paths[: rag_cfg.temporal_frames_per_track]:
                raw_bgr = cv2.imread(str(p_str))
                if raw_bgr is not None:
                    loaded_pils.append(prepare_full_frame_pil(raw_bgr, max_dim=rag_cfg.max_image_dim))

            if not loaded_pils:
                skipped += 1
                continue

            frame_groups.append(loaded_pils)
            contexts.append(
                {
                    "species": ev.qwen_species or ev.species_yolo,
                    "video_source": ev.video_source,
                    "best_frame": ev.best_frame,
                    "temporal_frame_indices": ev.temporal_frame_indices,
                    "start_hms": ev.start_hms,
                    "end_hms": ev.end_hms,
                    "best_hms": ev.best_hms,
                    "species_counts": ev.verified_species_counts or ev.frame_species_counts,
                    "avg_conf": ev.avg_conf,
                }
            )
            valid_tracks.append(ev)

        if not valid_tracks:
            continue

        scene_results = vlm.describe_scenes_batch(frame_groups, scene_contexts=contexts)
        for ev, scene_res in zip(valid_tracks, scene_results):
            doc = build_rag_document(
                track=ev,
                scene=scene_res,
                language=rag_cfg.language,
                conf_review_thresh=config.video_scan.conf_review_thresh,
                purity_review_thresh=config.video_scan.purity_review_thresh,
                flag_count_mismatch=rag_cfg.flag_count_mismatch,
            )
            rag_docs.append(doc)
        _clear_cuda_cache_if_needed(is_low_vram)

    rag_jsonl_path = config.out_dataset_dir / rag_cfg.output_filename
    save_rag_documents_jsonl(rag_docs, rag_jsonl_path)

    chroma_stats: Optional[Dict[str, Any]] = None
    if rag_cfg.chromadb_dir is not None and rag_docs:
        chroma_stats = upsert_rag_documents_to_chromadb(
            docs=rag_docs,
            persist_dir=Path(rag_cfg.chromadb_dir),
            collection_name=rag_cfg.chromadb_collection,
            embedding_model_id=rag_cfg.embedding_model_id,
        )

    corpus_summary = summarize_rag_corpus(rag_docs)
    stats = {
        "described": len(rag_docs),
        "skipped": skipped,
        "rag_jsonl_path": str(rag_jsonl_path),
        "chromadb": chroma_stats,
        "corpus_summary": corpus_summary,
    }
    return rag_docs, stats


def _preflight_check_vlm_dependencies(config: PipelineConfig, mode_flags: Dict[str, bool]) -> None:
    """Fail fast (<1s) BEFORE Stage 1 if PEFT/torchao is incompatible in Colab."""
    will_use_vlm = (config.run_stage2_distill and mode_flags["run_qwen_verify"]) or (
        config.rag.enabled and mode_flags["run_qwen_rag"]
    )
    if not will_use_vlm or not config.qwen.lora_dir:
        return
    try:
        from peft.import_utils import is_torchao_available

        is_torchao_available()
    except ImportError as exc:
        if "torchao" in str(exc).lower():
            raise ImportError(
                f"{exc}\n"
                "Fix on Colab in 5 seconds: run `!pip uninstall -y torchao && pip install -q -U peft`, "
                "then restart runtime (or re-run cell) BEFORE scanning videos."
            ) from exc
    except Exception:
        pass


def _load_stage1_checkpoint(
    checkpoint_path: Path,
    video_paths: Sequence[str],
) -> Optional[Tuple[List[TrackManifestItem], List[Dict[str, Any]]]]:
    """Load saved Stage 1 manifest & stats if checkpoint exists for the exact same video filenames."""
    if not checkpoint_path.exists():
        return None
    try:
        data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        expected_vids = [Path(p).name for p in video_paths]
        if data.get("video_names") != expected_vids:
            return None
        manifest = [TrackManifestItem.from_dict(item) for item in data.get("manifest", [])]
        s1_stats = list(data.get("stage1_stats", []))
        logger.info(
            "Resumed Stage 1 from checkpoint %s (%d tracks) — skipped rescanning!",
            checkpoint_path,
            len(manifest),
        )
        return manifest, s1_stats
    except Exception as exc:
        logger.warning("Could not load Stage 1 checkpoint (%s); running Stage 1 fresh.", exc)
        return None


def run_distillation_and_rag_pipeline(
    video_paths: Sequence[str],
    config: PipelineConfig,
) -> Dict[str, Any]:
    """Execute video scan (with Stage 1 checkpointing), optional Qwen verify/RAG, and always emit full summary."""
    run_mode = config.video_scan.run_mode
    mode_flags = resolve_mode_flags(run_mode)

    # Preflight check VLM / PEFT compatibility BEFORE spending minutes scanning video
    _preflight_check_vlm_dependencies(config, mode_flags)

    # Stage 1: Resume from checkpoint if available, otherwise run YOLO + ByteTrack scan
    loaded_ckpt = (
        _load_stage1_checkpoint(config.stage1_checkpoint_path, video_paths)
        if (config.resume_from_stage1 and run_mode != "display_only")
        else None
    )

    if loaded_ckpt is not None:
        manifest, s1_stats = loaded_ckpt
    else:
        scanner = VideoScanner(config.video_scan)
        manifest, s1_stats = scanner.scan_videos(
            video_paths=video_paths,
            staging_dir=config.staging_dir,
            qwen_review_bucket_dir=config.qwen_review_bucket_dir,
            trusted_rag_bucket_dir=config.trusted_rag_bucket_dir,
            unload_after=True,
        )
        # Immediately checkpoint Stage 1 BEFORE loading Qwen3-VL so Stage 1 work is never lost
        if run_mode != "display_only":
            config.out_dataset_dir.mkdir(parents=True, exist_ok=True)
            config.stage1_checkpoint_path.write_text(
                json.dumps(
                    {
                        "video_names": [Path(p).name for p in video_paths],
                        "stage1_stats": s1_stats,
                        "manifest": [m.to_dict() for m in manifest],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            logger.info("Saved Stage 1 checkpoint to %s", config.stage1_checkpoint_path)

    if run_mode == "display_only":
        video_summary = build_video_summary(
            manifest=manifest,
            stage1_stats=s1_stats,
            run_mode=run_mode,
        )
        logger.info("run_mode='display_only' completed; skipping buckets, Qwen VLM, ChromaDB, and training.")
        return {
            "run_mode": run_mode,
            "should_run_pipeline": False,
            "run_yolo_training": False,
            "stage1_yolo_per_video": s1_stats,
            "stage2_qwen_grpo": {"skipped_stage": True, "reason": "display_only"},
            "stage3_qwen_rag": {"skipped_stage": True, "reason": "display_only"},
            "video_summary": video_summary,
            "video_summary_path": "",
        }

    config.out_dataset_dir.mkdir(parents=True, exist_ok=True)
    s2_stats: Optional[Dict[str, Any]] = None
    s3_stats: Optional[Dict[str, Any]] = None

    effective_run_stage2 = config.run_stage2_distill and mode_flags["run_qwen_verify"]
    effective_run_stage3 = config.rag.enabled and mode_flags["run_qwen_rag"]

    has_rag_targets = any(
        (m.is_trusted_for_rag if config.rag.trusted_only else True) for m in manifest
    )
    need_vlm = (effective_run_stage2 and any(m.needs_rl_review for m in manifest)) or (
        effective_run_stage3 and has_rag_targets
    )

    if need_vlm:
        with QwenWildlifeVLM(config.qwen) as vlm:
            if effective_run_stage2:
                s2_stats = run_qwen_distillation(
                    manifest=manifest,
                    vlm=vlm,
                    out_dataset_dir=config.out_dataset_dir,
                    batch_size=config.qwen.batch_size,
                    zoom_scale=config.qwen.zoom_scale,
                    save_debug=config.save_debug,
                )

            if effective_run_stage3:
                _, s3_stats = run_qwen_rag_description(
                    manifest=manifest,
                    vlm=vlm,
                    config=config,
                )

    video_summary = build_video_summary(
        manifest=manifest,
        stage1_stats=s1_stats,
        run_mode=run_mode,
        stage2_stats=s2_stats,
        stage3_stats=s3_stats,
    )

    manifest_path = config.out_dataset_dir / "track_manifest.json"
    manifest_path.write_text(
        json.dumps([m.to_dict() for m in manifest], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_json_path = config.out_dataset_dir / "video_summary.json"
    summary_json_path.write_text(
        json.dumps(video_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_md_path = config.out_dataset_dir / "video_summary.md"
    summary_md_path.write_text(video_summary["summary_markdown"], encoding="utf-8")

    report = {
        "run_mode": run_mode,
        "should_run_pipeline": mode_flags["run_yolo_training"],
        "run_yolo_training": mode_flags["run_yolo_training"],
        "qwen_review_bucket_dir": str(config.qwen_review_bucket_dir),
        "trusted_rag_bucket_dir": str(config.trusted_rag_bucket_dir),
        "stage1_yolo_per_video": s1_stats,
        "stage2_qwen_grpo": s2_stats or {"skipped_stage": True, "reason": f"disabled_in_{run_mode}"},
        "stage3_qwen_rag": s3_stats or {"skipped_stage": True, "reason": f"disabled_in_{run_mode}"},
        "track_manifest_path": str(manifest_path),
        "video_summary_path": str(summary_json_path),
        "video_summary_markdown_path": str(summary_md_path),
        "video_summary": video_summary,
    }

    report_path = config.out_dataset_dir / "pipeline_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Pipeline report & summary written to %s", config.out_dataset_dir)
    return report
