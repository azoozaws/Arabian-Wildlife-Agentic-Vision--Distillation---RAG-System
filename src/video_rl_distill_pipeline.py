"""CLI entrypoint for Stage 1 (scan/display + dual buckets), Stage 2 (Qwen GRPO), and Stage 3 (RAG -> ChromaDB)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.src.config import (
    DEFAULT_QWEN_MODEL_ID,
    PipelineConfig,
    QwenVLMConfig,
    RAGDescriptionConfig,
    VideoScanConfig,
)
from app.src.pipelines.distillation_pipeline import run_distillation_and_rag_pipeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Arabian Wildlife Video Scan, Display Modes, Dual Buckets & ChromaDB RAG Pipeline"
    )
    parser.add_argument("--video_paths", nargs="+", required=True, help="One or more input video file paths")
    parser.add_argument("--yolo_weights", type=str, required=True, help="Path to YOLO .pt weights")
    parser.add_argument(
        "--run_mode",
        type=str,
        choices=[
            "display_only",
            "summary_only",
            "summary_with_rag",
            "summary_with_qwen",
            "train_only",
            "display_and_train",
        ],
        default="train_only",
        help=(
            "display_only: live view + summary | "
            "summary_only: scan + buckets + summary (no Qwen/training) | "
            "summary_with_rag: scan + Qwen 3-frame RAG + summary (no verify/training) | "
            "summary_with_qwen: scan + Qwen verify + RAG + summary (no training) | "
            "train_only / display_and_train: full workflow including YOLO training"
        ),
    )
    parser.add_argument("--lora_dir", type=str, default=None, help="Path to Qwen3-VL GRPO LoRA adapter directory")
    parser.add_argument("--qwen_model_id", type=str, default=DEFAULT_QWEN_MODEL_ID)
    parser.add_argument("--out_dataset_dir", type=str, required=True, help="Output workspace for buckets & RAG JSONL")
    parser.add_argument("--conf_min", type=float, default=0.15)
    parser.add_argument(
        "--conf_review_thresh",
        type=float,
        default=0.95,
        help="Tracks below this threshold go to qwen_review_below_95/{images,labels,debug}",
    )
    parser.add_argument(
        "--trusted_rag_min_conf",
        type=float,
        default=0.75,
        help="Tracks at or above this threshold go to trusted_rag_above_75/{images,labels,debug} -> Qwen RAG -> ChromaDB",
    )
    parser.add_argument("--purity_review_thresh", type=float, default=0.80)
    parser.add_argument("--audit_all", action="store_true")
    parser.add_argument("--idle_stride", type=int, default=30)
    parser.add_argument("--active_stride", type=int, default=5)
    parser.add_argument("--qwen_batch", type=int, default=8)
    parser.add_argument("--skip_distill", action="store_true", help="Skip Stage 2 closed-set Qwen classification")
    parser.add_argument("--skip_rag", action="store_true", help="Skip Stage 3 RAG scene description")
    parser.add_argument(
        "--rag_language",
        type=str,
        choices=["ar", "en", "bilingual"],
        default="ar",
        help="Primary narrative language for RAG documents",
    )
    parser.add_argument(
        "--chromadb_dir",
        type=str,
        default=None,
        help="Optional persistent ChromaDB directory for automatic RAG ingestion",
    )
    parser.add_argument("--chromadb_collection", type=str, default="arabian_wildlife_rag")
    parser.add_argument("--include_rejected_in_rag", action="store_true")
    parser.add_argument("--use_adapter_for_description", action="store_true")
    parser.add_argument("--no_debug", action="store_true", help="Disable debug image generation")
    parser.add_argument(
        "--clean_demo_display",
        action="store_true",
        help="Use clean emerald-only recording overlay (no confidence/orange) for live display modes",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    config = PipelineConfig(
        out_dataset_dir=Path(args.out_dataset_dir),
        video_scan=VideoScanConfig(
            yolo_weights=args.yolo_weights,
            run_mode=args.run_mode,
            conf_min=args.conf_min,
            conf_review_thresh=args.conf_review_thresh,
            trusted_rag_min_conf=args.trusted_rag_min_conf,
            purity_review_thresh=args.purity_review_thresh,
            idle_stride=args.idle_stride,
            active_stride=args.active_stride,
            audit_all=args.audit_all,
            save_bucket_debug=not args.no_debug,
            display_clean_demo_mode=args.clean_demo_display,
        ),
        qwen=QwenVLMConfig(
            model_id=args.qwen_model_id,
            lora_dir=args.lora_dir,
            batch_size=args.qwen_batch,
            use_adapter_for_description=args.use_adapter_for_description,
        ),
        rag=RAGDescriptionConfig(
            enabled=not args.skip_rag,
            language=args.rag_language,
            trusted_only=True,
            include_empty_or_rejected=args.include_rejected_in_rag,
            chromadb_dir=Path(args.chromadb_dir) if args.chromadb_dir else None,
            chromadb_collection=args.chromadb_collection,
        ),
        run_stage2_distill=not args.skip_distill,
        save_debug=not args.no_debug,
    )

    report = run_distillation_and_rag_pipeline(video_paths=args.video_paths, config=config)
    summary_md = (report.get("video_summary") or {}).get("summary_markdown", "")
    if summary_md:
        print("\n" + summary_md)
    print("\n================ END-TO-END BATCH REPORT ================")
    compact_report = {k: v for k, v in report.items() if k != "video_summary"}
    print(json.dumps(compact_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
