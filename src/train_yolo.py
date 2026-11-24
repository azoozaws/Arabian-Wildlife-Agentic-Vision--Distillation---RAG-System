"""CLI entrypoint supporting split YOLO training, evaluation (for eval_gate), or both."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.src.config import TrainingConfig, WandbConfig
from app.src.pipelines.training_pipeline import (
    run_evaluation_stage,
    run_training_pipeline,
    run_training_stage,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split Incremental YOLO Training & Champion-vs-Challenger Evaluation CLI"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["train", "eval", "both"],
        default="both",
        help="Run training only, evaluation only, or both sequentially",
    )
    parser.add_argument("--orig_dataset", type=str, default="./dataset_local/YOLO11_Dataset")
    parser.add_argument("--qwen_dataset", type=str, default="")
    parser.add_argument("--merged_dataset", type=str, default="./dataset_local/YOLO11_Dataset_v3")
    parser.add_argument(
        "--champion_weights",
        "--baseline_weights",
        "--yolo_v1_weights",
        dest="champion_weights",
        type=str,
        required=True,
        help="Path to Champion (baseline) YOLO weights",
    )
    parser.add_argument(
        "--challenger_weights",
        type=str,
        default="",
        help="Path to Challenger YOLO weights (required when --stage eval)",
    )
    parser.add_argument(
        "--eval_yaml",
        type=str,
        default="",
        help="Path to evaluation data.yaml (required when --stage eval)",
    )
    parser.add_argument(
        "--held_out_video",
        type=str,
        default="",
        help="Optional empty/negative held-out video for fp_per_hour computation",
    )
    parser.add_argument("--replay_samples", type=int, default=400, help="0 = use all original train images")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--version_tag", type=str, default="v4")
    parser.add_argument("--run_name", type=str, default="yolo_v4_rl_distilled")
    parser.add_argument("--model_id", type=str, default=None, help="Model ID to store in app/models and registry.json")
    parser.add_argument("--models_dir", type=str, default=None, help="Models directory (defaults to app/models)")
    parser.add_argument("--registry_path", type=str, default=None, help="Registry JSON path (defaults to app/registry.json)")
    parser.add_argument("--project_dir", type=str, default="./runs/yolo_v4_runs")
    parser.add_argument("--no_zip", action="store_true", help="Skip archiving merged dataset to ZIP")
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.stage == "eval":
        if not args.challenger_weights or not args.eval_yaml:
            raise SystemExit("--stage eval requires --challenger_weights and --eval_yaml.")
        merged_dir = Path(args.merged_dataset)
        test_img_dir = merged_dir / "test" / "images"
        summary = run_evaluation_stage(
            champion_weights_path=args.champion_weights,
            challenger_weights_path=args.challenger_weights,
            eval_yaml_path=args.eval_yaml,
            held_out_video_path=args.held_out_video,
            out_dir=Path(args.project_dir) / args.run_name,
            test_images_dir=test_img_dir if test_img_dir.exists() else None,
            version_tag=args.version_tag,
        )
    else:
        if not args.qwen_dataset:
            raise SystemExit("--qwen_dataset is required for --stage train or --stage both.")
        config = TrainingConfig(
            orig_dataset=Path(args.orig_dataset),
            qwen_dataset=Path(args.qwen_dataset),
            merged_dataset=Path(args.merged_dataset),
            baseline_weights=Path(args.champion_weights),
            project_dir=Path(args.project_dir),
            version_tag=args.version_tag,
            run_name=args.run_name,
            replay_samples=args.replay_samples,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            archive_dataset_zip=not args.no_zip,
            model_id=args.model_id,
            models_dir=Path(args.models_dir) if args.models_dir else None,
            registry_path=Path(args.registry_path) if args.registry_path else None,
            wandb=WandbConfig(enabled=not args.no_wandb),
        )
        if args.stage == "train":
            summary = run_training_stage(config)
        else:
            summary = run_training_pipeline(config, held_out_video_path=args.held_out_video)

    print(f"\n================ YOLO STAGE: {args.stage.upper()} ================")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
