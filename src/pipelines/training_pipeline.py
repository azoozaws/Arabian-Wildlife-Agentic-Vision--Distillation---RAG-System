"""Split training and evaluation pipelines ready for standalone CLI or LangGraph Agent execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from app.data.dataset_builder import build_merged_dataset
from app.src.config import TrainingConfig
from app.src.training.yolo_trainer import YOLOTrainer, evaluate_champion_vs_challenger
from app.src.utils.logging_utils import get_logger

logger = get_logger("TrainingPipeline")


def run_training_stage(config: TrainingConfig) -> Dict[str, Any]:
    """Stage A (Train): Merge Qwen + Replay dataset and fine-tune the Challenger YOLO model."""
    yaml_path = build_merged_dataset(
        orig_root=config.orig_dataset,
        qwen_root=config.qwen_dataset,
        out_root=config.merged_dataset,
        replay_samples=config.replay_samples,
        seed=config.seed,
        yaml_filename=f"data_{config.version_tag}.yaml",
    )

    trainer = YOLOTrainer(config)
    train_summary = trainer.train(data_yaml_path=yaml_path)
    train_summary["data_yaml_path"] = str(yaml_path)
    return train_summary


def run_evaluation_stage(
    champion_weights_path: str,
    challenger_weights_path: str,
    eval_yaml_path: str,
    held_out_video_path: str = "",
    split: str = "test",
    out_dir: Optional[Path] = None,
    test_images_dir: Optional[Path] = None,
    version_tag: str = "v3",
    diff_conf_thresh: float = 0.25,
    fp_conf_thresh: float = 0.25,
    fp_frame_stride: int = 10,
) -> Dict[str, Any]:
    """Stage B (Evaluate): Evaluate Champion vs Challenger to feed directly into eval_gate_node."""
    return evaluate_champion_vs_challenger(
        champion_weights_path=champion_weights_path,
        challenger_weights_path=challenger_weights_path,
        eval_yaml_path=eval_yaml_path,
        held_out_video_path=held_out_video_path,
        split=split,
        out_dir=out_dir,
        test_images_dir=test_images_dir,
        version_tag=version_tag,
        diff_conf_thresh=diff_conf_thresh,
        fp_conf_thresh=fp_conf_thresh,
        fp_frame_stride=fp_frame_stride,
    )


def run_training_pipeline(
    config: TrainingConfig,
    held_out_video_path: str = "",
) -> Dict[str, Any]:
    """End-to-end convenience wrapper running Stage A (Train) followed by Stage B (Evaluate)."""
    train_out = run_training_stage(config)
    test_img_dir = config.merged_dataset / config.eval_split / "images"
    eval_out = run_evaluation_stage(
        champion_weights_path=train_out["champion_weights_path"],
        challenger_weights_path=train_out["challenger_weights_path"],
        eval_yaml_path=train_out["eval_yaml_path"],
        held_out_video_path=held_out_video_path,
        split=config.eval_split,
        out_dir=Path(train_out["run_dir"]),
        test_images_dir=test_img_dir if test_img_dir.exists() else None,
        version_tag=config.version_tag,
        diff_conf_thresh=config.diff_conf_thresh,
        fp_conf_thresh=config.fp_conf_thresh,
        fp_frame_stride=config.fp_frame_stride,
    )
    return {**train_out, **eval_out}
