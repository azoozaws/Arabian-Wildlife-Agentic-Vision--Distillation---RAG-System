"""Split YOLO training and Champion-vs-Challenger evaluation services aligned with WildlifeAgentState."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import cv2

from app.src.agent.agent_state import MetricsSnapshot, load_yolo_cached
from app.src.config import TrainingConfig, WandbConfig
from app.src.training.model_registry import (
    DEFAULT_MODELS_DIR,
    DEFAULT_REGISTRY_PATH,
    ModelRegistry,
)
from app.src.utils.logging_utils import get_logger
from app.src.utils.visualization import save_model_disagreements

logger = get_logger("YOLOTrainer")


def configure_wandb_env(wandb_cfg: WandbConfig) -> None:
    """Configure W&B environment variables cleanly without hardcoding secrets in source code."""
    os.environ.pop("WANDB_START_METHOD", None)
    os.environ.pop("WANDB_NOTEBOOK_NAME", None)

    if not wandb_cfg.enabled or not wandb_cfg.api_key:
        os.environ["WANDB_DISABLED"] = "true"
        logger.info("W&B logging disabled (set WANDB_API_KEY in environment/.env to enable).")
        return

    os.environ.pop("WANDB_DISABLED", None)
    os.environ["WANDB_API_KEY"] = wandb_cfg.api_key
    if wandb_cfg.base_url:
        os.environ["WANDB_BASE_URL"] = wandb_cfg.base_url
    if wandb_cfg.entity:
        os.environ["WANDB_ENTITY"] = wandb_cfg.entity
    if wandb_cfg.project:
        os.environ["WANDB_PROJECT"] = wandb_cfg.project
    os.environ["WANDB_DISABLE_CODE"] = "true" if wandb_cfg.disable_code else "false"
    os.environ["WANDB_CONSOLE"] = wandb_cfg.console


def compute_fp_per_hour(
    model: Any,
    held_out_video_path: str,
    conf_thresh: float = 0.25,
    frame_stride: int = 30,
    max_eval_minutes: float = 15.0,
) -> float:
    """Measure false positives per hour on a held-out negative (empty desert) video.

    Caps evaluation to `max_eval_minutes` (when > 0) so multi-hour negative videos
    evaluate in seconds while preserving an accurate FP/hour rate.
    """
    if not held_out_video_path:
        return 0.0

    video_file = Path(held_out_video_path)
    if not video_file.exists():
        logger.warning("Held-out negative video not found (%s); defaulting fp_per_hour=0.0.", video_file)
        return 0.0

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        logger.warning("Could not open held-out negative video (%s); defaulting fp_per_hour=0.0.", video_file)
        return 0.0

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, int(frame_stride))
    max_frames = int(max_eval_minutes * 60.0 * fps) if max_eval_minutes > 0 else 0
    processed_frames = 0
    fp_detections = 0

    try:
        frame_idx = 0
        while cap.isOpened():
            if max_frames > 0 and frame_idx >= max_frames:
                break
            if stride > 1 and frame_idx > 0:
                for _ in range(stride - 1):
                    if not cap.grab():
                        break
                    frame_idx += 1
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            processed_frames += 1

            res = model(frame, verbose=False, conf=conf_thresh)[0]
            if res.boxes is not None and len(res.boxes) > 0:
                fp_detections += len(res.boxes)
    finally:
        cap.release()

    evaluated_frames = frame_idx if frame_idx > 0 else (processed_frames * stride)
    if max_frames == 0 and total_frames > 0:
        evaluated_frames = total_frames
    duration_hours = (evaluated_frames / fps) / 3600.0
    if duration_hours <= 0.0:
        return 0.0

    # Scale sampled detections back to full evaluated window FP/hr estimate
    estimated_total_fp = fp_detections * stride
    return round(estimated_total_fp / duration_hours, 4)


def evaluate_single_model(
    weights_path: str,
    eval_yaml_path: str,
    held_out_video_path: str = "",
    split: str = "test",
    fp_conf_thresh: float = 0.25,
    fp_frame_stride: int = 30,
    fp_max_eval_minutes: float = 15.0,
) -> MetricsSnapshot:
    """Evaluate a single YOLO checkpoint and return a MetricsSnapshot compatible with eval_gate_node."""
    model = load_yolo_cached(str(weights_path))
    val_res = model.val(data=str(eval_yaml_path), split=split, verbose=False)
    fp_hr = compute_fp_per_hour(
        model=model,
        held_out_video_path=held_out_video_path,
        conf_thresh=fp_conf_thresh,
        frame_stride=fp_frame_stride,
        max_eval_minutes=fp_max_eval_minutes,
    )
    return MetricsSnapshot(
        precision=round(float(val_res.box.mp), 4),
        recall=round(float(val_res.box.mr), 4),
        mAP50=round(float(val_res.box.map50), 4),
        fp_per_hour=fp_hr,
    )


def evaluate_champion_vs_challenger(
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
    """Evaluate Champion and Challenger checkpoints and return state updates for eval_gate_node."""
    logger.info("Evaluating Champion model (%s) on split='%s'...", champion_weights_path, split)
    champion_metrics = evaluate_single_model(
        weights_path=champion_weights_path,
        eval_yaml_path=eval_yaml_path,
        held_out_video_path=held_out_video_path,
        split=split,
        fp_conf_thresh=fp_conf_thresh,
        fp_frame_stride=fp_frame_stride,
    )

    logger.info("Evaluating Challenger model (%s) on split='%s'...", challenger_weights_path, split)
    challenger_metrics = evaluate_single_model(
        weights_path=challenger_weights_path,
        eval_yaml_path=eval_yaml_path,
        held_out_video_path=held_out_video_path,
        split=split,
        fp_conf_thresh=fp_conf_thresh,
        fp_frame_stride=fp_frame_stride,
    )

    diff_count = 0
    if out_dir is not None and test_images_dir is not None and test_images_dir.exists():
        champ_model = load_yolo_cached(str(champion_weights_path))
        chall_model = load_yolo_cached(str(challenger_weights_path))
        diff_count = save_model_disagreements(
            baseline_model=champ_model,
            candidate_model=chall_model,
            test_images_dir=test_images_dir,
            out_diff_dir=out_dir / f"champion_vs_{version_tag}_debug_diffs",
            conf_thresh=diff_conf_thresh,
            baseline_label="YOLO Champion",
            candidate_label=f"YOLO Challenger ({version_tag})",
        )

    eval_report_path = ""
    result: Dict[str, Any] = {
        "champion_metrics": champion_metrics,
        "challenger_metrics": challenger_metrics,
        "disagreement_debug_images_saved": diff_count,
    }

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        report_file = out_dir / f"champion_vs_{version_tag}_metrics.json"
        report_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        eval_report_path = str(report_file)
        logger.info("Saved evaluation report to %s", report_file)

    result["eval_report_path"] = eval_report_path
    return result


class YOLOTrainer:
    """Handles incremental fine-tuning separately from Champion-vs-Challenger evaluation."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def train(self, data_yaml_path: Path) -> Dict[str, Any]:
        """Fine-tune a Challenger YOLO model from baseline (Champion) weights without running gate eval."""
        configure_wandb_env(self.config.wandb)
        from ultralytics import YOLO

        logger.info(
            "Starting incremental fine-tuning (%s) from %s...",
            self.config.run_name,
            self.config.baseline_weights,
        )
        candidate_model = YOLO(str(self.config.baseline_weights))
        train_results = candidate_model.train(
            data=str(data_yaml_path),
            epochs=self.config.epochs,
            imgsz=self.config.imgsz,
            batch=self.config.batch,
            optimizer=self.config.optimizer,
            lr0=self.config.lr0,
            lrf=self.config.lrf,
            cos_lr=self.config.cos_lr,
            warmup_epochs=self.config.warmup_epochs,
            warmup_bias_lr=self.config.warmup_bias_lr,
            patience=self.config.patience,
            mosaic=self.config.mosaic,
            close_mosaic=self.config.close_mosaic,
            hsv_h=self.config.hsv_h,
            hsv_s=self.config.hsv_s,
            hsv_v=self.config.hsv_v,
            fliplr=self.config.fliplr,
            scale=self.config.scale,
            project=str(self.config.project_dir),
            name=self.config.run_name,
            exist_ok=True,
        )

        run_dir = Path(train_results.save_dir)
        best_weights_path = run_dir / "weights" / "best.pt"

        # Copy standalone weights into run_dir for clear Challenger tracking
        model_id = self.config.model_id or f"yolo_{self.config.version_tag}_best"
        exported_weights = run_dir / f"{model_id}.pt"
        shutil.copy2(best_weights_path, exported_weights)

        saved_zip_path = ""
        if self.config.archive_dataset_zip:
            zip_base = str(run_dir / f"YOLO11_Dataset_{self.config.version_tag}")
            shutil.make_archive(zip_base, "zip", root_dir=str(self.config.merged_dataset))
            saved_zip_path = f"{zip_base}.zip"
            logger.info("Archived merged dataset -> %s", saved_zip_path)

        # Register the new Challenger model inside app/models/ and app/registry.json
        registry_path = self.config.registry_path or DEFAULT_REGISTRY_PATH
        models_dir = self.config.models_dir or DEFAULT_MODELS_DIR
        registry = ModelRegistry(registry_path=registry_path, models_dir=models_dir)
        reg_entry = registry.register_trained_model(
            source_weights_path=exported_weights,
            model_id=model_id,
            version_tag=self.config.version_tag,
            parent_model_id=self.config.parent_model_id or Path(self.config.baseline_weights).stem,
            training_metadata={
                "run_dir": str(run_dir),
                "run_name": self.config.run_name,
                "eval_yaml_path": str(data_yaml_path),
                "saved_dataset_zip": saved_zip_path,
                "epochs": self.config.epochs,
                "imgsz": self.config.imgsz,
                "batch": self.config.batch,
                "replay_samples": self.config.replay_samples,
            },
            status="candidate",
            copy_to_models_dir=True,
        )

        return {
            "champion_weights_path": str(self.config.baseline_weights),
            "challenger_model_id": model_id,
            "challenger_weights_path": reg_entry["weights_path"],
            "best_weights_raw": str(best_weights_path),
            "eval_yaml_path": str(data_yaml_path),
            "saved_dataset_zip": saved_zip_path,
            "run_dir": str(run_dir),
            "registry_path": str(registry_path),
            "models_dir": str(models_dir),
        }

    def evaluate(
        self,
        challenger_weights_path: Path | str,
        eval_yaml_path: Path | str,
        held_out_video_path: str = "",
        out_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Evaluate Champion (baseline_weights) vs Challenger and return MetricsSnapshots for eval_gate_node."""
        test_img_dir = self.config.merged_dataset / self.config.eval_split / "images"
        return evaluate_champion_vs_challenger(
            champion_weights_path=str(self.config.baseline_weights),
            challenger_weights_path=str(challenger_weights_path),
            eval_yaml_path=str(eval_yaml_path),
            held_out_video_path=held_out_video_path,
            split=self.config.eval_split,
            out_dir=out_dir,
            test_images_dir=test_img_dir if test_img_dir.exists() else None,
            version_tag=self.config.version_tag,
            diff_conf_thresh=self.config.diff_conf_thresh,
            fp_conf_thresh=self.config.fp_conf_thresh,
            fp_frame_stride=self.config.fp_frame_stride,
        )

    def train_and_evaluate(
        self,
        data_yaml_path: Path,
        held_out_video_path: str = "",
    ) -> Dict[str, Any]:
        """Convenience wrapper combining train() and evaluate() for standalone CLI runs."""
        train_out = self.train(data_yaml_path=data_yaml_path)
        eval_out = self.evaluate(
            challenger_weights_path=train_out["challenger_weights_path"],
            eval_yaml_path=train_out["eval_yaml_path"],
            held_out_video_path=held_out_video_path,
            out_dir=Path(train_out["run_dir"]),
        )
        return {**train_out, **eval_out}
