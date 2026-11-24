"""Side-by-side and per-bucket debug image generators for VLM audits and YOLO comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np
from PIL import Image

from app.src.config import CLASS_NAME_TO_ID, VALID_IMAGE_EXTS
from app.src.utils.logging_utils import get_logger

logger = get_logger("Visualization")


def save_track_bucket_debug(
    raw_bgr: np.ndarray,
    all_frame_boxes: List[Dict[str, Any]],
    target_track_id: int,
    out_path: Path,
    bucket_label: str,
    avg_conf: float,
    best_frame: int,
    best_hms: str,
    max_width: int = 960,
    jpeg_quality: int = 80,
) -> None:
    """Render a compact annotated debug frame for a Stage 1 confidence bucket."""
    canvas = raw_bgr.copy()
    h, w = canvas.shape[:2]

    for box in all_frame_boxes:
        x1, y1, x2, y2 = map(int, box["xyxy"])
        is_target = int(box["track_id"]) == target_track_id
        color = (0, 230, 0) if ("TRUSTED" in bucket_label.upper()) else (0, 165, 255)
        if not is_target:
            color = (180, 180, 180)
        thickness = 3 if is_target else 1
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        lbl = f"ID:{box['track_id']} {box.get('species', box['class_id'])} {box['conf']:.2f}"
        cv2.putText(
            canvas,
            lbl,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    cv2.rectangle(canvas, (0, 0), (w, 64), (25, 25, 25), -1)
    header = (
        f"{bucket_label} | trk={target_track_id} | avg_conf={avg_conf:.2f} | "
        f"frame={best_frame} ({best_hms})"
    )
    cv2.putText(canvas, header, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    cv2.putText(
        canvas,
        "Colors: Green = Trusted (>=75%) | Orange = Needs Review (<75%) | Gray = Other Tracks in Frame",
        (12, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (200, 210, 220),
        1,
    )

    if max_width > 0 and w > max_width:
        scale = max_width / float(w)
        canvas = cv2.resize(canvas, (max_width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])


def save_side_by_side_qwen_debug(
    raw_bgr: np.ndarray,
    zoomed_prompt_pil: Image.Image,
    out_path: Path,
    yolo_species: str,
    yolo_conf: float,
    yolo_xyxy: Sequence[int],
    qwen_species: str,
) -> None:
    """Render LEFT = full frame with YOLO box, RIGHT = 3x zoomed prompt with Qwen verdict."""
    h, w = raw_bgr.shape[:2]

    left = raw_bgr.copy()
    x1, y1, x2, y2 = map(int, yolo_xyxy)
    cv2.rectangle(left, (x1, y1), (x2, y2), (0, 165, 255), 3)
    cv2.rectangle(left, (0, 0), (w, 42), (30, 30, 30), -1)
    cv2.putText(
        left,
        f"YOLO: {yolo_species} (avg_conf={yolo_conf:.2f})",
        (14, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 200, 255),
        2,
    )

    zoom_bgr = cv2.cvtColor(np.array(zoomed_prompt_pil), cv2.COLOR_RGB2BGR)
    zh, zw = zoom_bgr.shape[:2]
    scale = h / max(1, zh)
    right = cv2.resize(zoom_bgr, (int(zw * scale), h))
    rw = right.shape[1]

    is_pos = qwen_species in CLASS_NAME_TO_ID
    hdr_txt = f"QWEN GRPO: VERIFIED ({qwen_species})" if is_pos else "QWEN GRPO: REJECTED (NONE)"
    hdr_col = (0, 230, 0) if is_pos else (0, 0, 255)
    cv2.rectangle(right, (0, 0), (rw, 42), (30, 30, 30), -1)
    cv2.putText(right, hdr_txt, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.68, hdr_col, 2)

    divider = np.full((h, 6, 3), 255, dtype=np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.hstack([left, divider, right]))


def save_model_disagreements(
    baseline_model: Any,
    candidate_model: Any,
    test_images_dir: Path,
    out_diff_dir: Path,
    conf_thresh: float = 0.25,
    baseline_label: str = "YOLO Champion",
    candidate_label: str = "YOLO Challenger",
) -> int:
    """Save side-by-side comparison images whenever champion and challenger models disagree."""
    out_diff_dir.mkdir(parents=True, exist_ok=True)
    test_imgs = sorted(p for p in test_images_dir.iterdir() if p.suffix.lower() in VALID_IMAGE_EXTS)
    saved_diffs = 0

    for img_p in test_imgs:
        r1 = baseline_model(str(img_p), verbose=False, conf=conf_thresh)[0]
        r2 = candidate_model(str(img_p), verbose=False, conf=conf_thresh)[0]

        cls1 = sorted(int(c) for c in r1.boxes.cls.tolist()) if len(r1.boxes) else []
        cls2 = sorted(int(c) for c in r2.boxes.cls.tolist()) if len(r2.boxes) else []
        conf1 = float(r1.boxes.conf.mean().item()) if len(r1.boxes) else 0.0
        conf2 = float(r2.boxes.conf.mean().item()) if len(r2.boxes) else 0.0

        # Flag disagreement if predicted classes/counts differ or average confidence shifted > 0.20
        if cls1 != cls2 or abs(conf1 - conf2) > 0.20:
            left = r1.plot()
            right = r2.plot()
            h, w = left.shape[:2]

            cv2.rectangle(left, (0, 0), (w, 40), (30, 30, 30), -1)
            cv2.putText(
                left,
                f"{baseline_label} | Detections: {len(cls1)} (avg={conf1:.2f})",
                (12, 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 255),
                2,
            )

            cv2.rectangle(right, (0, 0), (w, 40), (30, 30, 30), -1)
            cv2.putText(
                right,
                f"{candidate_label} | Detections: {len(cls2)} (avg={conf2:.2f})",
                (12, 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 230, 0),
                2,
            )

            divider = np.full((h, 6, 3), 255, dtype=np.uint8)
            cv2.imwrite(str(out_diff_dir / f"diff_{img_p.stem}.jpg"), np.hstack([left, divider, right]))
            saved_diffs += 1

    logger.info("Saved %d model disagreement debug images to %s", saved_diffs, out_diff_dir)
    return saved_diffs
