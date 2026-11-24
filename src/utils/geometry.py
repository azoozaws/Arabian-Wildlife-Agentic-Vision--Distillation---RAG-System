"""Geometry, formatting, and visual-prompt image preparation utilities."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image


def format_hms(seconds: float) -> str:
    """Convert elapsed seconds to HH:MM:SS string."""
    safe_sec = max(0.0, float(seconds))
    return time.strftime("%H:%M:%S", time.gmtime(safe_sec))


def norm_species(name: str) -> str:
    """Normalize species label names to lowercase snake_case."""
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def sanitize_tag(video_path: str) -> str:
    """Create a filesystem-safe short tag from a video file path."""
    return re.sub(r"[^\w\-]+", "_", Path(video_path).stem).strip("_")[:35] or "vid"


def xyxy_to_yolo(xyxy: Sequence[int], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Convert pixel [x1, y1, x2, y2] coordinates to normalized YOLO (cx, cy, w, h)."""
    x1, y1, x2, y2 = xyxy
    cx = float(np.clip(((x1 + x2) / 2.0) / img_w, 0.0, 1.0))
    cy = float(np.clip(((y1 + y2) / 2.0) / img_h, 0.0, 1.0))
    bw = float(np.clip((x2 - x1) / img_w, 0.0, 1.0))
    bh = float(np.clip((y2 - y1) / img_h, 0.0, 1.0))
    return cx, cy, bw, bh


def build_zoomed_visual_prompt(
    raw_bgr: np.ndarray,
    target_xyxy: Sequence[int],
    zoom_scale: float = 3.0,
    max_dim: int = 768,
) -> Image.Image:
    """Crop a contextual window around YOLO's box and draw a temporary red box for Qwen classification."""
    h, w = raw_bgr.shape[:2]
    x1, y1, x2, y2 = map(int, target_xyxy)
    bw, bh = max(16, x2 - x1), max(16, y2 - y1)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    crop_w = max(int(bw * zoom_scale), 224)
    crop_h = max(int(bh * zoom_scale), 224)
    rx1 = max(0, cx - crop_w // 2)
    ry1 = max(0, cy - crop_h // 2)
    rx2 = min(w, rx1 + crop_w)
    ry2 = min(h, ry1 + crop_h)

    zoomed = raw_bgr[ry1:ry2, rx1:rx2].copy()
    lx1, ly1 = max(0, x1 - rx1), max(0, y1 - ry1)
    lx2, ly2 = min(zoomed.shape[1] - 1, x2 - rx1), min(zoomed.shape[0] - 1, y2 - ry1)
    cv2.rectangle(zoomed, (lx1, ly1), (lx2, ly2), (0, 0, 255), 2)

    pil_img = Image.fromarray(cv2.cvtColor(zoomed, cv2.COLOR_BGR2RGB))
    pil_img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    return pil_img


def prepare_full_frame_pil(raw_bgr: np.ndarray, max_dim: int = 1024) -> Image.Image:
    """Convert a full BGR frame to an RGB PIL image scaled for VLM scene description."""
    pil_img = Image.fromarray(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB))
    pil_img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    return pil_img
