"""Stage 1 video scanner with display modes, adaptive frame skipping, and dual confidence buckets."""

from __future__ import annotations

import gc
import os
import shutil
import sys
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from app.src.config import VideoScanConfig, resolve_mode_flags
from app.src.schemas import TrackManifestItem, TrackRecord
from app.src.utils.geometry import format_hms, norm_species, sanitize_tag, xyxy_to_yolo
from app.src.utils.logging_utils import get_logger
from app.src.utils.visualization import save_track_bucket_debug

logger = get_logger("VideoScanner")


def export_track_to_bucket(
    bucket_dir: Path,
    stem: str,
    raw_image_path: Path,
    all_frame_boxes: List[Dict[str, Any]],
    target_track_id: int,
    avg_conf: float,
    best_frame: int,
    best_hms: str,
    bucket_label: str,
    save_debug: bool = True,
    temporal_frame_paths: Optional[Sequence[str]] = None,
    preloaded_bgr: Optional[np.ndarray] = None,
) -> Tuple[str, str, Optional[str], List[str]]:
    """Export raw image, YOLO .txt label, optional debug frame, and up to 3 temporal frames into a bucket.

    Bucket layout:
        <bucket_dir>/
            images/<stem>.jpg
            images/<stem>_t1.jpg ... _t3.jpg   (when temporal_frame_paths provided)
            labels/<stem>.txt
            debug/<stem>_debug.jpg
    """
    raw_bgr = preloaded_bgr if preloaded_bgr is not None else cv2.imread(str(raw_image_path))
    if raw_bgr is None:
        raise FileNotFoundError(f"Could not read staged frame: {raw_image_path}")

    img_h, img_w = raw_bgr.shape[:2]
    images_dir = bucket_dir / "images"
    labels_dir = bucket_dir / "labels"
    debug_dir = bucket_dir / "debug"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    dst_img = images_dir / f"{stem}.jpg"
    dst_lbl = labels_dir / f"{stem}.txt"
    shutil.copy2(raw_image_path, dst_img)

    exported_temporal: List[str] = []
    if temporal_frame_paths:
        for t_idx, src_p_str in enumerate(list(temporal_frame_paths)[:3], start=1):
            src_p = Path(src_p_str)
            if src_p.exists():
                dst_t = images_dir / f"{stem}_t{t_idx}.jpg"
                shutil.copy2(src_p, dst_t)
                exported_temporal.append(str(dst_t))

    label_lines: List[str] = []
    for box in all_frame_boxes:
        cls_id = int(box["class_id"])
        cx, cy, bw, bh = xyxy_to_yolo(box["xyxy"], img_w, img_h)
        label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    dst_lbl.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

    dst_dbg_str: Optional[str] = None
    if save_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)
        dst_dbg = debug_dir / f"{stem}_debug.jpg"
        save_track_bucket_debug(
            raw_bgr=raw_bgr,
            all_frame_boxes=all_frame_boxes,
            target_track_id=target_track_id,
            out_path=dst_dbg,
            bucket_label=bucket_label,
            avg_conf=avg_conf,
            best_frame=best_frame,
            best_hms=best_hms,
        )
        dst_dbg_str = str(dst_dbg)

    return str(dst_img), str(dst_lbl), dst_dbg_str, exported_temporal


def collect_chronological_track_frames(ev: TrackRecord) -> Tuple[List[str], List[int]]:
    """Collect up to 3 unique chronological (start, best, end) frame paths and indices for a track."""
    candidates: List[Tuple[int, str]] = []
    if ev.first_frame_path and Path(ev.first_frame_path).exists():
        candidates.append((ev.start_frame, ev.first_frame_path))
    if ev.raw_image_path and Path(ev.raw_image_path).exists():
        candidates.append((ev.best_frame, ev.raw_image_path))
    if ev.last_frame_path and Path(ev.last_frame_path).exists():
        candidates.append((ev.last_saved_frame_idx if ev.last_saved_frame_idx >= 0 else ev.end_frame, ev.last_frame_path))

    # Deduplicate by frame index while preserving chronological order
    seen_frames = set()
    ordered: List[Tuple[int, str]] = []
    for f_idx, f_path in sorted(candidates, key=lambda item: item[0]):
        if f_idx in seen_frames:
            continue
        seen_frames.add(f_idx)
        ordered.append((f_idx, f_path))

    return [p for _, p in ordered[:3]], [f for f, _ in ordered[:3]]


_SPECIES_SHORT_LABEL: Dict[str, str] = {
    "arabian_oryx": "Oryx",
    "oryx": "Oryx",
    "arabian_gazelle": "Gazelle",
    "gazelle": "Gazelle",
}


def _classify_species_key(raw_species: str) -> str:
    """Normalize any model species name ('Oryx', 'arabian_oryx', 'Gazelle', etc.) to canonical key."""
    low = str(raw_species or "").strip().lower()
    if "oryx" in low:
        return "arabian_oryx"
    if "gazelle" in low:
        return "arabian_gazelle"
    return low


@lru_cache(maxsize=8)
def _load_ui_font(size: int, bold: bool = True) -> Any:
    """Load a crisp TrueType font (Windows Segoe UI / Arial, or Linux DejaVu) with graceful fallback."""
    candidates = (
        [
            r"C:\Windows\Fonts\seguisb.ttf",
            r"C:\Windows\Fonts\arialbd.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
        if bold
        else [
            r"C:\Windows\Fonts\segoeui.ttf",
            r"C:\Windows\Fonts\arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )
    for font_path in candidates:
        if Path(font_path).exists():
            try:
                return ImageFont.truetype(font_path, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def _draw_emerald_corner_brackets(
    draw: Any,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color_rgba: Tuple[int, int, int, int],
    shadow_rgba: Tuple[int, int, int, int],
    thickness: int = 3,
) -> None:
    """Draw sleek green tracking corner brackets instead of a heavy full rectangle."""
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    arm = max(8, min(22, int(min(bw, bh) * 0.28)))

    segments = [
        # Top-left
        ((x1, y1), (x1 + arm, y1)),
        ((x1, y1), (x1, y1 + arm)),
        # Top-right
        ((x2 - arm, y1), (x2, y1)),
        ((x2, y1), (x2, y1 + arm)),
        # Bottom-left
        ((x1, y2), (x1 + arm, y2)),
        ((x1, y2 - arm), (x1, y2)),
        # Bottom-right
        ((x2 - arm, y2), (x2, y2)),
        ((x2, y2 - arm), (x2, y2)),
    ]
    for p1, p2 in segments:
        draw.line([(p1[0] + 1, p1[1] + 1), (p2[0] + 1, p2[1] + 1)], fill=shadow_rgba, width=thickness + 1)
    for p1, p2 in segments:
        draw.line([p1, p2], fill=color_rgba, width=thickness)


def _render_clean_demo_overlay(
    vis_bgr: np.ndarray,
    frame_boxes: List[Dict[str, Any]],
    scale: float,
    hms: str,
    total_hms: str,
    paused: bool,
) -> np.ndarray:
    """Render a temporary cinema-grade emerald-only overlay (corner brackets, TrueType species+ID, no confidence)."""
    h, w = vis_bgr.shape[:2]
    base_rgba = Image.fromarray(cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    label_font = _load_ui_font(15 if w >= 960 else 13, bold=True)
    hud_font = _load_ui_font(16 if w >= 960 else 13, bold=True)
    hud_sub_font = _load_ui_font(14 if w >= 960 else 12, bold=False)

    emerald = (52, 232, 130, 245)       # Refined bright emerald green
    emerald_border = (52, 232, 130, 210)
    shadow_dark = (0, 0, 0, 145)
    pill_bg = (12, 24, 18, 195)         # Translucent dark emerald-slate glass
    text_white = (248, 252, 250, 255)
    hud_bar_h = 46

    oryx_now = sum(1 for b in frame_boxes if _classify_species_key(str(b.get("species", ""))) == "arabian_oryx")
    gazelle_now = sum(1 for b in frame_boxes if _classify_species_key(str(b.get("species", ""))) == "arabian_gazelle")

    # Sort detections top-to-bottom so anti-overlap label placement stays deterministic
    sorted_boxes = sorted(frame_boxes, key=lambda b: (b["xyxy"][1], b["xyxy"][0]))
    placed_pills: List[Tuple[int, int, int, int]] = []

    def _overlaps(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
        return not (a[2] + 3 < b[0] or a[0] > b[2] + 3 or a[3] + 2 < b[1] or a[1] > b[3] + 2)

    for box in sorted_boxes:
        x1 = int(round(box["xyxy"][0] * scale))
        y1 = int(round(box["xyxy"][1] * scale))
        x2 = int(round(box["xyxy"][2] * scale))
        y2 = int(round(box["xyxy"][3] * scale))

        _draw_emerald_corner_brackets(draw, x1, y1, x2, y2, emerald, shadow_dark, thickness=3)

        sp_key = _classify_species_key(str(box.get("species", "")))
        sp_short = _SPECIES_SHORT_LABEL.get(sp_key, str(box.get("species", "")).title())
        lbl = f"{sp_short}  ID:{int(box['track_id']) + 1}"

        tb = draw.textbbox((0, 0), lbl, font=label_font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        pad_x, pad_y = 9, 4
        dot_space = 12
        pill_w = tw + pad_x * 2 + dot_space
        pill_h = th + pad_y * 2 + 2

        # Try placing above box, then below box, then nudging upward to prevent overlapping neighbor labels
        cx_left = max(4, min(x1, w - pill_w - 4))
        candidates = [
            (cx_left, max(hud_bar_h + 6, y1 - pill_h - 5)),
            (cx_left, min(h - pill_h - 4, y2 + 5)),
            (cx_left, max(hud_bar_h + 6, y1 - pill_h * 2 - 9)),
            (max(4, min(x2 - pill_w, w - pill_w - 4)), max(hud_bar_h + 6, y1 - pill_h - 5)),
        ]
        chosen = candidates[0]
        for px, py in candidates:
            rect = (px, py, px + pill_w, py + pill_h)
            if not any(_overlaps(rect, prev) for prev in placed_pills):
                chosen = (px, py)
                break

        px, py = chosen
        pill_rect = (px, py, px + pill_w, py + pill_h)
        placed_pills.append(pill_rect)

        draw.rounded_rectangle(pill_rect, radius=6, fill=pill_bg, outline=emerald_border, width=1)
        dot_y = py + pill_h // 2
        draw.ellipse((px + 7, dot_y - 3, px + 13, dot_y + 3), fill=emerald)
        draw.text((px + pad_x + dot_space - 2, py + pad_y - 1), lbl, font=label_font, fill=text_white)

    # Minimal sleek top HUD bar (dark frosted glass strip with live counts and clean timecode)
    draw.rectangle((0, 0, w, hud_bar_h), fill=(10, 18, 15, 188))
    draw.line([(0, hud_bar_h), (w, hud_bar_h)], fill=(52, 232, 130, 140), width=1)

    # Left badge: live species counters
    cx = 16
    draw.ellipse((cx, 17, cx + 11, 28), fill=emerald)
    cx += 18
    title_txt = "Arabian Wildlife AI"
    draw.text((cx, 12), title_txt, font=hud_font, fill=text_white)
    cx += int(draw.textlength(title_txt, font=hud_font)) + 20

    draw.line([(cx, 12), (cx, 34)], fill=(255, 255, 255, 65), width=1)
    cx += 18

    count_txt = f"Arabian Oryx: {oryx_now}    |    Arabian Gazelle: {gazelle_now}"
    draw.text((cx, 13), count_txt, font=hud_font, fill=(175, 250, 205, 255))

    # Right side: clean timecode (+ PAUSED badge only when paused)
    time_str = f"{hms} / {total_hms}" if total_hms else hms
    if paused:
        time_str = f"PAUSED   |   {time_str}"
    tw_r = int(draw.textlength(time_str, font=hud_sub_font))
    draw.text(
        (w - tw_r - 18, 14),
        time_str,
        font=hud_sub_font,
        fill=(255, 220, 110, 255) if paused else (210, 225, 218, 240),
    )

    composited = Image.alpha_composite(base_rgba, layer).convert("RGB")
    return cv2.cvtColor(np.array(composited), cv2.COLOR_RGB2BGR)

# Cross-platform cv2.waitKeyEx codes for Left / Right arrow keys (Windows, Linux X11, macOS)
_LEFT_ARROW_KEYS = {2424832, 81, 65361, 63234}
_RIGHT_ARROW_KEYS = {2555904, 83, 65363, 63235}


def _decode_display_action(
    key_ex: int,
    paused: bool,
    seek_frames: int = 150,
    fine_step_frames: int = 15,
) -> Tuple[str, int]:
    """Translate a cv2.waitKeyEx key code into an interactive playback action."""
    if key_ex is None or key_ex < 0:
        return "none", 0

    if key_ex in _LEFT_ARROW_KEYS:
        return "seek_rel", -max(1, int(seek_frames))
    if key_ex in _RIGHT_ARROW_KEYS:
        return "seek_rel", max(1, int(seek_frames))

    ascii_key = key_ex & 0xFF if key_ex < 256 else -1
    if ascii_key in (ord("q"), ord("Q"), 27):
        return "quit", 0
    if ascii_key == ord(" "):
        return "toggle_pause", 0
    if ascii_key in (ord("+"), ord("=")):
        return "speed_up", 1
    if ascii_key in (ord("-"), ord("_")):
        return "speed_down", -1
    if ascii_key in (ord("a"), ord("A")):
        return "seek_rel", -max(1, int(seek_frames))
    if ascii_key in (ord("d"), ord("D")):
        return "seek_rel", max(1, int(seek_frames))
    if ascii_key in (ord(","), ord("<"), ord("[")):
        return "seek_rel", -1 if paused else -max(1, int(fine_step_frames))
    if ascii_key in (ord("."), ord(">"), ord("]")):
        return "seek_rel", 1 if paused else max(1, int(fine_step_frames))

    return "none", 0


def _fit_frame_for_display(
    frame: np.ndarray,
    max_width: int = 1280,
    max_height: int = 720,
) -> Tuple[np.ndarray, float]:
    """Resize a video frame preserving aspect ratio so the entire frame fits on screen without cropping."""
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return frame.copy(), 1.0
    scale = min(
        (max_width / float(w)) if max_width > 0 else 1.0,
        (max_height / float(h)) if max_height > 0 else 1.0,
        1.0,
    )
    if scale < 0.999:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA), scale
    return frame.copy(), 1.0


def _render_live_overlay(
    frame: np.ndarray,
    frame_boxes: List[Dict[str, Any]],
    frame_idx: int,
    hms: str,
    run_mode: str,
    trusted_thresh: float = 0.75,
    max_width: int = 1280,
    max_height: int = 720,
    playback_speed: float = 1.0,
    paused: bool = False,
    total_frames: int = 0,
    total_hms: str = "",
    clean_demo_mode: bool = False,
) -> np.ndarray:
    """Render an aspect-ratio-fitted live frame with bounding boxes, HUD, and color legend."""
    vis, scale = _fit_frame_for_display(frame, max_width=max_width, max_height=max_height)
    if clean_demo_mode:
        return _render_clean_demo_overlay(
            vis_bgr=vis,
            frame_boxes=frame_boxes,
            scale=scale,
            hms=hms,
            total_hms=total_hms,
            paused=paused,
        )

    h, w = vis.shape[:2]

    top_bar_h = 36
    legend_bar_h = 30
    hud_total_h = top_bar_h + legend_bar_h

    # Count detected species in the current frame
    oryx_now = sum(1 for b in frame_boxes if _classify_species_key(str(b.get("species", ""))) == "arabian_oryx")
    gazelle_now = sum(1 for b in frame_boxes if _classify_species_key(str(b.get("species", ""))) == "arabian_gazelle")

    # Draw bounding boxes and readable high-contrast label pills first so HUD stays cleanly on top
    green_bgr = (0, 220, 0)
    orange_bgr = (0, 165, 255)
    for box in frame_boxes:
        x1 = int(round(box["xyxy"][0] * scale))
        y1 = int(round(box["xyxy"][1] * scale))
        x2 = int(round(box["xyxy"][2] * scale))
        y2 = int(round(box["xyxy"][3] * scale))
        cf = float(box["conf"])
        is_trusted = cf >= trusted_thresh
        color = green_bgr if is_trusted else orange_bgr
        status_tag = "Trusted" if is_trusted else "Review"
        sp_short = _SPECIES_SHORT_LABEL.get(_classify_species_key(str(box.get("species", ""))), str(box.get("species", "")))

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        lbl = f"ID:{int(box['track_id']) + 1} {sp_short} {int(round(cf * 100))}% [{status_tag}]"
        font_scale = 0.52
        (tw, th), baseline = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        pill_w = tw + 8
        pill_h = th + baseline + 6
        px1 = max(0, min(x1, w - pill_w - 2))
        py2 = max(hud_total_h + pill_h + 2, y1)
        py1 = py2 - pill_h
        cv2.rectangle(vis, (px1, py1), (px1 + pill_w, py2), color, -1)
        cv2.putText(
            vis,
            lbl,
            (px1 + 4, py2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (15, 15, 15),
            1,
            cv2.LINE_AA,
        )

    # Semi-transparent top status bar + color legend bar
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, top_bar_h), (20, 24, 28), -1)
    cv2.rectangle(overlay, (0, top_bar_h), (w, hud_total_h), (12, 16, 20), -1)
    cv2.addWeighted(overlay, 0.80, vis, 0.20, 0.0, dst=vis)
    cv2.line(vis, (0, top_bar_h), (w, top_bar_h), (60, 68, 76), 1)
    cv2.line(vis, (0, hud_total_h), (w, hud_total_h), (80, 90, 100), 1)

    speed_str = "MAX" if playback_speed <= 0 else f"{playback_speed:g}x"
    time_part = f"{hms}/{total_hms}" if total_hms else hms
    frame_part = f"{frame_idx}/{total_frames}" if total_frames > 0 else str(frame_idx)
    status_txt = (
        f"Mode: {run_mode} | {time_part} | F:{frame_part} | "
        f"{speed_str} | Oryx:{oryx_now} Gazelle:{gazelle_now}"
    )
    if paused:
        status_txt += " [PAUSED]"

    hud_font_scale = 0.48 if w >= 960 else 0.40
    cv2.putText(
        vis,
        status_txt,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        hud_font_scale,
        (255, 255, 255) if not paused else (0, 235, 255),
        1,
        cv2.LINE_AA,
    )

    controls_txt = "[Space] Pause [<-/->,A/D] Seek [,/.] Step [+/-] Speed [Q] Quit"
    (sw, _), _ = cv2.getTextSize(status_txt, cv2.FONT_HERSHEY_SIMPLEX, hud_font_scale, 1)
    (cw, _), _ = cv2.getTextSize(controls_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    if w - cw - 16 > sw + 24:
        cv2.putText(
            vis,
            controls_txt,
            (w - cw - 12, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (195, 205, 215),
            1,
            cv2.LINE_AA,
        )

    # Row 2: Explicit Color Legend so any viewer understands Green vs Orange immediately
    pct = int(round(trusted_thresh * 100))
    ly_mid = top_bar_h + legend_bar_h // 2
    compact_legend = w < 920
    leg_font = 0.42 if compact_legend else 0.48
    cursor_x = 12
    guide_lbl = "Colors:" if compact_legend else "Color Guide:"
    cv2.putText(
        vis,
        guide_lbl,
        (cursor_x, ly_mid + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        leg_font,
        (210, 218, 226),
        1,
        cv2.LINE_AA,
    )
    (glw, _), _ = cv2.getTextSize(guide_lbl, cv2.FONT_HERSHEY_SIMPLEX, leg_font, 1)
    cursor_x += glw + 12

    # Green swatch + explanation
    cv2.rectangle(vis, (cursor_x, ly_mid - 7), (cursor_x + 14, ly_mid + 7), green_bgr, -1)
    cv2.rectangle(vis, (cursor_x, ly_mid - 7), (cursor_x + 14, ly_mid + 7), (255, 255, 255), 1)
    cursor_x += 20
    green_txt = f"Green: >={pct}% Trusted" if compact_legend else f"Green = High Confidence (>={pct}% Trusted)"
    cv2.putText(
        vis,
        green_txt,
        (cursor_x, ly_mid + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        leg_font,
        (140, 255, 140),
        1,
        cv2.LINE_AA,
    )
    (gw, _), _ = cv2.getTextSize(green_txt, cv2.FONT_HERSHEY_SIMPLEX, leg_font, 1)
    cursor_x += gw + (16 if compact_legend else 26)

    # Orange swatch + explanation
    cv2.rectangle(vis, (cursor_x, ly_mid - 7), (cursor_x + 14, ly_mid + 7), orange_bgr, -1)
    cv2.rectangle(vis, (cursor_x, ly_mid - 7), (cursor_x + 14, ly_mid + 7), (255, 255, 255), 1)
    cursor_x += 20
    orange_txt = f"Orange: <{pct}% Review" if compact_legend else f"Orange = Low Confidence (<{pct}% Needs Review)"
    cv2.putText(
        vis,
        orange_txt,
        (cursor_x, ly_mid + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        leg_font,
        (130, 215, 255),
        1,
        cv2.LINE_AA,
    )
    return vis


class VideoScanner:
    """Scans wildlife videos with display mode support and exports dual confidence buckets.

    Execution modes:
    - 'display_only': shows live annotated video without creating bucket folders or running downstream pipeline.
    - 'display_and_train': shows live annotated video AND writes both confidence buckets (<0.95 & >=0.75).
    - 'train_only': headless scan that writes both confidence buckets (<0.95 & >=0.75).
    """

    def __init__(self, config: VideoScanConfig) -> None:
        self.config = config
        self._model: Any = None

    def _load_model(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO

            logger.info("Loading YOLO weights from %s", self.config.yolo_weights)
            self._model = YOLO(self.config.yolo_weights)
        return self._model

    def unload(self) -> None:
        """Release YOLO model weights and reclaim GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @staticmethod
    def _count_frame_species(
        frame_boxes: List[Dict[str, Any]],
        class_names: Dict[int, str],
    ) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for box in frame_boxes:
            cls_id = int(box["class_id"])
            sp_name = norm_species(class_names.get(cls_id, f"class_{cls_id}"))
            counts[sp_name] = counts.get(sp_name, 0) + 1
        return counts

    def scan_videos(
        self,
        video_paths: Sequence[str],
        staging_dir: Path,
        qwen_review_bucket_dir: Optional[Path] = None,
        trusted_rag_bucket_dir: Optional[Path] = None,
        unload_after: bool = True,
    ) -> Tuple[List[TrackManifestItem], List[Dict[str, Any]]]:
        """Scan a batch of videos, optionally display live, and export review (<0.95) & trusted (>=0.75) buckets."""
        import supervision as sv
        import torch
        from trackers import ByteTrackTracker

        model = self._load_model()
        use_half = self.config.use_half and torch.cuda.is_available()

        mode_flags = resolve_mode_flags(self.config.run_mode)
        in_colab_or_headless = (
            "COLAB_RELEASE_TAG" in os.environ
            or "COLAB_GPU" in os.environ
            or (sys.platform.startswith("linux") and not os.environ.get("DISPLAY"))
        )
        should_display = mode_flags["display_video"] and not in_colab_or_headless
        if mode_flags["display_video"] and in_colab_or_headless:
            logger.warning(
                "Headless/Colab environment detected: disabling cv2.imshow live window cleanly."
            )
        should_export_buckets = mode_flags["export_buckets"]
        is_display_only = self.config.run_mode == "display_only"

        # Default bucket directories next to staging_dir if not explicitly passed
        base_out = staging_dir.parent
        review_dir = qwen_review_bucket_dir or (base_out / "qwen_review_below_95")
        trusted_dir = trusted_rag_bucket_dir or (base_out / "trusted_rag_above_75")

        all_manifest: List[TrackManifestItem] = []
        all_stats: List[Dict[str, Any]] = []
        user_quit = False

        try:
            for idx, video_path in enumerate(video_paths, start=1):
                if user_quit:
                    break

                video_tag = f"v{idx:02d}_{sanitize_tag(video_path)}"
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    logger.error(
                        "[%d/%d] Could not open video: %s (skipping)",
                        idx,
                        len(video_paths),
                        video_path,
                    )
                    all_stats.append(
                        {
                            "video_tag": video_tag,
                            "video": Path(video_path).name,
                            "status": "failed_to_open",
                        }
                    )
                    continue

                raw_dir = staging_dir / video_tag
                if should_export_buckets:
                    raw_dir.mkdir(parents=True, exist_ok=True)

                fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                total_hms = format_hms(total_frames / fps) if total_frames > 0 else ""

                def _make_tracker() -> Any:
                    return ByteTrackTracker(
                        track_activation_threshold=max(
                            self.config.conf_min, self.config.track_activation_floor
                        ),
                        lost_track_buffer=int(fps * self.config.lost_track_buffer_sec),
                        minimum_iou_threshold=self.config.minimum_iou_threshold,
                        frame_rate=int(round(fps)),
                    )

                tracker = _make_tracker()
                bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                    history=500, varThreshold=25, detectShadows=False
                )

                tracks: Dict[int, TrackRecord] = {}
                frame_idx = 0
                cooldown_counter = 0
                t0 = time.time()

                # Live playback pacing & interactive speed/pause/seek state
                speed_presets = [0.5, 1.0, 2.0, 4.0, 0.0]  # 0.0 = MAX unthrottled speed
                init_speed = float(self.config.display_playback_speed)
                if init_speed in speed_presets:
                    speed_idx = speed_presets.index(init_speed)
                else:
                    speed_idx = 1  # default 1.0x real-time
                window_initialized = False
                trackbar_initialized = False
                updating_trackbar = False
                is_paused = False
                pending_seek_frame: Optional[int] = None

                def _on_trackbar(pos: int) -> None:
                    nonlocal pending_seek_frame
                    if updating_trackbar or not is_display_only:
                        return
                    cur_zero_idx = max(0, frame_idx - 1)
                    if abs(int(pos) - cur_zero_idx) > 1:
                        pending_seek_frame = int(pos)

                def _show_frame_and_pace(
                    raw_frame: np.ndarray,
                    boxes_now: List[Dict[str, Any]],
                    f_idx: int,
                    stride_used: int,
                    iter_start_ts: float,
                ) -> bool:
                    """Render fitted overlay, sync trackbar, pace FPS, and handle Pause/Seek/Speed/Quit keys."""
                    nonlocal speed_idx, window_initialized, trackbar_initialized, updating_trackbar
                    nonlocal should_display, is_paused, pending_seek_frame
                    win_name = self.config.display_window_name
                    cur_speed = speed_presets[speed_idx]
                    vis = _render_live_overlay(
                        raw_frame,
                        boxes_now,
                        f_idx,
                        format_hms(f_idx / fps),
                        self.config.run_mode,
                        trusted_thresh=self.config.trusted_rag_min_conf,
                        max_width=self.config.display_max_width,
                        max_height=self.config.display_max_height,
                        playback_speed=cur_speed,
                        paused=is_paused,
                        total_frames=total_frames,
                        total_hms=total_hms,
                        clean_demo_mode=self.config.display_clean_demo_mode,
                    )
                    try:
                        if not window_initialized:
                            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                            cv2.resizeWindow(win_name, vis.shape[1], vis.shape[0])
                            window_initialized = True
                            if is_display_only and total_frames > 1:
                                cv2.createTrackbar("Frame", win_name, 0, total_frames - 1, _on_trackbar)
                                trackbar_initialized = True

                        if trackbar_initialized and total_frames > 1:
                            updating_trackbar = True
                            try:
                                cv2.setTrackbarPos(
                                    "Frame",
                                    win_name,
                                    max(0, min(total_frames - 1, f_idx - 1)),
                                )
                            finally:
                                updating_trackbar = False

                        cv2.imshow(win_name, vis)

                        def _apply_key(key_ex_val: int) -> Tuple[bool, bool]:
                            """Apply decoded key action; returns (should_quit, seek_requested)."""
                            nonlocal speed_idx, is_paused, pending_seek_frame
                            action, val = _decode_display_action(
                                key_ex_val,
                                paused=is_paused,
                                seek_frames=self.config.display_seek_frames,
                                fine_step_frames=self.config.display_fine_step_frames,
                            )
                            if action == "quit":
                                return True, False
                            if action == "speed_up":
                                speed_idx = min(len(speed_presets) - 1, speed_idx + 1)
                            elif action == "speed_down":
                                speed_idx = max(0, speed_idx - 1)
                            elif action == "toggle_pause":
                                is_paused = not is_paused
                            elif action == "seek_rel" and is_display_only:
                                cur_zero_idx = max(0, f_idx - 1)
                                pending_seek_frame = cur_zero_idx + val
                                return False, True
                            return False, False

                        # Pace to real-time FPS in display_only (or scaled by cur_speed); keep waitKey(1) in MAX/train
                        if is_paused:
                            wait_ms = 30
                        elif is_display_only and cur_speed > 0:
                            elapsed_ms = (time.perf_counter() - iter_start_ts) * 1000.0
                            target_ms = (1000.0 / max(1.0, fps)) * stride_used / cur_speed
                            wait_ms = max(1, int(round(target_ms - elapsed_ms)))
                        else:
                            wait_ms = 1

                        should_quit_now, _ = _apply_key(cv2.waitKeyEx(wait_ms))
                        if should_quit_now or cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                            return True

                        # While paused, stay responsive to Space, Arrow/A/D/,/. stepping, trackbar drag, and Quit
                        while is_paused:
                            if pending_seek_frame is not None:
                                break
                            paused_vis = _render_live_overlay(
                                raw_frame,
                                boxes_now,
                                f_idx,
                                format_hms(f_idx / fps),
                                self.config.run_mode,
                                trusted_thresh=self.config.trusted_rag_min_conf,
                                max_width=self.config.display_max_width,
                                max_height=self.config.display_max_height,
                                playback_speed=speed_presets[speed_idx],
                                paused=True,
                                total_frames=total_frames,
                                total_hms=total_hms,
                                clean_demo_mode=self.config.display_clean_demo_mode,
                            )
                            cv2.imshow(win_name, paused_vis)
                            quit_p, seek_p = _apply_key(cv2.waitKeyEx(30))
                            if quit_p or cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                                return True
                            if seek_p or pending_seek_frame is not None:
                                break
                    except cv2.error as exc:
                        logger.warning("GUI display unavailable (%s); continuing headless.", exc)
                        should_display = False
                    return False

                pbar = tqdm(
                    total=total_frames if total_frames > 0 else None,
                    desc=f"Stage 1 [{self.config.run_mode}] [{idx}/{len(video_paths)}: {video_tag[:18]}]",
                    unit="frames",
                )
                try:
                    while cap.isOpened():
                        iter_t0 = time.perf_counter()
                        is_active = cooldown_counter > 0

                        # Apply interactive seek (keyboard arrows/A/D/,/. or trackbar drag) cleanly in display_only
                        did_seek = False
                        if is_display_only and pending_seek_frame is not None:
                            max_valid = max(0, total_frames - 2) if total_frames > 1 else max(0, pending_seek_frame)
                            target_pos = max(0, min(max_valid, pending_seek_frame)) if total_frames > 0 else max(0, pending_seek_frame)
                            cap.set(cv2.CAP_PROP_POS_FRAMES, target_pos)
                            frame_idx = target_pos
                            tracker = _make_tracker()
                            pending_seek_frame = None
                            did_seek = True

                        # In display_only mode, default to smooth 1x playback (stride=1) unless user speeds up via +/-
                        if did_seek:
                            current_stride = 1
                        elif is_display_only:
                            cur_speed = speed_presets[speed_idx]
                            if cur_speed <= 0:
                                current_stride = max(1, self.config.active_stride)
                            elif cur_speed > 1.0:
                                current_stride = max(1, int(round(cur_speed))) * max(1, self.config.display_stride)
                            else:
                                current_stride = max(1, self.config.display_stride)
                        else:
                            current_stride = (
                                self.config.active_stride if is_active else self.config.idle_stride
                            )
                        skipped = 0

                        if current_stride > 1:
                            for _ in range(current_stride - 1):
                                if not cap.grab():
                                    break
                                frame_idx += 1
                                skipped += 1

                        ret, frame = cap.read()
                        if not ret:
                            pbar.update(skipped)
                            break
                        frame_idx += 1
                        if did_seek and pbar.total:
                            pbar.n = min(pbar.total, frame_idx)
                            pbar.refresh()
                        else:
                            pbar.update(skipped + 1)

                        # Skip YOLO inference on static background frames when idle (except when displaying live)
                        if not is_active and not is_display_only:
                            small = cv2.resize(frame, (320, 180))
                            if cv2.countNonZero(bg_subtractor.apply(small)) <= self.config.motion_thresh:
                                if should_display:
                                    if _show_frame_and_pace(frame, [], frame_idx, current_stride, iter_t0):
                                        user_quit = True
                                        break
                                continue

                        results = model(
                            frame,
                            verbose=False,
                            conf=self.config.conf_min,
                            quantize=16 if use_half else None,
                        )[0]
                        detections = sv.Detections.from_ultralytics(results)
                        detections = tracker.update(detections)

                        frame_boxes_snapshot: List[Dict[str, Any]] = []
                        if len(detections) > 0:
                            cooldown_counter = self.config.cooldown_frames
                            ts = frame_idx / fps

                            frame_boxes_snapshot = [
                                {
                                    "track_id": int(t_id),
                                    "class_id": int(c_id),
                                    "species": norm_species(model.names.get(int(c_id), f"class_{int(c_id)}")),
                                    "conf": float(cf),
                                    "xyxy": [int(v) for v in box],
                                }
                                for box, cf, c_id, t_id in zip(
                                    detections.xyxy,
                                    detections.confidence,
                                    detections.class_id,
                                    detections.tracker_id,
                                )
                                if t_id is not None and t_id >= 0
                            ]

                            for det in frame_boxes_snapshot:
                                trk_id = det["track_id"]
                                cls_id = det["class_id"]
                                conf_val = det["conf"]
                                bbox = (
                                    det["xyxy"][0],
                                    det["xyxy"][1],
                                    det["xyxy"][2],
                                    det["xyxy"][3],
                                )

                                if trk_id not in tracks:
                                    first_img_str = ""
                                    if should_export_buckets:
                                        first_img_out = raw_dir / f"trk_{trk_id:05d}_f1_start.jpg"
                                        cv2.imwrite(str(first_img_out), frame)
                                        first_img_str = str(first_img_out)
                                    tracks[trk_id] = TrackRecord(
                                        track_id=trk_id,
                                        start_time_sec=ts,
                                        end_time_sec=ts,
                                        start_frame=frame_idx,
                                        end_frame=frame_idx,
                                        best_frame=frame_idx,
                                        best_time_sec=ts,
                                        first_frame_path=first_img_str,
                                    )

                                ev = tracks[trk_id]
                                ev.end_time_sec = ts
                                ev.end_frame = frame_idx
                                ev.confidences.append(conf_val)
                                ev.class_votes.append(cls_id)

                                if conf_val > ev.max_conf:
                                    ev.max_conf = conf_val
                                    ev.best_frame = frame_idx
                                    ev.best_time_sec = ts
                                    ev.best_bbox = bbox
                                    ev.all_frame_boxes = frame_boxes_snapshot
                                    if should_export_buckets:
                                        img_out = raw_dir / f"trk_{trk_id:05d}_f2_best.jpg"
                                        cv2.imwrite(str(img_out), frame)
                                        ev.raw_image_path = str(img_out)

                                # Keep latest temporal frame updated (throttled to ~1 write/sec to avoid disk I/O bottleneck)
                                min_last_gap = max(1, int(round(fps)))
                                if (
                                    should_export_buckets
                                    and frame_idx > ev.start_frame
                                    and (ev.last_saved_frame_idx < 0 or (frame_idx - ev.last_saved_frame_idx) >= min_last_gap)
                                ):
                                    last_img_out = raw_dir / f"trk_{trk_id:05d}_f3_end.jpg"
                                    cv2.imwrite(str(last_img_out), frame)
                                    ev.last_frame_path = str(last_img_out)
                                    ev.last_saved_frame_idx = frame_idx
                        else:
                            cooldown_counter = max(0, cooldown_counter - 1)

                        if should_display:
                            if _show_frame_and_pace(
                                frame, frame_boxes_snapshot, frame_idx, current_stride, iter_t0
                            ):
                                user_quit = True
                                break
                finally:
                    cap.release()
                    pbar.close()

                duration_sec = (total_frames / fps) if total_frames > 0 else (frame_idx / fps)
                scan_time_sec = time.time() - t0
                vid_flagged_review = 0
                vid_trusted_rag = 0
                filtered_flickers = 0

                sorted_tracks = sorted(tracks.items(), key=lambda kv: kv[1].start_time_sec)
                logger.info(
                    "[%s] Scan finished in %.1fs (%d raw tracks). Exporting buckets...",
                    video_tag,
                    scan_time_sec,
                    len(sorted_tracks),
                )

                track_iter = (
                    tqdm(sorted_tracks, desc=f"Exporting [{video_tag[:18]}]", unit="track")
                    if should_export_buckets and sorted_tracks
                    else sorted_tracks
                )

                for trk_id, ev in track_iter:
                    if should_export_buckets and not ev.raw_image_path:
                        continue
                    if not ev.class_votes:
                        continue
                    # Drop 1-frame low-confidence flicker noise unless audit_all is explicitly enabled
                    if (
                        not self.config.audit_all
                        and len(ev.confidences) < self.config.min_track_hits
                        and ev.max_conf < self.config.min_single_hit_conf
                    ):
                        filtered_flickers += 1
                        continue

                    votes = Counter(ev.class_votes)
                    dom_cls, dom_cnt = votes.most_common(1)[0]
                    purity = dom_cnt / len(ev.class_votes)
                    avg_conf = float(np.mean(ev.confidences))
                    min_conf = float(np.min(ev.confidences))

                    # Bucket 1 (< 0.95 or low purity or audit_all): goes to Qwen GRPO review -> YOLO training
                    needs_review = (
                        self.config.audit_all
                        or (avg_conf < self.config.conf_review_thresh)
                        or (purity < self.config.purity_review_thresh)
                    )
                    # Bucket 2 (>= 0.75): trusted sightings -> Qwen scene description -> ChromaDB
                    is_trusted_rag = avg_conf >= self.config.trusted_rag_min_conf

                    stem = f"{video_tag}__trk_{trk_id:05d}"
                    best_hms = format_hms(ev.best_time_sec)
                    temporal_paths, temporal_indices = (
                        collect_chronological_track_frames(ev) if should_export_buckets else ([], [])
                    )

                    # Decode the staged frame ONCE even when exporting to both buckets
                    preloaded_bgr = (
                        cv2.imread(str(ev.raw_image_path))
                        if (should_export_buckets and (needs_review or is_trusted_rag))
                        else None
                    )

                    rev_img = rev_lbl = rev_dbg = None
                    if needs_review:
                        vid_flagged_review += 1
                        if should_export_buckets:
                            rev_img, rev_lbl, rev_dbg, _ = export_track_to_bucket(
                                bucket_dir=review_dir,
                                stem=stem,
                                raw_image_path=Path(ev.raw_image_path),
                                all_frame_boxes=ev.all_frame_boxes,
                                target_track_id=trk_id,
                                avg_conf=avg_conf,
                                best_frame=ev.best_frame,
                                best_hms=best_hms,
                                bucket_label=f"QWEN_REVIEW (<{self.config.conf_review_thresh:.2f})",
                                save_debug=self.config.save_bucket_debug,
                                preloaded_bgr=preloaded_bgr,
                            )

                    tru_img = tru_lbl = tru_dbg = None
                    tru_temporal_paths: List[str] = []
                    if is_trusted_rag:
                        vid_trusted_rag += 1
                        if should_export_buckets:
                            tru_img, tru_lbl, tru_dbg, tru_temporal_paths = export_track_to_bucket(
                                bucket_dir=trusted_dir,
                                stem=stem,
                                raw_image_path=Path(ev.raw_image_path),
                                all_frame_boxes=ev.all_frame_boxes,
                                target_track_id=trk_id,
                                avg_conf=avg_conf,
                                best_frame=ev.best_frame,
                                best_hms=best_hms,
                                bucket_label=f"TRUSTED_RAG (>={self.config.trusted_rag_min_conf:.2f})",
                                save_debug=self.config.save_bucket_debug,
                                temporal_frame_paths=temporal_paths,
                                preloaded_bgr=preloaded_bgr,
                            )

                    frame_counts = self._count_frame_species(ev.all_frame_boxes, model.names)
                    item = TrackManifestItem(
                        video_tag=video_tag,
                        video_source=Path(video_path).name,
                        track_id=trk_id,
                        species_yolo=norm_species(model.names.get(dom_cls, f"class_{dom_cls}")),
                        start_frame=ev.start_frame,
                        end_frame=ev.end_frame,
                        best_frame=ev.best_frame,
                        start_time_sec=round(ev.start_time_sec, 3),
                        end_time_sec=round(ev.end_time_sec, 3),
                        best_time_sec=round(ev.best_time_sec, 3),
                        start_hms=format_hms(ev.start_time_sec),
                        end_hms=format_hms(ev.end_time_sec),
                        best_hms=best_hms,
                        fps=round(fps, 3),
                        avg_conf=round(avg_conf, 3),
                        max_conf=round(ev.max_conf, 3),
                        min_conf=round(min_conf, 3),
                        class_purity=round(purity, 3),
                        yolo_bbox_xyxy=list(ev.best_bbox),
                        all_frame_boxes=ev.all_frame_boxes,
                        raw_image_path=ev.raw_image_path,
                        needs_rl_review=needs_review,
                        is_trusted_for_rag=is_trusted_rag,
                        frame_species_counts=frame_counts,
                        review_bucket_image_path=rev_img,
                        review_bucket_label_path=rev_lbl,
                        review_bucket_debug_path=rev_dbg,
                        trusted_bucket_image_path=tru_img,
                        trusted_bucket_label_path=tru_lbl,
                        trusted_bucket_debug_path=tru_dbg,
                        temporal_frame_paths=temporal_paths,
                        temporal_frame_indices=temporal_indices,
                        trusted_temporal_frame_paths=tru_temporal_paths,
                    )
                    all_manifest.append(item)

                all_stats.append(
                    {
                        "video_tag": video_tag,
                        "video": Path(video_path).name,
                        "status": "displayed_only" if is_display_only else "ok",
                        "run_mode": self.config.run_mode,
                        "fps": round(fps, 2),
                        "total_frames": total_frames if total_frames > 0 else frame_idx,
                        "duration_sec": round(duration_sec, 2),
                        "duration_min": round(duration_sec / 60.0, 2),
                        "scan_time_sec": round(scan_time_sec, 2),
                        "total_tracks": len(tracks),
                        "flagged_for_qwen_review": vid_flagged_review,
                        "trusted_for_rag": vid_trusted_rag,
                    }
                )
        finally:
            if should_display:
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass
            if unload_after:
                self.unload()

        return all_manifest, all_stats
