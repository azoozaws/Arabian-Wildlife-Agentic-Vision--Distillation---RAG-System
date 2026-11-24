"""Utility helpers for logging, geometry, string normalization, summaries, and debug visualization."""

from app.src.utils.geometry import (
    build_zoomed_visual_prompt,
    format_hms,
    norm_species,
    prepare_full_frame_pil,
    sanitize_tag,
    xyxy_to_yolo,
)
from app.src.utils.logging_utils import get_logger
from app.src.utils.summary_builder import (
    build_video_summary,
    classify_speedup_level,
    compute_processing_performance,
    render_video_summary_markdown,
)
from app.src.utils.visualization import (
    save_model_disagreements,
    save_side_by_side_qwen_debug,
    save_track_bucket_debug,
)

__all__ = [
    "get_logger",
    "format_hms",
    "norm_species",
    "sanitize_tag",
    "xyxy_to_yolo",
    "build_zoomed_visual_prompt",
    "prepare_full_frame_pil",
    "build_video_summary",
    "compute_processing_performance",
    "classify_speedup_level",
    "render_video_summary_markdown",
    "save_track_bucket_debug",
    "save_side_by_side_qwen_debug",
    "save_model_disagreements",
]
