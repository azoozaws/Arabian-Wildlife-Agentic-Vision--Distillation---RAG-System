"""Centralized configuration dataclasses for the Arabian Wildlife AI system."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Optional, Set

# Valid image file extensions for dataset and frame processing
VALID_IMAGE_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".webp"}

# Canonical species class mapping (must align with YOLO data.yaml)
CLASS_NAME_TO_ID: Dict[str, int] = {
    "arabian_gazelle": 0,
    "arabian_oryx": 1,
}

CLASS_ID_TO_NAME: Dict[int, str] = {
    0: "arabian_gazelle",
    1: "arabian_oryx",
}

SPECIES_DISPLAY_AR: Dict[str, str] = {
    "arabian_gazelle": "الغزال العربي",
    "arabian_oryx": "المها العربي",
}

DEFAULT_QWEN_MODEL_ID: str = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_EMBEDDING_MODEL_ID: str = "Qwen/Qwen3-Embedding-0.6B"

RunModeType = Literal[
    "display_only",
    "summary_only",
    "summary_with_rag",
    "summary_with_qwen",
    "train_only",
    "display_and_train",
]

RUN_MODE_FLAGS: Dict[str, Dict[str, bool]] = {
    "display_only": {
        "display_video": True,
        "export_buckets": False,
        "run_qwen_verify": False,
        "run_qwen_rag": False,
        "run_yolo_training": False,
        "generate_summary": True,
    },
    "summary_only": {
        "display_video": False,
        "export_buckets": True,
        "run_qwen_verify": False,
        "run_qwen_rag": False,
        "run_yolo_training": False,
        "generate_summary": True,
    },
    "summary_with_rag": {
        "display_video": False,
        "export_buckets": True,
        "run_qwen_verify": False,
        "run_qwen_rag": True,
        "run_yolo_training": False,
        "generate_summary": True,
    },
    "summary_with_qwen": {
        "display_video": False,
        "export_buckets": True,
        "run_qwen_verify": True,
        "run_qwen_rag": True,
        "run_yolo_training": False,
        "generate_summary": True,
    },
    "train_only": {
        "display_video": False,
        "export_buckets": True,
        "run_qwen_verify": True,
        "run_qwen_rag": True,
        "run_yolo_training": True,
        "generate_summary": True,
    },
    "display_and_train": {
        "display_video": True,
        "export_buckets": True,
        "run_qwen_verify": True,
        "run_qwen_rag": True,
        "run_yolo_training": True,
        "generate_summary": True,
    },
}


def resolve_mode_flags(run_mode: str) -> Dict[str, bool]:
    """Return execution feature flags for a given run_mode."""
    if run_mode not in RUN_MODE_FLAGS:
        valid = ", ".join(RUN_MODE_FLAGS.keys())
        raise ValueError(f"Unsupported run_mode '{run_mode}'. Valid options: {valid}")
    return dict(RUN_MODE_FLAGS[run_mode])


@dataclass
class VideoScanConfig:
    """Configuration for Stage 1 video scanning with YOLO, ByteTrack, display mode, and dual buckets."""

    yolo_weights: str
    run_mode: RunModeType = "train_only"
    conf_min: float = 0.15
    # Tracks below 0.95 (or low purity) go to qwen_review_below_95/{images,labels,debug} for Qwen GRPO review
    conf_review_thresh: float = 0.95
    # Tracks at or above 0.75 go to trusted_rag_above_75/{images,labels,debug} for Qwen RAG description -> ChromaDB
    trusted_rag_min_conf: float = 0.75
    purity_review_thresh: float = 0.80
    idle_stride: int = 30
    active_stride: int = 5
    cooldown_frames: int = 60
    motion_thresh: int = 500
    audit_all: bool = False
    save_bucket_debug: bool = True
    # Drop 1-frame low-confidence flicker tracks (<0.50 conf) to prevent thousands of noise exports
    min_track_hits: int = 2
    min_single_hit_conf: float = 0.50
    display_window_name: str = "Arabian Wildlife Tracker"
    # Live display playback defaults: smooth 1x real-time fitted cleanly to screen without cropping
    display_stride: int = 1
    display_playback_speed: float = 1.0
    display_max_width: int = 1280
    display_max_height: int = 720
    # Interactive seeking in display_only mode (Left/Right or A/D = ±seek_frames; ,/. = ±fine_step or ±1 when paused)
    display_seek_frames: int = 150
    display_fine_step_frames: int = 15
    # Optional clean demo recording overlay (False = standard full HUD & green/orange legend; True = clean emerald demo)
    display_clean_demo_mode: bool = False
    use_half: bool = True
    track_activation_floor: float = 0.20
    minimum_iou_threshold: float = 0.20
    lost_track_buffer_sec: float = 2.0


@dataclass
class QwenVLMConfig:
    """Configuration for loading and running Qwen3-VL inference."""

    model_id: str = DEFAULT_QWEN_MODEL_ID
    lora_dir: Optional[str] = None
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 512 * 28 * 28
    batch_size: int = 8
    zoom_scale: float = 3.0
    classification_max_new_tokens: int = 96
    description_max_new_tokens: int = 512
    # Disable classification GRPO LoRA adapter during open-ended RAG scene description
    use_adapter_for_description: bool = False
    device_map: str = "auto"


@dataclass
class RAGDescriptionConfig:
    """Configuration for generating 3-frame temporal RAG wildlife scene descriptions and ChromaDB indexing."""

    enabled: bool = True
    # 'ar' generates Arabic narrative, 'en' English, 'bilingual' embeds both
    language: Literal["ar", "en", "bilingual"] = "ar"
    # By default, describe tracks in the trusted (>= 0.75) bucket; set False to describe all non-rejected tracks
    trusted_only: bool = True
    # If True, describe tracks rejected as false positives too
    include_empty_or_rejected: bool = False
    # Flag document for human review if Qwen observed counts disagree with YOLO counts
    flag_count_mismatch: bool = True
    # Number of chronological frames per track fed to Qwen3-VL (start, best/mid, end)
    temporal_frames_per_track: int = 3
    max_image_dim: int = 512
    output_filename: str = "rag_documents.jsonl"
    # Optional persistent ChromaDB destination
    chromadb_dir: Optional[Path] = None
    chromadb_collection: str = "arabian_wildlife_rag"
    embedding_model_id: str = DEFAULT_EMBEDDING_MODEL_ID


@dataclass
class PipelineConfig:
    """Top-level configuration for the end-to-end video distillation and RAG pipeline."""

    out_dataset_dir: Path
    video_scan: VideoScanConfig
    qwen: QwenVLMConfig
    rag: RAGDescriptionConfig = field(default_factory=RAGDescriptionConfig)
    run_stage2_distill: bool = True
    save_debug: bool = True
    # Resume automatically from stage1_checkpoint.json if Stage 1 already finished in out_dataset_dir
    resume_from_stage1: bool = True

    @property
    def staging_dir(self) -> Path:
        # Avoid slow Google Drive FUSE writes for temporary per-frame staging on Colab
        out_str = str(self.out_dataset_dir).replace("\\", "/")
        if out_str.startswith("/content/drive"):
            return Path("/content/_wildlife_staging") / self.out_dataset_dir.name
        return self.out_dataset_dir / "_staging"

    @property
    def stage1_checkpoint_path(self) -> Path:
        return self.out_dataset_dir / "stage1_checkpoint.json"

    @property
    def qwen_review_bucket_dir(self) -> Path:
        return self.out_dataset_dir / "qwen_review_below_95"

    @property
    def trusted_rag_bucket_dir(self) -> Path:
        return self.out_dataset_dir / "trusted_rag_above_75"


@dataclass
class WandbConfig:
    """Weights & Biases tracking configuration loaded safely from environment variables."""

    enabled: bool = True
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("WANDB_API_KEY"))
    base_url: Optional[str] = field(default_factory=lambda: os.getenv("WANDB_BASE_URL"))
    entity: Optional[str] = field(default_factory=lambda: os.getenv("WANDB_ENTITY"))
    project: str = field(default_factory=lambda: os.getenv("WANDB_PROJECT", "Arabian_Gazelles"))
    disable_code: bool = True
    console: str = "off"


@dataclass
class TrainingConfig:
    """Configuration for incremental YOLO dataset merging, fine-tuning, and evaluation."""

    orig_dataset: Path
    qwen_dataset: Path
    merged_dataset: Path
    baseline_weights: Path
    project_dir: Path
    version_tag: str = "v3"
    run_name: str = "yolo_v3_rl_distilled"
    replay_samples: int = 400
    seed: int = 42
    epochs: int = 25
    imgsz: int = 640
    batch: int = 16
    optimizer: str = "AdamW"
    lr0: float = 0.0002
    lrf: float = 0.05
    cos_lr: bool = True
    warmup_epochs: float = 0.0
    warmup_bias_lr: float = 0.0002
    patience: int = 8
    mosaic: float = 0.2
    close_mosaic: int = 10
    hsv_h: float = 0.005
    hsv_s: float = 0.25
    hsv_v: float = 0.20
    fliplr: float = 0.5
    scale: float = 0.3
    diff_conf_thresh: float = 0.25
    fp_conf_thresh: float = 0.25
    fp_frame_stride: int = 30
    fp_max_eval_minutes: float = 15.0
    eval_split: str = "test"
    archive_dataset_zip: bool = True
    models_dir: Optional[Path] = None
    registry_path: Optional[Path] = None
    model_id: Optional[str] = None
    parent_model_id: Optional[str] = None
    wandb: WandbConfig = field(default_factory=WandbConfig)
