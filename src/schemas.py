"""Typed data schemas for video tracking, VLM inference, RAG documents, and agent tools."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DetectionBox:
    """Single tracked object bounding box within a video frame."""

    track_id: int
    class_id: int
    species: str
    conf: float
    xyxy: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrackRecord:
    """Mutable accumulator for an object track during Stage 1 video scanning."""

    track_id: int
    start_time_sec: float
    end_time_sec: float
    start_frame: int
    end_frame: int
    best_frame: int = 0
    best_time_sec: float = 0.0
    max_conf: float = 0.0
    confidences: List[float] = field(default_factory=list)
    class_votes: List[int] = field(default_factory=list)
    best_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    all_frame_boxes: List[Dict[str, Any]] = field(default_factory=list)
    raw_image_path: str = ""
    # Chronological temporal frames captured directly to disk (start, best, end)
    first_frame_path: str = ""
    last_frame_path: str = ""
    last_saved_frame_idx: int = -1


@dataclass
class TrackManifestItem:
    """Serializable summary of a finalized track from Stage 1 (and optionally enriched in Stage 2)."""

    video_tag: str
    video_source: str
    track_id: int
    species_yolo: str
    start_frame: int
    end_frame: int
    best_frame: int
    start_time_sec: float
    end_time_sec: float
    best_time_sec: float
    start_hms: str
    end_hms: str
    best_hms: str
    fps: float
    avg_conf: float
    max_conf: float
    min_conf: float
    class_purity: float
    yolo_bbox_xyxy: List[int]
    all_frame_boxes: List[Dict[str, Any]]
    raw_image_path: str
    needs_rl_review: bool
    is_trusted_for_rag: bool = False
    frame_species_counts: Dict[str, int] = field(default_factory=dict)
    # Stage 1 bucket artifact paths (qwen_review_below_95 & trusted_rag_above_75)
    review_bucket_image_path: Optional[str] = None
    review_bucket_label_path: Optional[str] = None
    review_bucket_debug_path: Optional[str] = None
    trusted_bucket_image_path: Optional[str] = None
    trusted_bucket_label_path: Optional[str] = None
    trusted_bucket_debug_path: Optional[str] = None
    # Up to 3 chronological frames (start, best, end) for multi-frame Qwen RAG scene description
    temporal_frame_paths: List[str] = field(default_factory=list)
    temporal_frame_indices: List[int] = field(default_factory=list)
    trusted_temporal_frame_paths: List[str] = field(default_factory=list)
    # Fields populated after Stage 2 Qwen verification
    qwen_verified: bool = False
    qwen_species: Optional[str] = None
    verified_species_counts: Dict[str, int] = field(default_factory=dict)
    distilled_image_path: Optional[str] = None
    distilled_label_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrackManifestItem":
        return cls(**data)


@dataclass
class VLMClassificationResult:
    """Result returned by Qwen3-VL closed-set species verification."""

    species: str
    raw_text: str
    parsed_ok: bool


@dataclass
class SceneDescriptionResult:
    """Structured wildlife scene analysis returned by Qwen3-VL for RAG indexing."""

    description_ar: str
    description_en: str
    estimated_time_of_day: str  # dawn | morning | midday | afternoon | dusk | night | unknown
    environment_type: str  # e.g., sand_dunes, gravel_plain, desert_shrubland, rocky_hills
    animal_states: List[str] = field(default_factory=list)
    background_elements: List[str] = field(default_factory=list)
    visibility_quality: str = "clear"  # clear | partial_occlusion | distant | blurry | low_light
    observed_counts: Dict[str, int] = field(default_factory=dict)
    raw_text: str = ""
    parsed_ok: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RAGMetadata:
    """Flat, vector-database-friendly metadata payload for filtering RAG queries."""

    doc_id: str
    video_tag: str
    video_source: str
    image_path: str
    track_id: int
    # Temporal & frame coordinates
    start_frame: int
    end_frame: int
    best_frame: int
    start_time_sec: float
    end_time_sec: float
    best_time_sec: float
    start_hms: str
    end_hms: str
    best_hms: str
    # Per-species counts (flat ints enable range queries in ChromaDB/Qdrant/SQL)
    oryx_count: int
    gazelle_count: int
    other_animals_count: int
    total_animals: int
    species_counts: Dict[str, int]
    species_present: List[str]
    primary_species: str
    # Detection quality & human-in-the-loop review flags
    avg_conf: float
    max_conf: float
    min_conf: float
    class_purity: float
    needs_review: bool
    review_reasons: List[str]
    qwen_verified: bool
    qwen_verdict: Optional[str]
    # Environmental & behavioral filters
    estimated_time_of_day: str
    environment_type: str
    animal_states: List[str]
    background_elements: List[str]
    visibility_quality: str
    temporal_frames_used: int = 1
    temporal_frame_paths: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RAGDocument:
    """Complete RAG record combining embeddable narrative text and structured filter metadata."""

    doc_id: str
    text: str
    text_ar: str
    text_en: str
    metadata: RAGMetadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "text_ar": self.text_ar,
            "text_en": self.text_en,
            "metadata": self.metadata.to_dict(),
        }

    def to_vector_db_payload(self) -> Dict[str, Any]:
        """Return a flat dict ready for Qdrant or LangChain Document ingestion."""
        payload = self.metadata.to_dict()
        payload.pop("species_counts", None)
        payload["text"] = self.text
        return payload

    def to_chromadb_record(self) -> Tuple[str, str, Dict[str, Any]]:
        """Return (id, document_text, chroma_metadata) with strictly primitive values (str|int|float|bool)."""
        m = self.metadata
        chroma_meta: Dict[str, Any] = {
            "video_tag": str(m.video_tag),
            "video_source": str(m.video_source),
            "image_path": str(m.image_path),
            "track_id": int(m.track_id),
            "start_frame": int(m.start_frame),
            "end_frame": int(m.end_frame),
            "best_frame": int(m.best_frame),
            "start_time_sec": float(m.start_time_sec),
            "end_time_sec": float(m.end_time_sec),
            "best_time_sec": float(m.best_time_sec),
            "start_hms": str(m.start_hms),
            "end_hms": str(m.end_hms),
            "best_hms": str(m.best_hms),
            "oryx_count": int(m.oryx_count),
            "gazelle_count": int(m.gazelle_count),
            "other_animals_count": int(m.other_animals_count),
            "total_animals": int(m.total_animals),
            "has_oryx": bool(m.oryx_count > 0),
            "has_gazelle": bool(m.gazelle_count > 0),
            "species_present_csv": ",".join(m.species_present),
            "primary_species": str(m.primary_species),
            "avg_conf": float(m.avg_conf),
            "max_conf": float(m.max_conf),
            "min_conf": float(m.min_conf),
            "class_purity": float(m.class_purity),
            "needs_review": bool(m.needs_review),
            "review_reasons_csv": ",".join(m.review_reasons),
            "qwen_verified": bool(m.qwen_verified),
            "qwen_verdict": str(m.qwen_verdict) if m.qwen_verdict is not None else "unverified",
            "estimated_time_of_day": str(m.estimated_time_of_day),
            "environment_type": str(m.environment_type),
            "animal_states_csv": ",".join(m.animal_states),
            "background_elements_csv": ",".join(m.background_elements),
            "visibility_quality": str(m.visibility_quality),
            "temporal_frames_used": int(m.temporal_frames_used),
            "temporal_frame_paths_csv": ",".join(m.temporal_frame_paths),
        }
        return self.doc_id, self.text, chroma_meta

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RAGDocument":
        return cls(
            doc_id=data["doc_id"],
            text=data["text"],
            text_ar=data.get("text_ar", ""),
            text_en=data.get("text_en", ""),
            metadata=RAGMetadata(**data["metadata"]),
        )


@dataclass
class RAGQueryFilter:
    """Structured query filters for searching wildlife RAG documents programmatically or via an Agent."""

    video_source: Optional[str] = None
    video_tag: Optional[str] = None
    species: Optional[List[str]] = None
    min_oryx_count: Optional[int] = None
    max_oryx_count: Optional[int] = None
    min_gazelle_count: Optional[int] = None
    max_gazelle_count: Optional[int] = None
    min_total_animals: Optional[int] = None
    max_total_animals: Optional[int] = None
    min_conf: Optional[float] = None
    max_conf: Optional[float] = None
    min_purity: Optional[float] = None
    needs_review: Optional[bool] = None
    qwen_verified: Optional[bool] = None
    time_of_day: Optional[List[str]] = None
    environment_type: Optional[str] = None
    animal_states: Optional[List[str]] = None
    visibility_quality: Optional[List[str]] = None
    frame_start: Optional[int] = None
    frame_end: Optional[int] = None
    time_start_sec: Optional[float] = None
    time_end_sec: Optional[float] = None
    text_contains: Optional[str] = None
