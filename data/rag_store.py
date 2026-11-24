"""RAG document builder, JSONL persistence, and metadata query filter engine."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence

if sys.platform == "win32":
    try:
        import ctypes

        _sys_msvcp = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "MSVCP140.dll"
        if _sys_msvcp.exists():
            ctypes.WinDLL(str(_sys_msvcp))
    except Exception:
        pass

from app.src.config import DEFAULT_EMBEDDING_MODEL_ID
from app.src.schemas import (
    RAGDocument,
    RAGMetadata,
    RAGQueryFilter,
    SceneDescriptionResult,
    TrackManifestItem,
)
from app.src.utils.geometry import norm_species
from app.src.utils.logging_utils import get_logger

logger = get_logger("RAGStore")

TIME_OF_DAY_AR: Dict[str, str] = {
    "dawn": "الفجر",
    "morning": "الصباح",
    "midday": "الظهيرة",
    "afternoon": "العصر",
    "dusk": "الغروب",
    "night": "الليل",
    "unknown": "غير محدد",
}


def _format_counts_ar(oryx_count: int, gazelle_count: int, other_count: int) -> str:
    parts: List[str] = []
    if oryx_count > 0:
        parts.append(f"{oryx_count} من المها العربي")
    if gazelle_count > 0:
        parts.append(f"{gazelle_count} من الغزال العربي")
    if other_count > 0:
        parts.append(f"{other_count} من حيوانات أخرى")
    return " و ".join(parts) if parts else "لا توجد حيوانات مستهدفة مؤكدة"


def _format_counts_en(oryx_count: int, gazelle_count: int, other_count: int) -> str:
    parts: List[str] = []
    if oryx_count > 0:
        parts.append(f"{oryx_count} Arabian Oryx")
    if gazelle_count > 0:
        parts.append(f"{gazelle_count} Arabian Gazelle")
    if other_count > 0:
        parts.append(f"{other_count} other animal(s)")
    return " and ".join(parts) if parts else "no confirmed target wildlife"


def build_rag_document(
    track: TrackManifestItem,
    scene: SceneDescriptionResult,
    language: Literal["ar", "en", "bilingual"] = "ar",
    conf_review_thresh: float = 0.75,
    purity_review_thresh: float = 0.80,
    flag_count_mismatch: bool = True,
) -> RAGDocument:
    """Combine Stage 1/2 track metadata and Stage 3 Qwen scene description into a RAGDocument."""
    # Prefer Stage 2 Qwen-verified counts when available, otherwise Stage 1 detector counts
    base_counts = track.verified_species_counts or track.frame_species_counts or {}
    det_oryx = int(base_counts.get("arabian_oryx", 0))
    det_gazelle = int(base_counts.get("arabian_gazelle", 0))

    obs_oryx = int(scene.observed_counts.get("arabian_oryx", det_oryx))
    obs_gazelle = int(scene.observed_counts.get("arabian_gazelle", det_gazelle))
    obs_other = int(scene.observed_counts.get("other", 0))

    # Use verified detector counts as canonical counts when Stage 2 ran, else fall back to VLM counts
    oryx_count = det_oryx if (track.verified_species_counts or det_oryx > 0) else obs_oryx
    gazelle_count = det_gazelle if (track.verified_species_counts or det_gazelle > 0) else obs_gazelle
    other_count = obs_other
    total_animals = oryx_count + gazelle_count + other_count

    species_counts: Dict[str, int] = {
        "arabian_oryx": oryx_count,
        "arabian_gazelle": gazelle_count,
    }
    if other_count > 0:
        species_counts["other"] = other_count

    species_present = [sp for sp, cnt in species_counts.items() if cnt > 0]
    primary_species = (
        track.qwen_species
        if (track.qwen_verified and track.qwen_species and track.qwen_species != "none")
        else track.species_yolo
    )

    # Compute transparent human-in-the-loop review flags
    review_reasons: List[str] = []
    if track.avg_conf < conf_review_thresh:
        review_reasons.append("low_confidence")
    if track.class_purity < purity_review_thresh:
        review_reasons.append("low_class_purity")
    if track.qwen_verified and track.qwen_species == "none":
        review_reasons.append("qwen_rejected_false_positive")
    if not scene.parsed_ok:
        review_reasons.append("vlm_description_parse_fallback")
    if scene.visibility_quality in {"blurry", "low_light", "distant"}:
        review_reasons.append(f"visibility_{scene.visibility_quality}")
    if flag_count_mismatch and scene.observed_counts:
        if obs_oryx != det_oryx or obs_gazelle != det_gazelle:
            review_reasons.append("vlm_detector_count_mismatch")

    needs_review = bool(review_reasons) or (track.needs_rl_review and not track.qwen_verified)

    # Build grounded Arabic & English embeddable texts including exact clip/frame/timestamp info
    counts_ar = _format_counts_ar(oryx_count, gazelle_count, other_count)
    counts_en = _format_counts_en(oryx_count, gazelle_count, other_count)
    tod_ar = TIME_OF_DAY_AR.get(scene.estimated_time_of_day, scene.estimated_time_of_day)
    review_ar = "تحتاج إلى مراجعة" if needs_review else "مؤكدة ولا تحتاج مراجعة"
    review_en = "needs review" if needs_review else "verified"

    header_ar = (
        f"في المقطع ({track.video_source}) عند الفريم رقم {track.best_frame} "
        f"(وقت الظهور من {track.start_hms} إلى {track.end_hms}، وأفضل لقطة عند {track.best_hms}): "
        f"يوجد {counts_ar}. "
        f"الوقت المتوقع من اليوم: {tod_ar}. "
        f"متوسط دقة الكشف: {track.avg_conf:.2f} (أعلى دقة {track.max_conf:.2f})، والحالة: {review_ar}."
    )
    text_ar = f"{header_ar}\n{scene.description_ar}".strip() if scene.description_ar else header_ar

    header_en = (
        f"In video '{track.video_source}' at frame {track.best_frame} "
        f"(visible from {track.start_hms} to {track.end_hms}, peak clarity at {track.best_hms}): "
        f"detected {counts_en}. "
        f"Estimated time of day: {scene.estimated_time_of_day}. "
        f"Average confidence: {track.avg_conf:.2f} (max {track.max_conf:.2f}), status: {review_en}."
    )
    text_en = f"{header_en}\n{scene.description_en}".strip() if scene.description_en else header_en

    if language == "ar":
        primary_text = text_ar
    elif language == "en":
        primary_text = text_en
    else:
        primary_text = f"{text_ar}\n\n---\n\n{text_en}"

    doc_id = f"{track.video_tag}__trk_{track.track_id:05d}__f{track.best_frame}"
    image_path = track.trusted_bucket_image_path or track.distilled_image_path or track.raw_image_path
    temporal_paths = (
        track.trusted_temporal_frame_paths
        or track.temporal_frame_paths
        or ([image_path] if image_path else [])
    )

    metadata = RAGMetadata(
        doc_id=doc_id,
        video_tag=track.video_tag,
        video_source=track.video_source,
        image_path=image_path,
        track_id=track.track_id,
        start_frame=track.start_frame,
        end_frame=track.end_frame,
        best_frame=track.best_frame,
        start_time_sec=track.start_time_sec,
        end_time_sec=track.end_time_sec,
        best_time_sec=track.best_time_sec,
        start_hms=track.start_hms,
        end_hms=track.end_hms,
        best_hms=track.best_hms,
        oryx_count=oryx_count,
        gazelle_count=gazelle_count,
        other_animals_count=other_count,
        total_animals=total_animals,
        species_counts=species_counts,
        species_present=species_present,
        primary_species=primary_species,
        avg_conf=track.avg_conf,
        max_conf=track.max_conf,
        min_conf=track.min_conf,
        class_purity=track.class_purity,
        needs_review=needs_review,
        review_reasons=review_reasons,
        qwen_verified=track.qwen_verified,
        qwen_verdict=track.qwen_species,
        estimated_time_of_day=scene.estimated_time_of_day,
        environment_type=scene.environment_type,
        animal_states=scene.animal_states,
        background_elements=scene.background_elements,
        visibility_quality=scene.visibility_quality,
        temporal_frames_used=len(temporal_paths),
        temporal_frame_paths=list(temporal_paths),
    )

    return RAGDocument(
        doc_id=doc_id,
        text=primary_text,
        text_ar=text_ar,
        text_en=text_en,
        metadata=metadata,
    )


def save_rag_documents_jsonl(docs: Iterable[RAGDocument], out_path: Path) -> Path:
    """Persist RAGDocument objects as newline-delimited JSON (JSONL)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    logger.info("Saved %d RAG documents to %s", count, out_path)
    return out_path


def load_rag_documents_jsonl(jsonl_path: Path) -> List[RAGDocument]:
    """Load RAGDocument objects from a JSONL file."""
    docs: List[RAGDocument] = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            docs.append(RAGDocument.from_dict(json.loads(line)))
    return docs


def export_vector_db_payloads(docs: Iterable[RAGDocument]) -> List[Dict[str, Any]]:
    """Convert RAGDocument records into flat payloads ready for Qdrant, ChromaDB, or LangChain."""
    return [doc.to_vector_db_payload() for doc in docs]


def filter_rag_documents(
    docs: Iterable[RAGDocument],
    query_filter: Optional[RAGQueryFilter] = None,
) -> List[RAGDocument]:
    """Filter RAG documents using rich wildlife metadata criteria (counts, confidence, review status, etc.)."""
    if query_filter is None:
        return list(docs)

    f = query_filter
    norm_species_filter = {norm_species(s) for s in f.species} if f.species else None
    norm_tod_filter = {norm_species(t) for t in f.time_of_day} if f.time_of_day else None
    norm_states_filter = {norm_species(s) for s in f.animal_states} if f.animal_states else None
    norm_vis_filter = {norm_species(v) for v in f.visibility_quality} if f.visibility_quality else None
    norm_env_filter = norm_species(f.environment_type) if f.environment_type else None
    needle = f.text_contains.lower() if f.text_contains else None

    matched: List[RAGDocument] = []
    for doc in docs:
        m = doc.metadata
        if f.video_source and f.video_source.lower() not in m.video_source.lower():
            continue
        if f.video_tag and f.video_tag != m.video_tag:
            continue
        if norm_species_filter and not (norm_species_filter & set(m.species_present)):
            continue
        if f.min_oryx_count is not None and m.oryx_count < f.min_oryx_count:
            continue
        if f.max_oryx_count is not None and m.oryx_count > f.max_oryx_count:
            continue
        if f.min_gazelle_count is not None and m.gazelle_count < f.min_gazelle_count:
            continue
        if f.max_gazelle_count is not None and m.gazelle_count > f.max_gazelle_count:
            continue
        if f.min_total_animals is not None and m.total_animals < f.min_total_animals:
            continue
        if f.max_total_animals is not None and m.total_animals > f.max_total_animals:
            continue
        if f.min_conf is not None and m.avg_conf < f.min_conf:
            continue
        if f.max_conf is not None and m.avg_conf > f.max_conf:
            continue
        if f.min_purity is not None and m.class_purity < f.min_purity:
            continue
        if f.needs_review is not None and m.needs_review != f.needs_review:
            continue
        if f.qwen_verified is not None and m.qwen_verified != f.qwen_verified:
            continue
        if norm_tod_filter and m.estimated_time_of_day not in norm_tod_filter:
            continue
        if norm_env_filter and norm_env_filter != m.environment_type:
            continue
        if norm_states_filter and not (norm_states_filter & set(m.animal_states)):
            continue
        if norm_vis_filter and m.visibility_quality not in norm_vis_filter:
            continue
        if f.frame_start is not None and m.best_frame < f.frame_start:
            continue
        if f.frame_end is not None and m.best_frame > f.frame_end:
            continue
        if f.time_start_sec is not None and m.best_time_sec < f.time_start_sec:
            continue
        if f.time_end_sec is not None and m.best_time_sec > f.time_end_sec:
            continue
        if needle:
            haystack = f"{doc.text}\n{doc.text_ar}\n{doc.text_en}".lower()
            if needle not in haystack:
                continue

        matched.append(doc)

    return matched


def summarize_rag_corpus(docs: Iterable[RAGDocument]) -> Dict[str, Any]:
    """Compute quick aggregate analytics over a collection of RAG documents for an Agent."""
    doc_list = list(docs)
    total_oryx = sum(d.metadata.oryx_count for d in doc_list)
    total_gazelle = sum(d.metadata.gazelle_count for d in doc_list)
    needs_review_count = sum(1 for d in doc_list if d.metadata.needs_review)
    tod_dist = Counter(d.metadata.estimated_time_of_day for d in doc_list)
    env_dist = Counter(d.metadata.environment_type for d in doc_list)
    reason_dist = Counter(r for d in doc_list for r in d.metadata.review_reasons)

    return {
        "total_documents": len(doc_list),
        "total_oryx_sightings": total_oryx,
        "total_gazelle_sightings": total_gazelle,
        "needs_review_count": needs_review_count,
        "time_of_day_distribution": dict(tod_dist),
        "environment_distribution": dict(env_dist),
        "review_reason_distribution": dict(reason_dist),
    }


DEFAULT_QUERY_INSTRUCT = (
    "Given a wildlife monitoring search query in Arabic or English, "
    "retrieve relevant wildlife camera-trap sighting descriptions."
)


def last_token_pool(last_hidden_states: Any, attention_mask: Any) -> Any:
    """Official Qwen3-Embedding last-token pooling supporting both left and right padding."""
    import torch

    left_padding = bool(attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
    ]


class QwenEmbeddingFunction:
    """Multilingual Arabic/English embedder backed by Qwen/Qwen3-Embedding-0.6B.

    Computes official last-token pooled, L2-normalized vectors for cosine similarity in ChromaDB.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_EMBEDDING_MODEL_ID,
        max_length: int = 1024,
        batch_size: int = 16,
        device: Optional[str] = None,
        query_instruction: str = DEFAULT_QUERY_INSTRUCT,
    ) -> None:
        self.model_id = model_id
        self.max_length = max_length
        self.batch_size = max(1, int(batch_size))
        self.device = device
        self.query_instruction = query_instruction
        self._tokenizer: Any = None
        self._model: Any = None
        self._resolved_device: str = "cpu"
        self._load_lock = threading.Lock()

    def _load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        with self._load_lock:
            if self._model is not None and self._tokenizer is not None:
                return
            import torch
            from transformers import AutoModel, AutoTokenizer

            dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._resolved_device = dev
            if dev.startswith("cuda"):
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                # Qwen3-Embedding weights are natively bfloat16; using bf16 on modern CPUs halves RAM (~1.6GB vs ~2.9GB)
                dtype = torch.bfloat16 if torch.backends.mkldnn.is_available() else torch.float32

            logger.info("Loading embedding model %s on %s (dtype=%s)...", self.model_id, dev, dtype)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding_side="left")
            try:
                self._model = AutoModel.from_pretrained(
                    self.model_id,
                    dtype=dtype,
                    low_cpu_mem_usage=True,
                ).to(dev)
            except OSError as exc:
                if "torchaudio" in str(exc).lower():
                    raise RuntimeError(
                        "Detected a broken/mismatched `torchaudio` binary in your Python environment. "
                        "This project does not require torchaudio — fix it immediately by running:\n"
                        "  pip uninstall -y torchaudio"
                    ) from exc
                raise
            self._model.eval()

    def unload(self) -> None:
        """Release embedding model weights from memory."""
        self._model = None
        self._tokenizer = None
        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _encode_texts(self, texts: Sequence[str]) -> List[List[float]]:
        import torch
        import torch.nn.functional as F

        if not texts:
            return []
        self._load()

        all_vecs: List[List[float]] = []
        text_list = [str(t) for t in texts]
        total_batches = (len(text_list) + self.batch_size - 1) // self.batch_size
        with torch.inference_mode():
            for batch_idx, i in enumerate(range(0, len(text_list), self.batch_size), start=1):
                if total_batches > 1:
                    logger.info(
                        "Encoding embedding batch %d/%d (%d docs total) on %s...",
                        batch_idx,
                        total_batches,
                        len(text_list),
                        self._resolved_device,
                    )
                chunk = text_list[i : i + self.batch_size]
                batch_dict = self._tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self._resolved_device)
                outputs = self._model(**batch_dict)
                pooled = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
                normed = F.normalize(pooled.float(), p=2, dim=1)
                all_vecs.extend(normed.cpu().tolist())
        return all_vecs

    def encode_documents(self, documents: Sequence[str]) -> List[List[float]]:
        """Encode raw wildlife sighting documents for ChromaDB indexing."""
        return self._encode_texts(documents)

    def encode_query(self, query_text: str) -> List[List[float]]:
        """Encode a search query using Qwen3-Embedding's official instruction prompt."""
        formatted = f"Instruct: {self.query_instruction}\nQuery: {query_text}"
        return self._encode_texts([formatted])

    def __call__(self, input: Sequence[str]) -> List[List[float]]:
        return self.encode_documents(input)

    def name(self) -> str:
        return f"qwen_embedding::{self.model_id}"


@lru_cache(maxsize=2)
def get_qwen_embedding_function(
    model_id: str = DEFAULT_EMBEDDING_MODEL_ID,
    batch_size: int = 16,
) -> QwenEmbeddingFunction:
    """Return a cached QwenEmbeddingFunction instance so weights are loaded only once."""
    return QwenEmbeddingFunction(model_id=model_id, batch_size=batch_size)


def upsert_rag_documents_to_chromadb(
    docs: Iterable[RAGDocument],
    persist_dir: Path,
    collection_name: str = "arabian_wildlife_rag",
    embedding_function: Any = None,
    embedding_model_id: Optional[str] = DEFAULT_EMBEDDING_MODEL_ID,
    reset_collection: bool = False,
) -> Dict[str, Any]:
    """Upsert RAGDocument records into ChromaDB using Qwen3-Embedding-0.6B (cosine space)."""
    doc_list = list(docs)
    if not doc_list:
        return {
            "upserted": 0,
            "persist_dir": str(persist_dir),
            "collection_name": collection_name,
            "embedding_model": embedding_model_id or "chromadb_default",
        }

    import chromadb

    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))

    if reset_collection:
        try:
            client.delete_collection(name=collection_name)
            logger.info("Deleted existing ChromaDB collection '%s' prior to rebuild.", collection_name)
        except Exception:
            pass

    ef = embedding_function
    if ef is None and embedding_model_id:
        ef = get_qwen_embedding_function(model_id=embedding_model_id)

    col_meta: Dict[str, Any] = {"hnsw:space": "cosine"}
    if embedding_model_id:
        col_meta["embedding_model"] = embedding_model_id

    collection = client.get_or_create_collection(name=collection_name, metadata=col_meta)

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    for doc in doc_list:
        doc_id, text, meta = doc.to_chromadb_record()
        ids.append(doc_id)
        documents.append(text)
        metadatas.append(meta)

    embedding_dim: Optional[int] = None
    if ef is not None:
        vecs = ef.encode_documents(documents) if hasattr(ef, "encode_documents") else ef(documents)
        embedding_dim = len(vecs[0]) if vecs else None
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=vecs)
    else:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    logger.info(
        "Upserted %d RAG documents into ChromaDB collection '%s' (model=%s, dim=%s) at %s",
        len(ids),
        collection_name,
        embedding_model_id or "chromadb_default",
        embedding_dim,
        persist_dir,
    )
    return {
        "upserted": len(ids),
        "persist_dir": str(persist_dir),
        "collection_name": collection_name,
        "embedding_model": embedding_model_id or "chromadb_default",
        "embedding_dim": embedding_dim,
        "reset_collection": reset_collection,
    }


def rebuild_chromadb_from_jsonl(
    rag_jsonl_paths: Path | str | Sequence[Path | str],
    persist_dir: Path | str,
    collection_name: str = "arabian_wildlife_rag",
    embedding_model_id: str = DEFAULT_EMBEDDING_MODEL_ID,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """Delete and rebuild a ChromaDB collection from one or more rag_documents.jsonl files."""
    if isinstance(rag_jsonl_paths, (str, Path)):
        path_list = [Path(rag_jsonl_paths)]
    else:
        path_list = [Path(p) for p in rag_jsonl_paths]

    all_docs: List[RAGDocument] = []
    for p in path_list:
        if not p.exists():
            raise FileNotFoundError(f"RAG JSONL file not found: {p}")
        all_docs.extend(load_rag_documents_jsonl(p))

    ef = get_qwen_embedding_function(model_id=embedding_model_id, batch_size=batch_size)
    return upsert_rag_documents_to_chromadb(
        docs=all_docs,
        persist_dir=Path(persist_dir),
        collection_name=collection_name,
        embedding_function=ef,
        embedding_model_id=embedding_model_id,
        reset_collection=True,
    )


def query_chromadb_rag(
    persist_dir: Path,
    query_text: str,
    collection_name: str = "arabian_wildlife_rag",
    where_filter: Optional[Dict[str, Any]] = None,
    n_results: int = 10,
    embedding_function: Any = None,
    embedding_model_id: Optional[str] = DEFAULT_EMBEDDING_MODEL_ID,
) -> Dict[str, Any]:
    """Run semantic + metadata-filtered retrieval against ChromaDB using matching Qwen3-Embedding vectors."""
    import chromadb

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(name=collection_name)

    # Auto-detect embedding model stored in collection metadata if available
    stored_model = (collection.metadata or {}).get("embedding_model")
    active_model_id = stored_model or embedding_model_id

    ef = embedding_function
    if ef is None and active_model_id:
        ef = get_qwen_embedding_function(model_id=active_model_id)

    query_kwargs: Dict[str, Any] = {"n_results": n_results}
    if ef is not None:
        q_vecs = ef.encode_query(query_text) if hasattr(ef, "encode_query") else ef([query_text])
        query_kwargs["query_embeddings"] = q_vecs
    else:
        query_kwargs["query_texts"] = [query_text]

    if where_filter:
        query_kwargs["where"] = where_filter
    return collection.query(**query_kwargs)


def _cli_rebuild() -> None:
    parser = argparse.ArgumentParser(description="Rebuild ChromaDB wildlife collection with Qwen3-Embedding")
    parser.add_argument("--jsonl", nargs="+", required=True, help="Path(s) to rag_documents.jsonl")
    parser.add_argument("--chroma_dir", type=str, required=True, help="ChromaDB persist directory")
    parser.add_argument("--collection", type=str, default="arabian_wildlife_rag")
    parser.add_argument("--model", type=str, default=DEFAULT_EMBEDDING_MODEL_ID)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    res = rebuild_chromadb_from_jsonl(
        rag_jsonl_paths=args.jsonl,
        persist_dir=args.chroma_dir,
        collection_name=args.collection,
        embedding_model_id=args.model,
        batch_size=args.batch_size,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli_rebuild()
