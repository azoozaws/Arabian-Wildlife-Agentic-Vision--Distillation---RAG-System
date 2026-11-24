"""Hybrid Arabic/English natural-language RAG query engine for wildlife sightings.

Supports queries such as:
- "غزال في النهار" (gazelle in daytime)
- "ستة غزلان في الليل" (six gazelles at night)
- "غزال تحت الشجرة" (gazelle under the tree)
- "مها عربي يرعى في الصباح" (Arabian oryx grazing in the morning)

Combines automatic Arabic/English entity & filter extraction with ChromaDB semantic search
and JSONL metadata + bilingual keyword scoring fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.data.rag_store import (
    filter_rag_documents,
    load_rag_documents_jsonl,
    query_chromadb_rag,
)
from app.src.config import DEFAULT_EMBEDDING_MODEL_ID, SPECIES_DISPLAY_AR
from app.src.schemas import RAGDocument, RAGQueryFilter
from app.src.utils.logging_utils import get_logger

logger = get_logger("RAGQueryEngine")

# Arabic & English number word mapping
ARABIC_EASTERN_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

NUMBER_WORDS: Dict[str, int] = {
    "واحد": 1,
    "واحدة": 1,
    "واحده": 1,
    "اثنين": 2,
    "اثنان": 2,
    "اثنتين": 2,
    "اثنتان": 2,
    "ثلاث": 3,
    "ثلاثة": 3,
    "ثلاثه": 3,
    "اربع": 4,
    "اربعة": 4,
    "اربعه": 4,
    "خمس": 5,
    "خمسة": 5,
    "خمسه": 5,
    "ست": 6,
    "ستة": 6,
    "سته": 6,
    "سبع": 7,
    "سبعة": 7,
    "سبعه": 7,
    "ثمان": 8,
    "ثماني": 8,
    "ثمانية": 8,
    "ثمانيه": 8,
    "تسع": 9,
    "تسعة": 9,
    "تسعه": 9,
    "عشر": 10,
    "عشرة": 10,
    "عشره": 10,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

# Bilingual concept synonym groups for scene/background/state matching
CONCEPT_SYNONYMS: List[Tuple[List[str], List[str], Optional[str], Optional[str]]] = [
    # (trigger_keywords, match_terms, state_tag, env_or_bg_hint)
    (
        ["شجر", "شجره", "اشجار", "تحت الشجر", "تحت الشجره", "سمر", "طلح", "سدر", "tree", "trees", "acacia"],
        ["شجر", "شجره", "اشجار", "سمر", "طلح", "سدر", "tree", "trees", "acacia", "shrub", "desert_shrubland"],
        None,
        "tree",
    ),
    (
        ["كثبان", "رمل", "رمال", "رملي", "رمليه", "dune", "dunes", "sand"],
        ["كثبان", "رمل", "رمال", "رملي", "sand", "sand_dunes", "dune"],
        None,
        "sand_dunes",
    ),
    (
        ["صخر", "صخور", "صخري", "تلال", "جبل", "جبال", "rock", "rocky", "hill", "hills"],
        ["صخر", "صخور", "صخري", "تلال", "جبل", "rocky", "rocky_hills", "hills"],
        None,
        "rocky_hills",
    ),
    (
        ["ماء", "مياه", "واحه", "بركه", "بئر", "water", "waterhole", "oasis", "drinking", "يشرب"],
        ["ماء", "مياه", "واحه", "يشرب", "شرب", "water", "waterhole", "oasis", "drinking"],
        "drinking",
        "waterhole",
    ),
    (
        ["يرعي", "ترعي", "رعي", "ياكل", "تاكل", "graze", "grazing", "eating"],
        ["يرعي", "ترعي", "رعي", "ياكل", "grazing", "graze"],
        "grazing",
        None,
    ),
    (
        ["يركض", "تركض", "يجري", "تجري", "ركض", "جري", "run", "running"],
        ["يركض", "تركض", "يجري", "ركض", "running", "run"],
        "running",
        None,
    ),
    (
        ["يمشي", "تمشي", "مشي", "يسير", "تسير", "يتحرك", "walk", "walking"],
        ["يمشي", "تمشي", "مشي", "يسير", "يتحرك", "walking", "walk"],
        "walking",
        None,
    ),
    (
        ["مستريح", "مستريحه", "نائم", "نائمه", "جالس", "جالسه", "راحه", "راقد", "rest", "resting", "sleeping"],
        ["مستريح", "نائم", "جالس", "راحه", "راقد", "resting", "rest", "lying"],
        "resting",
        None,
    ),
    (
        ["متيقظ", "حذر", "حذره", "واقف", "alert", "standing", "watching"],
        ["متيقظ", "حذر", "واقف", "alert", "standing"],
        "alert",
        None,
    ),
    (
        ["سياج", "شبك", "سور", "fence"],
        ["سياج", "شبك", "سور", "fence"],
        None,
        "fence",
    ),
    (
        ["سياره", "مركبه", "vehicle", "car", "truck"],
        ["سياره", "مركبه", "vehicle", "car"],
        None,
        "vehicle",
    ),
]

TIME_OF_DAY_PATTERNS: List[Tuple[List[str], List[str]]] = [
    (["النهار", "نهار", "نهارا", "في النهار", "day", "daytime", "daylight"], ["morning", "midday", "afternoon"]),
    (["الصباح", "صباح", "صباحا", "في الصباح", "morning"], ["morning"]),
    (["الظهر", "الظهيره", "ظهر", "ظهرا", "midday", "noon"], ["midday"]),
    (["العصر", "عصر", "عصرا", "بعد الظهر", "afternoon"], ["afternoon"]),
    (["الفجر", "فجر", "فجرا", "شروق", "dawn", "sunrise"], ["dawn"]),
    (["الغروب", "غروب", "المساء", "مساء", "dusk", "sunset", "evening"], ["dusk"]),
    (["الليل", "ليل", "ليلا", "في الليل", "ظلام", "night", "nighttime", "dark"], ["night"]),
]

STOPWORDS = {
    "في",
    "من",
    "على",
    "الى",
    "إلى",
    "عن",
    "مع",
    "او",
    "أو",
    "و",
    "هل",
    "كم",
    "اين",
    "أين",
    "تحت",
    "فوق",
    "قرب",
    "عند",
    "وقت",
    "صوره",
    "صور",
    "مقطع",
    "فيديو",
    "يوجد",
    "هنالك",
    "هناك",
    "ارني",
    "ابحث",
    "the",
    "in",
    "at",
    "on",
    "of",
    "a",
    "an",
    "under",
    "near",
    "with",
    "during",
    "show",
    "find",
}


def normalize_arabic_text(text: str) -> str:
    """Normalize Arabic orthographic variants and digits for reliable bilingual search."""
    s = str(text or "").translate(ARABIC_EASTERN_DIGITS).lower()
    # Remove Arabic diacritics (tashkeel) and tatweel
    s = re.sub(r"[\u064b-\u065f\u0670\u0640]", "", s)
    s = s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    s = s.replace("ة", "ه").replace("ى", "ي")
    return s


@dataclass
class ParsedRAGQuery:
    """Structured filters and semantic terms extracted from a natural-language wildlife query."""

    raw_query: str
    normalized_query: str
    species: List[str] = field(default_factory=list)
    min_oryx_count: Optional[int] = None
    min_gazelle_count: Optional[int] = None
    time_of_day: List[str] = field(default_factory=list)
    animal_states: List[str] = field(default_factory=list)
    needs_review: Optional[bool] = None
    concept_terms: List[str] = field(default_factory=list)
    free_keywords: List[str] = field(default_factory=list)

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "raw_query": self.raw_query,
            "species": self.species,
            "min_oryx_count": self.min_oryx_count,
            "min_gazelle_count": self.min_gazelle_count,
            "time_of_day": self.time_of_day,
            "animal_states": self.animal_states,
            "needs_review": self.needs_review,
            "concept_terms": self.concept_terms,
        }


def _extract_species_count(norm_q: str, species_tokens: Sequence[str], dual_tokens: Sequence[str]) -> Optional[int]:
    """Detect explicit counts preceding or following a species name (including Arabic dual forms)."""
    # Check dual forms first (e.g., "غزالين" / "غزالان" -> 2)
    for dt in dual_tokens:
        if re.search(rf"(?:^|\s){re.escape(dt)}(?:\s|$)", norm_q):
            return 2

    tokens = re.findall(r"[a-z0-9\u0600-\u06ff]+", norm_q)
    for i, tok in enumerate(tokens):
        is_species_tok = any(tok.startswith(st) or st in tok for st in species_tokens)
        if not is_species_tok:
            continue
        # Look 1-2 tokens before species word (e.g., "ستة غزلان", "6 gazelles")
        for offset in (1, 2):
            if i - offset >= 0:
                prev = tokens[i - offset]
                if prev.isdigit():
                    return int(prev)
                if prev in NUMBER_WORDS:
                    return NUMBER_WORDS[prev]
        # Look 1 token after species word (e.g., "غزال واحد", "مها 3")
        if i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if nxt.isdigit():
                return int(nxt)
            if nxt in NUMBER_WORDS:
                return NUMBER_WORDS[nxt]
    return None


def parse_natural_rag_query(query_text: str) -> ParsedRAGQuery:
    """Parse an Arabic or English natural-language wildlife query into structured filters and concepts."""
    norm_q = normalize_arabic_text(query_text)
    parsed = ParsedRAGQuery(raw_query=query_text, normalized_query=norm_q)

    if not norm_q.strip():
        return parsed

    # 1. Detect species & counts
    gazelle_words = ("غزال", "غزلان", "الريم", "الادمي", "gazelle", "gazelles")
    gazelle_dual = ("غزالين", "غزالان")
    oryx_words = ("مها", "المها", "وضيحي", "الوضيحي", "oryx")
    oryx_dual = ("مهاوين", "مهاين", "وضيحيين")

    has_gazelle = any(w in norm_q for w in gazelle_words) or any(w in norm_q for w in gazelle_dual)
    has_oryx = any(w in norm_q for w in oryx_words) or any(w in norm_q for w in oryx_dual)

    if has_gazelle:
        parsed.species.append("arabian_gazelle")
        g_cnt = _extract_species_count(norm_q, gazelle_words, gazelle_dual)
        parsed.min_gazelle_count = g_cnt if g_cnt is not None else 1

    if has_oryx:
        parsed.species.append("arabian_oryx")
        o_cnt = _extract_species_count(norm_q, oryx_words, oryx_dual)
        parsed.min_oryx_count = o_cnt if o_cnt is not None else 1

    # 2. Detect time of day
    tod_set: List[str] = []
    for triggers, tod_values in TIME_OF_DAY_PATTERNS:
        if any(normalize_arabic_text(tr) in norm_q for tr in triggers):
            for v in tod_values:
                if v not in tod_set:
                    tod_set.append(v)
    parsed.time_of_day = tod_set

    # 3. Detect review flag intent
    if any(k in norm_q for k in ("مراجعه", "تحتاج مراجعه", "تحتاج الي مراجعه", "needs review", "unverified")):
        parsed.needs_review = True
    elif any(k in norm_q for k in ("موكده", "موثوقه", "بدون مراجعه", "verified only")):
        parsed.needs_review = False

    # 4. Detect habitat/background/behavior concepts
    concept_terms: List[str] = []
    state_tags: List[str] = []
    for triggers, match_terms, state_tag, _ in CONCEPT_SYNONYMS:
        if any(normalize_arabic_text(tr) in norm_q for tr in triggers):
            for term in match_terms:
                nt = normalize_arabic_text(term)
                if nt not in concept_terms:
                    concept_terms.append(nt)
            if state_tag and state_tag not in state_tags:
                state_tags.append(state_tag)

    parsed.concept_terms = concept_terms
    parsed.animal_states = state_tags

    # 5. Retain meaningful free-text tokens for fuzzy boosting
    ignored = set(STOPWORDS) | set(NUMBER_WORDS.keys()) | set(gazelle_words) | set(oryx_words)
    for triggers, _ in TIME_OF_DAY_PATTERNS:
        ignored.update(normalize_arabic_text(t) for t in triggers)
    raw_tokens = re.findall(r"[a-z0-9\u0600-\u06ff]+", norm_q)
    parsed.free_keywords = [
        tok for tok in raw_tokens if len(tok) >= 3 and not tok.isdigit() and tok not in ignored
    ]

    return parsed


def _build_chromadb_where(query_filter: RAGQueryFilter) -> Optional[Dict[str, Any]]:
    """Convert RAGQueryFilter into a valid ChromaDB metadata `where` dictionary."""
    clauses: List[Dict[str, Any]] = []
    if query_filter.min_oryx_count is not None:
        clauses.append({"oryx_count": {"$gte": int(query_filter.min_oryx_count)}})
    if query_filter.max_oryx_count is not None:
        clauses.append({"oryx_count": {"$lte": int(query_filter.max_oryx_count)}})
    if query_filter.min_gazelle_count is not None:
        clauses.append({"gazelle_count": {"$gte": int(query_filter.min_gazelle_count)}})
    if query_filter.max_gazelle_count is not None:
        clauses.append({"gazelle_count": {"$lte": int(query_filter.max_gazelle_count)}})
    if query_filter.min_conf is not None:
        clauses.append({"avg_conf": {"$gte": float(query_filter.min_conf)}})
    if query_filter.max_conf is not None:
        clauses.append({"avg_conf": {"$lte": float(query_filter.max_conf)}})
    if query_filter.needs_review is not None:
        clauses.append({"needs_review": {"$eq": bool(query_filter.needs_review)}})
    if query_filter.time_of_day:
        if len(query_filter.time_of_day) == 1:
            clauses.append({"estimated_time_of_day": {"$eq": query_filter.time_of_day[0]}})
        else:
            clauses.append({"estimated_time_of_day": {"$in": list(query_filter.time_of_day)}})
    if query_filter.environment_type:
        clauses.append({"environment_type": {"$eq": query_filter.environment_type}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _score_document(doc: RAGDocument, parsed: ParsedRAGQuery) -> Tuple[float, bool]:
    """Score a candidate RAGDocument against parsed concepts and free keywords.

    Returns (score, concept_matched). When concepts are requested (e.g. 'تحت الشجرة'),
    requires at least one matching synonym in text/background/environment/states.
    """
    m = doc.metadata
    haystack = normalize_arabic_text(
        " ".join(
            [
                doc.text,
                doc.text_ar,
                doc.text_en,
                m.environment_type,
                " ".join(m.animal_states),
                " ".join(m.background_elements),
                m.video_source,
            ]
        )
    )

    score = float(m.avg_conf) * 0.5
    if parsed.concept_terms:
        hits = sum(1 for term in parsed.concept_terms if term in haystack)
        if hits == 0:
            return 0.0, False
        score += hits * 2.0

    for kw in parsed.free_keywords:
        if kw in haystack:
            score += 1.0

    # Reward exact count proximity when user specifies a count (e.g. 6 gazelles)
    if parsed.min_gazelle_count is not None and m.gazelle_count == parsed.min_gazelle_count:
        score += 1.5
    if parsed.min_oryx_count is not None and m.oryx_count == parsed.min_oryx_count:
        score += 1.5

    return round(score, 4), True


def _collect_media_search_roots(
    rag_jsonl_path: Optional[Path],
    chromadb_dir: Optional[Path],
) -> List[Path]:
    """Gather local candidate directories where Colab-exported wildlife images may reside."""
    candidates: List[Path] = []
    workspace_root = Path(__file__).resolve().parents[2]

    if rag_jsonl_path is not None:
        parent = rag_jsonl_path.parent
        candidates.extend([parent, parent / "buckets_backup"])

    if chromadb_dir is not None:
        base = chromadb_dir.resolve().parent.parent
        candidates.extend(
            [
                base / "out_wildlife_colab",
                base / "out_wildlife_colab" / "buckets_backup",
                base / "out_wildlife",
                base / "out_wildlife" / "buckets_backup",
            ]
        )

    candidates.extend(
        [
            workspace_root / "out_wildlife_colab",
            workspace_root / "out_wildlife_colab" / "buckets_backup",
            workspace_root / "out_wildlife",
            workspace_root / "out_wildlife" / "buckets_backup",
        ]
    )

    unique_roots: List[Path] = []
    seen: set[str] = set()
    for root in candidates:
        key = str(root)
        if key not in seen and root.exists():
            seen.add(key)
            unique_roots.append(root)
    return unique_roots


def resolve_local_media_path(raw_path: str, search_roots: List[Path]) -> str:
    """Remap cross-platform (e.g., Colab `/content/out_wildlife_colab/...`) image paths to local files."""
    if not raw_path:
        return ""
    direct = Path(raw_path)
    if direct.exists():
        return str(direct)

    norm = raw_path.replace("\\", "/")
    rel_candidates: List[str] = []
    for marker in (
        "trusted_rag_above_75/",
        "qwen_review_below_95/",
        "debug_compare/",
        "out_wildlife_colab/",
        "out_wildlife/",
    ):
        idx = norm.find(marker)
        if idx != -1:
            if marker in ("out_wildlife_colab/", "out_wildlife/"):
                rel_candidates.append(norm[idx + len(marker) :])
            else:
                rel_candidates.append(norm[idx:])

    filename = norm.rsplit("/", 1)[-1]
    if filename:
        rel_candidates.extend(
            [
                f"trusted_rag_above_75/images/{filename}",
                f"qwen_review_below_95/images/{filename}",
                f"images/{filename}",
                filename,
            ]
        )

    for root in search_roots:
        for rel in rel_candidates:
            candidate = root / Path(rel)
            if candidate.exists():
                return str(candidate)

    return raw_path


def _format_rag_answer_markdown(
    query_text: str,
    parsed: ParsedRAGQuery,
    matches: List[Dict[str, Any]],
    backend_used: str,
) -> str:
    """Format a clear Arabic answer with video name, timestamp, frame, counts, confidence, and description."""
    lines: List[str] = [
        f"### نتائج البحث عن: «{query_text or 'الكل'}»",
        f"- **مصدر البحث:** `{backend_used}` | **عدد النتائج المطابقة:** **{len(matches)}**",
    ]

    extracted_parts: List[str] = []
    if parsed.species:
        sp_names = "، ".join(SPECIES_DISPLAY_AR.get(s, s) for s in parsed.species)
        extracted_parts.append(f"النوع: {sp_names}")
    if parsed.min_oryx_count:
        extracted_parts.append(f"المها ≥ {parsed.min_oryx_count}")
    if parsed.min_gazelle_count:
        extracted_parts.append(f"الغزلان ≥ {parsed.min_gazelle_count}")
    if parsed.time_of_day:
        extracted_parts.append(f"الوقت: {', '.join(parsed.time_of_day)}")
    if parsed.animal_states:
        extracted_parts.append(f"السلوك: {', '.join(parsed.animal_states)}")
    if extracted_parts:
        lines.append(f"- **فلاتر مستخرجة تلقائياً:** {' | '.join(extracted_parts)}")

    lines.append("")
    if not matches:
        lines.append("_لم يتم العثور على مشاهد مطابقة لهذا الاستعلام والفلاتر المحددة._")
        return "\n".join(lines)

    for idx, item in enumerate(matches, start=1):
        review_badge = "تحتاج مراجعة" if item["needs_review"] else "موثّقة"
        lines.append(
            f"#### {idx}. 🎬 المقطع: `{item['video_source']}` — الفريم `{item['best_frame']}` — الوقت `{item['best_hms']}`"
        )
        lines.append(
            f"- **نافذة الظهور:** `{item['start_hms']} -> {item['end_hms']}` "
            f"(الفريمات `{item['start_frame']} -> {item['end_frame']}`)"
        )
        lines.append(
            f"- **الأعداد:** المها العربي `{item['oryx_count']}` | الغزال العربي `{item['gazelle_count']}` | "
            f"المجموع `{item['total_animals']}`"
        )
        lines.append(
            f"- **الدقة:** متوسط `{item['avg_conf'] * 100:.1f}%` (أعلى `{item['max_conf'] * 100:.1f}%`) | "
            f"**الوقت المتوقع:** `{item['estimated_time_of_day']}` | **البيئة:** `{item['environment_type']}` | "
            f"**الحالة:** {review_badge}"
        )
        lines.append(f"- **الوصف:** {item['text']}")
        lines.append(f"- **مسار الصورة:** `{item['image_path']}`")
        lines.append("")

    return "\n".join(lines)


def search_wildlife_rag(
    query_text: str = "",
    rag_jsonl_path: Optional[str | Path] = None,
    chromadb_dir: Optional[str | Path] = None,
    chromadb_collection: str = "arabian_wildlife_rag",
    embedding_model_id: str = DEFAULT_EMBEDDING_MODEL_ID,
    species_override: Optional[List[str]] = None,
    min_oryx_count: Optional[int] = None,
    min_gazelle_count: Optional[int] = None,
    time_of_day_override: Optional[List[str]] = None,
    min_conf: Optional[float] = None,
    needs_review: Optional[bool] = None,
    video_source: Optional[str] = None,
    limit: int = 12,
    include_temporal_frames_in_gallery: bool = False,
) -> Dict[str, Any]:
    """Execute hybrid wildlife RAG search across ChromaDB and/or JSONL and return answer + image gallery items."""
    parsed = parse_natural_rag_query(query_text)

    effective_species = species_override if species_override else (parsed.species or None)
    effective_min_oryx = min_oryx_count if (min_oryx_count is not None and min_oryx_count > 0) else parsed.min_oryx_count
    effective_min_gazelle = (
        min_gazelle_count if (min_gazelle_count is not None and min_gazelle_count > 0) else parsed.min_gazelle_count
    )
    effective_tod = time_of_day_override if time_of_day_override else (parsed.time_of_day or None)
    effective_review = needs_review if needs_review is not None else parsed.needs_review

    query_filter = RAGQueryFilter(
        video_source=video_source or None,
        species=effective_species,
        min_oryx_count=effective_min_oryx,
        min_gazelle_count=effective_min_gazelle,
        min_conf=min_conf if (min_conf is not None and min_conf > 0.0) else None,
        needs_review=effective_review,
        time_of_day=effective_tod,
    )

    matches: List[Dict[str, Any]] = []
    backend_used = "jsonl_hybrid"

    # Prefer ChromaDB semantic + metadata search when persist directory exists and chromadb is installed
    chroma_path = Path(chromadb_dir) if chromadb_dir else None
    if chroma_path and chroma_path.exists():
        try:
            where_clause = _build_chromadb_where(query_filter)
            raw_chroma = query_chromadb_rag(
                persist_dir=chroma_path,
                query_text=query_text or "wildlife",
                collection_name=chromadb_collection,
                where_filter=where_clause,
                n_results=max(1, limit),
                embedding_model_id=embedding_model_id,
            )
            ids_row = (raw_chroma.get("ids") or [[]])[0]
            docs_row = (raw_chroma.get("documents") or [[]])[0]
            metas_row = (raw_chroma.get("metadatas") or [[]])[0]
            dists_row = (raw_chroma.get("distances") or [[0.0] * len(ids_row)])[0]
            backend_used = f"chromadb:{chromadb_collection}"

            for doc_id, text_val, meta, dist in zip(ids_row, docs_row, metas_row, dists_row):
                meta = meta or {}
                if video_source and video_source.lower() not in str(meta.get("video_source", "")).lower():
                    continue
                t_csv = str(meta.get("temporal_frame_paths_csv", "")).strip()
                t_paths = [p for p in t_csv.split(",") if p] if t_csv else []
                matches.append(
                    {
                        "doc_id": doc_id,
                        "score": round(max(0.0, 1.0 - float(dist or 0.0)), 4),
                        "text": str(text_val or ""),
                        "video_source": str(meta.get("video_source", "")),
                        "image_path": str(meta.get("image_path", "")),
                        "temporal_frame_paths": t_paths,
                        "track_id": int(meta.get("track_id", 0)),
                        "start_frame": int(meta.get("start_frame", 0)),
                        "end_frame": int(meta.get("end_frame", 0)),
                        "best_frame": int(meta.get("best_frame", 0)),
                        "start_hms": str(meta.get("start_hms", "00:00:00")),
                        "end_hms": str(meta.get("end_hms", "00:00:00")),
                        "best_hms": str(meta.get("best_hms", "00:00:00")),
                        "oryx_count": int(meta.get("oryx_count", 0)),
                        "gazelle_count": int(meta.get("gazelle_count", 0)),
                        "total_animals": int(meta.get("total_animals", 0)),
                        "avg_conf": float(meta.get("avg_conf", 0.0)),
                        "max_conf": float(meta.get("max_conf", 0.0)),
                        "needs_review": bool(meta.get("needs_review", False)),
                        "estimated_time_of_day": str(meta.get("estimated_time_of_day", "unknown")),
                        "environment_type": str(meta.get("environment_type", "unknown")),
                    }
                )
        except Exception as exc:
            logger.warning("ChromaDB search unavailable (%s); falling back to JSONL hybrid search.", exc)
            matches = []

    # JSONL hybrid search (primary when ChromaDB is not configured or unavailable)
    jsonl_file = Path(rag_jsonl_path) if rag_jsonl_path else None
    if not matches and jsonl_file and jsonl_file.exists():
        backend_used = f"jsonl:{jsonl_file.name}"
        all_docs = load_rag_documents_jsonl(jsonl_file)
        filtered_docs = filter_rag_documents(all_docs, query_filter)

        scored_pairs: List[Tuple[float, RAGDocument]] = []
        for doc in filtered_docs:
            score, matched_concepts = _score_document(doc, parsed)
            if not matched_concepts:
                continue
            scored_pairs.append((score, doc))

        scored_pairs.sort(key=lambda pair: pair[0], reverse=True)
        for score, doc in scored_pairs[: max(1, limit)]:
            m = doc.metadata
            matches.append(
                {
                    "doc_id": doc.doc_id,
                    "score": score,
                    "text": doc.text,
                    "video_source": m.video_source,
                    "image_path": m.image_path,
                    "temporal_frame_paths": list(m.temporal_frame_paths),
                    "track_id": m.track_id,
                    "start_frame": m.start_frame,
                    "end_frame": m.end_frame,
                    "best_frame": m.best_frame,
                    "start_hms": m.start_hms,
                    "end_hms": m.end_hms,
                    "best_hms": m.best_hms,
                    "oryx_count": m.oryx_count,
                    "gazelle_count": m.gazelle_count,
                    "total_animals": m.total_animals,
                    "avg_conf": m.avg_conf,
                    "max_conf": m.max_conf,
                    "needs_review": m.needs_review,
                    "estimated_time_of_day": m.estimated_time_of_day,
                    "environment_type": m.environment_type,
                }
            )

    # Resolve cross-platform (Colab -> Windows/local) media paths and build Gradio gallery tuples
    media_roots = _collect_media_search_roots(jsonl_file, chroma_path)
    gallery_items: List[Tuple[str, str]] = []
    for item in matches:
        item["image_path"] = resolve_local_media_path(item["image_path"], media_roots)
        item["temporal_frame_paths"] = [
            resolve_local_media_path(p, media_roots) for p in item["temporal_frame_paths"]
        ]

        base_caption = (
            f"{item['video_source']} | ⏱ {item['best_hms']} ({item['start_hms']}->{item['end_hms']}) | "
            f"Frame {item['best_frame']} | مها:{item['oryx_count']} غزال:{item['gazelle_count']} | "
            f"ثقة:{item['avg_conf'] * 100:.0f}%"
        )
        paths_to_show: List[str] = []
        if include_temporal_frames_in_gallery and item["temporal_frame_paths"]:
            paths_to_show = [p for p in item["temporal_frame_paths"] if p and Path(p).exists()]
        if not paths_to_show and item["image_path"] and Path(item["image_path"]).exists():
            paths_to_show = [item["image_path"]]

        for idx_p, img_p in enumerate(paths_to_show, start=1):
            cap = f"[Frame {idx_p}/{len(paths_to_show)}] {base_caption}" if len(paths_to_show) > 1 else base_caption
            gallery_items.append((img_p, cap))

    answer_md = _format_rag_answer_markdown(query_text, parsed, matches, backend_used)
    return {
        "backend": backend_used,
        "parsed_query": parsed.to_summary_dict(),
        "total_matched": len(matches),
        "answer_markdown": answer_md,
        "matches": matches,
        "gallery_items": gallery_items,
    }
