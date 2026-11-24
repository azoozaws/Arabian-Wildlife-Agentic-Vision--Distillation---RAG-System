"""Gradio 6.4+ Dashboard for Arabian Wildlife RAG Search, LangGraph Agent Control, and Model Registry."""

from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

# Disable background telemetry network calls so UI starts instantaneously even offline
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
# Prevent matplotlib (imported by Gradio) from probing PyQt5 and loading PyQt5's outdated 2020 MSVCP140.dll
os.environ.setdefault("MPLBACKEND", "Agg")
if sys.platform == "win32":
    try:
        import ctypes

        _sys_msvcp = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "MSVCP140.dll"
        if _sys_msvcp.exists():
            ctypes.WinDLL(str(_sys_msvcp))
    except Exception:
        pass

import gradio as gr

_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from app.data.rag_query_engine import search_wildlife_rag
from app.data.rag_store import rebuild_chromadb_from_jsonl
from app.src.agent import WildlifeAgentRunner, get_graph_mermaid
from app.src.agent.agent_state import WildlifeAgentState
from app.src.config import DEFAULT_EMBEDDING_MODEL_ID
from app.src.training.model_registry import (
    DEFAULT_MODELS_DIR,
    DEFAULT_REGISTRY_PATH,
    ModelRegistry,
)

CUSTOM_CSS = """
.gradio-container {
    font-family: "Cairo", "Tajawal", "Segoe UI", system-ui, -apple-system, sans-serif !important;
    max-width: 1460px !important;
}
.rtl-box, .rtl-box textarea, .rtl-box input, .rtl-md {
    direction: rtl;
    text-align: right;
}
.rtl-box label, .rtl-box .wrap {
    direction: rtl;
    text-align: right;
}
.hero-banner {
    background: radial-gradient(circle at 12% 18%, rgba(245, 158, 11, 0.22), transparent 36%),
                linear-gradient(135deg, #072227 0%, #0f3d3e 48%, #165a4f 80%, #24483e 100%);
    border: 1px solid rgba(52, 211, 153, 0.35);
    border-bottom: 3px solid #d97706;
    color: #ffffff;
    padding: 20px 26px;
    border-radius: 16px;
    margin-bottom: 14px;
    direction: rtl;
    text-align: right;
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.32);
}
.hero-banner h1 {
    color: #ffffff !important;
    font-size: 1.55rem !important;
    font-weight: 800 !important;
    margin: 0 0 6px 0 !important;
    letter-spacing: -0.01em;
}
.hero-banner p {
    color: #d1fae5 !important;
    font-size: 0.94rem !important;
    margin: 0 0 12px 0 !important;
    opacity: 0.95;
}
.hero-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
}
.hero-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.80rem;
    font-weight: 600;
    background: rgba(6, 24, 22, 0.65);
    color: #ecfdf5;
    border: 1px solid rgba(52, 211, 153, 0.40);
    backdrop-filter: blur(6px);
}
.hero-chip-gold {
    background: rgba(180, 83, 9, 0.32);
    color: #fde68a;
    border-color: rgba(245, 158, 11, 0.55);
}
.info-callout {
    direction: rtl;
    text-align: right;
    background: linear-gradient(90deg, rgba(15, 42, 38, 0.72), rgba(17, 24, 39, 0.85));
    border: 1px solid rgba(52, 211, 153, 0.28);
    border-right: 4px solid #10b981;
    border-radius: 12px;
    padding: 12px 16px;
    color: #e2e8f0;
    font-size: 0.88rem;
    line-height: 1.75;
    margin: 4px 0 8px 0;
}
.info-callout kbd {
    display: inline-block;
    padding: 1px 7px;
    margin: 0 2px;
    border-radius: 6px;
    background: rgba(30, 41, 59, 0.95);
    border: 1px solid rgba(148, 163, 184, 0.45);
    color: #f8fafc;
    font-family: ui-monospace, Consolas, monospace;
    font-size: 0.78rem;
}
.swatch-green, .swatch-orange {
    display: inline-block;
    width: 11px;
    height: 11px;
    border-radius: 3px;
    vertical-align: middle;
    margin-left: 4px;
}
.swatch-green { background: #22c55e; border: 1px solid #86efac; }
.swatch-orange { background: #f59e0b; border: 1px solid #fde68a; }
.rag-detail-panel {
    background: rgba(15, 23, 34, 0.72);
    border: 1px solid rgba(52, 211, 153, 0.22);
    border-radius: 12px;
    padding: 14px 18px;
    max-height: 540px;
    overflow-y: auto;
}
.rag-summary-bar {
    direction: rtl;
    text-align: right;
    background: linear-gradient(90deg, rgba(16, 61, 52, 0.85), rgba(17, 24, 39, 0.92));
    border: 1px solid rgba(52, 211, 153, 0.35);
    border-radius: 10px;
    padding: 10px 16px;
    margin-bottom: 14px;
    color: #ecfdf5;
    font-size: 0.94rem;
}
.rag-cards-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
    gap: 16px;
    direction: rtl;
}
.rag-card {
    background: #111827;
    border: 1px solid rgba(52, 211, 153, 0.28);
    border-radius: 14px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.30);
    transition: transform 0.16s ease, border-color 0.16s ease, box-shadow 0.16s ease;
}
.rag-card:hover {
    transform: translateY(-2px);
    border-color: rgba(52, 211, 153, 0.78);
    box-shadow: 0 10px 26px rgba(16, 185, 129, 0.18);
}
.rag-card-header {
    background: linear-gradient(180deg, rgba(15, 45, 38, 0.96), rgba(17, 24, 39, 0.98));
    padding: 11px 14px;
    border-bottom: 1px solid rgba(52, 211, 153, 0.22);
    direction: rtl;
    text-align: right;
}
.rag-card-top-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 7px;
}
.rag-card-title {
    color: #f8fafc;
    font-weight: 700;
    font-size: 0.88rem;
    word-break: break-word;
    line-height: 1.35;
}
.rag-card-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
}
.rag-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.77rem;
    background: rgba(30, 41, 59, 0.92);
    color: #e2e8f0;
    border: 1px solid rgba(148, 163, 184, 0.25);
}
.rag-pill-accent {
    background: rgba(16, 185, 129, 0.16);
    color: #6ee7b7;
    border-color: rgba(52, 211, 153, 0.40);
}
.rag-badge-ok {
    padding: 2px 9px;
    border-radius: 999px;
    font-size: 0.73rem;
    font-weight: 700;
    background: rgba(16, 185, 129, 0.20);
    color: #6ee7b7;
    border: 1px solid rgba(16, 185, 129, 0.45);
    white-space: nowrap;
}
.rag-badge-rev {
    padding: 2px 9px;
    border-radius: 999px;
    font-size: 0.73rem;
    font-weight: 700;
    background: rgba(245, 158, 11, 0.20);
    color: #fcd34d;
    border: 1px solid rgba(245, 158, 11, 0.45);
    white-space: nowrap;
}
.rag-card-img-link {
    display: block;
    background: #090d12;
    text-decoration: none;
}
.rag-card-img {
    width: 100%;
    max-height: 300px;
    object-fit: contain;
    display: block;
    background: #090d12;
}
.rag-empty-box {
    direction: rtl;
    text-align: center;
    padding: 28px 16px;
    border-radius: 12px;
    border: 1px dashed rgba(148, 163, 184, 0.35);
    color: #94a3b8;
    font-size: 0.95rem;
}
/* Registry KPI & Benchmark Cards */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 14px;
    direction: rtl;
    margin-bottom: 16px;
}
.kpi-card {
    background: linear-gradient(155deg, rgba(17, 29, 38, 0.96), rgba(15, 23, 34, 0.98));
    border: 1px solid rgba(52, 211, 153, 0.26);
    border-radius: 14px;
    padding: 14px 16px;
    text-align: right;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.24);
}
.kpi-card-gold {
    border-color: rgba(245, 158, 11, 0.50);
    background: linear-gradient(155deg, rgba(42, 28, 14, 0.90), rgba(17, 24, 39, 0.98));
}
.kpi-label {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-bottom: 6px;
    font-weight: 600;
}
.kpi-value {
    font-size: 1.32rem;
    font-weight: 800;
    color: #f8fafc;
    margin-bottom: 4px;
    word-break: break-word;
}
.kpi-sub {
    font-size: 0.78rem;
    color: #6ee7b7;
}
.kpi-sub-gold {
    font-size: 0.78rem;
    color: #fcd34d;
}
.model-bench-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(290px, 1fr));
    gap: 14px;
    direction: rtl;
    margin-bottom: 14px;
}
.bench-card {
    background: #111827;
    border: 1px solid rgba(148, 163, 184, 0.24);
    border-radius: 14px;
    padding: 14px 16px;
    text-align: right;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.22);
}
.bench-card-champion {
    border: 1.5px solid rgba(16, 185, 129, 0.75);
    background: linear-gradient(180deg, rgba(13, 46, 38, 0.92), rgba(17, 24, 39, 0.98));
    box-shadow: 0 6px 22px rgba(16, 185, 129, 0.16);
}
.bench-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 10px;
}
.bench-title {
    font-weight: 800;
    font-size: 0.98rem;
    color: #f8fafc;
}
.bench-ver {
    font-size: 0.74rem;
    color: #94a3b8;
    margin-right: 4px;
}
.bench-row {
    margin-bottom: 8px;
}
.bench-row-head {
    display: flex;
    justify-content: space-between;
    font-size: 0.78rem;
    color: #cbd5e1;
    margin-bottom: 3px;
}
.bench-bar-bg {
    height: 7px;
    width: 100%;
    border-radius: 999px;
    background: rgba(51, 65, 85, 0.75);
    overflow: hidden;
}
.bench-bar-fill {
    height: 100%;
    border-radius: 999px;
}
.bench-delta-up { color: #4ade80; font-weight: 700; margin-right: 4px; }
.bench-delta-down { color: #fbbf24; font-weight: 700; margin-right: 4px; }
.bench-foot {
    margin-top: 10px;
    padding-top: 8px;
    border-top: 1px solid rgba(148, 163, 184, 0.16);
    font-size: 0.75rem;
    color: #94a3b8;
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}
"""

RAG_VIEW_MODE_CHOICES = [
    ("🖼️ عرض الصور مع تفاصيل المقطع فوقها (بدون شرح)", "cards_only"),
    ("📋 عرض تفصيلي مع الشرح على اليسار", "detailed"),
]

DISPLAY_STYLE_CHOICES = [
    ("الوضع الطبيعي — الألوان (أخضر/برتقالي) ودليل الثقة الكامل", "standard"),
    ("وضع الديمو النظيف للتصوير — زوايا خضراء واسم الحيوان و ID فقط", "clean_demo"),
]

RUN_MODE_CHOICES = [
    ("التلخيص فقط — بدون Qwen وبدون تدريب (summary_only)", "summary_only"),
    ("تلخيص + Qwen RAG للـ ChromaDB فقط — بدون تدريب (summary_with_rag)", "summary_with_rag"),
    ("تلخيص + Qwen تحقق و RAG — بدون تدريب YOLO (summary_with_qwen)", "summary_with_qwen"),
    ("تدريب كامل Headless + تلخيص + Qwen + بوابة التقييم (train_only)", "train_only"),
    ("عرض حي + تدريب كامل + تلخيص + Qwen + بوابة التقييم (display_and_train)", "display_and_train"),
    ("عرض الفيديو فقط مع التلخيص بدون حفظ أو تدريب (display_only)", "display_only"),
]

SPECIES_CHOICES = [
    ("المها العربي (arabian_oryx)", "arabian_oryx"),
    ("الغزال العربي (arabian_gazelle)", "arabian_gazelle"),
]

TIME_OF_DAY_CHOICES = [
    ("الصباح (morning)", "morning"),
    ("الظهيرة (midday)", "midday"),
    ("العصر (afternoon)", "afternoon"),
    ("الفجر (dawn)", "dawn"),
    ("الغروب (dusk)", "dusk"),
    ("الليل (night)", "night"),
]


def _default_champion_path() -> str:
    reg = ModelRegistry(DEFAULT_REGISTRY_PATH, DEFAULT_MODELS_DIR)
    active = reg.get_active_champion()
    return active["weights_path"] if active else str(DEFAULT_MODELS_DIR / "yolo_v3_best.pt")


def _discover_default_rag_jsonl() -> str:
    """Quickly check shallow expected paths without scanning .venv or large dataset directories."""
    shallow_candidates = [
        _WORKSPACE_ROOT / "out_wildlife_colab" / "rag_documents.jsonl",
        _WORKSPACE_ROOT / "out_wildlife" / "rag_documents.jsonl",
        _WORKSPACE_ROOT / "app" / "out_wildlife" / "rag_documents.jsonl",
        _WORKSPACE_ROOT / "app" / "rag_documents.jsonl",
        Path("/content/out_wildlife_colab/rag_documents.jsonl"),
        Path("/content/out_wildlife/rag_documents.jsonl"),
    ]
    for p in shallow_candidates:
        if p.exists():
            return str(p)
    return str(_WORKSPACE_ROOT / "out_wildlife_colab" / "rag_documents.jsonl")


def _to_gradio_file_url(img_path: str) -> str:
    """Build a Gradio 6.4+ `/gradio_api/file=...` URL for a local image inside allowed_paths."""
    resolved = Path(img_path).resolve().as_posix()
    return f"/gradio_api/file={quote(resolved, safe='/:')}"


def _render_rag_cards_html(
    res: Optional[Dict[str, Any]],
    query_text: str = "",
    show_3frames: bool = False,
) -> str:
    """Render full-width visual wildlife cards with clip metadata cleanly placed ABOVE each image."""
    if res is None:
        return (
            '<div class="rag-empty-box">'
            "اكتب استعلامك واضغط <b>«بحث في الـ RAG»</b> لعرض الصور مع تفاصيل المقطع فوق كل صورة."
            "</div>"
        )

    matches = res.get("matches") or []
    backend = html.escape(str(res.get("backend", "")))
    q_disp = html.escape(query_text.strip() or "الكل")
    summary_bar = (
        f'<div class="rag-summary-bar">'
        f"🔍 <b>نتائج البحث عن:</b> «{q_disp}» &nbsp;|&nbsp; "
        f"<b>المصدر:</b> <code>{backend}</code> &nbsp;|&nbsp; "
        f"<b>عدد المشاهد المطابقة:</b> <b>{len(matches)}</b>"
        f"</div>"
    )
    if not matches:
        return (
            summary_bar
            + '<div class="rag-empty-box">لم يتم العثور على مشاهد مطابقة لهذا الاستعلام والفلاتر المحددة.</div>'
        )

    cards: List[str] = []
    for idx, item in enumerate(matches, start=1):
        vid = html.escape(str(item.get("video_source", "")))
        best_hms = html.escape(str(item.get("best_hms", "00:00:00")))
        start_hms = html.escape(str(item.get("start_hms", "00:00:00")))
        end_hms = html.escape(str(item.get("end_hms", "00:00:00")))
        best_frame = int(item.get("best_frame", 0))
        track_id = int(item.get("track_id", 0))
        oryx_cnt = int(item.get("oryx_count", 0))
        gaz_cnt = int(item.get("gazelle_count", 0))
        avg_conf_pct = float(item.get("avg_conf", 0.0)) * 100.0
        needs_rev = bool(item.get("needs_review", False))
        badge_html = (
            '<span class="rag-badge-rev">تحتاج مراجعة</span>'
            if needs_rev
            else '<span class="rag-badge-ok">موثّقة</span>'
        )

        paths_to_show: List[str] = []
        if show_3frames and item.get("temporal_frame_paths"):
            paths_to_show = [p for p in item["temporal_frame_paths"] if p and Path(p).exists()]
        if not paths_to_show and item.get("image_path") and Path(item["image_path"]).exists():
            paths_to_show = [item["image_path"]]

        if not paths_to_show:
            paths_to_show = [""]

        for p_idx, img_p in enumerate(paths_to_show, start=1):
            frame_seq_badge = (
                f'<span class="rag-pill">لقطة {p_idx}/{len(paths_to_show)}</span>'
                if len(paths_to_show) > 1
                else ""
            )
            if img_p:
                img_url = _to_gradio_file_url(img_p)
                img_block = (
                    f'<a class="rag-card-img-link" href="{img_url}" target="_blank" '
                    f'title="اضغط لفتح الصورة بالحجم الكامل">'
                    f'<img class="rag-card-img" src="{img_url}" loading="lazy" alt="{vid}" />'
                    f"</a>"
                )
            else:
                img_block = '<div class="rag-empty-box">الصورة غير متوفرة على القرص المحلي</div>'

            cards.append(
                f'<div class="rag-card">'
                f'  <div class="rag-card-header">'
                f'    <div class="rag-card-top-row">'
                f'      <div class="rag-card-title">#{idx} 🎬 {vid}</div>'
                f"      {badge_html}"
                f"    </div>"
                f'    <div class="rag-card-meta">'
                f'      <span class="rag-pill rag-pill-accent">⏱ {best_hms} ({start_hms} ← {end_hms})</span>'
                f'      <span class="rag-pill">🎞️ فريم {best_frame}</span>'
                f'      <span class="rag-pill">Track #{track_id}</span>'
                f'      <span class="rag-pill rag-pill-accent">🦌 مها: {oryx_cnt} | غزال: {gaz_cnt}</span>'
                f'      <span class="rag-pill">🎯 {avg_conf_pct:.1f}%</span>'
                f"      {frame_seq_badge}"
                f"    </div>"
                f"  </div>"
                f"  {img_block}"
                f"</div>"
            )

    return summary_bar + '<div class="rag-cards-grid">' + "".join(cards) + "</div>"


def _fmt_pct(val: Any) -> str:
    if isinstance(val, (int, float)):
        return f"{val * 100:.2f}%"
    return "-"


def _render_hero_html() -> str:
    """Build dynamic hero header showing live active champion & RAG embedding badges."""
    reg = ModelRegistry(DEFAULT_REGISTRY_PATH, DEFAULT_MODELS_DIR)
    active = reg.get_active_champion()
    models = reg.list_models()
    champ_id = active["model_id"] if active else "—"
    champ_ver = active.get("version_tag", "") if active else ""
    champ_m = (active.get("metrics") or {}) if active else {}
    map50_txt = _fmt_pct(champ_m.get("mAP50"))
    prec_txt = _fmt_pct(champ_m.get("precision"))
    return (
        '<div class="hero-banner">'
        "  <h1>🦌 نظام مراقبة الحياة الفطرية العربية — Agentic AI & Multimodal RAG</h1>"
        "  <p>رصد وتتبع حي بـ YOLO + ByteTrack | تحقق وتقطير معرفي ووصف زمني (3 فريمات) عبر Qwen3-VL | بحث دلالي عبر Qwen3-Embedding + ChromaDB | سجل نماذج وبوابة ترقية ذكية</p>"
        '  <div class="hero-chips">'
        f'    <span class="hero-chip hero-chip-gold">🏆 Active Champion: {html.escape(champ_id)} ({html.escape(champ_ver)})</span>'
        f'    <span class="hero-chip">📈 mAP@50: {map50_txt}</span>'
        f'    <span class="hero-chip">🎯 Precision: {prec_txt}</span>'
        f'    <span class="hero-chip">🧠 {html.escape(DEFAULT_EMBEDDING_MODEL_ID)}</span>'
        f'    <span class="hero-chip">📦 الإصدارات المسجّلة: {len(models)}</span>'
        "  </div>"
        "</div>"
    )


def _format_registry_state() -> Tuple[str, List[List[Any]], Any]:
    """Render visual KPI summary cards, per-version benchmark progress bars, table rows, and dropdown update."""
    reg = ModelRegistry(DEFAULT_REGISTRY_PATH, DEFAULT_MODELS_DIR)
    active = reg.get_active_champion()
    prev = reg.get_previous_champion()
    models = reg.list_models()

    # Compute top KPIs across all registered versions
    baseline_m = (models[0].get("metrics") or {}) if models else {}
    best_map50_val, best_map50_id = -1.0, "-"
    best_prec_val, best_prec_id = -1.0, "-"
    best_rec_val, best_rec_id = -1.0, "-"
    for m in models:
        met = m.get("metrics") or {}
        if isinstance(met.get("mAP50"), (int, float)) and met["mAP50"] > best_map50_val:
            best_map50_val, best_map50_id = float(met["mAP50"]), m["model_id"]
        if isinstance(met.get("precision"), (int, float)) and met["precision"] > best_prec_val:
            best_prec_val, best_prec_id = float(met["precision"]), m["model_id"]
        if isinstance(met.get("recall"), (int, float)) and met["recall"] > best_rec_val:
            best_rec_val, best_rec_id = float(met["recall"]), m["model_id"]

    active_id = active["model_id"] if active else "لا يوجد"
    active_ver = active.get("version_tag", "") if active else ""
    active_m = (active.get("metrics") or {}) if active else {}
    prev_id = prev["model_id"] if prev else "لا يوجد"
    prev_ver = prev.get("version_tag", "") if prev else ""

    kpi_html = (
        '<div class="kpi-grid">'
        '  <div class="kpi-card kpi-card-gold">'
        '    <div class="kpi-label">🏆 النموذج المعتمد حالياً (Active Champion)</div>'
        f'    <div class="kpi-value">{html.escape(active_id)}</div>'
        f'    <div class="kpi-sub-gold">الإصدار {html.escape(active_ver)} | mAP@50: {_fmt_pct(active_m.get("mAP50"))} | Precision: {_fmt_pct(active_m.get("precision"))}</div>'
        "  </div>"
        '  <div class="kpi-card">'
        '    <div class="kpi-label">📈 أعلى mAP@50 مسجّل</div>'
        f'    <div class="kpi-value">{f"{best_map50_val * 100:.2f}%" if best_map50_val >= 0 else "-"}</div>'
        f'    <div class="kpi-sub">النموذج: {html.escape(best_map50_id)}</div>'
        "  </div>"
        '  <div class="kpi-card">'
        '    <div class="kpi-label">🎯 أعلى Precision / Recall</div>'
        f'    <div class="kpi-value">{f"{best_prec_val * 100:.2f}%" if best_prec_val >= 0 else "-"} / {f"{best_rec_val * 100:.2f}%" if best_rec_val >= 0 else "-"}</div>'
        f'    <div class="kpi-sub">P: {html.escape(best_prec_id)} | R: {html.escape(best_rec_id)}</div>'
        "  </div>"
        '  <div class="kpi-card">'
        '    <div class="kpi-label">⏪ الاسترجاع الفوري (Previous Champion)</div>'
        f'    <div class="kpi-value">{html.escape(prev_id)}</div>'
        f'    <div class="kpi-sub">الإصدار {html.escape(prev_ver)} | إجمالي النماذج: {len(models)}</div>'
        "  </div>"
        "</div>"
    )

    # Visual benchmark evolution cards for each registered model
    bench_cards: List[str] = []
    metric_specs = [
        ("Precision", "precision", "#10b981"),
        ("Recall", "recall", "#38bdf8"),
        ("mAP@50", "mAP50", "#f59e0b"),
        ("mAP@50-95", "mAP50_95", "#a78bfa"),
    ]
    status_badges = {
        "champion": '<span class="rag-badge-ok">🏆 Champion</span>',
        "candidate": '<span class="rag-badge-rev">⏳ Candidate</span>',
        "archived": '<span class="rag-pill">📦 Archived</span>',
        "rejected": '<span class="rag-badge-rev">❌ Rejected</span>',
    }

    rows: List[List[Any]] = []
    model_ids: List[str] = []
    for idx_m, m in enumerate(models):
        metrics = m.get("metrics") or {}
        tmeta = m.get("training_metadata") or {}
        model_ids.append(m["model_id"])
        status = str(m.get("status", ""))
        is_champ = status == "champion"
        card_cls = "bench-card bench-card-champion" if is_champ else "bench-card"
        badge = status_badges.get(status, f'<span class="rag-pill">{html.escape(status)}</span>')

        bars_html: List[str] = []
        for label, key, bar_col in metric_specs:
            val = metrics.get(key)
            if isinstance(val, (int, float)):
                pct_val = max(0.0, min(100.0, float(val) * 100.0))
                delta_html = ""
                if idx_m > 0 and isinstance(baseline_m.get(key), (int, float)):
                    diff_pts = (float(val) - float(baseline_m[key])) * 100.0
                    if abs(diff_pts) >= 0.005:
                        d_cls = "bench-delta-up" if diff_pts > 0 else "bench-delta-down"
                        sign = "+" if diff_pts > 0 else ""
                        delta_html = f'<span class="{d_cls}">({sign}{diff_pts:.2f}%)</span>'
                val_txt = f"{pct_val:.2f}%{delta_html}"
            else:
                pct_val = 0.0
                val_txt = "—"
            bars_html.append(
                f'<div class="bench-row">'
                f'  <div class="bench-row-head"><span>{label}</span><span>{val_txt}</span></div>'
                f'  <div class="bench-bar-bg"><div class="bench-bar-fill" style="width:{pct_val:.1f}%;background:{bar_col};"></div></div>'
                f"</div>"
            )

        foot_pills: List[str] = []
        if m.get("parent_model_id"):
            foot_pills.append(f'<span class="rag-pill">الأصل: {html.escape(str(m["parent_model_id"]))}</span>')
        if tmeta.get("disagreement_debug_images_saved") is not None:
            foot_pills.append(
                f'<span class="rag-pill rag-pill-accent">صور الاختلاف: {int(tmeta["disagreement_debug_images_saved"])}</span>'
            )
        if isinstance(metrics.get("fp_per_hour"), (int, float)):
            foot_pills.append(f'<span class="rag-pill">FP/hr: {metrics["fp_per_hour"]:.1f}</span>')
        if m.get("created_at"):
            foot_pills.append(f'<span class="rag-pill">📅 {html.escape(str(m["created_at"])[:10])}</span>')

        bench_cards.append(
            f'<div class="{card_cls}">'
            f'  <div class="bench-top">'
            f'    <div class="bench-title">{html.escape(m["model_id"])} <span class="bench-ver">({html.escape(str(m.get("version_tag", "")))})</span></div>'
            f"    {badge}"
            f"  </div>"
            f'  {"".join(bars_html)}'
            f'  <div class="bench-foot">{"".join(foot_pills)}</div>'
            f"</div>"
        )

        disagree_val = tmeta.get("disagreement_debug_images_saved", "-")
        fp_val = f"{metrics['fp_per_hour']:.1f}" if isinstance(metrics.get("fp_per_hour"), (int, float)) else "-"
        rows.append(
            [
                m["model_id"],
                m.get("version_tag", ""),
                status,
                m.get("parent_model_id", "") or "-",
                _fmt_pct(metrics.get("precision")),
                _fmt_pct(metrics.get("recall")),
                _fmt_pct(metrics.get("mAP50")),
                _fmt_pct(metrics.get("mAP50_95")),
                disagree_val,
                fp_val,
                m.get("gate_decision", "-"),
                m.get("weights_path", ""),
            ]
        )

    dashboard_html = kpi_html + '<div class="model-bench-grid">' + "".join(bench_cards) + "</div>"
    dropdown_update = gr.update(choices=model_ids, value=model_ids[-1] if model_ids else None)
    return dashboard_html, rows, dropdown_update


def build_gradio_app() -> gr.Blocks:
    """Create the full Gradio 6.4+ Blocks interface without starting the server."""
    runner = WildlifeAgentRunner(
        registry_path=DEFAULT_REGISTRY_PATH,
        models_dir=DEFAULT_MODELS_DIR,
        interrupt_on_hitl=True,
        allow_fallback=True,
    )

    with gr.Blocks(title="Arabian Wildlife AI Agent & RAG Dashboard", analytics_enabled=False) as demo:
        active_thread_id = gr.State(value="")
        rag_last_state = gr.State(value=None)

        gr.HTML(_render_hero_html())

        with gr.Tabs():
            # ==================== TAB 1: RAG SEARCH ====================
            with gr.Tab("🔍 البحث الذكي في المشاهد (RAG)"):
                gr.Markdown(
                    "اكتب سؤالك باللغة العربية أو الإنجليزية مثل: **«مها تحت الشجرة»**، **«Oryx under Tree»**، أو **«سيارة بيضاء»**، "
                    "واختر عرض الصور ببطاقات سريعة أو العرض التفصيلي مع الشرح الكامل.",
                    elem_classes=["rtl-md"],
                )

                with gr.Row():
                    rag_query_input = gr.Textbox(
                        label="استعلام البحث باللغة الطبيعية",
                        placeholder="مثال: مها تحت الشجرة  أو  Oryx under Tree  أو  مها يجري بجانب السياج",
                        scale=4,
                        elem_classes=["rtl-box"],
                    )
                    rag_search_btn = gr.Button("🔍 ابحث في المشاهد", variant="primary", scale=1)

                rag_view_mode = gr.Radio(
                    label="نمط عرض نتائج البحث (تبديل فوري)",
                    choices=RAG_VIEW_MODE_CHOICES,
                    value="cards_only",
                    elem_classes=["rtl-box"],
                )

                gr.Examples(
                    examples=[
                        ["مها تحت الشجرة"],
                        ["Oryx under Tree"],
                        ["سيارة بيضاء"],
                        ["مها يجري بجانب السياج"],
                        ["غزال في النهار"],
                        ["صور تحتاج مراجعة"],
                    ],
                    inputs=[rag_query_input],
                    label="أمثلة سريعة للاستعلام",
                )

                with gr.Accordion("⚙️ مصادر البيانات وفلاتر التصفية المتقدمة", open=False):
                    with gr.Row():
                        rag_jsonl_input = gr.Textbox(
                            label="مسار ملف rag_documents.jsonl",
                            value=_discover_default_rag_jsonl(),
                        )
                        rag_chroma_dir = gr.Textbox(
                            label="مجلد ChromaDB (اختياري)",
                            value=str(_WORKSPACE_ROOT / "app" / "chroma_db"),
                        )
                        rag_chroma_col = gr.Textbox(
                            label="اسم المجموعة (Collection)",
                            value="arabian_wildlife_rag",
                        )

                    with gr.Row():
                        rag_embed_model = gr.Textbox(
                            label="نموذج التضمين (Embedding Model)",
                            value=DEFAULT_EMBEDDING_MODEL_ID,
                            scale=3,
                        )
                        rag_rebuild_btn = gr.Button("♻️ إعادة بناء ChromaDB بهذا النموذج", variant="secondary", scale=1)
                    rag_rebuild_status = gr.Markdown(elem_classes=["rtl-md"])

                    with gr.Row():
                        rag_species_filter = gr.CheckboxGroup(
                            label="تصفية النوع (اختياري)",
                            choices=SPECIES_CHOICES,
                            value=[],
                        )
                        rag_tod_filter = gr.CheckboxGroup(
                            label="الوقت من اليوم (اختياري)",
                            choices=TIME_OF_DAY_CHOICES,
                            value=[],
                        )

                    with gr.Row():
                        rag_min_oryx = gr.Number(label="أدنى عدد للمها العربي", value=0, precision=0)
                        rag_min_gazelle = gr.Number(label="أدنى عدد للغزال العربي", value=0, precision=0)
                        rag_min_conf = gr.Slider(label="أدنى متوسط دقة", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        rag_review_filter = gr.Dropdown(
                            label="حالة المراجعة",
                            choices=[("الكل", "all"), ("تحتاج مراجعة فقط", "needs_review"), ("موثّقة فقط", "verified")],
                            value="all",
                        )

                    with gr.Row():
                        rag_video_filter = gr.Textbox(label="تصفية باسم المقطع (اختياري)", placeholder="مثال: desert_cam.mp4")
                        rag_limit = gr.Slider(label="أقصى عدد للنتائج", minimum=1, maximum=50, value=12, step=1)
                        rag_show_3frames = gr.Checkbox(label="عرض الفريمات الـ 3 الزمنية في المعرض إن وُجدت", value=False)

                rag_cards_html = gr.HTML(
                    value=_render_rag_cards_html(None),
                    visible=True,
                )

                with gr.Row(visible=False) as rag_detailed_row:
                    with gr.Column(scale=5, elem_classes=["rag-detail-panel"]):
                        rag_answer_md = gr.Markdown(
                            value="_اكتب استعلامك واضغط «ابحث في المشاهد» لعرض النتائج والتفاصيل._",
                            elem_classes=["rtl-md"],
                        )
                    with gr.Column(scale=5):
                        rag_gallery = gr.Gallery(
                            label="🖼️ صور المشاهد المطابقة (مع اسم المقطع والوقت والفريم)",
                            columns=2,
                            height=540,
                            object_fit="contain",
                            allow_preview=True,
                        )

                with gr.Accordion("📑 الجدول الهيكلي الكامل للنتائج", open=False):
                    rag_table = gr.Dataframe(
                        headers=["المقطع", "Track", "الفريم", "وقت الظهور", "ذروة الوضوح", "مها", "غزال", "الدقة", "مراجعة", "مسار الصورة"],
                        interactive=False,
                    )

                def _on_rag_search(
                    query_text: str,
                    view_mode: str,
                    jsonl_path: str,
                    chroma_dir: str,
                    chroma_col: str,
                    embed_model: str,
                    species_sel: List[str],
                    tod_sel: List[str],
                    min_oryx: float,
                    min_gaz: float,
                    min_conf_val: float,
                    review_opt: str,
                    video_src: str,
                    limit_val: int,
                    show_3frames: bool,
                ):
                    review_bool: Optional[bool] = None
                    if review_opt == "needs_review":
                        review_bool = True
                    elif review_opt == "verified":
                        review_bool = False

                    res = search_wildlife_rag(
                        query_text=query_text or "",
                        rag_jsonl_path=jsonl_path.strip() or None,
                        chromadb_dir=chroma_dir.strip() or None,
                        chromadb_collection=chroma_col.strip() or "arabian_wildlife_rag",
                        embedding_model_id=embed_model.strip() or DEFAULT_EMBEDDING_MODEL_ID,
                        species_override=species_sel or None,
                        min_oryx_count=int(min_oryx) if (min_oryx and int(min_oryx) > 0) else None,
                        min_gazelle_count=int(min_gaz) if (min_gaz and int(min_gaz) > 0) else None,
                        time_of_day_override=tod_sel or None,
                        min_conf=float(min_conf_val) if min_conf_val and min_conf_val > 0 else None,
                        needs_review=review_bool,
                        video_source=video_src.strip() if video_src else None,
                        limit=int(limit_val),
                        include_temporal_frames_in_gallery=bool(show_3frames),
                    )
                    rows = [
                        [
                            m["video_source"],
                            m["track_id"],
                            m["best_frame"],
                            f"{m['start_hms']} -> {m['end_hms']}",
                            m["best_hms"],
                            m["oryx_count"],
                            m["gazelle_count"],
                            f"{m['avg_conf'] * 100:.1f}%",
                            "نعم" if m["needs_review"] else "لا",
                            m["image_path"],
                        ]
                        for m in res["matches"]
                    ]
                    is_cards = view_mode == "cards_only"
                    cards_markup = _render_rag_cards_html(res, query_text=query_text or "", show_3frames=bool(show_3frames))
                    cached = {"res": res, "query": query_text or "", "show_3frames": bool(show_3frames)}
                    return (
                        gr.update(value=cards_markup, visible=is_cards),
                        gr.update(visible=not is_cards),
                        res["answer_markdown"],
                        res["gallery_items"],
                        rows,
                        cached,
                    )

                def _on_toggle_rag_view(view_mode: str, cached_state: Optional[Dict[str, Any]]):
                    is_cards = view_mode == "cards_only"
                    if cached_state and cached_state.get("res") is not None:
                        cards_markup = _render_rag_cards_html(
                            cached_state["res"],
                            query_text=cached_state.get("query", ""),
                            show_3frames=bool(cached_state.get("show_3frames", False)),
                        )
                    else:
                        cards_markup = _render_rag_cards_html(None)
                    return (
                        gr.update(value=cards_markup, visible=is_cards),
                        gr.update(visible=not is_cards),
                    )

                def _on_rebuild_chroma(jsonl_path: str, chroma_dir: str, chroma_col: str, embed_model: str):
                    if not jsonl_path.strip() or not Path(jsonl_path.strip()).exists():
                        return f"⚠️ ملف JSONL غير موجود: `{jsonl_path}`"
                    res = rebuild_chromadb_from_jsonl(
                        rag_jsonl_paths=jsonl_path.strip(),
                        persist_dir=chroma_dir.strip(),
                        collection_name=chroma_col.strip() or "arabian_wildlife_rag",
                        embedding_model_id=embed_model.strip() or DEFAULT_EMBEDDING_MODEL_ID,
                    )
                    return (
                        f"✅ تمت إعادة بناء ChromaDB بنجاح! "
                        f"الوثائق: **{res['upserted']}** | النموذج: `{res['embedding_model']}` | الأبعاد: `{res['embedding_dim']}`"
                    )

                rag_inputs = [
                    rag_query_input,
                    rag_view_mode,
                    rag_jsonl_input,
                    rag_chroma_dir,
                    rag_chroma_col,
                    rag_embed_model,
                    rag_species_filter,
                    rag_tod_filter,
                    rag_min_oryx,
                    rag_min_gazelle,
                    rag_min_conf,
                    rag_review_filter,
                    rag_video_filter,
                    rag_limit,
                    rag_show_3frames,
                ]
                rag_outputs = [rag_cards_html, rag_detailed_row, rag_answer_md, rag_gallery, rag_table, rag_last_state]
                rag_search_btn.click(_on_rag_search, inputs=rag_inputs, outputs=rag_outputs)
                rag_query_input.submit(_on_rag_search, inputs=rag_inputs, outputs=rag_outputs)
                rag_view_mode.change(
                    _on_toggle_rag_view,
                    inputs=[rag_view_mode, rag_last_state],
                    outputs=[rag_cards_html, rag_detailed_row],
                )
                rag_rebuild_btn.click(
                    _on_rebuild_chroma,
                    inputs=[rag_jsonl_input, rag_chroma_dir, rag_chroma_col, rag_embed_model],
                    outputs=[rag_rebuild_status],
                )

            # ==================== TAB 2: LANGGRAPH AGENT ====================
            with gr.Tab("🤖 تشغيل الوكيل وتحليل المقاطع (LangGraph)"):
                with gr.Row():
                    agent_video_paths = gr.Textbox(
                        label="مسار/مسارات الفيديو (مسار واحد في كل سطر)",
                        placeholder="./videos/desert_cam.mp4",
                        lines=2,
                    )
                    agent_video_upload = gr.File(
                        label="أو ارفع ملف/ملفات فيديو مباشرة",
                        file_count="multiple",
                        type="filepath",
                    )

                with gr.Row():
                    agent_run_mode = gr.Dropdown(
                        label="وضع التشغيل (run_mode)",
                        choices=RUN_MODE_CHOICES,
                        value="summary_only",
                    )
                    agent_champion_weights = gr.Textbox(
                        label="أوزان النموذج المعتمد (Champion .pt)",
                        value=_default_champion_path(),
                    )

                agent_display_style = gr.Radio(
                    label="🎨 نمط العرض الحي للفيديو (عند اختيار display_only أو display_and_train)",
                    choices=DISPLAY_STYLE_CHOICES,
                    value="standard",
                    elem_classes=["rtl-box"],
                )

                gr.HTML(
                    """
                    <div class="info-callout">
                        <div>
                            <b>🎨 دليل الألوان في الوضع الطبيعي:</b>
                            <span class="swatch-green"></span><b>الأخضر</b> = كشف موثوق بثقة عالية (≥ 75%)
                            &nbsp;|&nbsp;
                            <span class="swatch-orange"></span><b>البرتقالي</b> = كشف يحتاج مراجعة (&lt; 75%)
                        </div>
                        <div>
                            <b>⌨️ اختصارات نافذة الفيديو الحي:</b>
                            <kbd>Space</kbd> إيقاف/استئناف &nbsp;·&nbsp;
                            <kbd>←</kbd>/<kbd>→</kbd> أو <kbd>A</kbd>/<kbd>D</kbd> تقديم وتأخير (150 فريم ≈ 5 ثوانٍ) &nbsp;·&nbsp;
                            <kbd>,</kbd>/<kbd>.</kbd> تنقل دقيق (15 فريم، أو فريم واحد عند الإيقاف المؤقت) &nbsp;·&nbsp;
                            <kbd>+</kbd>/<kbd>-</kbd> تغيير السرعة &nbsp;·&nbsp;
                            <kbd>Q</kbd> إغلاق &nbsp;·&nbsp;
                            <b>شريط التمرير العلوي</b> للقفز المباشر لأي فريم
                        </div>
                    </div>
                    """
                )

                with gr.Row():
                    agent_out_dir = gr.Textbox(
                        label="مجلد المخرجات (Workspace)",
                        value=str(_WORKSPACE_ROOT / "out_wildlife"),
                    )
                    agent_lora_dir = gr.Textbox(
                        label="مسار Qwen GRPO LoRA (اختياري)",
                        value="",
                    )
                    agent_chroma_dir = gr.Textbox(
                        label="مجلد ChromaDB (اختياري)",
                        value=str(_WORKSPACE_ROOT / "app" / "chroma_db"),
                    )

                with gr.Accordion("🛠️ إعدادات التدريب وبوابة التقييم (تُستخدم فقط مع train_only / display_and_train)", open=False):
                    with gr.Row():
                        agent_orig_ds = gr.Textbox(
                            label="مجلد الداتا الأصلية (Replay Buffer)",
                            value=str(_WORKSPACE_ROOT / "YOLO_Dataset_Stratified"),
                        )
                        agent_merged_ds = gr.Textbox(
                            label="مجلد الداتا المدمجة",
                            value=str(_WORKSPACE_ROOT / "dataset_local" / "YOLO11_Dataset_v4"),
                        )
                        agent_held_out_vid = gr.Textbox(
                            label="فيديو الاختبار السلبي (لحساب FP/hour — اختياري)",
                            value="",
                        )
                    with gr.Row():
                        agent_train_proj = gr.Textbox(
                            label="مجلد نتائج التدريب",
                            value=str(_WORKSPACE_ROOT / "runs" / "yolo_v4_runs"),
                        )
                        agent_version_tag = gr.Textbox(label="وسم الإصدار (version_tag)", value="v4")

                run_agent_btn = gr.Button("🚀 تشغيل الوكيل (Run LangGraph Agent)", variant="primary")

                with gr.Row():
                    agent_summary_md = gr.Markdown(
                        value="_سيظهر هنا التلخيص الكامل (الأعداد، الثقة، أوقات الظهور والفريمات، ومعدل التسريع ومستوى التحسين)._",
                        elem_classes=["rtl-md"],
                    )
                    agent_status_json = gr.Code(label="حالة الوكيل وقرار البوابة (State / Gate Output)", language="json")

                with gr.Group():
                    gr.Markdown("### 🧑‍💻 تدخل المهندس (Human-in-the-Loop Override — عند توقف البوابة على `needs_hitl`)", elem_classes=["rtl-md"])
                    with gr.Row():
                        hitl_approve_btn = gr.Button("✅ اعتماد النموذج الجديد (Approve Challenger)", variant="primary")
                        hitl_reject_btn = gr.Button("❌ رفض وإبقاء النموذج الحالي (Reject Challenger)", variant="stop")
                    hitl_result_md = gr.Markdown(elem_classes=["rtl-md"])

                def _on_run_agent(
                    paths_text: str,
                    uploaded_files: Optional[List[str]],
                    run_mode: str,
                    display_style: str,
                    champ_weights: str,
                    out_dir: str,
                    lora_dir: str,
                    chroma_dir: str,
                    orig_ds: str,
                    merged_ds: str,
                    held_out: str,
                    train_proj: str,
                    ver_tag: str,
                ):
                    vids = [line.strip() for line in (paths_text or "").splitlines() if line.strip()]
                    if uploaded_files:
                        vids.extend(uploaded_files)
                    if not vids:
                        return "⚠️ يرجى إدخال مسار فيديو واحد على الأقل أو رفع ملف فيديو.", "{}", ""

                    state: WildlifeAgentState = {
                        "run_mode": run_mode,  # type: ignore[typeddict-item]
                        "display_clean_demo_mode": display_style == "clean_demo",
                        "video_paths": vids,
                        "out_workspace_dir": out_dir.strip(),
                        "champion_weights_path": champ_weights.strip(),
                        "qwen_lora_dir": lora_dir.strip() or None,
                        "chromadb_dir": chroma_dir.strip(),
                        "chromadb_collection": "arabian_wildlife_rag",
                        "orig_dataset_dir": orig_ds.strip(),
                        "merged_dataset_dir": merged_ds.strip(),
                        "held_out_video_path": held_out.strip(),
                        "train_project_dir": train_proj.strip(),
                        "version_tag": ver_tag.strip() or "v4",
                        "registry_path": str(DEFAULT_REGISTRY_PATH),
                        "models_dir": str(DEFAULT_MODELS_DIR),
                    }

                    res = runner.run(state)
                    st = res["state"]
                    summary_md = st.get("video_summary_markdown") or "تم التنفيذ بنجاح."
                    compact = {
                        "thread_id": res["thread_id"],
                        "engine": res["engine"],
                        "awaiting_hitl": res["awaiting_hitl"],
                        "run_mode": st.get("run_mode"),
                        "gate_decision": st.get("gate_decision"),
                        "gate_reason": st.get("gate_reason"),
                        "champion_metrics": st.get("champion_metrics"),
                        "challenger_metrics": st.get("challenger_metrics"),
                        "qwen_review_dir": st.get("qwen_review_dir"),
                        "trusted_rag_dir": st.get("trusted_rag_dir"),
                        "rag_jsonl_path": st.get("rag_jsonl_path"),
                        "video_summary_path": st.get("video_summary_path"),
                    }
                    return summary_md, json.dumps(compact, ensure_ascii=False, indent=2), res["thread_id"]

                def _on_hitl(tid: str, decision: str):
                    if not tid:
                        return "⚠️ لا توجد جلسة متوقفة بانتظار المراجعة حالياً.", "{}"
                    res = runner.resume_hitl(tid, engineer_override=decision)  # type: ignore[arg-type]
                    st = res["state"]
                    msg = (
                        f"تم تطبيق قرار المهندس (`{decision}`) — القرار النهائي: **`{st.get('gate_decision')}`** | "
                        f"النموذج المعتمد: `{st.get('active_champion_model_id', st.get('champion_weights_path'))}`"
                    )
                    return msg, json.dumps(
                        {
                            "thread_id": tid,
                            "gate_decision": st.get("gate_decision"),
                            "gate_reason": st.get("gate_reason"),
                            "active_champion_model_id": st.get("active_champion_model_id"),
                            "champion_weights_path": st.get("champion_weights_path"),
                        },
                        ensure_ascii=False,
                        indent=2,
                    )

                run_agent_btn.click(
                    _on_run_agent,
                    inputs=[
                        agent_video_paths,
                        agent_video_upload,
                        agent_run_mode,
                        agent_display_style,
                        agent_champion_weights,
                        agent_out_dir,
                        agent_lora_dir,
                        agent_chroma_dir,
                        agent_orig_ds,
                        agent_merged_ds,
                        agent_held_out_vid,
                        agent_train_proj,
                        agent_version_tag,
                    ],
                    outputs=[agent_summary_md, agent_status_json, active_thread_id],
                )
                hitl_approve_btn.click(
                    lambda tid: _on_hitl(tid, "approve_challenger"),
                    inputs=[active_thread_id],
                    outputs=[hitl_result_md, agent_status_json],
                )
                hitl_reject_btn.click(
                    lambda tid: _on_hitl(tid, "reject_challenger"),
                    inputs=[active_thread_id],
                    outputs=[hitl_result_md, agent_status_json],
                )

            # ==================== TAB 3: MODEL REGISTRY ====================
            with gr.Tab("📦 سجل النماذج والأداء (Registry & Rollback)"):
                init_dashboard_html, init_rows, _ = _format_registry_state()
                init_choices = [r[0] for r in init_rows]

                reg_header_html = gr.HTML(value=init_dashboard_html)
                reg_table = gr.Dataframe(
                    headers=[
                        "Model ID",
                        "Version",
                        "Status",
                        "Parent",
                        "Precision",
                        "Recall",
                        "mAP@50",
                        "mAP@50-95",
                        "Disagreements",
                        "FP/hr",
                        "Gate",
                        "Weights Path",
                    ],
                    label="جدول مقارنة جميع الإصدارات (app/registry.json)",
                    value=init_rows,
                    interactive=False,
                )

                with gr.Row():
                    reg_model_select = gr.Dropdown(
                        label="اختر نموذجاً للترقية أو الاسترجاع إليه",
                        choices=init_choices,
                        value=init_choices[-1] if init_choices else None,
                    )
                    reg_reason_input = gr.Textbox(
                        label="السبب (يُسجّل في history)",
                        value="Manual action via Gradio UI",
                    )

                with gr.Row():
                    reg_refresh_btn = gr.Button("🔄 تحديث السجل")
                    reg_promote_btn = gr.Button("⬆️ ترقية النموذج المختار إلى Champion", variant="primary")
                    reg_rollback_prev_btn = gr.Button("⏪ استرجاع النموذج السابق تلقائياً (Rollback)", variant="stop")
                    reg_rollback_target_btn = gr.Button("↩️ استرجاع للنموذج المختار")

                reg_action_md = gr.Markdown(elem_classes=["rtl-md"])

                def _on_promote(mid: str, reason: str):
                    if not mid:
                        hdr, rows, dd = _format_registry_state()
                        return hdr, rows, dd, "⚠️ اختر نموذجاً أولاً."
                    entry = runner.get_registry().promote_model(mid, gate_reason=reason, engineer_override="approve_challenger")
                    hdr, rows, dd = _format_registry_state()
                    return hdr, rows, dd, f"✅ تمت ترقية `{entry['model_id']}` إلى Active Champion."

                def _on_rollback(mid: Optional[str], reason: str):
                    entry = runner.rollback_champion(target_model_id=mid, reason=reason)
                    hdr, rows, dd = _format_registry_state()
                    return hdr, rows, dd, f"⏪ تم الاسترجاع بنجاح إلى `{entry['model_id']}` (`{entry['weights_path']}`)."

                reg_refresh_btn.click(_format_registry_state, outputs=[reg_header_html, reg_table, reg_model_select])
                reg_promote_btn.click(
                    _on_promote,
                    inputs=[reg_model_select, reg_reason_input],
                    outputs=[reg_header_html, reg_table, reg_model_select, reg_action_md],
                )
                reg_rollback_prev_btn.click(
                    lambda reason: _on_rollback(None, reason),
                    inputs=[reg_reason_input],
                    outputs=[reg_header_html, reg_table, reg_model_select, reg_action_md],
                )
                reg_rollback_target_btn.click(
                    _on_rollback,
                    inputs=[reg_model_select, reg_reason_input],
                    outputs=[reg_header_html, reg_table, reg_model_select, reg_action_md],
                )

                with gr.Accordion("🗺️ مخطط تدفق الوكيل (LangGraph Topology)", open=False):
                    gr.Code(value=get_graph_mermaid(), language="markdown", label="Mermaid Flowchart")

    return demo


def launch_ui(
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    share: bool = False,
    allowed_paths: Optional[List[str]] = None,
    debug: bool = False,
) -> None:
    """Build and launch the Gradio 6.4+ UI with local image serving enabled (Windows & Colab compatible)."""
    import os

    in_colab = "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ
    effective_share = True if in_colab else share
    effective_host = "0.0.0.0" if in_colab else server_name

    demo = build_gradio_app()
    if allowed_paths is not None:
        paths = list(allowed_paths)
    else:
        paths = [str(_WORKSPACE_ROOT), str(_WORKSPACE_ROOT / "app")]
        for colab_root in ("/content", "/content/drive"):
            if Path(colab_root).exists():
                paths.append(colab_root)

    theme = gr.themes.Soft(primary_hue="emerald", secondary_hue="amber")
    demo.launch(
        server_name=effective_host,
        server_port=server_port,
        share=effective_share,
        debug=debug,
        theme=theme,
        css=CUSTOM_CSS,
        allowed_paths=paths,
    )


def _cli_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Launch Arabian Wildlife Gradio Dashboard")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public gradio.live tunnel (auto-enabled on Colab)")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--allowed_paths", nargs="*", default=None)
    args = parser.parse_args()
    launch_ui(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=args.allowed_paths,
        debug=args.debug,
    )


if __name__ == "__main__":
    _cli_main()
