"""Comprehensive wildlife video analysis summary builder (JSON + Arabic Markdown report)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.src.config import SPECIES_DISPLAY_AR
from app.src.schemas import TrackManifestItem
from app.src.utils.geometry import format_hms, norm_species


def classify_speedup_level(speedup_factor: float) -> Tuple[str, str]:
    """Classify processing acceleration ratio into (arabic_label, english_code)."""
    if speedup_factor >= 10.0:
        return "ممتاز جداً (تسريع فائق)", "exceptional"
    if speedup_factor >= 5.0:
        return "ممتاز (تحسين عالٍ)", "excellent"
    if speedup_factor >= 2.0:
        return "جيد جداً (أسرع من الوقت الحقيقي)", "good"
    if speedup_factor >= 1.0:
        return "مقبول (قريب من الوقت الحقيقي)", "realtime"
    return "بطيء (أبطأ من الوقت الحقيقي)", "slower_than_realtime"


def compute_processing_performance(
    video_duration_sec: float,
    processing_time_sec: float,
) -> Dict[str, Any]:
    """Compute nominal video duration vs actual processing time, speedup factor, time saved, and level."""
    dur = max(0.0, float(video_duration_sec))
    proc = max(0.0, float(processing_time_sec))

    if dur > 0.0 and proc > 0.0:
        speedup = dur / proc
        time_saved = max(0.0, dur - proc)
        reduction_pct = ((dur - proc) / dur) * 100.0
    elif dur > 0.0 and proc == 0.0:
        speedup = float(dur)
        time_saved = dur
        reduction_pct = 100.0
    else:
        speedup = 0.0
        time_saved = 0.0
        reduction_pct = 0.0

    level_ar, level_en = classify_speedup_level(speedup)

    return {
        "video_duration_sec": round(dur, 2),
        "video_duration_min": round(dur / 60.0, 2),
        "video_duration_hms": format_hms(dur),
        "processing_time_sec": round(proc, 2),
        "processing_time_hms": format_hms(proc),
        "time_saved_sec": round(time_saved, 2),
        "time_saved_hms": format_hms(time_saved),
        "time_reduction_pct": round(reduction_pct, 2),
        "speedup_factor": round(speedup, 2),
        "improvement_level_ar": level_ar,
        "improvement_level_en": level_en,
    }


def _resolve_species_label(ev: TrackManifestItem) -> str:
    if ev.qwen_verified and ev.qwen_species:
        return norm_species(ev.qwen_species)
    return norm_species(ev.species_yolo)


def build_video_summary(
    manifest: Sequence[TrackManifestItem],
    stage1_stats: Sequence[Dict[str, Any]],
    run_mode: str,
    stage2_stats: Optional[Dict[str, Any]] = None,
    stage3_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a complete structured summary covering species counts, confidence, timestamps, frames, and speedup."""
    per_video_perf: List[Dict[str, Any]] = []
    total_video_dur_sec = 0.0
    total_proc_sec = 0.0

    for st in stage1_stats:
        dur_sec = float(st.get("duration_sec", float(st.get("duration_min", 0.0)) * 60.0))
        proc_sec = float(st.get("scan_time_sec", 0.0))
        total_video_dur_sec += dur_sec
        total_proc_sec += proc_sec
        perf = compute_processing_performance(dur_sec, proc_sec)
        per_video_perf.append(
            {
                "video_tag": st.get("video_tag", ""),
                "video": st.get("video", ""),
                "status": st.get("status", "ok"),
                "fps": st.get("fps", 0.0),
                "total_frames": st.get("total_frames", 0),
                "total_tracks": st.get("total_tracks", 0),
                **perf,
            }
        )

    overall_perf = compute_processing_performance(total_video_dur_sec, total_proc_sec)

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    timeline: List[Dict[str, Any]] = []
    confirmed_confs: List[float] = []
    rejected_tracks = 0

    for ev in sorted(manifest, key=lambda item: (item.video_tag, item.start_frame, item.track_id)):
        species = _resolve_species_label(ev)
        if species == "none":
            rejected_tracks += 1
            continue

        species_ar = SPECIES_DISPLAY_AR.get(species, species)
        appearance_entry = {
            "video_tag": ev.video_tag,
            "video": ev.video_source,
            "track_id": ev.track_id,
            "species": species,
            "species_ar": species_ar,
            "start_hms": ev.start_hms,
            "end_hms": ev.end_hms,
            "best_hms": ev.best_hms,
            "start_frame": ev.start_frame,
            "end_frame": ev.end_frame,
            "best_frame": ev.best_frame,
            "avg_conf": ev.avg_conf,
            "max_conf": ev.max_conf,
            "min_conf": ev.min_conf,
            "class_purity": ev.class_purity,
            "qwen_verified": ev.qwen_verified,
            "needs_review": ev.needs_rl_review,
            "is_trusted_for_rag": ev.is_trusted_for_rag,
        }
        grouped[species].append(appearance_entry)
        timeline.append(appearance_entry)
        confirmed_confs.append(float(ev.avg_conf))

    species_breakdown: Dict[str, Dict[str, Any]] = {}
    for sp, items in sorted(grouped.items()):
        confs = [float(x["avg_conf"]) for x in items]
        species_breakdown[sp] = {
            "display_name_ar": SPECIES_DISPLAY_AR.get(sp, sp),
            "appearances_count": len(items),
            "avg_conf": round(float(np.mean(confs)), 4),
            "min_conf": round(float(np.min(confs)), 4),
            "max_conf": round(float(np.max(confs)), 4),
            "appearances": items,
        }

    summary: Dict[str, Any] = {
        "run_mode": run_mode,
        "performance": overall_perf,
        "per_video_performance": per_video_perf,
        "totals": {
            "videos_processed": len(stage1_stats),
            "total_confirmed_appearances": len(timeline),
            "rejected_false_positives": rejected_tracks,
            "overall_avg_conf": round(float(np.mean(confirmed_confs)), 4) if confirmed_confs else 0.0,
            "overall_min_conf": round(float(np.min(confirmed_confs)), 4) if confirmed_confs else 0.0,
            "overall_max_conf": round(float(np.max(confirmed_confs)), 4) if confirmed_confs else 0.0,
        },
        "species_breakdown": species_breakdown,
        "appearances_timeline": timeline,
        "pipeline_stages": {
            "qwen_verify": stage2_stats or {"skipped_stage": True},
            "qwen_rag": stage3_stats or {"skipped_stage": True},
        },
    }
    summary["summary_markdown"] = render_video_summary_markdown(summary)
    return summary


def render_video_summary_markdown(summary: Dict[str, Any]) -> str:
    """Render a clear Arabic Markdown summary report from a build_video_summary() dict."""
    perf = summary["performance"]
    totals = summary["totals"]
    breakdown = summary["species_breakdown"]
    timeline = summary["appearances_timeline"]

    lines: List[str] = [
        f"# ملخص تحليل المقاطع البرية (`{summary['run_mode']}`)",
        "",
        "## أداء المعالجة ومستوى التحسين",
        f"- **مدة المقاطع (الوقت الافتراضي):** `{perf['video_duration_hms']}` ({perf['video_duration_min']:.2f} دقيقة / {perf['video_duration_sec']:.1f} ثانية)",
        f"- **وقت الإنجاز الفعلي:** `{perf['processing_time_hms']}` ({perf['processing_time_sec']:.2f} ثانية)",
        f"- **الوقت المُوفَّر:** `{perf['time_saved_hms']}` (نسبة توفير **{perf['time_reduction_pct']:.1f}%**)",
        f"- **معدل التسريع:** **{perf['speedup_factor']:.2f}x** أسرع من الوقت الحقيقي",
        f"- **مستوى التحسين:** **{perf['improvement_level_ar']}**",
        "",
    ]

    if len(summary.get("per_video_performance", [])) > 1:
        lines.append("### تفاصيل الأداء لكل مقطع")
        lines.append("| المقطع | المدة الافتراضية | وقت الإنجاز | التسريع | التوفير | مستوى التحسين |")
        lines.append("|---|---|---|---|---|---|")
        for vp in summary["per_video_performance"]:
            lines.append(
                f"| {vp['video']} | {vp['video_duration_hms']} | {vp['processing_time_hms']} | "
                f"{vp['speedup_factor']:.2f}x | {vp['time_reduction_pct']:.1f}% | {vp['improvement_level_ar']} |"
            )
        lines.append("")

    lines.extend(
        [
            "## إحصائيات الظهور حسب النوع",
            f"- **المجموع الكلي لمرات الظهور:** **{totals['total_confirmed_appearances']}**",
            f"- **متوسط الثقة العام:** **{totals['overall_avg_conf'] * 100:.1f}%** "
            f"(أدنى: {totals['overall_min_conf'] * 100:.1f}% | أعلى: {totals['overall_max_conf'] * 100:.1f}%)",
        ]
    )
    if totals["rejected_false_positives"] > 0:
        lines.append(f"- **الإنذارات الكاذبة المستبعدة عبر Qwen:** {totals['rejected_false_positives']}")

    if breakdown:
        for sp, info in breakdown.items():
            lines.append(
                f"- **{info['display_name_ar']} (`{sp}`):** **{info['appearances_count']}** ظهور | "
                f"متوسط الثقة: **{info['avg_conf'] * 100:.1f}%** "
                f"(أدنى: {info['min_conf'] * 100:.1f}%، أعلى: {info['max_conf'] * 100:.1f}%)"
            )
    else:
        lines.append("- لم يتم رصد أي حيوانات مستهدفة في المقاطع.")

    lines.append("")
    lines.append("## أوقات الظهور وأرقام الفريمات")
    if timeline:
        lines.append("| المقطع | الحيوان | Track | وقت الظهور | ذروة الوضوح | الفريمات | أفضل فريم | متوسط الثقة |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for item in timeline:
            lines.append(
                f"| {item['video']} | {item['species_ar']} | #{item['track_id']} | "
                f"{item['start_hms']} -> {item['end_hms']} | {item['best_hms']} | "
                f"{item['start_frame']} -> {item['end_frame']} | {item['best_frame']} | "
                f"{item['avg_conf'] * 100:.1f}% |"
            )
    else:
        lines.append("_لا توجد فترات ظهور مسجّلة._")

    return "\n".join(lines)
