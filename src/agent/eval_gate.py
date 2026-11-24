from typing import Dict

from app.src.agent.agent_state import WildlifeAgentState




def eval_gate_node(state: WildlifeAgentState) -> Dict:
    """Hybrid Champion-Challenger Gate: Hard Guardrails + Weighted Points."""
    champ = state["champion_metrics"]
    chall = state["challenger_metrics"]

    # 1. Hard Guardrails (veto if any core metric drops > 0.02)
    MAX_DROP = 0.02
    for key in ("precision", "recall", "mAP50"):
        if chall[key] < champ[key] - MAX_DROP:
            return {
                "champion_points": 0,
                "challenger_points": 0,
                "gate_decision": "needs_hitl",
                "gate_reason": f"Guardrail breached: {key} dropped from {champ[key]:.4f} to {chall[key]:.4f}.",
            }

    # 2. Weighted Points Scorecard
    champ_pts, chall_pts = 0, 0
    weights = {"precision": 1, "recall": 1, "mAP50": 1, "fp_per_hour": 2}

    for key in ("precision", "recall", "mAP50"):
        if chall[key] > champ[key]:
            chall_pts += weights[key]
        elif champ[key] > chall[key]:
            champ_pts += weights[key]

    # For FP/hr, LOWER is better
    if chall["fp_per_hour"] < champ["fp_per_hour"]:
        chall_pts += weights["fp_per_hour"]
    elif champ["fp_per_hour"] < chall["fp_per_hour"]:
        champ_pts += weights["fp_per_hour"]

    if chall_pts > champ_pts:
        decision = "promote"
        reason = f"Challenger won {chall_pts} vs {champ_pts} pts with all guardrails passed."
    else:
        decision = "needs_hitl"
        reason = f"Challenger did not beat Champion ({chall_pts} vs {champ_pts} pts) — escalated to engineer."

    return {
        "champion_points": champ_pts,
        "challenger_points": chall_pts,
        "gate_decision": decision,
        "gate_reason": reason,
    }