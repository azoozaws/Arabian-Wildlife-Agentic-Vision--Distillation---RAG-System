"""Runnable CLI entrypoint for the LangGraph Wildlife Agent.

Note: Core LangGraph definitions live in `app/src/agent/graph.py` (and are lazily exported
from `app.src.agent`) so the `app/src/agent/` package directory does not shadow this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from app.src.agent.graph import (
    WildlifeAgentRunner,
    build_wildlife_graph,
    get_graph_mermaid,
    hitl_review_node,
    route_after_eval_gate,
)

__all__ = [
    "WildlifeAgentRunner",
    "build_wildlife_graph",
    "get_graph_mermaid",
    "hitl_review_node",
    "route_after_eval_gate",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Arabian Wildlife LangGraph Agent CLI")
    parser.add_argument("--mermaid", action="store_true", help="Print the LangGraph Mermaid workflow diagram and exit")
    parser.add_argument("--video_paths", nargs="+", default=[], help="Input video file path(s)")
    parser.add_argument(
        "--run_mode",
        type=str,
        choices=[
            "display_only",
            "summary_only",
            "summary_with_rag",
            "summary_with_qwen",
            "train_only",
            "display_and_train",
        ],
        default="summary_only",
    )
    parser.add_argument("--out_workspace_dir", type=str, default="./out_wildlife")
    parser.add_argument("--champion_weights", type=str, default="")
    parser.add_argument("--lora_dir", type=str, default=None)
    parser.add_argument("--chromadb_dir", type=str, default="")
    parser.add_argument("--orig_dataset_dir", type=str, default="./YOLO_Dataset_Stratified")
    parser.add_argument("--merged_dataset_dir", type=str, default="./dataset_local/YOLO11_Dataset_v4")
    parser.add_argument("--train_project_dir", type=str, default="./runs/yolo_v4_runs")
    parser.add_argument("--held_out_video", type=str, default="")
    parser.add_argument("--version_tag", type=str, default="v4")
    parser.add_argument(
        "--clean_demo_display",
        action="store_true",
        help="Use clean emerald-only demo display overlay instead of standard full HUD",
    )
    args = parser.parse_args()

    if args.mermaid:
        print(get_graph_mermaid())
        return

    if not args.video_paths:
        raise SystemExit("Provide --video_paths <file1> ... or use --mermaid to inspect the graph.")

    runner = WildlifeAgentRunner()
    result = runner.run(
        {
            "run_mode": args.run_mode,  # type: ignore[typeddict-item]
            "video_paths": args.video_paths,
            "out_workspace_dir": args.out_workspace_dir,
            "champion_weights_path": args.champion_weights,
            "qwen_lora_dir": args.lora_dir,
            "chromadb_dir": args.chromadb_dir,
            "orig_dataset_dir": args.orig_dataset_dir,
            "merged_dataset_dir": args.merged_dataset_dir,
            "train_project_dir": args.train_project_dir,
            "held_out_video_path": args.held_out_video,
            "version_tag": args.version_tag,
            "display_clean_demo_mode": args.clean_demo_display,
        }
    )

    st = result["state"]
    if st.get("video_summary_markdown"):
        print("\n" + st["video_summary_markdown"])

    print("\n================ LANGGRAPH AGENT RESULT ================")
    print(
        json.dumps(
            {
                "thread_id": result["thread_id"],
                "engine": result["engine"],
                "awaiting_hitl": result["awaiting_hitl"],
                "gate_decision": st.get("gate_decision"),
                "gate_reason": st.get("gate_reason"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
