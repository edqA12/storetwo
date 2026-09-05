"""导出单段真实视频的近跌倒逐帧特征与状态机诊断。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import load_config  # noqa: E402
from core.detector import YoloPoseDetector  # noqa: E402
from core.pipeline import VisionPipeline, new_stream_state  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="诊断真实近跌倒逐帧状态")
    parser.add_argument("video", type=Path)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "exports" / "near_fall_frame_diagnostics.csv",
    )
    args = parser.parse_args()
    video_path = args.video if args.video.is_absolute() else PROJECT_ROOT / args.video
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"无法打开视频：{video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
    config = load_config()
    pipeline = VisionPipeline(config, YoloPoseDetector(config))
    state = new_stream_state()
    state["profile_persistence_enabled"] = False
    rows: list[dict[str, object]] = []
    emitted_events: list[dict[str, object]] = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        timestamp_s = frame_index / fps
        frame_index += 1
        if timestamp_s < args.start:
            continue
        if args.end is not None and timestamp_s > args.end:
            break
        result, state = pipeline.process_frame(frame, timestamp_s, state, confidence=0.30)
        for decision in result.decisions:
            track = state.get("tracks", {}).get(str(decision.track_id), {})
            features = dict(track.get("last_features", {}))
            pre_fall = track.get("pre_fall", {})
            history = pre_fall.get("history", [])
            latest = history[-1] if history else {}
            candidate = pre_fall.get("near_fall_candidate") or {}
            row = {
                "frame": frame_index - 1,
                "timestamp_s": round(timestamp_s, 3),
                "track_id": decision.track_id,
                "state": decision.state,
                "stability_score": getattr(track.get("last_pre_fall"), "stability_score", None),
                "sway_score": None,
                "candidate_started_at": candidate.get("started_at"),
                "near_fall_emitted": bool(result.near_fall_events),
                "sit_to_stand_phase": pre_fall.get("sit_to_stand_phase"),
                **features,
            }
            if history:
                _, sway_score = pipeline.pre_fall_analyzer._stability_scores(history)
                row["sway_score"] = round(sway_score, 3)
            rows.append(row)
        emitted_events.extend(result.near_fall_events)
    capture.release()
    if not rows:
        raise SystemExit("指定区间没有检测到可用人体姿态。")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "video": str(video_path),
        "fps": round(fps, 3),
        "rows": len(rows),
        "near_fall_events": emitted_events,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
