"""导出真实视频中的跟踪连续性、膝角和坐站状态机逐帧诊断。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import load_config  # noqa: E402
from core.detector import YoloPoseDetector  # noqa: E402
from core.pipeline import VisionPipeline, new_stream_state  # noqa: E402


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description="诊断真实Sit-to-Stand逐帧状态")
    parser.add_argument(
        "--clips", type=Path,
        default=PROJECT_ROOT / "test_videos" / "real_pre_fall_clips.csv",
    )
    parser.add_argument(
        "--annotations", type=Path,
        default=PROJECT_ROOT / "test_videos" / "real_pre_fall_event_annotations.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "data" / "exports" / "sit_to_stand_frame_diagnostics.csv",
    )
    args = parser.parse_args()
    clips = _read_csv(args.clips)
    annotations = _read_csv(args.annotations)
    truth_by_clip = {
        row["clip_id"]: (float(row["start_s"]), float(row["end_s"]))
        for row in annotations if row["event_type"] == "SIT_TO_STAND"
    }
    config = load_config()
    pipeline = VisionPipeline(config, YoloPoseDetector(config))
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for number, clip in enumerate(clips, start=1):
        clip_id = clip["clip_id"]
        print(f"[{number}/{len(clips)}] {clip_id}", flush=True)
        capture = cv2.VideoCapture(str(PROJECT_ROOT / clip["relative_path"]))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
        state = new_stream_state()
        state["profile_persistence_enabled"] = False
        frame_index = 0
        track_counts: Counter[int] = Counter()
        events = 0
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            timestamp_s = frame_index / fps
            result, state = pipeline.process_frame(frame, timestamp_s, state, confidence=0.30)
            active_ids = {decision.track_id for decision in result.decisions}
            for track_id in active_ids:
                track_counts[track_id] += 1
                track = state.get("tracks", {}).get(str(track_id), {})
                pre_fall = track.get("pre_fall", {})
                history = pre_fall.get("history", [])
                latest = history[-1] if history else {}
                transition = pre_fall.get("sit_to_stand_transition") or {}
                last_pose = track.get("last_pose", {})
                bbox = last_pose.get("bbox", [0.0, 0.0, 0.0, 0.0])
                frame_area = max(1.0, float(frame.shape[0] * frame.shape[1]))
                bbox_area_ratio = (
                    max(0.0, float(bbox[2]) - float(bbox[0]))
                    * max(0.0, float(bbox[3]) - float(bbox[1]))
                    / frame_area
                )
                last_result = pre_fall.get("last_result")
                rows.append({
                    "clip_id": clip_id,
                    "frame": frame_index,
                    "timestamp_s": round(timestamp_s, 3),
                    "track_id": track_id,
                    "bbox_area_ratio": round(bbox_area_ratio, 5),
                    "rgb_quality": getattr(last_result, "rgb_quality", None),
                    "knee_angle_deg": pre_fall.get("previous_knee_angle_deg"),
                    "torso_angle_deg": latest.get("angle"),
                    "center_y": latest.get("center_y"),
                    "phase": pre_fall.get("sit_to_stand_phase"),
                    "seated_since": pre_fall.get("sit_to_stand_seated_since"),
                    "transition_started_at": transition.get("started_at"),
                    "truth_sts": int(
                        clip_id in truth_by_clip
                        and truth_by_clip[clip_id][0] <= timestamp_s <= truth_by_clip[clip_id][1]
                    ),
                })
            events += len(result.sit_to_stand_events)
            frame_index += 1
        capture.release()
        summaries.append({
            "clip_id": clip_id,
            "truth_sts": clip_id in truth_by_clip,
            "truth_interval": truth_by_clip.get(clip_id),
            "track_frame_counts": dict(track_counts),
            "predicted_events": events,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
