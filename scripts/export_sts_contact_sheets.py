"""导出真实起身真值前后关键帧拼图，供阈值诊断。"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "exports" / "sit_to_stand_contact_sheets"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    clips = {row["clip_id"]: row for row in _rows(PROJECT_ROOT / "test_videos" / "real_pre_fall_clips.csv")}
    events = [
        row for row in _rows(PROJECT_ROOT / "test_videos" / "real_pre_fall_event_annotations.csv")
        if row["event_type"] == "SIT_TO_STAND"
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for event in events:
        clip_id = event["clip_id"]
        capture = cv2.VideoCapture(str(PROJECT_ROOT / clips[clip_id]["relative_path"]))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
        duration = float(capture.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
        start = float(event["start_s"])
        end = float(event["end_s"])
        times = [max(0.0, start - 0.5), start, (start + end) / 2.0, min(end, duration - 1.0 / fps)]
        frames: list[np.ndarray] = []
        for timestamp_s in times:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_s * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            frame = cv2.resize(frame, (480, 360))
            cv2.putText(
                frame, f"{timestamp_s:.2f}s", (14, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 255), 2, cv2.LINE_AA,
            )
            frames.append(frame)
        capture.release()
        if len(frames) == 4:
            sheet = np.vstack((np.hstack(frames[:2]), np.hstack(frames[2:])))
            cv2.imwrite(str(OUTPUT_DIR / f"{clip_id}.jpg"), sheet)
    print(OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
