"""按固定时间间隔导出视频关键帧拼图，供逐事件人工标注复核。"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="导出视频时间轴关键帧拼图")
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=0.0)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--width", type=int, default=400)
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"无法打开视频：{args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = frame_count / fps
    interval_s = max(0.05, float(args.interval))
    start_s = max(0.0, float(args.start))
    end_s = min(duration_s, float(args.end)) if args.end > 0 else duration_s
    timestamps = np.arange(start_s, max(start_s, end_s), interval_s).tolist()
    frames: list[np.ndarray] = []
    for timestamp_s in timestamps:
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_s * 1000.0)
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        height, width = frame.shape[:2]
        target_width = max(160, int(args.width))
        target_height = max(90, round(height * target_width / max(width, 1)))
        frame = cv2.resize(frame, (target_width, target_height))
        cv2.putText(
            frame,
            f"{timestamp_s:.2f}s",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        frames.append(frame)
    capture.release()
    if not frames:
        raise SystemExit("没有可导出的帧。")

    columns = max(1, int(args.columns))
    rows = math.ceil(len(frames) / columns)
    blank = np.zeros_like(frames[0])
    frames.extend(blank.copy() for _ in range(rows * columns - len(frames)))
    sheet = np.vstack(
        [np.hstack(frames[index:index + columns]) for index in range(0, len(frames), columns)]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), sheet):
        raise SystemExit(f"无法写入：{args.output}")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
