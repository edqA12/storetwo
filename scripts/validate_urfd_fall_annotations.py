"""核验URFD官方逐帧标签、本地视频帧序和派生跌倒时间边界。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABEL_PATH = PROJECT_ROOT / "test_videos" / "external" / "urfd" / "urfall-cam0-falls.csv"
ANNOTATION_PATH = PROJECT_ROOT / "test_videos" / "specialized_event_annotations.csv"
FALL_ANNOTATION_PATH = PROJECT_ROOT / "test_videos" / "fall_event_annotations.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "exports" / "urfd_fall_annotation_validation.json"


def _annotations(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            row["clip_id"]: row
            for row in csv.DictReader(handle)
            if row["event_type"] == "FALL"
        }


def main() -> int:
    labels: dict[str, list[tuple[int, int]]] = {}
    with LABEL_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            labels.setdefault(row[0], []).append((int(row[1]), int(row[2])))
    annotations = _annotations(ANNOTATION_PATH)
    fall_annotations = _annotations(FALL_ANNOTATION_PATH)
    results: list[dict[str, object]] = []
    for sequence_number in range(1, 7):
        sequence = f"fall-{sequence_number:02d}"
        clip_id = f"urfd-fall-{sequence_number:02d}"
        video_path = (
            PROJECT_ROOT
            / "test_videos"
            / "00_ready_to_upload"
            / "01_fall_positive"
            / f"URFD_FALL_{sequence_number:02d}_RGB_POS.mp4"
        )
        capture = cv2.VideoCapture(str(video_path))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 0.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        sequence_labels = labels[sequence]
        falling_frames = [frame for frame, label in sequence_labels if label == 0]
        lying_frames = [frame for frame, label in sequence_labels if label == 1]
        first_falling = min(falling_frames)
        first_lying = min(lying_frames)
        derived_start = (first_falling - 1) / fps
        derived_end = (first_lying - 1) / fps
        annotation = annotations[clip_id]
        fall_annotation = fall_annotations[clip_id]
        checks = {
            "fps_is_30": abs(fps - 30.0) <= 1e-6,
            "frame_count_matches_official": frame_count == len(sequence_labels),
            "falling_labels_are_contiguous": falling_frames
            == list(range(first_falling, first_lying)),
            "annotated_start_matches": abs(float(annotation["start_s"]) - derived_start) <= 0.0005,
            "annotated_end_matches": abs(float(annotation["end_s"]) - derived_end) <= 0.0005,
            "fall_table_start_matches": abs(float(fall_annotation["start_s"]) - derived_start) <= 0.0005,
            "fall_table_end_matches": abs(float(fall_annotation["end_s"]) - derived_end) <= 0.0005,
            "fall_table_frames_match": (
                int(fall_annotation["start_frame"]) == first_falling
                and int(fall_annotation["end_frame"]) == first_lying
            ),
        }
        results.append({
            "clip_id": clip_id,
            "fps": fps,
            "video_frames": frame_count,
            "official_label_rows": len(sequence_labels),
            "first_falling_frame": first_falling,
            "first_lying_frame": first_lying,
            "fall_transition_frames": len(falling_frames),
            "derived_start_s": round(derived_start, 3),
            "derived_end_s": round(derived_end, 3),
            "checks": checks,
            "passed": all(checks.values()),
        })
    report = {
        "method": "URFD official frame label 0 start to first label 1; t=(frame-1)/fps",
        "sequences": len(results),
        "passed": sum(bool(item["passed"]) for item in results),
        "all_passed": all(bool(item["passed"]) for item in results),
        "results": results,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUTPUT_PATH)
    print(f"URFD跌倒标注核验：{report['passed']}/{report['sequences']} 通过")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
