"""运行真实视频逐事件验收并生成近跌倒/起身指标。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import load_config  # noqa: E402
from core.detector import YoloPoseDetector  # noqa: E402
from core.evaluation import temporal_event_metrics  # noqa: E402
from core.pipeline import VisionPipeline, new_stream_state  # noqa: E402


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _scan_clip(
    path: Path,
    pipeline: VisionPipeline,
    confidence: float,
    stride: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"无法打开视频：{path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    state = new_stream_state()
    state["profile_persistence_enabled"] = False
    predictions: list[dict[str, Any]] = []
    index = 0
    inferred_frames = 0
    max_pre_fall_risk = 0.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if index % stride == 0:
                timestamp_s = index / fps
                result, state = pipeline.process_frame(
                    frame, timestamp_s, state, confidence=confidence
                )
                inferred_frames += 1
                if result.pre_fall_results:
                    max_pre_fall_risk = max(
                        max_pre_fall_risk,
                        max(item.pre_fall_risk_score for item in result.pre_fall_results),
                    )
                for event in result.near_fall_events:
                    predictions.append({
                        "event_type": "NEAR_FALL",
                        "candidate_start_s": event.get("start_time_s"),
                        "detection_time_s": event.get("timestamp_s", timestamp_s),
                        "track_id": event.get("track_id"),
                        "trigger_kind": event.get("trigger_kind"),
                        "start_knee_angle_deg": event.get("start_knee_angle_deg"),
                        "min_knee_angle_deg": event.get("min_knee_angle_deg"),
                        "max_center_speed": event.get("max_center_speed"),
                    })
                for event in result.sit_to_stand_attempts:
                    predictions.append({
                        "event_type": "SIT_TO_STAND",
                        "candidate_start_s": event.get("start_time_s"),
                        "detection_time_s": event.get("timestamp_s", timestamp_s),
                        "track_id": event.get("track_id"),
                        "succeeded": None,
                        "duration_s": None,
                        "knee_extension_deg": event.get("knee_extension_deg"),
                        "center_rise_norm": event.get("center_rise_norm"),
                        "torso_change_deg": event.get("torso_change_deg"),
                        "box_area_ratio": event.get("box_area_ratio"),
                    })
                for event in result.new_events:
                    predictions.append({
                        "event_type": "FALL",
                        "candidate_start_s": event.get("source_time_s", timestamp_s),
                        "detection_time_s": event.get("source_time_s", timestamp_s),
                        "track_id": event.get("track_id"),
                    })
            index += 1
    finally:
        capture.release()
    return predictions, {
        "fps": round(fps, 3),
        "frames": index,
        "declared_frames": frame_count,
        "inferred_frames": inferred_frames,
        "duration_s": round(index / fps, 3),
        "max_pre_fall_risk": round(max_pre_fall_risk, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="真实近跌倒/起身逐事件验收")
    parser.add_argument(
        "--clips",
        type=Path,
        default=PROJECT_ROOT / "test_videos" / "real_pre_fall_clips.csv",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=PROJECT_ROOT / "test_videos" / "real_pre_fall_event_annotations.csv",
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "data" / "exports"
    )
    parser.add_argument(
        "--report-prefix",
        default="real_pre_fall_event",
        help="输出文件名前缀；例如 specialized_scene_event",
    )
    args = parser.parse_args()

    clips = _read_csv(args.clips)
    annotations = _read_csv(args.annotations)
    if args.limit > 0:
        clips = clips[: args.limit]
    selected = {str(row["clip_id"]) for row in clips}
    annotations = [row for row in annotations if str(row["clip_id"]) in selected]
    if not clips:
        raise SystemExit("真实视频清单为空。")

    config = load_config()
    pipeline = VisionPipeline(config, YoloPoseDetector(config))
    prediction_rows: list[dict[str, Any]] = []
    clip_results: list[dict[str, Any]] = []
    for number, clip in enumerate(clips, start=1):
        clip_id = str(clip["clip_id"])
        path = PROJECT_ROOT / str(clip["relative_path"])
        print(f"[{number}/{len(clips)}] {clip_id}", flush=True)
        predictions, diagnostics = _scan_clip(
            path, pipeline, args.confidence, max(1, int(args.stride))
        )
        for event_number, prediction in enumerate(predictions, start=1):
            prediction_rows.append({
                "prediction_id": f"{clip_id}-pred-{event_number:03d}",
                "clip_id": clip_id,
                **prediction,
            })
        clip_results.append({
            "clip_id": clip_id,
            "dataset": clip.get("dataset", ""),
            "relative_path": clip.get("relative_path", ""),
            "annotation_status": clip.get("annotation_status", ""),
            "prediction_count": len(predictions),
            **diagnostics,
        })

    clip_ids = [str(row["clip_id"]) for row in clips]
    near_fall = temporal_event_metrics(
        annotations,
        prediction_rows,
        event_type="NEAR_FALL",
        clip_ids=clip_ids,
    )
    sit_to_stand = temporal_event_metrics(
        annotations,
        prediction_rows,
        event_type="SIT_TO_STAND",
        clip_ids=clip_ids,
    )
    fall = temporal_event_metrics(
        annotations,
        prediction_rows,
        event_type="FALL",
        clip_ids=clip_ids,
    )
    report = {
        "evaluation_unit": "one-to-one temporal event matching",
        "matching_tolerance_s": {"before": 0.5, "after": 0.5},
        "dataset": {
            "clips": len(clips),
            "annotated_events": len(annotations),
            "predicted_events": len(prediction_rows),
            "inference_stride": max(1, int(args.stride)),
        },
        "near_fall": near_fall,
        "sit_to_stand": sit_to_stand,
        "fall": fall,
        "clips": clip_results,
        "warnings": [],
    }
    if near_fall["status"] != "OK":
        report["warnings"].append(
            "当前本地验收集尚无可用于视觉推理的真实近跌倒正视频；近跌倒 Precision、Recall、F1、预警延迟保持为空，混淆矩阵仅反映负样本误报。"
        )
    if fall["status"] != "OK":
        report["warnings"].append(
            "当前验收清单没有跌倒正事件，跌倒 Precision、Recall、F1 和告警延迟保持为空。"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe_prefix = "".join(
        character for character in str(args.report_prefix)
        if character.isalnum() or character in {"_", "-"}
    ).strip("_-")
    if not safe_prefix:
        raise SystemExit("输出文件名前缀不能为空。")
    prediction_path = args.output_dir / f"{safe_prefix}_predictions.csv"
    report_path = args.output_dir / f"{safe_prefix}_report.json"
    fields = [
        "prediction_id", "clip_id", "event_type", "candidate_start_s",
        "detection_time_s", "track_id", "succeeded", "duration_s",
        "knee_extension_deg", "center_rise_norm", "torso_change_deg",
        "box_area_ratio", "trigger_kind", "start_knee_angle_deg",
        "min_knee_angle_deg", "max_center_speed",
    ]
    with prediction_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(prediction_rows)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"逐事件预测：{prediction_path}")
    print(f"验收报告：{report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
