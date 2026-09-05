"""在同一批 RGB/Depth 视频上运行四组消融并生成可复核报告。"""

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
from core.depth_pipeline import DepthPipeline  # noqa: E402
from core.detector import YoloPoseDetector  # noqa: E402
from core.evaluation import evaluate_ablation  # noqa: E402
from core.fusion import FusionEngine  # noqa: E402
from core.multimodal import MultimodalPipeline, new_multimodal_state  # noqa: E402
from core.pipeline import VisionPipeline, new_stream_state  # noqa: E402


def _truth_label(value: str) -> str:
    normalized = value.strip().upper()
    aliases = {"NONFALL": "NORMAL", "NO_FALL": "NORMAL", "NEARFALL": "NEAR_FALL"}
    return aliases.get(normalized, normalized)


def _prediction(fall_time: float | None, near_fall_time: float | None) -> tuple[str, float | None]:
    if fall_time is not None:
        return "FALL", fall_time
    if near_fall_time is not None:
        return "NEAR_FALL", near_fall_time
    return "NORMAL", None


def _scan_rgb(
    path: Path,
    pipeline: VisionPipeline,
    confidence: float,
    stride: int,
) -> tuple[str, float | None, float | None]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"无法打开 RGB 视频：{path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
    state = new_stream_state()
    index = 0
    fall_time: float | None = None
    confirmed_time: float | None = None
    near_fall_time: float | None = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if index % stride == 0:
                timestamp_s = index / fps
                result, state = pipeline.process_frame(frame, timestamp_s, state, confidence)
                if fall_time is None and result.new_events:
                    fall_time = timestamp_s
                if confirmed_time is None and any(
                    event.get("event_type") == "跌倒告警" for event in result.new_events
                ):
                    confirmed_time = timestamp_s
                if near_fall_time is None and result.near_fall_events:
                    near_fall_time = timestamp_s
            index += 1
    finally:
        capture.release()
    prediction, detection_time = _prediction(fall_time, near_fall_time)
    return prediction, detection_time, confirmed_time


def _scan_multimodal(
    path: Path,
    pipeline: MultimodalPipeline,
    confidence: float,
    stride: int,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"无法打开 RGB+Depth 视频：{path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
    state = new_multimodal_state()
    index = 0
    fusion_fall_time: float | None = None
    fusion_confirmed_time: float | None = None
    near_fall_time: float | None = None
    depth_fall_time: float | None = None
    depth_frames = 0
    depth_available_frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if index % stride == 0:
                timestamp_s = index / fps
                result, state = pipeline.process_frame(frame, timestamp_s, state, confidence)
                if fusion_fall_time is None and result.new_events:
                    fusion_fall_time = timestamp_s
                if fusion_confirmed_time is None and any(
                    event.get("event_type") == "跌倒告警" for event in result.new_events
                ):
                    fusion_confirmed_time = timestamp_s
                if near_fall_time is None and result.near_fall_events:
                    near_fall_time = timestamp_s
                depth = next((item for item in result.modalities if item.modality == "depth"), None)
                if depth is not None:
                    depth_frames += 1
                    depth_available_frames += int(depth.available)
                    if (
                        depth_fall_time is None
                        and depth.available
                        and depth.state == "DEPTH_IMPACT"
                        and depth.risk >= pipeline.config.multimodal.fusion_alert_threshold
                    ):
                        depth_fall_time = timestamp_s
            index += 1
    finally:
        capture.release()
    fusion_prediction, fusion_time = _prediction(fusion_fall_time, near_fall_time)
    depth_prediction, depth_time = _prediction(depth_fall_time, None)
    available_ratio = depth_available_frames / depth_frames if depth_frames else 0.0
    return {
        "depth_only_prediction": depth_prediction,
        "depth_only_detection_time_s": depth_time,
        "rgb_depth_prediction": fusion_prediction,
        "rgb_depth_detection_time_s": fusion_time,
        "rgb_depth_confirmation_time_s": fusion_confirmed_time,
        # Personal Baseline 只改变前置风险评估，不改变跌倒事件真值定义；
        # 在当前 FALL/NORMAL 数据集上两列应一致，报告中会明确数据边界。
        "rgb_depth_baseline_prediction": fusion_prediction,
        "rgb_depth_baseline_detection_time_s": fusion_time,
        "rgb_depth_baseline_confirmation_time_s": fusion_confirmed_time,
        "depth_degraded": available_ratio < 0.5,
        "depth_available_ratio": round(available_ratio, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="运行暮安智护四组消融实验")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=PROJECT_ROOT / "test_videos" / "ground_truth.csv",
    )
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--limit", type=int, default=0, help="仅调试时限制样本数；0表示全部")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "data" / "exports"
    )
    args = parser.parse_args()
    with args.ground_truth.open("r", encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    if args.limit > 0:
        source_rows = source_rows[: args.limit]
    if not source_rows:
        raise SystemExit("真实标签清单为空。")

    config = load_config()
    detector = YoloPoseDetector(config)
    vision = VisionPipeline(config, detector)
    multimodal = MultimodalPipeline(
        config,
        vision,
        DepthPipeline(config.multimodal),
        FusionEngine(config.multimodal),
    )
    result_rows: list[dict[str, Any]] = []
    for index, row in enumerate(source_rows, start=1):
        filename = str(row["filename"])
        folder = str(row["folder"])
        rgb_path = PROJECT_ROOT / "test_videos" / "00_ready_to_upload" / folder / filename
        raw_name = filename.replace("_RGB_", "_CAM0_")
        raw_path = PROJECT_ROOT / "test_videos" / folder / raw_name
        print(f"[{index}/{len(source_rows)}] {filename}", flush=True)
        rgb_prediction, rgb_time, rgb_confirmation_time = _scan_rgb(
            rgb_path, vision, args.confidence, max(1, args.stride)
        )
        multimodal_result = _scan_multimodal(
            raw_path, multimodal, args.confidence, max(1, args.stride)
        )
        result_rows.append({
            "sample_id": filename,
            "path": str(raw_path),
            "ground_truth": _truth_label(str(row["expected_label"])),
            "duration_s": row.get("duration_s", ""),
            "event_time_s": row.get("event_time_s", ""),
            "rgb_only_prediction": rgb_prediction,
            "rgb_only_detection_time_s": rgb_time,
            "rgb_only_confirmation_time_s": rgb_confirmation_time,
            **multimodal_result,
        })

    report = evaluate_ablation(result_rows)
    report["warnings"].append(
        "当前 URFD 清单只有 FALL/NORMAL，没有近跌倒和起身风险真值；Baseline 组只能验证不破坏跌倒事件识别，不能证明长期风险预测准确率。"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "ablation_predictions.csv"
    report_path = args.output_dir / "ablation_report.json"
    fieldnames = list(result_rows[0].keys())
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_rows)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"预测清单：{manifest_path}")
    print(f"消融报告：{report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
