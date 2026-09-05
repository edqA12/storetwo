from __future__ import annotations

import copy
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import AppConfig
from .detector import YoloPoseDetector
from .features import extract_pose_features
from .state_machine import FALLEN, NORMAL, REASON_LABELS, advance_state_machine
from .tracker import assign_track_ids
from .types import Decision, FrameResult, PersonPose


COCO_EDGES = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
)

STATE_COLORS = {
    "NORMAL": (70, 180, 80),
    "DESCENDING": (0, 190, 255),
    "SUSPECT": (0, 145, 255),
    "LYING": (0, 190, 255),
    "FALLEN": (35, 35, 220),
    "UNKNOWN": (150, 150, 150),
}


def new_stream_state() -> dict[str, Any]:
    return {
        "version": 1,
        "session_id": __import__("uuid").uuid4().hex,
        "next_track_id": 1,
        "tracks": {},
        "last_timestamp_s": None,
        "frame_count": 0,
        "last_db_event_id": None,
        "last_alert_text": "暂无告警",
    }


@lru_cache(maxsize=8)
def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _draw_chinese_panel(frame_bgr: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image, "RGBA")
    font = _font(max(18, min(frame_bgr.shape[1] // 35, 28)))
    line_height = int(getattr(font, "size", 22) * 1.55)
    panel_height = max(52, 18 + line_height * len(lines))
    draw.rounded_rectangle((12, 12, min(image.width - 12, 560), panel_height), radius=10, fill=(7, 22, 42, 205))
    y = 22
    for text, bgr in lines:
        draw.text((26, y), text, font=font, fill=(bgr[2], bgr[1], bgr[0], 255))
        y += line_height
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


class VisionPipeline:
    def __init__(self, config: AppConfig, detector: YoloPoseDetector | None = None) -> None:
        self.config = config
        self.detector = detector or YoloPoseDetector(config)

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp_s: float,
        stream_state: dict[str, Any] | None,
        confidence: float | None = None,
    ) -> tuple[FrameResult, dict[str, Any]]:
        state = copy.deepcopy(stream_state) if stream_state else new_stream_state()
        started = time.perf_counter()
        last_timestamp = state.get("last_timestamp_s")
        if last_timestamp is not None and timestamp_s <= float(last_timestamp):
            annotated = _draw_chinese_panel(frame_bgr.copy(), [("已丢弃乱序帧", (0, 190, 255))])
            return FrameResult(annotated, [], 0.0, "等待新画面", [], {"stale_frame": True}), state

        poses = self.detector.detect(frame_bgr, confidence=confidence)
        poses = assign_track_ids(
            poses,
            state,
            timestamp_s,
            self.config.detection.tracker_iou_threshold,
            self.config.detection.tracker_center_threshold,
            self.config.detection.missing_tolerance_s,
        )

        decisions: list[Decision] = []
        new_events: list[dict[str, Any]] = []
        for pose in poses:
            track_state = state["tracks"][str(pose.track_id)]
            features = extract_pose_features(
                pose,
                frame_bgr.shape,
                track_state,
                timestamp_s,
                self.config.model.keypoint_confidence,
                self.config.detection.history_seconds,
                self.config.detection.stale_derivative_s,
            )
            decision = advance_state_machine(
                pose.track_id,
                features,
                track_state.setdefault("machine", {}),
                timestamp_s,
                self.config.detection,
            )
            decisions.append(decision)
            track_state["last_pose"] = {
                "bbox": list(pose.bbox_xyxy),
                "keypoints": pose.keypoints_xy.tolist(),
                "confidence": pose.keypoint_confidence.tolist(),
                "box_confidence": pose.box_confidence,
            }
            track_state["last_decision"] = {
                "state": decision.state,
                "label": decision.label,
                "risk": decision.risk,
            }
            if decision.event_started:
                new_events.append({
                    "event_key": decision.event_key,
                    "event_type": "跌倒告警" if decision.state == FALLEN else "疑似跌倒",
                    "track_id": decision.track_id,
                    "source_time_s": round(float(timestamp_s), 3),
                    "risk": decision.risk,
                    "confidence": round(float(features.quality), 3),
                    "reasons": [REASON_LABELS.get(code, code) for code in decision.reason_codes],
                })

        annotated = self._draw(frame_bgr.copy(), poses, decisions)
        if decisions:
            highest = max(decisions, key=lambda item: item.risk)
            max_risk = highest.risk
            overall_label = highest.label
            panel_color = STATE_COLORS.get(highest.state, (255, 255, 255))
            panel_lines = [
                (f"状态：{overall_label}", panel_color),
                (f"风险值：{max_risk:.1f} / 100　人数：{len(poses)}", (245, 245, 245)),
            ]
        else:
            max_risk = 0.0
            overall_label = "未检测到完整人体"
            panel_lines = [(overall_label, (0, 190, 255)), ("请保持全身进入画面", (235, 235, 235))]
        annotated = _draw_chinese_panel(annotated, panel_lines)

        state["last_timestamp_s"] = float(timestamp_s)
        state["frame_count"] = int(state.get("frame_count", 0)) + 1
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        diagnostics = {
            "persons": len(poses),
            "inference_ms": round(float(self.detector.last_inference_ms), 1),
            "total_ms": round(elapsed_ms, 1),
            "processing_fps": round(1000.0 / elapsed_ms, 2) if elapsed_ms > 0 else 0.0,
        }
        result = FrameResult(
            annotated_bgr=annotated,
            decisions=decisions,
            max_risk=max_risk,
            overall_label=overall_label,
            new_events=new_events,
            diagnostics=diagnostics,
        )
        return result, state

    def render_cached(self, frame_bgr: np.ndarray, stream_state: dict[str, Any]) -> np.ndarray:
        poses: list[PersonPose] = []
        decisions: list[Decision] = []
        for key, track in stream_state.get("tracks", {}).items():
            raw_pose = track.get("last_pose")
            raw_decision = track.get("last_decision")
            if not raw_pose or not raw_decision:
                continue
            poses.append(PersonPose(
                track_id=int(key),
                bbox_xyxy=tuple(raw_pose["bbox"]),
                box_confidence=float(raw_pose["box_confidence"]),
                keypoints_xy=np.asarray(raw_pose["keypoints"], dtype=np.float32),
                keypoint_confidence=np.asarray(raw_pose["confidence"], dtype=np.float32),
            ))
            decisions.append(Decision(
                track_id=int(key),
                state=str(raw_decision["state"]),
                label=str(raw_decision["label"]),
                risk=float(raw_decision["risk"]),
            ))
        return self._draw(frame_bgr.copy(), poses, decisions)

    def _draw(self, frame: np.ndarray, poses: list[PersonPose], decisions: list[Decision]) -> np.ndarray:
        decisions_by_id = {item.track_id: item for item in decisions}
        threshold = self.config.model.keypoint_confidence
        for pose in poses:
            decision = decisions_by_id.get(pose.track_id)
            color = STATE_COLORS.get(decision.state if decision else NORMAL, (255, 255, 255))
            x1, y1, x2, y2 = (int(value) for value in pose.bbox_xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            for first, second in COCO_EDGES:
                if first >= len(pose.keypoints_xy) or second >= len(pose.keypoints_xy):
                    continue
                if pose.keypoint_confidence[first] < threshold or pose.keypoint_confidence[second] < threshold:
                    continue
                p1 = tuple(int(value) for value in pose.keypoints_xy[first][:2])
                p2 = tuple(int(value) for value in pose.keypoints_xy[second][:2])
                cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)
            for index, point in enumerate(pose.keypoints_xy):
                if index < len(pose.keypoint_confidence) and pose.keypoint_confidence[index] >= threshold:
                    cv2.circle(frame, tuple(int(value) for value in point[:2]), 4, (255, 255, 255), -1, cv2.LINE_AA)
            label = f"ID {pose.track_id}  RISK {decision.risk:.0f}" if decision else f"ID {pose.track_id}"
            cv2.rectangle(frame, (x1, max(0, y1 - 28)), (min(frame.shape[1], x1 + 190), y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        return frame
