from __future__ import annotations

import uuid
from typing import Any

import cv2
import numpy as np

from .config import AppConfig
from .depth_pipeline import DepthPipeline
from .fusion import FusionEngine
from .pre_fall import LEVEL_LABELS
from .pipeline import VisionPipeline, _draw_chinese_panel, new_stream_state
from .types import FrameResult, FusedDecision, ModalityResult, PreFallRiskResult


def new_multimodal_state() -> dict[str, Any]:
    return {
        "version": 2,
        "session_id": uuid.uuid4().hex,
        "vision": new_stream_state(),
        "depth": {},
        "fusion": {"last_event_ts": -1e12, "active_event_key": None},
        "last_fused": None,
    }


def split_rgb_depth_frame(
    frame_bgr: np.ndarray,
    config: AppConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] < 3:
        raise ValueError("多模态视频画面格式不正确。")
    width = int(frame_bgr.shape[1])
    split = int(round(width * float(config.multimodal.split_ratio)))
    minimum = int(config.multimodal.min_half_width)
    if split < minimum or width - split < minimum:
        raise ValueError(f"组合视频两侧画面宽度必须至少为{minimum}像素。")
    left, right = frame_bgr[:, :split].copy(), frame_bgr[:, split:].copy()
    if str(config.multimodal.depth_side).lower() == "right":
        return right, left
    return left, right


def _visual_modality(result: FrameResult, timestamp_s: float) -> ModalityResult:
    for modality in result.modalities:
        if modality.modality == "vision":
            return modality
    state = result.decisions[0].state if result.decisions else "UNKNOWN"
    return ModalityResult(
        modality="vision",
        timestamp_s=float(timestamp_s),
        risk=result.max_risk,
        quality=float(result.diagnostics.get("quality", 0.0)),
        state=state,
        label=result.overall_label,
        reasons=[],
        diagnostics=dict(result.diagnostics),
        available=bool(result.decisions),
    )


class MultimodalPipeline:
    def __init__(
        self,
        config: AppConfig,
        vision_pipeline: VisionPipeline,
        depth_pipeline: DepthPipeline,
        fusion_engine: FusionEngine,
    ) -> None:
        self.config = config
        self.vision_pipeline = vision_pipeline
        self.depth_pipeline = depth_pipeline
        self.fusion_engine = fusion_engine

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp_s: float,
        stream_state: dict[str, Any] | None,
        confidence: float | None = None,
    ) -> tuple[FrameResult, dict[str, Any]]:
        depth_frame, rgb_frame = split_rgb_depth_frame(frame_bgr, self.config)
        return self.process_rgb_depth(
            rgb_frame,
            depth_frame,
            timestamp_s,
            stream_state,
            confidence=confidence,
        )

    def process_rgb_depth(
        self,
        rgb_frame: np.ndarray,
        depth_frame: np.ndarray,
        timestamp_s: float,
        stream_state: dict[str, Any] | None,
        confidence: float | None = None,
    ) -> tuple[FrameResult, dict[str, Any]]:
        """Process synchronized RGB plus raw or visualized depth frames."""
        if rgb_frame is None or rgb_frame.ndim != 3 or rgb_frame.shape[2] < 3:
            raise ValueError("RGB画面格式不正确。")
        if depth_frame is None or depth_frame.ndim not in {2, 3}:
            raise ValueError("深度画面格式不正确。")
        state = dict(stream_state or new_multimodal_state())
        depth_result, depth_state, depth_rendered = self.depth_pipeline.process_frame(
            depth_frame,
            timestamp_s,
            state.get("depth"),
        )
        if isinstance(self.vision_pipeline, VisionPipeline):
            vision_result, vision_state = self.vision_pipeline.process_frame(
                rgb_frame,
                timestamp_s,
                state.get("vision"),
                confidence=confidence,
                depth_context={
                    "quality": depth_result.quality,
                    "available": depth_result.available,
                    "diagnostics": depth_result.diagnostics,
                },
            )
        else:
            vision_result, vision_state = self.vision_pipeline.process_frame(
                rgb_frame,
                timestamp_s,
                state.get("vision"),
                confidence=confidence,
            )
        visual_result = _visual_modality(vision_result, timestamp_s)
        fused = self.fusion_engine.fuse(visual_result, depth_result, timestamp_s)
        depth_available = bool(
            depth_result.available
            and depth_result.quality >= self.config.pre_fall.min_depth_quality
        )
        for pre_fall in vision_result.pre_fall_results:
            pre_fall.depth_quality = float(depth_result.quality)
            pre_fall.depth_degraded = not depth_available
        for near_fall in vision_result.near_fall_events:
            near_fall["depth_quality"] = round(float(depth_result.quality), 3)
            near_fall["depth_degraded"] = not depth_available
        for sit_to_stand in vision_result.sit_to_stand_events:
            sit_to_stand["depth_quality"] = round(float(depth_result.quality), 3)
            sit_to_stand["depth_degraded"] = not depth_available
        for attempt in vision_result.sit_to_stand_attempts:
            attempt["depth_quality"] = round(float(depth_result.quality), 3)
            attempt["depth_degraded"] = not depth_available
        state["vision"] = vision_state
        state["depth"] = depth_state
        state["last_visual"] = visual_result
        state["last_depth"] = depth_result
        state["last_fused"] = fused
        state["last_pre_fall"] = max(
            vision_result.pre_fall_results,
            key=lambda item: item.pre_fall_risk_score,
            default=None,
        )

        fusion_state = dict(state.get("fusion") or {})
        new_events: list[dict[str, Any]] = []
        alerting = fused.state in {"SUSPECT", "FALLEN"} and fused.risk >= self.config.multimodal.fusion_alert_threshold
        if alerting and not fusion_state.get("active_event_key"):
            if timestamp_s - float(fusion_state.get("last_event_ts", -1e12)) >= self.config.multimodal.event_cooldown_s:
                event_key = uuid.uuid4().hex
                fusion_state["active_event_key"] = event_key
                fusion_state["last_event_ts"] = float(timestamp_s)
                new_events.append({
                    "event_key": event_key,
                    "event_type": "跌倒告警" if fused.state == "FALLEN" else "疑似跌倒",
                    "track_id": 0,
                    "source_time_s": round(float(timestamp_s), 3),
                    "risk": fused.risk,
                    "fall_event_score": fused.fall_event_score,
                    "confidence": fused.quality,
                    "reasons": list(fused.explanation),
                    "fusion_method": fused.method,
                    "fusion_explanation": "；".join(fused.explanation),
                    "modalities": [visual_result.to_record(), depth_result.to_record()],
                })
        elif not alerting and fused.risk < 60.0:
            fusion_state["active_event_key"] = None
        state["fusion"] = fusion_state

        annotated_rgb = self._fusion_panel(
            vision_result.annotated_bgr,
            fused,
            visual_result,
            depth_result,
            state.get("last_pre_fall"),
        )
        annotated = self._compose(depth_rendered, annotated_rgb)
        diagnostics = dict(vision_result.diagnostics)
        diagnostics.update({
            "depth_quality": depth_result.quality,
            "depth_state": depth_result.state,
            "fusion_method": fused.method,
            "fusion_quality": fused.quality,
            "input_mode": "rgb_depth",
            "depth_degraded": not depth_available,
            "metric_depth_input": bool(
                depth_result.diagnostics.get("metric_depth_input", False)
            ),
        })
        return FrameResult(
            annotated_bgr=annotated,
            decisions=vision_result.decisions,
            max_risk=fused.risk,
            overall_label=fused.label,
            new_events=new_events,
            diagnostics=diagnostics,
            modalities=[visual_result, depth_result],
            fusion=fused,
            pre_fall_results=vision_result.pre_fall_results,
            near_fall_events=vision_result.near_fall_events,
            sit_to_stand_events=vision_result.sit_to_stand_events,
            sit_to_stand_attempts=vision_result.sit_to_stand_attempts,
        ), state

    def render_cached(self, frame_bgr: np.ndarray, stream_state: dict[str, Any]) -> np.ndarray:
        depth_frame, rgb_frame = split_rgb_depth_frame(frame_bgr, self.config)
        vision_state = stream_state.get("vision") or new_stream_state()
        depth_state = stream_state.get("depth") or {}
        rgb_rendered = self.vision_pipeline.render_cached(rgb_frame, vision_state)
        depth_rendered = self.depth_pipeline.render_cached(depth_frame, depth_state)
        fused = stream_state.get("last_fused")
        if isinstance(fused, FusedDecision):
            visual = stream_state.get("last_visual")
            if not isinstance(visual, ModalityResult):
                visual = ModalityResult("vision", fused.timestamp_s, 0.0, 0.0, "UNKNOWN", "视觉不可用", available=False)
            depth = stream_state.get("last_depth") or depth_state.get("last_result")
            if not isinstance(depth, ModalityResult):
                depth = ModalityResult("depth", fused.timestamp_s, 0.0, 0.0, "UNAVAILABLE", "深度不可用", available=False)
            rgb_rendered = self._fusion_panel(
                rgb_rendered,
                fused,
                visual,
                depth,
                stream_state.get("last_pre_fall"),
            )
        return self._compose(depth_rendered, rgb_rendered)

    def _fusion_panel(
        self,
        frame: np.ndarray,
        fused: FusedDecision,
        visual: ModalityResult,
        depth: ModalityResult,
        pre_fall: PreFallRiskResult | None = None,
    ) -> np.ndarray:
        color = (35, 35, 220) if fused.risk >= 80 else (0, 160, 255) if fused.risk >= 60 else (70, 180, 80)
        lines = [
            (f"融合状态：{fused.label}", color),
            (f"融合事件证据分：{fused.risk:.1f} / 100　质量：{fused.quality:.2f}", (245, 245, 245)),
            (f"视觉 {visual.risk:.0f}　深度 {depth.risk:.0f}　方式：{fused.method}", (225, 225, 225)),
        ]
        if isinstance(pre_fall, PreFallRiskResult):
            level_label = LEVEL_LABELS.get(pre_fall.pre_fall_risk_level, pre_fall.pre_fall_risk_level)
            lines.append((f"前置风险：{level_label} {pre_fall.pre_fall_risk_score:.1f} / 100", (245, 245, 245)))
        return _draw_chinese_panel(frame, lines)

    def _compose(self, depth_rendered: np.ndarray, rgb_rendered: np.ndarray) -> np.ndarray:
        if depth_rendered.shape[0] != rgb_rendered.shape[0]:
            depth_rendered = cv2.resize(depth_rendered, (depth_rendered.shape[1], rgb_rendered.shape[0]))
        if str(self.config.multimodal.depth_side).lower() == "right":
            return np.hstack([rgb_rendered, depth_rendered])
        return np.hstack([depth_rendered, rgb_rendered])
