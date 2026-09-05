from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .config import MultimodalConfig
from .types import ModalityResult


DEPTH_LABELS = {
    "DEPTH_NORMAL": "深度画面正常",
    "DEPTH_MOTION": "深度画面检测到运动",
    "DEPTH_EVIDENCE": "深度画面检测到下降证据",
    "DEPTH_IMPACT": "深度画面检测到强变化",
    "DEPTH_LOW_ACTIVITY": "深度画面检测到低位低活动",
    "UNAVAILABLE": "深度画面不可用",
}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _as_gray(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 3:
        array = cv2.cvtColor(array[:, :, :3], cv2.COLOR_BGR2GRAY)
    if array.dtype != np.uint8:
        finite = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        low, high = float(np.min(finite)), float(np.max(finite))
        array = np.zeros_like(finite, dtype=np.uint8) if high <= low else np.clip(
            (finite - low) * (255.0 / (high - low)), 0, 255
        ).astype(np.uint8)
    return array


def _metric_depth(frame: np.ndarray) -> np.ndarray | None:
    """Return calibrated/raw depth values when the input is not an 8-bit preview."""
    array = np.asarray(frame)
    if array.ndim != 2 or array.dtype == np.uint8:
        return None
    metric = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if not metric.size or float(np.max(metric)) <= 255.0:
        return None
    return metric


class DepthPipeline:
    """对视频中的深度可视化画面提取独立的时序运动证据。"""

    def __init__(self, config: MultimodalConfig) -> None:
        self.config = config

    def process_frame(
        self,
        depth_frame: np.ndarray,
        timestamp_s: float,
        stream_state: dict[str, Any] | None,
    ) -> tuple[ModalityResult, dict[str, Any], np.ndarray]:
        state = dict(stream_state or {})
        metric_depth = _metric_depth(depth_frame)
        gray = _as_gray(depth_frame)
        valid = (metric_depth > 0.0) if metric_depth is not None else (gray > 2)
        valid_ratio = float(np.mean(valid)) if gray.size else 0.0
        metric_depth_median = (
            float(np.median(metric_depth[valid]))
            if metric_depth is not None and np.any(valid) else None
        )
        contrast = float(np.std(gray[valid])) if np.any(valid) else 0.0
        quality = _clamp(valid_ratio / max(self.config.depth_min_valid_ratio, 1e-6))
        quality *= _clamp(contrast / 24.0)

        previous = state.get("previous_gray")
        last_timestamp = state.get("last_timestamp_s")
        reasons: list[str] = []
        diagnostics: dict[str, Any] = {
            "valid_ratio": round(valid_ratio, 4),
            "contrast": round(contrast, 2),
            "motion_ratio": 0.0,
            "motion_strength": 0.0,
            "motion_centroid_y": None,
            "downward_delta": 0.0,
            "metric_depth_input": metric_depth is not None,
            "median_depth": (
                round(metric_depth_median, 2) if metric_depth_median is not None else None
            ),
        }

        available = bool(quality >= self.config.min_depth_quality)
        risk = 0.0
        state_name = "DEPTH_NORMAL" if available else "UNAVAILABLE"
        motion_ratio = 0.0
        centroid_y: float | None = None
        distance_change_ratio = 0.0
        longitudinal_velocity = 0.0
        pre_fall_spatial_score = 0.0

        ordered = last_timestamp is None or timestamp_s > float(last_timestamp)
        if available and ordered and isinstance(previous, np.ndarray) and previous.shape == gray.shape:
            difference = cv2.absdiff(gray, previous)
            motion_mask = difference >= int(max(1.0, self.config.depth_motion_threshold))
            motion_ratio = float(np.mean(motion_mask))
            moving_values = difference[motion_mask]
            motion_strength = float(np.mean(moving_values)) if moving_values.size else 0.0
            if np.any(motion_mask):
                y_coordinates = np.nonzero(motion_mask)[0]
                centroid_y = float(np.mean(y_coordinates) / max(1, gray.shape[0] - 1))

            previous_centroid = state.get("last_motion_centroid_y")
            downward_delta = (
                max(0.0, centroid_y - float(previous_centroid))
                if centroid_y is not None and previous_centroid is not None else 0.0
            )
            motion_score = _clamp(motion_ratio / max(self.config.depth_motion_ratio, 1e-6))
            impact_score = _clamp(motion_strength / max(self.config.depth_impact_delta, 1e-6))
            downward_score = _clamp(downward_delta / max(self.config.depth_downward_delta, 1e-6))
            lower_score = _clamp(((centroid_y or 0.0) - 0.45) / 0.35)
            elapsed_s = max(1e-6, float(timestamp_s) - float(last_timestamp))
            if metric_depth_median is not None:
                previous_depth_median = state.get("previous_depth_median")
                if previous_depth_median is not None and float(previous_depth_median) > 0.0:
                    distance_change_ratio = abs(
                        metric_depth_median - float(previous_depth_median)
                    ) / float(previous_depth_median)
            elif np.any(motion_mask):
                signed_change = gray.astype(np.float32) - previous.astype(np.float32)
                distance_change_ratio = float(abs(np.median(signed_change[motion_mask])) / 255.0)
            if distance_change_ratio > 0.0:
                longitudinal_velocity = distance_change_ratio / elapsed_s
            distance_score = _clamp(
                distance_change_ratio / max(self.config.depth_distance_change_ratio, 1e-6)
            )
            longitudinal_score = _clamp(
                longitudinal_velocity / max(self.config.depth_longitudinal_velocity, 1e-6)
            )
            pre_fall_spatial_score = 100.0 * (
                0.40 * downward_score
                + 0.35 * max(distance_score, longitudinal_score) * motion_score
                + 0.25 * impact_score * motion_score
            )
            risk = 100.0 * (
                0.45 * downward_score
                + 0.30 * impact_score * motion_score
                + 0.15 * lower_score * motion_score
                + 0.10 * motion_score
            )

            evidence_strength = max(downward_score, impact_score * motion_score)
            if evidence_strength >= 0.55:
                state["evidence_until"] = float(timestamp_s + self.config.depth_evidence_hold_s)
                reasons.append("深度画面出现快速下降或强变化")
            evidence_active = timestamp_s <= float(state.get("evidence_until", -1.0))
            if evidence_active:
                risk = max(risk, 55.0)
                state_name = "DEPTH_EVIDENCE"
            if risk >= 70.0:
                state_name = "DEPTH_IMPACT"
            elif motion_ratio >= self.config.depth_motion_ratio and not evidence_active:
                state_name = "DEPTH_MOTION"
                reasons.append("深度画面检测到明显运动")
            if (
                evidence_active
                and motion_ratio < self.config.depth_motion_ratio * 0.25
                and float(state.get("last_motion_centroid_y", 0.0)) >= 0.55
            ):
                state_name = "DEPTH_LOW_ACTIVITY"
                risk = max(risk, 65.0)
                reasons.append("强变化后在画面低位保持低活动")

            diagnostics.update({
                "motion_ratio": round(motion_ratio, 5),
                "motion_strength": round(motion_strength, 2),
                "motion_centroid_y": round(centroid_y, 4) if centroid_y is not None else None,
                "downward_delta": round(downward_delta, 4),
                "distance_change_ratio": round(distance_change_ratio, 5),
                "longitudinal_velocity": round(longitudinal_velocity, 5),
                "pre_fall_spatial_score": round(pre_fall_spatial_score, 1),
            })
            if centroid_y is not None and motion_ratio >= self.config.depth_motion_ratio * 0.25:
                state["last_motion_centroid_y"] = centroid_y
        elif not ordered:
            reasons.append("深度帧时间戳无效")
            state_name = "UNAVAILABLE"
            available = False
            quality = 0.0
        elif not available:
            reasons.append("深度画面有效像素或对比度不足")

        state["previous_gray"] = gray.copy()
        if metric_depth_median is not None:
            state["previous_depth_median"] = metric_depth_median
        state["last_timestamp_s"] = float(timestamp_s)
        result = ModalityResult(
            modality="depth",
            timestamp_s=float(timestamp_s),
            risk=round(_clamp(risk, 0.0, 100.0), 1),
            quality=round(_clamp(quality), 3),
            state=state_name,
            label=DEPTH_LABELS[state_name],
            reasons=reasons,
            diagnostics=diagnostics,
            available=available,
        )
        state["last_result"] = result
        return result, state, self.render(depth_frame, result)

    def render(self, depth_frame: np.ndarray, result: ModalityResult | None) -> np.ndarray:
        colored = cv2.applyColorMap(_as_gray(depth_frame), cv2.COLORMAP_TURBO)
        state_name = result.state if result else "UNAVAILABLE"
        risk = result.risk if result else 0.0
        quality = result.quality if result else 0.0
        cv2.rectangle(colored, (10, 10), (min(colored.shape[1] - 10, 360), 84), (15, 15, 15), -1)
        cv2.putText(colored, "DEPTH EVIDENCE", (22, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        cv2.putText(colored, f"{state_name}  EVIDENCE {risk:.0f}", (22, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1)
        cv2.putText(colored, f"QUALITY {quality:.2f}", (22, 77), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 245), 1)
        return colored

    def render_cached(self, depth_frame: np.ndarray, stream_state: dict[str, Any]) -> np.ndarray:
        last_result = stream_state.get("last_result")
        return self.render(depth_frame, last_result if isinstance(last_result, ModalityResult) else None)
