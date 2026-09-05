from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class PersonPose:
    """单个人在一帧中的姿态检测结果。"""

    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    box_confidence: float
    keypoints_xy: np.ndarray
    keypoint_confidence: np.ndarray


@dataclass(slots=True)
class PoseFeatures:
    """供时序状态机使用的尺度无关特征。"""

    valid: bool
    torso_angle_deg: float | None = None
    box_aspect: float | None = None
    hip_y_norm: float | None = None
    center_x_norm: float | None = None
    center_y_norm: float | None = None
    center_speed: float = 0.0
    lateral_velocity: float = 0.0
    down_velocity: float = 0.0
    angle_velocity: float = 0.0
    motion_norm: float = 0.0
    lying_score: float = 0.0
    knee_angle_deg: float | None = None
    box_area_ratio: float = 0.0
    quality: float = 0.0


@dataclass(slots=True)
class Decision:
    track_id: int
    state: str
    label: str
    risk: float
    event_started: bool = False
    event_key: str | None = None
    reason_codes: list[str] = field(default_factory=list)

    @property
    def fall_event_score(self) -> float:
        """当前短时跌倒事件证据分；保留 risk 字段用于旧接口兼容。"""
        return float(self.risk)


@dataclass(slots=True)
class ModalityResult:
    """一个感知模态在同一时间点输出的统一结果。"""

    modality: str
    timestamp_s: float
    risk: float
    quality: float
    state: str
    label: str
    reasons: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    available: bool = True

    @property
    def fall_event_score(self) -> float:
        return float(self.risk)

    def to_record(self) -> dict[str, Any]:
        return {
            "modality": self.modality,
            "timestamp_s": round(float(self.timestamp_s), 3),
            "risk": round(float(self.risk), 1),
            "quality": round(float(self.quality), 3),
            "state": self.state,
            "label": self.label,
            "reasons": list(self.reasons),
            "diagnostics": dict(self.diagnostics),
            "available": bool(self.available),
        }


@dataclass(slots=True)
class FusedDecision:
    timestamp_s: float
    risk: float
    quality: float
    state: str
    label: str
    method: str
    modality_weights: dict[str, float] = field(default_factory=dict)
    explanation: list[str] = field(default_factory=list)

    @property
    def fall_event_score(self) -> float:
        return float(self.risk)


@dataclass(slots=True)
class PreFallRiskResult:
    """基于连续行为变化的工程前置风险评估，不表示临床跌倒概率。"""

    timestamp_s: float
    track_id: int
    movement_score: float
    stability_score: float
    near_fall_count: int
    sit_to_stand_score: float
    baseline_deviation: float
    pre_fall_risk_score: float
    pre_fall_risk_level: str
    risk_factors: list[str] = field(default_factory=list)
    rgb_quality: float = 0.0
    depth_quality: float = 0.0
    depth_degraded: bool = True
    level_changed: bool = False
    previous_level: str | None = None
    movement_speed: float = 0.0
    baseline_speed: float | None = None
    speed_change_pct: float = 0.0
    sit_to_stand_duration_s: float | None = None
    sit_to_stand_status: str = "未检测"
    sit_to_stand_count: int = 0
    person_id: str = "primary"
    short_term_score: float = 0.0
    medium_term_score: float = 0.0
    long_term_score: float = 0.0
    depth_spatial_score: float = 0.0
    depth_fusion_applied: bool = False

    def to_record(self) -> dict[str, Any]:
        return {
            "timestamp_s": round(float(self.timestamp_s), 3),
            "track_id": int(self.track_id),
            "movement_score": round(float(self.movement_score), 1),
            "stability_score": round(float(self.stability_score), 1),
            "near_fall_count": int(self.near_fall_count),
            "sit_to_stand_score": round(float(self.sit_to_stand_score), 1),
            "baseline_deviation": round(float(self.baseline_deviation), 1),
            "pre_fall_risk_score": round(float(self.pre_fall_risk_score), 1),
            "pre_fall_risk_level": str(self.pre_fall_risk_level),
            "risk_factors": list(self.risk_factors),
            "rgb_quality": round(float(self.rgb_quality), 3),
            "depth_quality": round(float(self.depth_quality), 3),
            "depth_degraded": bool(self.depth_degraded),
            "level_changed": bool(self.level_changed),
            "previous_level": self.previous_level,
            "movement_speed": round(float(self.movement_speed), 4),
            "baseline_speed": (
                round(float(self.baseline_speed), 4) if self.baseline_speed is not None else None
            ),
            "speed_change_pct": round(float(self.speed_change_pct), 1),
            "sit_to_stand_duration_s": (
                round(float(self.sit_to_stand_duration_s), 3)
                if self.sit_to_stand_duration_s is not None else None
            ),
            "sit_to_stand_status": str(self.sit_to_stand_status),
            "sit_to_stand_count": int(self.sit_to_stand_count),
            "person_id": str(self.person_id),
            "short_term_score": round(float(self.short_term_score), 1),
            "medium_term_score": round(float(self.medium_term_score), 1),
            "long_term_score": round(float(self.long_term_score), 1),
            "depth_spatial_score": round(float(self.depth_spatial_score), 1),
            "depth_fusion_applied": bool(self.depth_fusion_applied),
        }


@dataclass(slots=True)
class FrameResult:
    annotated_bgr: np.ndarray
    decisions: list[Decision]
    max_risk: float
    overall_label: str
    new_events: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    modalities: list[ModalityResult] = field(default_factory=list)
    fusion: FusedDecision | None = None
    pre_fall_results: list[PreFallRiskResult] = field(default_factory=list)
    near_fall_events: list[dict[str, Any]] = field(default_factory=list)
    sit_to_stand_events: list[dict[str, Any]] = field(default_factory=list)
    sit_to_stand_attempts: list[dict[str, Any]] = field(default_factory=list)

    @property
    def max_fall_event_score(self) -> float:
        """当前画面最高跌倒事件证据分；max_risk 为旧接口兼容名。"""
        return float(self.max_risk)
