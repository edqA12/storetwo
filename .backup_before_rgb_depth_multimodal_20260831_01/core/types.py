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
    center_y_norm: float | None = None
    down_velocity: float = 0.0
    angle_velocity: float = 0.0
    motion_norm: float = 0.0
    lying_score: float = 0.0
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


@dataclass(slots=True)
class FrameResult:
    annotated_bgr: np.ndarray
    decisions: list[Decision]
    max_risk: float
    overall_label: str
    new_events: list[dict[str, Any]]
    diagnostics: dict[str, Any]
