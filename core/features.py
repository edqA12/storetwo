from __future__ import annotations

import math
from typing import Any

import numpy as np

from .types import PersonPose, PoseFeatures


SHOULDERS = (5, 6)
HIPS = (11, 12)
KNEES = (13, 14)
ANKLES = (15, 16)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _linear_slope(samples: list[dict[str, Any]], key: str) -> float:
    points = [(float(item["ts"]), float(item[key])) for item in samples if item.get(key) is not None]
    if len(points) < 2:
        return 0.0
    times = np.asarray([point[0] for point in points], dtype=np.float64)
    values = np.asarray([point[1] for point in points], dtype=np.float64)
    times -= times.mean()
    denominator = float(np.dot(times, times))
    if denominator <= 1e-9:
        return 0.0
    return float(np.dot(times, values - values.mean()) / denominator)


def _mean_point(points: np.ndarray) -> np.ndarray:
    return np.mean(points, axis=0, dtype=np.float64)


def _joint_angle(first: np.ndarray, vertex: np.ndarray, third: np.ndarray) -> float | None:
    first_vector = first - vertex
    third_vector = third - vertex
    denominator = float(np.linalg.norm(first_vector) * np.linalg.norm(third_vector))
    if denominator <= 1e-9:
        return None
    cosine = float(np.dot(first_vector, third_vector) / denominator)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def extract_pose_features(
    pose: PersonPose,
    frame_shape: tuple[int, ...],
    track_state: dict[str, Any],
    timestamp_s: float,
    keypoint_threshold: float,
    history_seconds: float,
    stale_derivative_s: float,
) -> PoseFeatures:
    """从骨架关键点提取尺度无关特征，并用短窗口回归降低抖动。"""

    height, width = frame_shape[:2]
    x1, y1, x2, y2 = pose.bbox_xyxy
    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    if height <= 0 or width <= 0 or box_width < 2.0 or box_height < 2.0:
        return PoseFeatures(valid=False)

    keypoints = np.asarray(pose.keypoints_xy, dtype=np.float64)
    confidences = np.asarray(pose.keypoint_confidence, dtype=np.float64)
    if keypoints.ndim != 2 or keypoints.shape[0] < 13 or keypoints.shape[1] < 2:
        return PoseFeatures(valid=False)
    if confidences.ndim != 1 or confidences.shape[0] < keypoints.shape[0]:
        confidences = np.ones(keypoints.shape[0], dtype=np.float64)

    required = np.asarray([*SHOULDERS, *HIPS], dtype=np.int64)
    shoulder_visible = confidences[list(SHOULDERS)] >= keypoint_threshold
    hip_visible = confidences[list(HIPS)] >= keypoint_threshold
    # 侧身或短暂遮挡时，YOLO常会丢失一侧肩/髋。允许四个躯干点中
    # 有三个可靠点继续计算，但必须同时看到至少一个肩点和一个髋点。
    if (
        int(np.count_nonzero(shoulder_visible)) + int(np.count_nonzero(hip_visible)) < 3
        or not bool(np.any(shoulder_visible))
        or not bool(np.any(hip_visible))
    ):
        return PoseFeatures(valid=False, box_aspect=box_width / box_height)

    shoulder_center = _mean_point(keypoints[list(SHOULDERS), :2][shoulder_visible])
    hip_center = _mean_point(keypoints[list(HIPS), :2][hip_visible])
    torso_vector = shoulder_center - hip_center
    if float(np.linalg.norm(torso_vector)) < 2.0:
        return PoseFeatures(valid=False, box_aspect=box_width / box_height)

    torso_angle = math.degrees(math.atan2(abs(float(torso_vector[0])), abs(float(torso_vector[1]))))
    center = (shoulder_center + hip_center) / 2.0
    center_x_norm = float(center[0]) / float(width)
    center_y_norm = float(center[1]) / float(height)
    hip_y_norm = float(hip_center[1]) / float(height)
    box_aspect = box_width / box_height
    quality = float(np.mean(confidences[required]))

    knee_angles: list[float] = []
    for hip_index, knee_index, ankle_index in ((11, 13, 15), (12, 14, 16)):
        if min(confidences[hip_index], confidences[knee_index], confidences[ankle_index]) < keypoint_threshold:
            continue
        angle = _joint_angle(
            keypoints[hip_index, :2], keypoints[knee_index, :2], keypoints[ankle_index, :2]
        )
        if angle is not None:
            knee_angles.append(angle)
    knee_angle = float(np.mean(knee_angles)) if knee_angles else None
    box_area_ratio = box_width * box_height / max(float(width * height), 1.0)

    visible = confidences >= keypoint_threshold
    current_points = keypoints[visible, :2]
    previous_points_raw = track_state.get("last_visible_points")
    motion_norm = 0.0
    if previous_points_raw is not None:
        previous = np.asarray(previous_points_raw, dtype=np.float64)
        if previous.shape == current_points.shape and current_points.size:
            diagonal = max(math.hypot(box_width, box_height), 1.0)
            motion_norm = float(np.mean(np.linalg.norm(current_points - previous, axis=1)) / diagonal)
    track_state["last_visible_points"] = current_points.tolist()

    history: list[dict[str, Any]] = track_state.setdefault("history", [])
    last_ts = float(history[-1]["ts"]) if history else None
    if last_ts is not None and (timestamp_s <= last_ts or timestamp_s - last_ts > stale_derivative_s):
        history.clear()

    history.append({
        "ts": float(timestamp_s),
        "center_x": center_x_norm,
        "center_y": center_y_norm,
        "torso_angle": torso_angle,
    })
    minimum_ts = timestamp_s - history_seconds
    history[:] = [item for item in history if float(item["ts"]) >= minimum_ts]

    lateral_velocity = _linear_slope(history, "center_x")
    down_velocity = _linear_slope(history, "center_y")
    center_speed = math.hypot(lateral_velocity, down_velocity)
    angle_velocity = _linear_slope(history, "torso_angle")
    angle_component = _clamp((torso_angle - 40.0) / 30.0)
    aspect_component = _clamp((box_aspect - 0.55) / 0.75)
    lying_score = _clamp(0.78 * angle_component + 0.22 * aspect_component)

    return PoseFeatures(
        valid=True,
        torso_angle_deg=torso_angle,
        box_aspect=box_aspect,
        hip_y_norm=hip_y_norm,
        center_x_norm=center_x_norm,
        center_y_norm=center_y_norm,
        center_speed=center_speed,
        lateral_velocity=lateral_velocity,
        down_velocity=down_velocity,
        angle_velocity=angle_velocity,
        motion_norm=motion_norm,
        lying_score=lying_score,
        knee_angle_deg=knee_angle,
        box_area_ratio=box_area_ratio,
        quality=quality,
    )
