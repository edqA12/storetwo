from __future__ import annotations

import math
from typing import Any

from .types import PersonPose


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _center_distance_ratio(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    acx, acy = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    bcx, bcy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    diagonal = math.hypot(max(a[2] - a[0], b[2] - b[0]), max(a[3] - a[1], b[3] - b[1]))
    return math.hypot(acx - bcx, acy - bcy) / max(diagonal, 1.0)


def assign_track_ids(
    detections: list[PersonPose],
    stream_state: dict[str, Any],
    timestamp_s: float,
    iou_threshold: float,
    center_threshold: float,
    missing_tolerance_s: float,
) -> list[PersonPose]:
    """使用每个会话独立的轻量匹配器，避免YOLO跟踪器跨会话串状态。"""

    tracks: dict[str, dict[str, Any]] = stream_state.setdefault("tracks", {})
    expired = [
        key for key, item in tracks.items()
        if timestamp_s - float(item.get("last_seen", timestamp_s)) > missing_tolerance_s
    ]
    for key in expired:
        tracks.pop(key, None)

    candidates: list[tuple[float, float, int, str]] = []
    for det_index, detection in enumerate(detections):
        for track_key, track in tracks.items():
            previous_bbox = tuple(track.get("bbox", detection.bbox_xyxy))
            overlap = _iou(detection.bbox_xyxy, previous_bbox)  # type: ignore[arg-type]
            distance = _center_distance_ratio(detection.bbox_xyxy, previous_bbox)  # type: ignore[arg-type]
            if overlap >= iou_threshold or distance <= center_threshold:
                candidates.append((overlap, -distance, det_index, track_key))

    candidates.sort(reverse=True)
    assigned_detections: set[int] = set()
    assigned_tracks: set[str] = set()
    for _overlap, _negative_distance, det_index, track_key in candidates:
        if det_index in assigned_detections or track_key in assigned_tracks:
            continue
        detections[det_index].track_id = int(track_key)
        assigned_detections.add(det_index)
        assigned_tracks.add(track_key)

    next_id = int(stream_state.get("next_track_id", 1))
    for det_index, detection in enumerate(detections):
        if det_index not in assigned_detections:
            detection.track_id = next_id
            next_id += 1
        key = str(detection.track_id)
        previous = tracks.get(key, {})
        previous.update({
            "bbox": list(map(float, detection.bbox_xyxy)),
            "last_seen": float(timestamp_s),
            "history": previous.get("history", []),
            "machine": previous.get("machine", {}),
        })
        tracks[key] = previous

    stream_state["next_track_id"] = next_id
    return detections
