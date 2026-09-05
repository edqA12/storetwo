from __future__ import annotations

import numpy as np

from core.features import extract_pose_features
from core.types import PersonPose


def make_pose(lying: bool = False, center_shift: float = 0.0) -> PersonPose:
    points = np.zeros((17, 2), dtype=np.float32)
    confidence = np.zeros(17, dtype=np.float32)
    if lying:
        points[5], points[6] = (35, 50 + center_shift), (35, 60 + center_shift)
        points[11], points[12] = (72, 50 + center_shift), (72, 60 + center_shift)
        bbox = (20.0, 42.0 + center_shift, 92.0, 72.0 + center_shift)
    else:
        points[5], points[6] = (42, 28 + center_shift), (58, 28 + center_shift)
        points[11], points[12] = (44, 65 + center_shift), (56, 65 + center_shift)
        bbox = (30.0, 10.0 + center_shift, 70.0, 105.0 + center_shift)
    confidence[[5, 6, 11, 12]] = 0.95
    return PersonPose(1, bbox, 0.9, points, confidence)


def test_standing_and_lying_features_are_separated():
    standing = extract_pose_features(make_pose(False), (160, 120, 3), {}, 0.0, 0.3, 1.2, 0.8)
    lying = extract_pose_features(make_pose(True), (160, 120, 3), {}, 0.0, 0.3, 1.2, 0.8)
    assert standing.valid and lying.valid
    assert standing.torso_angle_deg is not None and standing.torso_angle_deg < 15
    assert lying.torso_angle_deg is not None and lying.torso_angle_deg > 75
    assert lying.lying_score > standing.lying_score


def test_window_velocity_detects_downward_motion():
    state: dict = {}
    last = None
    for timestamp, shift in [(0.0, 0.0), (0.35, 10.0), (0.7, 28.0)]:
        last = extract_pose_features(make_pose(False, shift), (160, 120, 3), state, timestamp, 0.3, 1.2, 0.8)
    assert last is not None
    assert last.down_velocity > 0.2


def test_one_occluded_torso_keypoint_remains_valid():
    pose = make_pose(False)
    pose.keypoint_confidence[11] = 0.05
    features = extract_pose_features(pose, (160, 120, 3), {}, 0.0, 0.3, 1.2, 0.8)
    assert features.valid is True
    assert features.torso_angle_deg is not None


def test_two_missing_torso_keypoints_is_invalid():
    pose = make_pose(False)
    pose.keypoint_confidence[[11, 12]] = 0.05
    features = extract_pose_features(pose, (160, 120, 3), {}, 0.0, 0.3, 1.2, 0.8)
    assert features.valid is False
    assert features.torso_angle_deg is None


def test_lower_body_keypoints_produce_knee_angle_for_sit_to_stand():
    pose = make_pose(False)
    pose.keypoints_xy[13], pose.keypoints_xy[15] = (45, 92), (45, 130)
    pose.keypoints_xy[14], pose.keypoints_xy[16] = (55, 92), (55, 130)
    pose.keypoint_confidence[[13, 14, 15, 16]] = 0.95
    features = extract_pose_features(pose, (160, 120, 3), {}, 0.0, 0.3, 1.2, 0.8)
    assert features.knee_angle_deg is not None
    assert features.knee_angle_deg > 150.0
