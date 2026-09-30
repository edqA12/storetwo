"""Metric body features. Calibration evidence is checked, never inferred from size.

Coordinates: OpenCV x right, y down, z forward; metres, depth -> color.
These features are observational until labelled metric sequences justify fusion.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class SpatialConfig:
    calibration_file: str = 'data/calibration/astra_spatial.json'
    keypoint_confidence: float = 0.5
    min_samples: int = 12
    min_valid_ratio: float = 0.5
    max_depth_spread_m: float = 0.15
    max_gap_s: float = 0.75
    low_height_m: float = 0.55
    max_speed_m_s: float = 4.0
    max_reprojection_px: float = 3.0
    max_floor_residual_m: float = 0.02


def array(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError('标定数组形状或数值无效')
    return result


def camera(value: dict) -> tuple[np.ndarray, np.ndarray]:
    matrix = array(value['matrix'], (3, 3))
    distortion = array(value['distortion'], (5,))
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or not np.allclose(matrix[2], [0, 0, 1]):
        raise ValueError('相机内参无效')
    if not np.allclose([matrix[0, 1], matrix[1, 0]], 0):
        raise ValueError('不支持倾斜内参')
    size = array(value['size'], (2,))
    if np.any(size <= 0) or np.any(size != np.floor(size)):
        raise ValueError('标定图像尺寸无效')
    return matrix, distortion


def unproject(uv: np.ndarray, z: np.ndarray, intrinsics: dict) -> np.ndarray:
    matrix, distortion = camera(intrinsics)
    if not len(uv):
        return np.empty((0, 3), np.float64)
    rays = cv2.undistortPoints(np.asarray(uv, np.float64).reshape(-1, 1, 2), matrix, distortion).reshape(-1, 2)
    return np.column_stack((rays, np.ones(len(rays)))) * np.asarray(z).reshape(-1, 1)


def project(points: np.ndarray, intrinsics: dict) -> np.ndarray:
    matrix, distortion = camera(intrinsics)
    return cv2.projectPoints(points, np.zeros(3), np.zeros(3), matrix, distortion)[0].reshape(-1, 2)


def fit_floor(points: np.ndarray, tolerance: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """RANSAC on an explicitly selected floor region, not the largest scene plane."""
    points = np.asarray(points, np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 100 or not np.isfinite(points).all():
        raise ValueError('地面至少需要100个有效三维点')
    rng = np.random.default_rng(17)
    best = np.zeros(len(points), bool)
    for _ in range(200):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        length = np.linalg.norm(normal)
        if length < 1e-8:
            continue
        normal /= length
        selected = np.abs((points - sample[0]) @ normal) < tolerance
        if selected.sum() > best.sum():
            best = selected
    if best.mean() < 0.8:
        raise ValueError('地面平面内点不足80%')
    center = points[best].mean(axis=0)
    _, singular, vectors = np.linalg.svd(points[best] - center, full_matrices=False)
    if singular[1] / np.sqrt(best.sum()) < 0.15:
        raise ValueError('地面区域覆盖不足或近似共线')
    normal = vectors[-1]
    offset = -float(center @ normal)
    if offset < 0:
        normal, offset = -normal, -offset
    if not 0.3 <= offset <= 3.0 or normal[1] > -0.4:
        raise ValueError('地面方向或相机离地高度不合理；请检查选区和相机姿态')
    return np.r_[normal, offset], best


class Calibration:
    def __init__(self, payload: dict, config: SpatialConfig):
        self.data = payload
        if payload.get('schema') != 1 or payload.get('convention') != 'opencv_depth_to_color_m':
            raise ValueError('标定版本、坐标方向或单位不明确')
        self.depth, self.color = payload['depth'], payload['color']
        camera(self.depth)
        camera(self.color)
        self.rotation = array(payload['rotation'], (3, 3))
        self.translation = array(payload['translation_m'], (3,))
        if not np.allclose(self.rotation.T @ self.rotation, np.eye(3), atol=1e-4) or abs(np.linalg.det(self.rotation) - 1) > 1e-4:
            raise ValueError('外参旋转矩阵无效')
        if np.linalg.norm(self.translation) > 0.5:
            raise ValueError('外参平移超出相机合理范围')
        if not all(isinstance(payload.get('device_ids', {}).get(k), str) and payload['device_ids'][k] for k in ('depth', 'color')):
            raise ValueError('缺少深度和彩色设备标识')
        evidence = payload['verification']
        if not evidence.get('independent_capture') or not evidence.get('capture_id'):
            raise ValueError('缺少独立采集的配准验证证据')
        if payload.get('fit_capture_id') == evidence['capture_id']:
            raise ValueError('拟合与验证采集不能相同')
        uvz = np.asarray(evidence['depth_uvz_m'], np.float64)
        rgb = np.asarray(evidence['color_uv'], np.float64)
        if uvz.ndim != 2 or uvz.shape[1] != 3 or len(uvz) < 12 or rgb.shape != (len(uvz), 2):
            raise ValueError('至少需要12个独立验证对应点')
        if not np.isfinite(uvz).all() or not np.isfinite(rgb).all() or np.any(uvz[:, 2] <= 0):
            raise ValueError('验证点无效')
        for uv, size in ((uvz[:, :2], self.depth['size']), (rgb, self.color['size'])):
            if np.any(uv < 0) or np.any(uv >= size) or np.any(np.ptp(uv, axis=0) < np.asarray(size) * 0.25):
                raise ValueError('验证点越界或图像覆盖不足')
        if np.ptp(uvz[:, 2]) < 0.3:
            raise ValueError('验证点需要覆盖至少0.3米距离变化')
        xyz = unproject(uvz[:, :2], uvz[:, 2], self.depth) @ self.rotation.T + self.translation
        if np.any(xyz[:, 2] <= 0):
            raise ValueError('对应点投影在相机后方')
        errors = np.linalg.norm(project(xyz, self.color) - rgb, axis=1)
        self.error_px = float(np.percentile(errors, 95))
        if self.error_px > config.max_reprojection_px or errors.max() > 2 * config.max_reprojection_px:
            raise ValueError('独立对应点配准误差超限')
        floor = payload['floor']
        if not floor.get('operator_confirmed_floor') or not floor.get('placement_id'):
            raise ValueError('尚未确认地面选区与固定安装位置')
        self.floor, _ = fit_floor(floor['points_color_m'], config.max_floor_residual_m)
        heldout = np.asarray(floor['validation_points_color_m'], np.float64)
        if heldout.ndim != 2 or heldout.shape[1] != 3 or len(heldout) < 30 or not np.isfinite(heldout).all():
            raise ValueError('缺少独立地面验证点')
        if float(np.percentile(np.abs(heldout @ self.floor[:3] + self.floor[3]), 95)) > config.max_floor_residual_m:
            raise ValueError('地面验证残差超限')
        self.identifier = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def register(self, depth: np.ndarray, rgb_shape: tuple) -> np.ndarray:
        dw, dh = self.depth['size']
        cw, ch = self.color['size']
        if depth.shape != (dh, dw) or tuple(rgb_shape[:2]) != (ch, cw):
            raise ValueError('当前分辨率与标定不一致，禁止缩放代替配准')
        y, x = np.nonzero(np.isfinite(depth) & (depth > 0))
        xyz = unproject(np.column_stack((x, y)), depth[y, x], self.depth) @ self.rotation.T + self.translation
        xyz = xyz[xyz[:, 2] > 0]
        output = np.full((ch, cw, 3), np.nan, np.float32)
        if not len(xyz):
            return output
        projected = project(xyz, self.color)
        finite = np.isfinite(projected).all(axis=1)
        xyz, projected = xyz[finite], projected[finite]
        uv = np.rint(projected).astype(np.int64)
        valid = (uv[:, 0] >= 0) & (uv[:, 0] < cw) & (uv[:, 1] >= 0) & (uv[:, 1] < ch)
        uv, xyz = uv[valid], xyz[valid]
        # Z-buffer: only nearest surface survives; never fill occluded holes.
        order = np.argsort(xyz[:, 2], kind='stable')
        index = uv[order, 1] * cw + uv[order, 0]
        _, first = np.unique(index, return_index=True)
        chosen = order[first]
        output[uv[chosen, 1], uv[chosen, 0]] = xyz[chosen]
        return output


class BodySpatialAnalyzer:
    def __init__(self, config: SpatialConfig):
        self.config = config
        self.calibration: Calibration | None = None
        self.reason = '尚未加载经验证的空间与地面标定'
        self.previous: dict | None = None

    def load(self, path: Path, mount_confirmed: bool) -> None:
        self.previous, self.calibration = None, None
        if not mount_confirmed:
            self.reason = '未确认相机固定位置与地面标定一致'
            return
        try:
            self.calibration = Calibration(json.loads(path.read_text(encoding='utf-8-sig')), self.config)
            self.reason = ''
        except (OSError, ValueError, KeyError, TypeError, cv2.error) as exc:
            self.reason = f'空间标定不可用：{exc}'

    def update(self, depth: np.ndarray | None, shape: tuple, poses: dict, timestamp: float, device_ids: dict) -> dict:
        base = {'available': False, 'fusion_applied': False, 'scope': 'single_person_torso', 'persons': [],
                'registration_verified': False, 'reason': self.reason}
        def unavailable(reason: str) -> dict:
            self.previous = None
            return {**base, 'reason': reason}
        if not np.isfinite(timestamp):
            return unavailable('三维帧时间戳无效')
        cal = self.calibration
        if cal is None:
            return unavailable(self.reason)
        base.update(calibration_id=cal.identifier, placement_id=cal.data['floor']['placement_id'], reprojection_p95_px=cal.error_px)
        if device_ids != cal.data['device_ids']:
            return unavailable('设备标识与标定不一致')
        if depth is None:
            return unavailable('深度缺失、质量不足或失步')
        if len(poses) != 1:
            return unavailable('当前仅支持单人三维分析；多人不进行深度归属' if poses else '未检测到人')
        try:
            xyz = cal.register(depth, shape)
            base['registration_verified'] = True
            track_id, pose = next(iter(poses.items()))
            points = np.asarray(pose['keypoints'], float)[[5, 6, 12, 11]]
            confidence = np.asarray(pose['confidence'], float)[[5, 6, 12, 11]]
            if not np.isfinite(points).all() or not np.isfinite(confidence).all() or np.any(confidence < self.config.keypoint_confidence):
                return unavailable('肩髋关键点质量不足')
            if np.any(points < 0) or np.any(points[:, 0] >= shape[1]) or np.any(points[:, 1] >= shape[0]):
                return unavailable('躯干关键点越界')
            # Shrink the shoulder/hip polygon to avoid clothing edges/background.
            center = points.mean(axis=0)
            polygon = np.rint(center + (points - center) * 0.6).astype(np.int32)
            mask = np.zeros(shape[:2], np.uint8)
            cv2.fillConvexPoly(mask, cv2.convexHull(polygon), 1)
            candidates = xyz[mask.astype(bool)]
            valid = candidates[np.isfinite(candidates).all(axis=1)]
            ratio = len(valid) / max(1, len(candidates))
            if len(valid) < self.config.min_samples or ratio < self.config.min_valid_ratio:
                return unavailable('躯干有效深度覆盖不足')
            spread = float(np.percentile(valid[:, 2], 90) - np.percentile(valid[:, 2], 10))
            if spread > self.config.max_depth_spread_m:
                return unavailable('躯干深度混合或遮挡，无法可靠归属')
            position = np.median(valid, axis=0)
            height = float(position @ cal.floor[:3] + cal.floor[3])
            if not 0 <= height <= 2.5:
                return unavailable('躯干离地高度不合理')
            previous = self.previous
            velocity, activity, low_since = None, None, timestamp if height < self.config.low_height_m else None
            if previous and previous['track_id'] == track_id:
                dt = timestamp - previous['timestamp']
                if dt <= 0:
                    return unavailable('重复或乱序三维帧')
                if dt <= self.config.max_gap_s:
                    activity = float(np.linalg.norm(position - previous['position']) / dt)
                    if activity > self.config.max_speed_m_s:
                        return unavailable('三维位置跳变，已清除运动历史')
                    velocity = (previous['height'] - height) / dt
                    if low_since is not None and previous['low_since'] is not None:
                        low_since = previous['low_since']
            self.previous = dict(track_id=track_id, timestamp=timestamp, height=height, position=position, low_since=low_since)
            feature = dict(track_id=str(track_id), torso_position_color_m=position.tolist(), torso_height_m=height,
                           descent_speed_m_s=velocity, low_duration_s=timestamp-low_since if low_since is not None else 0.0,
                           low_activity_m_s=activity if low_since is not None else None, valid_ratio=ratio,
                           depth_spread_m=spread, sample_count=len(valid), temporal_ready=velocity is not None)
            return {**base, 'available': True, 'reason': '三维特征观察模式；尚无标注依据，未用于风险加权', 'persons': [feature]}
        except (ValueError, KeyError, IndexError, TypeError, cv2.error) as exc:
            return unavailable(f'三维分析不可用：{exc}')


def spatial_markdown(value: dict | None) -> str:
    value = value or {}
    text = '\n- 人体三维：' + str(value.get('reason', '未启用／未标定'))
    for person in value.get('persons', []):
        speed = person['descent_speed_m_s']
        activity = person['low_activity_m_s']
        text += f"\n- 人物 {person['track_id']}：躯干离地 {person['torso_height_m']:.2f} m；下降速度 "
        text += f'{speed:.2f} m/s' if speed is not None else '预热中'
        text += f"；低位持续 {person['low_duration_s']:.2f} s；低位活动 "
        text += f'{activity:.2f} m/s' if activity is not None else '不可用'
        text += f"；有效覆盖 {person['valid_ratio']:.0%}"
    return text + '\n'
