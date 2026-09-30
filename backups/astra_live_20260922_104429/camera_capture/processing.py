"""Explicit metric depth and bounded software pairing; no model or storage access."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, fields, replace
import math
import time
from typing import Any

import cv2
import numpy as np

from .manager import CaptureFrame


@dataclass
class ProcessingConfig:
    raw_depth_min_m: float = 0.6
    raw_depth_max_m: float = 6.0
    raw_min_valid_ratio: float = 0.08
    raw_motion_m: float = 0.05
    raw_impact_m: float = 0.5
    raw_jump_m: float = 1.5
    raw_jump_ratio: float = 0.6
    raw_temporal_gap_s: float = 0.75
    sync_tolerance_s: float = 0.05
    frame_max_age_s: float = 0.25
    sync_buffer_size: int = 8
    clock_warmup_frames: int = 8
    clock_residual_limit_s: float = 0.15
    color_receive_offset_s: float = 0.0

    def validate(self) -> None:
        for item in fields(self):
            key, value = item.name, getattr(self, item.name)
            if isinstance(value, (float, int)) and not math.isfinite(value):
                raise ValueError(f'{key}必须是有限数值')
        if not 0 < self.raw_depth_min_m < self.raw_depth_max_m:
            raise ValueError('深度范围无效')
        if not 0 < self.raw_min_valid_ratio <= 1 or not 0 < self.raw_jump_ratio <= 1:
            raise ValueError('深度比例阈值无效')
        if not 0 < self.sync_tolerance_s <= self.frame_max_age_s:
            raise ValueError('同步容差或过期时间无效')
        if (type(self.sync_buffer_size) is not int or type(self.clock_warmup_frames) is not int
                or not 1 <= self.sync_buffer_size <= 64 or not 2 <= self.clock_warmup_frames <= 120):
            raise ValueError('同步缓存或预热帧数无效')
        if any(value <= 0 for value in (self.raw_motion_m, self.raw_impact_m, self.raw_jump_m,
                                        self.raw_temporal_gap_s, self.clock_residual_limit_s)):
            raise ValueError('时间或运动阈值必须为正数')
        if self.color_receive_offset_s < 0:
            raise ValueError('彩色接收偏移必须为非负数')


@dataclass(frozen=True)
class MetricDepth:
    meters: np.ndarray
    valid_mask: np.ndarray
    valid_ratio: float


def normalize_depth(image: np.ndarray, unit_m: float, config: ProcessingConfig) -> MetricDepth:
    config.validate()
    array = np.asarray(image)
    if array.ndim != 2 or array.dtype not in (np.dtype('uint16'), np.dtype('float32'), np.dtype('float64')):
        raise ValueError('原始深度必须是二维uint16/float32/float64，不能传入8位预览')
    if not math.isfinite(unit_m) or unit_m <= 0 or not array.size:
        raise ValueError('必须提供明确的有效深度单位和非空数组')
    with np.errstate(over='ignore', invalid='ignore'):
        values = array.astype(np.float64) * unit_m
    valid = np.isfinite(values) & (values >= config.raw_depth_min_m) & (values <= config.raw_depth_max_m)
    # OpenNI uint16 maximum is a sentinel, never a physical distance.
    if array.dtype == np.uint16:
        valid &= array != np.iinfo(np.uint16).max
    values = np.where(valid, values, np.nan).astype(np.float32)
    values.setflags(write=False)
    valid.setflags(write=False)
    return MetricDepth(values, valid, float(np.mean(valid)))


def depth_preview(depth: MetricDepth, config: ProcessingConfig) -> np.ndarray:
    """Fixed physical scale; invalid pixels black. Never used by motion analysis."""
    normalized = (np.nan_to_num(depth.meters, nan=config.raw_depth_min_m) - config.raw_depth_min_m)
    gray = np.clip(normalized * 255 / (config.raw_depth_max_m - config.raw_depth_min_m), 0, 255).astype(np.uint8)
    preview = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    preview[~depth.valid_mask] = 0
    return preview


def depth_motion(depth: MetricDepth, timestamp_s: float, state: dict[str, Any],
                 config: ProcessingConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Motion on common valid metric pixels; no cross-gap or cross-shape derivatives."""
    previous = state.get('metric')
    old_time = state.get('timestamp_s')
    metrics: dict[str, Any] = {
        'valid_ratio': depth.valid_ratio, 'available': depth.valid_ratio >= config.raw_min_valid_ratio,
        'temporal_valid': False, 'reset_reason': None, 'jump_ratio': 0.0,
        'usable_for_motion': False,
        'motion_ratio': 0.0, 'motion_strength_m': 0.0, 'motion_centroid_y': None,
        'downward_delta': 0.0, 'distance_change_ratio': 0.0, 'longitudinal_velocity': 0.0,
        'velocity_m_s': 0.0, 'elapsed_s': None,
        'median_depth_m': float(np.nanmedian(depth.meters)) if np.any(depth.valid_mask) else None,
    }
    if not math.isfinite(timestamp_s) or (old_time is not None and timestamp_s <= old_time):
        metrics.update(available=False, reset_reason='invalid_timestamp')
        return metrics, state  # Rejected input must not poison the accepted baseline.
    if not metrics['available']:
        metrics['reset_reason'] = 'insufficient_valid_depth'
        return metrics, {}
    next_state: dict[str, Any] = {'metric': depth, 'timestamp_s': timestamp_s}
    if previous is None or previous.meters.shape != depth.meters.shape:
        metrics['reset_reason'] = 'warmup_or_shape_change'
        return metrics, next_state
    elapsed = timestamp_s - old_time
    metrics['elapsed_s'] = elapsed
    if elapsed > config.raw_temporal_gap_s:
        metrics['reset_reason'] = 'time_gap'
        return metrics, next_state
    common = previous.valid_mask & depth.valid_mask
    if np.mean(common) < config.raw_min_valid_ratio:
        metrics['reset_reason'] = 'insufficient_common_pixels'
        return metrics, next_state
    difference = np.zeros(depth.meters.shape, dtype=np.float32)
    difference[common] = depth.meters[common] - previous.meters[common]
    metrics['jump_ratio'] = float(np.mean(np.abs(difference[common]) > config.raw_jump_m))
    if metrics['jump_ratio'] >= config.raw_jump_ratio:
        metrics.update(available=False, reset_reason='abrupt_depth_jump')
        return metrics, {}  # New stable scene must warm up again.
    moving = common & (np.abs(difference) >= config.raw_motion_m)
    metrics['motion_ratio'] = float(np.mean(moving))
    metrics['motion_strength_m'] = float(np.mean(np.abs(difference[moving]))) if np.any(moving) else 0.0
    if np.any(moving):
        centroid = float(np.mean(np.nonzero(moving)[0]) / max(1, depth.meters.shape[0] - 1))
        metrics['motion_centroid_y'] = centroid
        if state.get('centroid') is not None:
            metrics['downward_delta'] = max(0.0, centroid - state['centroid'])
        next_state['centroid'] = centroid
    # The same physical pixel set is used on both frames.
    distance_delta = float(np.median(difference[common]))
    previous_distance = float(np.median(previous.meters[common]))
    metrics['distance_change_ratio'] = abs(distance_delta) / previous_distance
    metrics['longitudinal_velocity'] = metrics['distance_change_ratio'] / elapsed
    metrics['velocity_m_s'] = distance_delta / elapsed
    metrics['temporal_valid'] = True
    metrics['usable_for_motion'] = True
    return metrics, next_state


@dataclass(frozen=True)
class PreparedPair:
    color: CaptureFrame
    depth: CaptureFrame
    metric_depth: MetricDepth
    color_time_s: float
    depth_time_s: float
    pair_delta_s: float
    timestamp_s: float
    depth_quality: dict[str, Any]
    clock_epoch: int = 0
    frame_max_age_s: float = 0.25
    timing_basis: str = 'software_host_receive_estimate'
    hardware_synchronized: bool = False
    exposure_alignment_verified: bool = False

    def algorithm_inputs(self, now_s: float | None = None) -> dict[str, Any]:
        """Validated kwargs for process_rgb_depth; caller still owns stream_state.

        This is a software-paired, unregistered input, not hardware synchronization.
        """
        now = time.monotonic() if now_s is None else now_s
        if not all(0 <= now - stamp <= self.frame_max_age_s
                   for stamp in (self.color_time_s, self.depth_time_s)):
            raise ValueError('配对数据已过期或时间无效')
        if not self.depth_quality['usable_for_motion']:
            raise ValueError('深度质量或时序未就绪，不能作为有效检测输入')
        return {'rgb_frame': self.color.image.copy(), 'depth_frame': self.metric_depth.meters.copy(),
                'timestamp_s': self.timestamp_s, 'depth_unit_m': 1.0}


class FramePreprocessor:
    """Single-consumer bounded matching. Hardware exposure offset remains unknown."""
    def __init__(self, config: ProcessingConfig) -> None:
        config.validate()
        self.config = config
        self.rejections: Counter[str] = Counter()
        self.session: str | None = None
        self._clock_epoch = 0
        self._retired: set[str] = set()
        self.reset()

    def reset(self) -> None:
        self._clock_epoch += 1
        if getattr(self, 'session', None):
            self._retired.add(self.session)
        self.session = None
        self._buffers: dict[str, list[tuple[float, CaptureFrame]]] = {'color': [], 'depth': []}
        self._last: dict[str, tuple[int, float, str]] = {}
        self._device_stamp: int | None = None
        self._offsets: list[float] = []
        self._offset: float | None = None
        self._depth_state: dict[str, Any] = {}
        self._pair_time: float | None = None
        self._poll_seen: dict[str, tuple[str, int]] = {}

    def _reject(self, reason: str) -> bool:
        self.rejections[reason] += 1
        return False

    def offer(self, frame: CaptureFrame, now_s: float | None = None) -> bool:
        now = time.monotonic() if now_s is None else now_s
        source = frame.source
        if source not in self._buffers:
            return self._reject('source')
        if type(frame.sequence) is not int or frame.sequence < 0 or not frame.device_id:
            return self._reject('frame_identity')
        if not math.isfinite(frame.received_monotonic_s) or not math.isfinite(now):
            return self._reject('timestamp')
        age = now - frame.received_monotonic_s
        if age < 0 or age > self.config.frame_max_age_s:
            return self._reject('stale_or_future')
        if not frame.session_id or frame.session_id in self._retired:
            return self._reject('retired_session')
        if self.session != frame.session_id:
            self.reset()
            self.session = frame.session_id
        previous = self._last.get(source)
        if previous and (frame.sequence <= previous[0] or frame.received_monotonic_s <= previous[1]):
            return self._reject('duplicate_or_out_of_order')
        if previous and frame.device_id != previous[2]:
            return self._reject('device_changed_without_session')
        if source == 'depth':
            if frame.timestamp_basis != 'device_us_and_host_receive_monotonic':
                return self._reject('clock_basis')
            if frame.pixel_format != 'depth_u16_mm' or frame.depth_unit_m != 0.001 or frame.image.dtype != np.uint16:
                return self._reject('depth_format_or_unit')
            if frame.image.ndim != 2 or not frame.image.size:
                return self._reject('depth_shape')
            stamp = frame.device_timestamp_us
            if type(stamp) is not int or stamp < 0 or (self._device_stamp is not None and stamp <= self._device_stamp):
                return self._reject('device_clock_regression')
            self._device_stamp = stamp
            offset = frame.received_monotonic_s - stamp / 1e6
            if self._offset is None:
                self._offsets.append(offset)
                if len(self._offsets) < self.config.clock_warmup_frames:
                    self._last[source] = (frame.sequence, frame.received_monotonic_s, frame.device_id)
                    return self._reject('clock_warmup')
                self._offset = min(self._offsets)
            mapped = stamp / 1e6 + self._offset
            residual = frame.received_monotonic_s - mapped
            if abs(residual) > self.config.clock_residual_limit_s:
                # Re-anchor only after discarding all queued/temporal history.
                self._buffers = {'color': [], 'depth': []}
                self._depth_state = {}
                self._clock_epoch += 1
                self._offset, self._offsets = None, []
                self._last[source] = (frame.sequence, frame.received_monotonic_s, frame.device_id)
                return self._reject('clock_residual')
            # A later read can have lower transport latency than warmup reads.
            # Do not treat ordinary receive jitter as a clock reset, nor emit a
            # timestamp later than receipt. This remains a software estimate.
            mapped = min(mapped, frame.received_monotonic_s)
        else:
            if frame.timestamp_basis != 'host_receive_monotonic_only':
                return self._reject('clock_basis')
            if frame.pixel_format != 'bgr8' or frame.image.dtype != np.uint8 or frame.image.ndim != 3 or frame.image.shape[2] != 3 or not frame.image.size:
                return self._reject('color_format')
            mapped = frame.received_monotonic_s - self.config.color_receive_offset_s
        self._last[source] = (frame.sequence, frame.received_monotonic_s, frame.device_id)
        if now - mapped > self.config.frame_max_age_s:
            return self._reject('mapped_stale')
        buffer = self._buffers[source]
        if len(buffer) >= self.config.sync_buffer_size:
            buffer.pop(0)
            self.rejections['buffer_overflow'] += 1
        buffer.append((mapped, replace(frame, image=frame.image.copy())))
        return True

    def next_pair(self, now_s: float | None = None) -> PreparedPair | None:
        now = time.monotonic() if now_s is None else now_s
        for source, buffer in self._buffers.items():
            fresh = [item for item in buffer if 0 <= now - item[0] <= self.config.frame_max_age_s]
            self.rejections['expired_pending'] += len(buffer) - len(fresh)
            self._buffers[source] = fresh
        color, depth = self._buffers['color'], self._buffers['depth']
        if not color or not depth:
            return None
        delta, ci, di = min((abs(c[0] - d[0]), i, j) for i, c in enumerate(color) for j, d in enumerate(depth))
        if delta > self.config.sync_tolerance_s:
            # A later frame of the lagging source might still arrive; bounded buffers
            # and expiration remove the unmatched frames rather than inventing a pair.
            self.rejections['outside_tolerance'] += 1
            return None
        ct, cf = color[ci]
        dt, df = depth[di]
        self.rejections['superseded_unpaired'] += ci + di
        del color[:ci + 1]
        del depth[:di + 1]
        if self._pair_time is not None and dt <= self._pair_time:
            self._reject('pair_time_regression')
            return None
        self._pair_time = dt
        metric = normalize_depth(df.image, df.depth_unit_m, self.config)
        quality, self._depth_state = depth_motion(metric, dt, self._depth_state, self.config)
        return PreparedPair(cf, df, metric, ct, dt, delta, dt, quality, self._clock_epoch,
                            self.config.frame_max_age_s)

    def poll(self, capture: Any) -> PreparedPair | None:
        status = capture.status()
        if status['state'] != 'running':
            self.reset()
            return None
        frames = capture.buffered_frames()
        if capture.status()['state'] != 'running':
            self.reset()
            return None
        for frame in sorted(frames, key=lambda f: f.received_monotonic_s):
            previous = self._poll_seen.get(frame.source)
            if previous and previous[0] == frame.session_id and frame.sequence <= previous[1]:
                continue
            self.offer(frame)
            self._poll_seen[frame.source] = (frame.session_id, frame.sequence)
        return self.next_pair()
