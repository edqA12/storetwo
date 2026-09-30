"""Session-owned capture and inference with accepted-frame profile commits."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import threading
import time
from typing import Any
import numpy as np

from camera_capture import AstraCapture, CaptureConfig
from camera_capture.processing import FramePreprocessor, depth_preview, normalize_depth
from core.multimodal import new_multimodal_state
from core.types import FrameResult
from core.spatial import BodySpatialAnalyzer, SpatialConfig


@dataclass
class LiveUpdate:
    status: str
    result: FrameResult | None = None
    new_result: bool = False
    depth_bgr: np.ndarray | None = None
    details: dict[str, Any] = field(default_factory=dict)


class AstraLiveMonitor:
    def __init__(self, config: Any, pipeline: Any, capture_factory=AstraCapture) -> None:
        self.config, self.pipeline = config, pipeline
        self.capture_factory = capture_factory
        self.capture: Any = None
        self.owner: str | None = None
        self.monitoring = False
        self._last_heartbeat = 0.0
        self._watcher: threading.Thread | None = None
        self._last_result_wall: float | None = None
        self._lock = threading.RLock()
        self.processor = FramePreprocessor(config.multimodal)
        self.state = new_multimodal_state()
        self._last_sequence = -1
        self._last_rgb_time: float | None = None
        self._capture_session: str | None = None
        self._clock_epoch: int | None = None
        self._last_update = LiveUpdate('尚未开始Astra监测')
        self.spatial = BodySpatialAnalyzer(getattr(config, 'spatial', SpatialConfig()))

    def connect(self, owner: str) -> None:
        if not owner:
            raise ValueError('缺少监测页面会话标识')
        with self._lock:
            if self.owner is not None and self.owner != owner:
                raise RuntimeError('Astra正在由另一个页面监测，请先在原页面停止。')
            if self.capture is not None:
                self._last_heartbeat = time.monotonic()
                return
            capture = self.capture_factory(CaptureConfig(Path(self.config.astra.sdk_dir), buffer_size=8))
            try:
                capture.connect()
            except Exception:
                capture.close()
                raise
            self.capture, self.owner = capture, owner
            self._last_heartbeat = time.monotonic()
            if self._watcher is None or not self._watcher.is_alive():
                self._watcher = threading.Thread(target=self._watch_page, daemon=True, name='astra-page-lease')
                self._watcher.start()

    def start(self, owner: str, mount_confirmed: bool = False) -> None:
        with self._lock:
            self.connect(owner)
            if self.monitoring:
                return
            self.capture.start()
            self.monitoring = True
            self.processor = FramePreprocessor(self.config.multimodal)
            self._reset()
            self.spatial.load(Path(self.config.project_root) / self.spatial.config.calibration_file, mount_confirmed)
            self._capture_session = self.capture.status()['session_id']
            self._clock_epoch = None
            self._last_heartbeat = time.monotonic()

    def _reset(self) -> None:
        self.spatial.previous = None
        self.state = new_multimodal_state()
        self.state['vision'].update(defer_profile_save=True, require_single_person_profile=True)
        self._last_result_wall = None
        self._last_sequence = -1
        self._last_rgb_time = None
        self._last_update = LiveUpdate('正在准备监测；深度时序尚未就绪')

    def stop(self, owner: str) -> None:
        with self._lock:
            if owner != self.owner or self.capture is None:
                return
            self.capture.stop()
            self.monitoring = False
            self.processor = FramePreprocessor(self.config.multimodal)
            self._reset()
            self._last_heartbeat = time.monotonic()

    def disconnect(self, owner: str, *, force: bool = False) -> None:
        with self._lock:
            if not force and self.owner is not None and self.owner != owner:
                return
            if self.capture is not None:
                self.capture.close()
            self.capture, self.owner = None, None
            self.monitoring = False
            self.processor = FramePreprocessor(self.config.multimodal)
            self._reset()
            self._last_update = LiveUpdate('Astra监测已停止')

    def check_page_timeout(self, now_s: float | None = None) -> bool:
        with self._lock:
            now = time.monotonic() if now_s is None else now_s
            if self.owner and now - self._last_heartbeat > self.config.astra.page_timeout_s:
                self.disconnect(self.owner)
                return True
            return False

    def _watch_page(self) -> None:
        while True:
            time.sleep(1)
            with self._lock:
                if self.capture is None:
                    self._watcher = None
                    return
                try:
                    if self.check_page_timeout():
                        self._watcher = None
                        return
                except Exception:
                    continue  # Failed native shutdown retains ownership; retry safely.

    def device_status(self, owner: str) -> dict[str, Any]:
        with self._lock:
            return {'connected': self.capture is not None, 'owned': owner == self.owner and self.owner is not None,
                    'monitoring': self.monitoring, 'device': self.capture.status() if self.capture else {}}

    def tick(self, owner: str, confidence: float) -> LiveUpdate:
        with self._lock:
            if owner != self.owner or self.capture is None:
                return LiveUpdate('已断开：未连接、被其他页面占用或页面超时后已释放')
            self._last_heartbeat = time.monotonic()
            if not self.monitoring:
                return LiveUpdate('等待数据：设备已连接，尚未开始监测')
            capture_status = self.capture.status()
            if capture_status['session_id'] != self._capture_session:
                self._reset()
                self.spatial.calibration = None
                self.spatial.reason = '采集会话已变化，请停止后重新核对安装位置再开始三维分析'
                self.processor = FramePreprocessor(self.config.multimodal)
                self._capture_session = capture_status['session_id']
                self._clock_epoch = None
            if capture_status['state'] not in {'running', 'degraded'}:
                self._reset()
                return LiveUpdate('监测中断：采集不可用，请停止后重新开始')
            try:
                pair = self.processor.poll(self.capture)
                color = self.capture.latest('color')
                depth = self.capture.latest('depth')
            except Exception as exc:
                self._reset()
                return LiveUpdate(f'监测中断：{exc}')
            now = time.monotonic()
            preview, valid_ratio = None, None
            if depth is not None and depth.source == 'depth' and 0 <= now - depth.received_monotonic_s <= self.config.multimodal.frame_max_age_s:
                metric = normalize_depth(depth.image, depth.depth_unit_m, self.config.multimodal)
                preview, valid_ratio = depth_preview(metric, self.config.multimodal), metric.valid_ratio
            if color is None or not 0 <= now - color.received_monotonic_s <= self.config.multimodal.frame_max_age_s:
                self._reset()
                return LiveUpdate('监测中断：RGB画面不可用，当前无法完成检测', depth_bgr=preview)
            if pair is not None and self._clock_epoch != pair.clock_epoch:
                if self._clock_epoch is not None:
                    self._reset()
                self._clock_epoch = pair.clock_epoch
            use_pair = (pair is not None and pair.color.sequence > self._last_sequence
                        and 0 <= now - pair.color.received_monotonic_s <= self.config.multimodal.frame_max_age_s
                        and not capture_status.get('source_errors', {}).get('depth'))
            if use_pair:
                color = pair.color
                preview, valid_ratio = depth_preview(pair.metric_depth, self.config.multimodal), pair.metric_depth.valid_ratio
            if color.sequence <= self._last_sequence:
                return LiveUpdate(self._last_update.status, self._last_update.result, False, preview)
            if self._last_rgb_time is not None and color.received_monotonic_s - self._last_rgb_time > self.config.multimodal.raw_temporal_gap_s:
                self._reset()
            raw_depth = None
            reason = capture_status.get('source_errors', {}).get('depth') or '深度缺失、过期或未满足软件配对容差'
            if use_pair:
                fresh = all(0 <= now - ts <= self.config.multimodal.frame_max_age_s
                            for ts in (pair.color_time_s, pair.depth_time_s))
                reason = {'abrupt_depth_jump': '深度异常跳变', 'low_valid_ratio': '深度有效像素不足',
                          'temporal_gap': '深度帧间隔过大'}.get(pair.depth_quality['reset_reason'], pair.depth_quality['reset_reason']) or reason
                if fresh and pair.depth_quality['available']:
                    # Initial valid depth primes the core temporal baseline; the
                    # core reports unavailable until that baseline is ready.
                    raw_depth = pair.metric_depth.meters
                elif not fresh:
                    reason = '配对数据已过期'
            try:
                result, state = self.pipeline.process_rgb_depth(
                    color.image, raw_depth, color.received_monotonic_s, self.state,
                    confidence=confidence, depth_unit_m=1.0 if raw_depth is not None else None,
                    depth_timestamp_s=pair.depth_time_s if raw_depth is not None else None,
                    depth_unavailable_reason=str(reason), compose_depth=False)
            except Exception as exc:
                self._reset()
                return LiveUpdate(f'监测中断：检测未完成（{exc}）')
            accepted = time.monotonic()
            if accepted - color.received_monotonic_s > self.config.astra.result_max_age_s:
                self._reset()
                return LiveUpdate('等待数据：本次检测耗时过长，已丢弃过期结果', depth_bgr=preview)
            self.state = state
            poses = {str(d.track_id): state['vision'].get('tracks', {}).get(str(d.track_id), {}).get('last_pose')
                     for d in result.decisions}
            poses = {key: value for key, value in poses.items() if value is not None}
            # Count every current detection; absent stored poses must not turn a
            # multi-person scene into an apparently single-person assignment.
            if len(result.decisions) > 1:
                poses = {str(d.track_id): {} for d in result.decisions}
            spatial = self.spatial.update(raw_depth, color.image.shape, poses,
                                          color.received_monotonic_s, capture_status.get('device_ids', {}))
            accepted = time.monotonic()
            if accepted - color.received_monotonic_s > self.config.astra.result_max_age_s:
                self._reset()
                return LiveUpdate('等待数据：三维处理后结果已过期，未更新基线', depth_bgr=preview)
            if hasattr(self.pipeline.vision_pipeline, 'commit_profiles'):
                self.pipeline.vision_pipeline.commit_profiles(state['vision'], color.received_monotonic_s)
            self._last_sequence = color.sequence
            self._last_rgb_time = color.received_monotonic_s
            method = result.fusion.method if result.fusion else 'visual_fallback'
            label = 'RGB＋深度融合（软件配对）' if method != 'visual_fallback' else 'RGB降级：' + str(reason)
            if raw_depth is not None and method == 'visual_fallback':
                label = 'RGB降级：深度质量不足或正在预热'
            fallback = label.split('：', 1)[-1] if method == 'visual_fallback' else ''
            if not result.decisions:
                label = '未检测到人 · ' + label
            elif not any(m.modality == 'vision' and m.available and m.quality >= self.config.pre_fall.min_rgb_quality for m in result.modalities):
                label = '人体质量不足 · ' + label
            fps = 1 / (accepted - self._last_result_wall) if self._last_result_wall is not None and accepted > self._last_result_wall else None
            self._last_result_wall = accepted
            result.diagnostics.update({'capture_source': 'Astra Pro', 'capture_session': self._capture_session,
                                       'live_mode': method, 'software_pair_delta_s': pair.pair_delta_s if use_pair else None,
                                       'exposure_alignment_verified': False,
                                       'spatial_registration': spatial['registration_verified'], 'body_3d': spatial,
                                       'device_ids': capture_status.get('device_ids', {'color': color.device_id}),
                                       'depth_valid_ratio': valid_ratio, 'fallback_reason': fallback,
                                       'frame_sequence': color.sequence, 'frame_time_s': color.received_monotonic_s,
                                       'live_processing_fps': fps})
            self._last_update = LiveUpdate(label, result, True, preview)
            return self._last_update
