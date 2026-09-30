"""Session-owned live inference. No UI, persistence, or assistant calls here."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Any

from camera_capture import AstraCapture, CaptureConfig
from camera_capture.processing import FramePreprocessor
from core.multimodal import new_multimodal_state
from core.types import FrameResult


@dataclass
class LiveUpdate:
    status: str
    result: FrameResult | None = None
    new_result: bool = False


class AstraLiveMonitor:
    def __init__(self, config: Any, pipeline: Any, capture_factory=AstraCapture) -> None:
        self.config, self.pipeline = config, pipeline
        self.capture_factory = capture_factory
        self.capture: Any = None
        self.owner: str | None = None
        self._lock = threading.RLock()
        self.processor = FramePreprocessor(config.multimodal)
        self.state = new_multimodal_state()
        self._last_sequence = -1
        self._last_rgb_time: float | None = None
        self._capture_session: str | None = None
        self._clock_epoch: int | None = None
        self._last_update = LiveUpdate('尚未开始Astra监测')

    def start(self, owner: str) -> None:
        if not owner:
            raise ValueError('缺少监测页面会话标识')
        with self._lock:
            if self.owner is not None and self.owner != owner:
                raise RuntimeError('Astra正在由另一个页面监测，请先在原页面停止。')
            self.stop(owner)
            capture = self.capture_factory(CaptureConfig(Path(self.config.astra.sdk_dir), buffer_size=8))
            try:
                capture.connect()
                capture.start()
            except Exception:
                capture.close()
                raise
            self.capture, self.owner = capture, owner
            self.processor = FramePreprocessor(self.config.multimodal)
            self._reset()
            self._capture_session = capture.status()['session_id']
            self._clock_epoch = None

    def _reset(self) -> None:
        self.state = new_multimodal_state()
        self._last_sequence = -1
        self._last_rgb_time = None
        self._last_update = LiveUpdate('正在准备监测；深度时序尚未就绪')

    def stop(self, owner: str) -> None:
        with self._lock:
            if self.owner is not None and self.owner != owner:
                return
            if self.capture is not None:
                self.capture.close()
            self.capture, self.owner = None, None
            self.processor = FramePreprocessor(self.config.multimodal)
            self._reset()
            self._last_update = LiveUpdate('Astra监测已停止')

    def tick(self, owner: str, confidence: float) -> LiveUpdate:
        with self._lock:
            if owner != self.owner or self.capture is None:
                return LiveUpdate('尚未开始Astra监测（设备可能由其他页面使用）')
            capture_status = self.capture.status()
            if capture_status['session_id'] != self._capture_session:
                self._reset()
                self.processor = FramePreprocessor(self.config.multimodal)
                self._capture_session = capture_status['session_id']
                self._clock_epoch = None
            if capture_status['state'] not in {'running', 'degraded'}:
                self._reset()
                return LiveUpdate('监测中断：采集不可用，请停止后重新开始')
            try:
                pair = self.processor.poll(self.capture)
                color = self.capture.latest('color')
            except Exception as exc:
                self._reset()
                return LiveUpdate(f'监测中断：{exc}')
            now = time.monotonic()
            if color is None or now - color.received_monotonic_s > self.config.multimodal.frame_max_age_s:
                self._reset()
                return LiveUpdate('监测中断：RGB画面不可用，当前无法完成检测')
            if pair is not None and self._clock_epoch != pair.clock_epoch:
                if self._clock_epoch is not None:
                    self._reset()
                self._clock_epoch = pair.clock_epoch
            use_pair = (pair is not None and pair.color.sequence > self._last_sequence
                        and 0 <= now - pair.color.received_monotonic_s <= self.config.multimodal.frame_max_age_s
                        and not capture_status.get('source_errors', {}).get('depth'))
            if use_pair:
                color = pair.color
            if color.sequence <= self._last_sequence:
                return LiveUpdate(self._last_update.status, self._last_update.result, False)
            if self._last_rgb_time is not None and color.received_monotonic_s - self._last_rgb_time > self.config.multimodal.raw_temporal_gap_s:
                self._reset()
            raw_depth = None
            reason = '深度缺失、过期或未满足软件配对容差'
            if use_pair:
                fresh = all(0 <= now - ts <= self.config.multimodal.frame_max_age_s
                            for ts in (pair.color_time_s, pair.depth_time_s))
                reason = pair.depth_quality['reset_reason'] or reason
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
                    depth_unavailable_reason=str(reason))
            except Exception as exc:
                self._reset()
                return LiveUpdate(f'监测中断：检测未完成（{exc}）')
            if time.monotonic() - color.received_monotonic_s > self.config.astra.result_max_age_s:
                self._reset()
                return LiveUpdate('正在准备监测：本次检测耗时过长，已丢弃过期结果')
            self.state = state
            self._last_sequence = color.sequence
            self._last_rgb_time = color.received_monotonic_s
            method = result.fusion.method if result.fusion else 'visual_fallback'
            label = 'RGB＋深度融合（软件配对）' if method != 'visual_fallback' else 'RGB降级：' + str(reason)
            if raw_depth is not None and method == 'visual_fallback':
                label = 'RGB降级：深度质量不足或正在预热'
            result.diagnostics.update({'capture_source': 'Astra Pro', 'capture_session': self._capture_session,
                                       'live_mode': method, 'software_pair_delta_s': pair.pair_delta_s if use_pair else None,
                                       'exposure_alignment_verified': False, 'spatial_registration': False})
            self._last_update = LiveUpdate(label, result, True)
            return self._last_update
