from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest

from camera_capture.manager import CaptureFrame
from camera_capture.processing import PreparedPair, ProcessingConfig, normalize_depth
from core.config import AppConfig
from core.depth_pipeline import DepthPipeline
from core.fusion import FusionEngine
from core.multimodal import MultimodalPipeline
from core.types import FrameResult, ModalityResult, Decision
from services.astra_live import AstraLiveMonitor


class FakeVision:
    def process_frame(self, image, timestamp_s, state, confidence=None):
        state = dict(state or {})
        state['count'] = state.get('count', 0) + 1
        modality = ModalityResult('vision', timestamp_s, 20, .9, 'NORMAL', '正常', available=True)
        decision = Decision(1, 'NORMAL', '正常', 20, False, None, [])
        return FrameResult(image, [decision], 20, '正常', [], {'persons': 1}, [modality]), state


class Capture:
    def __init__(self, config):
        self.session = 'capture1'; self.phase = 'running'; self.sequence = 0
        self.age = 0; self.has_color = True
    def connect(self): pass
    def start(self): self.phase = 'running'
    def stop(self): self.phase = 'connected'
    def close(self): self.phase = 'disconnected'
    def status(self): return {'state': self.phase, 'session_id': self.session}
    def latest(self, source):
        if not self.has_color: return None
        return CaptureFrame(np.zeros((80, 80, 3), np.uint8), 'color', self.sequence,
                            time.monotonic() - self.age, None, 'host_receive_monotonic_only',
                            'bgr8', None, 'uvc', self.session)


class Processor:
    pair = None
    def poll(self, capture): return self.pair


@pytest.fixture
def monitor(tmp_path):
    config = AppConfig(project_root=tmp_path)
    pipeline = MultimodalPipeline(config, FakeVision(), DepthPipeline(config.multimodal), FusionEngine(config.multimodal))
    value = AstraLiveMonitor(config, pipeline, Capture)
    value.start('owner')
    value.processor = Processor()
    yield value
    value.disconnect('owner')


def supply(monitor, metric=True, valid=True, epoch=1):
    monitor.capture.sequence += 1
    color = monitor.capture.latest('color')
    raw = np.full((80, 80), 1500, np.uint16)
    depth = CaptureFrame(raw, 'depth', color.sequence, color.received_monotonic_s, color.sequence * 33333,
                         'device_us_and_host_receive_monotonic', 'depth_u16_mm', .001, 'oni', color.session_id)
    monitor.processor.pair = PreparedPair(color, depth, normalize_depth(raw, .001, monitor.config.multimodal),
                                          color.received_monotonic_s, color.received_monotonic_s, 0,
                                          color.received_monotonic_s,
                                          {'available': valid, 'usable_for_motion': valid, 'reset_reason': None if valid else 'abrupt_depth_jump'}, epoch) if metric else None


def test_valid_depth_enters_actual_fusion(monitor):
    supply(monitor); first = monitor.tick('owner', .3)
    assert first.result.fusion.method == 'visual_fallback'
    supply(monitor); second = monitor.tick('owner', .3)
    assert second.result.fusion.method == 'quality_weighted_rules'
    assert second.result.fusion.modality_weights['depth'] > 0
    assert second.result.max_risk != 20
    assert second.result.diagnostics['metric_depth_input']
    assert 'body_3d' not in second.result.diagnostics
    assert not second.result.diagnostics['spatial_registration']


def test_live_monitor_does_not_construct_or_load_body_3d(tmp_path, monkeypatch):
    from core.spatial import BodySpatialAnalyzer

    def forbidden(*args, **kwargs):
        raise AssertionError('Formal live monitoring must not construct body 3D')

    monkeypatch.setattr(BodySpatialAnalyzer, '__init__', forbidden)
    config = AppConfig(project_root=tmp_path)
    pipeline = MultimodalPipeline(config, FakeVision(), DepthPipeline(config.multimodal), FusionEngine(config.multimodal))
    value = AstraLiveMonitor(config, pipeline, Capture)
    try:
        value.start('owner')
        value.processor = Processor()
        monkeypatch.setattr('services.astra_live.FramePreprocessor', lambda config: Processor())
        supply(value)
        assert value.tick('owner', .3).new_result
        value.capture.session = 'reconnected'
        assert value.tick('owner', .3).result is not None
        value.stop('owner')
        value.start('owner')
    finally:
        value.disconnect('owner')


@pytest.mark.parametrize('mode', ['missing', 'bad_quality', 'stale_pair'])
def test_rgb_fallback_reuses_visual_state(monitor, mode):
    supply(monitor); monitor.tick('owner', .3)
    supply(monitor); monitor.tick('owner', .3)
    count = monitor.state['vision']['count']
    supply(monitor, metric=mode != 'missing', valid=mode != 'bad_quality')
    if mode == 'stale_pair':
        object.__setattr__(monitor.processor.pair, 'depth_time_s', time.monotonic() - 10)
    update = monitor.tick('owner', .3)
    assert update.result.fusion.method == 'visual_fallback'
    assert update.result.fusion.modality_weights['depth'] == 0
    assert monitor.state['vision']['count'] == count + 1
    assert update.result.diagnostics['depth_degraded']


def test_rgb_loss_interrupts_not_zero_result(monitor):
    supply(monitor); monitor.tick('owner', .3)
    monitor.capture.has_color = False
    result = monitor.tick('owner', .3)
    assert result.result is None and '监测中断' in result.status
    assert not monitor.state['vision']['tracks']


def test_no_new_frame_does_not_infer_or_repeat_event(monitor):
    supply(monitor, metric=False)
    monitor.tick('owner', .3)
    count = monitor.state['vision']['count']
    monitor.processor.pair = None
    update = monitor.tick('owner', .3)
    assert not update.new_result
    assert monitor.state['vision']['count'] == count


def test_owner_cannot_read_or_stop_another_session(monitor):
    assert monitor.tick('other', .3).result is None
    monitor.stop('other')
    assert monitor.owner == 'owner'
    with pytest.raises(RuntimeError): monitor.start('other')


def test_restart_clears_temporal_state(monitor):
    supply(monitor); monitor.tick('owner', .3)
    old = monitor.state['session_id']
    monitor.stop('owner'); monitor.start('owner')
    assert monitor.state['session_id'] != old
    assert monitor._last_sequence == -1


def test_clock_epoch_change_resets_vision_and_depth(monitor):
    supply(monitor); monitor.tick('owner', .3)
    supply(monitor); monitor.tick('owner', .3)
    supply(monitor, epoch=2)
    update = monitor.tick('owner', .3)
    assert monitor.state['vision']['count'] == 1
    assert update.result.fusion.method == 'visual_fallback'


def test_degraded_capture_continues_rgb(monitor):
    monitor.capture.phase = 'degraded'
    supply(monitor, metric=False)
    result = monitor.tick('owner', .3)
    assert result.result is not None and 'RGB降级' in result.status


def test_failed_depth_cannot_reuse_buffered_pair(monitor):
    supply(monitor); monitor.tick('owner', .3)
    supply(monitor)
    monitor.capture.status = lambda: {'state': 'degraded', 'session_id': 'capture1', 'source_errors': {'depth': 'lost'}}
    result = monitor.tick('owner', .3)
    assert result.result.fusion.method == 'visual_fallback'
    assert result.result.fusion.modality_weights['depth'] == 0


def test_old_pair_uses_fresh_rgb_instead(monitor):
    supply(monitor)
    object.__setattr__(monitor.processor.pair.color, 'received_monotonic_s', time.monotonic() - 10)
    result = monitor.tick('owner', .3)
    assert result.new_result and result.result.fusion.method == 'visual_fallback'


def test_stop_keeps_connection_disconnect_releases(monitor):
    capture = monitor.capture
    monitor.stop('owner')
    assert monitor.capture is capture and not monitor.monitoring
    assert monitor.device_status('owner')['owned']
    assert monitor.tick('owner', .3).result is None
    monitor.disconnect('owner')
    assert monitor.capture is None and monitor.owner is None


def test_page_timeout_closes_stream_and_clears_state(monitor):
    supply(monitor); monitor.tick('owner', .3)
    assert monitor.check_page_timeout(monitor._last_heartbeat + monitor.config.astra.page_timeout_s + 1)
    assert monitor.capture is None and not monitor.monitoring
    assert monitor._last_sequence == -1


def test_another_page_can_explicitly_release_device(monitor):
    monitor.disconnect('other')
    assert monitor.owner == 'owner'
    monitor.disconnect('other', force=True)
    assert monitor.owner is None


def test_preview_is_separate_and_record_diagnostics_are_real(monitor):
    supply(monitor); monitor.tick('owner', .3)
    supply(monitor); update = monitor.tick('owner', .3)
    assert update.depth_bgr.shape == (80, 80, 3)
    assert update.result.annotated_bgr.shape == (80, 80, 3)
    assert update.result.diagnostics['depth_valid_ratio'] == 1
    assert update.result.diagnostics['fallback_reason'] == ''
    assert update.result.diagnostics['live_processing_fps'] > 0


def test_expired_inference_does_not_commit_profile(monitor):
    calls = []
    monitor.pipeline.vision_pipeline.commit_profiles = lambda *args: calls.append(args)
    monitor.config.astra.result_max_age_s = -1  # Inject a deadline already exceeded.
    supply(monitor)
    result = monitor.tick('owner', .3)
    assert result.result is None and not calls
    assert monitor._last_sequence == -1
