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
    def start(self): pass
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
    value.stop('owner')


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
