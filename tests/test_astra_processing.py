from dataclasses import replace
import numpy as np
import pytest

from camera_capture.manager import CaptureFrame
from camera_capture.processing import ProcessingConfig, FramePreprocessor, normalize_depth, depth_preview, depth_motion
from core.config import MultimodalConfig
from core.depth_pipeline import DepthPipeline


def frame(source, sequence, received, stamp=None, session='one', image=None):
    if image is None:
        image = np.full((10, 10), 1500, np.uint16) if source == 'depth' else np.zeros((10, 10, 3), np.uint8)
    return CaptureFrame(image, source, sequence, received, stamp,
                        'host_receive_monotonic_only' if source == 'color' else 'device_us_and_host_receive_monotonic',
                        'depth_u16_mm' if source == 'depth' else 'bgr8',
                        0.001 if source == 'depth' else None, source + '-device', session)


def ready():
    config = ProcessingConfig(clock_warmup_frames=2)
    proc = FramePreprocessor(config)
    assert not proc.offer(frame('depth', 1, 10, 0), 10)
    assert proc.offer(frame('depth', 2, 10.033, 33000), 10.033)
    return proc


def test_explicit_units_and_invalid_values():
    cfg = ProcessingConfig()
    raw = np.array([[0, 1500, 65535, 500, 6100]], np.uint16)
    depth = normalize_depth(raw, .001, cfg)
    assert depth.valid_mask.tolist() == [[False, True, False, False, False]]
    assert depth.meters[0, 1] == pytest.approx(1.5)
    floating = normalize_depth(np.array([[np.nan, np.inf, -1, 1.5]], np.float32), 1, cfg)
    assert floating.valid_ratio == .25


def test_metric_millimeters_and_meters_equivalent():
    cfg = ProcessingConfig()
    a = normalize_depth(np.full((4, 4), 1500, np.uint16), .001, cfg)
    b = normalize_depth(np.full((4, 4), 1.5, np.float32), 1, cfg)
    np.testing.assert_allclose(a.meters, b.meters)
    boundaries = normalize_depth(np.array([[600, 6000]], np.uint16), .001, cfg)
    assert boundaries.valid_mask.all()


@pytest.mark.parametrize('value', [0, -1, np.nan, np.inf])
def test_invalid_units_rejected(value):
    with pytest.raises(ValueError):
        normalize_depth(np.zeros((2, 2), np.uint16), value, ProcessingConfig())


def test_preview_does_not_rescale_when_outlier_changes():
    cfg = ProcessingConfig()
    a = np.full((10, 10), 1500, np.uint16)
    b = a.copy(); b[0, 0] = 5000
    pa = depth_preview(normalize_depth(a, .001, cfg), cfg)
    pb = depth_preview(normalize_depth(b, .001, cfg), cfg)
    np.testing.assert_array_equal(pa[1:, 1:], pb[1:, 1:])
    assert np.all(depth_preview(normalize_depth(np.zeros_like(a), .001, cfg), cfg) == 0)


def test_motion_uses_actual_elapsed_and_common_pixels():
    cfg = ProcessingConfig()
    a = normalize_depth(np.full((10, 10), 2000, np.uint16), .001, cfg)
    b = normalize_depth(np.full((10, 10), 2200, np.uint16), .001, cfg)
    _, state = depth_motion(a, 10, {}, cfg)
    fast, _ = depth_motion(b, 10.1, state, cfg)
    slow, _ = depth_motion(b, 10.5, state, cfg)
    assert fast['velocity_m_s'] == pytest.approx(5 * slow['velocity_m_s'])
    hole = np.full((10, 10), 2000, np.uint16); hole[:3] = 0
    unchanged, _ = depth_motion(normalize_depth(hole, .001, cfg), 10.2, state, cfg)
    assert unchanged['motion_ratio'] == 0


def test_gap_jump_and_backward_time_do_not_create_motion():
    cfg = ProcessingConfig()
    a = normalize_depth(np.full((10, 10), 2000, np.uint16), .001, cfg)
    b = normalize_depth(np.full((10, 10), 4500, np.uint16), .001, cfg)
    _, state = depth_motion(a, 1, {}, cfg)
    rejected, saved = depth_motion(b, .9, state, cfg)
    assert saved is state and not rejected['usable_for_motion']
    gap, _ = depth_motion(b, 3, state, cfg)
    assert gap['reset_reason'] == 'time_gap' and not gap['usable_for_motion']
    jump, saved = depth_motion(b, 1.1, state, cfg)
    assert jump['reset_reason'] == 'abrupt_depth_jump' and saved == {}


def test_pair_metadata_nearest_no_reuse():
    proc = ready()
    proc.offer(frame('color', 1, 10.020), 10.04)
    proc.offer(frame('color', 2, 10.032), 10.04)
    pair = proc.next_pair(10.04)
    assert pair.color.sequence == 2
    assert pair.pair_delta_s == pytest.approx(.001)
    assert not pair.hardware_synchronized and not pair.exposure_alignment_verified
    assert not pair.depth_quality['usable_for_motion']
    assert proc.next_pair(10.04) is None
    assert not proc.offer(frame('depth', 2, 10.033, 33000), 10.04)


def test_stale_future_and_tolerance_rejected():
    proc = ready()
    assert not proc.offer(frame('color', 1, 9), 10.04)
    assert not proc.offer(frame('color', 1, 11), 10.04)
    proc.offer(frame('color', 2, 10.12), 10.12)
    assert proc.next_pair(10.12) is None
    assert proc.next_pair(11) is None


def test_device_clock_regression_and_arrival_delay():
    proc = ready()
    assert not proc.offer(frame('depth', 3, 10.066, 10000), 10.066)
    assert not proc.offer(frame('depth', 4, 10.5, 66000), 10.5)
    assert proc.rejections['clock_residual'] == 1
    assert not proc._buffers['depth']


def test_session_change_clears_history_and_rejects_old_session():
    proc = ready()
    proc.offer(frame('color', 1, 10.04, session='two'), 10.04)
    assert proc.session == 'two' and not proc._buffers['depth']
    assert not proc.offer(frame('depth', 3, 10.066, 66000, 'one'), 10.066)


def test_buffer_is_bounded_and_owned():
    proc = ready()
    proc.config.sync_buffer_size = 2
    f = frame('color', 1, 10.034)
    proc.offer(f, 10.034); f.image[:] = 255
    assert not np.any(proc._buffers['color'][0][1].image)
    for n in range(2, 9):
        proc.offer(frame('color', n, 10.034 + n / 1000), 10.05)
    assert len(proc._buffers['color']) == 2


def test_wrong_unit_and_camera_change_rejected():
    proc = ready()
    assert not proc.offer(replace(frame('depth', 3, 10.066, 66000), depth_unit_m=1), 10.066)
    assert not proc.offer(replace(frame('depth', 4, 10.07, 70000), device_id='another'), 10.07)


def test_pipeline_float_requires_unit_and_invalid_input_not_safe():
    pipeline = DepthPipeline(MultimodalConfig())
    a = np.full((10, 10), 1.5, np.float32)
    with pytest.raises(ValueError):
        pipeline.process_frame(a, 0, {})
    first, state, _ = pipeline.process_frame(a, 0, {}, depth_unit_m=1)
    assert not first.available
    result, state, _ = pipeline.process_frame(a, .1, state, depth_unit_m=1)
    assert result.available and result.diagnostics['median_depth'] == 1500
    invalid, _, _ = pipeline.process_frame(np.zeros((10, 10), np.uint16), .2, state)
    assert not invalid.available and invalid.state == 'UNAVAILABLE'


def test_small_uint16_never_misclassified_as_preview():
    result, _, _ = DepthPipeline(MultimodalConfig()).process_frame(np.full((10, 10), 200, np.uint16), 0, {})
    assert result.diagnostics['metric_depth_input'] and not result.available


def test_nan_config_rejected():
    config = MultimodalConfig()
    config.sync_tolerance_s = float('nan')
    with pytest.raises(ValueError):
        config.validate()


def test_algorithm_adapter_checks_readiness_and_expiration():
    proc = ready()
    proc.offer(frame('color', 1, 10.034), 10.034)
    first = proc.next_pair(10.034)
    with pytest.raises(ValueError):
        first.algorithm_inputs(10.034)
    proc.offer(frame('depth', 3, 10.066, 66000), 10.066)
    proc.offer(frame('color', 2, 10.067), 10.067)
    pair = proc.next_pair(10.067)
    arguments = pair.algorithm_inputs(10.07)
    assert arguments['depth_unit_m'] == 1
    np.testing.assert_allclose(arguments['depth_frame'], 1.5, rtol=1e-6)
    with pytest.raises(ValueError):
        pair.algorithm_inputs(11)


def test_normal_negative_clock_residual_not_repeated_warmup():
    proc = FramePreprocessor(ProcessingConfig(clock_warmup_frames=2))
    proc.offer(frame('depth', 1, 10.020, 0), 10.020)
    proc.offer(frame('depth', 2, 10.053, 33000), 10.053)
    assert proc.offer(frame('depth', 3, 10.066, 66000), 10.066)
    proc.offer(frame('color', 1, 10.067), 10.067)
    pair = proc.next_pair(10.067)
    assert pair.depth_time_s <= pair.depth.received_monotonic_s
    assert proc.rejections['clock_residual'] == 0
