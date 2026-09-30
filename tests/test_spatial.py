from __future__ import annotations

import copy
import json
import numpy as np
import pytest

from core.spatial import (BodySpatialAnalyzer, Calibration, SpatialConfig, fit_floor,
                          project, unproject, spatial_markdown)


def payload():
    intr = {'matrix': [[100, 0, 50], [0, 100, 50], [0, 0, 1]],
            'distortion': [0, 0, 0, 0, 0], 'size': [100, 100]}
    uv = np.array([[x, y] for x in (20, 40, 60, 80) for y in (20, 50, 80)], float)
    z = np.linspace(1, 2, len(uv))
    points = [[x, 1, z] for x in np.linspace(-1, 1, 12) for z in np.linspace(1, 3, 12)]
    return dict(schema=1, convention='opencv_depth_to_color_m', depth=intr, color=copy.deepcopy(intr),
                rotation=np.eye(3).tolist(), translation_m=[0, 0, 0], device_ids={'depth': 'd', 'color': 'c'},
                fit_capture_id='fit', verification=dict(independent_capture=True, capture_id='heldout',
                depth_uvz_m=np.column_stack((uv, z)).tolist(), color_uv=uv.tolist()),
                floor=dict(operator_confirmed_floor=True, placement_id='fixed-camera',
                           points_color_m=points, validation_points_color_m=[[x, 1, z] for x in np.linspace(-.9, .9, 10) for z in np.linspace(1.1, 2.9, 10)]))


def analyzer():
    value = BodySpatialAnalyzer(SpatialConfig())
    value.calibration = Calibration(payload(), value.config)
    return value


def pose(y=70):
    points = np.zeros((17, 2))
    points[[5, 6, 12, 11]] = [[30, y-10], [70, y-10], [70, y+10], [30, y+10]]
    return {'1': {'keypoints': points.tolist(), 'confidence': [1] * 17}}


def update(value, timestamp, poses=None, depth=None, ids=None):
    return value.update(np.full((100, 100), 2.0) if depth is None else depth,
                        (100, 100, 3), pose() if poses is None else poses, timestamp,
                        {'depth': 'd', 'color': 'c'} if ids is None else ids)


def test_projection_distortion_round_trip():
    intr = payload()['depth']
    intr['distortion'] = [.1, -.03, .001, -.002, .001]
    uv = np.array([[20, 20], [40, 60], [80, 80]], float)
    assert np.allclose(project(unproject(uv, np.array([1, 2, 3]), intr), intr), uv, atol=1e-3)


@pytest.mark.parametrize('mutation', [
    lambda p: p.update(convention='millimeters'),
    lambda p: p.update(rotation=[[1, 1, 0], [0, 1, 0], [0, 0, 1]]),
    lambda p: p.update(translation_m=[10, 0, 0]),
    lambda p: p.update(device_ids={}),
    lambda p: p['verification'].update(independent_capture=False),
    lambda p: p['verification'].update(capture_id='fit'),
    lambda p: p['verification'].update(color_uv=(np.array(p['verification']['color_uv'])+10).tolist()),
    lambda p: p['verification'].update(depth_uvz_m=[[30, 30, 1]] * 12),
    lambda p: p['floor'].update(operator_confirmed_floor=False),
    lambda p: p['floor'].update(validation_points_color_m=[[0, 2, 2]] * 100),
])
def test_invalid_calibration_rejected(mutation):
    data = payload()
    mutation(data)
    with pytest.raises(ValueError):
        Calibration(data, SpatialConfig())


def test_registration_is_projection_not_resize():
    data = payload()
    data['translation_m'] = [.1, 0, 0]
    uvz = np.array(data['verification']['depth_uvz_m'])
    data['verification']['color_uv'] = project(unproject(uvz[:, :2], uvz[:, 2], data['depth']) + [.1, 0, 0], data['color']).tolist()
    cal = Calibration(data, SpatialConfig())
    depth = np.full((100, 100), np.nan)
    depth[50, 50] = 2
    xyz = cal.register(depth, (100, 100, 3))
    assert np.isfinite(xyz[50, 55]).all()
    assert np.isnan(xyz[50, 50]).all()
    with pytest.raises(ValueError, match='分辨率'):
        cal.register(depth, (50, 50, 3))


def test_occlusion_keeps_nearest_surface():
    cal = Calibration(payload(), SpatialConfig())
    # A very narrow color field mapping places the two depth rays at same pixel.
    cal.color = copy.deepcopy(cal.color)
    cal.color['matrix'][0][0] = 1
    depth = np.full((100, 100), np.nan)
    depth[50, 50], depth[50, 51] = 3, 1
    xyz = cal.register(depth, (100, 100, 3))
    assert xyz[50, 50, 2] == 1


def test_height_real_dt_low_duration_and_activity():
    value = analyzer()
    first = update(value, 10, pose(70))
    assert first['available'] and not first['fusion_applied']
    assert first['persons'][0]['torso_height_m'] == pytest.approx(.6, abs=.03)
    assert first['persons'][0]['descent_speed_m_s'] is None
    second = update(value, 10.5, pose(80))['persons'][0]
    assert second['descent_speed_m_s'] == pytest.approx(.4, abs=.03)
    third = update(value, 11, pose(80))['persons'][0]
    assert third['low_duration_s'] == pytest.approx(.5)
    assert third['low_activity_m_s'] == pytest.approx(0)
    assert '躯干离地' in spatial_markdown(first)
    late = update(value, 12, pose(80))['persons'][0]
    assert late['descent_speed_m_s'] is None and late['low_duration_s'] == 0


def test_multi_person_and_missing_depth_reset_history():
    value = analyzer()
    update(value, 10)
    multiple = {**pose(), '2': pose()['1']}
    assert '多人' in update(value, 10.2, multiple)['reason']
    assert value.previous is None
    assert update(value, 10.3)['persons'][0]['descent_speed_m_s'] is None
    assert not value.update(None, (100, 100, 3), pose(), 10.4, {'depth': 'd', 'color': 'c'})['available']
    assert value.previous is None


def test_bad_depth_quality_or_device_never_becomes_zero_height():
    value = analyzer()
    depth = np.full((100, 100), np.nan)
    assert not update(value, 1, depth=depth)['available']
    depth[:, ::2] = 1
    depth[:, 1::2] = 2
    assert '混合' in update(value, 2, depth=depth)['reason']
    assert '设备标识' in update(value, 3, ids={'depth': 'other', 'color': 'c'})['reason']


def test_repeated_time_track_change_and_unconfirmed_mount(tmp_path):
    value = analyzer()
    update(value, 1)
    assert '乱序' in update(value, 1)['reason']
    update(value, 2)
    changed = {'2': pose()['1']}
    assert update(value, 2.1, changed)['persons'][0]['descent_speed_m_s'] is None
    file = tmp_path / 'calibration.json'
    file.write_text(json.dumps(payload()), encoding='utf-8')
    value.load(file, False)
    assert not update(value, 3)['available']
    value.load(file, True)
    assert update(value, 4)['available']


def test_floor_rejects_wall_and_outliers():
    with pytest.raises(ValueError):
        fit_floor(np.array([[1, y, z] for y in np.linspace(-1, 1, 12) for z in np.linspace(1, 3, 12)]))
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError):
        fit_floor(rng.uniform(-1, 1, (300, 3)))
