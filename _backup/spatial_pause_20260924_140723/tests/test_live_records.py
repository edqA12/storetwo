import json
import sqlite3
from types import SimpleNamespace

import numpy as np

from services.database import EventRepository, EVENT_COLUMNS, PREFALL_COLUMNS
from services.events import EventService
from services.live_records import capture_context
from core.config import AppConfig
from core.pipeline import VisionPipeline, new_stream_state
from test_pipeline_sessions import FakeDetector


CONTEXT = {'device': 'Astra Pro', 'device_ids': {'depth': 'test'}, 'modalities_used': ['vision'],
           'quality': {'vision': .9, 'depth': 0}, 'fallback_reason': '深度失步'}


def test_all_record_kinds_persist_provenance_and_exports(tmp_path):
    repo = EventRepository(tmp_path / 'events.db')
    event = {'event_key': 'one', 'capture_context': CONTEXT}
    repo.add_event(event)
    repo.add_pre_fall_record({'source_time_s': 1, 'capture_context': CONTEXT}, '相机', 'Astra', 'one')
    repo.add_near_fall_event(event, '相机')
    repo.add_sit_to_stand_event(event, '相机')
    with sqlite3.connect(repo.database_path) as connection:
        for table in ['events', 'pre_fall_risk_records', 'near_fall_events', 'sit_to_stand_events']:
            value = connection.execute(f'SELECT capture_context FROM {table}').fetchone()[0]
            assert json.loads(value) == CONTEXT
    assert len(repo.table_rows()[0]) == len(EVENT_COLUMNS)
    assert len(repo.pre_fall_table_rows()[0]) == len(PREFALL_COLUMNS)
    assert repo.table_rows()[0][-4:] == ['Astra Pro', 'RGB', 'RGB 0.90 / 深度 0.00', '深度失步']
    assert '深度失步' in repo.export_csv(tmp_path / 'event.csv').read_text(encoding='utf-8-sig')


def test_duplicate_event_keeps_first_provenance_snapshot_and_confirmation(tmp_path):
    repo = EventRepository(tmp_path / 'events.db')
    service = EventService(repo, tmp_path / 'snapshots')
    event = {'event_key': 'one', 'capture_context': CONTEXT, 'modalities': [{'modality': 'depth', 'quality': .7}]}
    event_id = service.record(event, np.zeros((20, 20, 3), np.uint8), '相机')
    first_snapshot = (tmp_path / 'snapshots/one.jpg').read_bytes()
    repo.update_status(event_id, '已确认')
    event['capture_context'] = {'device': 'changed'}
    event['modalities'][0]['quality'] = .2
    assert service.record(event, np.full((20, 20, 3), 255, np.uint8), 'changed') == event_id
    assert repo.add_event(event) == event_id
    saved = repo.get_event(event_id)
    assert json.loads(saved['capture_context']) == CONTEXT
    assert saved['status'] == '已确认' and saved['modalities'][0]['quality'] == .7
    assert (tmp_path / 'snapshots/one.jpg').read_bytes() == first_snapshot
    assert len(repo.list_events()) == 1


def test_old_database_migration_is_repeatable_and_does_not_guess_source(tmp_path):
    repo = EventRepository(tmp_path / 'old.db')
    repo.add_event({'event_key': 'old'})
    with sqlite3.connect(repo.database_path) as connection:
        for table in ['events', 'pre_fall_risk_records', 'near_fall_events', 'sit_to_stand_events']:
            connection.execute(f'ALTER TABLE {table} DROP COLUMN capture_context')
    repo.initialize(); repo.initialize()
    assert repo.table_rows()[0][-4:] == ['未记录'] * 4


def test_valid_profile_update_is_deferred_until_frame_is_accepted(tmp_path):
    config = AppConfig(project_root=tmp_path)
    config.pre_fall.profile_save_interval_s = 0
    repo = EventRepository(tmp_path / 'baseline.db')
    pipeline = VisionPipeline(config, FakeDetector(), repo)
    state = new_stream_state()
    state.update(defer_profile_save=True, require_single_person_profile=True)
    _, next_state = pipeline.process_frame(np.zeros((160, 120, 3), np.uint8), 1, state)
    assert repo.get_personal_baseline('primary') is None
    pipeline.commit_profiles(next_state, 1)
    assert repo.get_personal_baseline('primary')['baseline_samples'] >= 1


def test_invalid_or_multiple_people_do_not_update_profile(tmp_path, monkeypatch):
    config = AppConfig(project_root=tmp_path)
    repo = EventRepository(tmp_path / 'baseline.db')
    detector = FakeDetector()
    original = detector.detect
    pipeline = VisionPipeline(config, detector, repo)
    for mode in ['invalid', 'low_quality', 'multiple']:
        def detect(frame, confidence=None):
            poses = original(frame, confidence)
            if mode == 'invalid':
                poses[0].keypoint_confidence[:] = 0
                return poses
            if mode == 'low_quality':
                poses[0].keypoint_confidence[:] = .3
                return poses
            return poses + original(frame, confidence)
        monkeypatch.setattr(detector, 'detect', detect)
        state = new_stream_state()
        state['require_single_person_profile'] = True
        result, updated = pipeline.process_frame(np.zeros((160, 120, 3), np.uint8), 1, state)
        assert not result.pre_fall_results and not updated['profile_update_tracks']
        assert repo.get_personal_baseline('primary') is None


def test_only_available_weighted_modalities_are_recorded():
    result = SimpleNamespace(diagnostics={'fallback_reason': '失步'}, decisions=[1],
        modalities=[SimpleNamespace(modality='vision', quality=.9, available=True),
                    SimpleNamespace(modality='depth', quality=.7, available=True)],
        fusion=SimpleNamespace(modality_weights={'vision': 1, 'depth': 0}))
    assert capture_context(result, '相机', 'Astra')['modalities_used'] == ['vision']
