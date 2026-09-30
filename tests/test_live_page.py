from types import SimpleNamespace
import numpy as np
import pytest

from core.pipeline import new_stream_state
from core.types import FrameResult, Decision, ModalityResult
from services.database import EventRepository
from services.events import EventService


@pytest.fixture
def page(tmp_path, monkeypatch):
    import app
    repo = EventRepository(tmp_path / 'test.db')
    monkeypatch.setattr(app, 'REPOSITORY', repo)
    monkeypatch.setattr(app, 'EVENT_SERVICE', EventService(repo, tmp_path / 'snapshots'))
    return app


def result(people=1, available=True, events=None):
    return FrameResult(np.zeros((80, 80, 3), np.uint8),
        [Decision(1, 'NORMAL', '正常', 10)] if people else [], 10, '正常', events or [],
        {'persons': people, 'live_processing_fps': 4},
        [ModalityResult('vision', 1, 10, .9 if available else 0, 'NORMAL', '正常', available=available)])


def test_waiting_no_person_and_invalid_are_not_zero_results(page):
    waiting = page.process_camera_frame(None, None, .3)
    assert waiting[2:4] == (None, None) and '等待数据' in waiting[1]
    for people, available, text in [(0, False, '未检测到人'), (1, False, '质量不足')]:
        out = page._present_camera_result(result(people, available), new_stream_state())
        assert out[2] is None and text in out[1]
    assert page._present_camera_result(result(), new_stream_state())[2] == 10


def test_ui_deduplicates_events_and_preserves_manual_confirmation(page, monkeypatch):
    state = new_stream_state()
    event = {'event_key': 'same-frame-event', 'risk': 90, 'event_type': '跌倒告警'}
    first = page._present_camera_result(result(events=[event]), state, '实时深度相机', 'Astra Pro')
    current_id = first[8]['last_db_event_id']
    monitor = SimpleNamespace(owner='owner', state=state, _lock=__import__('threading').RLock())
    monkeypatch.setattr(page, 'ASTRA_MONITOR', monitor)
    response = page.mark_live_confirmed(state, 'Astra Pro RGB＋深度', SimpleNamespace(session_hash='owner'))
    assert '已确认' in response[0]
    second = page._present_camera_result(result(events=[event]), state, '实时深度相机', 'Astra Pro')
    assert second[7] is None  # No duplicate sound.
    assert len(page.REPOSITORY.list_events()) == 1
    assert page.REPOSITORY.get_event(current_id)['status'] == '已确认'
    assert '已确认' in monitor.state['last_alert_text']


def test_no_person_cannot_prepare_valid_ai_summary(page):
    out = page.prepare_ai_current('实时监测', '', None, None, 0, 0, 4, None, None)
    assert out[1] == '' and out[2] is False
