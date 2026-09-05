from __future__ import annotations

from services.database import EventRepository


def test_event_crud_and_deduplication(tmp_path):
    repository = EventRepository(tmp_path / "events.db")
    payload = {
        "event_key": "unique-event",
        "source": "测试",
        "event_type": "跌倒告警",
        "risk": 91.2,
        "confidence": 0.88,
        "source_time_s": 2.3,
        "reasons": ["快速下降", "持续横卧"],
    }
    first_id = repository.add_event(payload)
    second_id = repository.add_event(payload)
    assert first_id == second_id
    assert len(repository.list_events()) == 1
    assert repository.update_status(first_id, "误报")
    assert repository.get_event(first_id)["status"] == "误报"
    exported = repository.export_csv(tmp_path / "events.csv")
    assert exported.exists()
    assert "跌倒告警" in exported.read_text(encoding="utf-8-sig")


def test_suspected_fall_event_type_is_preserved(tmp_path):
    repository = EventRepository(tmp_path / "events.db")
    event_id = repository.add_event({
        "event_key": "suspected-event",
        "source": "测试",
        "event_type": "疑似跌倒",
        "risk": 79.0,
        "confidence": 0.72,
        "source_time_s": 4.0,
        "reasons": ["身体持续接近水平"],
    })
    assert repository.get_event(event_id)["event_type"] == "疑似跌倒"
