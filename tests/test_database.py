from __future__ import annotations

import sqlite3

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


def test_event_stores_modality_results(tmp_path):
    repository = EventRepository(tmp_path / "events.db")
    event_id = repository.add_event({
        "event_key": "multimodal-event",
        "source": "测试",
        "event_type": "疑似跌倒",
        "risk": 79.0,
        "confidence": 0.81,
        "fusion_method": "quality_weighted_rules",
        "fusion_explanation": "视觉与深度共同支持",
        "modalities": [
            {"modality": "vision", "risk": 76.0, "quality": 0.9, "state": "SUSPECT", "label": "疑似"},
            {"modality": "depth", "risk": 70.0, "quality": 0.72, "state": "DEPTH_IMPACT", "label": "强变化"},
        ],
    })
    event = repository.get_event(event_id)
    assert event is not None
    assert event["fusion_method"] == "quality_weighted_rules"
    assert {item["modality"] for item in event["modalities"]} == {"vision", "depth"}


def test_existing_event_database_is_migrated_for_multimodal_fields(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_key TEXT NOT NULL UNIQUE,
                occurred_at TEXT NOT NULL,
                source TEXT NOT NULL,
                source_name TEXT NOT NULL DEFAULT '',
                event_type TEXT NOT NULL,
                risk REAL NOT NULL,
                confidence REAL NOT NULL,
                source_time_s REAL,
                snapshot_path TEXT,
                reasons TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT '待处理',
                note TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
    repository = EventRepository(path)
    event_id = repository.add_event({
        "event_key": "after-migration",
        "source": "测试",
        "risk": 50.0,
        "confidence": 0.5,
        "fusion_method": "visual_fallback",
    })
    assert repository.get_event(event_id)["fusion_method"] == "visual_fallback"


def test_pre_fall_and_near_fall_records_are_persisted(tmp_path):
    repository = EventRepository(tmp_path / "events.db")
    record_id = repository.add_pre_fall_record(
        {
            "timestamp_s": 12.5,
            "track_id": 3,
            "movement_score": 42.0,
            "stability_score": 55.0,
            "near_fall_count": 1,
            "sit_to_stand_score": 0.0,
            "baseline_deviation": 31.0,
            "pre_fall_risk_score": 47.0,
            "pre_fall_risk_level": "MEDIUM",
            "risk_factors": ["运动速度下降", "检测到近跌倒"],
            "rgb_quality": 0.9,
            "depth_quality": 0.0,
            "depth_degraded": True,
            "level_changed": True,
            "previous_level": "LOW",
        },
        "测试",
        "样例",
        "session-1",
    )
    near_fall_id = repository.add_near_fall_event(
        {
            "event_key": "near-fall-1",
            "timestamp_s": 12.5,
            "track_id": 3,
            "max_torso_angle": 38.0,
            "max_down_velocity": 0.24,
            "lowest_center_y": 0.68,
            "recovery_time_s": 0.9,
            "rgb_quality": 0.9,
            "depth_quality": 0.0,
            "depth_degraded": True,
            "developed_into_fall": False,
        },
        "测试",
        "样例",
        "session-1",
    )
    assert record_id > 0 and near_fall_id > 0
    rows = repository.pre_fall_table_rows()
    assert rows[0][8] == 47.0
    assert rows[0][9] == "中风险"
    exported = repository.export_pre_fall_csv(tmp_path / "pre_fall.csv")
    assert "前置风险分" in exported.read_text(encoding="utf-8-sig")


def test_sit_to_stand_event_and_movement_trend_fields_are_persisted(tmp_path):
    repository = EventRepository(tmp_path / "events.db")
    repository.add_pre_fall_record(
        {
            "timestamp_s": 2.0,
            "track_id": 1,
            "movement_speed": 0.042,
            "baseline_speed": 0.060,
            "speed_change_pct": -30.0,
            "sit_to_stand_duration_s": 2.4,
            "sit_to_stand_status": "成功",
            "sit_to_stand_count": 1,
        },
        "测试",
        session_id="session-sts",
    )
    event_id = repository.add_sit_to_stand_event(
        {
            "event_key": "sts-1",
            "timestamp_s": 2.0,
            "track_id": 1,
            "status": "成功",
            "succeeded": True,
            "duration_s": 2.4,
            "max_stability_score": 22.0,
            "start_knee_angle_deg": 102.0,
            "max_knee_angle_deg": 164.0,
            "sit_to_stand_score": 18.0,
            "rgb_quality": 0.9,
        },
        "测试",
        session_id="session-sts",
    )
    assert event_id > 0
    record = repository.list_pre_fall_records()[0]
    assert record["movement_speed"] == 0.042
    assert record["sit_to_stand_status"] == "成功"


def test_personal_baseline_profile_roundtrip(tmp_path):
    repository = EventRepository(tmp_path / "events.db")
    profile = {
        "baseline_speed": 0.072,
        "baseline_stability": 8.0,
        "baseline_samples": 42,
        "hourly_windows": {"2026-09-01T10:00Z": {"samples": 3}},
        "daily_windows": {},
    }
    repository.upsert_personal_baseline("elder-001", profile)
    assert repository.get_personal_baseline("elder-001") == profile
    assert repository.get_personal_baseline("missing") is None
