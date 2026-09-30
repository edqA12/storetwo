from __future__ import annotations

import csv
import json
import sqlite3
from .live_records import context_values
import threading
from datetime import datetime
from pathlib import Path
from typing import Any


EVENT_COLUMNS = [
    "事件ID", "发生时间", "来源", "事件类型", "跌倒事件证据分", "置信度", "视频时间", "处理状态", "原因", "融合方式",
]

PREFALL_COLUMNS = [
    "记录ID", "记录时间", "来源", "人员/Track ID", "运动异常", "稳定性异常",
    "近跌倒次数", "基线偏离", "前置风险分", "风险等级", "深度降级", "主要原因",
    "当前运动速度", "个人基线速度", "速度变化", "最近起身耗时", "起身状态", "起身次数",
    "监护对象", "短期风险", "中期风险", "长期风险", "深度空间异常", "深度参与融合",
]


EVENT_COLUMNS += ["检测设备", "所用模态", "模态质量", "回退原因"]
PREFALL_COLUMNS += ["检测设备", "所用模态", "模态质量", "回退原因"]


class EventRepository:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=20.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_key TEXT NOT NULL UNIQUE,
                    occurred_at TEXT NOT NULL,
                    source TEXT NOT NULL,
                    source_name TEXT NOT NULL DEFAULT '',
                    event_type TEXT NOT NULL,
                    risk REAL NOT NULL,
                    fall_event_score REAL,
                    confidence REAL NOT NULL,
                    source_time_s REAL,
                    snapshot_path TEXT,
                    reasons TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT '待处理',
                    note TEXT NOT NULL DEFAULT '',
                    fusion_method TEXT NOT NULL DEFAULT '',
                    fusion_explanation TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            existing_columns = {
                str(row["name"]) for row in connection.execute("PRAGMA table_info(events)").fetchall()
            }
            if "fusion_method" not in existing_columns:
                connection.execute("ALTER TABLE events ADD COLUMN fusion_method TEXT NOT NULL DEFAULT ''")
            if "fusion_explanation" not in existing_columns:
                connection.execute("ALTER TABLE events ADD COLUMN fusion_explanation TEXT NOT NULL DEFAULT ''")
            if "fall_event_score" not in existing_columns:
                connection.execute("ALTER TABLE events ADD COLUMN fall_event_score REAL")
            connection.execute(
                "UPDATE events SET fall_event_score = risk WHERE fall_event_score IS NULL"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS event_modalities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id INTEGER NOT NULL,
                    modality TEXT NOT NULL,
                    timestamp_s REAL,
                    risk REAL NOT NULL,
                    quality REAL NOT NULL,
                    state TEXT NOT NULL,
                    label TEXT NOT NULL DEFAULT '',
                    reasons TEXT NOT NULL DEFAULT '',
                    diagnostics TEXT NOT NULL DEFAULT '{}',
                    available INTEGER NOT NULL DEFAULT 1,
                    FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE,
                    UNIQUE(event_id, modality)
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_events_time ON events(occurred_at DESC)")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_time_id ON events(occurred_at DESC, id DESC)"
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_events_status ON events(status)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_event_modalities_event ON event_modalities(event_id)")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS pre_fall_risk_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recorded_at TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    source_name TEXT NOT NULL DEFAULT '',
                    source_time_s REAL,
                    track_id INTEGER NOT NULL,
                    movement_score REAL NOT NULL,
                    stability_score REAL NOT NULL,
                    near_fall_count INTEGER NOT NULL,
                    sit_to_stand_score REAL NOT NULL,
                    baseline_deviation REAL NOT NULL,
                    pre_fall_risk_score REAL NOT NULL,
                    pre_fall_risk_level TEXT NOT NULL,
                    risk_factors TEXT NOT NULL DEFAULT '[]',
                    rgb_quality REAL NOT NULL,
                    depth_quality REAL NOT NULL,
                    depth_degraded INTEGER NOT NULL DEFAULT 1,
                    level_changed INTEGER NOT NULL DEFAULT 0,
                    previous_level TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(session_id, source_time_s, track_id)
                )
                """
            )
            pre_fall_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(pre_fall_risk_records)").fetchall()
            }
            pre_fall_migrations = {
                "movement_speed": "REAL NOT NULL DEFAULT 0",
                "baseline_speed": "REAL",
                "speed_change_pct": "REAL NOT NULL DEFAULT 0",
                "sit_to_stand_duration_s": "REAL",
                "sit_to_stand_status": "TEXT NOT NULL DEFAULT '未检测'",
                "sit_to_stand_count": "INTEGER NOT NULL DEFAULT 0",
                "person_id": "TEXT NOT NULL DEFAULT 'primary'",
                "short_term_score": "REAL NOT NULL DEFAULT 0",
                "medium_term_score": "REAL NOT NULL DEFAULT 0",
                "long_term_score": "REAL NOT NULL DEFAULT 0",
                "depth_spatial_score": "REAL NOT NULL DEFAULT 0",
                "depth_fusion_applied": "INTEGER NOT NULL DEFAULT 0",
            }
            for column, definition in pre_fall_migrations.items():
                if column not in pre_fall_columns:
                    connection.execute(
                        f"ALTER TABLE pre_fall_risk_records ADD COLUMN {column} {definition}"
                    )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS personal_baselines (
                    person_id TEXT PRIMARY KEY,
                    profile_json TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS near_fall_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_key TEXT NOT NULL UNIQUE,
                    occurred_at TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    source_name TEXT NOT NULL DEFAULT '',
                    source_time_s REAL,
                    track_id INTEGER NOT NULL,
                    max_torso_angle REAL NOT NULL,
                    max_down_velocity REAL NOT NULL,
                    lowest_center_y REAL NOT NULL,
                    recovery_time_s REAL NOT NULL,
                    rgb_quality REAL NOT NULL,
                    depth_quality REAL NOT NULL,
                    depth_degraded INTEGER NOT NULL DEFAULT 1,
                    developed_into_fall INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sit_to_stand_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_key TEXT NOT NULL UNIQUE,
                    occurred_at TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    source_name TEXT NOT NULL DEFAULT '',
                    source_time_s REAL,
                    track_id INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    succeeded INTEGER NOT NULL,
                    duration_s REAL NOT NULL,
                    max_stability_score REAL NOT NULL,
                    start_knee_angle_deg REAL NOT NULL,
                    max_knee_angle_deg REAL NOT NULL,
                    sit_to_stand_score REAL NOT NULL,
                    rgb_quality REAL NOT NULL,
                    depth_quality REAL NOT NULL,
                    depth_degraded INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_pre_fall_time ON pre_fall_risk_records(recorded_at DESC)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_pre_fall_time_id "
                "ON pre_fall_risk_records(recorded_at DESC, id DESC)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_near_fall_time ON near_fall_events(occurred_at DESC)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_sit_to_stand_time ON sit_to_stand_events(occurred_at DESC)"
            )

            for table in ('events', 'pre_fall_risk_records', 'near_fall_events', 'sit_to_stand_events'):
                columns = {row['name'] for row in connection.execute(f'PRAGMA table_info({table})')}
                if 'capture_context' not in columns:
                    connection.execute(f"ALTER TABLE {table} ADD COLUMN capture_context TEXT NOT NULL DEFAULT '{{}}'")

    def find_event_by_key(self, event_key: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute('SELECT * FROM events WHERE event_key = ?', (event_key,)).fetchone()
        return dict(row) if row else None

    def add_event(self, event: dict[str, Any]) -> int:
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        values = (
            str(event["event_key"]),
            str(event.get("occurred_at", now)),
            str(event.get("source", "未知")),
            str(event.get("source_name", "")),
            str(event.get("event_type", "跌倒告警")),
            float(event.get("risk", 0.0)),
            float(event.get("confidence", 0.0)),
            event.get("source_time_s"),
            str(event.get("snapshot_path", "")),
            "；".join(str(item) for item in event.get("reasons", [])),
            str(event.get("status", "待处理")),
            str(event.get("note", "")),
            str(event.get("fusion_method", "")),
            str(event.get("fusion_explanation", "")),
            now,
            now,
            json.dumps(event.get("capture_context", {}), ensure_ascii=False),
        )
        with self._lock, self._connect() as connection:
            inserted = connection.execute(
                """
                INSERT OR IGNORE INTO events (
                    event_key, occurred_at, source, source_name, event_type, risk,
                    confidence, source_time_s, snapshot_path, reasons, status,
                    note, fusion_method, fusion_explanation, created_at, updated_at, capture_context
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            row = connection.execute("SELECT id FROM events WHERE event_key = ?", (values[0],)).fetchone()
            if row is None:
                raise RuntimeError("事件写入失败")
            event_id = int(row["id"])
            if inserted.rowcount == 0:
                return event_id
            connection.execute(
                "UPDATE events SET fall_event_score = risk WHERE id = ? AND fall_event_score IS NULL",
                (event_id,),
            )
            modalities = event.get("modalities", [])
            if isinstance(modalities, list):
                for modality in modalities:
                    if not isinstance(modality, dict) or not modality.get("modality"):
                        continue
                    connection.execute(
                        """
                        INSERT INTO event_modalities (
                            event_id, modality, timestamp_s, risk, quality, state,
                            label, reasons, diagnostics, available
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(event_id, modality) DO UPDATE SET
                            timestamp_s=excluded.timestamp_s,
                            risk=excluded.risk,
                            quality=excluded.quality,
                            state=excluded.state,
                            label=excluded.label,
                            reasons=excluded.reasons,
                            diagnostics=excluded.diagnostics,
                            available=excluded.available
                        """,
                        (
                            event_id,
                            str(modality["modality"]),
                            modality.get("timestamp_s"),
                            float(modality.get("risk", 0.0)),
                            float(modality.get("quality", 0.0)),
                            str(modality.get("state", "UNKNOWN")),
                            str(modality.get("label", "")),
                            "；".join(str(item) for item in modality.get("reasons", [])),
                            json.dumps(modality.get("diagnostics", {}), ensure_ascii=False, default=str),
                            1 if modality.get("available", True) else 0,
                        ),
                    )
            return event_id

    def list_events(self, limit: int = 200) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 5000))
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM events ORDER BY occurred_at DESC, id DESC LIMIT ?", (safe_limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def get_event(self, event_id: int) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT * FROM events WHERE id = ?", (int(event_id),)).fetchone()
        if row is None:
            return None
        event = dict(row)
        event["modalities"] = self.list_event_modalities(int(event_id))
        return event

    def list_event_modalities(self, event_id: int) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM event_modalities WHERE event_id = ? ORDER BY id", (int(event_id),)
            ).fetchall()
        return [dict(row) for row in rows]

    def update_status(self, event_id: int, status: str, note: str = "") -> bool:
        allowed = {"待处理", "已确认", "误报", "已处理"}
        if status not in allowed:
            raise ValueError(f"不支持的处理状态：{status}")
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE events SET status = ?, note = ?, updated_at = ? WHERE id = ?",
                (status, note, now, int(event_id)),
            )
            return cursor.rowcount > 0

    def add_pre_fall_record(
        self,
        record: dict[str, Any],
        source: str,
        source_name: str = "",
        session_id: str = "",
    ) -> int:
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        factors = record.get("risk_factors", [])
        if not isinstance(factors, list):
            factors = [str(factors)]
        values = (
            str(record.get("recorded_at", now)),
            str(session_id or record.get("session_id", "unknown")),
            str(source),
            str(source_name),
            record.get("source_time_s", record.get("timestamp_s")),
            int(record.get("track_id", 0)),
            float(record.get("movement_score", 0.0)),
            float(record.get("stability_score", 0.0)),
            int(record.get("near_fall_count", 0)),
            float(record.get("sit_to_stand_score", 0.0)),
            float(record.get("baseline_deviation", 0.0)),
            float(record.get("pre_fall_risk_score", 0.0)),
            str(record.get("pre_fall_risk_level", "LOW")),
            json.dumps(factors, ensure_ascii=False),
            float(record.get("rgb_quality", 0.0)),
            float(record.get("depth_quality", 0.0)),
            1 if record.get("depth_degraded", True) else 0,
            1 if record.get("level_changed", False) else 0,
            record.get("previous_level"),
            float(record.get("movement_speed", 0.0)),
            (
                float(record["baseline_speed"])
                if record.get("baseline_speed") is not None else None
            ),
            float(record.get("speed_change_pct", 0.0)),
            (
                float(record["sit_to_stand_duration_s"])
                if record.get("sit_to_stand_duration_s") is not None else None
            ),
            str(record.get("sit_to_stand_status", "未检测")),
            int(record.get("sit_to_stand_count", 0)),
            str(record.get("person_id", "primary")),
            float(record.get("short_term_score", 0.0)),
            float(record.get("medium_term_score", 0.0)),
            float(record.get("long_term_score", 0.0)),
            float(record.get("depth_spatial_score", 0.0)),
            1 if record.get("depth_fusion_applied", False) else 0,
            now,
            json.dumps(record.get("capture_context", {}), ensure_ascii=False),
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO pre_fall_risk_records (
                    recorded_at, session_id, source, source_name, source_time_s,
                    track_id, movement_score, stability_score, near_fall_count,
                    sit_to_stand_score, baseline_deviation, pre_fall_risk_score,
                    pre_fall_risk_level, risk_factors, rgb_quality, depth_quality,
                    depth_degraded, level_changed, previous_level, movement_speed,
                    baseline_speed, speed_change_pct, sit_to_stand_duration_s,
                    sit_to_stand_status, sit_to_stand_count, person_id,
                    short_term_score, medium_term_score, long_term_score,
                    depth_spatial_score, depth_fusion_applied, created_at, capture_context
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            row = connection.execute(
                """
                SELECT id FROM pre_fall_risk_records
                WHERE session_id = ? AND source_time_s IS ? AND track_id = ?
                ORDER BY id DESC LIMIT 1
                """,
                (values[1], values[4], values[5]),
            ).fetchone()
            if row is None:
                raise RuntimeError("前置风险记录写入失败")
            return int(row["id"])

    def get_personal_baseline(self, person_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT profile_json FROM personal_baselines WHERE person_id = ?",
                (str(person_id),),
            ).fetchone()
        if row is None:
            return None
        try:
            value = json.loads(str(row["profile_json"]))
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None

    def upsert_personal_baseline(self, person_id: str, profile: dict[str, Any]) -> None:
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        payload = json.dumps(profile, ensure_ascii=False, default=str)
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO personal_baselines (person_id, profile_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(person_id) DO UPDATE SET
                    profile_json=excluded.profile_json,
                    updated_at=excluded.updated_at
                """,
                (str(person_id), payload, now),
            )

    def add_sit_to_stand_event(
        self,
        event: dict[str, Any],
        source: str,
        source_name: str = "",
        session_id: str = "",
    ) -> int:
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        values = (
            str(event.get("event_key") or __import__("uuid").uuid4().hex),
            str(event.get("occurred_at", now)),
            str(session_id or event.get("session_id", "unknown")),
            str(source),
            str(source_name),
            event.get("source_time_s", event.get("timestamp_s")),
            int(event.get("track_id", 0)),
            str(event.get("status", "未知")),
            1 if event.get("succeeded", False) else 0,
            float(event.get("duration_s", 0.0)),
            float(event.get("max_stability_score", 0.0)),
            float(event.get("start_knee_angle_deg", 0.0)),
            float(event.get("max_knee_angle_deg", 0.0)),
            float(event.get("sit_to_stand_score", 0.0)),
            float(event.get("rgb_quality", 0.0)),
            float(event.get("depth_quality", 0.0)),
            1 if event.get("depth_degraded", True) else 0,
            now,
            json.dumps(event.get("capture_context", {}), ensure_ascii=False),
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO sit_to_stand_events (
                    event_key, occurred_at, session_id, source, source_name,
                    source_time_s, track_id, status, succeeded, duration_s,
                    max_stability_score, start_knee_angle_deg, max_knee_angle_deg,
                    sit_to_stand_score, rgb_quality, depth_quality, depth_degraded, created_at, capture_context
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            row = connection.execute(
                "SELECT id FROM sit_to_stand_events WHERE event_key = ?", (values[0],)
            ).fetchone()
            if row is None:
                raise RuntimeError("起身事件写入失败")
            return int(row["id"])

    def add_near_fall_event(
        self,
        event: dict[str, Any],
        source: str,
        source_name: str = "",
        session_id: str = "",
    ) -> int:
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        values = (
            str(event.get("event_key") or __import__("uuid").uuid4().hex),
            str(event.get("occurred_at", now)),
            str(session_id or event.get("session_id", "unknown")),
            str(source),
            str(source_name),
            event.get("source_time_s", event.get("timestamp_s")),
            int(event.get("track_id", 0)),
            float(event.get("max_torso_angle", 0.0)),
            float(event.get("max_down_velocity", 0.0)),
            float(event.get("lowest_center_y", 0.0)),
            float(event.get("recovery_time_s", 0.0)),
            float(event.get("rgb_quality", 0.0)),
            float(event.get("depth_quality", 0.0)),
            1 if event.get("depth_degraded", True) else 0,
            1 if event.get("developed_into_fall", False) else 0,
            now,
            json.dumps(event.get("capture_context", {}), ensure_ascii=False),
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO near_fall_events (
                    event_key, occurred_at, session_id, source, source_name,
                    source_time_s, track_id, max_torso_angle, max_down_velocity,
                    lowest_center_y, recovery_time_s, rgb_quality, depth_quality,
                    depth_degraded, developed_into_fall, created_at, capture_context
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            row = connection.execute(
                "SELECT id FROM near_fall_events WHERE event_key = ?",
                (values[0],),
            ).fetchone()
            if row is None:
                raise RuntimeError("近跌倒事件写入失败")
            return int(row["id"])

    def list_pre_fall_records(self, limit: int = 500) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 5000))
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM pre_fall_risk_records ORDER BY recorded_at DESC, id DESC LIMIT ?",
                (safe_limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def pre_fall_table_rows(self, limit: int = 500) -> list[list[Any]]:
        return self.pre_fall_rows_from_records(self.list_pre_fall_records(limit))

    @staticmethod
    def pre_fall_rows_from_records(records: list[dict[str, Any]]) -> list[list[Any]]:
        """Format an existing history snapshot without querying it a second time."""
        level_labels = {"LOW": "低风险", "MEDIUM": "中风险", "HIGH": "高风险"}
        rows: list[list[Any]] = []
        for item in records:
            source = item["source"]
            if item.get("source_name"):
                source = f"{source} · {item['source_name']}"
            try:
                factors = json.loads(item.get("risk_factors") or "[]")
            except json.JSONDecodeError:
                factors = [str(item.get("risk_factors") or "")]
            rows.append([
                item["id"], item["recorded_at"], source, item["track_id"],
                round(float(item["movement_score"]), 1),
                round(float(item["stability_score"]), 1),
                int(item["near_fall_count"]),
                round(float(item["baseline_deviation"]), 1),
                round(float(item["pre_fall_risk_score"]), 1),
                level_labels.get(str(item["pre_fall_risk_level"]), item["pre_fall_risk_level"]),
                "是" if item["depth_degraded"] else "否",
                "；".join(str(value) for value in factors),
                round(float(item.get("movement_speed", 0.0)), 4),
                (
                    round(float(item["baseline_speed"]), 4)
                    if item.get("baseline_speed") is not None else "—"
                ),
                f"{float(item.get('speed_change_pct', 0.0)):.1f}%",
                (
                    round(float(item["sit_to_stand_duration_s"]), 2)
                    if item.get("sit_to_stand_duration_s") is not None else "—"
                ),
                str(item.get("sit_to_stand_status", "未检测")),
                int(item.get("sit_to_stand_count", 0)),
                str(item.get("person_id", "primary")),
                round(float(item.get("short_term_score", 0.0)), 1),
                round(float(item.get("medium_term_score", 0.0)), 1),
                round(float(item.get("long_term_score", 0.0)), 1),
                round(float(item.get("depth_spatial_score", 0.0)), 1),
                "是" if item.get("depth_fusion_applied") else "否",
            ] + context_values(item.get("capture_context")))
        return rows

    def table_rows(self, limit: int = 200) -> list[list[Any]]:
        rows: list[list[Any]] = []
        for event in self.list_events(limit):
            source = event["source"]
            if event.get("source_name"):
                source = f"{source} · {event['source_name']}"
            source_time = "—" if event.get("source_time_s") is None else f"{float(event['source_time_s']):.1f}s"
            rows.append([
                event["id"], event["occurred_at"], source, event["event_type"],
                round(float(event.get("fall_event_score") or event["risk"]), 1),
                round(float(event["confidence"]), 2),
                source_time, event["status"], event["reasons"], event.get("fusion_method") or "—",
            ] + context_values(event.get("capture_context")))
        return rows

    def statistics(self) -> dict[str, int]:
        with self._lock, self._connect() as connection:
            total = int(connection.execute("SELECT COUNT(*) FROM events").fetchone()[0])
            pending = int(connection.execute("SELECT COUNT(*) FROM events WHERE status = '待处理'").fetchone()[0])
            confirmed = int(connection.execute("SELECT COUNT(*) FROM events WHERE status IN ('已确认','已处理')").fetchone()[0])
            false_alarm = int(connection.execute("SELECT COUNT(*) FROM events WHERE status = '误报'").fetchone()[0])
        return {"总事件": total, "待处理": pending, "已确认或处理": confirmed, "误报": false_alarm}

    def export_csv(self, output_path: str | Path) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(EVENT_COLUMNS)
            writer.writerows(self.table_rows(limit=5000))
        return output

    def export_pre_fall_csv(self, output_path: str | Path) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(PREFALL_COLUMNS)
            writer.writerows(self.pre_fall_table_rows(limit=5000))
        return output
