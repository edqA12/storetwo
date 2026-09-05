from __future__ import annotations

import csv
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any


EVENT_COLUMNS = [
    "事件ID", "发生时间", "来源", "事件类型", "风险值", "置信度", "视频时间", "处理状态", "原因",
]


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
            connection.execute("CREATE INDEX IF NOT EXISTS idx_events_time ON events(occurred_at DESC)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_events_status ON events(status)")

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
            now,
            now,
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO events (
                    event_key, occurred_at, source, source_name, event_type, risk,
                    confidence, source_time_s, snapshot_path, reasons, status,
                    note, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            row = connection.execute("SELECT id FROM events WHERE event_key = ?", (values[0],)).fetchone()
            if row is None:
                raise RuntimeError("事件写入失败")
            return int(row["id"])

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
        return dict(row) if row else None

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

    def table_rows(self, limit: int = 200) -> list[list[Any]]:
        rows: list[list[Any]] = []
        for event in self.list_events(limit):
            source = event["source"]
            if event.get("source_name"):
                source = f"{source} · {event['source_name']}"
            source_time = "—" if event.get("source_time_s") is None else f"{float(event['source_time_s']):.1f}s"
            rows.append([
                event["id"], event["occurred_at"], source, event["event_type"],
                round(float(event["risk"]), 1), round(float(event["confidence"]), 2),
                source_time, event["status"], event["reasons"],
            ])
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
