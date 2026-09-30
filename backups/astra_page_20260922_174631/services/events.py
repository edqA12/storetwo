from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .database import EventRepository


class EventService:
    def __init__(self, repository: EventRepository, snapshots_dir: str | Path) -> None:
        self.repository = repository
        self.snapshots_dir = Path(snapshots_dir)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        event: dict[str, Any],
        annotated_bgr: np.ndarray,
        source: str,
        source_name: str = "",
    ) -> int:
        event_key = str(event.get("event_key") or uuid.uuid4().hex)
        snapshot = self.snapshots_dir / f"{event_key}.jpg"
        snapshot_path = ""
        if annotated_bgr is not None and annotated_bgr.size:
            ok, encoded = cv2.imencode(".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ok:
                snapshot.write_bytes(encoded.tobytes())
                snapshot_path = str(snapshot)
        payload = dict(event)
        payload.update({
            "event_key": event_key,
            "occurred_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "source": source,
            "source_name": source_name,
            "event_type": str(event.get("event_type", "跌倒告警")),
            "snapshot_path": snapshot_path,
        })
        return self.repository.add_event(payload)
