from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.config import AppConfig
from core.types import FrameResult
from core.video_processor import VideoProcessor
from services.database import EventRepository
from services.events import EventService


class FakePipeline:
    def process_frame(self, frame, timestamp_s, state, confidence=None):
        annotated = frame.copy()
        cv2.putText(annotated, "OK", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return FrameResult(annotated, [], 0.0, "正常", [], {"persons": 0}), state

    def render_cached(self, frame, state):
        return frame


def _write_test_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (96, 64))
    assert writer.isOpened()
    for index in range(6):
        frame = np.full((64, 96, 3), index * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_video_roundtrip_with_fake_pipeline(tmp_path):
    config = AppConfig(project_root=tmp_path)
    config.ensure_directories()
    repository = EventRepository(config.resolve(config.storage.database))
    service = EventService(repository, config.resolve(config.storage.snapshots_dir))
    processor = VideoProcessor(config, FakePipeline(), service)
    source = tmp_path / "sample.avi"
    _write_test_video(source)
    result = processor.process(source, confidence=0.35)
    assert result.output_path.exists()
    assert result.event_count == 0
    capture = cv2.VideoCapture(str(result.output_path))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) >= 5
    capture.release()
