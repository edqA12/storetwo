from __future__ import annotations

import numpy as np

from core.config import AppConfig
from core.pipeline import VisionPipeline, new_stream_state
from core.types import PersonPose
from services.database import EventRepository


class FakeDetector:
    last_inference_ms = 1.0

    def detect(self, frame_bgr, confidence=None):
        points = np.zeros((17, 2), dtype=np.float32)
        conf = np.zeros(17, dtype=np.float32)
        points[5], points[6] = (42, 28), (58, 28)
        points[11], points[12] = (44, 65), (56, 65)
        conf[[5, 6, 11, 12]] = 0.95
        return [PersonPose(0, (30, 10, 70, 105), 0.9, points, conf)]


def test_two_stream_states_do_not_share_history(tmp_path):
    config = AppConfig(project_root=tmp_path)
    pipeline = VisionPipeline(config, FakeDetector())
    frame = np.zeros((160, 120, 3), dtype=np.uint8)
    state_a = new_stream_state()
    state_b = new_stream_state()
    _, updated_a = pipeline.process_frame(frame, 1.0, state_a)
    _, updated_b = pipeline.process_frame(frame, 5.0, state_b)
    assert updated_a["session_id"] != updated_b["session_id"]
    assert updated_a["tracks"]["1"]["history"][0]["ts"] == 1.0
    assert updated_b["tracks"]["1"]["history"][0]["ts"] == 5.0


def test_personal_baseline_is_loaded_in_a_new_monitoring_session(tmp_path):
    config = AppConfig(project_root=tmp_path)
    config.pre_fall.baseline_min_samples = 1
    config.pre_fall.profile_save_interval_s = 0.0
    repository = EventRepository(tmp_path / "events.db")
    pipeline = VisionPipeline(config, FakeDetector(), repository)
    frame = np.zeros((160, 120, 3), dtype=np.uint8)
    _, first_state = pipeline.process_frame(frame, 1.0, new_stream_state())
    first_profile = repository.get_personal_baseline(config.pre_fall.person_id)
    assert first_profile is not None
    assert int(first_profile["baseline_samples"]) >= 1
    _, second_state = pipeline.process_frame(frame, 2.0, new_stream_state())
    second_pre_fall = second_state["tracks"]["1"]["pre_fall"]
    assert int(second_pre_fall["baseline_samples"]) > int(first_profile["baseline_samples"])
