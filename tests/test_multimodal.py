from __future__ import annotations

import numpy as np

from core.config import AppConfig
from core.depth_pipeline import DepthPipeline
from core.fusion import FusionEngine
from core.multimodal import MultimodalPipeline, new_multimodal_state, split_rgb_depth_frame
from core.types import FrameResult, ModalityResult


def modality(
    name: str,
    risk: float,
    quality: float,
    state: str,
    available: bool = True,
) -> ModalityResult:
    return ModalityResult(
        modality=name,
        timestamp_s=1.0,
        risk=risk,
        quality=quality,
        state=state,
        label=state,
        available=available,
    )


def _depth_frame(top: int) -> np.ndarray:
    frame = np.full((80, 80, 3), 30, dtype=np.uint8)
    frame[top:top + 20, 20:60] = 130
    return frame


def test_split_rgb_depth_frame_uses_configured_sides(tmp_path):
    config = AppConfig(project_root=tmp_path)
    frame = np.zeros((60, 320, 3), dtype=np.uint8)
    frame[:, :160] = 25
    frame[:, 160:] = 200
    depth, rgb = split_rgb_depth_frame(frame, config)
    assert depth.shape == rgb.shape == (60, 160, 3)
    assert int(depth.mean()) == 25
    assert int(rgb.mean()) == 200


def test_depth_pipeline_detects_downward_strong_change(tmp_path):
    config = AppConfig(project_root=tmp_path)
    pipeline = DepthPipeline(config.multimodal)
    state: dict = {}
    _, state, _ = pipeline.process_frame(_depth_frame(8), 0.0, state)
    _, state, _ = pipeline.process_frame(_depth_frame(12), 0.2, state)
    result, _, rendered = pipeline.process_frame(_depth_frame(42), 0.4, state)
    assert result.available
    assert result.risk >= 70.0
    assert result.state in {"DEPTH_IMPACT", "DEPTH_EVIDENCE"}
    assert rendered.shape == (80, 80, 3)
    assert result.diagnostics["pre_fall_spatial_score"] > 0.0
    assert "distance_change_ratio" in result.diagnostics


def test_depth_pipeline_uses_raw_metric_distance_change(tmp_path):
    config = AppConfig(project_root=tmp_path)
    pipeline = DepthPipeline(config.multimodal)
    state: dict = {}
    first = np.zeros((80, 80), dtype=np.uint16)
    second = np.zeros((80, 80), dtype=np.uint16)
    first[15:65, 20:40] = 1900
    first[15:65, 40:60] = 2100
    second[18:68, 20:40] = 2280
    second[18:68, 40:60] = 2520
    _, state, _ = pipeline.process_frame(first, 0.0, state)
    result, _, _ = pipeline.process_frame(second, 0.5, state)
    assert result.available
    assert result.diagnostics["metric_depth_input"] is True
    assert result.diagnostics["median_depth"] == 2400.0
    assert result.diagnostics["distance_change_ratio"] == 0.2
    assert result.diagnostics["longitudinal_velocity"] == 0.4


def test_fusion_confirms_supported_visual_fall(tmp_path):
    config = AppConfig(project_root=tmp_path)
    engine = FusionEngine(config.multimodal)
    fused = engine.fuse(
        modality("vision", 91.0, 0.9, "FALLEN"),
        modality("depth", 78.0, 0.8, "DEPTH_IMPACT"),
        1.0,
    )
    assert fused.state == "FALLEN"
    assert fused.risk >= 85.0
    assert fused.modality_weights["vision"] > fused.modality_weights["depth"]


def test_fusion_downgrades_conflicting_visual_fall(tmp_path):
    config = AppConfig(project_root=tmp_path)
    engine = FusionEngine(config.multimodal)
    fused = engine.fuse(
        modality("vision", 91.0, 0.9, "FALLEN"),
        modality("depth", 5.0, 0.9, "DEPTH_NORMAL"),
        1.0,
    )
    assert fused.state == "SUSPECT"
    assert fused.risk == 79.0


def test_fusion_falls_back_when_depth_is_unavailable(tmp_path):
    config = AppConfig(project_root=tmp_path)
    engine = FusionEngine(config.multimodal)
    visual = modality("vision", 35.0, 0.8, "NORMAL")
    fused = engine.fuse(visual, modality("depth", 0.0, 0.0, "UNAVAILABLE", False), 1.0)
    assert fused.method == "visual_fallback"
    assert fused.risk == visual.risk


def test_depth_only_change_does_not_confirm_when_visual_is_normal(tmp_path):
    config = AppConfig(project_root=tmp_path)
    engine = FusionEngine(config.multimodal)
    fused = engine.fuse(
        modality("vision", 20.0, 0.9, "NORMAL"),
        modality("depth", 90.0, 0.9, "DEPTH_IMPACT"),
        1.0,
    )
    assert fused.state == "DESCENDING"
    assert fused.risk < config.multimodal.fusion_alert_threshold


class FakeVisionPipeline:
    def process_frame(self, frame_bgr, timestamp_s, stream_state, confidence=None):
        visual = ModalityResult(
            modality="vision",
            timestamp_s=timestamp_s,
            risk=20.0,
            quality=0.9,
            state="NORMAL",
            label="正常",
            available=True,
        )
        return FrameResult(frame_bgr.copy(), [], 20.0, "正常", [], {"persons": 1}, [visual]), {"ok": True}

    def render_cached(self, frame_bgr, stream_state):
        return frame_bgr.copy()


def test_multimodal_pipeline_preserves_composite_dimensions(tmp_path):
    config = AppConfig(project_root=tmp_path)
    config.multimodal.min_half_width = 40
    depth_pipeline = DepthPipeline(config.multimodal)
    pipeline = MultimodalPipeline(
        config,
        FakeVisionPipeline(),  # type: ignore[arg-type]
        depth_pipeline,
        FusionEngine(config.multimodal),
    )
    composite = np.hstack([_depth_frame(10), np.zeros((80, 80, 3), dtype=np.uint8)])
    result, state = pipeline.process_frame(composite, 0.0, new_multimodal_state())
    assert result.annotated_bgr.shape == composite.shape
    assert result.fusion is not None
    assert {item.modality for item in result.modalities} == {"vision", "depth"}
    assert state["vision"]["ok"] is True


def test_multimodal_pipeline_accepts_separate_raw_depth_input(tmp_path):
    config = AppConfig(project_root=tmp_path)
    pipeline = MultimodalPipeline(
        config,
        FakeVisionPipeline(),  # type: ignore[arg-type]
        DepthPipeline(config.multimodal),
        FusionEngine(config.multimodal),
    )
    rgb = np.zeros((80, 80, 3), dtype=np.uint8)
    first = np.zeros((80, 80), dtype=np.uint16)
    second = np.zeros((80, 80), dtype=np.uint16)
    first[15:65, 20:40] = 1900
    first[15:65, 40:60] = 2100
    second[18:68, 20:40] = 2280
    second[18:68, 40:60] = 2520
    _, state = pipeline.process_rgb_depth(rgb, first, 0.0, new_multimodal_state())
    result, _ = pipeline.process_rgb_depth(rgb, second, 0.5, state)
    depth = next(item for item in result.modalities if item.modality == "depth")
    assert result.diagnostics["metric_depth_input"] is True
    assert depth.diagnostics["distance_change_ratio"] == 0.2
