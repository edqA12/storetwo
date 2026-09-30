from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from camera_capture.processing import ProcessingConfig


@dataclass(slots=True)
class ModelConfig:
    path: str = "models/yolo11n-pose.pt"
    download_name: str = "yolo11n-pose.pt"
    device: str = "cpu"
    image_size: int = 512
    confidence: float = 0.30
    keypoint_confidence: float = 0.25


@dataclass(slots=True)
class DetectionConfig:
    history_seconds: float = 1.6
    missing_tolerance_s: float = 0.8
    tracker_iou_threshold: float = 0.25
    tracker_center_threshold: float = 0.45
    lying_enter_angle_deg: float = 60.0
    lying_exit_angle_deg: float = 40.0
    lying_enter_score: float = 0.70
    lying_exit_score: float = 0.40
    fast_down_velocity: float = 0.20
    fast_angle_velocity: float = 48.0
    descent_timeout_s: float = 2.2
    lying_hold_s: float = 0.6
    unexplained_lying_s: float = 2.0
    suspected_fall_hold_s: float = 3.0
    suspected_fall_min_quality: float = 0.55
    recovery_upright_s: float = 1.0
    event_cooldown_s: float = 9.0
    stale_derivative_s: float = 1.0


@dataclass(slots=True)
class VideoConfig:
    max_file_mb: int = 300
    max_duration_s: int = 600
    fallback_fps: float = 25.0
    default_frame_stride: int = 1


@dataclass(slots=True)
class MultimodalConfig(ProcessingConfig):
    enabled: bool = True
    depth_side: str = "left"
    split_ratio: float = 0.5
    min_half_width: int = 120
    min_depth_quality: float = 0.18
    depth_min_valid_ratio: float = 0.08
    depth_motion_threshold: float = 12.0
    depth_motion_ratio: float = 0.025
    depth_downward_delta: float = 0.06
    depth_impact_delta: float = 26.0
    depth_evidence_hold_s: float = 2.0
    visual_weight: float = 0.75
    depth_weight: float = 0.25
    depth_support_threshold: float = 45.0
    depth_contradiction_threshold: float = 15.0
    fusion_alert_threshold: float = 75.0
    event_cooldown_s: float = 9.0
    depth_distance_change_ratio: float = 0.08
    depth_longitudinal_velocity: float = 0.20


@dataclass(slots=True)
class PreFallConfig:
    enabled: bool = True
    person_id: str = "primary"
    short_window_s: float = 6.0
    baseline_min_samples: int = 20
    baseline_alpha: float = 0.04
    min_active_speed: float = 0.015
    min_active_motion_norm: float = 0.004
    movement_drop_start_ratio: float = 0.15
    movement_drop_full_ratio: float = 0.45
    sway_start_norm: float = 0.004
    sway_full_norm: float = 0.025
    torso_variability_start_deg: float = 3.0
    torso_variability_full_deg: float = 16.0
    near_fall_min_down_velocity: float = 0.14
    near_fall_min_angle_velocity: float = 36.0
    near_fall_min_torso_angle: float = 24.0
    near_fall_max_lying_score: float = 0.68
    near_fall_min_stability_score: float = 30.0
    near_fall_min_sway_score: float = 30.0
    near_fall_knee_collapse_max_deg: float = 128.0
    near_fall_knee_recovery_min_deg: float = 154.0
    near_fall_min_knee_drop_deg: float = 30.0
    near_fall_knee_collapse_min_speed: float = 0.04
    near_fall_knee_collapse_max_torso_deg: float = 15.0
    near_fall_recovery_min_s: float = 0.25
    near_fall_recovery_max_s: float = 2.8
    near_fall_window_s: float = 86400.0
    near_fall_count_full: int = 3
    movement_weight: float = 0.25
    stability_weight: float = 0.20
    near_fall_weight: float = 0.30
    sit_to_stand_weight: float = 0.10
    baseline_weight: float = 0.15
    medium_threshold: float = 31.0
    high_threshold: float = 61.0
    medium_exit_threshold: float = 25.0
    high_exit_threshold: float = 52.0
    upgrade_hold_s: float = 0.8
    high_upgrade_hold_s: float = 1.5
    downgrade_hold_s: float = 3.0
    score_smoothing_alpha: float = 0.30
    baseline_max_stability_score: float = 35.0
    baseline_stability_full_delta: float = 40.0
    risk_factor_min_score: float = 20.0
    record_interval_s: float = 2.0
    min_rgb_quality: float = 0.35
    min_depth_quality: float = 0.18
    sit_to_stand_seated_knee_max_deg: float = 128.0
    sit_to_stand_standing_knee_min_deg: float = 154.0
    sit_to_stand_max_torso_angle_deg: float = 32.0
    sit_to_stand_min_sit_hold_s: float = 1.00
    sit_to_stand_min_transition_s: float = 0.25
    sit_to_stand_max_transition_s: float = 8.0
    sit_to_stand_standing_hold_s: float = 0.35
    sit_to_stand_min_up_velocity: float = 0.025
    sit_to_stand_min_knee_change_deg: float = 8.0
    sit_to_stand_min_cumulative_knee_change_deg: float = 30.0
    sit_to_stand_min_center_rise_norm: float = 0.025
    sit_to_stand_min_torso_change_deg: float = 12.0
    sit_to_stand_attempt_hold_s: float = 0.10
    sit_to_stand_max_center_drop_norm: float = 0.015
    sit_to_stand_min_box_area_ratio: float = 0.025
    sit_to_stand_min_pose_quality: float = 0.50
    sit_to_stand_relative_standing_rise_norm: float = 0.07
    sit_to_stand_relative_standing_knee_min_deg: float = 138.0
    sit_to_stand_return_center_tolerance_norm: float = 0.015
    sit_to_stand_return_knee_tolerance_deg: float = 14.0
    sit_to_stand_return_torso_tolerance_deg: float = 12.0
    sit_to_stand_slow_start_s: float = 2.0
    sit_to_stand_slow_full_s: float = 5.0
    sit_to_stand_failure_score: float = 85.0
    sit_to_stand_instability_full_score: float = 75.0
    sit_to_stand_baseline_min_events: int = 3
    sit_to_stand_baseline_alpha: float = 0.10
    sit_to_stand_baseline_slow_start_ratio: float = 0.15
    sit_to_stand_baseline_slow_full_ratio: float = 0.50
    sit_to_stand_repeat_window_s: float = 30.0
    sit_to_stand_repeat_count_full: int = 3
    medium_window_s: float = 86400.0
    long_window_days: int = 30
    window_sample_interval_s: float = 10.0
    short_term_weight: float = 0.55
    medium_term_weight: float = 0.25
    long_term_weight: float = 0.20
    profile_save_interval_s: float = 30.0
    pre_fall_depth_weight: float = 0.25


@dataclass(slots=True)
class StorageConfig:
    database: str = "data/events.db"
    uploads_dir: str = "data/uploads"
    results_dir: str = "data/results"
    snapshots_dir: str = "data/snapshots"
    exports_dir: str = "data/exports"


@dataclass(slots=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 7860
    open_browser: bool = True


@dataclass(slots=True)
class AstraConfig:
    sdk_dir: str = ''
    result_max_age_s: float = 2.0
    page_timeout_s: float = 30.0


@dataclass(slots=True)
class AppConfig:
    project_root: Path
    model: ModelConfig = field(default_factory=ModelConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    multimodal: MultimodalConfig = field(default_factory=MultimodalConfig)
    pre_fall: PreFallConfig = field(default_factory=PreFallConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    astra: AstraConfig = field(default_factory=AstraConfig)

    def resolve(self, relative_or_absolute: str) -> Path:
        path = Path(relative_or_absolute)
        return path if path.is_absolute() else self.project_root / path

    def ensure_directories(self) -> None:
        for value in (
            self.storage.uploads_dir,
            self.storage.results_dir,
            self.storage.snapshots_dir,
            self.storage.exports_dir,
            str(Path(self.model.path).parent),
        ):
            self.resolve(value).mkdir(parents=True, exist_ok=True)
        self.resolve(self.storage.database).parent.mkdir(parents=True, exist_ok=True)


def _apply_values(instance: Any, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if hasattr(instance, key):
            setattr(instance, key, value)


def load_config(path: str | Path | None = None) -> AppConfig:
    project_root = Path(__file__).resolve().parents[1]
    config_path = Path(path) if path else project_root / "config.yaml"
    raw: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

    config = AppConfig(project_root=project_root)
    for section in ("model", "detection", "video", "multimodal", "pre_fall", "storage", "server", "astra"):
        values = raw.get(section, {})
        if isinstance(values, dict):
            _apply_values(getattr(config, section), values)
    config.ensure_directories()
    return config
