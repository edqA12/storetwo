from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


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
class AppConfig:
    project_root: Path
    model: ModelConfig = field(default_factory=ModelConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

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
    for section in ("model", "detection", "video", "storage", "server"):
        values = raw.get(section, {})
        if isinstance(values, dict):
            _apply_values(getattr(config, section), values)
    config.ensure_directories()
    return config
