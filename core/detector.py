from __future__ import annotations

import os
import shutil
import threading
import time
from pathlib import Path

import numpy as np

from .config import AppConfig
from .types import PersonPose


class ModelUnavailableError(RuntimeError):
    pass


class YoloPoseDetector:
    """线程安全、延迟加载的YOLO姿态检测器。"""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._model = None
        self._lock = threading.RLock()
        self.last_inference_ms = 0.0

    @property
    def model_path(self) -> Path:
        return self.config.resolve(self.config.model.path)

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from ultralytics import YOLO
            except Exception as exc:  # pragma: no cover - depends on optional runtime
                raise ModelUnavailableError(
                    "尚未安装Ultralytics。请先双击 setup.bat 完成环境安装。"
                ) from exc

            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                if self.model_path.exists():
                    self._model = YOLO(str(self.model_path))
                    return

                original_cwd = Path.cwd()
                try:
                    os.chdir(self.config.project_root)
                    self._model = YOLO(self.config.model.download_name)
                finally:
                    os.chdir(original_cwd)

                downloaded = self.config.project_root / self.config.model.download_name
                checkpoint = Path(str(getattr(self._model, "ckpt_path", downloaded)))
                if not checkpoint.is_absolute():
                    checkpoint = self.config.project_root / checkpoint
                if checkpoint.exists() and checkpoint.resolve() != self.model_path.resolve():
                    shutil.copy2(checkpoint, self.model_path)
            except Exception as exc:
                self._model = None
                raise ModelUnavailableError(
                    "姿态模型加载失败。请检查网络，或将yolo11n-pose.pt放入models文件夹。"
                ) from exc

    def detect(self, frame_bgr: np.ndarray, confidence: float | None = None) -> list[PersonPose]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        self.ensure_loaded()
        threshold = float(confidence if confidence is not None else self.config.model.confidence)
        started = time.perf_counter()
        with self._lock:
            try:
                results = self._model.predict(  # type: ignore[union-attr]
                    source=frame_bgr,
                    conf=threshold,
                    imgsz=int(self.config.model.image_size),
                    device=self.config.model.device,
                    verbose=False,
                )
            except Exception as exc:
                raise ModelUnavailableError(f"模型推理失败：{exc}") from exc
        self.last_inference_ms = (time.perf_counter() - started) * 1000.0

        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        keypoints = getattr(result, "keypoints", None)
        if boxes is None or keypoints is None or len(boxes) == 0:
            return []

        try:
            boxes_xyxy = boxes.xyxy.detach().cpu().numpy()
            box_confidence = boxes.conf.detach().cpu().numpy()
            keypoint_data = keypoints.data.detach().cpu().numpy()
        except Exception as exc:
            raise ModelUnavailableError("无法解析姿态模型输出。") from exc

        count = min(len(boxes_xyxy), len(keypoint_data))
        poses: list[PersonPose] = []
        for index in range(count):
            data = np.asarray(keypoint_data[index], dtype=np.float32)
            if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] < 2:
                continue
            xy = data[:, :2]
            confidence_values = data[:, 2] if data.shape[1] >= 3 else np.ones(data.shape[0], dtype=np.float32)
            poses.append(PersonPose(
                track_id=0,
                bbox_xyxy=tuple(float(value) for value in boxes_xyxy[index][:4]),  # type: ignore[arg-type]
                box_confidence=float(box_confidence[index]),
                keypoints_xy=xy,
                keypoint_confidence=np.asarray(confidence_values, dtype=np.float32),
            ))
        return poses
