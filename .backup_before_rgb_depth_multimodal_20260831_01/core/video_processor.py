from __future__ import annotations

import math
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from services.events import EventService

from .config import AppConfig
from .pipeline import VisionPipeline, new_stream_state


ProgressCallback = Callable[[float, str], None]


@dataclass(slots=True)
class VideoProcessResult:
    output_path: Path
    max_risk: float
    event_count: int
    risk_samples: list[tuple[float, float]]
    event_rows: list[list[Any]]
    message: str


class VideoProcessor:
    def __init__(self, config: AppConfig, pipeline: VisionPipeline, event_service: EventService) -> None:
        self.config = config
        self.pipeline = pipeline
        self.event_service = event_service

    def process(
        self,
        input_path: str | Path,
        confidence: float,
        frame_stride: int = 1,
        progress: ProgressCallback | None = None,
    ) -> VideoProcessResult:
        source = Path(input_path)
        if not source.exists() or not source.is_file():
            raise ValueError("未找到上传的视频文件。")
        if source.stat().st_size > self.config.video.max_file_mb * 1024 * 1024:
            raise ValueError(f"视频超过{self.config.video.max_file_mb}MB限制。")

        temporary_source: Path | None = None
        capture_source = source
        if not str(source).isascii():
            uploads_dir = self.config.resolve(self.config.storage.uploads_dir)
            uploads_dir.mkdir(parents=True, exist_ok=True)
            temporary_source = uploads_dir / f"{uuid.uuid4().hex}{source.suffix.lower()}"
            shutil.copy2(source, temporary_source)
            capture_source = temporary_source

        capture = cv2.VideoCapture(str(capture_source))
        if not capture.isOpened():
            capture.release()
            if temporary_source is not None:
                temporary_source.unlink(missing_ok=True)
            raise ValueError("视频无法打开，可能格式不受支持或文件已经损坏。")

        raw_output: Path | None = None
        final_output: Path | None = None
        completed = False
        writer: cv2.VideoWriter | None = None
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not math.isfinite(fps) or fps <= 1e-3:
                fps = self.config.video.fallback_fps
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width <= 0 or height <= 0:
                raise ValueError("无法读取视频分辨率。")
            if frame_count > 0 and frame_count / fps > self.config.video.max_duration_s:
                raise ValueError(f"视频超过{self.config.video.max_duration_s // 60}分钟限制。")

            results_dir = self.config.resolve(self.config.storage.results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)
            run_id = uuid.uuid4().hex
            raw_output = results_dir / f"{run_id}_raw.mp4"
            final_output = results_dir / f"{run_id}_detected.mp4"
            writer = cv2.VideoWriter(
                str(raw_output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
            if not writer.isOpened():
                raise RuntimeError("无法创建结果视频文件。")

            state = new_stream_state()
            stride = max(1, min(int(frame_stride), 5))
            index = 0
            processed = 0
            max_risk = 0.0
            event_count = 0
            risks: list[tuple[float, float]] = []
            event_rows: list[list[Any]] = []
            last_risk = 0.0
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                timestamp_s = index / fps
                if timestamp_s > self.config.video.max_duration_s:
                    raise ValueError(f"视频超过{self.config.video.max_duration_s // 60}分钟限制。")
                if index % stride == 0:
                    result, state = self.pipeline.process_frame(
                        frame, timestamp_s, state, confidence=confidence
                    )
                    rendered = result.annotated_bgr
                    last_risk = result.max_risk
                    max_risk = max(max_risk, result.max_risk)
                    for event in result.new_events:
                        event_type = str(event.get("event_type", "跌倒告警"))
                        try:
                            event_id = self.event_service.record(
                                event, rendered, "上传视频", source.name
                            )
                            event_count += 1
                            event_rows.append([
                                event_id,
                                f"{float(event['source_time_s']):.1f}s",
                                event_type,
                                float(event["risk"]),
                                "；".join(event.get("reasons", [])),
                            ])
                        except Exception:
                            event_count += 1
                            event_rows.append([
                                "保存失败",
                                f"{float(event['source_time_s']):.1f}s",
                                event_type,
                                float(event["risk"]),
                                "事件记录失败，请查看结果视频并人工确认",
                            ])
                    processed += 1
                else:
                    rendered = self.pipeline.render_cached(frame, state)
                writer.write(rendered)
                risks.append((round(timestamp_s, 3), round(last_risk, 1)))
                index += 1
                if progress and (index % max(1, int(fps)) == 0 or (frame_count > 0 and index == frame_count)):
                    fraction = min(0.95, index / frame_count) if frame_count > 0 else 0.5
                    progress(fraction, f"正在分析：{timestamp_s:.1f} 秒")

            if index == 0:
                raise ValueError("视频中没有可读取的画面。")
            if frame_count > 0 and index < max(1, int(frame_count * 0.90)):
                raise ValueError(
                    f"视频在第{index}帧提前结束（容器声明约{frame_count}帧），文件可能损坏。"
                )
            writer.release()
            writer = None
            capture.release()
            if progress:
                progress(0.97, "正在生成浏览器兼容的视频……")
            transcoded = self._transcode_h264(raw_output, final_output)
            if not transcoded:
                shutil.move(str(raw_output), str(final_output))
            elif raw_output.exists():
                raw_output.unlink()
            message = (
                f"检测完成：共读取{index}帧，实际推理{processed}帧，"
                f"发现{event_count}条告警事件。"
            )
            if not transcoded:
                message += " 当前环境未完成H.264转码，若网页无法播放可直接下载结果。"
            if progress:
                progress(1.0, "处理完成")
            completed = True
            return VideoProcessResult(
                output_path=final_output,
                max_risk=round(max_risk, 1),
                event_count=event_count,
                risk_samples=risks,
                event_rows=event_rows,
                message=message,
            )
        finally:
            capture.release()
            if writer is not None:
                writer.release()
            if temporary_source is not None:
                temporary_source.unlink(missing_ok=True)
            if not completed:
                if raw_output is not None:
                    raw_output.unlink(missing_ok=True)
                if final_output is not None:
                    final_output.unlink(missing_ok=True)

    @staticmethod
    def _transcode_h264(source: Path, destination: Path) -> bool:
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            command = [
                ffmpeg, "-y", "-loglevel", "error", "-i", str(source), "-an",
                "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(destination),
            ]
            result = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
            success = result.returncode == 0 and destination.exists() and destination.stat().st_size > 0
            if not success:
                destination.unlink(missing_ok=True)
            return success
        except Exception:
            return False
