"""Astra Pro acquisition only: no inference, persistence, pairing or registration."""
from __future__ import annotations

import atexit
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import time
from typing import Any, BinaryIO
import uuid

import imageio_ffmpeg
import numpy as np

from .openni import OpenNIDepth


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    sdk_dir: Path
    width: int = 640
    height: int = 480
    fps: int = 30
    buffer_size: int = 2
    startup_timeout_s: float = 10.0
    stale_timeout_s: float = 3.0

    def __post_init__(self) -> None:
        if (self.width, self.height, self.fps) != (640, 480, 30):
            raise ValueError('本阶段仅开放已实测的640×480、30 FPS模式。')
        if not 1 <= self.buffer_size <= 8:
            raise ValueError('缓冲帧数必须为1至8。')
        if not 0 < self.stale_timeout_s <= self.startup_timeout_s <= 60:
            raise ValueError('采集超时配置无效。')


@dataclass(frozen=True, slots=True)
class DeviceSelection:
    depth_uri: str
    depth_serial: str
    color_id: str
    color_name: str


@dataclass(frozen=True, slots=True)
class CaptureFrame:
    image: np.ndarray
    source: str
    sequence: int
    received_monotonic_s: float
    device_timestamp_us: int | None
    timestamp_basis: str
    pixel_format: str
    depth_unit_m: float | None
    device_id: str
    session_id: str


class DeviceLease:
    """Process-wide and cross-process lock for this Windows user's Astra source.

    Conservative: reserves all Astra devices, including during enumeration.
    OS closes the byte lock after a crash; lock file existence is not ownership.
    """
    _local = threading.Lock()

    def __init__(self) -> None:
        self.file: BinaryIO | None = None

    def acquire(self) -> None:
        import msvcrt
        if not self._local.acquire(blocking=False):
            raise RuntimeError('Astra设备已被另一个采集实例占用。')
        try:
            self.file = (Path(tempfile.gettempdir()) / 'muan-astra-capture.lock').open('a+b')
            self.file.seek(0, 2)
            if self.file.tell() == 0:
                self.file.write(b'0')
                self.file.flush()
            self.file.seek(0)
            msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
        except Exception:
            if self.file:
                self.file.close()
                self.file = None
            self._local.release()
            raise RuntimeError('Astra采集被其他进程占用，或无法建立设备占用锁。') from None

    def release(self) -> None:
        if self.file:
            self.file.close()
            self.file = None
            self._local.release()


def parse_color_devices(text: str) -> list[dict[str, str]]:
    devices: list[dict[str, str]] = []
    name: str | None = None
    for line in text.splitlines():
        match = re.search(r'"([^"]+)" \(video\)', line)
        if match:
            name = match.group(1)
        alternative = re.search(r'Alternative name "([^"]+)"', line)
        if alternative and name:
            identifier = alternative.group(1)
            if 'vid_2bc5&pid_0501' in identifier.lower():
                devices.append({'name': name, 'id': identifier})
            name = None
    return devices


def enumerate_color() -> list[dict[str, str]]:
    completed = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-list_devices', 'true',
         '-f', 'dshow', '-i', 'dummy'], capture_output=True, timeout=15,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
    )
    # FFmpeg device enumeration intentionally exits nonzero after listing.
    return parse_color_devices(completed.stderr.decode('utf-8', errors='replace'))


def read_exact(pipe: BinaryIO, size: int) -> bytes:
    chunks, remaining = [], size
    while remaining:
        chunk = pipe.read(remaining)
        if not chunk:
            raise EOFError('彩色采集已结束或帧数据不完整。')
        chunks.append(chunk)
        remaining -= len(chunk)
    return b''.join(chunks)


class AstraCapture:
    def __init__(self, config: CaptureConfig) -> None:
        self.config = config
        self._operation = threading.RLock()
        self._data = threading.Lock()
        self._lease = DeviceLease()
        self._depth: OpenNIDepth | None = None
        self._color: subprocess.Popen[bytes] | None = None
        self._threads: list[threading.Thread] = []
        self._halt = threading.Event()
        self._buffers: dict[str, deque[CaptureFrame]] = {
            name: deque(maxlen=config.buffer_size) for name in ('depth', 'color')
        }
        self._counts = {'depth': 0, 'color': 0}
        self._overwrites = {'depth': 0, 'color': 0}
        self._error: str | None = None
        self._source_errors: dict[str, str] = {}
        self._stderr: deque[str] = deque(maxlen=12)
        self._state = 'disconnected'
        self._selection: DeviceSelection | None = None
        self._session = ''
        atexit.register(self.close)

    @classmethod
    def enumerate_devices(cls, config: CaptureConfig) -> dict[str, Any]:
        lease, depth = DeviceLease(), None
        lease.acquire()
        try:
            depth = OpenNIDepth(config.sdk_dir)
            return {'depth': [d for d in depth.enumerate() if d['vid'] == 0x2BC5 and d['pid'] == 0x0403],
                    'color': enumerate_color()}
        finally:
            if depth:
                depth.close()
            lease.release()

    def connect(self, selection: DeviceSelection | None = None) -> DeviceSelection:
        with self._operation:
            if self._state != 'disconnected':
                raise RuntimeError('请先断开当前设备。')
            self._lease.acquire()
            try:
                self._depth = OpenNIDepth(self.config.sdk_dir)
                depths = [d for d in self._depth.enumerate() if d['vid'] == 0x2BC5 and d['pid'] == 0x0403]
                colors = enumerate_color()
                if selection is None:
                    if len(depths) != 1 or len(colors) != 1:
                        raise RuntimeError('需要恰好一台Astra Pro；多设备时须显式指定深度URI和彩色设备标识。')
                    selection = DeviceSelection(depths[0]['uri'], depths[0]['serial'], colors[0]['id'], colors[0]['name'])
                if not any(d['uri'] == selection.depth_uri and d['serial'] == selection.depth_serial for d in depths):
                    raise RuntimeError('所选深度设备不在当前Astra Pro列表中。')
                if not any(d['id'] == selection.color_id for d in colors):
                    raise RuntimeError('所选彩色设备不在当前Astra Pro列表中。')
                self._depth.connect(selection.depth_uri, self.config.width, self.config.height, self.config.fps)
                self._selection = selection
                self._state, self._error = 'connected', None
                return selection
            except Exception:
                if self._depth:
                    self._depth.close()
                    self._depth = None
                self._lease.release()
                raise

    def start(self) -> None:
        with self._operation:
            if self._state != 'connected' or self._depth is None or self._selection is None:
                raise RuntimeError('请先连接设备，且不要重复启动。')
            self._halt.clear()
            with self._data:
                for buffer in self._buffers.values():
                    buffer.clear()
                self._counts = {'depth': 0, 'color': 0}
                self._overwrites = {'depth': 0, 'color': 0}
                self._error = None
                self._source_errors.clear()
                self._stderr.clear()
            self._session = uuid.uuid4().hex
            self._state = 'starting'
            try:
                try:
                    self._depth.start()
                except Exception as exc:
                    self._source_errors['depth'] = f'深度启动失败：{exc}'
                self._color = subprocess.Popen(
                    [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error',
                     '-f', 'dshow', '-rtbufsize', '2M', '-video_size', '640x480',
                     '-framerate', '30', '-vcodec', 'mjpeg', '-i', f'video={self._selection.color_id}',
                     '-an', '-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1'],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
                )
                self._threads = [threading.Thread(target=fn, daemon=True, name=f'astra-{name}')
                                 for name, fn in [('depth', self._depth_loop), ('color', self._color_loop),
                                                  ('errors', self._error_loop), ('watchdog', self._watchdog_loop)]]
                for thread in self._threads:
                    thread.start()
                deadline = time.monotonic() + self.config.startup_timeout_s
                while time.monotonic() < deadline:
                    with self._data:
                        if self._error:
                            raise RuntimeError(self._error)
                        if 'color' in self._source_errors:
                            raise RuntimeError(self._source_errors['color'])
                        if self._buffers['color'] and (self._buffers['depth'] or 'depth' in self._source_errors):
                            self._state = 'degraded' if self._source_errors else 'running'
                            return
                    time.sleep(0.02)
                raise TimeoutError('双流启动超时；检查设备占用、USB和UVC访问权限。')
            except Exception:
                self.stop()
                raise

    def _publish(self, frame: CaptureFrame) -> None:
        frame.image.setflags(write=False)
        with self._data:
            if self._halt.is_set() or frame.source in self._source_errors:
                return
            buffer = self._buffers[frame.source]
            if len(buffer) == buffer.maxlen:
                self._overwrites[frame.source] += 1
            buffer.append(frame)
            self._counts[frame.source] += 1

    def _fail(self, message: str, source: str | None = None) -> None:
        with self._data:
            if not self._halt.is_set():
                if source is not None:
                    self._source_errors[source] = message
                    self._buffers[source].clear()
                    if self._state != 'starting':
                        self._state = 'degraded' if len(self._source_errors) == 1 else 'error'
                    return
                self._error = message
                self._state = 'error'
                for buffer in self._buffers.values():
                    buffer.clear()
                self._halt.set()
        # Release a potentially blocked pipe reader, without joining this thread.
        if self._color and self._color.poll() is None:
            self._color.terminate()

    def _depth_loop(self) -> None:
        assert self._depth is not None and self._selection is not None
        if 'depth' in self._source_errors:
            return
        last = time.monotonic()
        try:
            while not self._halt.is_set():
                try:
                    image, index, stamp = self._depth.read()
                except TimeoutError:
                    if time.monotonic() - last > self.config.stale_timeout_s:
                        raise TimeoutError('深度流持续超时。')
                    continue
                last = time.monotonic()
                self._publish(CaptureFrame(image, 'depth', index, last, stamp,
                                          'device_us_and_host_receive_monotonic', 'depth_u16_mm',
                                          0.001, self._selection.depth_uri, self._session))
        except Exception as exc:
            self._fail(f'深度采集失败：{exc}', 'depth')

    def _color_loop(self) -> None:
        assert self._color is not None and self._color.stdout is not None and self._selection is not None
        try:
            sequence = 0
            while not self._halt.is_set():
                raw = read_exact(self._color.stdout, self.config.width * self.config.height * 3)
                received = time.monotonic()
                sequence += 1
                image = np.frombuffer(raw, dtype=np.uint8).reshape(self.config.height, self.config.width, 3).copy()
                self._publish(CaptureFrame(image, 'color', sequence, received, None,
                                          'host_receive_monotonic_only', 'bgr8', None,
                                          self._selection.color_id, self._session))
        except Exception as exc:
            self._fail(f'彩色采集失败：{exc}', 'color')

    def _error_loop(self) -> None:
        assert self._color is not None and self._color.stderr is not None
        for line in self._color.stderr:
            with self._data:
                self._stderr.append(line.decode('utf-8', errors='replace').strip())

    def _watchdog_loop(self) -> None:
        while not self._halt.wait(0.1):
            with self._data:
                stale = [source for source, buffer in self._buffers.items()
                         if self._state in {'running', 'degraded'} and source not in self._source_errors
                         and (not buffer or time.monotonic() - buffer[-1].received_monotonic_s > self.config.stale_timeout_s)]
            for source in stale:
                self._fail('采集流停止更新，旧帧已失效；请停止后重新连接。', source)

    def latest(self, source: str, after_sequence: int | None = None) -> CaptureFrame | None:
        """Return a defensive copy of one fresh frame; streams are NOT paired."""
        if source not in self._buffers:
            raise ValueError('source必须为depth或color。')
        with self._data:
            if self._state in {'connected', 'disconnected', 'starting'}:
                return None
            if self._error:
                raise RuntimeError(self._error)
            if self._state not in {'running', 'degraded'} or source in self._source_errors or not self._buffers[source]:
                return None
            frame = self._buffers[source][-1]
            if (time.monotonic() - frame.received_monotonic_s > self.config.stale_timeout_s
                    or (after_sequence is not None and frame.sequence <= after_sequence)):
                return None
            return replace(frame, image=frame.image.copy())

    def status(self) -> dict[str, Any]:
        with self._data:
            ages = {name: time.monotonic() - buffer[-1].received_monotonic_s if buffer else None
                    for name, buffer in self._buffers.items()}
            return {'state': self._state, 'error': self._error, 'session_id': self._session,
                    'device_ids': {'depth': self._selection.depth_uri, 'color': self._selection.color_id} if self._selection else {},
                    'frames': dict(self._counts), 'buffer_overwrites': dict(self._overwrites),
                    'buffer_lengths': {name: len(buffer) for name, buffer in self._buffers.items()},
                    'age_s': ages, 'fresh': {name: age is not None and age <= self.config.stale_timeout_s
                                           for name, age in ages.items()},
                    'color_errors': list(self._stderr), 'source_errors': dict(self._source_errors), 'synchronized': False}

    def buffered_frames(self) -> list[CaptureFrame]:
        """Atomic bounded snapshot for a single preprocessing consumer."""
        with self._data:
            if self._state not in {'running', 'degraded'} or self._error:
                return []
            return [replace(frame, image=frame.image.copy())
                    for source, buffer in self._buffers.items() if source not in self._source_errors for frame in buffer]

    def stop(self) -> None:
        with self._operation:
            self._halt.set()
            if self._color:
                if self._color.poll() is None:
                    self._color.terminate()
                try:
                    self._color.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._color.kill()
                    self._color.wait(timeout=3)
            for thread in self._threads:
                thread.join(timeout=3)
            if any(thread.is_alive() for thread in self._threads):
                # Never destroy an SDK handle still being accessed by a native reader.
                self._state = 'error'
                raise RuntimeError('采集线程未退出；保留设备占用锁，请关闭采集进程。')
            if self._depth:
                self._depth.stop()
            if self._color:
                for pipe in (self._color.stdout, self._color.stderr):
                    if pipe:
                        pipe.close()
                self._color = None
            self._threads.clear()
            with self._data:
                for buffer in self._buffers.values():
                    buffer.clear()
            self._state = 'connected' if self._depth else 'disconnected'

    def close(self) -> None:
        with self._operation:
            self.stop()
            if self._depth:
                self._depth.close()
                self._depth = None
            self._selection = None
            self._lease.release()
            self._state = 'disconnected'

    def __enter__(self) -> AstraCapture:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
