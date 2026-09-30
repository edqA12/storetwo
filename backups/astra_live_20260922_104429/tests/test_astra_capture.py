from __future__ import annotations

import io
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera_capture import AstraCapture, CaptureConfig, CaptureFrame, DeviceSelection
from camera_capture import manager


DEPTH = {'uri': 'astra-uri', 'serial': 'serial', 'vid': 0x2BC5, 'pid': 0x0403}
COLOR = {'id': '@device_pnp_usb#vid_2bc5&pid_0501#test', 'name': 'Astra Pro HD Camera'}


class FakeDepth:
    def __init__(self, _: Path) -> None:
        self.index = 0
        self.closed = False

    def enumerate(self) -> list[dict]:
        return [DEPTH]

    def connect(self, *args: object) -> None:
        pass

    def start(self) -> None:
        self.index = 0

    def read(self) -> tuple[np.ndarray, int, int]:
        time.sleep(0.01)
        self.index += 1
        return np.full((2, 2), 1500, dtype=np.uint16), self.index, self.index * 33333

    def stop(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class FakePipe:
    def __init__(self) -> None:
        self.halt = threading.Event()
        self.pause = False

    def read(self, size: int) -> bytes:
        if self.pause:
            self.halt.wait(2)
        if self.halt.wait(0.01):
            return b''
        return bytes([42]) * size

    def close(self) -> None:
        self.halt.set()


class FakeProcess:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.stdout, self.stderr = FakePipe(), io.BytesIO()
        self.code = None

    def poll(self) -> int | None:
        return self.code

    def terminate(self) -> None:
        self.code = 0
        self.stdout.close()

    kill = terminate

    def wait(self, **kwargs: object) -> int:
        return 0


@pytest.fixture
def capture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(manager, 'OpenNIDepth', FakeDepth)
    monkeypatch.setattr(manager, 'enumerate_color', lambda: [COLOR])
    monkeypatch.setattr(manager.imageio_ffmpeg, 'get_ffmpeg_exe', lambda: 'fake-ffmpeg')
    monkeypatch.setattr(manager.subprocess, 'Popen', FakeProcess)
    value = AstraCapture(CaptureConfig(tmp_path, startup_timeout_s=1, stale_timeout_s=0.2))
    yield value
    value.close()


def test_selects_astra_not_laptop() -> None:
    text = '\n'.join(['[dshow] "FHD Camera" (video)', '[dshow] Alternative name "usb#vid_3277&pid_00a0"',
                      '[dshow] "Astra Pro HD Camera" (video)', '[dshow] Alternative name "usb#vid_2bc5&pid_0501#abc"'])
    assert manager.parse_color_devices(text) == [{'name': 'Astra Pro HD Camera', 'id': 'usb#vid_2bc5&pid_0501#abc'}]


def test_lifecycle_and_metadata(capture: AstraCapture) -> None:
    capture.connect()
    capture.start()
    depth, color = capture.latest('depth'), capture.latest('color')
    assert depth.depth_unit_m == 0.001 and depth.image.dtype == np.uint16
    assert depth.device_timestamp_us is not None and depth.device_id == DEPTH['uri']
    assert color.device_timestamp_us is None and color.pixel_format == 'bgr8'
    assert color.device_id == COLOR['id'] and depth.session_id == color.session_id
    first_session = depth.session_id
    capture.stop()
    assert capture.latest('depth') is None
    capture.start()
    assert capture.latest('depth').session_id != first_session
    capture.close()
    capture.connect()
    capture.start()
    assert capture.status()['state'] == 'running'


def test_slow_consumer_bounded_and_defensive_copy(capture: AstraCapture) -> None:
    capture.connect()
    capture.start()
    time.sleep(0.15)
    status = capture.status()
    assert max(status['buffer_lengths'].values()) <= 2
    assert min(status['buffer_overwrites'].values()) > 0
    frame = capture.latest('depth')
    frame.image[:] = 0
    assert np.all(capture.latest('depth').image == 1500)
    assert capture.latest('depth', after_sequence=10**9) is None


def test_duplicate_start_rejected(capture: AstraCapture) -> None:
    capture.connect()
    capture.start()
    with pytest.raises(RuntimeError):
        capture.start()


def test_second_manager_excluded(capture: AstraCapture) -> None:
    capture.connect()
    second = AstraCapture(capture.config)
    try:
        with pytest.raises(RuntimeError, match='占用'):
            second.connect()
        capture.close()
        second.connect()
    finally:
        second.close()


def test_connection_failure_releases_lease(capture: AstraCapture, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(manager, 'enumerate_color', lambda: [])
    with pytest.raises(RuntimeError):
        capture.connect()
    monkeypatch.setattr(manager, 'enumerate_color', lambda: [COLOR])
    capture.connect()


def test_ambiguous_device_requires_selection(capture: AstraCapture, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(manager, 'enumerate_color', lambda: [COLOR, {**COLOR, 'id': COLOR['id'] + '2'}])
    with pytest.raises(RuntimeError, match='显式'):
        capture.connect()
    capture.connect(DeviceSelection(DEPTH['uri'], DEPTH['serial'], COLOR['id'], COLOR['name']))


def test_wrong_selection_rejected(capture: AstraCapture) -> None:
    with pytest.raises(RuntimeError):
        capture.connect(DeviceSelection('wrong', 'serial', COLOR['id'], COLOR['name']))


def test_start_failure_cleanup(capture: AstraCapture, monkeypatch: pytest.MonkeyPatch) -> None:
    capture.connect()
    def fail(*args: object, **kwargs: object) -> None:
        raise OSError('fake launch failure')
    monkeypatch.setattr(manager.subprocess, 'Popen', fail)
    with pytest.raises(OSError):
        capture.start()
    assert capture.status()['state'] == 'connected'
    assert capture.latest('color') is None


def test_frozen_color_invalidates_all_frames(capture: AstraCapture) -> None:
    capture.connect()
    capture.start()
    capture._color.stdout.pause = True
    time.sleep(0.5)
    assert capture.status()['state'] == 'error'
    with pytest.raises(RuntimeError):
        capture.latest('depth')


def test_depth_failure_invalidates_buffers(capture: AstraCapture, monkeypatch: pytest.MonkeyPatch) -> None:
    capture.connect()
    capture.start()
    def fail() -> None:
        raise OSError('unplugged')
    monkeypatch.setattr(capture._depth, 'read', fail)
    time.sleep(0.05)
    assert capture.status()['state'] == 'error'
    assert capture.status()['buffer_lengths'] == {'color': 0, 'depth': 0}


def test_partial_pipe_never_publishes_partial_frame() -> None:
    with pytest.raises(EOFError):
        manager.read_exact(io.BytesIO(b'123'), 6)
    assert manager.read_exact(io.BytesIO(b'123456'), 6) == b'123456'


def test_cross_process_lease() -> None:
    lease = manager.DeviceLease()
    lease.acquire()
    try:
        code = ('from camera_capture.manager import DeviceLease\n'
                'import sys\n'
                'try: DeviceLease().acquire()\n'
                'except RuntimeError: sys.exit(42)\n')
        result = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).parents[1], capture_output=True, timeout=10)
        assert result.returncode == 42
    finally:
        lease.release()


@pytest.mark.parametrize('kwargs', [{'buffer_size': 0}, {'fps': 15}, {'stale_timeout_s': 20}])
def test_invalid_config(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        CaptureConfig(Path('.'), **kwargs)
