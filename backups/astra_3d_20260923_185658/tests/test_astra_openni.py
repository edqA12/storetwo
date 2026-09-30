from __future__ import annotations

import ctypes as ct
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera_capture.openni import NativeFrame, OpenNIDepth, VideoMode


def reader(pixel_format: int = 100, wait_code: int = 0):
    source = np.array([[1200, 1500, 999], [1400, 0, 999]], dtype=np.uint16)
    frame = NativeFrame(source.nbytes, source.ctypes.data, 3, 123456, 8, 2, 2,
                        VideoMode(pixel_format, 2, 2, 30), 0, 0, 0, 6)
    released = []
    def read(_stream, output):
        ct.cast(output, ct.POINTER(ct.POINTER(NativeFrame)))[0] = ct.pointer(frame)
        return 0
    obj = OpenNIDepth.__new__(OpenNIDepth)
    obj.stream = ct.c_void_p(1)
    obj.lib = SimpleNamespace(oniWaitForAnyStream=lambda *args: wait_code,
                              oniStreamReadFrame=read, oniFrameRelease=lambda f: released.append(True),
                              oniGetExtendedError=lambda: b'fake')
    return obj, source, released


def test_raw_depth_stride_copy_unit_and_release() -> None:
    obj, source, released = reader()
    result, index, timestamp = obj.read()
    assert result.tolist() == [[1200, 1500], [1400, 0]]
    assert index == 8 and timestamp == 123456 and released == [True]
    source[:] = 42
    assert result[0, 0] == 1200


def test_wrong_depth_unit_rejected_and_released() -> None:
    obj, _, released = reader(pixel_format=101)
    with pytest.raises(RuntimeError, match='格式'):
        obj.read()
    assert released == [True]


def test_native_timeout_has_no_frame_to_release() -> None:
    obj, _, released = reader(wait_code=102)
    with pytest.raises(TimeoutError):
        obj.read()
    assert released == []
