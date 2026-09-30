"""Minimal bindings for the installed Orbbec OpenNI 2.3 ABI (Win64)."""
from __future__ import annotations

import ctypes as ct
import os
from pathlib import Path
from typing import Any

import numpy as np


class VideoMode(ct.Structure):
    _fields_ = [(name, ct.c_int) for name in ('pixel_format', 'width', 'height', 'fps')]


class DeviceInfo(ct.Structure):
    _fields_ = [(name, ct.c_char * 256) for name in ('uri', 'vendor', 'name', 'serial')] + [
        ('vid', ct.c_uint16), ('pid', ct.c_uint16),
    ]


class NativeFrame(ct.Structure):
    _fields_ = [
        ('size', ct.c_int), ('data', ct.c_void_p), ('sensor', ct.c_int),
        ('timestamp', ct.c_uint64), ('index', ct.c_int), ('width', ct.c_int),
        ('height', ct.c_int), ('mode', VideoMode), ('cropping', ct.c_int),
        ('crop_x', ct.c_int), ('crop_y', ct.c_int), ('stride', ct.c_int),
    ]


class OBCameraParams(ct.Structure):
    # Exact layout from this SDK's Win64 Include/OniCTypes.h (property 14).
    _fields_ = [(name, ct.c_float * count) for name, count in (
        ('l_intr_p', 4), ('r_intr_p', 4), ('r2l_r', 9), ('r2l_t', 3), ('l_k', 5), ('r_k', 5))]


class OpenNIDepth:
    """Caller must hold the application device lease for this object's lifetime."""

    def __init__(self, sdk_dir: Path) -> None:
        if os.name != 'nt' or ct.sizeof(ct.c_void_p) != 8:
            raise RuntimeError('Astra采集要求Windows 64位Python。')
        self.device = ct.c_void_p()
        self.stream = ct.c_void_p()
        self.initialized = False
        self.running = False
        self.dll_directory = os.add_dll_directory(str(sdk_dir.resolve()))
        try:
            self.lib = ct.CDLL(str(sdk_dir.resolve() / 'OpenNI2.dll'))
            self._bind()
            self._check(self.lib.oniInitialize(2003))
            self.initialized = True
        except Exception:
            self.dll_directory.close()
            raise

    def _bind(self) -> None:
        pointer = ct.c_void_p
        declarations = {
            'oniInitialize': ([ct.c_int], ct.c_int),
            'oniShutdown': ([], None),
            'oniGetExtendedError': ([], ct.c_char_p),
            'oniGetDeviceList': ([ct.POINTER(ct.POINTER(DeviceInfo)), ct.POINTER(ct.c_int)], ct.c_int),
            'oniReleaseDeviceList': ([ct.POINTER(DeviceInfo)], ct.c_int),
            'oniDeviceOpen': ([ct.c_char_p, ct.POINTER(pointer)], ct.c_int),
            'oniDeviceClose': ([pointer], ct.c_int),
            'oniDeviceGetProperty': ([pointer, ct.c_int, pointer, ct.POINTER(ct.c_int)], ct.c_int),
            'oniDeviceCreateStream': ([pointer, ct.c_int, ct.POINTER(pointer)], ct.c_int),
            'oniStreamSetProperty': ([pointer, ct.c_int, pointer, ct.c_int], ct.c_int),
            'oniStreamStart': ([pointer], ct.c_int),
            'oniStreamStop': ([pointer], None),
            'oniStreamDestroy': ([pointer], None),
            'oniWaitForAnyStream': ([ct.POINTER(pointer), ct.c_int, ct.POINTER(ct.c_int), ct.c_int], ct.c_int),
            'oniStreamReadFrame': ([pointer, ct.POINTER(ct.POINTER(NativeFrame))], ct.c_int),
            'oniFrameRelease': ([ct.POINTER(NativeFrame)], None),
        }
        for name, (args, result) in declarations.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = args, result

    def _check(self, code: int) -> None:
        if code:
            detail = (self.lib.oniGetExtendedError() or b'').decode('utf-8', errors='replace')
            raise RuntimeError(f'OpenNI错误 {code}: {detail}')

    def enumerate(self) -> list[dict[str, Any]]:
        devices, count = ct.POINTER(DeviceInfo)(), ct.c_int()
        self._check(self.lib.oniGetDeviceList(ct.byref(devices), ct.byref(count)))
        try:
            return [
                {**{key: bytes(getattr(devices[i], key)).decode('utf-8', errors='replace')
                    for key in ('uri', 'vendor', 'name', 'serial')},
                 'vid': devices[i].vid, 'pid': devices[i].pid}
                for i in range(count.value)
            ]
        finally:
            self.lib.oniReleaseDeviceList(devices)

    def connect(self, uri: str, width: int, height: int, fps: int) -> None:
        self._check(self.lib.oniDeviceOpen(uri.encode('utf-8'), ct.byref(self.device)))
        self._check(self.lib.oniDeviceCreateStream(self.device, 3, ct.byref(self.stream)))
        # Explicit millimetre depth; never infer the unit from numeric magnitude.
        mode = VideoMode(100, width, height, fps)
        self._check(self.lib.oniStreamSetProperty(self.stream, 3, ct.byref(mode), ct.sizeof(mode)))

    def start(self) -> None:
        self._check(self.lib.oniStreamStart(self.stream))
        self.running = True

    def camera_parameters(self) -> dict[str, Any]:
        """Raw factory metadata only; does not certify UVC geometry or units."""
        params = OBCameraParams()
        size = ct.c_int(ct.sizeof(params))
        self._check(self.lib.oniDeviceGetProperty(self.device, 14, ct.byref(params), ct.byref(size)))
        if size.value != ct.sizeof(params):
            raise RuntimeError('相机参数ABI长度不符')
        values = {name: list(getattr(params, name)) for name, _ in params._fields_}
        invalid = [name for name, value in values.items() if not np.isfinite(value).all()]
        if invalid:
            raise RuntimeError('出厂标定参数含无效数值：' + ', '.join(invalid))
        return {'sdk_property': 14, 'raw': values, 'verified': False,
                'notice': '原始SDK参数；左右相机、外参方向、平移单位及当前UVC模式须独立验证'}

    def read(self) -> tuple[np.ndarray, int, int]:
        index = ct.c_int()
        code = self.lib.oniWaitForAnyStream(ct.byref(self.stream), 1, ct.byref(index), 200)
        if code == 102:
            raise TimeoutError('等待深度帧超时')
        self._check(code)
        frame = ct.POINTER(NativeFrame)()
        self._check(self.lib.oniStreamReadFrame(self.stream, ct.byref(frame)))
        try:
            value = frame.contents
            if (value.mode.pixel_format != 100 or not value.data or value.width <= 0
                    or value.height <= 0 or value.stride < value.width * 2
                    or value.stride % 2 or value.size < value.stride * value.height):
                raise RuntimeError('原始深度格式或缓冲区无效')
            raw = np.ctypeslib.as_array(
                ct.cast(value.data, ct.POINTER(ct.c_uint16)),
                shape=(value.stride * value.height // 2,),
            ).reshape(value.height, value.stride // 2)[:, :value.width].copy()
            return raw, int(value.index), int(value.timestamp)
        finally:
            self.lib.oniFrameRelease(frame)

    def stop(self) -> None:
        if self.running:
            self.lib.oniStreamStop(self.stream)
            self.running = False

    def close(self) -> None:
        self.stop()
        if self.stream:
            self.lib.oniStreamDestroy(self.stream)
            self.stream = ct.c_void_p()
        if self.device:
            self.lib.oniDeviceClose(self.device)
            self.device = ct.c_void_p()
        if self.initialized:
            self.lib.oniShutdown()
            self.initialized = False
        self.dll_directory.close()
