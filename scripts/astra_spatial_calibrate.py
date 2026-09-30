"""Offline calibration evidence tools. Never silently approve factory metadata.

Run from project root. All paths are explicit; existing outputs are refused.
See Astra人体三维分析说明.md for the acquisition and evidence schemas.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from camera_capture import AstraCapture, CaptureConfig
from camera_capture.manager import DeviceLease, enumerate_color
from camera_capture.openni import OpenNIDepth
from camera_capture.processing import FramePreprocessor
from core.config import load_config
from core.spatial import Calibration, camera, unproject, project


def read(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path: str, value: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def capture(args: argparse.Namespace) -> None:
    config = load_config()
    if Path(args.output).exists():
        raise ValueError('输出已存在，拒绝覆盖')
    if args.factory_only:
        lease, native = DeviceLease(), None
        lease.acquire()
        try:
            native = OpenNIDepth(Path(config.astra.sdk_dir))
            devices = [item for item in native.enumerate() if item['vid'] == 0x2bc5 and item['pid'] == 0x0403]
            if len(devices) != 1:
                raise RuntimeError('出厂参数读取需要恰好一台Astra Pro深度设备')
            native.connect(devices[0]['uri'], 640, 480, 30)
            metadata = {'depth_device': devices[0], 'color_devices': enumerate_color()}
            try:
                metadata['factory'] = native.camera_parameters()
            except RuntimeError as exc:
                metadata['factory_error'] = str(exc)
            write(args.output, metadata)
            print('出厂参数探测已保存；' + metadata.get('factory_error', '尚未验证UVC配准。'))
            return
        finally:
            if native is not None:
                native.close()
            lease.release()
    capture = AstraCapture(CaptureConfig(Path(config.astra.sdk_dir), buffer_size=8))
    try:
        capture.connect()
        metadata = {'device_ids': capture.status()['device_ids'], 'size': [640, 480]}
        try:
            metadata['factory'] = capture.camera_parameters()
        except RuntimeError as exc:
            metadata['factory_error'] = str(exc)
        capture.start()
        processor = FramePreprocessor(config.multimodal)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            pair = processor.poll(capture)
            if pair is not None and pair.depth_quality['available']:
                metadata.update(pair_delta_s=pair.pair_delta_s, capture_id=pair.color.session_id,
                                color_timestamp_s=pair.color_time_s, depth_timestamp_s=pair.depth_time_s)
                # Original metres and RGB stay local. No compressed preview used as depth.
                target = Path(args.output)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as handle:
                    np.savez_compressed(handle, depth_m=pair.metric_depth.meters, color_bgr=pair.color.image,
                                        metadata=json.dumps(metadata, ensure_ascii=False))
                print('已保存一组软件配对RGB与原始米制深度；静态场景用于标定。')
                return
            time.sleep(0.01)
        raise RuntimeError('15秒内没有合格帧对')
    finally:
        capture.close()


def intrinsics(args: argparse.Namespace) -> None:
    specification = read(args.input)
    columns, rows = specification['inner_corners']
    side = float(specification['square_size_m'])
    if columns < 4 or rows < 4 or not 0.005 <= side <= 0.2:
        raise ValueError('内角点数量或实测方格尺寸无效')
    grid = np.zeros((columns * rows, 3), np.float32)
    grid[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * side
    objects, images, size = [], [], None
    for path in specification['images']:
        frame = cv2.imread(str(Path(args.input).parent / path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise ValueError(f'无法读取标定图片：{path}')
        current = (frame.shape[1], frame.shape[0])
        if size is not None and current != size:
            raise ValueError('标定图片尺寸不一致')
        size = current
        found, corners = cv2.findChessboardCornersSB(frame, (columns, rows))
        if found:
            objects.append(grid)
            images.append(corners)
    if len(images) < 12:
        raise ValueError('至少需要12张成功识别的不同位置和倾角棋盘图；深度内参必须使用同光路IR图，不能用伪彩深度图')
    error, matrix, distortion, _, _ = cv2.calibrateCamera(objects, images, size, None, None)
    write(args.output, dict(matrix=matrix.tolist(), distortion=distortion.ravel()[:5].tolist(), size=list(size),
                            reprojection_rms_px=error, image_count=len(images), verified=False))


def extrinsics(args: argparse.Namespace) -> None:
    data = read(args.input)
    uvz = np.asarray(data['fit_depth_uvz_m'], np.float64)
    rgb = np.asarray(data['fit_color_uv'], np.float64)
    if uvz.ndim != 2 or uvz.shape[1] != 3 or len(uvz) < 12 or rgb.shape != (len(uvz), 2):
        raise ValueError('至少需要12个拟合对应点；另留独立采集点验证')
    if not np.isfinite(uvz).all() or not np.isfinite(rgb).all() or np.any(uvz[:, 2] <= 0):
        raise ValueError('拟合对应点无效')
    xyz = unproject(uvz[:, :2], uvz[:, 2], data['depth'])
    matrix, distortion = camera(data['color'])
    ok, rotation, translation, inliers = cv2.solvePnPRansac(xyz, rgb, matrix, distortion, iterationsCount=300,
                                                           reprojectionError=3.0, confidence=0.999)
    if not ok or inliers is None or len(inliers) < 12 or len(inliers) / len(xyz) < 0.8:
        raise ValueError('外参拟合失败或对应点内点不足')
    data.update(schema=1, convention='opencv_depth_to_color_m', rotation=cv2.Rodrigues(rotation)[0].tolist(),
                translation_m=translation.ravel().tolist(), fit_capture_id=data['capture_id'])
    write(args.output, data)


def verify(args: argparse.Namespace) -> None:
    payload = read(args.input)
    if payload.get('fit_capture_id') == payload.get('verification', {}).get('capture_id'):
        raise ValueError('拟合与验证必须来自独立采集')
    calibrated = Calibration(payload, load_config().spatial)
    write(args.output, {'calibration_sha256': calibrated.identifier, 'reprojection_p95_px': calibrated.error_px,
                        'floor_plane_color_m': calibrated.floor.tolist(), 'status': 'evidence_passed',
                        'notice': '仍需现场核对真实目标和安装位置；不代表检测性能或硬件曝光同步验证'})


def floor(args: argparse.Namespace) -> None:
    """Build floor evidence from operator-selected, independent static captures."""
    payload = read(args.input)
    rotation = np.asarray(payload['rotation'], float)
    translation = np.asarray(payload['translation_m'], float)
    batches = []
    ids = []
    for item in (args.floor_capture, args.floor_validation):
        with np.load(item, allow_pickle=False) as source:
            metadata = json.loads(str(source['metadata']))
            if metadata['device_ids'] != payload['device_ids']:
                raise ValueError('地面采集设备不匹配')
            ids.append(metadata['capture_id'])
            depth = source['depth_m']
        if list(reversed(depth.shape)) != payload['depth']['size']:
            raise ValueError('地面深度分辨率不匹配')
        # Region is defined in raw depth pixels; the operator must identify floor.
        polygon = np.asarray(read(args.region)['depth_polygon'], np.int32)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
            raise ValueError('地面选区至少3个顶点')
        mask = np.zeros(depth.shape, np.uint8)
        cv2.fillPoly(mask, [polygon], 1)
        y, x = np.nonzero((mask > 0) & np.isfinite(depth) & (depth > 0))
        step = max(1, len(x) // 5000)
        xyz = unproject(np.column_stack((x[::step], y[::step])), depth[y[::step], x[::step]], payload['depth'])
        batches.append((xyz @ rotation.T + translation).tolist())
    if ids[0] == ids[1]:
        raise ValueError('地面拟合与验证需要独立采集')
    payload['floor'] = dict(points_color_m=batches[0], validation_points_color_m=batches[1],
                            operator_confirmed_floor=args.confirm_floor, placement_id=args.placement,
                            capture_ids=ids)
    Calibration(payload, load_config().spatial)
    write(args.output, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    cmd = commands.add_parser('capture')
    cmd.add_argument('--output', required=True)
    cmd.add_argument('--factory-only', action='store_true')
    cmd.set_defaults(action=capture)
    for name, action in (('intrinsics', intrinsics), ('extrinsics', extrinsics), ('verify', verify), ('floor', floor)):
        cmd = commands.add_parser(name)
        cmd.add_argument('--input', required=True)
        cmd.add_argument('--output', required=True)
        cmd.set_defaults(action=action)
        if name == 'floor':
            cmd.add_argument('--floor-capture', required=True)
            cmd.add_argument('--floor-validation', required=True)
            cmd.add_argument('--region', required=True)
            cmd.add_argument('--placement', required=True)
            cmd.add_argument('--confirm-floor', action='store_true', required=True)
    args = parser.parse_args()
    args.action(args)


if __name__ == '__main__':
    main()
