"""Explicit hardware smoke test; never imports the app or writes media/database."""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera_capture import AstraCapture, CaptureConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk-dir', type=Path, required=True, help='Directory containing OpenNI2.dll and OpenNI2/Drivers')
    parser.add_argument('--seconds', type=float, default=15.0)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--restart', action='store_true', help='Also verify stop/start and disconnect/reconnect')
    parser.add_argument('--output', type=Path, help='Optional statistics-only JSON file')
    args = parser.parse_args()
    if not 1 <= args.seconds <= 3600:
        parser.error('--seconds must be between 1 and 3600')
    config = CaptureConfig(args.sdk_dir)
    if args.list:
        print(json.dumps(AstraCapture.enumerate_devices(config), ensure_ascii=False, indent=2))
        return
    report: dict = {'scope': 'acquisition_only', 'runs': []}
    with AstraCapture(config) as capture:
        report['device'] = asdict(capture.connect())
        for run in range(3 if args.restart else 1):
            if run == 2:
                capture.close()
                capture.connect()
            capture.start()
            initial = capture.status()
            deadline = time.monotonic() + args.seconds
            sample: dict = {}
            while time.monotonic() < deadline:
                for source in ('color', 'depth'):
                    frame = capture.latest(source)
                    if frame:
                        sample[source] = {field.name: getattr(frame, field.name) for field in fields(frame) if field.name != 'image'}
                        sample[source].update(shape=list(frame.image.shape), dtype=str(frame.image.dtype))
                if capture.status()['state'] != 'running':
                    raise RuntimeError(capture.status())
                time.sleep(0.1)
            status = capture.status()
            if not all(status['fresh'].values()) or set(sample) != {'color', 'depth'}:
                raise RuntimeError('双流未持续更新')
            report['runs'].append({'initial': initial, 'final': status, 'last_frames': sample})
            capture.stop()
            if capture.latest('depth') is not None or capture.latest('color') is not None:
                raise RuntimeError('停止后仍返回旧帧')
    report['final_state'] = capture.status()['state']
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
