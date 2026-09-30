"""Hardware software-pairing check; stores statistics only, never media."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera_capture import AstraCapture, CaptureConfig
from camera_capture.processing import FramePreprocessor, ProcessingConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'config.yaml')
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 3600:
        parser.error('seconds must be between 1 and 3600')
    raw = yaml.safe_load(args.config.read_text(encoding='utf-8'))['multimodal']
    settings = ProcessingConfig(**{f.name: raw[f.name] for f in fields(ProcessingConfig) if f.name in raw})
    processor = FramePreprocessor(settings)
    deltas, ratios, elapsed, usable, resets = [], [], [], 0, {}
    with AstraCapture(CaptureConfig(args.sdk_dir, buffer_size=8)) as capture:
        capture.connect()
        capture.start()
        start = time.monotonic()
        initial = capture.status()['frames']
        while time.monotonic() - start < args.seconds:
            pair = processor.poll(capture)
            if capture.status()['state'] != 'running':
                raise RuntimeError(capture.status())
            if pair:
                deltas.append(pair.pair_delta_s * 1000)
                ratios.append(pair.metric_depth.valid_ratio)
                usable += bool(pair.depth_quality['usable_for_motion'])
                reason = pair.depth_quality['reset_reason']
                if reason:
                    resets[reason] = resets.get(reason, 0) + 1
                if pair.depth_quality['elapsed_s'] is not None:
                    elapsed.append(pair.depth_quality['elapsed_s'])
            time.sleep(.01)
        counts = capture.status()['frames']
        report = {'duration_s': time.monotonic() - start, 'pairs': len(deltas),
                  'received_during_window': {k: counts[k] - initial[k] for k in counts},
                  'usable_for_motion': usable, 'quality_resets': resets,
                  'pair_delta_ms_p50_p95_max': np.percentile(deltas, [50, 95, 100]).tolist() if deltas else None,
                  'valid_ratio_min_mean': [min(ratios), float(np.mean(ratios))] if ratios else None,
                  'elapsed_s_min_max': [min(elapsed), max(elapsed)] if elapsed else None,
                  'rejections': dict(processor.rejections),
                  'timing_basis': 'software_host_receive_estimate', 'exposure_alignment_verified': False,
                  'spatial_registration': False}
    report['final_state'] = capture.status()['state']
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not deltas:
        raise RuntimeError('没有得到有效的软件配对')


if __name__ == '__main__':
    main()
