"""Local camera/model smoke check. No database, media saving or assistant API."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera_capture import AstraCapture
from core.config import load_config
from core.depth_pipeline import DepthPipeline
from core.detector import YoloPoseDetector
from core.fusion import FusionEngine
from core.multimodal import MultimodalPipeline
from core.pipeline import VisionPipeline
from services.astra_live import AstraLiveMonitor


class FaultGate:
    """Inject unavailable streams above real capture; never claim a USB unplug."""
    def __init__(self, config):
        self.inner = AstraCapture(config)
        self.mode = 'normal'

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def status(self):
        result = self.inner.status()
        if self.mode == 'no_depth':
            result.update(state='degraded', source_errors={'depth': 'injected'})
        elif self.mode == 'no_streams':
            result.update(state='error')
        return result

    def buffered_frames(self):
        return [frame for frame in self.inner.buffered_frames()
                if self.mode == 'normal' or (self.mode == 'no_depth' and frame.source == 'color')]

    def latest(self, source):
        return None if self.mode == 'no_streams' else self.inner.latest(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error('existing local model required')
    config = load_config()
    config.model.path = str(args.model.resolve())
    detector = YoloPoseDetector(config)
    detector.ensure_loaded()
    pipeline = MultimodalPipeline(config, VisionPipeline(config, detector),
                                  DepthPipeline(config.multimodal), FusionEngine(config.multimodal))
    monitor = AstraLiveMonitor(config, pipeline, FaultGate)
    report = {}
    try:
        monitor.start('smoke')
        for mode, seconds in [('normal', 12), ('no_depth', 4), ('no_streams', 1), ('reconnect', 6)]:
            if mode == 'reconnect':
                monitor.stop('smoke')
                monitor.start('smoke')
            else:
                monitor.capture.mode = mode
            counts, weights, deltas, statuses = Counter(), [], [], Counter()
            start = time.monotonic()
            while time.monotonic() - start < seconds:
                update = monitor.tick('smoke', config.model.confidence)
                statuses[update.status] += 1
                if update.result is None:
                    counts['no_result'] += 1
                elif update.new_result:
                    result = update.result
                    counts[result.fusion.method] += 1
                    counts['person_frames'] += bool(result.decisions)
                    weights.append(result.fusion.modality_weights.get('depth', 0))
                    delta = result.diagnostics.get('software_pair_delta_s')
                    if delta is not None:
                        deltas.append(delta * 1000)
                time.sleep(.2)
            report[mode] = {'counts': dict(counts), 'depth_weight_max': max(weights, default=0),
                            'pair_delta_ms_max': max(deltas, default=None), 'statuses': dict(statuses)}
    finally:
        monitor.stop('smoke')
    report['checks'] = {
        'real_depth_fused': report['normal']['depth_weight_max'] > 0,
        'injected_depth_loss_falls_back': report['no_depth']['counts'].get('visual_fallback', 0) > 0 and report['no_depth']['depth_weight_max'] == 0,
        'injected_total_loss_interrupts': report['no_streams']['counts'].get('no_result', 0) > 0,
        'reconnect_fuses_again': report['reconnect']['depth_weight_max'] > 0,
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(payload, encoding='utf-8')
    print(payload)
    if not all(report['checks'].values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
