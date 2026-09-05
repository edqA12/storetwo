"""使用真实模型对一段左右组合视频执行多模态端到端烟测。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import load_config  # noqa: E402
from core.depth_pipeline import DepthPipeline  # noqa: E402
from core.detector import YoloPoseDetector  # noqa: E402
from core.fusion import FusionEngine  # noqa: E402
from core.multimodal import MultimodalPipeline  # noqa: E402
from core.pipeline import VisionPipeline  # noqa: E402
from core.video_processor import VideoProcessor  # noqa: E402
from services.database import EventRepository  # noqa: E402
from services.events import EventService  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="运行RGB+深度组合视频烟测")
    parser.add_argument("video", type=Path, help="左深度、右RGB的组合视频")
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()

    config = load_config()
    repository = EventRepository(config.resolve(config.storage.database))
    event_service = EventService(repository, config.resolve(config.storage.snapshots_dir))
    vision = VisionPipeline(config, YoloPoseDetector(config))
    multimodal = MultimodalPipeline(
        config,
        vision,
        DepthPipeline(config.multimodal),
        FusionEngine(config.multimodal),
    )
    processor = VideoProcessor(config, vision, event_service, multimodal)
    result = processor.process(
        args.video,
        confidence=args.confidence,
        frame_stride=args.stride,
        input_mode="rgb_depth",
    )
    print(result.message)
    print(f"输出：{result.output_path}")
    print(f"最高融合跌倒事件证据分：{result.max_risk:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
