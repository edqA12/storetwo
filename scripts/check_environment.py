from __future__ import annotations

import importlib
import sys
from pathlib import Path


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    required = ["cv2", "gradio", "ultralytics", "numpy", "plotly", "yaml", "PIL", "imageio_ffmpeg"]
    failed = False
    for package in required:
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "已安装")
            print(f"[OK] {package}: {version}")
        except Exception as exc:
            failed = True
            print(f"[缺失] {package}: {exc}")
    model = Path(__file__).resolve().parents[1] / "models" / "yolo11n-pose.pt"
    print(f"模型: {'已就绪' if model.exists() else '尚未下载'} ({model})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
