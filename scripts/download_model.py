from __future__ import annotations

import shutil
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    target = project_root / "models" / "yolo11n-pose.pt"
    if target.exists() and target.stat().st_size > 1_000_000:
        print(f"模型已经存在：{target}")
        return

    from ultralytics import YOLO

    target.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO("yolo11n-pose.pt")
    source = Path(str(getattr(model, "ckpt_path", project_root / "yolo11n-pose.pt")))
    if not source.is_absolute():
        source = Path.cwd() / source
    if not source.exists():
        fallback = project_root / "yolo11n-pose.pt"
        source = fallback if fallback.exists() else source
    if not source.exists():
        raise RuntimeError("模型已加载，但无法定位下载后的权重文件。")
    if source.resolve() != target.resolve():
        if source.parent.resolve() == project_root.resolve():
            shutil.move(str(source), str(target))
        else:
            shutil.copy2(source, target)
    print(f"模型准备完成：{target}")


if __name__ == "__main__":
    main()
