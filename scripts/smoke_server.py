"""用于本地界面检查的无浏览器启动脚本。"""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import ALARM_FILE, CONFIG, CSS, build_app  # noqa: E402
import gradio as gr  # noqa: E402


def main() -> None:
    port = int(os.environ.get("MUAN_SMOKE_PORT", "7861"))
    app = build_app()
    app.queue(default_concurrency_limit=1, max_size=8).launch(
        server_name="127.0.0.1",
        server_port=port,
        inbrowser=False,
        share=False,
        show_error=True,
        allowed_paths=[
            str(CONFIG.resolve(CONFIG.storage.results_dir).resolve()),
            str(CONFIG.resolve(CONFIG.storage.snapshots_dir).resolve()),
            str(CONFIG.resolve(CONFIG.storage.exports_dir).resolve()),
            str(ALARM_FILE.resolve()),
        ],
        css=CSS,
        theme=gr.themes.Soft(primary_hue="blue"),
    )


if __name__ == "__main__":
    main()
