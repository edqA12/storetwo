from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np
import plotly.graph_objects as go

from core.config import load_config
from core.detector import ModelUnavailableError, YoloPoseDetector
from core.pipeline import VisionPipeline, new_stream_state
from core.video_processor import VideoProcessor
from services.alarm import ensure_alarm_wav
from services.database import EVENT_COLUMNS, EventRepository
from services.events import EventService


CONFIG = load_config()
REPOSITORY = EventRepository(CONFIG.resolve(CONFIG.storage.database))
EVENT_SERVICE = EventService(REPOSITORY, CONFIG.resolve(CONFIG.storage.snapshots_dir))
DETECTOR = YoloPoseDetector(CONFIG)
PIPELINE = VisionPipeline(CONFIG, DETECTOR)
VIDEO_PROCESSOR = VideoProcessor(CONFIG, PIPELINE, EVENT_SERVICE)
ALARM_FILE = ensure_alarm_wav(CONFIG.resolve("data/alarm.wav"))

UPLOAD_EVENT_HEADERS = ["事件ID", "视频时间", "事件类型", "风险值", "触发原因"]


def _empty_risk_figure(message: str = "等待检测") -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        template="plotly_white",
        height=330,
        margin=dict(l=45, r=25, t=45, b=40),
        title=message,
        xaxis_title="视频时间（秒）",
        yaxis_title="风险值",
        yaxis=dict(range=[0, 100]),
    )
    return figure


def _risk_figure(samples: list[tuple[float, float]]) -> go.Figure:
    if not samples:
        return _empty_risk_figure("未获得风险数据")
    x_values = [item[0] for item in samples]
    y_values = [item[1] for item in samples]
    figure = go.Figure()
    figure.add_hrect(y0=80, y1=100, fillcolor="#fee2e2", opacity=0.6, line_width=0)
    figure.add_hrect(y0=60, y1=80, fillcolor="#fef3c7", opacity=0.55, line_width=0)
    figure.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode="lines",
        name="跌倒风险",
        line=dict(color="#0066cc", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(0,102,204,0.10)",
    ))
    figure.update_layout(
        template="plotly_white",
        height=330,
        margin=dict(l=45, r=25, t=45, b=40),
        title="视频风险变化曲线",
        xaxis_title="视频时间（秒）",
        yaxis_title="风险值",
        yaxis=dict(range=[0, 100]),
        hovermode="x unified",
        showlegend=False,
    )
    return figure


def _normalise_video_input(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("path") or value.get("video")
    if isinstance(value, (tuple, list)) and value:
        return str(value[0])
    return str(value)


def process_uploaded_video(
    video_value: Any,
    confidence: float,
    frame_stride: int,
    progress: gr.Progress = gr.Progress(),
):
    path = _normalise_video_input(video_value)
    if not path:
        return None, "❌ 请先选择一个视频文件。", 0.0, 0, _empty_risk_figure(), [], None
    try:
        progress(0.01, desc="正在读取视频……")
        result = VIDEO_PROCESSOR.process(
            path,
            confidence=float(confidence),
            frame_stride=int(frame_stride),
            progress=lambda fraction, text: progress(fraction, desc=text),
        )
        status = f"✅ {result.message}"
        return (
            str(result.output_path), status, result.max_risk, result.event_count,
            _risk_figure(result.risk_samples), result.event_rows, str(result.output_path),
        )
    except (ValueError, RuntimeError, ModelUnavailableError) as exc:
        return None, f"❌ {exc}", 0.0, 0, _empty_risk_figure("检测未完成"), [], None
    except Exception as exc:  # pragma: no cover - final UI guard
        return None, f"❌ 检测过程中发生异常：{exc}", 0.0, 0, _empty_risk_figure("检测未完成"), [], None


def process_camera_frame(frame_rgb: np.ndarray | None, state: dict[str, Any] | None, confidence: float):
    session = state or new_stream_state()
    if frame_rgb is None:
        return None, "### 等待摄像头画面", 0.0, 0.0, session.get("last_alert_text", "暂无告警"), None, session
    try:
        array = np.asarray(frame_rgb)
        if array.ndim != 3 or array.shape[2] < 3:
            raise ValueError("摄像头画面格式不正确。")
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(array[:, :, :3], cv2.COLOR_RGB2BGR)
        result, session = PIPELINE.process_frame(
            frame_bgr,
            timestamp_s=time.monotonic(),
            stream_state=session,
            confidence=float(confidence),
        )
        alarm_output: str | None = None
        for event in result.new_events:
            event_type = str(event.get("event_type", "跌倒告警"))
            try:
                event_id = EVENT_SERVICE.record(event, result.annotated_bgr, "实时摄像头", "本机摄像头")
                session["last_db_event_id"] = event_id
                icon = "🚨" if event_type == "跌倒告警" else "⚠️"
                session["last_alert_text"] = (
                    f"{icon} **{event_type} #{event_id}**　风险 {float(event['risk']):.1f}　"
                    f"原因：{'；'.join(event.get('reasons', []))}"
                )
            except Exception as persist_error:
                session["last_alert_text"] = (
                    f"⚠️ **已检测到{event_type}，但事件保存失败**　风险 {float(event['risk']):.1f}。"
                    f"请立即人工确认；记录错误：{persist_error}"
                )
            if event_type == "跌倒告警":
                alarm_output = str(ALARM_FILE)

        color = "#dc2626" if result.max_risk >= 80 else "#d97706" if result.max_risk >= 60 else "#15803d"
        status = (
            f"### <span style='color:{color}'>{result.overall_label}</span>  "
            f"\n检测到 {result.diagnostics.get('persons', 0)} 人 · "
            f"单帧耗时 {result.diagnostics.get('total_ms', 0):.0f} ms"
        )
        annotated_rgb = cv2.cvtColor(result.annotated_bgr, cv2.COLOR_BGR2RGB)
        return (
            annotated_rgb,
            status,
            result.max_risk,
            float(result.diagnostics.get("processing_fps", 0.0)),
            session.get("last_alert_text", "暂无告警"),
            alarm_output,
            session,
        )
    except (ValueError, ModelUnavailableError) as exc:
        return frame_rgb, f"### ⚠️ {exc}", 0.0, 0.0, session.get("last_alert_text", "暂无告警"), None, session
    except Exception as exc:  # pragma: no cover - final UI guard
        return frame_rgb, f"### ⚠️ 摄像头检测异常：{exc}", 0.0, 0.0, session.get("last_alert_text", "暂无告警"), None, session


def reset_camera_state():
    return None, "### 监测状态已重置", 0.0, 0.0, "暂无告警", None, new_stream_state()


def update_last_camera_event(state: dict[str, Any] | None, target_status: str):
    session = state or new_stream_state()
    event_id = session.get("last_db_event_id")
    if not event_id:
        return "当前会话还没有可处理的告警。", session
    if REPOSITORY.update_status(int(event_id), target_status):
        message = f"事件 #{event_id} 已标记为“{target_status}”。"
        session["last_alert_text"] = message
        return message, session
    return f"未找到事件 #{event_id}。", session


def refresh_events():
    stats = REPOSITORY.statistics()
    summary = "　".join(f"**{key}：** {value}" for key, value in stats.items())
    return REPOSITORY.table_rows(), summary


def select_event(evt: gr.SelectData):
    row_value = getattr(evt, "row_value", None)
    event_id: int | None = None
    if isinstance(row_value, (list, tuple)) and row_value:
        try:
            event_id = int(row_value[0])
        except (TypeError, ValueError):
            event_id = None
    if event_id is None:
        value = getattr(evt, "value", None)
        try:
            event_id = int(value)
        except (TypeError, ValueError):
            return None, "未能识别所选事件。", None
    event = REPOSITORY.get_event(event_id)
    if not event:
        return None, "所选事件不存在。", None
    details = (
        f"### 事件 #{event_id}\n"
        f"- 时间：{event['occurred_at']}\n"
        f"- 来源：{event['source']}\n"
        f"- 风险值：{float(event['risk']):.1f}\n"
        f"- 状态：{event['status']}\n"
        f"- 原因：{event['reasons'] or '未记录'}"
    )
    snapshot = event.get("snapshot_path")
    return event_id, details, snapshot if snapshot and Path(snapshot).exists() else None


def update_selected_event(event_id: int | float | None, status: str):
    if event_id is None:
        return "请先在表格中选择一条事件。", REPOSITORY.table_rows(), None, "请从表格中选择一条事件。", None, refresh_events()[1]
    try:
        selected = int(event_id)
    except (TypeError, ValueError):
        return "事件编号无效。", REPOSITORY.table_rows(), None, "请从表格中选择一条事件。", None, refresh_events()[1]
    updated = REPOSITORY.update_status(selected, status)
    message = f"事件 #{selected} 已标记为“{status}”。" if updated else f"未找到事件 #{selected}。"
    return message, REPOSITORY.table_rows(), None, "请从表格中选择一条事件。", None, refresh_events()[1]


def export_events():
    export_dir = CONFIG.resolve(CONFIG.storage.exports_dir)
    path = export_dir / f"events_{uuid.uuid4().hex[:8]}.csv"
    REPOSITORY.export_csv(path)
    return str(path), f"已导出 {len(REPOSITORY.list_events(limit=5000))} 条记录。"


CSS = """
:root {
  --muan-blue: #0066cc;
  --muan-blue-focus: #0071e3;
  --muan-ink: #1d1d1f;
  --muan-muted: #6e6e73;
  --muan-canvas: #ffffff;
  --muan-parchment: #f5f5f7;
  --muan-pearl: #fafafc;
  --muan-hairline: #e0e0e0;
}

body, .gradio-container {
  background: var(--muan-parchment) !important;
  color: var(--muan-ink) !important;
  font-family: Inter, "SF Pro Text", system-ui, -apple-system, BlinkMacSystemFont,
    "Segoe UI", "Microsoft YaHei UI", sans-serif !important;
}
.gradio-container {
  max-width: none !important;
  padding: 0 !important;
}
.main, .contain {max-width: none !important; padding: 0 !important;}

#muan-global-nav {
  height: 44px;
  padding: 0 max(24px, calc((100vw - 1440px) / 2));
  background: #000000;
  color: #ffffff;
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 12px;
  letter-spacing: -0.01em;
}
#muan-global-nav .brand {font-weight: 600; letter-spacing: -0.02em;}
#muan-global-nav .nav-meta {color: #cccccc;}

#muan-sub-nav {
  position: sticky;
  top: 0;
  z-index: 50;
  min-height: 52px;
  padding: 0 max(24px, calc((100vw - 1440px) / 2));
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
  background: rgba(245, 245, 247, 0.82);
  border-bottom: 1px solid rgba(0, 0, 0, 0.08);
  backdrop-filter: saturate(180%) blur(20px);
  -webkit-backdrop-filter: saturate(180%) blur(20px);
}
#muan-sub-nav .product-name {font-size: 21px; font-weight: 600; letter-spacing: -0.02em;}
#muan-sub-nav .product-meta {font-size: 12px; color: var(--muan-muted);}

#hero {
  min-height: 340px;
  padding: 76px 32px 64px;
  background: var(--muan-canvas);
  color: var(--muan-ink);
  text-align: center;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}
#hero .eyebrow {
  margin-bottom: 12px;
  color: var(--muan-blue);
  font-size: 14px;
  font-weight: 600;
  letter-spacing: .06em;
}
#hero h1 {
  margin: 0;
  max-width: 1100px;
  color: var(--muan-ink) !important;
  font-family: "SF Pro Display", Inter, system-ui, -apple-system, sans-serif !important;
  font-size: clamp(38px, 5vw, 56px);
  font-weight: 600;
  line-height: 1.07;
  letter-spacing: -0.035em;
}
#hero .hero-lead {
  margin: 20px 0 0;
  color: var(--muan-ink) !important;
  font-size: clamp(20px, 2.4vw, 28px);
  font-weight: 400;
  line-height: 1.25;
  letter-spacing: -0.02em;
}
#hero .hero-note {
  margin: 18px 0 0;
  color: var(--muan-muted) !important;
  font-size: 14px;
  line-height: 1.43;
}

#app-shell {
  max-width: 1440px;
  margin: 0 auto;
  padding: 0 24px 80px;
}
.section-intro {
  max-width: 920px;
  margin: 0 auto 32px;
  text-align: center;
}
.section-intro h2 {
  margin: 0 0 10px;
  font-size: clamp(28px, 3.2vw, 40px);
  font-weight: 600;
  line-height: 1.1;
  letter-spacing: -0.03em;
}
.section-intro p {margin: 0; color: var(--muan-muted); font-size: 17px; line-height: 1.47;}

.tabs {background: transparent !important;}
.tab-container[role="tablist"] {
  position: sticky;
  top: 52px;
  z-index: 40;
  display: flex !important;
  justify-content: center !important;
  gap: 6px !important;
  padding: 12px 0 !important;
  background: rgba(245, 245, 247, .88) !important;
  border: 0 !important;
  backdrop-filter: saturate(180%) blur(20px);
}
.tab-container[role="tablist"] button[role="tab"] {
  min-height: 44px !important;
  padding: 10px 20px !important;
  border: 0 !important;
  border-radius: 9999px !important;
  color: var(--muan-ink) !important;
  font-size: 14px !important;
  font-weight: 400 !important;
}
.tab-container[role="tablist"] button[role="tab"].selected {
  background: var(--muan-ink) !important;
  color: #ffffff !important;
}
[role="tabpanel"] {
  padding: 54px 0 0 !important;
  border: 0 !important;
  background: transparent !important;
}

.muan-card, .metric-card, .gradio-container .form,
.gradio-container .panel, .gradio-container .block {
  border-color: var(--muan-hairline) !important;
}
.muan-card {
  padding: 24px !important;
  border: 1px solid var(--muan-hairline) !important;
  border-radius: 18px !important;
  background: var(--muan-canvas) !important;
}
.metric-card {
  min-height: 112px;
  padding: 18px 22px !important;
  border: 1px solid var(--muan-hairline) !important;
  border-radius: 18px !important;
  background: var(--muan-canvas) !important;
}
.metric-card input {font-size: 30px !important; font-weight: 600 !important; letter-spacing: -0.03em;}
.metric-card label span {color: var(--muan-muted) !important; font-size: 14px !important;}

.warning-note {
  margin: 0 0 24px;
  padding: 17px 20px;
  border: 1px solid var(--muan-hairline);
  border-radius: 18px;
  background: var(--muan-canvas);
  color: var(--muan-muted);
  font-size: 14px;
  line-height: 1.43;
}
.warning-note strong {color: var(--muan-ink);}

.gradio-container button {
  min-height: 44px !important;
  border-radius: 9999px !important;
  font-size: 15px !important;
  font-weight: 400 !important;
  transition: transform .12s ease, background-color .12s ease !important;
}
.gradio-container button:active {transform: scale(.95);}
.gradio-container button:focus-visible {
  outline: 2px solid var(--muan-blue-focus) !important;
  outline-offset: 2px;
}
.gradio-container button.primary {
  border-color: var(--muan-blue) !important;
  background: var(--muan-blue) !important;
  color: #ffffff !important;
}
.gradio-container button.secondary {
  border: 1px solid var(--muan-blue) !important;
  background: var(--muan-canvas) !important;
  color: var(--muan-blue) !important;
}

.gradio-container input, .gradio-container textarea, .gradio-container select {
  border-radius: 11px !important;
  font-size: 16px !important;
}
.gradio-container input[type="range"] {accent-color: var(--muan-blue);}
.gradio-container a {color: var(--muan-blue) !important;}

#upload-video, #result-video, #camera-input, #camera-output {
  overflow: hidden;
  border: 1px solid var(--muan-hairline) !important;
  border-radius: 18px !important;
  background: #000000 !important;
}
#events-table {overflow: hidden; border-radius: 18px !important; background: var(--muan-canvas);}
.gradio-container table {font-size: 14px !important;}
.gradio-container thead {background: var(--muan-parchment) !important;}

#status-panel, #alert-panel, #event-detail {
  min-height: 76px;
  padding: 17px 20px !important;
  border: 1px solid var(--muan-hairline) !important;
  border-radius: 18px !important;
  background: var(--muan-canvas) !important;
}
#event-stats {padding: 17px 20px; border-radius: 18px; background: #1d1d1f;}
#event-stats, #event-stats p, #event-stats span, #event-stats strong {
  color: #ffffff !important;
}
#event-stats strong {font-size: 17px;}

#muan-footer {
  margin-top: 64px;
  padding: 48px 24px;
  border-top: 1px solid var(--muan-hairline);
  color: var(--muan-muted);
  font-size: 12px;
  line-height: 1.6;
  text-align: center;
}
footer {display: none !important;}

@media (max-width: 833px) {
  #muan-global-nav .nav-meta {display: none;}
  #muan-sub-nav .product-meta {display: none;}
  #hero {min-height: 300px; padding: 56px 24px 48px;}
  #app-shell {padding: 0 16px 56px;}
  [role="tabpanel"] {padding-top: 40px !important;}
  .muan-card {padding: 18px !important;}
}
@media (max-width: 640px) {
  #muan-global-nav {padding: 0 16px;}
  #muan-sub-nav {padding: 0 16px;}
  #hero h1 {font-size: 34px;}
  #hero .hero-lead {font-size: 20px;}
  .tab-container[role="tablist"] {justify-content: flex-start !important; overflow-x: auto;}
  .tab-container[role="tablist"] button[role="tab"] {flex: 0 0 auto !important; padding: 10px 16px !important;}
  .section-intro h2 {font-size: 30px;}
}
@media (max-width: 419px) {
  #hero h1 {font-size: 28px;}
  #hero {min-height: 270px; padding-inline: 18px;}
}
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="暮安智护·视觉跌倒检测MVP") as app:
        gr.HTML(
            "<nav id='muan-global-nav'><span class='brand'>暮安智护</span>"
            "<span class='nav-meta'>本地 AI 安全监测</span></nav>"
            "<div id='muan-sub-nav'><span class='product-name'>视觉跌倒检测</span>"
            "<span class='product-meta'>YOLO Pose · 时序姿态判断 · 本地隐私处理</span></div>"
        )
        gr.HTML(
            "<section id='hero'><span class='eyebrow'>MUAN VISION · MVP</span>"
            "<h1>让每一次异常姿态，都被及时看见。</h1>"
            "<p class='hero-lead'>视频分析、实时监测与告警管理，一体化完成。</p>"
            "<p class='hero-note'>初代研究原型 · 数据默认留在本机 · 不属于医疗器械或临床诊断工具</p></section>"
        )

        with gr.Column(elem_id="app-shell"):
            with gr.Tabs():
                with gr.Tab("视频检测"):
                    gr.HTML(
                        "<div class='section-intro'><h2>分析一段视频</h2>"
                        "<p>上传本地视频，系统将标注人体骨架、风险变化与跌倒事件。</p></div>"
                    )
                    gr.HTML(
                        "<div class='warning-note'><strong>运行提示</strong>　"
                        "上传分析前请停止实时摄像头，避免 CPU 推理任务互相等待。</div>"
                    )
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=5, elem_classes="muan-card"):
                            upload_video = gr.Video(
                                label="上传待检测视频",
                                sources=["upload"],
                                elem_id="upload-video",
                            )
                            with gr.Row():
                                upload_confidence = gr.Slider(
                                    0.15, 0.75, value=CONFIG.model.confidence, step=0.05,
                                    label="人体检测置信度",
                                )
                                frame_stride = gr.Radio(
                                    choices=[
                                        ("逐帧检测（最准）", 1),
                                        ("每2帧检测", 2),
                                        ("每3帧检测（最快）", 3),
                                    ],
                                    value=CONFIG.video.default_frame_stride,
                                    label="处理速度",
                                )
                            analyse_button = gr.Button("开始检测", variant="primary", size="lg")
                            upload_status = gr.Markdown("请选择视频后开始检测。")
                        with gr.Column(scale=6, elem_classes="muan-card"):
                            result_video = gr.Video(
                                label="标注结果视频", format="mp4", elem_id="result-video"
                            )
                            result_download = gr.File(label="下载检测结果")
                    with gr.Row():
                        max_risk = gr.Number(
                            label="最高风险值", value=0, interactive=False,
                            elem_classes="metric-card",
                        )
                        event_count = gr.Number(
                            label="告警事件数量", value=0, interactive=False,
                            elem_classes="metric-card",
                        )
                    risk_plot = gr.Plot(value=_empty_risk_figure())
                    upload_events = gr.Dataframe(
                        headers=UPLOAD_EVENT_HEADERS,
                        value=[],
                        interactive=False,
                        label="本次视频事件",
                    )
                    analyse_button.click(
                        fn=process_uploaded_video,
                        inputs=[upload_video, upload_confidence, frame_stride],
                        outputs=[
                            result_video, upload_status, max_risk, event_count,
                            risk_plot, upload_events, result_download,
                        ],
                        concurrency_limit=1,
                        concurrency_id="pose-inference",
                    )

                with gr.Tab("实时监测"):
                    gr.HTML(
                        "<div class='section-intro'><h2>打开实时守护</h2>"
                        "<p>通过浏览器摄像头连续观察人体姿态，并在证据链成立时发出告警。</p></div>"
                    )
                    camera_state = gr.State(value=new_stream_state())
                    gr.Markdown(
                        "<div class='warning-note'><strong>拍摄建议</strong>　"
                        "请允许浏览器使用摄像头，并让人物全身进入画面。"
                        "声音告警可能受浏览器自动播放规则限制，请以红色画面提示为主要告警。</div>"
                    )
                    with gr.Row():
                        camera_input = gr.Image(
                            label="摄像头",
                            sources=["webcam"],
                            type="numpy",
                            streaming=True,
                            elem_id="camera-input",
                        )
                        camera_output = gr.Image(
                            label="实时检测结果",
                            type="numpy",
                            streaming=True,
                            elem_id="camera-output",
                        )
                    with gr.Row():
                        camera_confidence = gr.Slider(
                            0.15, 0.75, value=CONFIG.model.confidence, step=0.05,
                            label="人体检测置信度",
                        )
                        camera_risk = gr.Number(
                            label="当前风险值", value=0, interactive=False,
                            elem_classes="metric-card",
                        )
                        camera_fps = gr.Number(
                            label="处理帧率 FPS", value=0, interactive=False,
                            elem_classes="metric-card",
                        )
                    camera_status = gr.Markdown(
                        "### 等待摄像头画面", elem_id="status-panel"
                    )
                    recent_alert = gr.Markdown("暂无告警", elem_id="alert-panel")
                    alarm_audio = gr.Audio(label="声音告警", autoplay=True, interactive=False)
                    with gr.Row():
                        reset_button = gr.Button("重置监测状态")
                        safe_button = gr.Button("我没事（标记误报）", variant="secondary")
                        confirm_button = gr.Button("确认告警", variant="primary")

                    camera_input.stream(
                        fn=process_camera_frame,
                        inputs=[camera_input, camera_state, camera_confidence],
                        outputs=[
                            camera_output, camera_status, camera_risk, camera_fps,
                            recent_alert, alarm_audio, camera_state,
                        ],
                        time_limit=60,
                        stream_every=0.2,
                        trigger_mode="always_last",
                        concurrency_limit=1,
                        concurrency_id="pose-inference",
                        show_progress="hidden",
                    )
                    reset_button.click(
                        fn=reset_camera_state,
                        outputs=[
                            camera_output, camera_status, camera_risk, camera_fps,
                            recent_alert, alarm_audio, camera_state,
                        ],
                    )
                    camera_input.clear(
                        fn=reset_camera_state,
                        outputs=[
                            camera_output, camera_status, camera_risk, camera_fps,
                            recent_alert, alarm_audio, camera_state,
                        ],
                    )
                    safe_button.click(
                        fn=lambda state: update_last_camera_event(state, "误报"),
                        inputs=[camera_state],
                        outputs=[recent_alert, camera_state],
                    )
                    confirm_button.click(
                        fn=lambda state: update_last_camera_event(state, "已确认"),
                        inputs=[camera_state],
                        outputs=[recent_alert, camera_state],
                    )

                with gr.Tab("事件中心"):
                    gr.HTML(
                        "<div class='section-intro'><h2>管理告警事件</h2>"
                        "<p>复核告警截图、更新处理状态，并导出用于实验分析的事件记录。</p></div>"
                    )
                    selected_event_id = gr.State(value=None)
                    stats_markdown = gr.Markdown(elem_id="event-stats")
                    with gr.Row():
                        refresh_button = gr.Button("刷新记录", variant="primary")
                        export_button = gr.Button("导出CSV")
                        export_file = gr.File(label="事件记录文件")
                    export_status = gr.Markdown()
                    events_table = gr.Dataframe(
                        headers=EVENT_COLUMNS,
                        value=REPOSITORY.table_rows(),
                        interactive=False,
                        label="历史告警记录（点击一行查看详情）",
                        elem_id="events-table",
                    )
                    with gr.Row():
                        selected_details = gr.Markdown(
                            "请从表格中选择一条事件。", elem_id="event-detail"
                        )
                        selected_snapshot = gr.Image(label="告警截图", type="filepath")
                    with gr.Row():
                        mark_confirmed = gr.Button("标记为已确认")
                        mark_false = gr.Button("标记为误报")
                        mark_handled = gr.Button("标记为已处理")
                    action_status = gr.Markdown()

                    app.load(
                        fn=refresh_events,
                        outputs=[events_table, stats_markdown],
                        show_progress="hidden",
                    )
                    refresh_button.click(
                        fn=refresh_events,
                        outputs=[events_table, stats_markdown],
                        show_progress="hidden",
                    )
                    events_table.select(
                        fn=select_event,
                        outputs=[selected_event_id, selected_details, selected_snapshot],
                    )
                    mark_confirmed.click(
                        fn=lambda event_id: update_selected_event(event_id, "已确认"),
                        inputs=[selected_event_id],
                        outputs=[
                            action_status, events_table, selected_event_id,
                            selected_details, selected_snapshot, stats_markdown,
                        ],
                    )
                    mark_false.click(
                        fn=lambda event_id: update_selected_event(event_id, "误报"),
                        inputs=[selected_event_id],
                        outputs=[
                            action_status, events_table, selected_event_id,
                            selected_details, selected_snapshot, stats_markdown,
                        ],
                    )
                    mark_handled.click(
                        fn=lambda event_id: update_selected_event(event_id, "已处理"),
                        inputs=[selected_event_id],
                        outputs=[
                            action_status, events_table, selected_event_id,
                            selected_details, selected_snapshot, stats_markdown,
                        ],
                    )
                    export_button.click(
                        fn=export_events, outputs=[export_file, export_status]
                    )

            gr.HTML(
                "<div id='muan-footer'><strong>安全说明</strong><br>"
                "本系统用于学生科研和演示。跌倒模拟必须由健康成年人在体操垫、护具和观察员保护下进行；"
                "不得要求老年人模拟跌倒。系统输出不能替代照护人员判断、医疗诊断或紧急救援。</div>"
            )
    return app


def main() -> None:
    app = build_app()
    app.queue(default_concurrency_limit=1, max_size=8).launch(
        server_name=CONFIG.server.host,
        server_port=int(CONFIG.server.port),
        inbrowser=bool(CONFIG.server.open_browser),
        share=False,
        show_error=False,
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
