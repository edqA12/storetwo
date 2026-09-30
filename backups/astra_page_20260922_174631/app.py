from __future__ import annotations

import time
from base64 import b64encode
import uuid
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np
import plotly.graph_objects as go

from core.config import load_config
from core.depth_pipeline import DepthPipeline
from core.detector import ModelUnavailableError, YoloPoseDetector
from core.fusion import FusionEngine
from core.multimodal import MultimodalPipeline
from core.pipeline import VisionPipeline, new_stream_state
from core.pre_fall import LEVEL_LABELS
from core.video_processor import VideoProcessor
from services.alarm import ensure_alarm_wav
from services.database import EVENT_COLUMNS, PREFALL_COLUMNS, EventRepository
from services.events import EventService
from services.history import HistoryPage
from services import ai_assistant as ai
from services import knowledge_ui
from services.astra_live import AstraLiveMonitor
import json
from datetime import datetime


CONFIG = load_config()
REPOSITORY = EventRepository(CONFIG.resolve(CONFIG.storage.database))
EVENT_SERVICE = EventService(REPOSITORY, CONFIG.resolve(CONFIG.storage.snapshots_dir))
DETECTOR = YoloPoseDetector(CONFIG)
PIPELINE = VisionPipeline(CONFIG, DETECTOR, REPOSITORY)
DEPTH_PIPELINE = DepthPipeline(CONFIG.multimodal)
FUSION_ENGINE = FusionEngine(CONFIG.multimodal)
MULTIMODAL_PIPELINE = MultimodalPipeline(CONFIG, PIPELINE, DEPTH_PIPELINE, FUSION_ENGINE)
ASTRA_MONITOR = AstraLiveMonitor(CONFIG, MULTIMODAL_PIPELINE)
VIDEO_PROCESSOR = VideoProcessor(CONFIG, PIPELINE, EVENT_SERVICE, MULTIMODAL_PIPELINE)
ALARM_FILE = ensure_alarm_wav(CONFIG.resolve("data/alarm.wav"))

UPLOAD_EVENT_HEADERS = ["事件ID", "视频时间", "事件类型", "跌倒事件证据分", "触发原因"]


def _empty_risk_figure(message: str = "等待检测", y_title: str = "跌倒事件证据分") -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        template="plotly_white",
        height=330,
        margin=dict(l=45, r=25, t=45, b=40),
        title=message,
        xaxis_title="视频时间（秒）",
        yaxis_title=y_title,
        yaxis=dict(range=[0, 100]),
    )
    return figure


def _risk_figure(samples: list[tuple[float, float]]) -> go.Figure:
    if not samples:
        return _empty_risk_figure("未获得事件证据数据")
    x_values = [item[0] for item in samples]
    y_values = [item[1] for item in samples]
    figure = go.Figure()
    figure.add_hrect(y0=80, y1=100, fillcolor="#fee2e2", opacity=0.6, line_width=0)
    figure.add_hrect(y0=60, y1=80, fillcolor="#fef3c7", opacity=0.55, line_width=0)
    figure.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode="lines",
        name="跌倒事件证据分",
        line=dict(color="#0066cc", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(0,102,204,0.10)",
    ))
    figure.update_layout(
        template="plotly_white",
        height=330,
        margin=dict(l=45, r=25, t=45, b=40),
        title="跌倒事件证据分变化曲线",
        xaxis_title="视频时间（秒）",
        yaxis_title="跌倒事件证据分",
        yaxis=dict(range=[0, 100]),
        hovermode="x unified",
        showlegend=False,
    )
    return figure


def _pre_fall_figure(samples: list[tuple[Any, float]], title: str = "前置风险变化趋势") -> go.Figure:
    if not samples:
        return _empty_risk_figure("暂无前置风险记录", "Pre-Fall Risk Score")
    figure = go.Figure()
    figure.add_hrect(y0=61, y1=100, fillcolor="#fee2e2", opacity=0.55, line_width=0)
    figure.add_hrect(y0=31, y1=61, fillcolor="#fef3c7", opacity=0.50, line_width=0)
    figure.add_trace(go.Scatter(
        x=[item[0] for item in samples],
        y=[item[1] for item in samples],
        mode="lines+markers",
        name="前置风险分",
        line=dict(color="#7c3aed", width=2.5),
    ))
    figure.update_layout(
        template="plotly_white",
        height=330,
        margin=dict(l=45, r=25, t=45, b=40),
        title=title,
        xaxis_title="时间",
        yaxis_title="Pre-Fall Risk Score",
        yaxis=dict(range=[0, 100]),
        hovermode="x unified",
        showlegend=False,
    )
    return figure


def _pre_fall_history_figure(records: list[dict[str, Any]]) -> go.Figure:
    if not records:
        return _empty_risk_figure("暂无前置风险记录", "工程评分")
    ordered = list(reversed(records))
    figure = go.Figure()
    series = (
        ("pre_fall_risk_score", "前置风险分", "#7c3aed"),
        ("short_term_score", "短时窗口", "#2563eb"),
        ("medium_term_score", "24小时窗口", "#ea580c"),
        ("long_term_score", "30天窗口", "#059669"),
        ("depth_spatial_score", "深度空间证据", "#dc2626"),
    )
    for key, label, color in series:
        figure.add_trace(go.Scatter(
            x=[item["recorded_at"] for item in ordered],
            y=[float(item.get(key, 0.0)) for item in ordered],
            mode="lines+markers",
            name=label,
            line=dict(color=color, width=2.2),
        ))
    figure.update_layout(
        template="plotly_white",
        height=380,
        margin=dict(l=45, r=25, t=45, b=40),
        title="前置风险与短时、24小时、30天、深度空间证据趋势",
        xaxis_title="时间",
        yaxis_title="工程异常评分",
        yaxis=dict(range=[0, 100]),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.12),
    )
    return figure


def _pre_fall_summary(level: str, score: float, factors: list[str]) -> str:
    label = LEVEL_LABELS.get(level, level)
    factor_lines = "\n".join(f"- {item}" for item in factors) or "- 暂无明显异常"
    return (
        f"### 前置跌倒风险：{label}（{score:.1f}/100）\n"
        f"{factor_lines}\n\n"
        "> 该分数是连续行为工程风险评分，不等同于临床医学跌倒概率。"
    )


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
    input_mode: str = "rgb",
    progress: gr.Progress = gr.Progress(),
):
    path = _normalise_video_input(video_value)
    if not path:
        return (
            None, "❌ 请先选择一个视频文件。", 0.0, 0,
            _empty_risk_figure(), [], None, 0.0,
            _pre_fall_summary("LOW", 0.0, []),
            _empty_risk_figure("暂无前置风险记录", "Pre-Fall Risk Score"), 0,
        )
    try:
        progress(0.01, desc="正在读取视频……")
        result = VIDEO_PROCESSOR.process(
            path,
            confidence=float(confidence),
            frame_stride=int(frame_stride),
            input_mode=str(input_mode),
            progress=lambda fraction, text: progress(fraction, desc=text),
        )
        status = f"✅ {result.message}"
        return (
            str(result.output_path), status, result.max_risk, result.event_count,
            _risk_figure(result.risk_samples), result.event_rows, str(result.output_path),
            result.max_pre_fall_risk,
            _pre_fall_summary(
                result.pre_fall_risk_level,
                result.max_pre_fall_risk,
                result.pre_fall_risk_factors,
            ),
            _pre_fall_figure(result.pre_fall_risk_samples, "视频前置风险变化曲线"),
            result.near_fall_count,
        )
    except (ValueError, RuntimeError, ModelUnavailableError) as exc:
        return (
            None, f"❌ {exc}", 0.0, 0, _empty_risk_figure("检测未完成"), [], None,
            0.0, _pre_fall_summary("LOW", 0.0, []),
            _empty_risk_figure("检测未完成", "Pre-Fall Risk Score"), 0,
        )
    except Exception as exc:  # pragma: no cover - final UI guard
        return (
            None, f"❌ 检测过程中发生异常：{exc}", 0.0, 0,
            _empty_risk_figure("检测未完成"), [], None, 0.0,
            _pre_fall_summary("LOW", 0.0, []),
            _empty_risk_figure("检测未完成", "Pre-Fall Risk Score"), 0,
        )


def process_camera_frame(frame_rgb: np.ndarray | None, state: dict[str, Any] | None, confidence: float, source="浏览器RGB"):
    if source != "浏览器RGB":
        return tuple(gr.skip() for _ in range(9))
    session = state or new_stream_state()
    if frame_rgb is None:
        return (
            None, "### 等待摄像头画面", 0.0, 0.0,
            _pre_fall_summary("LOW", 0.0, []), 0.0,
            session.get("last_alert_text", "暂无告警"), None, session,
        )
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
        return _present_camera_result(result, session)
    except (ValueError, ModelUnavailableError) as exc:
        return (
            frame_rgb, f"### ⚠️ {exc}", 0.0, 0.0,
            _pre_fall_summary("LOW", 0.0, []), 0.0,
            session.get("last_alert_text", "暂无告警"), None, session,
        )
    except Exception as exc:  # pragma: no cover - final UI guard
        return (
            frame_rgb, f"### ⚠️ 摄像头检测异常：{exc}", 0.0, 0.0,
            _pre_fall_summary("LOW", 0.0, []), 0.0,
            session.get("last_alert_text", "暂无告警"), None, session,
        )



def _present_camera_result(result, session, source="实时摄像头", source_name="本机摄像头", prefix=""):
    highest_pre_fall = max(
        result.pre_fall_results,
        key=lambda item: item.pre_fall_risk_score,
        default=None,
    )
    record_times = session.setdefault("pre_fall_last_record_ts", {})
    for pre_fall in result.pre_fall_results:
        previous_record_ts = float(record_times.get(str(pre_fall.track_id), -1e12))
        should_record = (
            pre_fall.level_changed
            or pre_fall.timestamp_s - previous_record_ts >= CONFIG.pre_fall.record_interval_s
            or bool(result.near_fall_events)
            or bool(result.sit_to_stand_events)
        )
        if should_record:
            record = pre_fall.to_record()
            record["source_time_s"] = pre_fall.timestamp_s
            REPOSITORY.add_pre_fall_record(
                record,
                source,
                source_name,
                str(session.get("session_id", "camera-session")),
            )
            record_times[str(pre_fall.track_id)] = float(pre_fall.timestamp_s)
    for near_fall in result.near_fall_events:
        near_fall_id = REPOSITORY.add_near_fall_event(
            near_fall,
            source,
            source_name,
            str(session.get("session_id", "camera-session")),
        )
        session["last_alert_text"] = (
            f"⚠️ **近跌倒 #{near_fall_id}**　人员在失衡后恢复站立，"
            "未记录为确认跌倒；前置风险已更新。"
        )
    for sit_to_stand in result.sit_to_stand_events:
        REPOSITORY.add_sit_to_stand_event(
            sit_to_stand,
            source,
            source_name,
            str(session.get("session_id", "camera-session")),
        )
        session["last_alert_text"] = (
            f"🪑 **坐站评估：{sit_to_stand['status']}**　"
            f"耗时 {float(sit_to_stand['duration_s']):.1f} 秒，"
            f"异常分 {float(sit_to_stand['sit_to_stand_score']):.0f}/100。"
        )
    if result.sit_to_stand_attempts and not result.sit_to_stand_events:
        session["last_alert_text"] = "🪑 **检测到起身动作**　正在评估是否稳定站立。"
    alarm_output: str | None = None
    for event in result.new_events:
        event_type = str(event.get("event_type", "跌倒告警"))
        try:
            event_id = EVENT_SERVICE.record(event, result.annotated_bgr, source, source_name)
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
        f"### <span style='color:{color}'>{prefix}{result.overall_label}</span>  "
        f"\n检测到 {result.diagnostics.get('persons', 0)} 人 · "
        f"跌倒事件证据分 {result.max_risk:.1f}/100 · "
        f"单帧耗时 {result.diagnostics.get('total_ms', 0):.0f} ms"
    )
    pre_fall_score = highest_pre_fall.pre_fall_risk_score if highest_pre_fall else 0.0
    pre_fall_summary = _pre_fall_summary(
        highest_pre_fall.pre_fall_risk_level if highest_pre_fall else "LOW",
        pre_fall_score,
        list(highest_pre_fall.risk_factors) if highest_pre_fall else [],
    )
    if highest_pre_fall is None:
        pre_fall_summary = "暂无有效前置风险结果"
    if not result.decisions:
        status = f"### {prefix}当前未获得有效人体判断，分数不可用。"
    annotated_rgb = cv2.cvtColor(result.annotated_bgr, cv2.COLOR_BGR2RGB)
    return (
        annotated_rgb,
        status,
        result.max_risk if result.decisions else None,
        pre_fall_score if highest_pre_fall is not None else None,
        pre_fall_summary,
        float(result.diagnostics.get("processing_fps", 0.0)),
        session.get("last_alert_text", "暂无告警"),
        alarm_output,
        session,
    )



def _astra_empty(message, active=False):
    prefix = 'Astra监测中 · ' if active else ''
    return (None, "### " + prefix + message, None, None, "暂无有效前置风险结果", None,
            "暂无新告警", None, new_stream_state())


def start_astra(request: gr.Request):
    try:
        if not DETECTOR.model_path.exists():
            raise RuntimeError("本地姿态模型不存在，请先检查模型文件。")
        DETECTOR.ensure_loaded()
        ASTRA_MONITOR.start(request.session_hash)
        return (*_astra_empty("Astra已连接，正在准备监测", active=True), gr.Timer(active=True))
    except Exception as exc:
        return (*_astra_empty(f"无法开始Astra监测：{exc}"), gr.Timer(active=False))


def stop_astra(request: gr.Request):
    ASTRA_MONITOR.stop(request.session_hash)
    return (*_astra_empty("Astra监测已停止"), gr.Timer(active=False))


def tick_astra(source, confidence, request: gr.Request):
    if source != "Astra Pro RGB＋深度" or ASTRA_MONITOR.owner != request.session_hash:
        return tuple(gr.skip() for _ in range(9))
    update = ASTRA_MONITOR.tick(request.session_hash, float(confidence))
    if update.result is None:
        return _astra_empty(update.status, active=ASTRA_MONITOR.owner == request.session_hash)
    if not update.new_result:
        return tuple(gr.skip() for _ in range(9))
    outputs = list(_present_camera_result(update.result, ASTRA_MONITOR.state,
                                         "实时深度相机", "Astra Pro", "Astra监测中 · " + update.status + " · "))
    return tuple(outputs)


def switch_camera_source(source, request: gr.Request):
    ASTRA_MONITOR.stop(request.session_hash)
    astra = source == "Astra Pro RGB＋深度"
    return (gr.update(visible=not astra, value=None), gr.update(visible=astra),
            *_astra_empty("请选择开始监测" if astra else "等待浏览器摄像头画面"), gr.Timer(active=False))


def reset_selected_camera(source, request: gr.Request):
    if source == "Astra Pro RGB＋深度":
        return stop_astra(request)
    return (*reset_camera_state(), gr.Timer(active=False))


def clear_browser_camera(source):
    return reset_camera_state() if source == "浏览器RGB" else tuple(gr.skip() for _ in range(9))


def release_astra_page(request: gr.Request):
    ASTRA_MONITOR.stop(request.session_hash)


def reset_camera_state():
    return (
        None, "### 监测状态已重置", 0.0, 0.0,
        _pre_fall_summary("LOW", 0.0, []), 0.0,
        "暂无告警", None, new_stream_state(),
    )


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


def refresh_pre_fall_records():
    records = REPOSITORY.list_pre_fall_records(limit=500)
    if records:
        latest = records[0]
        label = LEVEL_LABELS.get(str(latest["pre_fall_risk_level"]), latest["pre_fall_risk_level"])
        summary = (
            f"**最新前置风险：** {label} · {float(latest['pre_fall_risk_score']):.1f}/100　"
            f"**历史记录：** {len(records)} 条"
        )
    else:
        summary = "暂无前置风险历史记录。"
    return REPOSITORY.pre_fall_rows_from_records(records), _pre_fall_history_figure(records), summary


def _history_page_outputs(history: HistoryPage):
    return (
        history.visible_rows, history, history.label,
        gr.update(interactive=history.current_page > 1),
        gr.update(interactive=history.current_page < history.page_count),
    )


def _event_page_outputs(history: HistoryPage):
    # A selection belongs to the displayed page, never to its row position on another page.
    return (*_history_page_outputs(history), None, "请从表格中选择一条事件。", None)


def load_event_history(page_size: int = 20):
    rows, summary = refresh_events()
    return (*_event_page_outputs(HistoryPage(rows=rows, loaded=True, page_size=page_size)), summary)


def open_event_history(history: HistoryPage | None):
    # Re-entering a tab reuses its session snapshot; Refresh explicitly fetches new records.
    if history is not None and history.loaded:
        return tuple(gr.skip() for _ in range(9))
    return load_event_history(history.page_size if history is not None else 20)


def turn_event_history(history: HistoryPage | None, delta: int):
    return _event_page_outputs((history or HistoryPage()).moved(delta))


def resize_event_history(page_size: int, history: HistoryPage | None):
    return _event_page_outputs((history or HistoryPage()).resized(page_size))


def load_pre_fall_history(page_size: int = 20):
    rows, figure, summary = refresh_pre_fall_records()
    return (*_history_page_outputs(HistoryPage(rows=rows, loaded=True, page_size=page_size)), figure, summary)


def open_pre_fall_history(history: HistoryPage | None):
    if history is not None and history.loaded:
        return tuple(gr.skip() for _ in range(7))
    return load_pre_fall_history(history.page_size if history is not None else 20)


def turn_pre_fall_history(history: HistoryPage | None, delta: int):
    # Deliberately no plot output: paging must not redraw or truncate the history curve.
    return _history_page_outputs((history or HistoryPage()).moved(delta))


def resize_pre_fall_history(page_size: int, history: HistoryPage | None):
    return _history_page_outputs((history or HistoryPage()).resized(page_size))


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
    modalities = event.get("modalities", [])
    modality_lines = []
    for modality in modalities if isinstance(modalities, list) else []:
        modality_lines.append(
            f"  - {modality.get('modality', '未知')}：风险 {float(modality.get('risk', 0.0)):.1f}，"
            f"质量 {float(modality.get('quality', 0.0)):.2f}，状态 {modality.get('label') or modality.get('state', '未知')}"
        )
    details = (
        f"### 事件 #{event_id}\n"
        f"- 时间：{event['occurred_at']}\n"
        f"- 来源：{event['source']}\n"
        f"- 跌倒事件证据分：{float(event['risk']):.1f}\n"
        f"- 状态：{event['status']}\n"
        f"- 原因：{event['reasons'] or '未记录'}\n"
        f"- 融合方式：{event.get('fusion_method') or '单模态视觉'}\n"
        f"- 模态明细：\n" + ("\n".join(modality_lines) if modality_lines else "  - 未记录")
    )
    snapshot = event.get("snapshot_path")
    return event_id, details, snapshot if snapshot and Path(snapshot).exists() else None


def update_selected_event(event_id: int | float | None, status: str):
    if event_id is None:
        message = "请先在表格中选择一条事件。"
    else:
        try:
            selected = int(event_id)
        except (TypeError, ValueError):
            message = "事件编号无效。"
        else:
            updated = REPOSITORY.update_status(selected, status)
            message = f"事件 #{selected} 已标记为“{status}”。" if updated else f"未找到事件 #{selected}。"
    rows, summary = refresh_events()
    return message, rows, None, "请从表格中选择一条事件。", None, summary


def update_event_history(event_id, status: str, history: HistoryPage | None):
    if event_id is not None and history is not None:
        try:
            is_current = int(event_id) in {int(row[0]) for row in history.visible_rows}
        except (TypeError, ValueError):
            is_current = False
        if not is_current:
            return (
                "所选事件已不在当前页，请重新选择。",
                *_event_page_outputs(history), gr.skip(),
            )
    message, rows, _, _, _, summary = update_selected_event(event_id, status)
    page = history.current_page if history is not None else 1
    page_size = history.page_size if history is not None else 20
    refreshed = HistoryPage(rows=rows, page=page, loaded=True, page_size=page_size)
    return (message, *_event_page_outputs(refreshed), summary)


def export_events():
    export_dir = CONFIG.resolve(CONFIG.storage.exports_dir)
    path = export_dir / f"events_{uuid.uuid4().hex[:8]}.csv"
    REPOSITORY.export_csv(path)
    return str(path), f"已导出 {len(REPOSITORY.list_events(limit=5000))} 条记录。"


def export_pre_fall_records():
    export_dir = CONFIG.resolve(CONFIG.storage.exports_dir)
    path = export_dir / f"pre_fall_risk_{uuid.uuid4().hex[:8]}.csv"
    REPOSITORY.export_pre_fall_csv(path)
    return str(path), f"已导出 {len(REPOSITORY.list_pre_fall_records(limit=5000))} 条前置风险记录。"


ASSETS_DIR = Path(__file__).resolve().parent / "assets"
CSS = "\n".join(
    (ASSETS_DIR / filename).read_text(encoding="utf-8")
    for filename in ("welcome.css", "interface.css", "assistant.css")
)
NAVIGATION_JS = (ASSETS_DIR / "navigation.js").read_text(encoding="utf-8")


def _welcome_html() -> str:
    image = b64encode((ASSETS_DIR / "muan-dusk-brand.jpg").read_bytes()).decode("ascii")
    return (ASSETS_DIR / "welcome.html").read_text(encoding="utf-8").replace(
        "__MUAN_BRAND_IMAGE__", f"data:image/jpeg;base64,{image}"
    )



def prepare_ai_current(source, upload_status, event_score, pre_score, count, near_count,
                       camera_fps, camera_event_score, camera_pre_score):
    is_video = source == "视频检测"
    ready = str(upload_status).startswith("✅") if is_video else (ai.number(camera_fps) or 0) > 0
    if not ready:
        return "尚无可用结果，请先完成所选来源的检测。", "", False
    data = {"摘要类型": source + "结果快照", "摘要生成时间": datetime.now().isoformat(timespec="seconds"),
            "跌倒事件证据分" + ("（本次最高）" if is_video else "（当前）"): ai.number(event_score if is_video else camera_event_score),
            "前置风险分" + ("（本次最高）" if is_video else "（当前）"): ai.number(pre_score if is_video else camera_pre_score),
            "说明": "两种分数并非医学概率；这是点击准备时的结果，不会自动更新。"}
    if is_video:
        data.update({"告警事件数量": ai.number(count), "近跌倒次数": ai.number(near_count)})
    value = json.dumps(data, ensure_ascii=False, indent=2)
    return value, value, False


def select_ai_risk(evt: gr.SelectData):
    row = getattr(evt, "row_value", None)
    try:
        return int(row[0]) if row else None
    except (TypeError, ValueError, IndexError):
        return None


def prepare_ai_record(source, event_id, risk_id, events, risks):
    history = events if source == "事件中心" else risks
    selected = event_id if source == "事件中心" else risk_id
    headers = EVENT_COLUMNS if source == "事件中心" else PREFALL_COLUMNS
    rows = history.visible_rows if history is not None and history.loaded else []
    row = next((row for row in rows if selected is not None and str(row[0]) == str(selected)), None)
    if row is None:
        return "请先在所选记录页点击一行；翻页或刷新后请重新选择。", "", False
    data = {"摘要类型": source + "选中单条记录（非全部历史）", **ai.record_summary(row, headers)}
    value = json.dumps(data, ensure_ascii=False, indent=2)
    return value, value, False


def build_app() -> gr.Blocks:
    with gr.Blocks(title="暮安智护·跌倒全周期风险监测") as app:
        with gr.Column(elem_id="muan-interface"):
            gr.HTML(
                _welcome_html(), elem_id="welcome-block", apply_default_css=False,
                js_on_load=NAVIGATION_JS,
            )

            with gr.Column(elem_id="app-shell"):
                gr.HTML(
                    (ASSETS_DIR / "workspace.html").read_text(encoding="utf-8"),
                    elem_id="workspace-chrome", apply_default_css=False, js_on_load=None,
                )
                with gr.Tabs(elem_id="workspace-tabs", selected="video"):
                    with gr.Tab("视频检测", id="video", elem_id="video-tab"):
                        gr.HTML(
                            "<div class='section-intro'><h2>视频检测</h2>"
                            "<p>支持普通RGB视频，以及左侧深度图、右侧RGB的组合视频。</p></div>"
                        )
                        gr.HTML(
                            "<div class='warning-note'><strong>运行提示</strong>　"
                            "上传分析前请停止实时摄像头，避免 CPU 推理任务互相等待。"
                            "双模态模式要求左右画面来自同一时刻。</div>"
                        )
                        with gr.Row(equal_height=False, elem_id="video-workspace"): 
                            with gr.Column(scale=3, elem_classes="muan-card", elem_id="video-controls"):
                                upload_video = gr.Video(
                                    label="上传待检测视频",
                                    sources=["upload"],
                                    elem_id="upload-video",
                                )
                                input_mode = gr.Radio(
                                    choices=[
                                        ("普通RGB视频", "rgb"),
                                        ("RGB+深度组合视频（左深度、右RGB）", "rgb_depth"),
                                    ],
                                    value="rgb",
                                    label="输入模式",
                                )
                                with gr.Accordion("高级设置", open=False):
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
                                upload_status = gr.Markdown("请选择视频后开始检测。", elem_id="upload-status")
                            with gr.Column(scale=7, elem_classes="muan-card", elem_id="video-stage"):
                                result_video = gr.Video(
                                    label="标注结果视频", format="mp4", elem_id="result-video"
                                )
                                result_download = gr.File(label="下载检测结果", elem_id="result-download")
                                gr.HTML("<div class='video-empty-guide'><strong>等待检测结果</strong><p>上传视频，选择输入模式，然后点击开始检测。</p></div>", elem_id="video-empty-guide")
                        with gr.Row(elem_id="video-metrics"):
                            max_risk = gr.Number(
                                label="最高跌倒事件证据分", value=0, interactive=False,
                                elem_classes="metric-card",
                            )
                            max_pre_fall_risk = gr.Number(
                                label="最高前置风险分", value=0, interactive=False,
                                elem_classes="metric-card",
                            )
                            event_count = gr.Number(
                                label="告警事件数量", value=0, interactive=False,
                                elem_classes="metric-card",
                            )
                            upload_near_fall_count = gr.Number(
                                label="近跌倒次数", value=0, interactive=False,
                                elem_classes="metric-card",
                            )
                        upload_pre_fall_summary = gr.Markdown(
                            _pre_fall_summary("LOW", 0.0, []), elem_id="pre-fall-summary"
                        )
                        risk_plot = gr.Plot(value=_empty_risk_figure())
                        upload_pre_fall_plot = gr.Plot(
                            value=_empty_risk_figure("暂无前置风险记录", "Pre-Fall Risk Score")
                        )
                        upload_events = gr.Dataframe(
                            headers=UPLOAD_EVENT_HEADERS,
                            value=[],
                            interactive=False,
                            label="本次视频事件",
                        )
                        analyse_button.click(
                            fn=process_uploaded_video,
                            inputs=[upload_video, upload_confidence, frame_stride, input_mode],
                            outputs=[
                                result_video, upload_status, max_risk, event_count,
                                risk_plot, upload_events, result_download,
                                max_pre_fall_risk, upload_pre_fall_summary,
                                upload_pre_fall_plot, upload_near_fall_count,
                            ],
                            concurrency_limit=1,
                            concurrency_id="pose-inference",
                        )

                    with gr.Tab("实时监测", id="live", elem_id="live-tab"):
                        gr.HTML(
                            "<div class='section-intro'><h2>实时监测</h2>"
                            "<p>选择浏览器RGB或Astra Pro深度相机，连续监测人体活动并呈现判断依据。</p></div>"
                        )
                        camera_state = gr.State(value=new_stream_state())
                        camera_source = gr.Radio(["浏览器RGB", "Astra Pro RGB＋深度"], value="浏览器RGB", label="实时监测来源")
                        with gr.Row(visible=False) as astra_controls:
                            astra_start = gr.Button("开始Astra监测", variant="primary")
                            astra_stop = gr.Button("停止Astra监测")
                        astra_timer = gr.Timer(value=0.2, active=False)

                        gr.Markdown(
                            "<div class='warning-note'><strong>拍摄建议</strong>　"
                            "请让人物全身进入画面；浏览器RGB需授予摄像头权限，Astra使用上方开始与停止按钮。"
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
                            camera_event_score = gr.Number(
                                label="当前跌倒事件证据分", value=0, interactive=False,
                                elem_classes="metric-card",
                            )
                            camera_pre_fall_score = gr.Number(
                                label="当前前置风险分", value=0, interactive=False,
                                elem_classes="metric-card",
                            )
                            camera_fps = gr.Number(
                                label="处理帧率 FPS", value=0, interactive=False,
                                elem_classes="metric-card",
                            )
                        camera_status = gr.Markdown(
                            "### 等待摄像头画面", elem_id="status-panel"
                        )
                        camera_pre_fall_summary = gr.Markdown(
                            _pre_fall_summary("LOW", 0.0, []), elem_id="pre-fall-summary-live"
                        )
                        recent_alert = gr.Markdown("暂无告警", elem_id="alert-panel")
                        alarm_audio = gr.Audio(label="声音告警", autoplay=True, interactive=False)
                        with gr.Row():
                            reset_button = gr.Button("重置监测状态")
                            safe_button = gr.Button("我没事（标记误报）", variant="secondary")
                            confirm_button = gr.Button("确认告警", variant="primary")

                        camera_input.stream(
                            fn=process_camera_frame,
                            inputs=[camera_input, camera_state, camera_confidence, camera_source],
                            outputs=[
                                camera_output, camera_status, camera_event_score,
                                camera_pre_fall_score, camera_pre_fall_summary, camera_fps,
                                recent_alert, alarm_audio, camera_state,
                            ],
                            time_limit=60,
                            stream_every=0.2,
                            trigger_mode="always_last",
                            concurrency_limit=1,
                            concurrency_id="pose-inference",
                            show_progress="hidden",
                        )
                        live_outputs = [camera_output, camera_status, camera_event_score,
                                        camera_pre_fall_score, camera_pre_fall_summary, camera_fps,
                                        recent_alert, alarm_audio, camera_state]
                        astra_start.click(start_astra, outputs=live_outputs + [astra_timer],
                                          concurrency_id="pose-inference", concurrency_limit=1)
                        astra_stop.click(stop_astra, outputs=live_outputs + [astra_timer],
                                         concurrency_id="pose-inference", concurrency_limit=1)
                        astra_timer.tick(tick_astra, inputs=[camera_source, camera_confidence], outputs=live_outputs,
                                         concurrency_id="pose-inference", concurrency_limit=1,
                                         trigger_mode="always_last", show_progress="hidden")
                        camera_source.change(switch_camera_source, inputs=[camera_source],
                                             outputs=[camera_input, astra_controls] + live_outputs + [astra_timer],
                                             concurrency_id="pose-inference", concurrency_limit=1)
                        reset_button.click(
                            fn=reset_selected_camera,
                            inputs=[camera_source],
                            concurrency_id="pose-inference", concurrency_limit=1,
                            outputs=[
                                camera_output, camera_status, camera_event_score,
                                camera_pre_fall_score, camera_pre_fall_summary, camera_fps,
                                recent_alert, alarm_audio, camera_state, astra_timer,
                            ],
                        )
                        camera_input.clear(
                            fn=clear_browser_camera,
                            inputs=[camera_source],
                            outputs=[
                                camera_output, camera_status, camera_event_score,
                                camera_pre_fall_score, camera_pre_fall_summary, camera_fps,
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

                    with gr.Tab("事件中心", id="events", elem_id="events-tab") as events_tab:
                        gr.HTML(
                            "<div class='section-intro'><h2>事件中心</h2>"
                            "<p>复核告警截图、更新处理状态，并导出用于实验分析的事件记录。</p></div>"
                        )
                        selected_event_id = gr.State(value=None)
                        event_history_state = gr.State(value=HistoryPage())
                        stats_markdown = gr.Markdown("正在读取事件记录…", elem_id="event-stats")
                        with gr.Row(elem_classes="history-toolbar"):
                            refresh_button = gr.Button("刷新记录", variant="primary")
                            export_button = gr.Button("导出CSV")
                            export_file = gr.File(label="事件记录文件", elem_id="event-export-file")
                        export_status = gr.Markdown()
                        events_table = gr.Dataframe(
                            headers=EVENT_COLUMNS,
                            value=[],
                            interactive=False,
                            label="历史告警记录（点击一行查看详情）",
                            elem_id="events-table",
                        )
                        with gr.Row(elem_classes="history-pagination"):
                            events_previous = gr.Button("上一页", scale=0, min_width=96, interactive=False)
                            events_page_info = gr.Markdown("尚未加载", elem_classes="history-page-info")
                            events_page_size = gr.Dropdown(
                                choices=[("20 条/页", 20), ("50 条/页", 50), ("全部（原表格模式）", 0)],
                                value=20, label="事件每页条数", show_label=False, scale=0, min_width=180,
                                filterable=False,
                            )
                            events_next = gr.Button("下一页", scale=0, min_width=96, interactive=False)
                        gr.Markdown(
                            "浏览最近 200 条告警；选择“全部”可整表排序、复制和全屏查看。CSV 导出范围保持不变。",
                            elem_classes="history-hint",
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

                        event_page_components = [
                            events_table, event_history_state, events_page_info,
                            events_previous, events_next,
                            selected_event_id, selected_details, selected_snapshot,
                        ]
                        events_tab.select(
                            fn=open_event_history,
                            inputs=[event_history_state],
                            outputs=[*event_page_components, stats_markdown],
                            queue=False,
                            show_progress="hidden",
                        )
                        refresh_button.click(
                            fn=load_event_history,
                            inputs=[events_page_size],
                            outputs=[*event_page_components, stats_markdown],
                            queue=False,
                            show_progress="hidden",
                        )
                        events_page_size.change(
                            fn=resize_event_history,
                            inputs=[events_page_size, event_history_state], outputs=event_page_components,
                            queue=False, show_progress="hidden",
                        )
                        events_previous.click(
                            fn=lambda history: turn_event_history(history, -1),
                            inputs=[event_history_state], outputs=event_page_components,
                            queue=False, show_progress="hidden",
                        )
                        events_next.click(
                            fn=lambda history: turn_event_history(history, 1),
                            inputs=[event_history_state], outputs=event_page_components,
                            queue=False, show_progress="hidden",
                        )
                        events_table.select(
                            fn=select_event,
                            outputs=[selected_event_id, selected_details, selected_snapshot],
                            queue=False,
                            show_progress="hidden",
                        )
                        mark_confirmed.click(
                            fn=lambda event_id, history: update_event_history(event_id, "已确认", history),
                            inputs=[selected_event_id, event_history_state],
                            outputs=[action_status, *event_page_components, stats_markdown],
                            queue=False, show_progress="hidden",
                        )
                        mark_false.click(
                            fn=lambda event_id, history: update_event_history(event_id, "误报", history),
                            inputs=[selected_event_id, event_history_state],
                            outputs=[action_status, *event_page_components, stats_markdown],
                            queue=False, show_progress="hidden",
                        )
                        mark_handled.click(
                            fn=lambda event_id, history: update_event_history(event_id, "已处理", history),
                            inputs=[selected_event_id, event_history_state],
                            outputs=[action_status, *event_page_components, stats_markdown],
                            queue=False, show_progress="hidden",
                        )
                        export_button.click(
                            fn=export_events, outputs=[export_file, export_status]
                        )

                    with gr.Tab("前置风险记录", id="risk", elem_id="risk-tab") as pre_fall_tab:
                        gr.HTML(
                            "<div class='section-intro'><h2>前置风险记录</h2>"
                            "<p>前置风险分用于反映相对个人行为基线的变化，不等同于临床跌倒概率。</p></div>"
                        )
                        pre_fall_history_state = gr.State(value=HistoryPage())
                        pre_fall_history_summary = gr.Markdown("正在读取前置风险记录…")
                        with gr.Row(elem_classes="history-toolbar"):
                            refresh_pre_fall_button = gr.Button("刷新风险记录", variant="primary")
                            export_pre_fall_button = gr.Button("导出前置风险CSV")
                            pre_fall_export_file = gr.File(label="前置风险记录文件", elem_id="risk-export-file")
                        pre_fall_export_status = gr.Markdown()
                        pre_fall_history_plot = gr.Plot(
                            value=_empty_risk_figure("暂无前置风险记录", "Pre-Fall Risk Score")
                        )
                        pre_fall_history_table = gr.Dataframe(
                            headers=PREFALL_COLUMNS,
                            value=[],
                            interactive=False,
                            label="前置风险历史记录",
                            elem_id="pre-fall-table",
                        )
                        with gr.Row(elem_classes="history-pagination"):
                            pre_fall_previous = gr.Button("上一页", scale=0, min_width=96, interactive=False)
                            pre_fall_page_info = gr.Markdown("尚未加载", elem_classes="history-page-info")
                            pre_fall_page_size = gr.Dropdown(
                                choices=[("20 条/页", 20), ("50 条/页", 50), ("全部（原表格模式）", 0)],
                                value=20, label="风险每页条数", show_label=False, scale=0, min_width=180,
                                filterable=False,
                            )
                            pre_fall_next = gr.Button("下一页", scale=0, min_width=96, interactive=False)
                        gr.Markdown(
                            "浏览最近 500 条风险记录；选择“全部”可整表操作。趋势图始终保留完整历史窗口。",
                            elem_classes="history-hint",
                        )
                        pre_fall_page_components = [
                            pre_fall_history_table, pre_fall_history_state, pre_fall_page_info,
                            pre_fall_previous, pre_fall_next,
                        ]
                        pre_fall_tab.select(
                            fn=open_pre_fall_history,
                            inputs=[pre_fall_history_state],
                            outputs=[*pre_fall_page_components, pre_fall_history_plot, pre_fall_history_summary],
                            queue=False,
                            show_progress="hidden",
                        )
                        refresh_pre_fall_button.click(
                            fn=load_pre_fall_history,
                            inputs=[pre_fall_page_size],
                            outputs=[*pre_fall_page_components, pre_fall_history_plot, pre_fall_history_summary],
                            queue=False,
                            show_progress="hidden",
                        )
                        pre_fall_page_size.change(
                            fn=resize_pre_fall_history,
                            inputs=[pre_fall_page_size, pre_fall_history_state], outputs=pre_fall_page_components,
                            queue=False, show_progress="hidden",
                        )
                        pre_fall_previous.click(
                            fn=lambda history: turn_pre_fall_history(history, -1),
                            inputs=[pre_fall_history_state], outputs=pre_fall_page_components,
                            queue=False, show_progress="hidden",
                        )
                        pre_fall_next.click(
                            fn=lambda history: turn_pre_fall_history(history, 1),
                            inputs=[pre_fall_history_state], outputs=pre_fall_page_components,
                            queue=False, show_progress="hidden",
                        )
                        export_pre_fall_button.click(
                            fn=export_pre_fall_records,
                            outputs=[pre_fall_export_file, pre_fall_export_status],
                        )

                # AI owns a separate session and queue; inference callbacks are untouched.
                ai_history = gr.State(value=[])
                ai_context = gr.State(value="")
                ai_risk_selection = gr.State(value=None)
                pre_fall_history_table.select(fn=select_ai_risk, outputs=ai_risk_selection,
                                              queue=False, show_progress="hidden")
                for control, event_name in [(refresh_pre_fall_button, "click"),
                                            (pre_fall_previous, "click"), (pre_fall_next, "click"),
                                            (pre_fall_page_size, "change")]:
                    getattr(control, event_name)(fn=lambda: None, outputs=ai_risk_selection,
                                                 queue=False, show_progress="hidden")
                with gr.Column(elem_id="muan-ai-panel"):
                    gr.HTML('<div class="ai-heading"><div id="muan-ai-drag" tabindex="0" '
                            'aria-label="移动助手窗口：拖动或使用方向键"><strong>暮安小助手</strong>'
                            '<small>DEEPSEEK · 为每一份关心答疑</small></div><div class="ai-window-actions">'
                            '<button type="button" id="muan-ai-reset" aria-label="恢复默认大小和位置" title="恢复默认大小和位置">↺</button>'
                            '<button type="button" id="muan-ai-close" aria-label="关闭助手">×</button></div></div>', elem_id="ai-window-heading")
                    with gr.Column(elem_id="ai-window-body"):
                        ai_route_label = gr.Markdown("直接提问即可", elem_id="ai-route-label")
                        ai_chat = gr.Chatbot(value=[], height=220, show_label=False, label="与暮安小助手对话",
                                             elem_id="ai-chat", render_markdown=False, buttons=["copy"],
                                             placeholder="你好，我是暮安小助手。\n聊日常、问健康，也能陪你读懂检测结果。")
                    with gr.Column(elem_id="ai-composer"):
                        with gr.Row(elem_id="ai-shortcuts"):
                            ai_health_start = gr.Button("健康问题", elem_id="ai-health-start")
                            ai_result_start = gr.Button("解读结果", elem_id="ai-result-start")
                            ai_record_start = gr.Button("总结记录", elem_id="ai-record-start")
                            gr.HTML('<button type="button" id="ai-tools-toggle" aria-expanded="false" aria-controls="ai-tools-drawer">知识与引用</button>', elem_id="ai-tools-button")
                        ai_consent = gr.Checkbox(value=False, label="本次发送附带已核对的摘要")
                        ai_question = gr.Textbox(label="你的问题", placeholder="你好，有什么可以帮你？", lines=2, max_lines=4, elem_id="ai-question")
                        with gr.Row():
                            ai_send = gr.Button("发送", variant="primary", elem_id="ai-send")
                            ai_stop = gr.Button("停止回答", elem_id="ai-stop")
                            ai_clear = gr.Button("清空对话", elem_id="ai-clear")
                        gr.HTML("<p class='ai-privacy-brief'>视频本地处理，AI 问答按需联网。健康回答不能替代诊断。</p>")
                    with gr.Column(elem_id="ai-tools-drawer"):
                        gr.HTML('<div class="ai-drawer-heading"><strong id="ai-tools-title">知识与引用</strong><button type="button" id="ai-tools-close" aria-label="关闭操作面板">×</button></div>')
                        with gr.Column(elem_id="ai-knowledge-tools"):
                            ai_use_knowledge = gr.Checkbox(value=True, label="使用知识库（切换此开关会清空对话）", elem_id="ai-use-knowledge")
                            gr.HTML('<button type="button" id="muan-kb-open">打开知识库管理</button>')
                            with gr.Accordion("本轮参考资料 · 查看引用原文", open=False):
                                ai_references = gr.Textbox(value="尚未检索。", label="本轮提供给模型的资料，不代表每句回答均已核实", lines=7, interactive=False, elem_id="ai-references")
                        with gr.Column(elem_id="ai-summary-tools"):
                            with gr.Column(elem_id="ai-current-tools"):
                                ai_current_source = gr.Radio(["视频检测", "实时监测"], value="视频检测", label="结果来源")
                                ai_current_button = gr.Button("准备当前结果解读")
                            with gr.Column(elem_id="ai-record-tools"):
                                ai_record_source = gr.Radio(["事件中心", "前置风险记录"], value="事件中心", label="选中记录来源")
                                ai_record_button = gr.Button("准备选中记录总结")
                            ai_preview = gr.Textbox(value="", label="当前已准备的摘要（请核对来源与内容后再勾选发送）", lines=5, interactive=False, elem_id="ai-prepared-summary")
                        with gr.Accordion("数据发送与隐私说明", open=False, elem_id="ai-privacy-tools"):
                            gr.Markdown("视频检测在本地完成；发送后，问题、最近 4 轮对话及勾选的摘要将提交 DeepSeek；部分问题会先请求模型判断意图，再生成回答。"
                                        "开启知识库时还会发送必要的检索片段；文档向量在本机生成。不发送原始视频或截图。已发送的摘要可能保留在本轮对话中，清空可移除后续上下文。"
                                        "AI 解释仅供参考，不能替代医疗诊断。密钥请在本机 ai_config.yaml 中配置。", elem_id="ai-privacy")
                    gr.HTML('<div class="ai-window-footer"><span>拖动标题移动 · 拖动右下角调整大小</span>'
                            '<button type="button" id="muan-ai-resize" aria-label="调整助手大小：拖动或使用方向键" '
                            'title="拖动调整大小，也可使用方向键">↘</button></div>', elem_id="ai-window-footer")
                ai_current_button.click(fn=prepare_ai_current,
                    inputs=[ai_current_source, upload_status, max_risk, max_pre_fall_risk, event_count,
                            upload_near_fall_count, camera_fps, camera_event_score, camera_pre_fall_score],
                    outputs=[ai_preview, ai_context, ai_consent], queue=False).then(
                    fn=lambda: "请解释这份检测结果，区分两种分数，并说明结果的局限。", outputs=ai_question, queue=False)
                ai_record_button.click(fn=prepare_ai_record,
                    inputs=[ai_record_source, selected_event_id, ai_risk_selection, event_history_state, pre_fall_history_state],
                    outputs=[ai_preview, ai_context, ai_consent], queue=False).then(
                    fn=lambda: "请总结这条选中记录，说明记录事实与不能确定的部分。", outputs=ai_question, queue=False)
                ai_requests = [gr.on(triggers=[ai_send.click, ai_question.submit], fn=ai.chat_auto,
                        inputs=[ai_question, ai_history, ai_context, ai_consent, ai_use_knowledge],
                        outputs=[ai_chat, ai_history, ai_question, ai_references, ai_route_label],
                        concurrency_id="ai-chat", concurrency_limit=1, trigger_mode="once", show_progress="minimal")]
                ai_health_start.click(fn=lambda: "我想咨询一个健康问题。", outputs=ai_question, queue=False)
                ai_use_knowledge.change(fn=lambda: ([], [], "", "", False, "", "尚未检索。", "直接提问即可"),
                    outputs=[ai_chat, ai_history, ai_context, ai_preview, ai_consent, ai_question, ai_references, ai_route_label], cancels=ai_requests, queue=False)
                ai_stop.click(fn=None, cancels=ai_requests, queue=False)
                for clear_event in (ai_clear.click, ai_chat.clear):
                    clear_event(fn=lambda: ([], [], "", "", False, "", "尚未检索。", "直接提问即可"),
                        outputs=[ai_chat, ai_history, ai_context, ai_preview, ai_consent, ai_question, ai_references, ai_route_label],
                        cancels=ai_requests, queue=False)

                knowledge_ui.mount()
                gr.HTML(
                    "<div id='muan-footer'><strong>安全说明</strong><br>"
                    "本系统用于学生科研和演示。跌倒模拟必须由健康成年人在体操垫、护具和观察员保护下进行；"
                    "不得要求老年人模拟跌倒。系统输出不能替代照护人员判断、医疗诊断或紧急救援。</div>"
                )
    app.unload(release_astra_page)
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
            str(Path(__file__).resolve().parent / "data/knowledge/originals"),
            str(CONFIG.resolve(CONFIG.storage.results_dir).resolve()),
            str(CONFIG.resolve(CONFIG.storage.snapshots_dir).resolve()),
            str(CONFIG.resolve(CONFIG.storage.exports_dir).resolve()),
            str(ALARM_FILE.resolve()),
        ],
        css=CSS,
        theme=gr.themes.Soft(primary_hue="slate", font=["Microsoft YaHei", "Segoe UI", "sans-serif"], font_mono=["Consolas", "monospace"]),
    )


if __name__ == "__main__":
    main()
