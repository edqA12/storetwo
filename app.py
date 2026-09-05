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


CONFIG = load_config()
REPOSITORY = EventRepository(CONFIG.resolve(CONFIG.storage.database))
EVENT_SERVICE = EventService(REPOSITORY, CONFIG.resolve(CONFIG.storage.snapshots_dir))
DETECTOR = YoloPoseDetector(CONFIG)
PIPELINE = VisionPipeline(CONFIG, DETECTOR, REPOSITORY)
DEPTH_PIPELINE = DepthPipeline(CONFIG.multimodal)
FUSION_ENGINE = FusionEngine(CONFIG.multimodal)
MULTIMODAL_PIPELINE = MultimodalPipeline(CONFIG, PIPELINE, DEPTH_PIPELINE, FUSION_ENGINE)
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


def process_camera_frame(frame_rgb: np.ndarray | None, state: dict[str, Any] | None, confidence: float):
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
                    "实时摄像头",
                    "本机摄像头",
                    str(session.get("session_id", "camera-session")),
                )
                record_times[str(pre_fall.track_id)] = float(pre_fall.timestamp_s)
        for near_fall in result.near_fall_events:
            near_fall_id = REPOSITORY.add_near_fall_event(
                near_fall,
                "实时摄像头",
                "本机摄像头",
                str(session.get("session_id", "camera-session")),
            )
            session["last_alert_text"] = (
                f"⚠️ **近跌倒 #{near_fall_id}**　人员在失衡后恢复站立，"
                "未记录为确认跌倒；前置风险已更新。"
            )
        for sit_to_stand in result.sit_to_stand_events:
            REPOSITORY.add_sit_to_stand_event(
                sit_to_stand,
                "实时摄像头",
                "本机摄像头",
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
            f"跌倒事件证据分 {result.max_risk:.1f}/100 · "
            f"单帧耗时 {result.diagnostics.get('total_ms', 0):.0f} ms"
        )
        pre_fall_score = highest_pre_fall.pre_fall_risk_score if highest_pre_fall else 0.0
        pre_fall_summary = _pre_fall_summary(
            highest_pre_fall.pre_fall_risk_level if highest_pre_fall else "LOW",
            pre_fall_score,
            list(highest_pre_fall.risk_factors) if highest_pre_fall else [],
        )
        annotated_rgb = cv2.cvtColor(result.annotated_bgr, cv2.COLOR_BGR2RGB)
        return (
            annotated_rgb,
            status,
            result.max_risk,
            pre_fall_score,
            pre_fall_summary,
            float(result.diagnostics.get("processing_fps", 0.0)),
            session.get("last_alert_text", "暂无告警"),
            alarm_output,
            session,
        )
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
  min-height: 76px;
  box-sizing: border-box;
  padding: 12px max(24px, calc((100vw - 1440px) / 2));
  background: #102033;
  color: #f7fafc;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
  font-size: 12px;
  letter-spacing: -0.01em;
}
#muan-global-nav .brand {
  display: flex;
  align-items: center;
  gap: 12px;
  min-width: 0;
}
#muan-global-nav .brand-mark {
  display: grid;
  place-items: center;
  width: 40px;
  height: 40px;
  flex: 0 0 40px;
  border-radius: 12px;
  background: #293744;
  color: #f5cd85 !important;
}
#muan-global-nav .brand-mark svg {
  display: block;
  width: 24px;
  height: 24px;
  fill: none;
  stroke: #f5cd85;
  stroke-width: 1.8;
  stroke-linecap: round;
  stroke-linejoin: round;
}
/* Explicit foregrounds keep the wordmark legible under Gradio themes. */
#muan-global-nav .brand-name {
  display: block;
  color: #f7fafc !important;
  font-size: 24px;
  font-weight: 600;
  line-height: 1.3;
  letter-spacing: 0.08em;
  white-space: nowrap;
}
#muan-global-nav .brand-english {
  display: block;
  margin-top: 1px;
  color: #f5cd85 !important;
  font-size: 11px;
  font-weight: 400;
  line-height: 1.4;
  letter-spacing: 0.18em;
  white-space: nowrap;
}
#muan-global-nav .nav-meta {
  color: #c5d1df !important;
  font-size: 12px;
  line-height: 1.5;
  white-space: nowrap;
}

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

.gradio-container button.primary,
.gradio-container button.secondary,
.gradio-container button.stop {
  min-height: 44px !important;
  border-radius: 9999px !important;
  font-size: 15px !important;
  font-weight: 400 !important;
  transition: transform .12s ease, background-color .12s ease !important;
}
.gradio-container button.primary:active,
.gradio-container button.secondary:active,
.gradio-container button.stop:active {transform: scale(.95);}
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
#events-table, #pre-fall-table {overflow: hidden; border-radius: 18px !important; background: var(--muan-canvas);}
#events-table [role="button"], #pre-fall-table [role="button"] {
  font-family: "Segoe UI", "Microsoft YaHei UI", system-ui, sans-serif !important;
}
.history-pagination {
  align-items: center !important;
  gap: 12px !important;
  padding: 12px 16px !important;
  border: 1px solid var(--muan-hairline) !important;
  border-radius: 18px !important;
  background: var(--muan-canvas) !important;
}
.history-page-info {text-align: center; color: var(--muan-muted); font-size: 14px;}
.history-page-info p {margin: 0 !important;}
.history-hint {color: var(--muan-muted); font-size: 12px;}
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
  #muan-global-nav {padding: 12px 16px;}
  #muan-global-nav .brand-name {font-size: 23px;}
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
    with gr.Blocks(title="暮安智护·跌倒全周期风险监测") as app:
        gr.HTML(
            "<nav id='muan-global-nav' aria-label='暮安智护品牌'>"
            "<div class='brand'><span class='brand-mark' aria-hidden='true'>"
            "<svg viewBox='0 0 24 24' focusable='false' aria-hidden='true'>"
            "<path d='M20.9 13.3A9 9 0 0 1 10.7 3.1 9 9 0 1 0 20.9 13.3Z'/>"
            "<path d='m18 2 .9 2.6L21.5 5l-2.6.9L18 8.5l-.9-2.6L14.5 5l2.6-.4Z'/>"
            "</svg></span><div class='brand-copy'>"
            "<span class='brand-name'>暮安智护</span>"
            "<span class='brand-english'>MUAN CARE</span></div></div>"
            "<span class='nav-meta'>本地 AI 安全监测</span></nav>"
            "<div id='muan-sub-nav'><span class='product-name'>跌倒全周期风险监测</span>"
            "<span class='product-meta'>前置风险评估 · 跌倒事件识别 · 本地隐私处理</span></div>"
        )
        gr.HTML(
            "<section id='hero'><span class='eyebrow'>MUAN VISION · MVP</span>"
            "<h1>在跌倒发生前，看见风险变化。</h1>"
            "<p class='hero-lead'>前置风险评估、实时事件识别与告警处置，一体化完成。</p>"
            "<p class='hero-note'>初代研究原型 · 数据默认留在本机 · 不属于医疗器械或临床诊断工具</p></section>"
        )

        with gr.Column(elem_id="app-shell"):
            with gr.Tabs():
                with gr.Tab("视频检测"):
                    gr.HTML(
                        "<div class='section-intro'><h2>分析一段视频</h2>"
                        "<p>支持普通RGB视频，以及左侧深度图、右侧RGB的组合视频。</p></div>"
                    )
                    gr.HTML(
                        "<div class='warning-note'><strong>运行提示</strong>　"
                        "上传分析前请停止实时摄像头，避免 CPU 推理任务互相等待。"
                        "双模态模式要求左右画面来自同一时刻。</div>"
                    )
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=5, elem_classes="muan-card"):
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
                        inputs=[camera_input, camera_state, camera_confidence],
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
                    reset_button.click(
                        fn=reset_camera_state,
                        outputs=[
                            camera_output, camera_status, camera_event_score,
                            camera_pre_fall_score, camera_pre_fall_summary, camera_fps,
                            recent_alert, alarm_audio, camera_state,
                        ],
                    )
                    camera_input.clear(
                        fn=reset_camera_state,
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

                with gr.Tab("事件中心") as events_tab:
                    gr.HTML(
                        "<div class='section-intro'><h2>管理告警事件</h2>"
                        "<p>复核告警截图、更新处理状态，并导出用于实验分析的事件记录。</p></div>"
                    )
                    selected_event_id = gr.State(value=None)
                    event_history_state = gr.State(value=HistoryPage())
                    stats_markdown = gr.Markdown("正在读取事件记录…", elem_id="event-stats")
                    with gr.Row():
                        refresh_button = gr.Button("刷新记录", variant="primary")
                        export_button = gr.Button("导出CSV")
                        export_file = gr.File(label="事件记录文件")
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

                with gr.Tab("前置风险记录") as pre_fall_tab:
                    gr.HTML(
                        "<div class='section-intro'><h2>查看连续行为风险趋势</h2>"
                        "<p>前置风险分用于反映相对个人行为基线的变化，不等同于临床跌倒概率。</p></div>"
                    )
                    pre_fall_history_state = gr.State(value=HistoryPage())
                    pre_fall_history_summary = gr.Markdown("正在读取前置风险记录…")
                    with gr.Row():
                        refresh_pre_fall_button = gr.Button("刷新风险记录", variant="primary")
                        export_pre_fall_button = gr.Button("导出前置风险CSV")
                        pre_fall_export_file = gr.File(label="前置风险记录文件")
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
