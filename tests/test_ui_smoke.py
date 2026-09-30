from __future__ import annotations


def test_build_app_returns_blocks():
    import gradio as gr

    from app import build_app

    app = build_app()
    assert isinstance(app, gr.Blocks)


def test_camera_frame_handler_uses_session_state(monkeypatch):
    import app as app_module
    from core.pipeline import new_stream_state
    from core.types import FrameResult

    def fake_process(frame_bgr, timestamp_s, stream_state, confidence):
        stream_state["frame_count"] = int(stream_state.get("frame_count", 0)) + 1
        return FrameResult(
            annotated_bgr=frame_bgr,
            decisions=[],
            max_risk=0.0,
            overall_label="未检测到完整人体",
            new_events=[],
            diagnostics={"persons": 0, "total_ms": 5.0, "processing_fps": 10.0},
        ), stream_state

    monkeypatch.setattr(app_module.PIPELINE, "process_frame", fake_process)
    state = new_stream_state()
    frame = __import__("numpy").zeros((48, 64, 3), dtype="uint8")
    output = app_module.process_camera_frame(frame, state, 0.35)
    assert output[0].shape == frame.shape
    assert output[2] is None
    assert output[8]["frame_count"] == 1
