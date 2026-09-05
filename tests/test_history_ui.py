from __future__ import annotations

import csv

import gradio as gr
import pytest

from services.database import EVENT_COLUMNS, PREFALL_COLUMNS, EventRepository
from services.history import HISTORY_PAGE_SIZE, HistoryPage


@pytest.fixture
def history_app(tmp_path, monkeypatch):
    import app

    repository = EventRepository(tmp_path / "events.db")
    monkeypatch.setattr(app, "REPOSITORY", repository)
    for index in range(45):
        repository.add_event({
            "event_key": f"event-{index}", "occurred_at": "2026-09-01T12:00:00+08:00",
            "source": "测试", "source_name": "样例.mp4", "risk": 70 + index / 10,
            "confidence": 0.8, "reasons": ["测试原因"], "source_time_s": index,
        })
        repository.add_pre_fall_record({
            "recorded_at": "2026-09-01T12:00:00+08:00", "source_time_s": index,
            "track_id": 1, "pre_fall_risk_score": index, "risk_factors": ["测试风险"],
            "short_term_score": index + 1, "medium_term_score": index + 2,
            "long_term_score": index + 3, "depth_spatial_score": index + 4,
        }, "测试", "样例.mp4", "test-session")
    return app


def test_page_bounds_and_all_rows_are_reachable():
    rows = [[index] for index in range(47)]
    first = HistoryPage(rows, loaded=True)
    second = first.moved(1)
    last = second.moved(1)
    assert first.page == 1  # No shared mutable page cursor.
    assert first.page_count == 3
    assert first.visible_rows + second.visible_rows + last.visible_rows == rows
    assert last.moved(100).current_page == 3
    assert first.moved(-100).current_page == 1
    assert HistoryPage().visible_rows == []
    assert HistoryPage().page_count == 1
    assert "暂无记录" in HistoryPage().label
    assert HistoryPage(rows, page=999).current_page == 3


def test_empty_history_and_end_buttons(tmp_path, monkeypatch):
    import app

    monkeypatch.setattr(app, "REPOSITORY", EventRepository(tmp_path / "empty.db"))
    for output in (app.load_event_history(), app.load_pre_fall_history()):
        assert output[0] == []
        assert output[1].loaded
        assert output[3]["interactive"] is False
        assert output[4]["interactive"] is False


def test_prefall_single_query_full_curve_and_page_only_navigation(history_app, monkeypatch):
    app = history_app
    query = app.REPOSITORY.list_pre_fall_records
    calls = []

    def counted_query(limit=500):
        calls.append(limit)
        return query(limit)

    monkeypatch.setattr(app.REPOSITORY, "list_pre_fall_records", counted_query)
    loaded = app.load_pre_fall_history()
    assert calls == [500]
    assert len(loaded[0]) == HISTORY_PAGE_SIZE
    assert len(loaded[1].rows) == 45
    figure = loaded[5]
    assert len(figure.data) == 5
    for index, trace in enumerate(figure.data):
        assert list(trace.y) == [value + index for value in range(45)]
        assert len(trace.x) == 45
    page_two = app.turn_pre_fall_history(loaded[1], 1)
    assert len(page_two) == 5  # No figure or summary emitted when paging.
    assert calls == [500]
    assert page_two[0] == loaded[1].rows[20:40]
    assert len(page_two[0][0]) == len(PREFALL_COLUMNS)


def test_session_snapshot_does_not_shift_when_detection_adds_records(history_app):
    app = history_app
    session_a = app.load_event_history()[1]
    session_b = app.load_event_history()[1]
    new_id = app.REPOSITORY.add_event({"event_key": "newest", "occurred_at": "2099-01-01"})
    page_two = app.turn_event_history(session_a, 1)
    assert page_two[0] == session_a.rows[20:40]
    assert page_two[1].page == 2
    assert session_b.page == 1
    assert all(row[0] != new_id for row in page_two[0])
    assert app.load_event_history()[0][0][0] == new_id
    assert page_two[5:] == (None, "请从表格中选择一条事件。", None)


@pytest.mark.parametrize("status", ["已确认", "误报", "已处理"])
def test_select_and_mark_event_on_second_page(history_app, status):
    app = history_app
    history = app.load_event_history()[1].moved(1)
    row = history.visible_rows[3]
    # Gradio emits row_value even when a non-ID cell or a locally sorted row is selected.
    evt = gr.SelectData(None, {"index": [3, 4], "value": row[4], "row_value": row})
    selected, details, snapshot = app.select_event(evt)
    assert selected == row[0]
    assert f"事件 #{selected}" in details
    output = app.update_event_history(selected, status, history)
    assert app.REPOSITORY.get_event(selected)["status"] == status
    assert output[2].current_page == 2
    assert len(output[1]) == HISTORY_PAGE_SIZE
    assert output[6:9] == (None, "请从表格中选择一条事件。", None)
    assert sum(item["status"] == status for item in app.REPOSITORY.list_events()) == 1


def test_no_selection_does_not_change_database(history_app):
    app = history_app
    before = app.REPOSITORY.list_events()
    output = app.update_event_history(None, "已确认", app.load_event_history()[1])
    assert "请先" in output[0]
    assert app.REPOSITORY.list_events() == before


def test_stale_selection_after_page_change_cannot_mark_another_page(history_app):
    app = history_app
    first_page = app.load_event_history()[1]
    previous_id = first_page.visible_rows[0][0]
    output = app.update_event_history(previous_id, "已处理", first_page.moved(1))
    assert "重新选择" in output[0]
    assert app.REPOSITORY.get_event(previous_id)["status"] == "待处理"
    assert output[6] is None


def test_maximum_history_window_remains_reachable_without_large_ui_payload():
    rows = [[index] + ["内容"] * 23 for index in range(500)]
    page = HistoryPage(rows, loaded=True)
    seen = []
    for _ in range(page.page_count):
        assert len(page.visible_rows) <= HISTORY_PAGE_SIZE
        seen.extend(page.visible_rows)
        page = page.moved(1)
    assert seen == rows
    assert page.current_page == 25


def test_full_table_mode_preserves_original_rows_and_can_return_to_paging(history_app):
    app = history_app
    for load, resize in (
        (app.load_event_history, app.resize_event_history),
        (app.load_pre_fall_history, app.resize_pre_fall_history),
    ):
        state = load()[1]
        full = resize(0, state)
        assert full[0] == state.rows
        assert full[1].page_count == 1
        assert full[3]["interactive"] is False and full[4]["interactive"] is False
        assert resize(20, full[1])[0] == state.rows[:20]
        assert load(0)[1].page_size == 0


def test_status_update_keeps_user_selected_page_size(history_app):
    app = history_app
    full = app.load_event_history(0)[1]
    output = app.update_event_history(full.rows[-1][0], "已处理", full)
    assert output[2].page_size == 0
    assert len(output[1]) == 45


@pytest.mark.parametrize("value", [None, -1, 999])
def test_cleared_or_invalid_page_size_is_safe(value):
    page = HistoryPage(rows=[[i] for i in range(50)], page_size=value)
    assert len(page.visible_rows) == HISTORY_PAGE_SIZE


def test_reentering_loaded_tabs_skips_query_and_render(history_app, monkeypatch):
    app = history_app
    event_state = app.load_event_history()[1]
    risk_state = app.load_pre_fall_history()[1]

    def forbidden(*args, **kwargs):
        raise AssertionError("Re-entering a loaded tab must not refetch or redraw")

    monkeypatch.setattr(app, "refresh_events", forbidden)
    monkeypatch.setattr(app, "refresh_pre_fall_records", forbidden)
    assert app.open_event_history(event_state) == tuple(gr.skip() for _ in range(9))
    assert app.open_pre_fall_history(risk_state) == tuple(gr.skip() for _ in range(7))


def test_csv_exports_are_not_limited_to_current_page(history_app, tmp_path):
    repository = history_app.REPOSITORY
    for export, rows, headers, name in (
        (repository.export_csv, repository.table_rows(limit=5000), EVENT_COLUMNS, "events.csv"),
        (repository.export_pre_fall_csv, repository.pre_fall_table_rows(limit=5000), PREFALL_COLUMNS, "risk.csv"),
    ):
        path = export(tmp_path / name)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            actual = list(csv.reader(handle))
        assert actual[0] == headers
        assert actual[1:] == [[str(value) for value in row] for row in rows]
        assert len(actual) == 46


def test_compound_indexes_keep_order_without_temporary_sort(history_app):
    repository = history_app.REPOSITORY
    with repository._connect() as connection:
        for table, time_column in (("events", "occurred_at"), ("pre_fall_risk_records", "recorded_at")):
            plan = connection.execute(
                f"EXPLAIN QUERY PLAN SELECT * FROM {table} ORDER BY {time_column} DESC, id DESC LIMIT 500"
            ).fetchall()
            assert all("TEMP B-TREE" not in row[3] for row in plan)
    assert [row["id"] for row in repository.list_events()] == list(range(45, 0, -1))


def test_ui_starts_empty_and_history_does_not_share_inference_queue(monkeypatch):
    import app

    def forbidden(*args, **kwargs):
        raise AssertionError("Building UI must not query hidden history tables")

    monkeypatch.setattr(app.REPOSITORY, "table_rows", forbidden)
    monkeypatch.setattr(app.REPOSITORY, "pre_fall_table_rows", forbidden)
    monkeypatch.setattr(app.REPOSITORY, "list_pre_fall_records", forbidden)
    ui = app.build_app()
    config = ui.get_config_file()
    tables = [c for c in config["components"] if c.get("props", {}).get("elem_id") in ("events-table", "pre-fall-table")]
    assert len(tables) == 2
    assert all(not c["props"]["value"]["data"] for c in tables)
    assert not any(target[1] == "load" for dep in config["dependencies"] for target in dep["targets"])
    history_names = {"open_event_history", "open_pre_fall_history", "load_event_history", "load_pre_fall_history", "select_event"}
    callbacks = [fn for fn in ui.fns.values() if fn.fn and fn.fn.__name__ in history_names]
    assert len(callbacks) == 5
    assert all(fn.queue is False for fn in callbacks)
    inference = [fn for fn in ui.fns.values() if fn.concurrency_id == "pose-inference"]
    assert len(inference) == 2
    assert all(fn.concurrency_limit == 1 and fn.queue for fn in inference)
