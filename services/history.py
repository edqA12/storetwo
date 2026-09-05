from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


# Presentation only: keep detection policy and the existing history/export limits unchanged.
HISTORY_PAGE_SIZE = 20


@dataclass(slots=True)
class HistoryPage:
    """A browser-session snapshot; new detections cannot shift rows during paging."""

    rows: list[list[Any]] = field(default_factory=list)
    page: int = 1
    loaded: bool = False
    page_size: int = HISTORY_PAGE_SIZE

    def __post_init__(self) -> None:
        if self.page_size not in (0, 20, 50):
            self.page_size = HISTORY_PAGE_SIZE

    @property
    def effective_page_size(self) -> int:
        # Zero restores the original full-table copy/sort/fullscreen behavior on demand.
        return max(1, len(self.rows)) if self.page_size == 0 else max(1, int(self.page_size))

    @property
    def page_count(self) -> int:
        size = self.effective_page_size
        return max(1, (len(self.rows) + size - 1) // size)

    @property
    def current_page(self) -> int:
        return max(1, min(int(self.page), self.page_count))

    @property
    def visible_rows(self) -> list[list[Any]]:
        start = (self.current_page - 1) * self.effective_page_size
        return self.rows[start:start + self.effective_page_size]

    @property
    def label(self) -> str:
        if not self.rows:
            return "暂无记录"
        if self.page_size == 0:
            return f"全部显示 · 共 {len(self.rows)} 条"
        start = (self.current_page - 1) * self.effective_page_size + 1
        end = min(start + self.effective_page_size - 1, len(self.rows))
        return (
            f"第 {self.current_page} / {self.page_count} 页 · "
            f"第 {start}–{end} 条 / 共 {len(self.rows)} 条"
        )

    def moved(self, delta: int) -> HistoryPage:
        return replace(self, page=max(1, min(self.current_page + delta, self.page_count)))

    def resized(self, page_size: int) -> HistoryPage:
        return replace(self, page=1, page_size=page_size)
