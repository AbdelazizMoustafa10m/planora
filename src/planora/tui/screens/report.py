"""Markdown report viewer screen."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Footer, Header, MarkdownViewer

if TYPE_CHECKING:
    from pathlib import Path

    from textual.app import ComposeResult


class ReportScreen(Screen[None]):
    """Display the generated workflow report as markdown."""

    BINDINGS = [
        Binding("escape", "dismiss_screen", "Back"),
        Binding("q", "quit_app", "Quit"),
    ]

    def __init__(
        self,
        report_path: Path | None = None,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._report_path = report_path

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield MarkdownViewer(
            self._load_report(),
            show_table_of_contents=False,
            id="report-viewer",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#report-viewer", MarkdownViewer).border_title = "Plan Report"

    async def action_dismiss_screen(self) -> None:
        await self.dismiss(None)

    def action_quit_app(self) -> None:
        self.app.exit()

    def _load_report(self) -> str:
        if self._report_path is None or not self._report_path.exists():
            return "# Report unavailable\n\nNo report file was generated for this run."
        return self._report_path.read_text(encoding="utf-8")
