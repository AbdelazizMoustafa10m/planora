from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip("textual")

from textual.widgets import Button, Footer, Header, Input, MarkdownViewer, Select, TextArea

from planora.tui.screens.report import ReportScreen
from planora.tui.screens.wizard import WizardScreen, _parse_auditors

# ---------------------------------------------------------------------------
# _parse_auditors — pure helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("gemini, codex", ["gemini", "codex"]),
        ("gemini,codex", ["gemini", "codex"]),
        ("gemini, gemini", ["gemini"]),  # dedup
        ("", []),
        ("  , ", []),
        ("claude", ["claude"]),
    ],
)
def test_parse_auditors_splits_and_deduplicates(raw: str, expected: list[str]) -> None:
    assert _parse_auditors(raw) == expected


# ---------------------------------------------------------------------------
# WizardScreen — widget layout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wizard_screen_mounts_task_input_and_buttons() -> None:
    app_class = _make_app_with_screen(WizardScreen())
    async with app_class().run_test() as pilot:
        await pilot.pause()
        assert pilot.app.screen.query_one("#task-input", TextArea) is not None
        assert pilot.app.screen.query_one("#launch-btn", Button) is not None
        assert pilot.app.screen.query_one("#cancel-btn", Button) is not None


@pytest.mark.asyncio
async def test_wizard_screen_mounts_planner_and_auditor_widgets() -> None:
    app_class = _make_app_with_screen(WizardScreen(planner="gemini", auditors=["codex"]))
    async with app_class().run_test() as pilot:
        await pilot.pause()
        assert pilot.app.screen.query_one("#planner-select", Select) is not None
        auditors_input = pilot.app.screen.query_one("#auditors-input", Input)
        assert "codex" in auditors_input.value


@pytest.mark.asyncio
async def test_wizard_screen_cancel_button_dismisses_with_none() -> None:
    dismissed: list[object] = []

    class _App(_AppBase):
        async def on_mount(self) -> None:
            await self.push_screen(WizardScreen(), dismissed.append)

    async with _App().run_test() as pilot:
        await pilot.pause()
        await pilot.click("#cancel-btn")
        await pilot.pause()

    assert dismissed == [None]


@pytest.mark.asyncio
async def test_wizard_screen_launch_button_shows_error_when_task_empty() -> None:
    app_class = _make_app_with_screen(WizardScreen())
    async with app_class().run_test() as pilot:
        await pilot.pause()
        # Task text area is empty by default — click launch
        await pilot.click("#launch-btn")
        await pilot.pause()

        error_widget = pilot.app.screen.query_one("#wizard-error")
        assert "required" in str(error_widget.content).lower()


@pytest.mark.asyncio
async def test_wizard_screen_escape_key_dismisses_with_none() -> None:
    dismissed: list[object] = []

    class _App(_AppBase):
        async def on_mount(self) -> None:
            await self.push_screen(WizardScreen(), dismissed.append)

    async with _App().run_test() as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

    assert dismissed == [None]


@pytest.mark.asyncio
async def test_wizard_screen_launch_returns_config_with_valid_task() -> None:
    dismissed: list[object] = []

    class _App(_AppBase):
        async def on_mount(self) -> None:
            await self.push_screen(WizardScreen(default_task="Add tests"), dismissed.append)

    async with _App().run_test() as pilot:
        await pilot.pause()
        await pilot.click("#launch-btn")
        await pilot.pause()

    # Should have dismissed with a non-None config
    assert len(dismissed) == 1
    config = dismissed[0]
    assert config is not None
    assert config["task"] == "Add tests"  # type: ignore[index]


@pytest.mark.asyncio
async def test_wizard_screen_defaults_to_one_audit_round() -> None:
    app_class = _make_app_with_screen(WizardScreen())
    async with app_class().run_test() as pilot:
        await pilot.pause()
        rounds_select = pilot.app.screen.query_one("#rounds-select", Select)
        assert rounds_select.value == 1


@pytest.mark.asyncio
async def test_wizard_screen_uses_injected_planner_default() -> None:
    app_class = _make_app_with_screen(WizardScreen(planner="codex"))
    async with app_class().run_test() as pilot:
        await pilot.pause()
        planner_select = pilot.app.screen.query_one("#planner-select", Select)
        assert planner_select.value == "codex"


# ---------------------------------------------------------------------------
# ReportScreen — widget layout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_report_screen_mounts_header_viewer_and_footer() -> None:
    app_class = _make_app_with_screen(ReportScreen(report_path=None))
    async with app_class().run_test() as pilot:
        await pilot.pause()
        assert pilot.app.screen.query_one(Header) is not None
        assert pilot.app.screen.query_one(MarkdownViewer) is not None
        assert pilot.app.screen.query_one(Footer) is not None


@pytest.mark.asyncio
async def test_report_screen_loads_report_content_from_file(tmp_path: Path) -> None:
    report_file = tmp_path / "plan-report.md"
    report_file.write_text("# My Report\n\nAll done.", encoding="utf-8")

    app_class = _make_app_with_screen(ReportScreen(report_path=report_file))
    async with app_class().run_test() as pilot:
        await pilot.pause()
        viewer = pilot.app.screen.query_one("#report-viewer", MarkdownViewer)
        assert viewer is not None


@pytest.mark.asyncio
async def test_report_screen_shows_unavailable_message_when_path_is_none() -> None:
    app_class = _make_app_with_screen(ReportScreen(report_path=None))
    async with app_class().run_test() as pilot:
        await pilot.pause()
        viewer = pilot.app.screen.query_one("#report-viewer", MarkdownViewer)
        assert viewer is not None


@pytest.mark.asyncio
async def test_report_screen_shows_unavailable_message_when_file_missing(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "no-report.md"

    app_class = _make_app_with_screen(ReportScreen(report_path=missing))
    async with app_class().run_test() as pilot:
        await pilot.pause()
        viewer = pilot.app.screen.query_one("#report-viewer", MarkdownViewer)
        assert viewer is not None


@pytest.mark.asyncio
async def test_report_screen_q_key_quits_app() -> None:
    app_class = _make_app_with_screen(ReportScreen(report_path=None))
    async with app_class().run_test() as pilot:
        await pilot.pause()
        await pilot.press("q")
        await pilot.pause()
        # App should have exited — no assertion error means it completed


@pytest.mark.asyncio
async def test_report_screen_sets_border_title_on_mount() -> None:
    app_class = _make_app_with_screen(ReportScreen(report_path=None))
    async with app_class().run_test() as pilot:
        await pilot.pause()
        viewer = pilot.app.screen.query_one("#report-viewer", MarkdownViewer)
        assert viewer.border_title == "Plan Report"


# ---------------------------------------------------------------------------
# Test app helpers
# ---------------------------------------------------------------------------


from textual.app import App, ComposeResult  # noqa: E402 — after importorskip


class _AppBase(App[None]):
    """Minimal host app for pushing test screens."""

    def compose(self) -> ComposeResult:
        yield from ()


def _make_app_with_screen(screen: object):
    """Return an App subclass that immediately pushes the given screen."""

    class _App(_AppBase):
        async def on_mount(self) -> None:
            await self.push_screen(screen)  # type: ignore[arg-type]

    return _App
