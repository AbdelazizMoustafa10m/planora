from __future__ import annotations

import pytest

pytest.importorskip("textual")

from textual.widgets import Footer, Header

from planora.tui.app import PlanoraTUI
from planora.tui.screens.dashboard import DashboardScreen
from planora.tui.widgets.agent_activity import AgentActivityPanel
from planora.tui.widgets.event_log import EventLog
from planora.tui.widgets.pipeline import PipelineProgress
from planora.tui.widgets.status_panel import StatusPanel


def test_app_declares_expected_key_bindings() -> None:
    bindings = {(binding.key, binding.action) for binding in PlanoraTUI.BINDINGS}

    assert bindings == {
        ("p", "pause"),
        ("c", "cancel"),
        ("s", "skip_phase"),
        ("l", "toggle_log"),
        ("q", "quit"),
        ("ctrl+p", "command_palette"),
    }


@pytest.mark.asyncio
async def test_app_mounts_dashboard_widgets(monkeypatch) -> None:
    monkeypatch.setattr(PlanoraTUI, "_launch_workflow", lambda self: None)
    app = PlanoraTUI(task_input="show dashboard")

    async with app.run_test():
        screen = app.screen
        assert isinstance(screen, DashboardScreen)
        assert isinstance(screen.query_one(Header), Header)
        assert isinstance(screen.query_one(PipelineProgress), PipelineProgress)
        assert isinstance(screen.query_one(AgentActivityPanel), AgentActivityPanel)
        assert isinstance(screen.query_one(StatusPanel), StatusPanel)
        assert isinstance(screen.query_one(EventLog), EventLog)
        assert isinstance(screen.query_one(Footer), Footer)


@pytest.mark.asyncio
async def test_on_mount_starts_worker_when_task_input_is_present(monkeypatch) -> None:
    started = []

    def fake_launch(self) -> None:
        started.append(self.task_input)

    monkeypatch.setattr(PlanoraTUI, "_launch_workflow", fake_launch)

    app = PlanoraTUI(task_input="ship the test suite")
    async with app.run_test():
        pass

    assert len(started) == 1


@pytest.mark.asyncio
async def test_toggle_log_hides_and_restores_event_log(monkeypatch) -> None:
    monkeypatch.setattr(PlanoraTUI, "_launch_workflow", lambda self: None)
    app = PlanoraTUI(task_input="show dashboard")

    async with app.run_test() as pilot:
        log = app.screen.query_one(EventLog)
        assert log.has_class("hidden") is False

        app.action_toggle_log()
        await pilot.pause()
        assert log.has_class("hidden") is True

        app.action_toggle_log()
        await pilot.pause()
        assert log.has_class("hidden") is False


@pytest.mark.asyncio
async def test_action_quit_requests_cancellation_for_active_workflow(
    monkeypatch,
) -> None:
    app = PlanoraTUI()

    async with app.run_test():
        cancelled = []
        monkeypatch.setattr(app, "_workflow_is_active", lambda: True)
        monkeypatch.setattr(
            app,
            "_request_workflow_cancel",
            lambda reason: cancelled.append(reason),
        )

        await app.action_quit()

    assert cancelled == ["Quit requested. Cancelling workflow before exit."]
