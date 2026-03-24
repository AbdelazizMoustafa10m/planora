from __future__ import annotations

from datetime import timedelta

import pytest

pytest.importorskip("textual")

from textual.widgets import Footer, Header

from planora.core.events import PlanResult
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


@pytest.mark.asyncio
async def test_execute_workflow_uses_injected_dependencies(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}
    posted_messages = []
    registry = object()
    runner = object()

    class DummyWorkflow:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        async def run(self, task_input: str) -> PlanResult:
            captured["task_input"] = task_input
            return PlanResult(
                phases=[],
                final_plan_path=None,
                report_path=None,
                archive_path=None,
                total_duration=timedelta(0),
                total_cost_usd=None,
                agent_results={},
                success=True,
            )

    monkeypatch.setattr("planora.workflow.plan.PlanWorkflow", DummyWorkflow)

    app = PlanoraTUI(
        task_input="ship the dashboard",
        planner="claude",
        auditors=["gemini"],
        audit_rounds=2,
        max_concurrency=4,
        project_root=tmp_path,
        registry=registry,
        runner=runner,
        plan_template_path=tmp_path / "plan.md",
        audit_template_path=tmp_path / "audit.md",
        refine_template_path=tmp_path / "refine.md",
        prompt_base_dir=tmp_path / "prompts",
    )
    monkeypatch.setattr(app, "post_message", lambda message: posted_messages.append(message))

    await PlanoraTUI._execute_workflow.__wrapped__(app)

    assert captured["registry"] is registry
    assert captured["runner"] is runner
    assert captured["planner"] == "claude"
    assert captured["auditors"] == ["gemini"]
    assert captured["audit_rounds"] == 2
    assert captured["max_concurrency"] == 4
    assert captured["plan_template_path"] == tmp_path / "plan.md"
    assert captured["audit_template_path"] == tmp_path / "audit.md"
    assert captured["refine_template_path"] == tmp_path / "refine.md"
    assert captured["prompt_base_dir"] == tmp_path / "prompts"
    assert captured["task_input"] == "ship the dashboard"
    assert posted_messages
