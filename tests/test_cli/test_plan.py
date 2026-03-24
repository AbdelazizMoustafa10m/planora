from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from planora.cli.app import app
from planora.cli.plan import _resolve_input_mode, parse_auditor_csv

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("csv_input", "expected"),
    [
        ("gemini,codex", ["gemini", "codex"]),
        ("gemini, codex", ["gemini", "codex"]),
        ("gemini,,codex", ["gemini", "codex"]),  # empty entry stripped
        ("gemini, gemini", ["gemini"]),  # dedup preserves order
        ("claude,gemini,claude", ["claude", "gemini"]),
        ("", []),
        ("  ", []),
    ],
)
def test_parse_auditor_csv_splits_and_deduplicates(
    csv_input: str, expected: list[str]
) -> None:
    assert parse_auditor_csv(csv_input) == expected


# ---------------------------------------------------------------------------
# _resolve_input_mode
# ---------------------------------------------------------------------------


def test_resolve_input_mode_returns_run_when_task_is_given() -> None:
    assert _resolve_input_mode("do the thing", None, False, False) == "run"


def test_resolve_input_mode_returns_run_when_task_file_is_given(tmp_path: Path) -> None:
    assert _resolve_input_mode(None, tmp_path / "task.md", False, False) == "run"


def test_resolve_input_mode_returns_wizard_when_interactive_flag_set() -> None:
    assert _resolve_input_mode(None, None, True, False) == "wizard"


def test_resolve_input_mode_prefers_wizard_over_run_when_interactive_flag_set() -> None:
    # Even with task provided, --interactive wins if interactive=True
    assert _resolve_input_mode("task text", None, True, False) == "wizard"


# ---------------------------------------------------------------------------
# plan run — basic invocation
# ---------------------------------------------------------------------------


runner = CliRunner()


def test_plan_run_exits_1_when_task_content_is_empty(tmp_path: Path) -> None:
    result = runner.invoke(app, ["plan", "run", ""])

    assert result.exit_code == 1
    assert "empty" in result.output.lower()


def test_plan_run_exits_1_when_task_file_is_missing(tmp_path: Path) -> None:
    missing = tmp_path / "no-such-file.md"

    result = runner.invoke(app, ["plan", "run", "--task-file", str(missing)])

    assert result.exit_code != 0


def test_plan_run_exits_1_when_task_file_is_empty(tmp_path: Path) -> None:
    empty_file = tmp_path / "empty.md"
    empty_file.write_text("", encoding="utf-8")

    result = runner.invoke(app, ["plan", "run", "--task-file", str(empty_file)])

    assert result.exit_code == 1
    assert "empty" in result.output.lower()


def test_plan_run_exits_1_when_planner_binary_not_on_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("planora.agents.registry.shutil.which", lambda _: None)

    result = runner.invoke(
        app,
        ["plan", "run", "Add auth", "--planner", "nonexistent-agent-xyz"],
    )

    assert result.exit_code == 1


def test_plan_run_skip_planning_exits_1_without_existing_initial_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        ["plan", "run", "Add auth", "--skip-planning", "--project-root", str(tmp_path)],
    )

    assert result.exit_code == 1
    assert "skip-planning" in result.output.lower()


def test_plan_run_invalid_config_override_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = runner.invoke(
        app,
        ["plan", "run", "Add auth", "--config", "no-equals-sign"],
    )

    assert result.exit_code == 1


def test_plan_run_unknown_profile_exits_1() -> None:
    result = runner.invoke(
        app,
        ["plan", "run", "Add auth", "--profile", "nonexistent-profile-xyz"],
    )

    assert result.exit_code == 1


def _make_fake_workflow_class(captured: dict[str, object]):
    """Return a fake PlanWorkflow class that records constructor kwargs and run calls."""
    from datetime import timedelta  # noqa: PLC0415

    from planora.core.events import PlanResult  # noqa: PLC0415

    class FakeWorkflow:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        async def run(self, task_content: str) -> PlanResult:
            captured["task_content"] = task_content
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

    return FakeWorkflow


def test_plan_run_dry_run_succeeds_without_invoking_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With --dry-run and a mocked workflow, the command should complete successfully."""
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "planora.workflow.plan.PlanWorkflow", _make_fake_workflow_class(captured)
    )
    monkeypatch.setattr("planora.agents.registry.shutil.which", lambda name: f"/usr/bin/{name}")

    result = runner.invoke(
        app,
        [
            "plan",
            "run",
            "Add unit tests",
            "--dry-run",
            "--project-root",
            str(tmp_path),
            "--auditors",
            "",
        ],
    )

    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# plan run — skip flags
# ---------------------------------------------------------------------------


def test_plan_run_skip_audit_uses_empty_auditor_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--skip-audit should set auditors to [] and skip audit phases."""
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "planora.workflow.plan.PlanWorkflow", _make_fake_workflow_class(captured)
    )
    monkeypatch.setattr("planora.agents.registry.shutil.which", lambda name: f"/usr/bin/{name}")

    runner.invoke(
        app,
        [
            "plan",
            "run",
            "Add unit tests",
            "--skip-audit",
            "--project-root",
            str(tmp_path),
        ],
    )

    assert captured.get("auditors") == []


def test_plan_run_skip_refinement_passes_flag_to_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--skip-refinement should be forwarded to PlanWorkflow as skip_refinement=True."""
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "planora.workflow.plan.PlanWorkflow", _make_fake_workflow_class(captured)
    )
    monkeypatch.setattr("planora.agents.registry.shutil.which", lambda name: f"/usr/bin/{name}")

    runner.invoke(
        app,
        [
            "plan",
            "run",
            "Add unit tests",
            "--skip-refinement",
            "--project-root",
            str(tmp_path),
        ],
    )

    assert captured.get("skip_refinement") is True


# ---------------------------------------------------------------------------
# plan run — output-format events
# ---------------------------------------------------------------------------


def test_plan_run_output_format_events_uses_events_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "planora.workflow.plan.PlanWorkflow", _make_fake_workflow_class(captured)
    )
    monkeypatch.setattr("planora.agents.registry.shutil.which", lambda name: f"/usr/bin/{name}")

    runner.invoke(
        app,
        [
            "plan",
            "run",
            "Add unit tests",
            "--output-format",
            "events",
            "--project-root",
            str(tmp_path),
        ],
    )

    from planora.cli.callbacks import EventsOutputCallback  # noqa: PLC0415

    assert isinstance(captured.get("ui"), EventsOutputCallback)


# ---------------------------------------------------------------------------
# plan resume
# ---------------------------------------------------------------------------


def test_plan_resume_exits_1_when_no_workspace_exists(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["plan", "resume", "--project-root", str(tmp_path)]
    )

    assert result.exit_code == 1
    assert "workspace" in result.output.lower()


def test_plan_resume_exits_1_when_task_input_md_missing(tmp_path: Path) -> None:
    workspace = tmp_path / ".plan-workspace"
    workspace.mkdir()

    result = runner.invoke(
        app, ["plan", "resume", "--project-root", str(tmp_path)]
    )

    assert result.exit_code == 1
    assert "task-input.md" in result.output


def test_plan_resume_succeeds_when_run_already_complete(tmp_path: Path) -> None:
    workspace = tmp_path / ".plan-workspace"
    workspace.mkdir()
    (workspace / "task-input.md").write_text("Add tests\n", encoding="utf-8")
    (workspace / "initial-plan.md").write_text("# Plan\n", encoding="utf-8")
    (workspace / "final-plan.md").write_text("# Final\n", encoding="utf-8")
    (workspace / "plan-report.md").write_text("# Report\n", encoding="utf-8")

    result = runner.invoke(
        app, ["plan", "resume", "--project-root", str(tmp_path)]
    )

    assert result.exit_code == 0
    assert "already complete" in result.output.lower()


def test_plan_resume_restarts_from_plan_phase_when_no_initial_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / ".plan-workspace"
    workspace.mkdir()
    (workspace / "task-input.md").write_text("Add tests\n", encoding="utf-8")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "planora.workflow.plan.PlanWorkflow", _make_fake_workflow_class(captured)
    )
    monkeypatch.setattr("planora.agents.registry.shutil.which", lambda name: f"/usr/bin/{name}")

    result = runner.invoke(
        app, ["plan", "resume", "--project-root", str(tmp_path)]
    )

    assert "Restarting from plan phase" in result.output


def test_plan_resume_output_format_events_uses_events_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / ".plan-workspace"
    workspace.mkdir()
    (workspace / "task-input.md").write_text("Add tests\n", encoding="utf-8")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "planora.workflow.plan.PlanWorkflow", _make_fake_workflow_class(captured)
    )
    monkeypatch.setattr("planora.agents.registry.shutil.which", lambda name: f"/usr/bin/{name}")

    runner.invoke(
        app,
        [
            "plan",
            "resume",
            "--project-root",
            str(tmp_path),
            "--output-format",
            "events",
        ],
    )

    from planora.cli.callbacks import EventsOutputCallback  # noqa: PLC0415

    assert isinstance(captured.get("ui"), EventsOutputCallback)
