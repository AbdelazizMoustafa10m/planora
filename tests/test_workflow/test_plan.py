from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING

import pytest

from planora.agents.registry import AgentConfig, AgentMode, OutputExtraction, StreamFormat
from planora.core.events import (
    AgentMonitorSnapshot,
    AgentResult,
    AgentState,
    PhaseResult,
    PhaseStatus,
    ToolExecution,
)
from planora.core.workspace import WorkspaceManager
from planora.workflow.plan import PlanWorkflow

if TYPE_CHECKING:
    from decimal import Decimal


def _agent_config(name: str) -> AgentConfig:
    return AgentConfig(
        name=name,
        binary="echo",
        model=f"{name}-model",
        subcommand="-p",
        flags={AgentMode.PLAN: [], AgentMode.FIX: []},
        stream_format=StreamFormat.CLAUDE,
        output_extraction=OutputExtraction(
            strategy=OutputExtraction.Strategy.STDOUT_CAPTURE,
        ),
    )


def _agent_result(
    agent_name: str,
    output_path,
    *,
    exit_code: int = 0,
    output_empty: bool = False,
    error: str | None = None,
    cost_usd: Decimal | None = None,
) -> AgentResult:
    return AgentResult(
        agent_name=agent_name,
        output_path=output_path,
        stream_path=output_path.with_suffix(".stream"),
        log_path=output_path.with_suffix(".log"),
        exit_code=exit_code,
        duration=timedelta(seconds=1),
        output_empty=output_empty,
        error=error,
        cost_usd=cost_usd,
    )


class _RegistryStub:
    def __init__(self, agents: dict[str, AgentConfig], missing: list[str] | None = None) -> None:
        self._agents = agents
        self._missing = missing or []
        self.validated: list[str] = []

    def validate(self, names: list[str]) -> list[str]:
        self.validated = names
        return list(self._missing)

    def get(self, name: str) -> AgentConfig:
        return self._agents[name]


@dataclass
class _RecordingUI:
    logs: list[tuple[str, str]] = field(default_factory=list)
    pipeline_updates: list[dict[str, PhaseStatus]] = field(default_factory=list)

    def on_phase_start(self, phase: str, label: str) -> None:
        del phase, label

    def on_phase_end(self, phase: str, result: PhaseResult) -> None:
        del phase, result

    def on_agent_start(self, agent: str, phase: str) -> None:
        del agent, phase

    def on_agent_end(self, agent: str, result: AgentResult) -> None:
        del agent, result

    def on_agent_state_change(self, agent: str, state: AgentState) -> None:
        del agent, state

    def on_tool_start(self, agent: str, tool: ToolExecution) -> None:
        del agent, tool

    def on_tool_done(self, agent: str, tool: ToolExecution) -> None:
        del agent, tool

    def on_cost_update(self, agent: str, cost_usd: Decimal) -> None:
        del agent, cost_usd

    def on_stall(self, agent: str, idle_seconds: float) -> None:
        del agent, idle_seconds

    def on_rate_limit(self, agent: str, retry_after_ms: int | None) -> None:
        del agent, retry_after_ms

    def on_retry(self, agent: str, attempt: int, max_retries: int, error: str) -> None:
        del agent, attempt, max_retries, error

    def on_snapshot(self, snapshot: AgentMonitorSnapshot) -> None:
        del snapshot

    def on_log(self, level: str, message: str) -> None:
        self.logs.append((level, message))

    def on_pipeline_update(self, statuses: dict[str, PhaseStatus]) -> None:
        self.pipeline_updates.append(dict(statuses))

    def dispatch_agent_event(self, agent: str, event) -> None:
        del agent, event


class _PhaseRunnerStub:
    def __init__(self, handlers: dict[str, object]) -> None:
        self._handlers = handlers
        self.calls: list[tuple[str, str]] = []

    async def run_phase(self, name, agent, prompt, output_path, dry_run):
        del prompt, dry_run
        self.calls.append(("phase", name))
        return await self._handlers[name](agent, output_path)

    async def run_parallel(self, name, agents, dry_run):
        del dry_run
        self.calls.append(("parallel", name))
        return await self._handlers[name](agents)


def _install_report_stubs(monkeypatch, workspace: WorkspaceManager, tmp_path) -> None:
    def fake_generate_plan_report(**kwargs):
        del kwargs
        return workspace.write_file("plan-report.md", "# report\n")

    def fake_archive():
        archive_path = tmp_path / "reports" / "plans" / "archive"
        archive_path.mkdir(parents=True, exist_ok=True)
        return archive_path

    monkeypatch.setattr("planora.workflow.plan.generate_plan_report", fake_generate_plan_report)
    monkeypatch.setattr(workspace, "archive", fake_archive)


@pytest.mark.asyncio
async def test_run_aborts_after_failed_plan(monkeypatch, tmp_path) -> None:
    planner = _agent_config("claude")
    registry = _RegistryStub({"claude": planner})
    workspace = WorkspaceManager(tmp_path)
    ui = _RecordingUI()

    async def plan_handler(agent: AgentConfig, output_path):
        del agent, output_path
        return PhaseResult(name="plan", status=PhaseStatus.FAILED, error="plan failed")

    workflow = PlanWorkflow(
        workspace=workspace,
        registry=registry,
        runner=object(),
        ui=ui,
        planner="claude",
        auditors=[],
    )
    workflow._phase_runner = _PhaseRunnerStub({"plan": plan_handler})
    monkeypatch.setattr(
        "planora.workflow.plan.build_plan_prompt",
        lambda *_args, **_kwargs: "plan prompt",
    )

    result = await workflow.run("Add tests for phase 4")

    assert result.success is False
    assert [phase.name for phase in result.phases] == ["plan"]
    assert result.report_path is None
    assert result.final_plan_path is None
    assert registry.validated == ["claude"]
    assert (workspace.workspace_dir / "task-input.md").read_text(encoding="utf-8") == (
        "Add tests for phase 4"
    )
    assert ui.pipeline_updates == [{}, {"plan": PhaseStatus.FAILED}]


@pytest.mark.asyncio
async def test_run_skips_refinement_when_all_auditors_fail(monkeypatch, tmp_path) -> None:
    planner = _agent_config("claude")
    gemini = _agent_config("gemini")
    codex = _agent_config("codex")
    registry = _RegistryStub({"claude": planner, "gemini": gemini, "codex": codex})
    workspace = WorkspaceManager(tmp_path)
    ui = _RecordingUI()

    async def plan_handler(agent: AgentConfig, output_path):
        del agent
        output_path.write_text("# Initial plan\n", encoding="utf-8")
        return PhaseResult(
            name="plan",
            status=PhaseStatus.DONE,
            output_files=[output_path],
            agent_results=[_agent_result("claude", output_path)],
        )

    async def audit_handler(agents):
        results = []
        for agent, _prompt, output_path in agents:
            results.append(
                _agent_result(
                    agent.name,
                    output_path,
                    exit_code=1,
                    output_empty=True,
                    error="audit failed",
                )
            )
        return PhaseResult(
            name="audit",
            status=PhaseStatus.FAILED,
            agent_results=results,
            error="all auditors failed",
        )

    workflow = PlanWorkflow(
        workspace=workspace,
        registry=registry,
        runner=object(),
        ui=ui,
        planner="claude",
        auditors=["gemini", "codex"],
    )
    workflow._phase_runner = _PhaseRunnerStub(
        {
            "plan": plan_handler,
            "audit": audit_handler,
        }
    )
    _install_report_stubs(monkeypatch, workspace, tmp_path)
    monkeypatch.setattr(
        "planora.workflow.plan.build_plan_prompt",
        lambda *_args, **_kwargs: "plan prompt",
    )
    monkeypatch.setattr("planora.workflow.plan.build_audit_prompt", lambda **_kwargs: "audit")

    result = await workflow.run("Add tests for phase 4")

    assert result.success is True
    assert [phase.name for phase in result.phases] == ["plan", "audit", "report"]
    assert result.final_plan_path == workspace.workspace_dir / "initial-plan.md"
    assert any("Skipping refinement" in message for _level, message in ui.logs)
    assert ("phase", "refine") not in workflow._phase_runner.calls


@pytest.mark.asyncio
async def test_run_copies_previous_plan_when_refine_fails(monkeypatch, tmp_path) -> None:
    planner = _agent_config("claude")
    gemini = _agent_config("gemini")
    registry = _RegistryStub({"claude": planner, "gemini": gemini})
    workspace = WorkspaceManager(tmp_path)
    ui = _RecordingUI()

    async def plan_handler(agent: AgentConfig, output_path):
        del agent
        output_path.write_text("# Initial plan\nKeep this content\n", encoding="utf-8")
        return PhaseResult(
            name="plan",
            status=PhaseStatus.DONE,
            output_files=[output_path],
            agent_results=[_agent_result("claude", output_path)],
        )

    async def audit_handler(agents):
        agent, _prompt, output_path = agents[0]
        output_path.write_text("# Audit report\nFix issue\n", encoding="utf-8")
        return PhaseResult(
            name="audit",
            status=PhaseStatus.DONE,
            output_files=[output_path],
            agent_results=[_agent_result(agent.name, output_path)],
        )

    async def refine_handler(agent: AgentConfig, output_path):
        del agent, output_path
        return PhaseResult(
            name="refine",
            status=PhaseStatus.FAILED,
            agent_results=[
                _agent_result(
                    "claude",
                    workspace.workspace_dir / "final-plan.md",
                    exit_code=1,
                    output_empty=True,
                    error="refine failed",
                )
            ],
            error="refine failed",
        )

    workflow = PlanWorkflow(
        workspace=workspace,
        registry=registry,
        runner=object(),
        ui=ui,
        planner="claude",
        auditors=["gemini"],
    )
    workflow._phase_runner = _PhaseRunnerStub(
        {
            "plan": plan_handler,
            "audit": audit_handler,
            "refine": refine_handler,
        }
    )
    _install_report_stubs(monkeypatch, workspace, tmp_path)
    monkeypatch.setattr(
        "planora.workflow.plan.build_plan_prompt",
        lambda *_args, **_kwargs: "plan prompt",
    )
    monkeypatch.setattr("planora.workflow.plan.build_audit_prompt", lambda **_kwargs: "audit")
    monkeypatch.setattr(
        "planora.workflow.plan.build_refinement_prompt",
        lambda **_kwargs: "refine",
    )

    result = await workflow.run("Add tests for phase 4")

    final_plan = workspace.workspace_dir / "final-plan.md"
    assert result.success is True
    assert [phase.name for phase in result.phases] == ["plan", "audit", "refine", "report"]
    assert result.final_plan_path == final_plan
    assert final_plan.read_text(encoding="utf-8") == "# Initial plan\nKeep this content\n"
    assert any("Using previous plan content as final" in message for _level, message in ui.logs)


@pytest.mark.asyncio
async def test_run_report_phase_failure_is_non_fatal(monkeypatch, tmp_path) -> None:
    planner = _agent_config("claude")
    registry = _RegistryStub({"claude": planner})
    workspace = WorkspaceManager(tmp_path)
    ui = _RecordingUI()

    async def plan_handler(agent: AgentConfig, output_path):
        del agent
        output_path.write_text("# Initial plan\n", encoding="utf-8")
        return PhaseResult(
            name="plan",
            status=PhaseStatus.DONE,
            output_files=[output_path],
            agent_results=[_agent_result("claude", output_path)],
        )

    workflow = PlanWorkflow(
        workspace=workspace,
        registry=registry,
        runner=object(),
        ui=ui,
        planner="claude",
        auditors=[],
    )
    workflow._phase_runner = _PhaseRunnerStub({"plan": plan_handler})
    monkeypatch.setattr(
        "planora.workflow.plan.build_plan_prompt",
        lambda *_args, **_kwargs: "plan prompt",
    )
    monkeypatch.setattr(
        "planora.workflow.plan.generate_plan_report",
        lambda **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    result = await workflow.run("Add tests for phase 4")

    assert result.success is True
    assert [phase.name for phase in result.phases] == ["plan", "audit"]
    assert result.phases[1].status == PhaseStatus.SKIPPED
    assert result.report_path is None
    assert result.archive_path is None
    assert result.final_plan_path == workspace.workspace_dir / "initial-plan.md"
    assert any("Report/archive step failed" in message for _level, message in ui.logs)


@pytest.mark.asyncio
async def test_run_skip_planning_reuses_existing_initial_plan(monkeypatch, tmp_path) -> None:
    planner = _agent_config("claude")
    registry = _RegistryStub({"claude": planner})
    workspace = WorkspaceManager(tmp_path)
    workspace.ensure_dirs()
    initial_plan = workspace.write_file("initial-plan.md", "# Existing plan\n")
    ui = _RecordingUI()

    workflow = PlanWorkflow(
        workspace=workspace,
        registry=registry,
        runner=object(),
        ui=ui,
        planner="claude",
        auditors=[],
        skip_planning=True,
        reuse_workspace=True,
    )
    _install_report_stubs(monkeypatch, workspace, tmp_path)

    result = await workflow.run("Resume the existing planning run")

    assert result.success is True
    assert [phase.name for phase in result.phases] == ["plan", "audit", "report"]
    assert result.phases[0].status == PhaseStatus.SKIPPED
    assert initial_plan.read_text(encoding="utf-8") == "# Existing plan\n"
    assert result.final_plan_path == initial_plan
