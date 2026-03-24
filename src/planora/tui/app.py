"""Main Textual application for the Planora TUI."""

from __future__ import annotations

import asyncio
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from textual import work
from textual.app import App
from textual.binding import Binding
from textual.message import Message
from textual.worker import Worker, WorkerState

from planora.core.events import PhaseStatus, PlanResult
from planora.tui.callbacks import (
    AgentFinished,
    AgentOutputChunk,
    AgentSnapshotUpdated,
    AgentStarted,
    CostUpdated,
    LogEvent,
    PhaseFinished,
    PhaseStarted,
    PipelineUpdated,
    TextualUICallback,
    ToolFinished,
    ToolStarted,
)
from planora.tui.screens.dashboard import DashboardScreen
from planora.tui.screens.report import ReportScreen
from planora.tui.screens.wizard import WizardLaunch, WizardScreen

if TYPE_CHECKING:
    from planora.agents.registry import AgentRegistry
    from planora.agents.runner import AgentRunner
    from planora.core.config import PlanораSettings
    from planora.workflow.engine import WorkflowControl


class WorkflowCompleted(Message, bubble=False):
    """Internal message emitted when the workflow worker finishes cleanly."""

    def __init__(self, result: PlanResult) -> None:
        self.result = result
        super().__init__()


class WorkflowFailed(Message, bubble=False):
    """Internal message emitted when the workflow worker fails."""

    def __init__(self, error: BaseException) -> None:
        self.error = error
        super().__init__()


class PlanoraTUI(App[None]):
    """Interactive TUI entrypoint for the Planora planning workflow."""

    CSS_PATH = "styles/app.tcss"
    BINDINGS = [
        Binding("p", "pause", "Pause"),
        Binding("c", "cancel", "Cancel"),
        Binding("s", "skip_phase", "Skip phase"),
        Binding("l", "toggle_log", "Toggle log"),
        Binding("q", "quit", "Quit"),
        Binding("ctrl+p", "command_palette", "Commands"),
    ]

    def __init__(
        self,
        task_input: str = "",
        planner: str = "claude",
        auditors: list[str] | None = None,
        audit_rounds: int = 1,
        max_concurrency: int = 3,
        project_root: Path | None = None,
        registry: AgentRegistry | None = None,
        runner: AgentRunner | None = None,
        plan_template_path: Path | None = None,
        audit_template_path: Path | None = None,
        refine_template_path: Path | None = None,
        prompt_base_dir: Path = Path("."),
        settings: PlanораSettings | None = None,
    ) -> None:
        super().__init__()
        self.task_input = task_input
        self._planner = planner
        self._auditors = _dedupe(auditors or ["gemini", "codex"])
        self._audit_rounds = audit_rounds
        self._max_concurrency = max_concurrency
        self._project_root = project_root or Path.cwd()
        self._agent_registry = registry
        self._agent_runner = runner
        self._plan_template_path = plan_template_path
        self._audit_template_path = audit_template_path
        self._refine_template_path = refine_template_path
        self._prompt_base_dir = prompt_base_dir
        self._settings = settings

        self._dashboard_screen: DashboardScreen | None = None
        self._workflow_worker: Worker[None] | None = None

        self._shutting_down = False
        self._exit_when_stopped = False
        self._workflow_control: WorkflowControl | None = None
        self._workflow_phase_runner: object = None  # PhaseRunner, set when workflow starts
        self._is_paused = False

        self._workflow_started_at: datetime | None = None
        self._phase_display_label = "Waiting"
        self._current_phase_key: str | None = None
        self._current_phase_started_at: datetime | None = None
        self._pipeline_statuses: dict[str, PhaseStatus] = {}
        self._agent_costs: dict[str, Decimal] = {}
        self._agent_running_tools: dict[str, int] = {}
        self._tool_total = 0
        self._tool_succeeded = 0
        self._tool_failed = 0
        self._turn_total = 0
        self._counted_cost_runs: set[tuple[str, int]] = set()
        self._counted_turn_runs: set[tuple[str, int]] = set()

    async def on_mount(self) -> None:
        dashboard = DashboardScreen()
        self._dashboard_screen = dashboard
        await self.push_screen(dashboard)

        if self.task_input.strip():
            self._launch_workflow()
            return

        await self.push_screen(self._build_wizard(), callback=self._handle_wizard_result)

    async def action_quit(self) -> None:
        if self._workflow_is_active():
            if self._shutting_down:
                self._log_system("Force quit requested while cancellation is in progress.")
                self.exit()
                return

            self._exit_when_stopped = True
            self._request_workflow_cancel(
                "Quit requested. Cancelling workflow before exit.",
            )
            return

        self.exit()

    def action_cancel(self) -> None:
        if not self._workflow_is_active():
            self.notify("No workflow is running.", severity="information")
            return

        self._request_workflow_cancel("Cancellation requested.")

    def action_pause(self) -> None:
        if not self._workflow_is_active():
            self.notify("No workflow is running.", severity="information")
            return

        if self._workflow_control is None:
            self.notify("Workflow control not available.", severity="warning")
            return

        if self._is_paused:
            self._is_paused = False
            self._workflow_control.resume()
            self.notify("Workflow resumed.", severity="information")
            self._log_system("Workflow resumed.", level="info")
        else:
            self._is_paused = True
            self._workflow_control.pause()
            self.notify("Workflow paused. Press P to resume.", severity="warning")
            self._log_system("Workflow paused at next phase boundary.", level="warning")

    def action_skip_phase(self) -> None:
        if not self._workflow_is_active():
            self.notify("No workflow is running.", severity="information")
            return

        if self._workflow_control is None:
            self.notify("Workflow control not available.", severity="warning")
            return

        self._workflow_control.request_skip()
        self._phase_runner_terminate_active()
        self.notify("Skipping current phase...", severity="warning")
        self._log_system("Phase skip requested — terminating current agent.", level="warning")

    def action_toggle_log(self) -> None:
        if self._dashboard_screen is not None:
            self._dashboard_screen.toggle_event_log()

    def on_worker_state_changed(self, message: Worker.StateChanged) -> None:
        if message.worker is not self._workflow_worker:
            return

        if message.state == WorkerState.CANCELLED:
            self._handle_workflow_cancelled()
        elif message.state == WorkerState.ERROR and message.worker.error is not None:
            self._handle_workflow_failure(message.worker.error)

    def on_planora_phase_started(self, message: PhaseStarted) -> None:
        self._current_phase_key = message.phase
        self._phase_display_label = _phase_label(message.phase)
        self._current_phase_started_at = datetime.now()
        self._pipeline_statuses[message.phase] = PhaseStatus.RUNNING
        self._sync_phase_widgets()

    def on_planora_phase_finished(self, message: PhaseFinished) -> None:
        self._pipeline_statuses[message.phase] = message.result.status
        if self._current_phase_key == message.phase:
            self._current_phase_key = None
            self._phase_display_label = _phase_label(message.phase)
            self._current_phase_started_at = None
        self._sync_phase_widgets()

    def on_planora_pipeline_updated(self, message: PipelineUpdated) -> None:
        self._pipeline_statuses.update(message.statuses)
        self._sync_phase_widgets()

    def on_planora_agent_started(self, message: AgentStarted) -> None:
        self._agent_running_tools.setdefault(message.agent, 0)
        self._sync_tool_counts()

    def on_planora_agent_snapshot_updated(self, message: AgentSnapshotUpdated) -> None:
        self._agent_running_tools[message.snapshot.agent_name] = len(message.snapshot.active_tools)
        if self._dashboard_screen is not None:
            self._dashboard_screen.apply_snapshot(message.snapshot)
        self._sync_tool_counts()

    def on_planora_tool_started(self, message: ToolStarted) -> None:
        self._tool_total += 1
        self._sync_tool_counts()
        if self._dashboard_screen is not None:
            self._dashboard_screen.append_log(
                message=f"{message.tool.friendly_name} started",
                agent=message.agent,
                icon="⟳",
                detail=message.tool.detail,
                timestamp=message.tool.started_at,
            )

    def on_planora_tool_finished(self, message: ToolFinished) -> None:
        if message.tool.status == "error":
            self._tool_failed += 1
            level = "error"
            icon = "✖"
        else:
            self._tool_succeeded += 1
            level = "info"
            icon = "✔"

        self._sync_tool_counts()
        if self._dashboard_screen is not None:
            self._dashboard_screen.append_log(
                message=message.tool.friendly_name,
                agent=message.agent,
                icon=icon,
                detail=message.tool.detail,
                duration=message.tool.duration,
                timestamp=message.tool.completed_at or datetime.now(),
                level=level,
            )

    def on_planora_cost_updated(self, message: CostUpdated) -> None:
        run_key = (message.agent, message.run_id)
        if run_key in self._counted_cost_runs:
            return

        self._counted_cost_runs.add(run_key)
        self._agent_costs[message.agent] = (
            self._agent_costs.get(message.agent, Decimal("0")) + message.cost_usd
        )
        self._sync_costs()

    def on_planora_agent_finished(self, message: AgentFinished) -> None:
        run_key = (message.agent, message.run_id)
        if message.result.num_turns is not None and run_key not in self._counted_turn_runs:
            self._counted_turn_runs.add(run_key)
            self._turn_total += message.result.num_turns
            self._sync_turns()

    def on_planora_agent_output_chunk(self, message: AgentOutputChunk) -> None:
        if self._dashboard_screen is not None:
            self._dashboard_screen.append_output(message.agent, message.text)

    def on_planora_log_event(self, message: LogEvent) -> None:
        if self._dashboard_screen is not None:
            self._dashboard_screen.append_log(
                message=message.message,
                timestamp=message.timestamp,
                level=message.level,
                agent=message.agent,
            )

        if message.notify:
            self.notify(message.message, severity=_notification_severity(message.level))

    async def on_workflow_completed(self, message: WorkflowCompleted) -> None:
        self._workflow_worker = None
        self._shutting_down = False
        result = message.result

        self._pipeline_statuses.update({phase.name: phase.status for phase in result.phases})
        if "report" not in self._pipeline_statuses and result.success:
            self._pipeline_statuses["report"] = (
                PhaseStatus.DONE if result.report_path is not None else PhaseStatus.FAILED
            )

        self._current_phase_key = None
        self._current_phase_started_at = None
        self._phase_display_label = "Completed" if result.success else "Failed"
        self._agent_running_tools = {agent: 0 for agent in self._agent_running_tools}
        self._sync_phase_widgets()
        self._sync_tool_counts()

        if result.success:
            self.notify("Planning complete.", severity="information")
            self._log_system("Workflow completed successfully.")
        else:
            self.notify("Planning failed.", severity="error")
            self._log_system("Workflow completed with failures.", level="error")

        if self._exit_when_stopped:
            self.exit()
            return

        if result.report_path is not None:
            await self.push_screen(ReportScreen(result.report_path))

    def on_workflow_failed(self, message: WorkflowFailed) -> None:
        self._handle_workflow_failure(message.error)

    @work(
        group="workflow",
        exclusive=True,
        exit_on_error=False,
        description="Execute the Planora planning workflow",
    )
    async def _execute_workflow(self) -> None:
        from planora.agents.registry import AgentRegistry
        from planora.agents.runner import AgentRunner
        from planora.core.workspace import WorkspaceManager
        from planora.workflow.engine import WorkflowControl
        from planora.workflow.plan import PlanWorkflow

        control = WorkflowControl()
        self._workflow_control = control

        # Resolve workspace with settings-driven reports_dir when available
        settings = self._settings
        if settings is not None:
            workspace = WorkspaceManager(
                self._project_root,
                reports_dir=Path(settings.effective_reports_dir),
            )
            snapshot_interval = settings.effective_cli_status_interval
            stall_check_interval = settings.effective_monitor_interval
        else:
            workspace = WorkspaceManager(self._project_root)
            snapshot_interval = None
            stall_check_interval = 5.0

        workflow = PlanWorkflow(
            workspace=workspace,
            registry=self._agent_registry or AgentRegistry(),
            runner=self._agent_runner or AgentRunner(),
            ui=TextualUICallback(self),
            planner=self._planner,
            auditors=self._auditors,
            audit_rounds=self._audit_rounds,
            max_concurrency=self._max_concurrency,
            control=control,
            plan_template_path=self._plan_template_path,
            audit_template_path=self._audit_template_path,
            refine_template_path=self._refine_template_path,
            prompt_base_dir=self._prompt_base_dir,
            snapshot_interval=snapshot_interval,
            stall_check_interval=stall_check_interval,
            settings=settings,
        )
        self._workflow_phase_runner = workflow._phase_runner

        try:
            result = await workflow.run(self.task_input)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            self.post_message(WorkflowFailed(exc))
        else:
            self.post_message(WorkflowCompleted(result))

    def _build_wizard(self) -> WizardScreen:
        return WizardScreen(
            default_task=self.task_input,
            planner=self._planner,
            auditors=self._auditors,
            audit_rounds=self._audit_rounds,
            max_concurrency=self._max_concurrency,
        )

    def _handle_wizard_result(self, result: WizardLaunch | None) -> None:
        if result is None:
            self.exit()
            return

        self.task_input = result["task"]
        self._planner = result["planner"]
        self._auditors = _dedupe(result["auditors"])
        self._audit_rounds = result["audit_rounds"]
        self._max_concurrency = result["max_concurrency"]
        self._launch_workflow()

    def _launch_workflow(self) -> None:
        if self._dashboard_screen is None:
            raise RuntimeError("Dashboard screen is not mounted")

        self._reset_runtime_state()

        self._dashboard_screen.configure_run(
            task=self.task_input,
            planner=self._planner,
            auditors=self._auditors,
            audit_rounds=self._audit_rounds,
            max_concurrency=self._max_concurrency,
            agents=self._agent_order(),
        )

        self._workflow_started_at = datetime.now()
        self._dashboard_screen.set_run_started(self._workflow_started_at)
        self._sync_phase_widgets()
        self._sync_tool_counts()
        self._sync_costs()
        self._sync_turns()
        self._log_system("Workflow starting.")

        self._workflow_worker = self._execute_workflow()

    def _phase_runner_terminate_active(self) -> None:
        """Terminate active subprocess in the current phase (for skip-phase support)."""
        from planora.workflow.engine import PhaseRunner

        if isinstance(self._workflow_phase_runner, PhaseRunner):
            self._workflow_phase_runner.terminate_active_processes()

    def _request_workflow_cancel(self, reason: str) -> None:
        if not self._workflow_is_active():
            self.notify("No workflow is running.", severity="information")
            return

        if self._shutting_down:
            self.notify("Cancellation already in progress.", severity="warning")
            return

        self._shutting_down = True
        self.notify("Cancelling workflow...", severity="warning")
        self._log_system(reason, level="warning")

        if self._workflow_worker is not None:
            self._workflow_worker.cancel()

    def _handle_workflow_cancelled(self) -> None:
        self._workflow_worker = None
        self._shutting_down = False

        if self._current_phase_key is not None:
            self._pipeline_statuses[self._current_phase_key] = PhaseStatus.FAILED

        self._current_phase_key = None
        self._current_phase_started_at = None
        self._phase_display_label = "Cancelled"
        self._agent_running_tools = {agent: 0 for agent in self._agent_running_tools}
        self._sync_phase_widgets()
        self._sync_tool_counts()

        self.notify("Workflow cancelled.", severity="warning")
        self._log_system("Workflow cancelled.", level="warning")

        if self._exit_when_stopped:
            self.exit()

    def _handle_workflow_failure(self, error: BaseException) -> None:
        self._workflow_worker = None
        self._shutting_down = False

        if self._current_phase_key is not None:
            self._pipeline_statuses[self._current_phase_key] = PhaseStatus.FAILED

        self._current_phase_key = None
        self._current_phase_started_at = None
        self._phase_display_label = "Failed"
        self._agent_running_tools = {agent: 0 for agent in self._agent_running_tools}
        self._sync_phase_widgets()
        self._sync_tool_counts()

        self.notify(f"Workflow failed: {error}", severity="error")
        self._log_system(f"Workflow failed: {error}", level="error")

        if self._exit_when_stopped:
            self.exit()

    def _sync_phase_widgets(self) -> None:
        if self._dashboard_screen is None:
            return

        self._dashboard_screen.set_current_phase(
            self._phase_display_label,
            self._current_phase_started_at,
        )
        self._dashboard_screen.update_pipeline(self._pipeline_statuses)

    def _sync_costs(self) -> None:
        if self._dashboard_screen is None:
            return
        total_cost = sum(self._agent_costs.values(), Decimal("0"))
        self._dashboard_screen.update_costs(
            total_cost=total_cost,
            agent_costs=dict(self._agent_costs),
        )

    def _sync_tool_counts(self) -> None:
        if self._dashboard_screen is None:
            return

        running = sum(self._agent_running_tools.values())
        self._dashboard_screen.update_tool_counts(
            total=self._tool_total,
            succeeded=self._tool_succeeded,
            failed=self._tool_failed,
            running=running,
        )

    def _sync_turns(self) -> None:
        if self._dashboard_screen is None:
            return
        self._dashboard_screen.update_turn_total(self._turn_total)

    def _workflow_is_active(self) -> bool:
        if self._workflow_worker is None:
            return False
        return self._workflow_worker.state in {WorkerState.PENDING, WorkerState.RUNNING}

    def _reset_runtime_state(self) -> None:
        self._shutting_down = False
        self._exit_when_stopped = False
        self._workflow_control = None
        self._workflow_phase_runner = None
        self._is_paused = False
        self._workflow_started_at = None
        self._phase_display_label = "Waiting"
        self._current_phase_key = None
        self._current_phase_started_at = None
        self._pipeline_statuses.clear()
        self._agent_costs.clear()
        self._agent_running_tools = {agent: 0 for agent in self._agent_order()}
        self._tool_total = 0
        self._tool_succeeded = 0
        self._tool_failed = 0
        self._turn_total = 0
        self._counted_cost_runs.clear()
        self._counted_turn_runs.clear()

    def _agent_order(self) -> list[str]:
        return _dedupe([self._planner, *self._auditors])

    def _log_system(self, message: str, *, level: str = "info") -> None:
        if self._dashboard_screen is None:
            return
        self._dashboard_screen.append_log(message=message, level=level)


def _phase_label(phase: str) -> str:
    if phase == "plan":
        return "Plan"
    if phase == "report":
        return "Report"
    if phase == "audit":
        return "Audit R1"
    if phase == "refine":
        return "Refine R1"
    if phase.startswith("audit-r"):
        return f"Audit R{phase.removeprefix('audit-r')}"
    if phase.startswith("refine-r"):
        return f"Refine R{phase.removeprefix('refine-r')}"
    return phase.replace("-", " ").title()


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _notification_severity(level: str) -> Literal["information", "warning", "error"]:
    if level == "warning":
        return "warning"
    if level == "error":
        return "error"
    return "information"
