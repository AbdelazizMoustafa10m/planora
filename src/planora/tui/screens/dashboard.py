"""Main execution dashboard screen."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

from planora.tui.widgets import (
    AgentActivityPanel,
    AgentOutputPanel,
    CostTracker,
    EventLog,
    PipelineProgress,
    StatusPanel,
)

if TYPE_CHECKING:
    from datetime import datetime, timedelta
    from decimal import Decimal

    from textual.app import ComposeResult

    from planora.core.events import AgentMonitorSnapshot, PhaseStatus


class DashboardScreen(Screen[None]):
    """Main monitoring dashboard for a workflow run."""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("Waiting for run configuration...", id="run-summary")
        yield PipelineProgress(id="pipeline-progress")
        yield Horizontal(
            Vertical(
                AgentActivityPanel(id="agent-activity"),
                AgentOutputPanel(id="agent-output"),
                id="left-panel",
            ),
            Vertical(
                StatusPanel(id="status-panel"),
                CostTracker(id="cost-tracker"),
                id="right-panel",
            ),
            id="dashboard-main",
        )
        yield EventLog(id="event-log")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#agent-activity", AgentActivityPanel).border_title = "Agent Activity"
        self.query_one("#agent-output", AgentOutputPanel).border_title = "Agent Output"
        self.query_one("#status-panel", StatusPanel).border_title = "Live Status"
        self.query_one("#cost-tracker", CostTracker).border_title = "Cost Tracker"
        self.query_one("#event-log", EventLog).border_title = "Event Log"

    def configure_run(
        self,
        *,
        task: str,
        planner: str,
        auditors: list[str],
        audit_rounds: int,
        max_concurrency: int,
        agents: list[str],
    ) -> None:
        """Reset the dashboard for a new workflow run."""
        task_preview = " ".join(task.split())
        if len(task_preview) > 120:
            task_preview = f"{task_preview[:117]}..."

        auditors_label = ", ".join(auditors) if auditors else "none"
        summary_lines = [
            f"Task: {task_preview or '(none)'}",
            (
                f"Planner: {planner}  |  Auditors: {auditors_label}  |  "
                f"Rounds: {audit_rounds}  |  Concurrency: {max_concurrency}"
            ),
        ]
        self.query_one("#run-summary", Static).update("\n".join(summary_lines))

        self.query_one("#pipeline-progress", PipelineProgress).configure(audit_rounds=audit_rounds)
        self.query_one("#agent-activity", AgentActivityPanel).set_agents(agents)
        self.query_one("#agent-activity", AgentActivityPanel).reset()
        self.query_one("#agent-output", AgentOutputPanel).set_agents(agents)
        self.query_one("#agent-output", AgentOutputPanel).reset()
        self.query_one("#status-panel", StatusPanel).reset()
        self.query_one("#status-panel", StatusPanel).set_run_context(
            agents=agents,
            max_concurrency=max_concurrency,
        )
        self.query_one("#cost-tracker", CostTracker).set_agents(agents)
        self.query_one("#cost-tracker", CostTracker).reset()
        self.query_one("#event-log", EventLog).reset()
        self.query_one("#event-log", EventLog).remove_class("hidden")

    def set_run_started(self, started_at: datetime) -> None:
        self.query_one("#status-panel", StatusPanel).set_run_started(started_at)

    def set_current_phase(self, label: str, started_at: datetime | None) -> None:
        self.query_one("#status-panel", StatusPanel).set_current_phase(
            label=label,
            started_at=started_at,
        )

    def update_pipeline(self, statuses: dict[str, PhaseStatus]) -> None:
        self.query_one("#pipeline-progress", PipelineProgress).update_statuses(statuses)

    def apply_snapshot(self, snapshot: AgentMonitorSnapshot) -> None:
        self.query_one("#agent-activity", AgentActivityPanel).apply_snapshot(snapshot)
        self.query_one("#status-panel", StatusPanel).apply_snapshot(snapshot)

    def update_costs(self, *, total_cost: Decimal, agent_costs: dict[str, Decimal]) -> None:
        self.query_one("#status-panel", StatusPanel).set_costs(
            total_cost=total_cost,
            agent_costs=agent_costs,
        )
        self.query_one("#cost-tracker", CostTracker).set_totals(agent_costs)

    def update_tool_counts(
        self,
        *,
        total: int,
        succeeded: int,
        failed: int,
        running: int,
    ) -> None:
        self.query_one("#status-panel", StatusPanel).set_tool_counts(
            total=total,
            succeeded=succeeded,
            failed=failed,
            running=running,
        )

    def update_turn_total(self, turns: int) -> None:
        self.query_one("#status-panel", StatusPanel).set_turn_total(turns)

    def append_output(self, agent: str, text: str) -> None:
        self.query_one("#agent-output", AgentOutputPanel).append_output(agent, text)

    def append_log(
        self,
        *,
        message: str,
        timestamp: datetime | None = None,
        level: str = "info",
        agent: str | None = None,
        icon: str | None = None,
        detail: str | None = None,
        duration: timedelta | float | None = None,
    ) -> None:
        self.query_one("#event-log", EventLog).append_entry(
            message=message,
            timestamp=timestamp,
            level=level,
            agent=agent,
            icon=icon,
            detail=detail,
            duration=duration,
        )

    def toggle_event_log(self) -> None:
        self.query_one("#event-log", EventLog).toggle_class("hidden")
