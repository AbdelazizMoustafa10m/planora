"""Live workflow status sidebar for the Planora dashboard."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

from textual.widgets import Static

from planora.core.events import AgentMonitorSnapshot, AgentState


class StatusPanel(Static):
    """Sidebar summarising phase, cost, agent status, counters, and stalls."""

    def __init__(
        self,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        super().__init__("", name=name, id=id, classes=classes, disabled=disabled)
        self._agent_order: list[str] = []
        self._max_concurrency = 0
        self._run_started_at: datetime | None = None
        self._current_phase_label = "Waiting"
        self._phase_started_at: datetime | None = None
        self._agent_snapshots: dict[str, AgentMonitorSnapshot] = {}
        self._agent_costs: dict[str, Decimal] = {}
        self._total_cost = Decimal("0")
        self._tool_total = 0
        self._tool_succeeded = 0
        self._tool_failed = 0
        self._tool_running = 0
        self._turn_total = 0
        self._stalled_agents: dict[str, float] = {}

    def on_mount(self) -> None:
        self.set_interval(1.0, self._rebuild)
        self._rebuild()

    def set_run_context(self, *, agents: list[str], max_concurrency: int) -> None:
        """Configure the set of agents and concurrency for the current run."""
        self._agent_order = list(dict.fromkeys(agents))
        self._max_concurrency = max_concurrency
        self._rebuild()

    def reset(self) -> None:
        """Reset transient workflow state for a new run."""
        self._run_started_at = None
        self._current_phase_label = "Waiting"
        self._phase_started_at = None
        self._agent_snapshots.clear()
        self._agent_costs.clear()
        self._total_cost = Decimal("0")
        self._tool_total = 0
        self._tool_succeeded = 0
        self._tool_failed = 0
        self._tool_running = 0
        self._turn_total = 0
        self._stalled_agents.clear()
        self._rebuild()

    def set_run_started(self, started_at: datetime) -> None:
        """Record when the workflow run started."""
        self._run_started_at = started_at
        self._rebuild()

    def set_current_phase(
        self,
        *,
        label: str,
        started_at: datetime | None,
    ) -> None:
        """Update the active phase label and timer anchor."""
        self._current_phase_label = label
        self._phase_started_at = started_at
        self._rebuild()

    def apply_snapshot(self, snapshot: AgentMonitorSnapshot) -> None:
        """Store a new point-in-time snapshot for an agent."""
        self._agent_snapshots[snapshot.agent_name] = snapshot
        if snapshot.state == AgentState.STALLED:
            self._stalled_agents[snapshot.agent_name] = snapshot.idle_seconds
        else:
            self._stalled_agents.pop(snapshot.agent_name, None)
        self._rebuild()

    def set_costs(self, *, total_cost: Decimal, agent_costs: dict[str, Decimal]) -> None:
        """Update pipeline-level cost state."""
        self._total_cost = total_cost
        self._agent_costs = dict(agent_costs)
        self._rebuild()

    def set_tool_counts(
        self,
        *,
        total: int,
        succeeded: int,
        failed: int,
        running: int,
    ) -> None:
        """Update aggregated tool counters."""
        self._tool_total = total
        self._tool_succeeded = succeeded
        self._tool_failed = failed
        self._tool_running = running
        self._rebuild()

    def set_turn_total(self, turns: int) -> None:
        """Update the total completed-turn count."""
        self._turn_total = turns
        self._rebuild()

    def _rebuild(self) -> None:
        lines = [
            f"Phase: {self._current_phase_label}",
            f"Elapsed: {_format_elapsed(self._run_started_at)}",
            f"Phase Elapsed: {_format_elapsed(self._phase_started_at)}",
            f"Cost: ${self._total_cost:.4f}",
            f"Concurrency: {self._max_concurrency}",
            "",
            "Agents:",
        ]

        if not self._agent_order:
            lines.append("  (no agents configured)")
        else:
            for agent in self._agent_order:
                snapshot = self._agent_snapshots.get(agent)
                symbol, label = _agent_status(snapshot)
                lines.append(f"  {symbol} {agent:<10} {label}")

        lines.extend(
            [
                "",
                f"Tools: ✔{self._tool_succeeded}  ⟳{self._tool_running}  ✖{self._tool_failed}",
                f"Turns: {self._turn_total}",
            ]
        )

        if self._stalled_agents:
            stalled = ", ".join(
                f"{agent} ({seconds:.0f}s)"
                for agent, seconds in sorted(self._stalled_agents.items())
            )
            lines.append(f"Stalled: {stalled}")
        else:
            lines.append("Stalled: none")

        self.update("\n".join(lines))


def _format_elapsed(started_at: datetime | None) -> str:
    if started_at is None:
        return "--"
    return _format_duration(datetime.now() - started_at)


def _format_duration(duration: timedelta) -> str:
    total_seconds = int(duration.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def _agent_status(snapshot: AgentMonitorSnapshot | None) -> tuple[str, str]:
    if snapshot is None:
        return "◯", "waiting"

    if snapshot.state == AgentState.TOOL_ACTIVE and snapshot.active_tools:
        active_tool = snapshot.active_tools[-1]
        return "●", f"tool:{active_tool.name}"

    match snapshot.state:
        case AgentState.STARTING:
            return "◯", "starting"
        case AgentState.THINKING:
            return "●", "thinking"
        case AgentState.WRITING:
            return "●", "writing"
        case AgentState.IDLE:
            return "◯", "idle"
        case AgentState.STALLED:
            return "!", "stalled"
        case AgentState.RATE_LIMITED:
            return "!", "rate-limited"
        case AgentState.COMPLETED:
            return "✔", "completed"
        case AgentState.FAILED:
            return "✖", "failed"
        case AgentState.TOOL_ACTIVE:
            return "●", "tool"
