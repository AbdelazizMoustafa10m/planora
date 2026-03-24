"""Agent activity widget for the Planora TUI dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from textual.widgets import Static

from planora.core.events import AgentState

if TYPE_CHECKING:
    from planora.core.events import AgentMonitorSnapshot, ToolExecution


@dataclass(slots=True)
class _SnapshotView:
    snapshot: AgentMonitorSnapshot
    recorded_at: datetime


class AgentActivityPanel(Static):
    """Render per-agent tool activity, state, and recent tool history."""

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
        self._snapshots: dict[str, _SnapshotView] = {}

    def on_mount(self) -> None:
        self.set_interval(1.0, self._refresh_display)
        self._refresh_display()

    def set_agents(self, agents: list[str]) -> None:
        """Set the display order for the known agent set."""
        self._agent_order = list(dict.fromkeys(agents))
        self._refresh_display()

    def reset(self) -> None:
        """Clear all agent snapshots for a new run."""
        self._snapshots.clear()
        self._refresh_display()

    def apply_snapshot(self, snapshot: AgentMonitorSnapshot) -> None:
        """Update a single agent snapshot."""
        if snapshot.agent_name not in self._agent_order:
            self._agent_order.append(snapshot.agent_name)
        self._snapshots[snapshot.agent_name] = _SnapshotView(snapshot, datetime.now())
        self._refresh_display()

    def _refresh_display(self) -> None:
        if not self._agent_order:
            self.update("No agents running")
            return

        lines: list[str] = []
        for agent in self._agent_order:
            view = self._snapshots.get(agent)
            if view is None:
                lines.append(f"▸ {agent} [waiting]")
                lines.append("")
                continue

            snapshot = view.snapshot
            lines.append(_render_header(snapshot, view.recorded_at))
            for tool in snapshot.active_tools:
                lines.append(f"  ⟳ {_render_tool(tool, running=True)}")

            recent_tools = list(snapshot.recent_tools)[-5:]
            for tool in recent_tools:
                icon = "✔" if tool.status == "done" else "✖"
                lines.append(f"  {icon} {_render_tool(tool, running=False)}")
            lines.append("")

        self.update("\n".join(lines).rstrip())


def _render_header(snapshot: AgentMonitorSnapshot, recorded_at: datetime) -> str:
    state_label = snapshot.state.value.replace("_", " ")
    elapsed = snapshot.elapsed
    if snapshot.state not in _TERMINAL_STATES:
        elapsed += datetime.now() - recorded_at
    return f"▸ {snapshot.agent_name} [{state_label}] {_format_duration(elapsed)}"


def _render_tool(tool: ToolExecution, *, running: bool) -> str:
    detail = f" {tool.detail}" if tool.detail else ""
    duration = datetime.now() - tool.started_at if running else tool.duration
    suffix = f" ({_format_duration(duration)})" if duration is not None else ""
    return f"{tool.name}{detail}{suffix}"


def _format_duration(duration: timedelta | None) -> str:
    if duration is None:
        return "--"

    total_seconds = int(duration.total_seconds())
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


_TERMINAL_STATES = {
    AgentState.COMPLETED,
    AgentState.FAILED,
}
