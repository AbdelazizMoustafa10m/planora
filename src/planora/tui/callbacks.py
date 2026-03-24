"""Textual UI callback bridge for worker-safe dashboard updates."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from textual.message import Message

from planora.agents.monitor import AgentMonitor
from planora.core.events import (
    AgentMonitorSnapshot,
    AgentResult,
    AgentState,
    PhaseResult,
    PhaseStatus,
    StreamEvent,
    StreamEventType,
    ToolExecution,
    UICallback,
)

if TYPE_CHECKING:
    from decimal import Decimal

    from planora.tui.app import PlanoraTUI


class PhaseStarted(Message, bubble=False, namespace="planora"):
    """Phase lifecycle message posted to the Textual app."""

    def __init__(self, phase: str, label: str) -> None:
        self.phase = phase
        self.label = label
        super().__init__()


class PhaseFinished(Message, bubble=False, namespace="planora"):
    """Completed phase result posted to the Textual app."""

    def __init__(self, phase: str, result: PhaseResult) -> None:
        self.phase = phase
        self.result = result
        super().__init__()


class AgentStarted(Message, bubble=False, namespace="planora"):
    """Agent start event posted to the Textual app."""

    def __init__(self, agent: str, phase: str, run_id: int) -> None:
        self.agent = agent
        self.phase = phase
        self.run_id = run_id
        super().__init__()


class AgentFinished(Message, bubble=False, namespace="planora"):
    """Agent completion event posted to the Textual app."""

    def __init__(self, agent: str, result: AgentResult, run_id: int) -> None:
        self.agent = agent
        self.result = result
        self.run_id = run_id
        super().__init__()


class AgentSnapshotUpdated(Message, bubble=False, namespace="planora"):
    """Latest point-in-time agent snapshot."""

    def __init__(self, snapshot: AgentMonitorSnapshot, run_id: int) -> None:
        self.snapshot = snapshot
        self.run_id = run_id
        super().__init__()


class ToolStarted(Message, bubble=False, namespace="planora"):
    """Tool start event for dashboard counters and event log."""

    def __init__(self, agent: str, tool: ToolExecution, run_id: int) -> None:
        self.agent = agent
        self.tool = tool
        self.run_id = run_id
        super().__init__()


class ToolFinished(Message, bubble=False, namespace="planora"):
    """Tool completion event for dashboard counters and event log."""

    def __init__(self, agent: str, tool: ToolExecution, run_id: int) -> None:
        self.agent = agent
        self.tool = tool
        self.run_id = run_id
        super().__init__()


class CostUpdated(Message, bubble=False, namespace="planora"):
    """Running cost update for the current agent run."""

    def __init__(self, agent: str, cost_usd: Decimal, run_id: int) -> None:
        self.agent = agent
        self.cost_usd = cost_usd
        self.run_id = run_id
        super().__init__()


class AgentOutputChunk(Message, bubble=False, namespace="planora"):
    """Streaming text preview chunk for the agent output widget."""

    def __init__(self, agent: str, text: str, run_id: int) -> None:
        self.agent = agent
        self.text = text
        self.run_id = run_id
        super().__init__()


class LogEvent(Message, bubble=False, namespace="planora"):
    """Structured log line destined for the event log widget."""

    def __init__(
        self,
        *,
        level: str,
        message: str,
        timestamp: datetime | None = None,
        agent: str | None = None,
        notify: bool = False,
    ) -> None:
        self.level = level
        self.message = message
        self.timestamp = timestamp or datetime.now()
        self.agent = agent
        self.notify = notify
        super().__init__()


class PipelineUpdated(Message, bubble=False, namespace="planora"):
    """Pipeline status refresh message."""

    def __init__(self, statuses: dict[str, PhaseStatus]) -> None:
        self.statuses = statuses
        super().__init__()


class TextualUICallback(UICallback):
    """UICallback implementation that posts thread-safe Textual messages."""

    def __init__(self, app: PlanoraTUI) -> None:
        self._app = app
        self._monitors: dict[str, AgentMonitor] = {}
        self._snapshots: dict[str, AgentMonitorSnapshot] = {}
        self._run_ids: dict[str, int] = {}
        self._snapshot_recorded_at: dict[str, datetime] = {}

    def on_phase_start(self, phase: str, label: str) -> None:
        self._post(PhaseStarted(phase, label))
        self._post(LogEvent(level="info", message=f"Phase started: {label}"))

    def on_phase_end(self, phase: str, result: PhaseResult) -> None:
        level = "error" if result.status == PhaseStatus.FAILED else "info"
        self._post(PhaseFinished(phase, result))
        self._post(
            LogEvent(
                level=level,
                message=f"Phase {phase} finished with status: {result.status.value}",
                notify=result.status == PhaseStatus.FAILED,
            )
        )

    def on_agent_start(self, agent: str, phase: str) -> None:
        run_id = self._run_ids.get(agent, 0) + 1
        self._run_ids[agent] = run_id
        self._monitors[agent] = AgentMonitor(agent)

        self._post(AgentStarted(agent, phase, run_id))
        self._emit_snapshot(agent)
        self._post(
            LogEvent(
                level="info",
                agent=agent,
                message=f"Agent started in phase {phase}",
            )
        )

    def on_agent_end(self, agent: str, result: AgentResult) -> None:
        run_id = self._run_ids.get(agent, 1)
        existing = self._snapshots.get(agent)

        if existing is None:
            snapshot = AgentMonitorSnapshot(
                agent_name=agent,
                state=AgentState.COMPLETED if result.exit_code == 0 else AgentState.FAILED,
                elapsed=result.duration,
                counters=_empty_counters(),
                active_tools=[],
                recent_tools=[],
                cost_usd=result.cost_usd,
                num_turns=result.num_turns,
                session_id=result.session_id,
            )
        else:
            snapshot = existing.model_copy(
                update={
                    "state": AgentState.COMPLETED if result.exit_code == 0 else AgentState.FAILED,
                    "elapsed": result.duration,
                    "active_tools": [],
                    "cost_usd": result.cost_usd
                    if result.cost_usd is not None
                    else existing.cost_usd,
                    "num_turns": result.num_turns
                    if result.num_turns is not None
                    else existing.num_turns,
                    "session_id": result.session_id
                    if result.session_id is not None
                    else existing.session_id,
                }
            )

        self._record_snapshot(snapshot)
        self._post(AgentFinished(agent, result, run_id))
        self._post(
            LogEvent(
                level="error" if result.exit_code != 0 else "info",
                agent=agent,
                message=(
                    f"Agent finished with exit code {result.exit_code}"
                    if result.exit_code != 0
                    else "Agent finished successfully"
                ),
                notify=result.exit_code != 0,
            )
        )

    def on_agent_state_change(self, agent: str, state: AgentState) -> None:
        existing = self._snapshots.get(agent)
        if existing is None:
            snapshot = AgentMonitorSnapshot(
                agent_name=agent,
                state=state,
                elapsed=timedelta(0),
                counters=_empty_counters(),
                active_tools=[],
                recent_tools=[],
            )
        else:
            snapshot = existing.model_copy(update={"state": state})
        self._record_snapshot(snapshot)

    def on_tool_start(self, agent: str, tool: ToolExecution) -> None:
        event = StreamEvent(
            event_type=StreamEventType.TOOL_START,
            timestamp=tool.started_at,
            tool_id=tool.tool_id,
            tool_name=tool.name,
            tool_detail=tool.detail,
            tool_status="running",
        )
        self._update_monitor(agent, event)

        snapshot = self._snapshots[agent]
        resolved_tool = _find_active_tool(snapshot, tool.tool_id) or tool
        self._post(ToolStarted(agent, resolved_tool, self._current_run_id(agent)))

    def on_tool_done(self, agent: str, tool: ToolExecution) -> None:
        event = StreamEvent(
            event_type=StreamEventType.TOOL_DONE,
            timestamp=tool.completed_at or datetime.now(),
            tool_id=tool.tool_id,
            tool_name=tool.name,
            tool_detail=tool.detail,
            tool_status=tool.status,
            tool_duration_ms=(
                int(tool.duration.total_seconds() * 1000) if tool.duration is not None else None
            ),
        )
        self._update_monitor(agent, event)

        snapshot = self._snapshots[agent]
        resolved_tool = _find_recent_tool(snapshot, tool.tool_id) or tool
        self._post(ToolFinished(agent, resolved_tool, self._current_run_id(agent)))

    def on_cost_update(self, agent: str, cost_usd: Decimal) -> None:
        self._post(CostUpdated(agent, cost_usd, self._current_run_id(agent)))

    def on_stall(self, agent: str, idle_seconds: float) -> None:
        idle = idle_seconds
        if idle <= 0:
            idle = self._current_idle_seconds(agent)
        self._post(
            LogEvent(
                level="warning",
                agent=agent,
                message=f"Agent stalled after {idle:.0f}s of inactivity",
                notify=True,
            )
        )

    def on_rate_limit(self, agent: str, retry_after_ms: int | None) -> None:
        retry_after = (
            f" Retry after {retry_after_ms / 1000:.1f}s." if retry_after_ms is not None else ""
        )
        self._post(
            LogEvent(
                level="warning",
                agent=agent,
                message=f"Rate limited.{retry_after}".strip(),
                notify=True,
            )
        )

    def on_retry(self, agent: str, attempt: int, max_retries: int, error: str) -> None:
        self._post(
            LogEvent(
                level="warning",
                agent=agent,
                message=f"Retry {attempt}/{max_retries}: {error}",
                notify=True,
            )
        )

    def on_snapshot(self, snapshot: AgentMonitorSnapshot) -> None:
        self._record_snapshot(snapshot)

    def on_log(self, level: str, message: str) -> None:
        self._post(
            LogEvent(
                level=level,
                message=message,
                notify=level.lower() in {"warning", "error"},
            )
        )

    def on_pipeline_update(self, statuses: dict[str, PhaseStatus]) -> None:
        self._post(PipelineUpdated(statuses))

    def dispatch_agent_event(self, agent: str, event: StreamEvent) -> None:
        snapshot = self._update_monitor(agent, event)

        if event.event_type == StreamEventType.TEXT and event.text_preview:
            self._post(AgentOutputChunk(agent, event.text_preview, self._current_run_id(agent)))

        match event.event_type:
            case StreamEventType.TOOL_START:
                tool = _find_active_tool(snapshot, event.tool_id) or _tool_from_start_event(event)
                self._post(ToolStarted(agent, tool, self._current_run_id(agent)))
            case StreamEventType.TOOL_DONE:
                tool = _find_recent_tool(snapshot, event.tool_id) or _tool_from_done_event(event)
                self._post(ToolFinished(agent, tool, self._current_run_id(agent)))
            case StreamEventType.RESULT:
                if event.cost_usd is not None:
                    self.on_cost_update(agent, event.cost_usd)
            case StreamEventType.STALL:
                self.on_stall(agent, self._current_idle_seconds(agent))
            case StreamEventType.RATE_LIMIT:
                self.on_rate_limit(agent, event.retry_delay_ms)
            case StreamEventType.RETRY:
                self.on_retry(
                    agent,
                    event.retry_attempt or 0,
                    event.retry_max or 0,
                    event.error_category or "retry requested",
                )
            case _:
                pass

    def _post(self, message: Message) -> None:
        self._app.post_message(message)

    def _current_run_id(self, agent: str) -> int:
        run_id = self._run_ids.get(agent)
        if run_id is None:
            run_id = 1
            self._run_ids[agent] = run_id
        return run_id

    def _ensure_monitor(self, agent: str) -> AgentMonitor:
        monitor = self._monitors.get(agent)
        if monitor is None:
            monitor = AgentMonitor(agent)
            self._monitors[agent] = monitor
            self._run_ids[agent] = self._run_ids.get(agent, 0) + 1
        return monitor

    def _emit_snapshot(self, agent: str) -> AgentMonitorSnapshot:
        monitor = self._ensure_monitor(agent)
        snapshot = monitor.snapshot()
        self._record_snapshot(snapshot)
        return snapshot

    def _record_snapshot(self, snapshot: AgentMonitorSnapshot) -> None:
        self._snapshots[snapshot.agent_name] = snapshot
        self._snapshot_recorded_at[snapshot.agent_name] = datetime.now()
        self._post(AgentSnapshotUpdated(snapshot, self._current_run_id(snapshot.agent_name)))

    def _update_monitor(self, agent: str, event: StreamEvent) -> AgentMonitorSnapshot:
        monitor = self._ensure_monitor(agent)
        monitor.update(event)
        return self._emit_snapshot(agent)

    def _current_idle_seconds(self, agent: str) -> float:
        recorded_at = self._snapshot_recorded_at.get(agent)
        snapshot = self._snapshots.get(agent)
        if recorded_at is None or snapshot is None:
            return 0.0
        return snapshot.idle_seconds + (datetime.now() - recorded_at).total_seconds()


def _tool_from_start_event(event: StreamEvent) -> ToolExecution:
    return ToolExecution(
        tool_id=event.tool_id or "unknown-tool",
        name=event.tool_name or "Unknown",
        friendly_name=event.tool_name or "Unknown",
        detail=event.tool_detail,
        started_at=event.timestamp,
    )


def _tool_from_done_event(event: StreamEvent) -> ToolExecution:
    duration = (
        timedelta(milliseconds=event.tool_duration_ms)
        if event.tool_duration_ms is not None
        else None
    )
    return ToolExecution(
        tool_id=event.tool_id or "unknown-tool",
        name=event.tool_name or "Unknown",
        friendly_name=event.tool_name or "Unknown",
        detail=event.tool_detail,
        started_at=event.timestamp - duration if duration is not None else event.timestamp,
        completed_at=event.timestamp,
        status=event.tool_status or "done",
        duration=duration,
    )


def _find_active_tool(
    snapshot: AgentMonitorSnapshot,
    tool_id: str | None,
) -> ToolExecution | None:
    if tool_id is None:
        return None
    for tool in snapshot.active_tools:
        if tool.tool_id == tool_id:
            return tool
    return None


def _find_recent_tool(
    snapshot: AgentMonitorSnapshot,
    tool_id: str | None,
) -> ToolExecution | None:
    if tool_id is None:
        return None
    for tool in reversed(snapshot.recent_tools):
        if tool.tool_id == tool_id:
            return tool
    return None


def _empty_counters() -> object:
    from planora.core.events import ToolCounters

    return ToolCounters()
