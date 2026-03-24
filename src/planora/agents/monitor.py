from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import TYPE_CHECKING

from planora.core.events import (
    AgentMonitorSnapshot,
    AgentState,
    StreamEvent,
    StreamEventType,
    ToolCounters,
    ToolExecution,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from datetime import timedelta
    from decimal import Decimal

_FRIENDLY_NAMES: dict[str, str] = {
    "Read": "Read file",
    "Write": "Write file",
    "Edit": "Edit file",
    "Bash": "Shell command",
    "Agent": "Sub-agent",
    "Glob": "Search files",
    "Grep": "Search content",
    "LS": "List directory",
}


def _friendly_name(tool_name: str) -> str:
    """Map a tool name to a human-readable label."""
    return _FRIENDLY_NAMES.get(tool_name, tool_name)


class AgentMonitor:
    """Event-driven agent monitor with real-time state tracking."""

    def __init__(self, agent_name: str, max_recent_tools: int = 20) -> None:
        self._agent_name = agent_name
        self._state = AgentState.STARTING
        self._start_time = datetime.now()
        self._last_activity = datetime.now()
        self._active_tools: dict[str, ToolExecution] = {}
        self._recent_tools: deque[ToolExecution] = deque(maxlen=max_recent_tools)
        self._counters = ToolCounters()
        self._cost_usd: Decimal | None = None
        self._text_count = 0
        self._subagent_count = 0
        self._session_id: str | None = None
        self._num_turns: int | None = None

    def _handle_init(self, event: StreamEvent) -> None:
        if event.session_id is not None:
            self._session_id = event.session_id

    def _handle_state_change(self, event: StreamEvent) -> None:
        if event.tool_detail == "thinking":
            self._state = AgentState.THINKING
        else:
            self._state = AgentState.WRITING

    def _handle_tool_start(self, event: StreamEvent) -> None:
        tool_id = event.tool_id or f"tool-{self._counters.total}"
        name = event.tool_name or tool_id
        execution = ToolExecution(
            tool_id=tool_id,
            name=name,
            friendly_name=_friendly_name(name),
            detail=event.tool_detail,
            started_at=datetime.now(),
        )
        self._active_tools[tool_id] = execution
        self._counters.running += 1
        self._counters.total += 1
        self._state = AgentState.TOOL_ACTIVE

    def _handle_tool_exec(self, event: StreamEvent) -> None:
        if event.tool_id is not None and event.tool_id in self._active_tools:
            self._active_tools[event.tool_id].detail = event.tool_detail
        self._state = AgentState.TOOL_ACTIVE

    def _handle_tool_done(self, event: StreamEvent) -> None:
        tool_id = event.tool_id
        if tool_id is None or tool_id not in self._active_tools:
            return

        execution = self._active_tools.pop(tool_id)
        now = datetime.now()
        execution.completed_at = now
        execution.duration = now - execution.started_at

        status = event.tool_status or "done"
        execution.status = status

        self._counters.running -= 1
        if status == "error":
            self._counters.failed += 1
        else:
            self._counters.succeeded += 1

        self._recent_tools.append(execution)
        self._state = AgentState.TOOL_ACTIVE if self._active_tools else AgentState.THINKING

    def _handle_text(self, event: StreamEvent) -> None:  # noqa: ARG002
        self._text_count += 1
        self._state = AgentState.WRITING

    def _handle_subagent(self, event: StreamEvent) -> None:  # noqa: ARG002
        self._subagent_count += 1
        self._state = AgentState.TOOL_ACTIVE

    def _handle_result(self, event: StreamEvent) -> None:
        if event.cost_usd is not None:
            self._cost_usd = event.cost_usd
        if event.num_turns is not None:
            self._num_turns = event.num_turns
        if event.session_id is not None:
            self._session_id = event.session_id
        self._state = AgentState.COMPLETED

    def _handle_rate_limit(self, event: StreamEvent) -> None:  # noqa: ARG002
        self._state = AgentState.RATE_LIMITED

    def _handle_stall(self, event: StreamEvent) -> None:  # noqa: ARG002
        self._state = AgentState.STALLED

    def update(self, event: StreamEvent) -> None:
        """Apply a stream event to the state machine."""
        match event.event_type:
            case StreamEventType.INIT:
                self._handle_init(event)
            case StreamEventType.STATE_CHANGE:
                self._handle_state_change(event)
            case StreamEventType.TOOL_START:
                self._handle_tool_start(event)
            case StreamEventType.TOOL_EXEC:
                self._handle_tool_exec(event)
            case StreamEventType.TOOL_DONE:
                self._handle_tool_done(event)
            case StreamEventType.TEXT:
                self._handle_text(event)
            case StreamEventType.SUBAGENT:
                self._handle_subagent(event)
            case StreamEventType.RESULT:
                self._handle_result(event)
            case StreamEventType.RETRY:
                pass  # state unchanged; retry logged upstream
            case StreamEventType.RATE_LIMIT:
                self._handle_rate_limit(event)
            case StreamEventType.STALL:
                self._handle_stall(event)

        self._last_activity = datetime.now()

    def snapshot(self) -> AgentMonitorSnapshot:
        """Return a point-in-time snapshot. Computes elapsed and idle_seconds from now."""
        now = datetime.now()
        elapsed: timedelta = now - self._start_time
        idle_seconds: float = (now - self._last_activity).total_seconds()

        last_tool: str | None = None
        last_tool_detail: str | None = None
        if self._recent_tools:
            last = self._recent_tools[-1]
            last_tool = last.name
            last_tool_detail = last.detail

        return AgentMonitorSnapshot(
            agent_name=self._agent_name,
            state=self._state,
            elapsed=elapsed,
            counters=self._counters.model_copy(),
            active_tools=list(self._active_tools.values()),
            recent_tools=list(self._recent_tools),
            last_tool=last_tool,
            last_tool_detail=last_tool_detail,
            cost_usd=self._cost_usd,
            text_count=self._text_count,
            subagent_count=self._subagent_count,
            session_id=self._session_id,
            num_turns=self._num_turns,
            idle_seconds=idle_seconds,
        )

    async def consume(
        self,
        events: AsyncIterator[StreamEvent],
        on_event: Callable[[StreamEvent], None] | None = None,
    ) -> None:
        """Consume stream events, updating state on each, with optional per-event callback."""
        async for event in events:
            self.update(event)
            if on_event is not None:
                on_event(event)
