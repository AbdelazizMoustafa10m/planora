from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal  # noqa: TC003 — required by Pydantic runtime validation
from enum import StrEnum
from pathlib import Path  # noqa: TC003 — required by Pydantic runtime validation
from typing import Literal, Protocol

from pydantic import BaseModel, Field


class StreamEventType(StrEnum):
    """Rich event taxonomy covering the full agent lifecycle."""

    INIT = "init"
    STATE_CHANGE = "state"
    TOOL_START = "tool_start"
    TOOL_EXEC = "tool_exec"
    TOOL_DONE = "tool_done"
    TEXT = "text"
    SUBAGENT = "subagent"
    RESULT = "result"
    RETRY = "retry"
    RATE_LIMIT = "rate_limit"
    STALL = "stall"


class StreamEvent(BaseModel):
    """Normalized event from any agent stream."""

    event_type: StreamEventType
    timestamp: datetime = Field(default_factory=datetime.now)

    # Tool lifecycle fields (TOOL_START, TOOL_EXEC, TOOL_DONE)
    tool_id: str | None = None
    tool_name: str | None = None
    tool_detail: str | None = None
    tool_status: Literal["running", "done", "error"] | None = None
    tool_duration_ms: int | None = None

    # Text fields
    text_preview: str | None = None

    # Result fields (RESULT)
    cost_usd: Decimal | None = None
    duration_ms: int | None = None
    num_turns: int | None = None
    session_id: str | None = None
    token_usage: dict[str, int] | None = None

    # Retry/rate limit/stall fields
    retry_attempt: int | None = None
    retry_max: int | None = None
    retry_delay_ms: int | None = None
    error_category: str | None = None
    idle_seconds: float | None = None

    # Raw event for debugging
    raw: dict[str, object] | None = None


class AgentState(StrEnum):
    """Agent state machine."""

    STARTING = "starting"
    THINKING = "thinking"
    WRITING = "writing"
    TOOL_ACTIVE = "tool_active"
    IDLE = "idle"
    STALLED = "stalled"
    RATE_LIMITED = "rate_limited"
    COMPLETED = "completed"
    FAILED = "failed"


class ToolExecution(BaseModel):
    """Tracks a single tool invocation through its lifecycle."""

    tool_id: str
    name: str
    friendly_name: str
    detail: str | None = None
    started_at: datetime
    completed_at: datetime | None = None
    status: Literal["running", "done", "error"] = "running"
    duration: timedelta | None = None


class ToolCounters(BaseModel):
    """Aggregated tool execution statistics."""

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    running: int = 0


class AgentMonitorSnapshot(BaseModel):
    """Point-in-time state of a monitored agent."""

    agent_name: str
    state: AgentState
    elapsed: timedelta
    counters: ToolCounters
    active_tools: list[ToolExecution]
    recent_tools: list[ToolExecution]
    last_tool: str | None = None
    last_tool_detail: str | None = None
    cost_usd: Decimal | None = None
    token_usage: dict[str, int] | None = None
    text_count: int = 0
    subagent_count: int = 0
    session_id: str | None = None
    num_turns: int | None = None
    idle_seconds: float = 0.0


class AgentResult(BaseModel):
    """Return type of AgentRunner.run(). Captures all outputs from a single agent execution."""

    agent_name: str
    output_path: Path
    stream_path: Path
    log_path: Path
    exit_code: int
    duration: timedelta
    cost_usd: Decimal | None = None
    token_usage: dict[str, int] | None = None
    num_turns: int | None = None
    session_id: str | None = None
    output_empty: bool = False
    error: str | None = None


class PhaseStatus(StrEnum):
    """Status of a workflow phase."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class PhaseResult:
    """Result of a single workflow phase."""

    name: str
    status: PhaseStatus
    duration: timedelta | None = None
    output_files: list[Path] = field(default_factory=list)
    agent_results: list[AgentResult] = field(default_factory=list)
    error: str | None = None
    cost_usd: Decimal | None = None


@dataclass
class PlanResult:
    """Return type of PlanWorkflow.run(). Aggregates all phase results."""

    phases: list[PhaseResult]
    final_plan_path: Path | None
    report_path: Path | None
    archive_path: Path | None
    total_duration: timedelta
    total_cost_usd: Decimal | None
    agent_results: dict[str, list[AgentResult]]  # Per-agent results keyed by agent name
    success: bool

    @property
    def audit_success_rate(self) -> tuple[int, int]:
        """Returns (succeeded, total) audit count across all rounds."""
        succeeded = total = 0
        for phase in self.phases:
            if phase.name.startswith("audit"):
                for ar in phase.agent_results:
                    total += 1
                    if not ar.output_empty and ar.exit_code == 0:
                        succeeded += 1
        return succeeded, total


class UICallback(Protocol):
    """Contract between workflow engine and UI."""

    def on_phase_start(self, phase: str, label: str) -> None: ...

    def on_phase_end(self, phase: str, result: PhaseResult) -> None: ...

    def on_agent_start(self, agent: str, phase: str) -> None: ...

    def on_agent_end(self, agent: str, result: AgentResult) -> None: ...

    def on_agent_state_change(self, agent: str, state: AgentState) -> None: ...

    def on_tool_start(self, agent: str, tool: ToolExecution) -> None: ...

    def on_tool_done(self, agent: str, tool: ToolExecution) -> None: ...

    def on_cost_update(self, agent: str, cost_usd: Decimal) -> None: ...

    def on_stall(self, agent: str, idle_seconds: float) -> None: ...

    def on_rate_limit(self, agent: str, retry_after_ms: int | None) -> None: ...

    def on_retry(self, agent: str, attempt: int, max_retries: int, error: str) -> None: ...

    def on_snapshot(self, snapshot: AgentMonitorSnapshot) -> None: ...

    def on_log(self, level: str, message: str) -> None: ...

    def on_pipeline_update(self, statuses: dict[str, PhaseStatus]) -> None: ...

    def dispatch_agent_event(self, agent: str, event: StreamEvent) -> None:
        """Route a raw StreamEvent to the appropriate typed callback."""
        match event.event_type:
            case StreamEventType.TOOL_START:
                tool = ToolExecution(
                    tool_id=event.tool_id or "",
                    name=event.tool_name or "unknown",
                    friendly_name=event.tool_name or "unknown",
                    detail=event.tool_detail,
                    started_at=event.timestamp,
                )
                self.on_tool_start(agent, tool)
            case StreamEventType.TOOL_DONE:
                duration = (
                    timedelta(milliseconds=event.tool_duration_ms)
                    if event.tool_duration_ms
                    else None
                )
                tool = ToolExecution(
                    tool_id=event.tool_id or "",
                    name=event.tool_name or "unknown",
                    friendly_name=event.tool_name or "unknown",
                    detail=event.tool_detail,
                    started_at=event.timestamp,
                    completed_at=event.timestamp,
                    status=event.tool_status or "done",
                    duration=duration,
                )
                self.on_tool_done(agent, tool)
            case StreamEventType.STATE_CHANGE:
                state = (
                    AgentState(event.text_preview) if event.text_preview else AgentState.THINKING
                )
                self.on_agent_state_change(agent, state)
            case StreamEventType.RESULT:
                if event.cost_usd is not None:
                    self.on_cost_update(agent, event.cost_usd)
            case StreamEventType.STALL:
                self.on_stall(agent, event.idle_seconds or 0.0)
            case StreamEventType.RATE_LIMIT:
                self.on_rate_limit(agent, event.retry_delay_ms)
            case StreamEventType.RETRY:
                self.on_retry(
                    agent,
                    event.retry_attempt or 0,
                    event.retry_max or 0,
                    event.error_category or "",
                )
            case _:
                pass
