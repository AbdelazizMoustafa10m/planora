from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from decimal import Decimal

from rich.console import Console
from rich.panel import Panel

from planora.core.events import (
    AgentMonitorSnapshot,
    AgentResult,
    AgentState,
    PhaseResult,
    PhaseStatus,
    StreamEvent,
    ToolExecution,
    UICallback,
)


class CLICallback(UICallback):
    """UICallback implementation for CLI mode using Rich.

    All output is written to stderr to keep stdout clean for pipeline use.
    """

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console(stderr=True)
        self._agent_costs: dict[str, Decimal] = {}

    def on_phase_start(self, phase: str, label: str) -> None:
        """Rich panel header with phase name."""
        self._console.print(
            Panel(f"[bold]{label}[/bold]", title=f"Phase: {phase}", border_style="blue")
        )

    def on_phase_end(self, phase: str, result: PhaseResult) -> None:
        """Summary line: status, duration, cost."""
        duration_str = str(result.duration) if result.duration else "N/A"
        cost_str = f"${result.cost_usd}" if result.cost_usd else "N/A"
        style = "green" if result.status == PhaseStatus.DONE else "red"
        self._console.print(
            f"  [{style}]{phase}[/{style}] {result.status.value}"
            f" | Duration: {duration_str} | Cost: {cost_str}"
        )

    def on_agent_start(self, agent: str, phase: str) -> None:
        """Print agent start message."""
        self._console.print(f"  [cyan][{agent}][/cyan] Starting in {phase}...")

    def on_agent_end(self, agent: str, result: AgentResult) -> None:
        """Print agent completion summary with duration, cost, and exit code."""
        duration_str = str(result.duration)
        cost_str = f"${result.cost_usd}" if result.cost_usd else "N/A"
        style = "green" if result.exit_code == 0 else "red"
        self._console.print(
            f"  [{style}][{agent}][/{style}] Completed"
            f" \u2014 {duration_str}, cost: {cost_str}, exit: {result.exit_code}"
        )

    def on_agent_state_change(self, agent: str, state: AgentState) -> None:
        """Print agent state transition."""
        self._console.print(f"  [dim][{agent}] State: {state.value}[/dim]")

    def on_tool_start(self, agent: str, tool: ToolExecution) -> None:
        """Print tool invocation start."""
        detail = f" {tool.detail}" if tool.detail else ""
        self._console.print(f"  [cyan][{agent}][/cyan] \u27f3 {tool.name}{detail}")

    def on_tool_done(self, agent: str, tool: ToolExecution) -> None:
        """Print tool completion with duration and status symbol."""
        detail = f" {tool.detail}" if tool.detail else ""
        duration_s = f" ({tool.duration.total_seconds():.1f}s)" if tool.duration else ""
        symbol = "\u2714" if tool.status == "done" else "\u2716"
        style = "green" if tool.status == "done" else "red"
        self._console.print(
            f"  [{style}][{agent}][/{style}] {symbol} {tool.name}{detail}{duration_s}"
        )

    def on_cost_update(self, agent: str, cost_usd: Decimal) -> None:
        """Print per-agent cost update and running total across all agents."""
        self._agent_costs[agent] = cost_usd
        total = sum(self._agent_costs.values())
        self._console.print(f"  [yellow][{agent}][/yellow] Cost: ${cost_usd} (total: ${total})")

    def on_stall(self, agent: str, idle_seconds: float) -> None:
        """Print stall warning with idle duration."""
        self._console.print(
            f"  [bold yellow][{agent}] \u26a0 Stalled"
            f" \u2014 no activity for {idle_seconds:.0f}s[/bold yellow]"
        )

    def on_rate_limit(self, agent: str, retry_after_ms: int | None) -> None:
        """Print rate limit warning with optional retry delay."""
        after = f" (retry after {retry_after_ms}ms)" if retry_after_ms else ""
        self._console.print(f"  [bold yellow][{agent}] \u26a0 Rate limited{after}[/bold yellow]")

    def on_retry(self, agent: str, attempt: int, max_retries: int, error: str) -> None:
        """Print retry attempt with error context."""
        self._console.print(f"  [yellow][{agent}] Retry {attempt}/{max_retries}: {error}[/yellow]")

    def on_snapshot(self, snapshot: AgentMonitorSnapshot) -> None:
        """Status line: [agent] MM:SS | done/running/failed counters | cost | active tool."""
        elapsed = snapshot.elapsed
        minutes = int(elapsed.total_seconds()) // 60
        seconds = int(elapsed.total_seconds()) % 60
        cost_str = f"${snapshot.cost_usd}" if snapshot.cost_usd else "$0.00"

        active_str = ""
        if snapshot.active_tools:
            tool = snapshot.active_tools[0]
            detail = f" {tool.detail}" if tool.detail else ""
            tool_elapsed = datetime.now(UTC) - tool.started_at
            dur = f" ({int(tool_elapsed.total_seconds())}s)"
            active_str = f" | \u27f3 {tool.name}{detail}{dur}"

        self._console.print(
            f"  [dim][{snapshot.agent_name}] \u23f1 {minutes:02d}:{seconds:02d}"
            f" | \u2714{snapshot.counters.succeeded}"
            f" \u27f3{snapshot.counters.running}"
            f" \u2716{snapshot.counters.failed}"
            f" | \U0001f4b0{cost_str}{active_str}[/dim]"
        )

    def on_log(self, level: str, message: str) -> None:
        """Print timestamped log line with level-specific styling."""
        ts = datetime.now(UTC).strftime("%H:%M:%S")
        style_map: dict[str, str] = {
            "debug": "dim",
            "info": "blue",
            "warning": "yellow",
            "error": "red bold",
            "success": "green",
        }
        style = style_map.get(level, "")
        self._console.print(f"  [{style}]{ts} [{level.upper()}] {message}[/{style}]")

    def on_pipeline_update(self, statuses: dict[str, PhaseStatus]) -> None:
        """Print pipeline phase status summary."""
        parts = [f"{phase}: {status.value}" for phase, status in statuses.items()]
        self._console.print(f"  [dim]Pipeline: {' | '.join(parts)}[/dim]")


class EventsOutputCallback(UICallback):
    """UICallback implementation for headless/CI mode.

    Every method serializes its data as a JSONL line to stderr.
    Format: {"ts": "<ISO8601>", "agent": "<name>", "event": "<type>", ...fields}
    """

    @staticmethod
    def _emit(record: dict[str, Any]) -> None:
        print(json.dumps(record), file=sys.stderr)

    @staticmethod
    def _ts() -> str:
        return datetime.now(UTC).isoformat()

    def on_phase_start(self, phase: str, label: str) -> None:
        self._emit({"ts": self._ts(), "event": "phase_start", "phase": phase, "label": label})

    def on_phase_end(self, phase: str, result: PhaseResult) -> None:
        self._emit(
            {
                "ts": self._ts(),
                "event": "phase_end",
                "phase": phase,
                "status": result.status.value,
                "duration_s": result.duration.total_seconds() if result.duration else None,
                "cost_usd": float(result.cost_usd) if result.cost_usd else None,
                "error": result.error,
            }
        )

    def on_agent_start(self, agent: str, phase: str) -> None:
        self._emit({"ts": self._ts(), "agent": agent, "event": "agent_start", "phase": phase})

    def on_agent_end(self, agent: str, result: AgentResult) -> None:
        self._emit(
            {
                "ts": self._ts(),
                "agent": agent,
                "event": "agent_end",
                "exit_code": result.exit_code,
                "duration_s": result.duration.total_seconds(),
                "cost_usd": float(result.cost_usd) if result.cost_usd else None,
                "num_turns": result.num_turns,
                "error": result.error,
            }
        )

    def on_agent_state_change(self, agent: str, state: AgentState) -> None:
        self._emit(
            {"ts": self._ts(), "agent": agent, "event": "state_change", "state": state.value}
        )

    def on_tool_start(self, agent: str, tool: ToolExecution) -> None:
        self._emit(
            {
                "ts": self._ts(),
                "agent": agent,
                "event": "tool_start",
                "tool": tool.name,
                "tool_id": tool.tool_id,
                "detail": tool.detail,
            }
        )

    def on_tool_done(self, agent: str, tool: ToolExecution) -> None:
        self._emit(
            {
                "ts": self._ts(),
                "agent": agent,
                "event": "tool_done",
                "tool": tool.name,
                "tool_id": tool.tool_id,
                "detail": tool.detail,
                "duration_ms": int(tool.duration.total_seconds() * 1000) if tool.duration else None,
                "status": tool.status,
            }
        )

    def on_cost_update(self, agent: str, cost_usd: Decimal) -> None:
        self._emit(
            {
                "ts": self._ts(),
                "agent": agent,
                "event": "cost_update",
                "cost_usd": float(cost_usd),
            }
        )

    def on_stall(self, agent: str, idle_seconds: float) -> None:
        self._emit(
            {
                "ts": self._ts(),
                "agent": agent,
                "event": "stall",
                "idle_seconds": idle_seconds,
            }
        )

    def on_rate_limit(self, agent: str, retry_after_ms: int | None) -> None:
        self._emit(
            {
                "ts": self._ts(),
                "agent": agent,
                "event": "rate_limit",
                "retry_after_ms": retry_after_ms,
            }
        )

    def on_retry(self, agent: str, attempt: int, max_retries: int, error: str) -> None:
        self._emit(
            {
                "ts": self._ts(),
                "agent": agent,
                "event": "retry",
                "attempt": attempt,
                "max_retries": max_retries,
                "error": error,
            }
        )

    def on_snapshot(self, snapshot: AgentMonitorSnapshot) -> None:
        self._emit(
            {
                "ts": self._ts(),
                "agent": snapshot.agent_name,
                "event": "snapshot",
                "state": snapshot.state.value,
                "elapsed_s": snapshot.elapsed.total_seconds(),
                "cost_usd": float(snapshot.cost_usd) if snapshot.cost_usd else None,
                "counters": {
                    "total": snapshot.counters.total,
                    "succeeded": snapshot.counters.succeeded,
                    "failed": snapshot.counters.failed,
                    "running": snapshot.counters.running,
                },
                "active_tools": [t.name for t in snapshot.active_tools],
                "num_turns": snapshot.num_turns,
                "idle_seconds": snapshot.idle_seconds,
            }
        )

    def on_log(self, level: str, message: str) -> None:
        self._emit({"ts": self._ts(), "event": "log", "level": level, "message": message})

    def on_pipeline_update(self, statuses: dict[str, PhaseStatus]) -> None:
        self._emit(
            {
                "ts": self._ts(),
                "event": "pipeline_update",
                "statuses": {phase: status.value for phase, status in statuses.items()},
            }
        )

    def dispatch_agent_event(self, agent: str, event: StreamEvent) -> None:
        """Serialize the StreamEvent directly as a JSONL line."""
        record: dict[str, Any] = {
            "ts": self._ts(),
            "agent": agent,
            "event": event.event_type.value,
        }
        if event.tool_name is not None:
            record["tool"] = event.tool_name
        if event.tool_id is not None:
            record["tool_id"] = event.tool_id
        if event.tool_detail is not None:
            record["detail"] = event.tool_detail
        if event.tool_status is not None:
            record["status"] = event.tool_status
        if event.tool_duration_ms is not None:
            record["duration_ms"] = event.tool_duration_ms
        if event.cost_usd is not None:
            record["cost_usd"] = float(event.cost_usd)
        if event.retry_attempt is not None:
            record["attempt"] = event.retry_attempt
        if event.retry_max is not None:
            record["max_retries"] = event.retry_max
        if event.retry_delay_ms is not None:
            record["retry_after_ms"] = event.retry_delay_ms
        if event.error_category is not None:
            record["error_category"] = event.error_category
        if event.text_preview is not None:
            record["text_preview"] = event.text_preview
        if event.num_turns is not None:
            record["num_turns"] = event.num_turns
        if event.session_id is not None:
            record["session_id"] = event.session_id
        self._emit(record)
