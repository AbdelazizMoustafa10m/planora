from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from decimal import Decimal

import pytest

from planora.agents.registry import AgentConfig, AgentMode, OutputExtraction, StreamFormat
from planora.core.events import (
    AgentMonitorSnapshot,
    AgentResult,
    AgentState,
    PhaseResult,
    PhaseStatus,
    StreamEvent,
    StreamEventType,
    ToolExecution,
)
from planora.workflow.engine import PhaseRunner, _failed_phase


def _agent_result(
    agent_name: str,
    output_path,
    *,
    exit_code: int = 0,
    output_empty: bool = False,
    cost_usd: Decimal | None = None,
    error: str | None = None,
) -> AgentResult:
    return AgentResult(
        agent_name=agent_name,
        output_path=output_path,
        stream_path=output_path.with_suffix(".stream"),
        log_path=output_path.with_suffix(".log"),
        exit_code=exit_code,
        duration=timedelta(seconds=1),
        cost_usd=cost_usd,
        output_empty=output_empty,
        error=error,
    )


@dataclass
class _RecordingUI:
    phase_starts: list[tuple[str, str]] = field(default_factory=list)
    phase_ends: list[tuple[str, PhaseResult]] = field(default_factory=list)
    agent_starts: list[tuple[str, str]] = field(default_factory=list)
    agent_ends: list[tuple[str, AgentResult]] = field(default_factory=list)
    dispatched: list[tuple[str, StreamEventType]] = field(default_factory=list)

    def on_phase_start(self, phase: str, label: str) -> None:
        self.phase_starts.append((phase, label))

    def on_phase_end(self, phase: str, result: PhaseResult) -> None:
        self.phase_ends.append((phase, result))

    def on_agent_start(self, agent: str, phase: str) -> None:
        self.agent_starts.append((agent, phase))

    def on_agent_end(self, agent: str, result: AgentResult) -> None:
        self.agent_ends.append((agent, result))

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
        del level, message

    def on_pipeline_update(self, statuses: dict[str, PhaseStatus]) -> None:
        del statuses

    def dispatch_agent_event(self, agent: str, event: StreamEvent) -> None:
        self.dispatched.append((agent, event.event_type))


class _StubRunner:
    def __init__(self, outcomes: dict[str, AgentResult | BaseException]) -> None:
        self._outcomes = outcomes
        self.calls: list[tuple[str, AgentMode, bool]] = []

    async def run(
        self,
        *,
        agent: AgentConfig,
        prompt: str,
        output_path,
        mode: AgentMode,
        dry_run: bool,
        on_event,
    ) -> AgentResult:
        del prompt, output_path
        self.calls.append((agent.name, mode, dry_run))
        on_event(StreamEvent(event_type=StreamEventType.TEXT, text_preview="streamed"))
        outcome = self._outcomes[agent.name]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _make_agent(name: str) -> AgentConfig:
    return AgentConfig(
        name=name,
        binary="echo",
        model="test-model",
        subcommand="-p",
        flags={AgentMode.PLAN: ["--verbose"], AgentMode.FIX: []},
        stream_format=StreamFormat.CLAUDE,
        output_extraction=OutputExtraction(
            strategy=OutputExtraction.Strategy.STDOUT_CAPTURE,
        ),
    )


@pytest.mark.asyncio
async def test_run_phase_reports_done_and_dispatches_events(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(PhaseRunner, "_install_signal_handlers", lambda self: None)

    ui = _RecordingUI()
    agent = _make_agent("claude")
    result = _agent_result(
        agent_name=agent.name,
        output_path=tmp_path / "initial-plan.md",
        cost_usd=Decimal("1.50"),
    )
    runner = _StubRunner({agent.name: result})

    phase = await PhaseRunner(runner, ui).run_phase(
        "plan",
        agent,
        "prompt text",
        result.output_path,
    )

    assert phase.status == PhaseStatus.DONE
    assert phase.output_files == [result.output_path]
    assert phase.cost_usd == Decimal("1.50")
    assert ui.phase_starts == [("plan", "plan")]
    assert ui.agent_starts == [("claude", "plan")]
    assert ui.agent_ends[0][0] == "claude"
    assert ui.dispatched == [("claude", StreamEventType.TEXT)]
    assert runner.calls == [("claude", AgentMode.PLAN, False)]


@pytest.mark.asyncio
async def test_run_phase_fails_when_output_is_empty(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(PhaseRunner, "_install_signal_handlers", lambda self: None)

    ui = _RecordingUI()
    agent = _make_agent("claude")
    result = _agent_result(
        agent_name=agent.name,
        output_path=tmp_path / "initial-plan.md",
        output_empty=True,
    )

    phase = await PhaseRunner(_StubRunner({agent.name: result}), ui).run_phase(
        "plan",
        agent,
        "prompt text",
        result.output_path,
    )

    assert phase.status == PhaseStatus.FAILED
    assert phase.output_files == []


@pytest.mark.asyncio
async def test_run_parallel_aggregates_costs_and_normalizes_exceptions(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(PhaseRunner, "_install_signal_handlers", lambda self: None)

    ui = _RecordingUI()
    first = _make_agent("gemini")
    second = _make_agent("codex")
    runner = _StubRunner(
        {
            "gemini": _agent_result(
                agent_name="gemini",
                output_path=tmp_path / "audit-gemini.md",
                cost_usd=Decimal("0.40"),
            ),
            "codex": RuntimeError("subprocess crashed"),
        }
    )

    phase = await PhaseRunner(runner, ui, max_concurrency=2).run_parallel(
        "audit",
        [
            (first, "prompt 1", tmp_path / "audit-gemini.md"),
            (second, "prompt 2", tmp_path / "audit-codex.md"),
        ],
    )

    assert phase.status == PhaseStatus.FAILED
    assert phase.cost_usd == Decimal("0.40")
    assert phase.output_files == [tmp_path / "audit-gemini.md"]
    assert "subprocess crashed" in (phase.error or "")
    assert [result.agent_name for result in phase.agent_results] == ["gemini", "codex"]
    assert phase.agent_results[1].exit_code == 1
    assert ui.agent_starts == [("gemini", "gemini"), ("codex", "codex")]
    assert [agent for agent, _result in ui.agent_ends] == ["gemini", "codex"]


def test_failed_phase_helper_returns_failed_status() -> None:
    phase = _failed_phase("audit", "shutdown")

    assert phase.name == "audit"
    assert phase.status == PhaseStatus.FAILED
    assert phase.error == "shutdown"
