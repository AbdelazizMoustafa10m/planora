"""Integration tests for the agent runner → filter → parser → stall → monitor pipeline.

These tests wire the full pipeline using fixture JSONL data and a fake subprocess
instead of the real agent binaries.  All assertions are made against the public
stream-event and AgentResult contracts.
"""
from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import TYPE_CHECKING

import pytest

from planora.agents.registry import AgentConfig, AgentMode, OutputExtraction, StreamFormat
from planora.agents.runner import AgentRunner
from planora.core.events import StreamEvent, StreamEventType

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Fake process helpers (adapted from test_runner.py pattern)
# ---------------------------------------------------------------------------


def _make_stream_reader(lines: list[str]) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    for line in lines:
        reader.feed_data(line.encode("utf-8"))
    reader.feed_eof()
    return reader


class _FakeProcess:
    def __init__(
        self,
        stdout_lines: list[str],
        stderr_lines: list[str],
        *,
        exit_code: int = 0,
    ) -> None:
        self.stdout = _make_stream_reader(stdout_lines)
        self.stderr = _make_stream_reader(stderr_lines)
        self.returncode = exit_code
        self._exit_code = exit_code
        self.terminated = False

    async def wait(self) -> int:
        return self._exit_code

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = self._exit_code

    def kill(self) -> None:
        self.returncode = self._exit_code


def _make_agent(
    name: str,
    stream_format: StreamFormat,
    *,
    stderr_as_stream: bool = False,
) -> AgentConfig:
    return AgentConfig(
        name=name,
        binary=name,
        model=f"{name}-model",
        subcommand="-p",
        flags={AgentMode.PLAN: [], AgentMode.FIX: []},
        stream_format=stream_format,
        output_extraction=OutputExtraction(
            strategy=OutputExtraction.Strategy.STDOUT_CAPTURE,
            stderr_as_stream=stderr_as_stream,
        ),
    )


def _fake_process_factory(
    stdout_lines: list[str], stderr_lines: list[str], *, exit_code: int = 0
):
    """Return a coroutine function that yields a _FakeProcess."""

    async def factory(*_args, **_kwargs) -> _FakeProcess:
        return _FakeProcess(
            stdout_lines=stdout_lines,
            stderr_lines=stderr_lines,
            exit_code=exit_code,
        )

    return factory


# ---------------------------------------------------------------------------
# Claude pipeline — full event sequence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_claude_pipeline_emits_correct_event_sequence(
    monkeypatch: pytest.MonkeyPatch,
    claude_jsonl_lines: list[str],
    tmp_path: Path,
) -> None:
    agent = _make_agent("claude", StreamFormat.CLAUDE)
    # Terminate each line with a newline so readline() returns each separately
    stdout_lines = [line + "\n" for line in claude_jsonl_lines]

    monkeypatch.setattr(
        "planora.agents.runner.asyncio.create_subprocess_exec",
        _fake_process_factory(stdout_lines=stdout_lines, stderr_lines=[]),
    )

    events: list[StreamEvent] = []
    output_path = tmp_path / "claude.md"
    result = await AgentRunner().run(
        agent=agent,
        prompt="plan the work",
        output_path=output_path,
        on_event=events.append,
    )

    event_types = [e.event_type for e in events]
    assert StreamEventType.INIT in event_types
    assert StreamEventType.TOOL_START in event_types
    assert StreamEventType.TOOL_DONE in event_types
    assert StreamEventType.TEXT in event_types
    assert result.exit_code == 0
    assert result.agent_name == "claude"


@pytest.mark.asyncio
async def test_claude_pipeline_writes_extracted_text_to_output_file(
    monkeypatch: pytest.MonkeyPatch,
    claude_jsonl_lines: list[str],
    tmp_path: Path,
) -> None:
    agent = _make_agent("claude", StreamFormat.CLAUDE)
    stdout_lines = [line + "\n" for line in claude_jsonl_lines]

    monkeypatch.setattr(
        "planora.agents.runner.asyncio.create_subprocess_exec",
        _fake_process_factory(stdout_lines=stdout_lines, stderr_lines=[]),
    )

    output_path = tmp_path / "claude.md"
    await AgentRunner().run(
        agent=agent,
        prompt="plan the work",
        output_path=output_path,
    )

    content = output_path.read_text(encoding="utf-8")
    assert "# Plan" in content


@pytest.mark.asyncio
async def test_claude_pipeline_archives_raw_stream_file(
    monkeypatch: pytest.MonkeyPatch,
    claude_jsonl_lines: list[str],
    tmp_path: Path,
) -> None:
    agent = _make_agent("claude", StreamFormat.CLAUDE)
    stdout_lines = [line + "\n" for line in claude_jsonl_lines]

    monkeypatch.setattr(
        "planora.agents.runner.asyncio.create_subprocess_exec",
        _fake_process_factory(stdout_lines=stdout_lines, stderr_lines=[]),
    )

    output_path = tmp_path / "claude.md"
    result = await AgentRunner().run(
        agent=agent,
        prompt="plan the work",
        output_path=output_path,
    )

    assert result.stream_path.exists()
    raw = result.stream_path.read_text(encoding="utf-8")
    assert len(raw) > 0


# ---------------------------------------------------------------------------
# Codex pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_codex_pipeline_emits_tool_start_and_done_events(
    monkeypatch: pytest.MonkeyPatch,
    codex_jsonl_lines: list[str],
    tmp_path: Path,
) -> None:
    agent = _make_agent("codex", StreamFormat.CODEX)
    stdout_lines = [line + "\n" for line in codex_jsonl_lines]

    monkeypatch.setattr(
        "planora.agents.runner.asyncio.create_subprocess_exec",
        _fake_process_factory(stdout_lines=stdout_lines, stderr_lines=[]),
    )

    events: list[StreamEvent] = []
    output_path = tmp_path / "codex.md"
    result = await AgentRunner().run(
        agent=agent,
        prompt="plan the work",
        output_path=output_path,
        on_event=events.append,
    )

    event_types = [e.event_type for e in events]
    assert StreamEventType.TOOL_START in event_types
    assert StreamEventType.TOOL_DONE in event_types
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Copilot pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_copilot_pipeline_extracts_text_from_jsonl(
    monkeypatch: pytest.MonkeyPatch,
    copilot_jsonl_lines: list[str],
    tmp_path: Path,
) -> None:
    agent = _make_agent("copilot", StreamFormat.COPILOT)
    stdout_lines = [line + "\n" for line in copilot_jsonl_lines]

    monkeypatch.setattr(
        "planora.agents.runner.asyncio.create_subprocess_exec",
        _fake_process_factory(stdout_lines=stdout_lines, stderr_lines=[]),
    )

    events: list[StreamEvent] = []
    output_path = tmp_path / "copilot.md"
    result = await AgentRunner().run(
        agent=agent,
        prompt="plan the work",
        output_path=output_path,
        on_event=events.append,
    )

    text_events = [e for e in events if e.event_type == StreamEventType.TEXT]
    assert len(text_events) > 0
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Gemini pipeline — stderr-as-stream path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gemini_pipeline_reads_events_from_stderr_stream(
    monkeypatch: pytest.MonkeyPatch,
    gemini_jsonl_lines: list[str],
    tmp_path: Path,
) -> None:
    agent = AgentConfig(
        name="gemini",
        binary="gemini",
        model="gemini-model",
        subcommand="-p",
        flags={AgentMode.PLAN: [], AgentMode.FIX: []},
        stream_format=StreamFormat.GEMINI,
        output_extraction=OutputExtraction(
            strategy=OutputExtraction.Strategy.STDOUT_CAPTURE,
            stderr_as_stream=True,
        ),
    )
    stdout_lines = ["# Final gemini plan\n", "Step 1: Do the thing\n"]
    stderr_lines = [line + "\n" for line in gemini_jsonl_lines]

    monkeypatch.setattr(
        "planora.agents.runner.asyncio.create_subprocess_exec",
        _fake_process_factory(stdout_lines=stdout_lines, stderr_lines=stderr_lines),
    )

    events: list[StreamEvent] = []
    output_path = tmp_path / "gemini.md"
    result = await AgentRunner().run(
        agent=agent,
        prompt="plan the work",
        output_path=output_path,
        on_event=events.append,
    )

    # Gemini output is captured from stdout (secondary stream)
    content = output_path.read_text(encoding="utf-8")
    assert "# Final gemini plan" in content
    assert result.exit_code == 0
    event_types = [e.event_type for e in events]
    assert StreamEventType.TOOL_START in event_types


# ---------------------------------------------------------------------------
# OpenCode pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_opencode_pipeline_emits_tool_events(
    monkeypatch: pytest.MonkeyPatch,
    opencode_jsonl_lines: list[str],
    tmp_path: Path,
) -> None:
    agent = _make_agent("opencode", StreamFormat.OPENCODE)
    stdout_lines = [line + "\n" for line in opencode_jsonl_lines]

    monkeypatch.setattr(
        "planora.agents.runner.asyncio.create_subprocess_exec",
        _fake_process_factory(stdout_lines=stdout_lines, stderr_lines=[]),
    )

    events: list[StreamEvent] = []
    output_path = tmp_path / "opencode.md"
    result = await AgentRunner().run(
        agent=agent,
        prompt="plan the work",
        output_path=output_path,
        on_event=events.append,
    )

    event_types = [e.event_type for e in events]
    assert StreamEventType.TOOL_START in event_types
    assert StreamEventType.TEXT in event_types
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Non-zero exit code produces error field
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_captures_stderr_tail_on_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    agent = _make_agent("claude", StreamFormat.CLAUDE)
    stderr_lines = ["fatal: authentication failed\n"]

    monkeypatch.setattr(
        "planora.agents.runner.asyncio.create_subprocess_exec",
        _fake_process_factory(
            stdout_lines=[],
            stderr_lines=stderr_lines,
            exit_code=1,
        ),
    )

    output_path = tmp_path / "claude.md"
    result = await AgentRunner().run(
        agent=agent,
        prompt="plan the work",
        output_path=output_path,
    )

    assert result.exit_code == 1
    assert result.error is not None
    assert "authentication failed" in result.error


# ---------------------------------------------------------------------------
# Dry run skips subprocess and returns zero-duration empty result
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_dry_run_returns_empty_result_and_skips_subprocess(
    tmp_path: Path,
) -> None:
    agent = _make_agent("claude", StreamFormat.CLAUDE)

    result = await AgentRunner().run(
        agent=agent,
        prompt="dry run",
        output_path=tmp_path / "result.md",
        dry_run=True,
    )

    assert result.exit_code == 0
    assert result.duration == timedelta(0)
    assert result.output_empty is True
    assert not result.output_path.exists()
