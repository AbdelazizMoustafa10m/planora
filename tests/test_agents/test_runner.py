from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from planora.agents.registry import AgentConfig, AgentMode, OutputExtraction, StreamFormat
from planora.agents.runner import AgentRunner, _write_output
from planora.core.events import StreamEventType


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
        self.killed = False

    async def wait(self) -> int:
        return self._exit_code

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = self._exit_code

    def kill(self) -> None:
        self.killed = True
        self.returncode = self._exit_code


def test_build_command_handles_agent_specific_output_flags(registry, tmp_path) -> None:
    runner = AgentRunner()

    claude_cmd = runner._build_command(
        registry.get("claude"),
        "prompt text",
        tmp_path / "claude.md",
        AgentMode.PLAN,
    )
    codex_cmd = runner._build_command(
        registry.get("codex"),
        "prompt text",
        tmp_path / "codex.md",
        AgentMode.PLAN,
    )

    assert claude_cmd[:3] == ["claude", "-p", "prompt text"]
    assert "--include-partial-messages" in claude_cmd
    assert "-o" not in claude_cmd

    assert codex_cmd[:3] == ["codex", "exec", "prompt text"]
    assert "--json" in codex_cmd
    assert codex_cmd[-2:] == ["-o", str(tmp_path / "codex.md")]


@pytest.mark.asyncio
async def test_run_dry_run_returns_empty_result_without_creating_files(
    sample_agent_config,
    tmp_path,
) -> None:
    result = await AgentRunner().run(
        agent=sample_agent_config,
        prompt="dry run",
        output_path=tmp_path / "result.md",
        dry_run=True,
    )

    assert result.exit_code == 0
    assert result.duration == timedelta(0)
    assert result.output_empty is True
    assert result.output_path.exists() is False


def test_write_output_strips_preamble_for_jq_filter(tmp_path) -> None:
    output_path = tmp_path / "plan.md"
    agent = AgentConfig(
        name="claude",
        binary="claude",
        model="model",
        subcommand="-p",
        flags={AgentMode.PLAN: [], AgentMode.FIX: []},
        stream_format=StreamFormat.CLAUDE,
        output_extraction=OutputExtraction(
            strategy=OutputExtraction.Strategy.JQ_FILTER,
            strip_preamble=True,
        ),
    )

    _write_output(
        agent=agent,
        output_path=output_path,
        text_chunks=["Preamble\nstill preamble\n# Title\nBody"],
        stdout_chunks=[],
    )

    assert output_path.read_text(encoding="utf-8") == "# Title\nBody"


def test_write_output_uses_secondary_chunks_for_stderr_stream(tmp_path) -> None:
    output_path = tmp_path / "gemini.md"
    agent = AgentConfig(
        name="gemini",
        binary="gemini",
        model="model",
        subcommand="-p",
        flags={AgentMode.PLAN: [], AgentMode.FIX: []},
        stream_format=StreamFormat.GEMINI,
        output_extraction=OutputExtraction(
            strategy=OutputExtraction.Strategy.STDOUT_CAPTURE,
            stderr_as_stream=True,
        ),
    )

    _write_output(
        agent=agent,
        output_path=output_path,
        text_chunks=["ignored"],
        stdout_chunks=["# Final\n", "Plan\n"],
    )

    assert output_path.read_text(encoding="utf-8") == "# Final\nPlan\n"


@pytest.mark.asyncio
async def test_run_processes_stream_pipeline_and_writes_outputs(monkeypatch, tmp_path) -> None:
    agent = AgentConfig(
        name="copilot-test",
        binary="copilot",
        model="claude-sonnet-4.5",
        subcommand="-p",
        flags={AgentMode.PLAN: ["--stream", "on"], AgentMode.FIX: []},
        stream_format=StreamFormat.COPILOT,
        env_vars={"COPILOT_TEST_FLAG": "1"},
        output_extraction=OutputExtraction(
            strategy=OutputExtraction.Strategy.STDOUT_CAPTURE,
        ),
    )
    stdout_lines = [
        '{"id":"cp-1","toolName":"Read","parameters":{"file_path":"README.md"}}\n',
        '{"id":"cp-1","toolName":"Read","parameters":{"file_path":"README.md"},"done":true,"result":"ok"}\n',
        '{"text":"Copilot response text"}\n',
    ]
    stderr_lines = ["warning: nothing to report\n"]
    captured: dict[str, object] = {}

    async def fake_create_subprocess_exec(*cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = kwargs["env"]
        return _FakeProcess(stdout_lines=stdout_lines, stderr_lines=stderr_lines)

    monkeypatch.setattr(
        "planora.agents.runner.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    events = []
    output_path = tmp_path / "result.md"
    result = await AgentRunner().run(
        agent=agent,
        prompt="build a plan",
        output_path=output_path,
        dry_run=False,
        on_event=events.append,
    )

    assert captured["cmd"] == [
        "copilot",
        "-p",
        "build a plan",
        "--model",
        "claude-sonnet-4.5",
        "--stream",
        "on",
    ]
    assert captured["env"]["COPILOT_TEST_FLAG"] == "1"

    assert output_path.read_text(encoding="utf-8") == "Copilot response text"
    assert result.stream_path.read_text(encoding="utf-8") == "".join(stdout_lines)
    assert result.log_path.read_text(encoding="utf-8") == "".join(stderr_lines)
    assert [event.event_type for event in events] == [
        StreamEventType.INIT,
        StreamEventType.TOOL_START,
        StreamEventType.TOOL_DONE,
        StreamEventType.TEXT,
    ]
    assert result.agent_name == "copilot-test"
    assert result.exit_code == 0
    assert result.output_empty is False
    assert result.duration >= timedelta(0)


@pytest.mark.asyncio
async def test_run_keeps_gemini_stderr_in_log_and_emits_snapshots(
    monkeypatch,
    tmp_path,
) -> None:
    agent = AgentConfig(
        name="gemini-test",
        binary="gemini",
        model="gemini-3.1-pro-preview",
        subcommand="-p",
        flags={AgentMode.PLAN: ["--approval-mode", "plan"], AgentMode.FIX: []},
        stream_format=StreamFormat.GEMINI,
        output_extraction=OutputExtraction(
            strategy=OutputExtraction.Strategy.STDOUT_CAPTURE,
            stderr_as_stream=True,
        ),
    )
    stdout_lines = ["# Final plan\n", "Ship it\n"]
    stderr_lines = [
        '{"functionCall":{"name":"Read","args":{"file_path":"README.md"}}}\n',
        '{"candidates":[{"content":{"parts":[{"text":"Gemini preview"}]}}],"usageMetadata":{}}\n',
    ]

    async def fake_create_subprocess_exec(*_cmd, **_kwargs):
        return _FakeProcess(stdout_lines=stdout_lines, stderr_lines=stderr_lines)

    monkeypatch.setattr(
        "planora.agents.runner.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    output_path = tmp_path / "gemini.md"
    events = []
    snapshots = []
    result = await AgentRunner().run(
        agent=agent,
        prompt="review the plan",
        output_path=output_path,
        on_event=events.append,
        on_snapshot=snapshots.append,
    )

    assert output_path.read_text(encoding="utf-8") == "".join(stdout_lines)
    assert result.stream_path.read_text(encoding="utf-8") == "".join(stderr_lines)
    assert result.log_path.read_text(encoding="utf-8") == "".join(stderr_lines)
    assert [event.event_type for event in events] == [
        StreamEventType.TOOL_START,
        StreamEventType.TEXT,
        StreamEventType.RESULT,
    ]
    assert snapshots
    assert snapshots[-1].state.value == "completed"
    assert result.exit_code == 0
