from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path  # noqa: TC003 — used at runtime in path operations
from typing import TYPE_CHECKING

from planora.agents.filter import StreamFilter
from planora.agents.monitor import AgentMonitor
from planora.agents.registry import AgentConfig, AgentMode, OutputExtraction
from planora.agents.stall import StallDetector
from planora.agents.stream import StreamParser
from planora.core.events import AgentResult, StreamEvent, StreamEventType

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from planora.core.events import AgentMonitorSnapshot

logger = logging.getLogger(__name__)


async def _read_lines(stream: asyncio.StreamReader) -> AsyncIterator[str]:
    """Yield decoded lines from an asyncio StreamReader."""
    while True:
        line_bytes = await stream.readline()
        if not line_bytes:
            break
        yield line_bytes.decode("utf-8", errors="replace")


class AgentRunner:
    """Executes an agent as an async subprocess and streams output through the pipeline."""

    async def run(
        self,
        agent: AgentConfig,
        prompt: str,
        output_path: Path,
        mode: AgentMode = AgentMode.PLAN,
        dry_run: bool = False,
        on_event: Callable[[StreamEvent], None] | None = None,
        on_snapshot: Callable[[AgentMonitorSnapshot], None] | None = None,
        snapshot_interval: float | None = None,
        on_process_start: Callable[[asyncio.subprocess.Process], None] | None = None,
        on_process_end: Callable[[asyncio.subprocess.Process], None] | None = None,
    ) -> AgentResult:
        """Run the agent subprocess and return a fully-populated AgentResult."""
        # Step 1 — Build command
        cmd = self._build_command(agent, prompt, output_path, mode)

        # Step 2 — Set up environment
        env = {**os.environ, **agent.env_vars}

        # Step 3 — Handle dry_run
        if dry_run:
            logger.info("dry_run=True, skipping subprocess. Command: %s", cmd)
            return AgentResult(
                agent_name=agent.name,
                output_path=output_path,
                stream_path=output_path.with_suffix(".stream"),
                log_path=output_path.with_suffix(".log"),
                exit_code=0,
                duration=timedelta(0),
                output_empty=True,
            )

        started_at = datetime.now()

        # Step 4 — Launch subprocess
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        if on_process_start is not None:
            on_process_start(proc)

        # Step 5 — Create pipeline components and derived paths
        stream_path = output_path.with_suffix(".stream")
        log_path = output_path.with_suffix(".log")

        stream_filter = StreamFilter(agent.stream_format)
        stream_parser = StreamParser(agent.stream_format)
        stall_detector = StallDetector(
            normal_timeout=agent.stall_timeout,
            deep_timeout=agent.deep_tool_timeout,
        )
        monitor = AgentMonitor(agent.name)

        # Collect TEXT event content for output extraction (JQ_FILTER / STDOUT_CAPTURE)
        text_chunks: list[str] = []

        def _collecting_on_event(event: StreamEvent) -> None:
            if event.event_type == StreamEventType.TEXT and event.text_preview is not None:
                text_chunks.append(event.text_preview)
            if on_event is not None:
                on_event(event)

        # Determine which pipe to read as the primary stream
        # For Gemini with stderr_as_stream=True, run stderr through the pipeline
        # and capture stdout as plain text; otherwise the opposite.
        assert proc.stdout is not None  # noqa: S101 — guaranteed by PIPE flag
        assert proc.stderr is not None  # noqa: S101 — guaranteed by PIPE flag

        primary_is_stderr = agent.output_extraction.stderr_as_stream
        if primary_is_stderr:
            primary_stream = proc.stderr
            secondary_stream = proc.stdout
        else:
            primary_stream = proc.stdout
            secondary_stream = proc.stderr

        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        async def _read_secondary() -> None:
            async for line in _read_lines(secondary_stream):
                if primary_is_stderr:
                    stdout_chunks.append(line)
                else:
                    stderr_chunks.append(line)

        secondary_task = asyncio.create_task(_read_secondary())
        snapshot_task = _start_snapshot_task(
            monitor=monitor,
            on_snapshot=on_snapshot,
            snapshot_interval=snapshot_interval,
        )

        try:
            # Build the 4-stage pipeline:
            #   raw_lines → tee to stream file → filter → parser → stall_detector → monitor
            raw_lines = _read_lines(primary_stream)
            filtered = stream_filter.filter_stream(
                _tee_to_file(
                    raw_lines,
                    stream_path,
                    collector=stderr_chunks if primary_is_stderr else None,
                )
            )
            parsed = stream_parser.parse_stream(filtered)
            watched = stall_detector.watch(parsed)

            await monitor.consume(watched, on_event=_collecting_on_event)

            # Wait for secondary reader to finish
            await secondary_task

            # Step 7 — Write stderr to log file
            log_path.write_text("".join(stderr_chunks), encoding="utf-8")

            # Step 8 — Wait for process exit
            exit_code = await proc.wait()

            duration = datetime.now() - started_at

            # Step 6 — Extract output
            _write_output(
                agent=agent,
                output_path=output_path,
                text_chunks=text_chunks,
                stdout_chunks=stdout_chunks,
            )

            # Build result from monitor snapshot
            snap = monitor.snapshot()
            if on_snapshot is not None:
                on_snapshot(snap)

            return AgentResult(
                agent_name=agent.name,
                output_path=output_path,
                stream_path=stream_path,
                log_path=log_path,
                exit_code=exit_code,
                duration=duration,
                cost_usd=snap.cost_usd,
                num_turns=snap.num_turns,
                session_id=snap.session_id,
                output_empty=not output_path.exists() or output_path.stat().st_size == 0,
            )
        except BaseException:
            await _shutdown_process(proc)
            raise
        finally:
            await _cancel_task(snapshot_task)
            await _cancel_task(secondary_task)
            if on_process_end is not None:
                on_process_end(proc)

    @staticmethod
    def _build_command(
        agent: AgentConfig,
        prompt: str,
        output_path: Path,
        mode: AgentMode,
    ) -> list[str]:
        """Build the subprocess command list."""
        cmd = [agent.binary, agent.subcommand, prompt]
        cmd.extend(["--model", agent.model])
        cmd.extend(agent.flags.get(mode, []))
        # Codex uses -o flag for direct file output
        if agent.output_extraction.strategy == OutputExtraction.Strategy.DIRECT_FILE:
            cmd.extend(["-o", str(output_path)])
        return cmd

    @staticmethod
    def _strip_preamble(text: str) -> str:
        """Remove text before the first markdown heading (^# )."""
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if re.match(r"^#\s", line):
                return "\n".join(lines[i:])
        return text  # No heading found, return as-is


async def _tee_to_file(
    raw: AsyncIterator[str],
    path: Path,
    collector: list[str] | None = None,
) -> AsyncIterator[str]:
    """Yield each line from raw while also writing it to path (tee for archival)."""
    with path.open("w", encoding="utf-8") as fh:
        async for line in raw:
            if collector is not None:
                collector.append(line)
            fh.write(line)
            yield line


def _write_output(
    agent: AgentConfig,
    output_path: Path,
    text_chunks: list[str],
    stdout_chunks: list[str],
) -> None:
    """Write extracted text to output_path based on the agent's OutputExtraction strategy."""
    extraction = agent.output_extraction

    match extraction.strategy:
        case OutputExtraction.Strategy.DIRECT_FILE:
            # Agent wrote directly to output_path — nothing to do.
            return

        case OutputExtraction.Strategy.JQ_FILTER:
            # text_preview fields are already the extracted text fragments.
            text = "".join(text_chunks)
            if extraction.strip_preamble:
                text = _strip_preamble_text(text)
            output_path.write_text(text, encoding="utf-8")

        case OutputExtraction.Strategy.STDOUT_CAPTURE:
            text = (
                "".join(stdout_chunks)
                if extraction.stderr_as_stream
                else "".join(text_chunks)
            )
            output_path.write_text(text, encoding="utf-8")


def _strip_preamble_text(text: str) -> str:
    """Remove text before the first markdown heading (^# )."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if re.match(r"^#\s", line):
            return "\n".join(lines[i:])
    return text


def _start_snapshot_task(
    *,
    monitor: AgentMonitor,
    on_snapshot: Callable[[AgentMonitorSnapshot], None] | None,
    snapshot_interval: float | None,
) -> asyncio.Task[None] | None:
    """Start the periodic snapshot emitter when snapshot delivery is enabled."""
    if on_snapshot is None or snapshot_interval is None or snapshot_interval <= 0:
        return None

    async def _emit_snapshots() -> None:
        while True:
            await asyncio.sleep(snapshot_interval)
            on_snapshot(monitor.snapshot())

    return asyncio.create_task(_emit_snapshots())


async def _cancel_task(task: asyncio.Task[None] | None) -> None:
    """Cancel and await a background task, ignoring normal cancellation noise."""
    if task is None or task.done():
        return

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def _shutdown_process(proc: asyncio.subprocess.Process) -> None:
    """Terminate a live subprocess and wait for it to exit."""
    if proc.returncode is not None:
        return

    proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=5.0)
    except TimeoutError:
        proc.kill()
        await proc.wait()
