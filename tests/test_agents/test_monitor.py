from __future__ import annotations

from decimal import Decimal

import pytest

from planora.agents.monitor import AgentMonitor, _friendly_name
from planora.core.events import AgentState, StreamEvent, StreamEventType


def test_friendly_name_maps_known_tools() -> None:
    assert _friendly_name("Read") == "Read file"
    assert _friendly_name("CustomTool") == "CustomTool"


def test_monitor_tracks_state_counters_and_results() -> None:
    monitor = AgentMonitor("claude")

    monitor.update(StreamEvent(event_type=StreamEventType.INIT, session_id="sess-1"))
    monitor.update(
        StreamEvent(
            event_type=StreamEventType.STATE_CHANGE,
            tool_detail="thinking",
        )
    )
    monitor.update(
        StreamEvent(
            event_type=StreamEventType.TOOL_START,
            tool_name="Read",
            tool_id="tool-1",
            tool_detail="README.md",
            tool_status="running",
        )
    )
    monitor.update(
        StreamEvent(
            event_type=StreamEventType.TOOL_EXEC,
            tool_id="tool-1",
            tool_detail="src/planora/core/config.py",
        )
    )
    monitor.update(StreamEvent(event_type=StreamEventType.TEXT, text_preview="draft"))
    monitor.update(
        StreamEvent(
            event_type=StreamEventType.TOOL_DONE,
            tool_name="Read",
            tool_id="tool-1",
            tool_detail="src/planora/core/config.py",
            tool_status="done",
        )
    )
    monitor.update(StreamEvent(event_type=StreamEventType.SUBAGENT))
    monitor.update(
        StreamEvent(
            event_type=StreamEventType.RESULT,
            cost_usd=Decimal("1.25"),
            num_turns=4,
            session_id="sess-2",
        )
    )

    snap = monitor.snapshot()

    assert snap.state == AgentState.COMPLETED
    assert snap.counters.total == 1
    assert snap.counters.running == 0
    assert snap.counters.succeeded == 1
    assert snap.text_count == 1
    assert snap.subagent_count == 1
    assert snap.last_tool == "Read"
    assert snap.last_tool_detail == "src/planora/core/config.py"
    assert snap.recent_tools[0].friendly_name == "Read file"
    assert snap.cost_usd == Decimal("1.25")
    assert snap.num_turns == 4
    assert snap.session_id == "sess-2"
    assert snap.idle_seconds >= 0


def test_monitor_records_failed_tools_and_recent_tool_limit() -> None:
    monitor = AgentMonitor("codex", max_recent_tools=1)

    for tool_id in ("tool-1", "tool-2"):
        monitor.update(
            StreamEvent(
                event_type=StreamEventType.TOOL_START,
                tool_name="Bash",
                tool_id=tool_id,
                tool_status="running",
            )
        )
        monitor.update(
            StreamEvent(
                event_type=StreamEventType.TOOL_DONE,
                tool_name="Bash",
                tool_id=tool_id,
                tool_status="error" if tool_id == "tool-2" else "done",
            )
        )

    snap = monitor.snapshot()

    assert snap.counters.total == 2
    assert snap.counters.succeeded == 1
    assert snap.counters.failed == 1
    assert len(snap.recent_tools) == 1
    assert snap.recent_tools[0].tool_id == "tool-2"


@pytest.mark.asyncio
async def test_monitor_consume_updates_state_and_emits_callback() -> None:
    events = [
        StreamEvent(event_type=StreamEventType.INIT, session_id="sess-1"),
        StreamEvent(event_type=StreamEventType.TEXT, text_preview="hello"),
    ]
    seen: list[StreamEventType] = []

    async def raw_events():
        for event in events:
            yield event

    monitor = AgentMonitor("copilot")
    await monitor.consume(raw_events(), on_event=lambda event: seen.append(event.event_type))

    snap = monitor.snapshot()
    assert seen == [StreamEventType.INIT, StreamEventType.TEXT]
    assert snap.state == AgentState.WRITING
    assert snap.text_count == 1
