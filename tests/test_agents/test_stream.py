from __future__ import annotations

from decimal import Decimal

import pytest

from planora.agents.registry import StreamFormat
from planora.agents.stream import StreamParser
from planora.core.events import StreamEventType


def _flatten_events(parser: StreamParser, lines: list[str]) -> list:
    events = []
    for line in lines:
        events.extend(parser.parse_line(line))
    return events


def test_parse_claude_stream_events(claude_jsonl_lines) -> None:
    events = _flatten_events(StreamParser(StreamFormat.CLAUDE), claude_jsonl_lines)

    assert [event.event_type for event in events] == [
        StreamEventType.INIT,
        StreamEventType.TOOL_START,
        StreamEventType.TOOL_EXEC,  # content_block_delta with input_json_delta
        StreamEventType.TOOL_DONE,
        StreamEventType.TEXT,
        StreamEventType.RESULT,
    ]
    assert events[0].session_id == "sess-123"
    assert events[1].tool_name == "Read"
    assert events[1].tool_id == "tb-1"
    assert events[3].tool_detail == "foo.py"
    assert events[4].text_preview == "# Plan\n\nAdd the requested tests."
    assert events[5].cost_usd == Decimal("0.0123")
    assert events[5].num_turns == 3


def test_parse_codex_stream_events(codex_jsonl_lines) -> None:
    events = _flatten_events(StreamParser(StreamFormat.CODEX), codex_jsonl_lines)

    assert [event.event_type for event in events] == [
        StreamEventType.INIT,
        StreamEventType.TOOL_START,
        StreamEventType.TOOL_EXEC,  # response.output_item.delta
        StreamEventType.TOOL_DONE,
        StreamEventType.TEXT,
        StreamEventType.RESULT,
    ]
    assert events[1].tool_name == "function_call"
    assert events[3].tool_status == "done"
    assert events[4].text_preview == "Codex summary"


def test_parse_copilot_stream_events(copilot_jsonl_lines) -> None:
    events = _flatten_events(StreamParser(StreamFormat.COPILOT), copilot_jsonl_lines)

    assert [event.event_type for event in events] == [
        StreamEventType.INIT,
        StreamEventType.TOOL_START,
        StreamEventType.TOOL_DONE,
        StreamEventType.TEXT,
    ]
    assert events[1].tool_name == "Read"
    assert events[1].tool_detail == "src/planora/core/config.py"
    assert events[2].tool_status == "done"
    assert events[3].text_preview == "Copilot response text"


def test_parse_opencode_stream_events(opencode_jsonl_lines) -> None:
    events = _flatten_events(StreamParser(StreamFormat.OPENCODE), opencode_jsonl_lines)

    assert [event.event_type for event in events] == [
        StreamEventType.INIT,
        StreamEventType.TOOL_START,
        StreamEventType.TOOL_DONE,
        StreamEventType.TEXT,
        StreamEventType.RESULT,
    ]
    assert events[1].tool_name == "bash"
    assert events[1].tool_detail == "$ ls -la"
    assert events[3].text_preview == "OpenCode response text"


def test_parse_gemini_stream_events(gemini_jsonl_lines) -> None:
    events = _flatten_events(StreamParser(StreamFormat.GEMINI), gemini_jsonl_lines)

    assert [event.event_type for event in events] == [
        StreamEventType.TOOL_START,
        StreamEventType.TEXT,
        StreamEventType.RESULT,
    ]
    assert events[0].tool_name == "web_search_exa"
    assert events[0].tool_detail == 'Web "planora tests"'
    assert events[1].text_preview == "Gemini response text"


def test_parse_line_returns_empty_for_malformed_json(caplog) -> None:
    parser = StreamParser(StreamFormat.CLAUDE)

    with caplog.at_level("WARNING"):
        events = parser.parse_line("{not-json")

    assert events == []
    assert "Malformed JSON line" in caplog.text


@pytest.mark.parametrize(
    ("tool_name", "tool_input", "expected"),
    [
        ("Read", {"file_path": "README.md"}, "README.md"),
        ("Grep", {"pattern": "needle" * 20}, ("needle" * 20)[:50]),
        ("Bash", {"command": "pytest tests/test_stream.py"}, "$ pytest tests/test_stream.py"),
        ("Agent", {"description": "review auth flow"}, 'Agent "review auth flow"'),
        ("mcp__exa__web_search_exa", {"query": "planora"}, 'Web "planora"'),
    ],
)
def test_extract_tool_detail_covers_common_tool_shapes(
    tool_name: str,
    tool_input: dict[str, str],
    expected: str,
) -> None:
    assert StreamParser._extract_tool_detail(tool_name, tool_input) == expected


def test_to_decimal_returns_none_for_invalid_values() -> None:
    assert StreamParser._to_decimal("not-a-decimal") is None


@pytest.mark.asyncio
async def test_parse_stream_yields_async_events(copilot_jsonl_lines) -> None:
    async def raw_lines():
        for line in copilot_jsonl_lines:
            yield line

    parser = StreamParser(StreamFormat.COPILOT)
    events = [event async for event in parser.parse_stream(raw_lines())]

    assert [event.event_type for event in events] == [
        StreamEventType.INIT,
        StreamEventType.TOOL_START,
        StreamEventType.TOOL_DONE,
        StreamEventType.TEXT,
    ]
