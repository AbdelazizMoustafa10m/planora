from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from decimal import Decimal, InvalidOperation
from typing import Any

from planora.agents.registry import StreamFormat
from planora.core.events import StreamEvent, StreamEventType

logger = logging.getLogger(__name__)

type _ParseFn = Callable[[dict[str, Any]], list[StreamEvent]]


class StreamParser:
    """Parses JSONL streams from a given agent format into typed StreamEvent objects."""

    def __init__(self, stream_format: StreamFormat) -> None:
        self._format = stream_format
        self._copilot_init_seen: bool = False
        self._parsers: dict[StreamFormat, _ParseFn] = {
            StreamFormat.CLAUDE: self._parse_claude,
            StreamFormat.CODEX: self._parse_codex,
            StreamFormat.COPILOT: self._parse_copilot,
            StreamFormat.OPENCODE: self._parse_opencode,
            StreamFormat.GEMINI: self._parse_gemini,
        }

    def parse_line(self, line: str) -> list[StreamEvent]:
        """Parse one JSONL line. Returns zero or more StreamEvents.

        On malformed JSON: logs warning, returns empty list.
        On unknown event type: logs debug, returns empty list.
        """
        stripped = line.strip()
        if not stripped:
            return []
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            logger.warning("Malformed JSON line: %s", stripped[:200])
            return []
        if not isinstance(data, dict):
            return []
        parser = self._parsers.get(self._format)
        if parser is None:
            return []
        return parser(data)

    async def parse_stream(self, stream: AsyncIterator[str]) -> AsyncGenerator[StreamEvent, None]:
        """Async generator yielding StreamEvents from an async line iterator."""
        async for line in stream:
            for event in self.parse_line(line):
                yield event

    # ------------------------------------------------------------------
    # Tool detail extraction (shared across all format parsers)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_tool_detail(
        tool_name: str | None, tool_input: dict[str, Any] | None
    ) -> str | None:
        if not tool_name or not tool_input:
            return None
        name_lower = tool_name.lower()
        if name_lower == "read":
            return str(tool_input.get("file_path") or tool_input.get("filename") or "")
        if name_lower in ("write", "edit"):
            return str(tool_input.get("file_path") or tool_input.get("filename") or "")
        if name_lower == "glob":
            return str(tool_input.get("pattern", ""))
        if name_lower == "grep":
            pattern = str(tool_input.get("pattern", ""))
            return pattern[:50]
        if name_lower == "bash":
            cmd = str(tool_input.get("command", ""))
            return f"$ {cmd}"[:80]
        if name_lower == "agent":
            desc = str(tool_input.get("description", ""))
            return f'Agent "{desc}"'
        if "web_search" in name_lower or "deep_search" in name_lower:
            query = str(tool_input.get("query", ""))
            return f'Web "{query}"'[:50]
        if name_lower.startswith("mcp__"):
            return f"MCP {tool_name}"
        if name_lower == "ls":
            return str(tool_input.get("path", ""))
        return None

    # ------------------------------------------------------------------
    # Decimal conversion helper
    # ------------------------------------------------------------------

    @staticmethod
    def _to_decimal(value: object) -> Decimal | None:
        if value is None:
            return None
        try:
            return Decimal(str(value))
        except InvalidOperation:
            return None

    # ------------------------------------------------------------------
    # Claude format parser
    # ------------------------------------------------------------------

    def _parse_claude(self, data: dict[str, Any]) -> list[StreamEvent]:
        event_type: str = data.get("type", "")

        if event_type == "system":
            return self._parse_claude_system(data)
        if event_type == "content_block_start":
            return self._parse_claude_content_block_start(data)
        if event_type == "assistant":
            return self._parse_claude_assistant(data)
        if event_type == "result":
            return self._parse_claude_result(data)

        logger.debug("Claude: unknown event type %r", event_type)
        return []

    def _parse_claude_system(self, data: dict[str, Any]) -> list[StreamEvent]:
        subtype: str = data.get("subtype", "")
        if subtype == "init":
            return [
                StreamEvent(
                    event_type=StreamEventType.INIT,
                    session_id=_str_or_none(data.get("session_id")),
                    raw=data,
                )
            ]
        if subtype == "api_retry":
            return [
                StreamEvent(
                    event_type=StreamEventType.RETRY,
                    retry_attempt=_int_or_none(data.get("attempt")),
                    retry_max=_int_or_none(data.get("max_retries")),
                    retry_delay_ms=_int_or_none(data.get("delay")),
                    raw=data,
                )
            ]
        logger.debug("Claude system: unknown subtype %r", subtype)
        return []

    def _parse_claude_content_block_start(self, data: dict[str, Any]) -> list[StreamEvent]:
        content_block = data.get("content_block")
        if not isinstance(content_block, dict):
            return []
        if content_block.get("type") != "tool_use":
            return []
        return [
            StreamEvent(
                event_type=StreamEventType.TOOL_START,
                tool_name=_str_or_none(content_block.get("name")),
                tool_id=_str_or_none(content_block.get("id")),
                tool_status="running",
                raw=data,
            )
        ]

    def _parse_claude_assistant(self, data: dict[str, Any]) -> list[StreamEvent]:
        message = data.get("message")
        if not isinstance(message, dict):
            return []
        content = message.get("content")
        if not isinstance(content, list):
            return []

        events: list[StreamEvent] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type: str = item.get("type", "")
            if item_type == "tool_use":
                tool_name = _str_or_none(item.get("name"))
                tool_input = item.get("input")
                tool_input_dict: dict[str, Any] | None = (
                    tool_input if isinstance(tool_input, dict) else None
                )
                events.append(
                    StreamEvent(
                        event_type=StreamEventType.TOOL_DONE,
                        tool_name=tool_name,
                        tool_id=_str_or_none(item.get("id")),
                        tool_detail=self._extract_tool_detail(tool_name, tool_input_dict),
                        tool_status="done",
                        raw=data,
                    )
                )
            elif item_type == "text":
                text: str = str(item.get("text", ""))
                if text:
                    events.append(
                        StreamEvent(
                            event_type=StreamEventType.TEXT,
                            text_preview=text[:200],
                            raw=data,
                        )
                    )
        return events

    def _parse_claude_result(self, data: dict[str, Any]) -> list[StreamEvent]:
        return [
            StreamEvent(
                event_type=StreamEventType.RESULT,
                cost_usd=self._to_decimal(data.get("total_cost_usd")),
                duration_ms=_int_or_none(data.get("duration_ms")),
                num_turns=_int_or_none(data.get("num_turns")),
                session_id=_str_or_none(data.get("session_id")),
                raw=data,
            )
        ]

    # ------------------------------------------------------------------
    # Codex format parser
    # ------------------------------------------------------------------

    def _parse_codex(self, data: dict[str, Any]) -> list[StreamEvent]:
        event_type: str = data.get("type", "")

        if event_type == "thread.started":
            return [StreamEvent(event_type=StreamEventType.INIT, raw=data)]

        if event_type == "item.started":
            item = data.get("item", {})
            item_dict: dict[str, Any] = item if isinstance(item, dict) else {}
            return [
                StreamEvent(
                    event_type=StreamEventType.TOOL_START,
                    tool_id=_str_or_none(item_dict.get("id")),
                    tool_name=_str_or_none(item_dict.get("type")),
                    tool_status="running",
                    raw=data,
                )
            ]

        if event_type == "item.completed":
            item = data.get("item", {})
            item_dict = item if isinstance(item, dict) else {}
            item_type: str = str(item_dict.get("type", ""))
            if item_type == "agent_message":
                content = item_dict.get("content")
                text_val = ""
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "output_text":
                            text_val = str(part.get("text", ""))
                            break
                elif isinstance(content, str):
                    text_val = content
                return [
                    StreamEvent(
                        event_type=StreamEventType.TEXT,
                        text_preview=text_val[:200] if text_val else None,
                        raw=data,
                    )
                ]
            return [
                StreamEvent(
                    event_type=StreamEventType.TOOL_DONE,
                    tool_id=_str_or_none(item_dict.get("id")),
                    tool_name=item_type or None,
                    tool_status="done",
                    raw=data,
                )
            ]

        if event_type == "turn.completed":
            return [
                StreamEvent(
                    event_type=StreamEventType.RESULT,
                    raw=data,
                )
            ]

        logger.debug("Codex: unknown event type %r", event_type)
        return []

    # ------------------------------------------------------------------
    # Copilot format parser
    # ------------------------------------------------------------------

    def _parse_copilot(self, data: dict[str, Any]) -> list[StreamEvent]:
        events: list[StreamEvent] = []

        if not self._copilot_init_seen:
            self._copilot_init_seen = True
            events.append(StreamEvent(event_type=StreamEventType.INIT, raw=data))

        # Tool invocation: presence of "toolName" or "tool" key
        tool_name = _str_or_none(data.get("toolName") or data.get("tool"))
        if tool_name:
            tool_input = data.get("parameters") or data.get("input") or data.get("args")
            tool_input_dict: dict[str, Any] | None = (
                tool_input if isinstance(tool_input, dict) else None
            )
            is_done = bool(data.get("result") or data.get("output") or data.get("done"))
            if is_done:
                events.append(
                    StreamEvent(
                        event_type=StreamEventType.TOOL_DONE,
                        tool_name=tool_name,
                        tool_id=_str_or_none(data.get("id")),
                        tool_detail=self._extract_tool_detail(tool_name, tool_input_dict),
                        tool_status="done",
                        raw=data,
                    )
                )
            else:
                events.append(
                    StreamEvent(
                        event_type=StreamEventType.TOOL_START,
                        tool_name=tool_name,
                        tool_id=_str_or_none(data.get("id")),
                        tool_detail=self._extract_tool_detail(tool_name, tool_input_dict),
                        tool_status="running",
                        raw=data,
                    )
                )
            return events

        # Text/message chunk
        text_val = _str_or_none(
            data.get("text") or data.get("message") or data.get("content")
        )
        if text_val:
            events.append(
                StreamEvent(
                    event_type=StreamEventType.TEXT,
                    text_preview=text_val[:200],
                    raw=data,
                )
            )
            return events

        logger.debug("Copilot: unrecognised object keys: %s", list(data.keys())[:10])
        return events

    # ------------------------------------------------------------------
    # OpenCode format parser
    # ------------------------------------------------------------------

    def _parse_opencode(self, data: dict[str, Any]) -> list[StreamEvent]:
        event_type: str = data.get("type", "")

        if event_type == "step_start":
            return [StreamEvent(event_type=StreamEventType.INIT, raw=data)]

        if event_type == "tool_use":
            has_result = "result" in data
            tool_name = _str_or_none(data.get("tool") or data.get("name"))
            tool_input = data.get("input")
            tool_input_dict: dict[str, Any] | None = (
                tool_input if isinstance(tool_input, dict) else None
            )
            if has_result:
                return [
                    StreamEvent(
                        event_type=StreamEventType.TOOL_DONE,
                        tool_name=tool_name,
                        tool_id=_str_or_none(data.get("id")),
                        tool_detail=self._extract_tool_detail(tool_name, tool_input_dict),
                        tool_status="done",
                        raw=data,
                    )
                ]
            return [
                StreamEvent(
                    event_type=StreamEventType.TOOL_START,
                    tool_name=tool_name,
                    tool_id=_str_or_none(data.get("id")),
                    tool_detail=self._extract_tool_detail(tool_name, tool_input_dict),
                    tool_status="running",
                    raw=data,
                )
            ]

        if event_type == "text":
            part = data.get("part")
            text_val = ""
            if isinstance(part, dict):
                text_val = str(part.get("text", ""))
            if text_val:
                return [
                    StreamEvent(
                        event_type=StreamEventType.TEXT,
                        text_preview=text_val[:200],
                        raw=data,
                    )
                ]
            return []

        if event_type == "step_finish":
            return [StreamEvent(event_type=StreamEventType.RESULT, raw=data)]

        logger.debug("OpenCode: unknown event type %r", event_type)
        return []

    # ------------------------------------------------------------------
    # Gemini format parser (best-effort)
    # ------------------------------------------------------------------

    def _parse_gemini(self, data: dict[str, Any]) -> list[StreamEvent]:
        events: list[StreamEvent] = []

        # Detect tool info: presence of "functionCall" or "tool_calls"
        tool_calls = data.get("functionCall") or data.get("tool_calls")
        if tool_calls:
            if isinstance(tool_calls, dict):
                tool_calls = [tool_calls]
            if isinstance(tool_calls, list):
                for call in tool_calls:
                    if not isinstance(call, dict):
                        continue
                    tool_name = _str_or_none(call.get("name"))
                    tool_args = call.get("args")
                    tool_args_dict: dict[str, Any] | None = (
                        tool_args if isinstance(tool_args, dict) else None
                    )
                    events.append(
                        StreamEvent(
                            event_type=StreamEventType.TOOL_START,
                            tool_name=tool_name,
                            tool_detail=self._extract_tool_detail(tool_name, tool_args_dict),
                            tool_status="running",
                            raw=data,
                        )
                    )
            return events

        # Final result blob: presence of "usageMetadata" or "candidates"
        if "usageMetadata" in data or "candidates" in data:
            candidates = data.get("candidates")
            text_val = ""
            if isinstance(candidates, list) and candidates:
                first = candidates[0]
                if isinstance(first, dict):
                    content = first.get("content")
                    if isinstance(content, dict):
                        parts = content.get("parts")
                        if isinstance(parts, list) and parts:
                            first_part = parts[0]
                            if isinstance(first_part, dict):
                                text_val = str(first_part.get("text", ""))
            if text_val:
                events.append(
                    StreamEvent(
                        event_type=StreamEventType.TEXT,
                        text_preview=text_val[:200],
                        raw=data,
                    )
                )
            if "usageMetadata" in data:
                events.append(StreamEvent(event_type=StreamEventType.RESULT, raw=data))
            return events

        logger.debug("Gemini: unrecognised object keys: %s", list(data.keys())[:10])
        return []


# ------------------------------------------------------------------
# Module-level helpers (pure functions, no side effects)
# ------------------------------------------------------------------


def _str_or_none(value: object) -> str | None:
    if value is None:
        return None
    s = str(value)
    return s if s else None


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None
