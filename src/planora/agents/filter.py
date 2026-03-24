from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from planora.agents.registry import StreamFormat

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class StreamFilter:
    """
    Drops high-volume events before parsing.

    Uses string-level matching BEFORE json.loads() to avoid parsing ~95% of lines
    that are dropped. This is the key optimization.
    """

    SKIP_TYPES_CLAUDE: ClassVar[set[str]] = {
        "content_block_stop",
        "message_start",
        "message_stop",
    }

    SKIP_TYPES_CODEX: ClassVar[set[str]] = set()

    def __init__(self, stream_format: StreamFormat) -> None:
        self._format = stream_format
        # Pre-build string patterns for fast matching
        if stream_format == StreamFormat.CLAUDE:
            self._skip_patterns = [f'"type":"{t}"' for t in self.SKIP_TYPES_CLAUDE]
            # Also match with space after colon
            self._skip_patterns += [f'"type": "{t}"' for t in self.SKIP_TYPES_CLAUDE]
        elif stream_format == StreamFormat.CODEX:
            self._skip_patterns = [f'"type":"{t}"' for t in self.SKIP_TYPES_CODEX]
            self._skip_patterns += [f'"type": "{t}"' for t in self.SKIP_TYPES_CODEX]
        else:
            self._skip_patterns = []

    def should_keep(self, line: str) -> bool:
        """
        Fast-path: use string search ('"type":"content_block_delta"' in line)
        BEFORE json.loads(). This avoids parsing ~95% of lines.

        For CLAUDE format: check against SKIP_TYPES_CLAUDE
        For CODEX format: check against SKIP_TYPES_CODEX
        For other formats: keep all lines (no known high-volume types)

        Empty lines and whitespace-only lines are always dropped.
        If the line doesn't contain a "type" field at all, keep it
        (could be malformed but important).
        Only drop lines that positively match a skip type.
        """
        if not line or not line.strip():
            return False

        return all(pattern not in line for pattern in self._skip_patterns)

    async def filter_stream(self, raw: AsyncIterator[str]) -> AsyncIterator[str]:
        """Async generator yielding only actionable JSONL lines."""
        async for line in raw:
            if self.should_keep(line):
                yield line
