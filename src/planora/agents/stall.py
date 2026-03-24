from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, ClassVar

from planora.core.events import StreamEvent, StreamEventType

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_SENTINEL = object()


async def _next_or_sentinel(
    aiter: AsyncIterator[StreamEvent],
) -> StreamEvent | object:
    """Await the next item from an async iterator, returning a sentinel on exhaustion."""
    try:
        return await aiter.__anext__()
    except StopAsyncIteration:
        return _SENTINEL


class StallDetector:
    """Wraps an event stream, injecting STALL events when the agent goes silent."""

    DEEP_TOOL_PATTERNS: ClassVar[set[str]] = {
        "deep_search_exa",
        "deep_researcher_start",
        "deep_researcher_check",
        "tavily_research",
        "tavily_crawl",
        "deep_research_exa",
    }

    def __init__(
        self,
        normal_timeout: float = 300.0,
        deep_timeout: float = 600.0,
        check_interval: float = 5.0,
    ) -> None:
        self._normal_timeout = normal_timeout
        self._deep_timeout = deep_timeout
        self._check_interval = check_interval
        self._active_tools: set[str] = set()

    def _is_deep_tool_active(self) -> bool:
        """Return True if any active tool matches DEEP_TOOL_PATTERNS."""
        return bool(self._active_tools & self.DEEP_TOOL_PATTERNS)

    def _current_timeout(self) -> float:
        """Return the applicable stall timeout based on current active tools."""
        return self._deep_timeout if self._is_deep_tool_active() else self._normal_timeout

    def _track_tool(self, event: StreamEvent) -> None:
        """Update active tool set from TOOL_START / TOOL_DONE events."""
        if event.event_type == StreamEventType.TOOL_START and event.tool_name is not None:
            self._active_tools.add(event.tool_name)
        elif event.event_type == StreamEventType.TOOL_DONE and event.tool_name is not None:
            self._active_tools.discard(event.tool_name)

    async def watch(self, events: AsyncIterator[StreamEvent]) -> AsyncIterator[StreamEvent]:
        """Yield events from the wrapped stream, injecting STALL events on silence.

        Uses asyncio.wait_for with check_interval as the polling window. When no event
        arrives within check_interval seconds, idle time is measured against
        _current_timeout(). A synthetic STALL event is yielded once per stall period;
        the flag resets when a real event arrives.
        """
        aiter_obj = events.__aiter__()
        last_event_at = time.monotonic()
        stall_injected = False

        while True:
            try:
                result = await asyncio.wait_for(
                    _next_or_sentinel(aiter_obj),
                    timeout=self._check_interval,
                )
            except TimeoutError:
                idle = time.monotonic() - last_event_at
                if idle >= self._current_timeout() and not stall_injected:
                    stall_injected = True
                    yield StreamEvent(event_type=StreamEventType.STALL)
                continue

            if result is _SENTINEL:
                break

            event: StreamEvent = result  # type: ignore[assignment]
            last_event_at = time.monotonic()
            stall_injected = False
            self._track_tool(event)
            yield event
