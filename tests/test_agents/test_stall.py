from __future__ import annotations

from itertools import chain, repeat

import pytest

from planora.agents.stall import StallDetector
from planora.core.events import StreamEvent, StreamEventType


class _AsyncEventIterator:
    def __init__(self, events: list[StreamEvent]) -> None:
        self._events = iter(events)

    def __aiter__(self):
        return self

    async def __anext__(self) -> StreamEvent:
        try:
            return next(self._events)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


def _patch_watch_timing(
    monkeypatch: pytest.MonkeyPatch,
    *,
    steps: list[str],
    monotonic_values: list[float],
) -> None:
    step_iter = iter(steps)
    time_iter = chain(monotonic_values, repeat(monotonic_values[-1]))

    async def fake_wait_for(coro, timeout):
        del timeout
        step = next(step_iter)
        if step == "timeout":
            coro.close()
            raise TimeoutError
        return await coro

    monkeypatch.setattr("planora.agents.stall.asyncio.wait_for", fake_wait_for)
    monkeypatch.setattr("planora.agents.stall.time.monotonic", lambda: next(time_iter))


@pytest.mark.asyncio
async def test_watch_injects_single_stall_during_long_idle_period(monkeypatch) -> None:
    _patch_watch_timing(
        monkeypatch,
        steps=["event", "timeout", "event", "event"],
        monotonic_values=[0.0, 0.0, 0.02, 0.02],
    )
    detector = StallDetector(normal_timeout=0.01, deep_timeout=0.05, check_interval=0.002)
    source = _AsyncEventIterator(
        [
            StreamEvent(event_type=StreamEventType.INIT),
            StreamEvent(event_type=StreamEventType.TEXT, text_preview="ready"),
        ]
    )

    events = [event async for event in detector.watch(source)]

    assert [event.event_type for event in events] == [
        StreamEventType.INIT,
        StreamEventType.STALL,
        StreamEventType.TEXT,
    ]


@pytest.mark.asyncio
async def test_watch_uses_deep_timeout_for_long_running_research_tools(monkeypatch) -> None:
    _patch_watch_timing(
        monkeypatch,
        steps=["event", "timeout", "event", "event"],
        monotonic_values=[0.0, 0.0, 0.02, 0.02],
    )
    detector = StallDetector(normal_timeout=0.005, deep_timeout=0.04, check_interval=0.002)
    source = _AsyncEventIterator(
        [
            StreamEvent(
                event_type=StreamEventType.TOOL_START,
                tool_name="deep_search_exa",
                tool_id="tool-1",
            ),
            StreamEvent(
                event_type=StreamEventType.TOOL_DONE,
                tool_name="deep_search_exa",
                tool_id="tool-1",
                tool_status="done",
            ),
        ]
    )

    events = [event async for event in detector.watch(source)]

    assert [event.event_type for event in events] == [
        StreamEventType.TOOL_START,
        StreamEventType.TOOL_DONE,
    ]


@pytest.mark.asyncio
async def test_watch_resets_after_real_activity_and_can_stall_again(monkeypatch) -> None:
    _patch_watch_timing(
        monkeypatch,
        steps=["event", "timeout", "event", "timeout", "event", "event"],
        monotonic_values=[0.0, 0.0, 0.02, 0.02, 0.04, 0.04],
    )
    detector = StallDetector(normal_timeout=0.01, deep_timeout=0.05, check_interval=0.002)
    source = _AsyncEventIterator(
        [
            StreamEvent(event_type=StreamEventType.INIT),
            StreamEvent(event_type=StreamEventType.TEXT, text_preview="part 1"),
            StreamEvent(event_type=StreamEventType.RESULT),
        ]
    )

    events = [event async for event in detector.watch(source)]

    assert [event.event_type for event in events].count(StreamEventType.STALL) == 2
