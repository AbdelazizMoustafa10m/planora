"""Tests for CLICallback dispatch_agent_event routing and cost tracking."""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from planora.cli.callbacks import CLICallback
from planora.core.events import StreamEvent, StreamEventType


@pytest.fixture
def cli_callback() -> CLICallback:
    console = MagicMock()
    return CLICallback(console=console)


@pytest.mark.parametrize(
    ("event_type", "event_kwargs", "expected_method"),
    [
        (
            StreamEventType.TOOL_START,
            {"tool_id": "t1", "tool_name": "Read", "tool_detail": "foo.py"},
            "on_tool_start",
        ),
        (
            StreamEventType.TOOL_DONE,
            {
                "tool_id": "t1",
                "tool_name": "Read",
                "tool_status": "done",
                "tool_duration_ms": 100,
            },
            "on_tool_done",
        ),
        (
            StreamEventType.STATE_CHANGE,
            {"text_preview": "thinking"},
            "on_agent_state_change",
        ),
        (
            StreamEventType.RESULT,
            {"cost_usd": Decimal("0.05")},
            "on_cost_update",
        ),
        (
            StreamEventType.STALL,
            {},
            "on_stall",
        ),
        (
            StreamEventType.RATE_LIMIT,
            {"retry_delay_ms": 5000},
            "on_rate_limit",
        ),
        (
            StreamEventType.RETRY,
            {"retry_attempt": 1, "retry_max": 3, "error_category": "overloaded"},
            "on_retry",
        ),
    ],
)
def test_cli_dispatch_routes_event(
    cli_callback: CLICallback,
    event_type: StreamEventType,
    event_kwargs: dict[str, object],
    expected_method: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock = MagicMock()
    monkeypatch.setattr(cli_callback, expected_method, mock)
    event = StreamEvent(event_type=event_type, **event_kwargs)
    cli_callback.dispatch_agent_event("test-agent", event)
    assert mock.called, f"{expected_method} was not called for {event_type}"


def test_cli_dispatch_ignores_text_events(cli_callback: CLICallback) -> None:
    """TEXT events should not trigger any typed callback."""
    event = StreamEvent(event_type=StreamEventType.TEXT, text_preview="hello")
    # Should not raise
    cli_callback.dispatch_agent_event("test-agent", event)


def test_cli_cost_tracks_per_agent(cli_callback: CLICallback) -> None:
    """Cost update for same agent should replace, not accumulate."""
    cli_callback.on_cost_update("claude", Decimal("0.10"))
    cli_callback.on_cost_update("claude", Decimal("0.15"))
    assert cli_callback._agent_costs["claude"] == Decimal("0.15")


def test_cli_cost_tracks_multiple_agents(cli_callback: CLICallback) -> None:
    """Cost updates for different agents are tracked independently."""
    cli_callback.on_cost_update("claude", Decimal("0.10"))
    cli_callback.on_cost_update("gemini", Decimal("0.20"))
    assert cli_callback._agent_costs["claude"] == Decimal("0.10")
    assert cli_callback._agent_costs["gemini"] == Decimal("0.20")
