"""Scrollable event log widget for the TUI dashboard."""

from __future__ import annotations

from datetime import datetime, timedelta

from rich.text import Text
from textual.widgets import RichLog

_LEVEL_STYLES = {
    "info": "white",
    "warning": "yellow",
    "error": "bold red",
    "debug": "dim",
}
_AGENT_PALETTE = ("cyan", "magenta", "green", "blue", "yellow", "bright_white")


class EventLog(RichLog):
    """Rich, persistent event log with agent-aware formatting."""

    def __init__(
        self,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        super().__init__(
            max_lines=2_000,
            min_width=80,
            wrap=True,
            highlight=False,
            markup=False,
            auto_scroll=True,
            name=name,
            id=id,
            classes=classes,
            disabled=disabled,
        )

    def append_entry(
        self,
        *,
        message: str,
        timestamp: datetime | None = None,
        level: str = "info",
        agent: str | None = None,
        icon: str | None = None,
        detail: str | None = None,
        duration: timedelta | float | None = None,
    ) -> None:
        """Append a formatted log line to the widget."""
        when = timestamp or datetime.now()
        line = Text()
        line.append(when.strftime("%H:%M:%S"), style="dim")
        line.append("  ")

        if agent is not None:
            line.append(f"{agent:<10}", style=_agent_style(agent))
        else:
            line.append("system     ", style="bold")

        line.append("  ")

        if icon:
            line.append(f"{icon} ", style=_LEVEL_STYLES.get(level, "white"))

        line.append(message, style=_LEVEL_STYLES.get(level, "white"))

        if detail:
            line.append(f" {detail}", style="dim")

        if duration is not None:
            line.append(f" ({_format_duration(duration)})", style="dim")

        self.write(line)

    def reset(self) -> None:
        """Clear the existing event history."""
        self.clear()


def _agent_style(agent: str) -> str:
    index = sum(ord(char) for char in agent) % len(_AGENT_PALETTE)
    return _AGENT_PALETTE[index]


def _format_duration(duration: timedelta | float) -> str:
    seconds = duration.total_seconds() if isinstance(duration, timedelta) else duration
    if seconds >= 60:
        minutes, remainder = divmod(int(seconds), 60)
        return f"{minutes}m {remainder:02d}s"
    return f"{seconds:.1f}s"
