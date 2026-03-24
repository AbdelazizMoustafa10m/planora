"""Optional-friendly exports for Planora TUI widgets."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "AgentActivityPanel",
    "AgentOutputPanel",
    "CostTracker",
    "EventLog",
    "PipelineProgress",
    "StatusPanel",
]

_EXPORT_TO_MODULE = {
    "AgentActivityPanel": "planora.tui.widgets.agent_activity",
    "AgentOutputPanel": "planora.tui.widgets.agent_output",
    "CostTracker": "planora.tui.widgets.cost_tracker",
    "EventLog": "planora.tui.widgets.event_log",
    "PipelineProgress": "planora.tui.widgets.pipeline",
    "StatusPanel": "planora.tui.widgets.status_panel",
}
_OPTIONAL_TUI_MESSAGE = (
    "Planora TUI requires the optional 'tui' dependencies. "
    "Install with: uv pip install 'planora[tui]'"
)
_OPTIONAL_TUI_IMPORTS = {"textual", "textual_speedups", "trogon"}

if TYPE_CHECKING:
    from .agent_activity import AgentActivityPanel
    from .agent_output import AgentOutputPanel
    from .cost_tracker import CostTracker
    from .event_log import EventLog
    from .pipeline import PipelineProgress
    from .status_panel import StatusPanel


def __getattr__(name: str) -> object:
    """Lazily import widgets to avoid hard dependency on Textual."""
    module_name = _EXPORT_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return _load_export(module_name, name)


def __dir__() -> list[str]:
    """Return the module attributes exposed by this package."""
    return sorted(set(globals()) | set(__all__))


def _load_export(module_name: str, export_name: str) -> object:
    try:
        module = import_module(module_name)
    except ImportError as exc:
        if _is_optional_tui_import(exc):
            raise ImportError(_OPTIONAL_TUI_MESSAGE) from exc
        raise
    exported = getattr(module, export_name)
    globals()[export_name] = exported
    return exported


def _is_optional_tui_import(exc: ImportError) -> bool:
    module_name = exc.name or ""
    root_module = module_name.partition(".")[0]
    return root_module in _OPTIONAL_TUI_IMPORTS
