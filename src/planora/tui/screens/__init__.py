"""Optional-friendly exports for Planora TUI screens."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = ["DashboardScreen", "ReportScreen", "WizardLaunch", "WizardScreen"]

_EXPORT_TO_MODULE = {
    "DashboardScreen": "planora.tui.screens.dashboard",
    "ReportScreen": "planora.tui.screens.report",
    "WizardLaunch": "planora.tui.screens.wizard",
    "WizardScreen": "planora.tui.screens.wizard",
}
_OPTIONAL_TUI_MESSAGE = (
    "Planora TUI requires the optional 'tui' dependencies. "
    "Install with: uv pip install 'planora[tui]'"
)
_OPTIONAL_TUI_IMPORTS = {"textual", "textual_speedups", "trogon"}

if TYPE_CHECKING:
    from .dashboard import DashboardScreen
    from .report import ReportScreen
    from .wizard import WizardLaunch, WizardScreen


def __getattr__(name: str) -> object:
    """Lazily import screens to avoid importing Textual unless needed."""
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
