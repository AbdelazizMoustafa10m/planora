"""Optional Textual-based terminal user interface for Planora."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = ["PlanoraTUI", "TextualUICallback"]

_OPTIONAL_TUI_MESSAGE = (
    "Planora TUI requires the optional 'tui' dependencies. "
    "Install with: uv pip install 'planora[tui]'"
)
_OPTIONAL_TUI_IMPORTS = {"textual", "textual_speedups", "trogon"}

if TYPE_CHECKING:
    from .app import PlanoraTUI
    from .callbacks import TextualUICallback


def __getattr__(name: str) -> object:
    """Lazily import TUI exports so the package stays optional-friendly."""
    match name:
        case "PlanoraTUI":
            return _load_export("planora.tui.app", name)
        case "TextualUICallback":
            return _load_export("planora.tui.callbacks", name)
        case _:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
