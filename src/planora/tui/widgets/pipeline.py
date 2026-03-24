"""Pipeline progress widget for the execution dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.text import Text
from textual.widgets import Static

from planora.core.events import PhaseStatus

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True, slots=True)
class _PhaseSpec:
    key: str
    label: str


class PipelineProgress(Static):
    """Horizontal phase indicator for the current workflow run."""

    def __init__(
        self,
        *,
        audit_rounds: int = 1,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        super().__init__("", name=name, id=id, classes=classes, disabled=disabled)
        self._phase_order: list[_PhaseSpec] = []
        self._statuses: dict[str, PhaseStatus] = {}
        self.configure(audit_rounds=audit_rounds)

    def configure(self, *, audit_rounds: int) -> None:
        """Set the canonical phase order for the active run."""
        phases: list[_PhaseSpec] = [_PhaseSpec("plan", "Plan")]
        for round_index in range(1, audit_rounds + 1):
            audit_key = "audit" if round_index == 1 else f"audit-r{round_index}"
            refine_key = "refine" if round_index == 1 else f"refine-r{round_index}"
            phases.append(_PhaseSpec(audit_key, f"Audit R{round_index}"))
            phases.append(_PhaseSpec(refine_key, f"Refine R{round_index}"))
        phases.append(_PhaseSpec("report", "Report"))

        self._phase_order = phases
        self.reset()

    def set_phase_order(self, phases: Iterable[tuple[str, str]]) -> None:
        """Override the rendered phase order with explicit key/label pairs."""
        self._phase_order = [_PhaseSpec(key, label) for key, label in phases]
        self.reset()

    def reset(self) -> None:
        """Reset all configured phases to pending."""
        self._statuses = {}
        self._refresh_display()

    def update_statuses(self, statuses: dict[str, PhaseStatus]) -> None:
        """Merge a batch of phase statuses into the widget state."""
        self._statuses.update(statuses)
        self._refresh_display()

    def set_phase_status(self, phase: str, status: PhaseStatus) -> None:
        """Update a single phase status."""
        self._statuses[phase] = status
        self._refresh_display()

    def _refresh_display(self) -> None:
        self.update(self._render_pipeline())

    def _render_pipeline(self) -> Text:
        if not self._phase_order:
            return Text("Pipeline unavailable", style="dim")

        rendered = Text()
        for index, phase in enumerate(self._phase_order):
            status = self._statuses.get(phase.key, PhaseStatus.PENDING)
            symbol, style = _phase_style(status)

            if index:
                rendered.append(" ─── ", style="dim")

            rendered.append(f"{symbol} ", style=style)
            rendered.append(phase.label, style=style if status != PhaseStatus.PENDING else "dim")
        return rendered


def _phase_style(status: PhaseStatus) -> tuple[str, str]:
    match status:
        case PhaseStatus.RUNNING:
            return "●", "bold yellow"
        case PhaseStatus.DONE:
            return "●", "bold green"
        case PhaseStatus.SKIPPED:
            return "◌", "dim"
        case PhaseStatus.FAILED:
            return "✖", "bold red"
        case PhaseStatus.PENDING:
            return "◯", "dim"
