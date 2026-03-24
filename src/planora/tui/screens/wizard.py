"""Interactive launch wizard for the Planora TUI."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Input, Label, Select, Static, TextArea

if TYPE_CHECKING:
    from textual.app import ComposeResult


class WizardLaunch(TypedDict):
    """Collected launch configuration returned by the wizard."""

    task: str
    planner: str
    auditors: list[str]
    audit_rounds: int
    max_concurrency: int


class WizardScreen(Screen[WizardLaunch | None]):
    """Collect task and workflow settings before launching the dashboard."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(
        self,
        *,
        default_task: str = "",
        planner: str = "claude",
        auditors: list[str] | None = None,
        audit_rounds: int = 1,
        max_concurrency: int = 3,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._default_task = default_task
        self._planner = planner
        self._auditors = auditors if auditors is not None else ["gemini", "codex"]
        self._audit_rounds = audit_rounds
        self._max_concurrency = max_concurrency

    def compose(self) -> ComposeResult:
        with Vertical(id="wizard-container"):
            yield Label("Planora Run Setup", id="wizard-title")
            yield Static(
                "Configure the task, planner, auditors, and concurrency for this run.",
                id="wizard-subtitle",
            )

            yield Label("Task")
            yield TextArea(
                self._default_task,
                id="task-input",
                soft_wrap=True,
                show_line_numbers=False,
                highlight_cursor_line=False,
                placeholder="Describe the implementation task...",
            )

            yield Label("Planner")
            yield Select(
                [(name, name) for name in ("claude", "codex", "copilot", "gemini")],
                value=self._planner,
                allow_blank=False,
                id="planner-select",
            )

            yield Label("Auditors (comma-separated)")
            yield Input(
                value=", ".join(self._auditors),
                placeholder="gemini, codex",
                id="auditors-input",
            )

            with Horizontal(id="wizard-select-row"):
                with Vertical(classes="wizard-select-group"):
                    yield Label("Audit Rounds")
                    yield Select(
                        [("1 round", 1), ("2 rounds", 2)],
                        value=self._audit_rounds,
                        allow_blank=False,
                        id="rounds-select",
                    )
                with Vertical(classes="wizard-select-group"):
                    yield Label("Concurrency")
                    yield Select(
                        [(str(value), value) for value in range(1, 5)],
                        value=self._max_concurrency,
                        allow_blank=False,
                        id="concurrency-select",
                    )

            yield Static("", id="wizard-error")

            with Horizontal(id="wizard-actions"):
                yield Button("Launch Pipeline", variant="primary", id="launch-btn")
                yield Button("Cancel", variant="default", id="cancel-btn")

    def on_mount(self) -> None:
        self.query_one("#task-input", TextArea).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "launch-btn":
            self._launch()
        elif event.button.id == "cancel-btn":
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _launch(self) -> None:
        launch = self._collect_launch_config()
        if launch is None:
            return
        self.dismiss(launch)

    def _collect_launch_config(self) -> WizardLaunch | None:
        task = self.query_one("#task-input", TextArea).text.strip()
        planner = self.query_one("#planner-select", Select).value
        auditors_text = self.query_one("#auditors-input", Input).value
        rounds = self.query_one("#rounds-select", Select).value
        concurrency = self.query_one("#concurrency-select", Select).value

        if not task:
            self._set_error("A task description is required.")
            return None
        if not isinstance(planner, str):
            self._set_error("Select a planner agent.")
            return None
        if not isinstance(rounds, int):
            self._set_error("Select the number of audit rounds.")
            return None
        if not isinstance(concurrency, int):
            self._set_error("Select the max concurrency.")
            return None

        self._set_error("")
        return {
            "task": task,
            "planner": planner,
            "auditors": _parse_auditors(auditors_text),
            "audit_rounds": rounds,
            "max_concurrency": concurrency,
        }

    def _set_error(self, message: str) -> None:
        self.query_one("#wizard-error", Static).update(message)


def _parse_auditors(raw_value: str) -> list[str]:
    seen: set[str] = set()
    auditors: list[str] = []

    for value in raw_value.split(","):
        agent = value.strip()
        if agent and agent not in seen:
            seen.add(agent)
            auditors.append(agent)
    return auditors
