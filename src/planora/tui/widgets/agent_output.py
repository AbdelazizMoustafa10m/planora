"""Tabbed streaming agent output widget."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from textual.css.query import NoMatches
from textual.widgets import MarkdownViewer, Static, TabbedContent, TabPane

if TYPE_CHECKING:
    from textual.app import ComposeResult


class AgentOutputPanel(TabbedContent):
    """Tabbed markdown output streams for the active agents."""

    def __init__(
        self,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self._agent_order: list[str] = []
        self._buffers: dict[str, str] = {}

    def set_agents(self, agents: list[str]) -> None:
        """Configure the panel with the known agent set."""
        self._agent_order = list(dict.fromkeys(agents))
        self._buffers = {agent: self._buffers.get(agent, "") for agent in self._agent_order}
        self.refresh(recompose=True)

    def reset(self) -> None:
        """Clear all buffered agent output."""
        for agent in list(self._buffers):
            self._buffers[agent] = ""
        self.refresh(recompose=True)

    def append_output(self, agent: str, text: str) -> None:
        """Append a markdown chunk to an agent tab."""
        if not text.strip():
            return

        added_agent = False
        if agent not in self._agent_order:
            self._agent_order.append(agent)
            added_agent = True
        existing = self._buffers.get(agent, "")
        separator = "\n\n" if existing else ""
        self._buffers[agent] = f"{existing}{separator}{text}"

        if added_agent or not self.is_mounted:
            self.refresh(recompose=True)
        self.run_worker(
            self._sync_agent(agent),
            group=f"agent-output-{agent}",
            exclusive=True,
            exit_on_error=False,
        )

    def compose(self) -> ComposeResult:
        if not self._agent_order:
            yield TabPane("Output", Static("Waiting for agent output...", id="output-empty"))
            return

        for agent in self._agent_order:
            yield TabPane(
                agent,
                MarkdownViewer(
                    self._render_markdown(agent),
                    show_table_of_contents=False,
                    id=self._viewer_id(agent),
                ),
            )

    async def _sync_agent(self, agent: str) -> None:
        try:
            viewer = self.query_one(f"#{self._viewer_id(agent)}", MarkdownViewer)
        except NoMatches:
            return

        await viewer.document.update(self._render_markdown(agent))
        viewer.scroll_end(animate=False)

    def _render_markdown(self, agent: str) -> str:
        content = self._buffers.get(agent, "").strip()
        if not content:
            return f"# {agent}\n\n_Waiting for output..._"
        return f"# {agent}\n\n{content}"

    @staticmethod
    def _viewer_id(agent: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", agent.lower()).strip("-")
        return f"agent-output-{slug or 'agent'}"
