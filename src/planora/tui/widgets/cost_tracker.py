"""Running cost breakdown widget for the Planora dashboard."""

from __future__ import annotations

from decimal import Decimal

from textual.widgets import Static


class CostTracker(Static):
    """Render per-agent cost totals and the pipeline aggregate."""

    def __init__(
        self,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        super().__init__("", name=name, id=id, classes=classes, disabled=disabled)
        self._agent_order: list[str] = []
        self._agent_costs: dict[str, Decimal] = {}
        self._total_cost = Decimal("0")
        self._rebuild()

    def set_agents(self, agents: list[str]) -> None:
        """Set the display order for known agents."""
        self._agent_order = list(dict.fromkeys(agents))
        self._rebuild()

    def reset(self) -> None:
        """Clear all recorded costs for a new run."""
        self._agent_costs.clear()
        self._total_cost = Decimal("0")
        self._rebuild()

    def record_cost(self, agent: str, cost_usd: Decimal) -> None:
        """Add a cost update for a single agent."""
        self._agent_costs[agent] = self._agent_costs.get(agent, Decimal("0")) + cost_usd
        self._total_cost = sum(self._agent_costs.values(), Decimal("0"))
        self._rebuild()

    def set_totals(self, agent_costs: dict[str, Decimal]) -> None:
        """Replace the full cost mapping."""
        self._agent_costs = dict(agent_costs)
        self._total_cost = sum(self._agent_costs.values(), Decimal("0"))
        self._rebuild()

    def _rebuild(self) -> None:
        agents = self._agent_order or list(self._agent_costs)

        lines = ["Per-Agent Cost"]
        if not agents:
            lines.append("  (no agents configured)")
        else:
            for agent in agents:
                if agent in self._agent_costs:
                    lines.append(f"  {agent:<10} ${self._agent_costs[agent]:.4f}")
                else:
                    lines.append(f"  {agent:<10} N/A")

        lines.append("")
        lines.append(f"Total: ${self._total_cost:.4f}")
        self.update("\n".join(lines))
