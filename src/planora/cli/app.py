from __future__ import annotations

import typer

app = typer.Typer(name="planora", help="Multi-agent implementation plan orchestrator")
plan_app = typer.Typer(name="plan", help="Implementation planning workflow")
agents_app = typer.Typer(name="agents", help="Agent management")

app.add_typer(plan_app, name="plan")
app.add_typer(agents_app, name="agents")


@app.command("tui")
def launch_tui() -> None:
    """Launch interactive TUI (auto-generated from CLI)."""
    try:
        from trogon import Trogon  # type: ignore[import-not-found]

        Trogon(app).run()
    except ImportError:
        from rich.console import Console

        Console().print("[yellow]TUI requires the 'tui' extra: uv pip install 'planora[tui]'[/]")
        raise typer.Exit(1) from None
