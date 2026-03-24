from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from planora.agents.registry import AgentRegistry, StreamFormat
from planora.cli.app import agents_app

console = Console()

# Per-agent auth env var candidates (any one being set satisfies the check).
_AUTH_ENV_VARS: dict[str, list[str]] = {
    "claude": ["ANTHROPIC_API_KEY"],
    "copilot": ["COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"],
    "codex": ["OPENAI_API_KEY"],
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "opencode-kimi": ["OPENAI_API_KEY"],
    "opencode-glm": ["OPENAI_API_KEY"],
    "opencode-minimax": ["OPENAI_API_KEY"],
}

# Display label for each stream format in the table.
_STREAM_FORMAT_LABEL: dict[StreamFormat, str] = {
    StreamFormat.CLAUDE: "Claude JSONL",
    StreamFormat.COPILOT: "Copilot JSONL",
    StreamFormat.CODEX: "Codex JSONL",
    StreamFormat.GEMINI: "Gemini",
    StreamFormat.OPENCODE: "OpenCode JSONL",
}


@agents_app.command("list")
def agents_list(
    output_format: Annotated[
        str,
        typer.Option("--format", help="Output format: table, json"),
    ] = "table",
) -> None:
    """Show registered agents and availability."""
    registry = AgentRegistry()

    if output_format == "json":
        result = [
            {
                "name": config.name,
                "binary": config.binary,
                "model": config.model,
                "stream_format": config.stream_format.value,
                "available": shutil.which(config.binary) is not None,
            }
            for config in registry.agents.values()
        ]
        console.print_json(json.dumps(result))
        return

    table = Table()
    table.add_column("Agent", style="cyan")
    table.add_column("Binary")
    table.add_column("Model")
    table.add_column("Stream Format")
    table.add_column("Available")

    for config in registry.agents.values():
        is_available = shutil.which(config.binary) is not None
        avail_str = "✓" if is_available else "✗ (not on PATH)"
        avail_style = "green" if is_available else "red"
        stream_label = _STREAM_FORMAT_LABEL.get(config.stream_format, config.stream_format.value)
        table.add_row(
            config.name,
            config.binary,
            config.model,
            stream_label,
            f"[{avail_style}]{avail_str}[/{avail_style}]",
        )

    console.print(table)


@agents_app.command("check")
def agents_check(
    agents: Annotated[
        str | None,
        typer.Option(help="Comma-separated agent names to check (default: all)"),
    ] = None,
) -> None:
    """Validate agent CLIs are installed and configured."""
    registry = AgentRegistry()

    names: list[str] = (
        [n.strip() for n in agents.split(",") if n.strip()]
        if agents
        else list(registry.agents.keys())
    )

    ready = 0

    for name in names:
        console.print(f"\n[bold cyan]{name}[/bold cyan]")

        try:
            config = registry.get(name)
        except KeyError:
            console.print(f"  [red]✗ Unknown agent: {name}[/red]")
            continue

        is_agent_ok = True

        # 1. Binary on PATH
        binary_path = shutil.which(config.binary)
        if binary_path:
            console.print(f"  [green]✓[/green] Binary: {binary_path}")
        else:
            console.print(f"  [red]✗[/red] Binary: {config.binary} not found on PATH")
            is_agent_ok = False

        # 2. Version check — only if binary is present
        if binary_path:
            try:
                proc = subprocess.run(
                    [config.binary, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                version_output = proc.stdout.strip() or proc.stderr.strip()
                if version_output:
                    console.print(
                        f"  [green]✓[/green] Version: {version_output.splitlines()[0]}"
                    )
                else:
                    console.print("  [yellow]?[/yellow] Version: no output")
            except subprocess.TimeoutExpired:
                console.print("  [yellow]?[/yellow] Version: check timed out")
            except OSError:
                console.print("  [yellow]?[/yellow] Version: check failed")

        # 3. Auth env var validation
        auth_vars = _AUTH_ENV_VARS.get(name, [])
        if auth_vars:
            set_vars = [v for v in auth_vars if os.environ.get(v)]
            if set_vars:
                console.print(f"  [green]✓[/green] Auth: {set_vars[0]} is set")
            else:
                expected = " or ".join(auth_vars)
                console.print(f"  [red]✗[/red] Auth: none of {expected} set")
                is_agent_ok = False

        if is_agent_ok:
            ready += 1

    total = len(names)
    summary_style = "green" if ready == total else "yellow" if ready > 0 else "red"
    console.print(f"\n[{summary_style}]{ready}/{total} agents ready[/{summary_style}]")

    if ready < total:
        raise typer.Exit(1)
