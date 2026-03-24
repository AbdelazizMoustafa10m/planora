from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from planora.agents.registry import AgentRegistry, StreamFormat
from planora.cli.app import agents_app

console = Console()

_STATUS_TIMEOUT_SECONDS = 10
_COPILOT_TOKEN_VARS = ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")
_OPENCODE_PROVIDER_KEYS = {
    "github-copilot": "github-copilot",
    "openai": "openai",
    "requesty": "requesty",
    "zai-coding-plan": "zai-coding-plan",
}

# Display label for each stream format in the table.
_STREAM_FORMAT_LABEL: dict[StreamFormat, str] = {
    StreamFormat.CLAUDE: "Claude JSONL",
    StreamFormat.COPILOT: "Copilot JSONL",
    StreamFormat.CODEX: "Codex JSONL",
    StreamFormat.GEMINI: "Gemini",
    StreamFormat.OPENCODE: "OpenCode JSONL",
}


@dataclass(frozen=True)
class AuthStatus:
    """Result of an agent auth readiness probe."""

    is_ready: bool
    detail: str


def _first_output_line(text: str) -> str:
    """Return the first non-empty line from a subprocess output string."""
    for line in text.splitlines():
        if stripped := line.strip():
            return stripped
    return ""


def _run_status_command(command: list[str]) -> subprocess.CompletedProcess[str] | AuthStatus:
    """Run a short-lived status command and return either the process or a failure."""
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=_STATUS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return AuthStatus(False, f"{' '.join(command)} timed out")
    except OSError as exc:
        return AuthStatus(False, f"{' '.join(command)} failed: {exc}")


def _check_claude_auth(binary: str) -> AuthStatus:
    """Require a real Claude account login rather than an API-key env var."""
    proc = _run_status_command([binary, "auth", "status"])
    if isinstance(proc, AuthStatus):
        return proc

    payload = proc.stdout.strip() or proc.stderr.strip()
    if proc.returncode != 0:
        reason = _first_output_line(payload) or f"exit code {proc.returncode}"
        return AuthStatus(False, reason)

    try:
        status = json.loads(payload)
    except json.JSONDecodeError:
        reason = _first_output_line(payload) or "unparseable auth status output"
        return AuthStatus(False, reason)

    if not isinstance(status, dict) or status.get("loggedIn") is not True:
        return AuthStatus(False, "not logged in")

    auth_method = str(status.get("authMethod") or "unknown")
    subscription = status.get("subscriptionType")
    email = status.get("email")
    detail_parts = [auth_method]
    if subscription:
        detail_parts.append(f"subscription={subscription}")
    if email:
        detail_parts.append(str(email))
    return AuthStatus(True, ", ".join(detail_parts))


def _check_codex_auth(binary: str) -> AuthStatus:
    """Require Codex CLI to be logged in with a ChatGPT account."""
    proc = _run_status_command([binary, "login", "status"])
    if isinstance(proc, AuthStatus):
        return proc

    output = (proc.stdout.strip() or proc.stderr.strip()).strip()
    if proc.returncode != 0:
        reason = _first_output_line(output) or f"exit code {proc.returncode}"
        return AuthStatus(False, reason)

    if not output:
        return AuthStatus(False, "login status returned no output")

    normalized = output.lower()
    if "api key" in normalized:
        return AuthStatus(False, output)
    if "chatgpt" in normalized or "logged in" in normalized:
        return AuthStatus(True, output)
    return AuthStatus(False, output)


def _copilot_config_dir() -> Path:
    """Return the resolved Copilot configuration directory."""
    if config_dir := os.environ.get("COPILOT_HOME"):
        return Path(config_dir).expanduser()
    return Path.home() / ".copilot"


def _check_copilot_auth() -> AuthStatus:
    """Accept either Copilot OAuth login state or a headless token for the CLI."""
    for env_var in _COPILOT_TOKEN_VARS:
        if os.environ.get(env_var):
            return AuthStatus(True, f"token via {env_var}")

    config_path = _copilot_config_dir() / "config.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return AuthStatus(False, f"{config_path} not found")
    except OSError as exc:
        return AuthStatus(False, f"failed reading {config_path}: {exc}")
    except json.JSONDecodeError:
        return AuthStatus(False, f"failed parsing {config_path}")

    if not isinstance(payload, dict):
        return AuthStatus(False, f"invalid config data in {config_path}")

    users: list[object] = []
    logged_in_users = payload.get("logged_in_users")
    if isinstance(logged_in_users, list):
        users.extend(logged_in_users)

    if last_user := payload.get("last_logged_in_user"):
        users.append(last_user)

    for user in users:
        if not isinstance(user, dict):
            continue
        login = user.get("login")
        host = user.get("host")
        if isinstance(login, str) and login:
            if isinstance(host, str) and host:
                return AuthStatus(True, f"stored login {login} @ {host}")
            return AuthStatus(True, f"stored login {login}")

    return AuthStatus(False, "no stored Copilot login found")


def _gemini_config_dir() -> Path:
    """Return the default Gemini CLI configuration directory."""
    return Path.home() / ".gemini"


def _read_json_file(path: Path) -> object:
    """Read and decode a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _check_gemini_auth() -> AuthStatus:
    """Require Gemini CLI to be configured for OAuth-backed personal auth."""
    config_dir = _gemini_config_dir()
    settings_path = config_dir / "settings.json"
    oauth_creds_path = config_dir / "oauth_creds.json"
    accounts_path = config_dir / "google_accounts.json"

    try:
        settings = _read_json_file(settings_path)
    except FileNotFoundError:
        return AuthStatus(False, f"{settings_path} not found")
    except OSError as exc:
        return AuthStatus(False, f"failed reading {settings_path}: {exc}")
    except json.JSONDecodeError:
        return AuthStatus(False, f"failed parsing {settings_path}")

    selected_type = ""
    if isinstance(settings, dict):
        security = settings.get("security")
        if isinstance(security, dict):
            auth = security.get("auth")
            if isinstance(auth, dict):
                raw_type = auth.get("selectedType")
                if isinstance(raw_type, str):
                    selected_type = raw_type

    if not selected_type.casefold().startswith("oauth"):
        if selected_type:
            return AuthStatus(False, f"configured for {selected_type}")
        return AuthStatus(False, "OAuth login not configured")

    if not oauth_creds_path.is_file():
        return AuthStatus(False, f"{oauth_creds_path} not found")

    email = ""
    try:
        accounts = _read_json_file(accounts_path)
    except FileNotFoundError:
        accounts = None
    except OSError:
        accounts = None
    except json.JSONDecodeError:
        accounts = None

    if isinstance(accounts, dict):
        active_account = accounts.get("active")
        if isinstance(active_account, str):
            email = active_account

    detail = selected_type
    if email:
        detail = f"{detail}, {email}"
    return AuthStatus(True, detail)


def _opencode_auth_file() -> Path:
    """Return the OpenCode auth file path."""
    return Path.home() / ".local" / "share" / "opencode" / "auth.json"


def _check_opencode_auth(model: str) -> AuthStatus:
    """Check provider credentials for OpenCode models without requiring env vars."""
    provider_prefix, _, _ = model.partition("/")
    if provider_prefix == "opencode":
        return AuthStatus(True, "built-in opencode free model")

    required_provider = _OPENCODE_PROVIDER_KEYS.get(provider_prefix)
    if required_provider is None:
        return AuthStatus(False, f"unknown provider prefix '{provider_prefix}'")

    auth_path = _opencode_auth_file()
    try:
        auth_data = _read_json_file(auth_path)
    except FileNotFoundError:
        return AuthStatus(False, f"{auth_path} not found")
    except OSError as exc:
        return AuthStatus(False, f"failed reading {auth_path}: {exc}")
    except json.JSONDecodeError:
        return AuthStatus(False, f"failed parsing {auth_path}")

    if not isinstance(auth_data, dict):
        return AuthStatus(False, f"invalid auth data in {auth_path}")

    if required_provider not in auth_data:
        return AuthStatus(False, f"provider credential missing: {required_provider}")

    provider_auth = auth_data[required_provider]
    auth_kind = ""
    if isinstance(provider_auth, dict):
        raw_type = provider_auth.get("type")
        if isinstance(raw_type, str):
            auth_kind = raw_type

    detail = required_provider
    if auth_kind:
        detail = f"{detail} ({auth_kind})"
    return AuthStatus(True, detail)


def _check_auth(name: str, binary: str, model: str) -> AuthStatus:
    """Return the auth readiness state for an agent."""
    if name == "claude":
        return _check_claude_auth(binary)
    if name == "codex":
        return _check_codex_auth(binary)
    if name == "copilot":
        return _check_copilot_auth()
    if name == "gemini":
        return _check_gemini_auth()
    if name.startswith("opencode-"):
        return _check_opencode_auth(model)
    return AuthStatus(True, "no auth check required")


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
                    console.print(f"  [green]✓[/green] Version: {version_output.splitlines()[0]}")
                else:
                    console.print("  [yellow]?[/yellow] Version: no output")
            except subprocess.TimeoutExpired:
                console.print("  [yellow]?[/yellow] Version: check timed out")
            except OSError:
                console.print("  [yellow]?[/yellow] Version: check failed")

        # 3. Auth validation
        auth_status = _check_auth(name, config.binary, config.model)
        if auth_status.is_ready:
            console.print(f"  [green]✓[/green] Auth: {auth_status.detail}")
        else:
            console.print(f"  [red]✗[/red] Auth: {auth_status.detail}")
            is_agent_ok = False

        if is_agent_ok:
            ready += 1

    total = len(names)
    summary_style = "green" if ready == total else "yellow" if ready > 0 else "red"
    console.print(f"\n[{summary_style}]{ready}/{total} agents ready[/{summary_style}]")

    if ready < total:
        raise typer.Exit(1)
