from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from planora.cli.agents import (
    AuthStatus,
    _check_claude_auth,
    _check_codex_auth,
    _check_copilot_auth,
    _check_gemini_auth,
    _check_opencode_auth,
    _first_output_line,
    _run_status_command,
)
from planora.cli.app import app

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", ""),
        ("  \n  \n", ""),
        ("hello\nworld", "hello"),
        ("\n  first non-empty  \n", "first non-empty"),
    ],
)
def test_first_output_line_returns_first_non_empty(text: str, expected: str) -> None:
    assert _first_output_line(text) == expected


# ---------------------------------------------------------------------------
# _run_status_command
# ---------------------------------------------------------------------------


def test_run_status_command_returns_completed_process_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_proc = subprocess.CompletedProcess(
        args=["echo", "hi"], returncode=0, stdout="hi\n", stderr=""
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: fake_proc)

    result = _run_status_command(["echo", "hi"])

    assert isinstance(result, subprocess.CompletedProcess)
    assert result.returncode == 0


def test_run_status_command_returns_auth_status_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["cmd"], timeout=10)

    monkeypatch.setattr(subprocess, "run", raise_timeout)

    result = _run_status_command(["cmd"])

    assert isinstance(result, AuthStatus)
    assert result.is_ready is False
    assert "timed out" in result.detail


def test_run_status_command_returns_auth_status_on_os_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_os_error(*args, **kwargs):
        raise OSError("no such file")

    monkeypatch.setattr(subprocess, "run", raise_os_error)

    result = _run_status_command(["missing-cmd"])

    assert isinstance(result, AuthStatus)
    assert result.is_ready is False


# ---------------------------------------------------------------------------
# _check_claude_auth
# ---------------------------------------------------------------------------


def test_check_claude_auth_returns_ready_for_valid_logged_in_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.dumps(
        {"loggedIn": True, "authMethod": "oauth", "email": "user@example.com"}
    )
    fake_proc = subprocess.CompletedProcess(
        args=["claude", "auth", "status"], returncode=0, stdout=payload, stderr=""
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: fake_proc)

    status = _check_claude_auth("claude")

    assert status.is_ready is True
    assert "oauth" in status.detail
    assert "user@example.com" in status.detail


def test_check_claude_auth_returns_not_ready_when_not_logged_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.dumps({"loggedIn": False})
    fake_proc = subprocess.CompletedProcess(
        args=["claude", "auth", "status"], returncode=0, stdout=payload, stderr=""
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: fake_proc)

    status = _check_claude_auth("claude")

    assert status.is_ready is False
    assert "not logged in" in status.detail


def test_check_claude_auth_returns_not_ready_on_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_proc = subprocess.CompletedProcess(
        args=["claude", "auth", "status"], returncode=1, stdout="", stderr="auth failed"
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: fake_proc)

    status = _check_claude_auth("claude")

    assert status.is_ready is False


def test_check_claude_auth_returns_not_ready_for_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_proc = subprocess.CompletedProcess(
        args=["claude", "auth", "status"],
        returncode=0,
        stdout="not-json",
        stderr="",
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: fake_proc)

    status = _check_claude_auth("claude")

    assert status.is_ready is False


# ---------------------------------------------------------------------------
# _check_codex_auth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("output", "is_ready"),
    [
        ("Logged in as ChatGPT user", True),
        ("logged in successfully", True),
        ("Not logged in; use API key", False),
        ("api key configured", False),
        ("", False),  # empty output
    ],
)
def test_check_codex_auth_various_outputs(
    output: str, is_ready: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_proc = subprocess.CompletedProcess(
        args=["codex", "login", "status"],
        returncode=0,
        stdout=output,
        stderr="",
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: fake_proc)

    status = _check_codex_auth("codex")

    assert status.is_ready is is_ready


# ---------------------------------------------------------------------------
# _check_copilot_auth
# ---------------------------------------------------------------------------


def test_check_copilot_auth_ready_when_env_token_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "ghp_token123")

    status = _check_copilot_auth()

    assert status.is_ready is True
    assert "COPILOT_GITHUB_TOKEN" in status.detail


def test_check_copilot_auth_ready_when_config_file_has_stored_login(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / ".copilot"
    config_dir.mkdir()
    config = {
        "logged_in_users": [{"login": "octocat", "host": "github.com"}],
    }
    (config_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setenv("COPILOT_HOME", str(config_dir))
    for var in ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
        monkeypatch.delenv(var, raising=False)

    status = _check_copilot_auth()

    assert status.is_ready is True
    assert "octocat" in status.detail


def test_check_copilot_auth_not_ready_when_config_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty_dir = tmp_path / ".copilot"
    empty_dir.mkdir()
    monkeypatch.setenv("COPILOT_HOME", str(empty_dir))
    for var in ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
        monkeypatch.delenv(var, raising=False)

    status = _check_copilot_auth()

    assert status.is_ready is False
    assert "not found" in status.detail


# ---------------------------------------------------------------------------
# _check_gemini_auth
# ---------------------------------------------------------------------------


def test_check_gemini_auth_ready_with_oauth_settings_and_creds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gemini_dir = tmp_path / ".gemini"
    gemini_dir.mkdir()
    settings = {
        "security": {"auth": {"selectedType": "oauth-personal"}}
    }
    (gemini_dir / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    (gemini_dir / "oauth_creds.json").write_text("{}", encoding="utf-8")
    accounts = {"active": "user@example.com"}
    (gemini_dir / "google_accounts.json").write_text(json.dumps(accounts), encoding="utf-8")

    monkeypatch.setattr(
        "planora.cli.agents._gemini_config_dir", lambda: gemini_dir
    )

    status = _check_gemini_auth()

    assert status.is_ready is True
    assert "user@example.com" in status.detail


def test_check_gemini_auth_not_ready_when_settings_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty_dir = tmp_path / ".gemini"
    empty_dir.mkdir()
    monkeypatch.setattr(
        "planora.cli.agents._gemini_config_dir", lambda: empty_dir
    )

    status = _check_gemini_auth()

    assert status.is_ready is False


def test_check_gemini_auth_not_ready_when_auth_type_is_not_oauth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gemini_dir = tmp_path / ".gemini"
    gemini_dir.mkdir()
    settings = {"security": {"auth": {"selectedType": "api-key"}}}
    (gemini_dir / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    monkeypatch.setattr(
        "planora.cli.agents._gemini_config_dir", lambda: gemini_dir
    )

    status = _check_gemini_auth()

    assert status.is_ready is False
    assert "api-key" in status.detail


# ---------------------------------------------------------------------------
# _check_opencode_auth
# ---------------------------------------------------------------------------


def test_check_opencode_auth_built_in_model_is_always_ready() -> None:
    status = _check_opencode_auth("opencode/free-model")

    assert status.is_ready is True
    assert "built-in" in status.detail


def test_check_opencode_auth_unknown_provider_prefix_returns_not_ready() -> None:
    status = _check_opencode_auth("unknownprovider/model")

    assert status.is_ready is False
    assert "unknown provider prefix" in status.detail


def test_check_opencode_auth_ready_when_provider_credential_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(
        json.dumps({"openai": {"type": "api-key", "key": "sk-test"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "planora.cli.agents._opencode_auth_file", lambda: auth_file
    )

    status = _check_opencode_auth("openai/gpt-4o")

    assert status.is_ready is True
    assert "openai" in status.detail


def test_check_opencode_auth_not_ready_when_provider_missing_from_auth_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({"some-other": {}}), encoding="utf-8")
    monkeypatch.setattr(
        "planora.cli.agents._opencode_auth_file", lambda: auth_file
    )

    status = _check_opencode_auth("openai/gpt-4o")

    assert status.is_ready is False
    assert "missing" in status.detail


# ---------------------------------------------------------------------------
# agents list — CLI command
# ---------------------------------------------------------------------------


def test_agents_list_table_mode_contains_known_agent_names() -> None:
    result = runner.invoke(app, ["agents", "list"])

    assert result.exit_code == 0
    assert "claude" in result.output


def test_agents_list_json_format_outputs_valid_json() -> None:
    result = runner.invoke(app, ["agents", "list", "--format", "json"])

    assert result.exit_code == 0
    # Strip Rich markup if any
    clean = result.output.strip()
    # Find the first '[' to locate JSON
    json_start = clean.find("[")

    agents = json.loads(clean[json_start:])

    assert isinstance(agents, list)
    assert len(agents) > 0
    assert all("name" in a for a in agents)
    assert all("binary" in a for a in agents)
    assert all("available" in a for a in agents)


# ---------------------------------------------------------------------------
# agents check — CLI command
# ---------------------------------------------------------------------------


def test_agents_check_exits_1_when_binary_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("shutil.which", lambda _: None)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: None)

    result = runner.invoke(app, ["agents", "check", "--agents", "claude"])

    assert result.exit_code == 1


def test_agents_check_exits_1_for_unknown_agent() -> None:
    result = runner.invoke(app, ["agents", "check", "--agents", "unknown-xyz-agent"])

    assert result.exit_code == 1
    assert "Unknown agent" in result.output

