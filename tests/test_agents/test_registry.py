from __future__ import annotations

import pytest

from planora.agents.registry import AgentMode, OutputExtraction


def test_registry_contains_expected_builtins(registry) -> None:
    assert sorted(registry.agents) == [
        "claude",
        "codex",
        "copilot",
        "gemini",
        "opencode-glm",
        "opencode-kimi",
        "opencode-minimax",
    ]


def test_registry_get_unknown_agent_raises_key_error(registry) -> None:
    with pytest.raises(KeyError, match="Agent 'missing' not found"):
        registry.get("missing")


def test_registry_available_filters_by_binary(monkeypatch, registry) -> None:
    binaries = {
        "claude": "/usr/bin/claude",
        "copilot": None,
        "codex": "/usr/bin/codex",
        "gemini": None,
        "opencode": "/usr/bin/opencode",
    }

    monkeypatch.setattr(
        "planora.agents.registry.shutil.which",
        lambda binary: binaries.get(binary),
    )

    assert registry.available() == [
        "claude",
        "codex",
        "opencode-kimi",
        "opencode-glm",
        "opencode-minimax",
    ]


def test_registry_validate_reports_missing_and_unknown(monkeypatch, registry) -> None:
    monkeypatch.setattr(
        "planora.agents.registry.shutil.which",
        lambda binary: None if binary in {"gemini", "opencode"} else f"/bin/{binary}",
    )

    assert registry.validate(["claude", "gemini", "unknown", "opencode-kimi"]) == [
        "gemini",
        "unknown",
        "opencode-kimi",
    ]


def test_claude_and_codex_configs_capture_special_behavior(registry) -> None:
    claude = registry.get("claude")
    codex = registry.get("codex")

    assert "--include-partial-messages" in claude.flags[AgentMode.PLAN]
    assert claude.output_extraction.strategy == OutputExtraction.Strategy.JQ_FILTER
    assert claude.output_extraction.strip_preamble is True

    assert "--json" in codex.flags[AgentMode.PLAN]
    assert codex.output_extraction.strategy == OutputExtraction.Strategy.DIRECT_FILE
