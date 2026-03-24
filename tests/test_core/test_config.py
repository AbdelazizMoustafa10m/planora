from __future__ import annotations

from pathlib import Path

import pytest

from planora.core.config import (
    AgentOverrideConfig,
    DefaultsConfig,
    PlanораSettings,
    ProfileConfig,
    _deep_merge,
    _parse_toml_value,
    _set_nested,
)


def test_parse_toml_value_string() -> None:
    assert _parse_toml_value("gemini") == "gemini"


def test_parse_toml_value_int() -> None:
    assert _parse_toml_value("42") == 42


def test_parse_toml_value_float() -> None:
    assert _parse_toml_value("300.0") == 300.0


def test_parse_toml_value_bool_true() -> None:
    assert _parse_toml_value("true") is True


def test_parse_toml_value_array() -> None:
    assert _parse_toml_value('["gemini", "codex"]') == ["gemini", "codex"]


def test_set_nested_builds_intermediate_tables() -> None:
    data: dict[str, object] = {}

    _set_nested(data, ["defaults", "planner"], "gemini")

    assert data == {"defaults": {"planner": "gemini"}}


def test_deep_merge_non_overlapping() -> None:
    base = {"a": 1, "b": {"x": 10}}
    override = {"c": 3}

    assert _deep_merge(base, override) == {"a": 1, "b": {"x": 10}, "c": 3}


def test_deep_merge_nested_override() -> None:
    base = {"agents": {"claude": {"model": "old", "timeout": 300}}}
    override = {"agents": {"claude": {"model": "new"}}}

    result = _deep_merge(base, override)

    assert result["agents"]["claude"] == {"model": "new", "timeout": 300}


def test_model_post_init_backfills_flat_fields_from_nested_config() -> None:
    settings = PlanораSettings(
        _env_file=None,
        defaults=DefaultsConfig(
            planner="gemini",
            auditors=["codex"],
            concurrency=5,
            audit_rounds=2,
            project_root=Path("workspace"),
            reports_dir="custom-reports",
        ),
        agents={"claude": AgentOverrideConfig(model="claude-haiku-4-5")},
    )

    assert settings.default_planner == "gemini"
    assert settings.default_auditors == ["codex"]
    assert settings.default_concurrency == 5
    assert settings.default_audit_rounds == 2
    assert settings.project_root == Path("workspace")
    assert settings.reports_dir == "custom-reports"
    assert settings.claude_model == "claude-haiku-4-5"


def test_with_config_overrides_updates_nested_and_flat_values() -> None:
    settings = PlanораSettings(_env_file=None)

    updated = settings.with_config_overrides(
        [
            "defaults.planner=gemini",
            'defaults.auditors=["codex"]',
            "agents.claude.model='claude-sonnet'",
        ]
    )

    assert updated.defaults.planner == "gemini"
    assert updated.default_planner == "gemini"
    assert updated.defaults.auditors == ["codex"]
    assert updated.default_auditors == ["codex"]
    assert updated.agents["claude"].model == "claude-sonnet"
    assert updated.claude_model == "claude-sonnet"


def test_with_config_overrides_invalid_format_raises() -> None:
    settings = PlanораSettings(_env_file=None)

    with pytest.raises(ValueError, match="Expected format"):
        settings.with_config_overrides(["no-equals-sign"])


def test_with_config_overrides_unknown_key_raises() -> None:
    settings = PlanораSettings(_env_file=None)

    with pytest.raises(ValueError, match="Unknown config override key"):
        settings.with_config_overrides(["defaults.unknown_key=1"])


def test_with_profile_unknown_raises() -> None:
    settings = PlanораSettings(_env_file=None)

    with pytest.raises(ValueError, match="Profile 'missing' not found"):
        settings.with_profile("missing")


def test_with_profile_applies_defaults_and_agent_overrides() -> None:
    settings = PlanораSettings(
        _env_file=None,
        profiles={
            "fast": ProfileConfig(
                planner="gemini",
                auditors=["codex"],
                concurrency=2,
                audit_rounds=2,
                agents={"claude": AgentOverrideConfig(model="claude-haiku-4-5")},
            )
        },
    )

    updated = settings.with_profile("fast")

    assert updated.defaults.planner == "gemini"
    assert updated.default_planner == "gemini"
    assert updated.defaults.auditors == ["codex"]
    assert updated.default_auditors == ["codex"]
    assert updated.defaults.concurrency == 2
    assert updated.default_concurrency == 2
    assert updated.defaults.audit_rounds == 2
    assert updated.default_audit_rounds == 2
    assert updated.agents["claude"].model == "claude-haiku-4-5"
    assert updated.claude_model == "claude-haiku-4-5"


def test_effective_properties_reflect_flat_compatibility_fields() -> None:
    settings = PlanораSettings(
        _env_file=None,
        defaults=DefaultsConfig(
            planner="gemini",
            auditors=["copilot"],
            concurrency=4,
            audit_rounds=2,
            project_root=Path("relative-root"),
        ),
    )

    assert settings.effective_planner == "gemini"
    assert settings.effective_auditors == ["copilot"]
    assert settings.effective_concurrency == 4
    assert settings.effective_audit_rounds == 2
    assert settings.effective_project_root.is_absolute() is True
