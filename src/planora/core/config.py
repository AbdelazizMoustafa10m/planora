from __future__ import annotations

import tomllib
from decimal import Decimal  # noqa: TCH003 - Pydantic needs at runtime
from pathlib import Path
from typing import Any, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    PyprojectTomlConfigSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

_DEFAULT_PLANNER = "claude"
_DEFAULT_AUDITORS = ["gemini", "codex"]
_DEFAULT_CONCURRENCY = 3
_DEFAULT_AUDIT_ROUNDS = 1
_DEFAULT_PROJECT_ROOT = Path(".")
_DEFAULT_REPORTS_DIR = "reports"
_DEFAULT_STALL_TIMEOUT = 300.0
_DEFAULT_DEEP_TOOL_TIMEOUT = 600.0
_DEFAULT_MONITOR_INTERVAL = 5.0
_DEFAULT_CLI_STATUS_INTERVAL = 5.0
_DEFAULT_TELEMETRY_PROTOCOL = "grpc"
_DEFAULT_TELEMETRY_SERVICE_NAME = "planora"
_DEFAULT_AGENT_MODELS = {
    "claude": "claude-opus-4-6",
    "copilot": "claude-sonnet-4.5",
    "codex": "gpt-5.4",
    "gemini": "gemini-3.1-pro-preview",
}
_LEGACY_FILE_KEY_PATHS: dict[str, tuple[str, ...]] = {
    "default_planner": ("defaults", "planner"),
    "default_auditors": ("defaults", "auditors"),
    "default_concurrency": ("defaults", "concurrency"),
    "default_audit_rounds": ("defaults", "audit_rounds"),
    "project_root": ("defaults", "project_root"),
    "reports_dir": ("defaults", "reports_dir"),
    "claude_model": ("agents", "claude", "model"),
    "copilot_model": ("agents", "copilot", "model"),
    "codex_model": ("agents", "codex", "model"),
    "gemini_model": ("agents", "gemini", "model"),
    "stall_timeout": ("observability", "stall_timeout"),
    "deep_tool_timeout": ("observability", "deep_tool_timeout"),
    "monitor_check_interval": ("observability", "monitor_interval"),
    "telemetry_enabled": ("telemetry", "enabled"),
    "telemetry_otlp_endpoint": ("telemetry", "otlp_endpoint"),
    "telemetry_otlp_protocol": ("telemetry", "otlp_protocol"),
    "telemetry_service_name": ("telemetry", "service_name"),
    "telemetry_log_prompts": ("telemetry", "log_prompts"),
}


def _csv_string_to_list(value: str) -> list[str]:
    """Split a comma-separated string into a de-duplicated list."""
    seen: set[str] = set()
    result: list[str] = []
    for item in value.split(","):
        stripped = item.strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            result.append(stripped)
    return result


def _coerce_auditor_list(value: Any) -> Any:
    """Accept either TOML/JSON arrays or legacy comma-separated strings."""
    if isinstance(value, str):
        return _csv_string_to_list(value)
    return value


def _is_string_sequence_field(field: Any) -> bool:
    """Return True when the field is a list[str]-like collection."""
    if field is None:
        return False

    origin = get_origin(field.annotation)
    if origin not in {list, tuple, set, frozenset}:
        return False

    args = [arg for arg in get_args(field.annotation) if arg is not Ellipsis]
    return bool(args) and all(arg is str for arg in args)


def _project_config_path() -> Path | None:
    path = Path.cwd() / "planora.toml"
    return path.resolve() if path.is_file() else None


def _user_config_path() -> Path | None:
    path = Path.home() / ".config" / "planora" / "config.toml"
    return path.resolve() if path.is_file() else None


def _pyproject_config_path() -> Path | None:
    path = Path.cwd() / "pyproject.toml"
    return path.resolve() if path.is_file() else None


def _effective_config_base_dir() -> Path:
    for candidate in (_project_config_path(), _user_config_path(), _pyproject_config_path()):
        if candidate is not None:
            return candidate.parent
    return Path.cwd()


def _path_exists(data: Any, keys: list[str]) -> bool:
    """Return True if a nested dot-notation path exists inside the data tree."""
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    return True


def _set_nested(d: dict[str, Any], keys: list[str], value: Any) -> None:
    """Set a value in a nested dict using a sequence of keys."""
    current = d
    for key in keys[:-1]:
        existing = current.get(key)
        if existing is None:
            existing = {}
            current[key] = existing
        if not isinstance(existing, dict):
            raise ValueError(
                f"Cannot apply nested override at '{'.'.join(keys)}': "
                f"'{key}' already holds a non-table value."
            )
        current = existing
    current[keys[-1]] = value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base without mutating either input."""
    result = dict(base)
    for key, value in override.items():
        base_value = result.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            result[key] = _deep_merge(base_value, value)
        else:
            result[key] = value
    return result


def _parse_toml_value(raw: str) -> Any:
    """Parse a CLI override value as TOML, falling back to a bare string."""
    try:
        return tomllib.loads(f"value = {raw}")["value"]
    except tomllib.TOMLDecodeError:
        return raw


def _normalize_legacy_file_data(data: dict[str, Any]) -> dict[str, Any]:
    """Map legacy flat TOML keys into the nested Phase 4 layout."""
    normalized = dict(data)
    for legacy_key, nested_path in _LEGACY_FILE_KEY_PATHS.items():
        if legacy_key not in normalized:
            continue
        value = normalized.pop(legacy_key)
        if _path_exists(normalized, list(nested_path)):
            continue
        _set_nested(normalized, list(nested_path), value)
    return normalized


def _sync_flat_patch_fields(patch: dict[str, Any]) -> dict[str, Any]:
    """Mirror nested override patches into legacy flat compatibility keys."""
    synced = dict(patch)

    defaults_patch = synced.get("defaults")
    if isinstance(defaults_patch, dict):
        if "planner" in defaults_patch:
            synced["default_planner"] = defaults_patch["planner"]
        if "auditors" in defaults_patch:
            synced["default_auditors"] = defaults_patch["auditors"]
        if "concurrency" in defaults_patch:
            synced["default_concurrency"] = defaults_patch["concurrency"]
        if "audit_rounds" in defaults_patch:
            synced["default_audit_rounds"] = defaults_patch["audit_rounds"]
        if "project_root" in defaults_patch:
            synced["project_root"] = defaults_patch["project_root"]
        if "reports_dir" in defaults_patch:
            synced["reports_dir"] = defaults_patch["reports_dir"]

    observability_patch = synced.get("observability")
    if isinstance(observability_patch, dict):
        if "stall_timeout" in observability_patch:
            synced["stall_timeout"] = observability_patch["stall_timeout"]
        if "deep_tool_timeout" in observability_patch:
            synced["deep_tool_timeout"] = observability_patch["deep_tool_timeout"]
        if "monitor_interval" in observability_patch:
            synced["monitor_check_interval"] = observability_patch["monitor_interval"]

    telemetry_patch = synced.get("telemetry")
    if isinstance(telemetry_patch, dict):
        if "enabled" in telemetry_patch:
            synced["telemetry_enabled"] = telemetry_patch["enabled"]
        if "otlp_endpoint" in telemetry_patch:
            synced["telemetry_otlp_endpoint"] = telemetry_patch["otlp_endpoint"]
        if "otlp_protocol" in telemetry_patch:
            synced["telemetry_otlp_protocol"] = telemetry_patch["otlp_protocol"]
        if "service_name" in telemetry_patch:
            synced["telemetry_service_name"] = telemetry_patch["service_name"]
        if "log_prompts" in telemetry_patch:
            synced["telemetry_log_prompts"] = telemetry_patch["log_prompts"]

    agents_patch = synced.get("agents")
    if isinstance(agents_patch, dict):
        for agent_name in ("claude", "copilot", "codex", "gemini"):
            agent_patch = agents_patch.get(agent_name)
            if isinstance(agent_patch, dict) and "model" in agent_patch:
                synced[f"{agent_name}_model"] = agent_patch["model"]

    return synced


class _PlanoraEnvSettingsSource(EnvSettingsSource):
    """Environment source with CSV fallback for legacy string-list vars."""

    def decode_complex_value(self, field_name: str, field: Any, value: Any) -> Any:
        try:
            return super().decode_complex_value(field_name, field, value)
        except ValueError:
            if isinstance(value, str) and _is_string_sequence_field(field):
                return _csv_string_to_list(value)
            raise


class _PlanoraDotEnvSettingsSource(DotEnvSettingsSource):
    """Dotenv source with CSV fallback for legacy string-list vars."""

    def decode_complex_value(self, field_name: str, field: Any, value: Any) -> Any:
        try:
            return super().decode_complex_value(field_name, field, value)
        except ValueError:
            if isinstance(value, str) and _is_string_sequence_field(field):
                return _csv_string_to_list(value)
            raise


class _NormalizedTomlConfigSettingsSource(TomlConfigSettingsSource):
    """TOML source that rewrites legacy flat keys into nested structures."""

    def __init__(self, settings_cls: type[BaseSettings], toml_file: Path) -> None:
        super().__init__(settings_cls, toml_file=toml_file)
        self.toml_data = _normalize_legacy_file_data(self.toml_data)
        super(TomlConfigSettingsSource, self).__init__(settings_cls, self.toml_data)


class _NormalizedPyprojectTomlConfigSettingsSource(PyprojectTomlConfigSettingsSource):
    """Pyproject source that rewrites legacy flat keys into nested structures."""

    def __init__(self, settings_cls: type[BaseSettings], toml_file: Path) -> None:
        super().__init__(settings_cls, toml_file=toml_file)
        self.toml_data = _normalize_legacy_file_data(self.toml_data)
        super(TomlConfigSettingsSource, self).__init__(settings_cls, self.toml_data)


class AgentOverrideConfig(BaseModel):
    """Per-agent overrides from [agents.<name>]."""

    model_config = ConfigDict(extra="ignore")

    model: str | None = None
    stall_timeout: float | None = None
    deep_tool_timeout: float | None = None
    env: dict[str, str] = Field(default_factory=dict)


class PromptsConfig(BaseModel):
    """Prompt template paths from [prompts]."""

    model_config = ConfigDict(extra="ignore")

    plan: Path | None = None
    audit: Path | None = None
    refine: Path | None = None


class ObservabilityConfig(BaseModel):
    """Observability settings from [observability]."""

    model_config = ConfigDict(extra="ignore")

    stall_timeout: float = _DEFAULT_STALL_TIMEOUT
    deep_tool_timeout: float = _DEFAULT_DEEP_TOOL_TIMEOUT
    monitor_interval: float = _DEFAULT_MONITOR_INTERVAL


class TelemetryConfig(BaseModel):
    """Telemetry settings from [telemetry]."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    otlp_endpoint: str | None = None
    otlp_protocol: str = _DEFAULT_TELEMETRY_PROTOCOL
    service_name: str = _DEFAULT_TELEMETRY_SERVICE_NAME
    log_prompts: bool = False


class DefaultsConfig(BaseModel):
    """Workflow defaults from [defaults]."""

    model_config = ConfigDict(extra="ignore")

    planner: str = _DEFAULT_PLANNER
    auditors: list[str] = Field(default_factory=lambda: list(_DEFAULT_AUDITORS))
    concurrency: int = _DEFAULT_CONCURRENCY
    audit_rounds: int = _DEFAULT_AUDIT_ROUNDS
    project_root: Path = _DEFAULT_PROJECT_ROOT
    reports_dir: str = _DEFAULT_REPORTS_DIR

    @model_validator(mode="before")
    @classmethod
    def _coerce_auditors(cls, value: Any) -> Any:
        if isinstance(value, dict) and "auditors" in value:
            return {**value, "auditors": _coerce_auditor_list(value["auditors"])}
        return value


class ProfileConfig(BaseModel):
    """Named profile overrides from [profiles.<name>]."""

    model_config = ConfigDict(extra="ignore")

    planner: str | None = None
    auditors: list[str] | None = None
    concurrency: int | None = None
    audit_rounds: int | None = None
    agents: dict[str, AgentOverrideConfig] = Field(default_factory=dict)
    prompts: PromptsConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_auditors(cls, value: Any) -> Any:
        if isinstance(value, dict) and "auditors" in value:
            return {**value, "auditors": _coerce_auditor_list(value["auditors"])}
        return value


class PlanораSettings(BaseSettings):
    """Settings loaded from env, dotenv, planora.toml, and pyproject.toml."""

    model_config = SettingsConfigDict(
        env_prefix="PLANORA_",
        env_file=(".env", ".env.local"),
        env_nested_delimiter="__",
        env_ignore_empty=True,
        extra="ignore",
        pyproject_toml_table_header=("tool", "planora"),
    )

    # Backward-compatible flat fields
    default_planner: str = _DEFAULT_PLANNER
    default_auditors: list[str] = Field(default_factory=lambda: list(_DEFAULT_AUDITORS))
    default_concurrency: int = _DEFAULT_CONCURRENCY
    default_audit_rounds: int = _DEFAULT_AUDIT_ROUNDS
    project_root: Path = _DEFAULT_PROJECT_ROOT
    reports_dir: str = _DEFAULT_REPORTS_DIR

    claude_model: str = _DEFAULT_AGENT_MODELS["claude"]
    copilot_model: str = _DEFAULT_AGENT_MODELS["copilot"]
    codex_model: str = _DEFAULT_AGENT_MODELS["codex"]
    gemini_model: str = _DEFAULT_AGENT_MODELS["gemini"]

    stall_timeout: float = _DEFAULT_STALL_TIMEOUT
    deep_tool_timeout: float = _DEFAULT_DEEP_TOOL_TIMEOUT
    monitor_check_interval: float = _DEFAULT_MONITOR_INTERVAL
    cli_status_interval: float = _DEFAULT_CLI_STATUS_INTERVAL

    telemetry_enabled: bool = False
    telemetry_otlp_endpoint: str | None = None
    telemetry_otlp_protocol: str = _DEFAULT_TELEMETRY_PROTOCOL
    telemetry_service_name: str = _DEFAULT_TELEMETRY_SERVICE_NAME
    telemetry_log_prompts: bool = False

    # Per-model token pricing in USD, keyed by model name then "input"/"output".
    # Example: {"claude-opus-4-6": {"input": "0.000015", "output": "0.000075"}}
    # Reserved for future cost-tracking integration; not yet consumed at runtime.
    token_pricing: dict[str, dict[str, Decimal]] = Field(default_factory=dict)

    # New nested fields
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    agents: dict[str, AgentOverrideConfig] = Field(default_factory=dict)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    profiles: dict[str, ProfileConfig] = Field(default_factory=dict)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Register sources in the Phase 4 priority order."""
        config = settings_cls.model_config
        sources: list[PydanticBaseSettingsSource] = [
            init_settings,
            _PlanoraEnvSettingsSource(
                settings_cls,
                case_sensitive=config.get("case_sensitive"),
                env_prefix=config.get("env_prefix"),
                env_prefix_target=config.get("env_prefix_target"),
                env_nested_delimiter=config.get("env_nested_delimiter"),
                env_nested_max_split=config.get("env_nested_max_split"),
                env_ignore_empty=config.get("env_ignore_empty"),
                env_parse_none_str=config.get("env_parse_none_str"),
                env_parse_enums=config.get("env_parse_enums"),
            ),
            _PlanoraDotEnvSettingsSource(
                settings_cls,
                env_file=config.get("env_file"),
                env_file_encoding=config.get("env_file_encoding"),
                case_sensitive=config.get("case_sensitive"),
                env_prefix=config.get("env_prefix"),
                env_prefix_target=config.get("env_prefix_target"),
                env_nested_delimiter=config.get("env_nested_delimiter"),
                env_nested_max_split=config.get("env_nested_max_split"),
                env_ignore_empty=config.get("env_ignore_empty"),
                env_parse_none_str=config.get("env_parse_none_str"),
                env_parse_enums=config.get("env_parse_enums"),
            ),
            file_secret_settings,
        ]

        if (project_toml := _project_config_path()) is not None:
            sources.append(_NormalizedTomlConfigSettingsSource(settings_cls, project_toml))

        if (user_toml := _user_config_path()) is not None:
            sources.append(_NormalizedTomlConfigSettingsSource(settings_cls, user_toml))

        pyproject = _pyproject_config_path() or (Path.cwd() / "pyproject.toml")
        sources.append(_NormalizedPyprojectTomlConfigSettingsSource(settings_cls, pyproject))

        return tuple(sources)

    def model_post_init(self, __context: Any) -> None:
        """Backfill legacy flat attributes from the resolved nested config."""
        if self.default_planner == _DEFAULT_PLANNER:
            self.default_planner = self.defaults.planner
        if self.default_auditors == _DEFAULT_AUDITORS:
            self.default_auditors = list(self.defaults.auditors)
        if self.default_concurrency == _DEFAULT_CONCURRENCY:
            self.default_concurrency = self.defaults.concurrency
        if self.default_audit_rounds == _DEFAULT_AUDIT_ROUNDS:
            self.default_audit_rounds = self.defaults.audit_rounds
        if self.project_root == _DEFAULT_PROJECT_ROOT:
            self.project_root = self.defaults.project_root
        if self.reports_dir == _DEFAULT_REPORTS_DIR:
            self.reports_dir = self.defaults.reports_dir

        if self.claude_model == _DEFAULT_AGENT_MODELS["claude"]:
            self.claude_model = (
                self.agents.get("claude", AgentOverrideConfig()).model
                or self.claude_model
            )
        if self.copilot_model == _DEFAULT_AGENT_MODELS["copilot"]:
            self.copilot_model = (
                self.agents.get("copilot", AgentOverrideConfig()).model
                or self.copilot_model
            )
        if self.codex_model == _DEFAULT_AGENT_MODELS["codex"]:
            self.codex_model = (
                self.agents.get("codex", AgentOverrideConfig()).model
                or self.codex_model
            )
        if self.gemini_model == _DEFAULT_AGENT_MODELS["gemini"]:
            self.gemini_model = (
                self.agents.get("gemini", AgentOverrideConfig()).model
                or self.gemini_model
            )

        if self.stall_timeout == _DEFAULT_STALL_TIMEOUT:
            self.stall_timeout = self.observability.stall_timeout
        if self.deep_tool_timeout == _DEFAULT_DEEP_TOOL_TIMEOUT:
            self.deep_tool_timeout = self.observability.deep_tool_timeout
        if self.monitor_check_interval == _DEFAULT_MONITOR_INTERVAL:
            self.monitor_check_interval = self.observability.monitor_interval

        if self.telemetry_enabled is False:
            self.telemetry_enabled = self.telemetry.enabled
        if self.telemetry_otlp_endpoint is None:
            self.telemetry_otlp_endpoint = self.telemetry.otlp_endpoint
        if self.telemetry_otlp_protocol == _DEFAULT_TELEMETRY_PROTOCOL:
            self.telemetry_otlp_protocol = self.telemetry.otlp_protocol
        if self.telemetry_service_name == _DEFAULT_TELEMETRY_SERVICE_NAME:
            self.telemetry_service_name = self.telemetry.service_name
        if self.telemetry_log_prompts is False:
            self.telemetry_log_prompts = self.telemetry.log_prompts

    def with_profile(self, profile_name: str) -> PlanораSettings:
        """Return a new settings object with a named profile merged on top."""
        if profile_name not in self.profiles:
            available = ", ".join(sorted(self.profiles)) or "none defined"
            raise ValueError(
                f"Profile '{profile_name}' not found. Available profiles: {available}"
            )

        profile = self.profiles[profile_name]
        patch: dict[str, Any] = {}
        defaults_patch: dict[str, Any] = {}
        if profile.planner is not None:
            defaults_patch["planner"] = profile.planner
        if profile.auditors is not None:
            defaults_patch["auditors"] = profile.auditors
        if profile.concurrency is not None:
            defaults_patch["concurrency"] = profile.concurrency
        if profile.audit_rounds is not None:
            defaults_patch["audit_rounds"] = profile.audit_rounds
        if defaults_patch:
            patch["defaults"] = defaults_patch

        if profile.agents:
            agent_patches: dict[str, dict[str, Any]] = {}
            for name, override in profile.agents.items():
                if agent_patch := _agent_override_patch(override):
                    agent_patches[name] = agent_patch
            if agent_patches:
                patch["agents"] = agent_patches

        if profile.prompts is not None:
            prompt_patch = profile.prompts.model_dump(mode="python", exclude_none=True)
            if prompt_patch:
                patch["prompts"] = prompt_patch

        merged = _deep_merge(
            self.model_dump(mode="python"),
            _sync_flat_patch_fields(patch),
        )
        return type(self)(**merged)

    def with_config_overrides(self, overrides: list[str]) -> PlanораSettings:
        """Apply repeatable --config key=value overrides using dot notation."""
        if not overrides:
            return self

        patch: dict[str, Any] = {}
        parsed_keys: list[list[str]] = []
        for override in overrides:
            if "=" not in override:
                raise ValueError(
                    f"Invalid --config value '{override}'. "
                    "Expected format: key=value (for example defaults.planner=gemini)."
                )

            key, _, raw_value = override.partition("=")
            key = key.strip()
            raw_value = raw_value.strip()
            if not key:
                raise ValueError(
                    f"Invalid --config value '{override}'. "
                    "Config keys must not be empty."
                )

            keys = key.split(".")
            parsed_keys.append(keys)
            _set_nested(patch, keys, _parse_toml_value(raw_value))

        merged = _deep_merge(self.model_dump(mode="python"), _sync_flat_patch_fields(patch))
        updated = type(self)(**merged)
        dumped = updated.model_dump(mode="python")
        for keys in parsed_keys:
            if not _path_exists(dumped, keys):
                raise ValueError(f"Unknown config override key: {'.'.join(keys)}")
        return updated

    @property
    def effective_planner(self) -> str:
        return self.default_planner

    @property
    def effective_auditors(self) -> list[str]:
        return list(self.default_auditors)

    @property
    def effective_concurrency(self) -> int:
        return self.default_concurrency

    @property
    def effective_audit_rounds(self) -> int:
        return self.default_audit_rounds

    @property
    def effective_project_root(self) -> Path:
        if self.project_root.is_absolute():
            return self.project_root
        return (_effective_config_base_dir() / self.project_root).resolve()

    @property
    def effective_reports_dir(self) -> str:
        return self.reports_dir

    @property
    def effective_stall_timeout(self) -> float:
        return self.stall_timeout

    @property
    def effective_deep_tool_timeout(self) -> float:
        return self.deep_tool_timeout

    @property
    def effective_monitor_interval(self) -> float:
        return self.monitor_check_interval

    @property
    def effective_cli_status_interval(self) -> float:
        return self.cli_status_interval

    @property
    def effective_telemetry_enabled(self) -> bool:
        return self.telemetry_enabled

    @property
    def effective_telemetry_otlp_endpoint(self) -> str:
        return self.telemetry_otlp_endpoint or ""

    @property
    def effective_telemetry_otlp_protocol(self) -> str:
        return self.telemetry_otlp_protocol

    @property
    def effective_telemetry_service_name(self) -> str:
        return self.telemetry_service_name

    @property
    def effective_telemetry_log_prompts(self) -> bool:
        return self.telemetry_log_prompts

    @property
    def config_base_dir(self) -> Path:
        return self.effective_prompt_base_dir

    @property
    def effective_prompt_base_dir(self) -> Path:
        return _effective_config_base_dir()


def _agent_override_patch(override: AgentOverrideConfig) -> dict[str, Any]:
    patch = override.model_dump(mode="python", exclude_none=True)
    if not override.env:
        patch.pop("env", None)
    return patch


# Public ASCII alias for the settings type. Some internal references still use the
# original identifier, but external imports should not depend on a confusable name.
PlanoraSettings = PlanораSettings
