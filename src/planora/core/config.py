from __future__ import annotations

from decimal import Decimal  # noqa: TCH003 — Pydantic needs at runtime
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PlanораSettings(BaseSettings):
    """Settings loaded from env, .env, or pyproject.toml."""

    model_config = SettingsConfigDict(
        env_prefix="PLANORA_",
        env_file=".env",
        toml_file="pyproject.toml",
    )

    # Defaults
    default_planner: str = "claude"
    default_auditors: list[str] = Field(default=["gemini", "codex"])
    default_concurrency: int = 3
    default_audit_rounds: int = 1

    # Paths
    project_root: Path = Path(".")
    reports_dir: str = "reports"

    # Agent model overrides (optional per-project)
    claude_model: str = "claude-opus-4-6"
    copilot_model: str = "claude-sonnet-4.5"
    codex_model: str = "gpt-5.4"
    gemini_model: str = "gemini-3.1-pro-preview"

    # Observability
    stall_timeout: float = 300.0
    deep_tool_timeout: float = 600.0
    monitor_check_interval: float = 5.0
    cli_status_interval: float = 5.0

    # Telemetry (optional)
    telemetry_enabled: bool = False
    telemetry_otlp_endpoint: str | None = None
    telemetry_otlp_protocol: str = "grpc"
    telemetry_service_name: str = "planora"
    telemetry_log_prompts: bool = False

    # Token-to-cost estimation (optional)
    token_pricing: dict[str, dict[str, Decimal]] = Field(default_factory=dict)
