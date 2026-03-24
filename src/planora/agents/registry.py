from __future__ import annotations

import shutil
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from planora.core.config import AgentOverrideConfig, PlanораSettings


class StreamFormat(StrEnum):
    """JSONL stream format variants."""

    CLAUDE = "claude"
    CODEX = "codex"
    OPENCODE = "opencode"
    GEMINI = "gemini"
    COPILOT = "copilot"


class AgentMode(StrEnum):
    """Agent execution modes."""

    PLAN = "plan"
    FIX = "fix"


class OutputExtraction(BaseModel):
    """Defines how to extract final text output from an agent's raw stream."""

    class Strategy(StrEnum):
        JQ_FILTER = "jq_filter"
        DIRECT_FILE = "direct_file"
        STDOUT_CAPTURE = "stdout"

    strategy: Strategy
    # Declarative reference for the extraction logic applied by StreamParser.
    # The stream parser extracts text natively (no jq binary required); this
    # field documents the equivalent jq expression for human reference and
    # future tooling that may invoke jq directly.
    jq_expression: str | None = None
    strip_preamble: bool = False
    stderr_as_stream: bool = False


class AgentConfig(BaseModel):
    """Single agent definition."""

    name: str
    binary: str
    model: str
    subcommand: str
    flags: dict[AgentMode, list[str]]
    stream_format: StreamFormat
    env_vars: dict[str, str] = Field(default_factory=dict)
    output_extraction: OutputExtraction
    stall_timeout: float = 300.0
    deep_tool_timeout: float = 600.0


def _builtin_agents() -> dict[str, AgentConfig]:
    """Create all built-in agent definitions."""
    return {
        "claude": AgentConfig(
            name="claude",
            binary="claude",
            model="claude-opus-4-6",
            subcommand="-p",
            flags={
                AgentMode.PLAN: [
                    "--allowedTools",
                    "Agent,Read,Glob,Grep,LS,mcp__exa__web_search_exa,mcp__exa__web_search_advanced_exa,mcp__exa__deep_search_exa,mcp__exa__crawling_exa,mcp__exa__get_code_context_exa",
                    "--verbose",
                    "--output-format",
                    "stream-json",
                    "--include-partial-messages",
                    "--max-turns",
                    "50",
                ],
                AgentMode.FIX: [],
            },
            stream_format=StreamFormat.CLAUDE,
            env_vars={"CLAUDE_CODE_EFFORT_LEVEL": "high"},
            output_extraction=OutputExtraction(
                strategy=OutputExtraction.Strategy.JQ_FILTER,
                jq_expression=(
                    'select(.type == "assistant")'
                    " | .message.content[]?"
                    ' | select(.type == "text")'
                    " | .text // empty"
                ),
                strip_preamble=True,
            ),
        ),
        "copilot": AgentConfig(
            name="copilot",
            binary="copilot",
            model="claude-sonnet-4.5",
            subcommand="-p",
            flags={
                AgentMode.PLAN: [
                    "--output-format",
                    "json",
                    "--stream",
                    "on",
                    "--autopilot",
                    "--no-ask-user",
                    "--silent",
                    "--allow-tool",
                    "read",
                    "--deny-tool",
                    "shell(rm),shell(git push),write",
                    "--max-autopilot-continues",
                    "50",
                ],
                AgentMode.FIX: [],
            },
            stream_format=StreamFormat.COPILOT,
            output_extraction=OutputExtraction(
                strategy=OutputExtraction.Strategy.STDOUT_CAPTURE,
            ),
        ),
        "codex": AgentConfig(
            name="codex",
            binary="codex",
            model="gpt-5.4",
            subcommand="exec",
            flags={
                AgentMode.PLAN: [
                    "-c",
                    "model_reasoning_effort=xhigh",
                    "-s",
                    "read-only",
                    "--json",
                ],
                AgentMode.FIX: [],
            },
            stream_format=StreamFormat.CODEX,
            output_extraction=OutputExtraction(
                strategy=OutputExtraction.Strategy.DIRECT_FILE,
            ),
        ),
        "gemini": AgentConfig(
            name="gemini",
            binary="gemini",
            model="gemini-3.1-pro-preview",
            subcommand="-p",
            flags={
                AgentMode.PLAN: [
                    "--approval-mode",
                    "plan",
                ],
                AgentMode.FIX: [],
            },
            stream_format=StreamFormat.GEMINI,
            output_extraction=OutputExtraction(
                strategy=OutputExtraction.Strategy.STDOUT_CAPTURE,
                stderr_as_stream=True,
            ),
        ),
        "opencode-kimi": AgentConfig(
            name="opencode-kimi",
            binary="opencode",
            model="opencode/kimi-k2.5-free",
            subcommand="run",
            flags={
                AgentMode.PLAN: [
                    "--agent",
                    "plan",
                    "--format",
                    "json",
                ],
                AgentMode.FIX: [],
            },
            stream_format=StreamFormat.OPENCODE,
            output_extraction=OutputExtraction(
                strategy=OutputExtraction.Strategy.JQ_FILTER,
                jq_expression='select(.type == "text") | .part.text // empty',
            ),
        ),
        "opencode-glm": AgentConfig(
            name="opencode-glm",
            binary="opencode",
            model="zai-coding-plan/glm-4.7",
            subcommand="run",
            flags={
                AgentMode.PLAN: [
                    "--agent",
                    "plan",
                    "--format",
                    "json",
                ],
                AgentMode.FIX: [],
            },
            stream_format=StreamFormat.OPENCODE,
            output_extraction=OutputExtraction(
                strategy=OutputExtraction.Strategy.JQ_FILTER,
                jq_expression='select(.type == "text") | .part.text // empty',
            ),
        ),
        "opencode-minimax": AgentConfig(
            name="opencode-minimax",
            binary="opencode",
            model="opencode/minimax-m2.5-free",
            subcommand="run",
            flags={
                AgentMode.PLAN: [
                    "--agent",
                    "plan",
                    "--format",
                    "json",
                ],
                AgentMode.FIX: [],
            },
            stream_format=StreamFormat.OPENCODE,
            output_extraction=OutputExtraction(
                strategy=OutputExtraction.Strategy.JQ_FILTER,
                jq_expression='select(.type == "text") | .part.text // empty',
            ),
        ),
    }


def _agent_override_data(override: AgentOverrideConfig) -> dict[str, Any]:
    """Convert a typed override model into an update dict."""
    update: dict[str, Any] = {}
    if override.model is not None:
        update["model"] = override.model
    if override.stall_timeout is not None:
        update["stall_timeout"] = override.stall_timeout
    if override.deep_tool_timeout is not None:
        update["deep_tool_timeout"] = override.deep_tool_timeout
    if override.env:
        update["env_vars"] = override.env
    return update


class AgentRegistry:
    """Registry of all available agents."""

    def __init__(self, agents: dict[str, AgentConfig] | None = None) -> None:
        self._agents: dict[str, AgentConfig] = agents if agents is not None else _builtin_agents()
        self.agents = self._agents

    @classmethod
    def default(cls) -> AgentRegistry:
        """Return a registry with the built-in agent definitions."""
        return cls(_builtin_agents())

    @classmethod
    def from_settings(cls, settings: PlanораSettings) -> AgentRegistry:
        """Build a registry with global and per-agent settings overrides applied."""
        from planora.core.config import AgentOverrideConfig

        global_stall_timeout = settings.effective_stall_timeout
        global_deep_tool_timeout = settings.effective_deep_tool_timeout

        merged_overrides: dict[str, AgentOverrideConfig] = {}
        default_registry = cls.default()
        for name, config in default_registry._agents.items():
            patch: dict[str, Any] = {}
            if global_stall_timeout != config.stall_timeout:
                patch["stall_timeout"] = global_stall_timeout
            if global_deep_tool_timeout != config.deep_tool_timeout:
                patch["deep_tool_timeout"] = global_deep_tool_timeout
            if patch:
                merged_overrides[name] = AgentOverrideConfig(**patch)

        legacy_models = {
            "claude": settings.claude_model,
            "copilot": settings.copilot_model,
            "codex": settings.codex_model,
            "gemini": settings.gemini_model,
        }
        for name, model in legacy_models.items():
            builtin_model = default_registry.get(name).model
            if model == builtin_model:
                continue

            existing = merged_overrides.get(name)
            update = existing.model_dump(mode="python", exclude_none=True) if existing else {}
            update["model"] = model
            merged_overrides[name] = AgentOverrideConfig(**update)

        for name, override in settings.agents.items():
            existing = merged_overrides.get(name)
            if existing is None:
                merged_overrides[name] = override
                continue

            update = existing.model_dump(mode="python", exclude_none=True)
            override_update = override.model_dump(mode="python", exclude_none=True)
            if existing.env or override.env:
                update["env"] = {**existing.env, **override.env}
            for key, value in override_update.items():
                if key == "env":
                    continue
                update[key] = value
            merged_overrides[name] = AgentOverrideConfig(**update)

        return default_registry.with_overrides(merged_overrides)

    def with_overrides(self, agent_overrides: dict[str, AgentOverrideConfig]) -> AgentRegistry:
        """Return a new registry with per-agent overrides applied."""
        patched: dict[str, AgentConfig] = {}
        for name, config in self._agents.items():
            override = agent_overrides.get(name)
            if override is None:
                patched[name] = config
                continue

            update = _agent_override_data(override)
            if env_vars := update.pop("env_vars", None):
                update["env_vars"] = {**config.env_vars, **env_vars}
            patched[name] = config.model_copy(update=update)

        return type(self)(patched)

    def get(self, name: str) -> AgentConfig:
        """Get agent config by name. Raises KeyError if not found."""
        if name not in self._agents:
            raise KeyError(f"Agent '{name}' not found in registry")
        return self._agents[name]

    def available(self) -> list[str]:
        """Return names of agents whose binary is on PATH."""
        return [
            name for name, config in self._agents.items() if shutil.which(config.binary) is not None
        ]

    def validate(self, names: list[str]) -> list[str]:
        """Return agent names that are unknown or whose binaries are missing."""
        return [
            name
            for name in names
            if name not in self._agents or shutil.which(self._agents[name].binary) is None
        ]
