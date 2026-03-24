from __future__ import annotations

import shutil
from enum import StrEnum

from pydantic import BaseModel, Field


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
    """Create all 7 built-in agent definitions."""
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


class AgentRegistry:
    """Registry of all available agents."""

    def __init__(self) -> None:
        self.agents: dict[str, AgentConfig] = _builtin_agents()

    def get(self, name: str) -> AgentConfig:
        """Get agent config by name. Raises KeyError if not found."""
        if name not in self.agents:
            raise KeyError(f"Agent '{name}' not found in registry")
        return self.agents[name]

    def available(self) -> list[str]:
        """Return names of agents whose binary is on PATH."""
        return [
            name
            for name, config in self.agents.items()
            if shutil.which(config.binary) is not None
        ]

    def validate(self, names: list[str]) -> list[str]:
        """Return list of agent names that are NOT available (binary not on PATH)."""
        return [
            name
            for name in names
            if name not in self.agents or shutil.which(self.agents[name].binary) is None
        ]
