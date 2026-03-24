from __future__ import annotations

import logging
import stat
from contextlib import suppress
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class _AgentConfigLike(Protocol):
    name: str


class ClaudeHooksManager:
    """Manage temporary Claude Code hook scripts for a Planora run."""

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace
        self._hooks_dir = workspace / ".planora-hooks"
        self._events_log = self._hooks_dir / "hook-events.jsonl"
        self._created_paths: list[Path] = []

    def install_hooks(self, agent_config: _AgentConfigLike) -> dict[str, object]:
        """Create hook scripts and return a Claude-compatible hooks config."""
        if agent_config.name != "claude":
            return {}

        self._hooks_dir.mkdir(parents=True, exist_ok=True)
        pre_hook = self._hooks_dir / "pre_tool_use.sh"
        post_hook = self._hooks_dir / "post_tool_use.sh"

        self._write_hook(pre_hook, self.create_pre_tool_use_hook())
        self._write_hook(post_hook, self.create_post_tool_use_hook())

        self._created_paths = [pre_hook, post_hook]
        return {
            "hooks": {
                "PreToolUse": [{"type": "command", "command": str(pre_hook)}],
                "PostToolUse": [{"type": "command", "command": str(post_hook)}],
            }
        }

    def create_pre_tool_use_hook(self) -> str:
        """Return the shell script used for Claude's `PreToolUse` hook."""
        return self._build_hook_script("PreToolUse")

    def create_post_tool_use_hook(self) -> str:
        """Return the shell script used for Claude's `PostToolUse` hook."""
        return self._build_hook_script("PostToolUse")

    def cleanup(self) -> None:
        """Remove generated hook scripts and the hook directory when empty."""
        for path in self._created_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Failed to unlink hook file %s: %s", path, exc)
                continue

        with suppress(OSError):
            self._hooks_dir.rmdir()

        self._created_paths = []

    def _build_hook_script(self, hook_name: str) -> str:
        log_path = self._events_log
        return f"""#!/usr/bin/env bash
set -euo pipefail
python3 -c '
import datetime
import json
import pathlib
import sys

log_path = pathlib.Path(sys.argv[1])
hook_name = sys.argv[2]
payload = sys.stdin.read()
log_path.parent.mkdir(parents=True, exist_ok=True)
record = {{
    "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
    "hook": hook_name,
    "payload": payload,
}}
with log_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(record) + "\\n")
' "{log_path}" "{hook_name}"
"""

    @staticmethod
    def _write_hook(path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
