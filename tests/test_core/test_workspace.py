from __future__ import annotations

import gzip
from typing import TYPE_CHECKING

import pytest

from planora.core.workspace import WorkspaceManager

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_workspace(root: Path) -> WorkspaceManager:
    return WorkspaceManager(root)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_workspace_dir_is_dot_plan_workspace_under_project_root(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)

    assert ws.workspace_dir == tmp_path / ".plan-workspace"


def test_archive_dir_contains_timestamp_and_slug(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    ws.set_task_slug("Add unit tests")

    archive = ws.archive_dir

    assert archive.parent == tmp_path / "reports" / "plans"
    assert "add-unit-tests" in archive.name


# ---------------------------------------------------------------------------
# set_task_slug
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("description", "expected_slug"),
    [
        ("Add unit tests", "add-unit-tests"),
        ("Fix the bug!", "fix-the-bug"),
        ("  leading and trailing  ", "leading-and-trailing"),
        ("", "untitled"),
        ("   ", "untitled"),
        ("Already-Clean-Slug", "already-clean-slug"),
        ("A" * 50, "a" * 40),  # truncated to 40 chars
        ("---special---chars---", "special-chars"),
    ],
)
def test_set_task_slug_normalises_description(
    tmp_path: Path, description: str, expected_slug: str
) -> None:
    ws = _make_workspace(tmp_path)
    ws.set_task_slug(description)

    assert ws._task_slug == expected_slug


# ---------------------------------------------------------------------------
# ensure_dirs
# ---------------------------------------------------------------------------


def test_ensure_dirs_creates_workspace_directory(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)

    ws.ensure_dirs()

    assert ws.workspace_dir.is_dir()


def test_ensure_dirs_wipes_existing_workspace_by_default(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    ws.ensure_dirs()
    stale_file = ws.workspace_dir / "stale.md"
    stale_file.write_text("old content", encoding="utf-8")

    ws.ensure_dirs(reuse=False)

    assert not stale_file.exists()
    assert ws.workspace_dir.is_dir()


def test_ensure_dirs_reuse_preserves_existing_files(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    ws.ensure_dirs()
    existing = ws.workspace_dir / "initial-plan.md"
    existing.write_text("# Existing plan\n", encoding="utf-8")

    ws.ensure_dirs(reuse=True)

    assert existing.exists()
    assert existing.read_text(encoding="utf-8") == "# Existing plan\n"


def test_ensure_dirs_creates_missing_dir_when_reuse_is_true(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)

    ws.ensure_dirs(reuse=True)

    assert ws.workspace_dir.is_dir()


# ---------------------------------------------------------------------------
# write_file / read_file
# ---------------------------------------------------------------------------


def test_write_file_creates_file_under_workspace_dir(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    ws.ensure_dirs()

    path = ws.write_file("initial-plan.md", "# Plan\n")

    assert path == ws.workspace_dir / "initial-plan.md"
    assert path.read_text(encoding="utf-8") == "# Plan\n"


def test_read_file_returns_content_of_existing_file(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    ws.ensure_dirs()
    ws.write_file("task-input.md", "Build the feature\n")

    content = ws.read_file("task-input.md")

    assert content == "Build the feature\n"


def test_read_file_returns_none_for_missing_file(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    ws.ensure_dirs()

    result = ws.read_file("nonexistent.md")

    assert result is None


# ---------------------------------------------------------------------------
# archive
# ---------------------------------------------------------------------------


def test_archive_copies_md_and_log_files_uncompressed(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    ws.ensure_dirs()
    ws.write_file("initial-plan.md", "# Plan\n")
    ws.write_file("audit-claude.log", "agent log\n")

    archive_path = ws.archive()

    assert (archive_path / "initial-plan.md").exists()
    assert (archive_path / "initial-plan.md").read_text(encoding="utf-8") == "# Plan\n"
    assert (archive_path / "audit-claude.log").exists()


def test_archive_compresses_stream_files_with_gzip(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    ws.ensure_dirs()
    ws.write_file("agent.stream", '{"type":"result"}\n')

    archive_path = ws.archive()

    compressed = archive_path / "agent.stream.gz"
    assert compressed.exists()
    with gzip.open(compressed, "rt", encoding="utf-8") as f:
        assert f.read() == '{"type":"result"}\n'
    assert not (archive_path / "agent.stream").exists()


def test_archive_creates_latest_symlink(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    ws.ensure_dirs()
    ws.write_file("initial-plan.md", "# Plan\n")

    archive_path = ws.archive()

    latest = archive_path.parent / "latest"
    assert latest.is_symlink()
    assert latest.resolve() == archive_path


def test_archive_updates_latest_symlink_on_second_call(tmp_path: Path) -> None:
    ws1 = WorkspaceManager(tmp_path)
    ws1._timestamp = "2024-01-01_00-00-00"
    ws1.ensure_dirs()
    ws1.write_file("initial-plan.md", "first run\n")
    ws1.archive()

    ws2 = WorkspaceManager(tmp_path)
    ws2._timestamp = "2024-01-02_00-00-00"
    ws2.ensure_dirs()
    ws2.write_file("initial-plan.md", "second run\n")
    archive_path = ws2.archive()

    latest = archive_path.parent / "latest"
    assert latest.resolve() == archive_path


def test_archive_returns_path_to_archive_directory(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    ws.ensure_dirs()
    ws.write_file("plan.md", "content\n")

    archive_path = ws.archive()

    assert archive_path.is_dir()
    assert archive_path == ws.archive_dir


# ---------------------------------------------------------------------------
# cleanup
# ---------------------------------------------------------------------------


def test_cleanup_removes_workspace_directory(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    ws.ensure_dirs()
    ws.write_file("initial-plan.md", "# Plan\n")

    ws.cleanup()

    assert not ws.workspace_dir.exists()


def test_cleanup_is_safe_when_workspace_does_not_exist(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)

    # Must not raise
    ws.cleanup()


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


def test_context_manager_cleans_up_on_success(tmp_path: Path) -> None:
    with WorkspaceManager(tmp_path) as ws:
        ws.ensure_dirs()
        ws.write_file("initial-plan.md", "# Plan\n")

    assert not ws.workspace_dir.exists()


def test_context_manager_preserves_workspace_on_exception(tmp_path: Path) -> None:
    try:
        with WorkspaceManager(tmp_path) as ws:
            ws.ensure_dirs()
            ws.write_file("initial-plan.md", "# Plan\n")
            raise RuntimeError("deliberate failure")
    except RuntimeError:
        pass

    assert ws.workspace_dir.exists()
    assert (ws.workspace_dir / "initial-plan.md").exists()
