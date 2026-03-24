from __future__ import annotations

import asyncio
import gzip
import re
import shutil
from datetime import datetime
from pathlib import Path  # noqa: TCH003
from types import TracebackType  # noqa: TCH003


class WorkspaceManager:
    """Manages .plan-workspace/ and reports/plans/ directories.

    The archive root is ``reports_dir / "plans"`` where ``reports_dir`` is a
    path (relative to ``project_root`` or absolute) sourced from
    ``PlanораSettings.effective_reports_dir``.  When not supplied it defaults
    to ``project_root / "reports"``, preserving the original behaviour.
    """

    def __init__(self, project_root: Path, reports_dir: Path | None = None) -> None:
        self._project_root = project_root
        # Relative paths are anchored to project_root; absolute paths are used as-is.
        reports_path = reports_dir if reports_dir is not None else Path("reports")
        self._reports_root = (
            reports_path if reports_path.is_absolute() else project_root / reports_path
        )
        self._task_slug: str = "untitled"
        self._timestamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    @property
    def workspace_dir(self) -> Path:
        """Active workspace: {project_root}/.plan-workspace/"""
        return self._project_root / ".plan-workspace"

    @property
    def archive_dir(self) -> Path:
        """Archive: {reports_root}/plans/{timestamp}_{slug}/"""
        return self._reports_root / "plans" / f"{self._timestamp}_{self._task_slug}"

    def set_task_slug(self, description: str) -> None:
        """Generate slug from task description: first 40 chars, lowercase, non-alnum → hyphens."""
        slug = description[:40].lower()
        slug = re.sub(r"[^a-z0-9]+", "-", slug)
        slug = slug.strip("-")
        self._task_slug = slug or "untitled"

    def ensure_dirs(self, reuse: bool = False) -> None:
        """
        Create workspace directories.

        reuse=False (default): wipe existing workspace before creating fresh dirs.
        reuse=True: preserve existing workspace contents, only create dirs that don't exist.
        """
        if not reuse and self.workspace_dir.exists():
            shutil.rmtree(self.workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    def write_file(self, name: str, content: str) -> Path:
        """Write a file to the workspace directory."""
        path = self.workspace_dir / name
        path.write_text(content, encoding="utf-8")
        return path

    def read_file(self, name: str) -> str | None:
        """Read a file from the workspace directory. Returns None if missing."""
        path = self.workspace_dir / name
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    async def aread_file(self, name: str) -> str | None:
        """Async variant of read_file; offloads blocking I/O to the thread pool."""
        return await asyncio.to_thread(self.read_file, name)

    async def awrite_file(self, name: str, content: str) -> Path:
        """Async variant of write_file; offloads blocking I/O to the thread pool."""
        return await asyncio.to_thread(self.write_file, name, content)

    async def aarchive(self) -> Path:
        """Async variant of archive; offloads blocking I/O to the thread pool."""
        return await asyncio.to_thread(self.archive)

    def archive(self) -> Path:
        """
        Copy workspace to archive directory.

        - Compresses .stream files with gzip
        - .md and .log files stored uncompressed
        - Creates 'latest' symlink
        """
        archive = self.archive_dir
        archive.mkdir(parents=True, exist_ok=True)

        for src_file in self.workspace_dir.iterdir():
            if not src_file.is_file():
                continue
            if src_file.suffix == ".stream":
                # Compress .stream files with gzip
                dest = archive / f"{src_file.name}.gz"
                with src_file.open("rb") as f_in, gzip.open(dest, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            else:
                shutil.copy2(src_file, archive / src_file.name)

        # Create 'latest' symlink
        latest = archive.parent / "latest"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(archive.name)

        return archive

    def cleanup(self) -> None:
        """Remove the workspace directory."""
        if self.workspace_dir.exists():
            shutil.rmtree(self.workspace_dir)

    def __enter__(self) -> WorkspaceManager:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Preserve workspace on error, cleanup on success."""
        if exc_type is None:
            self.cleanup()
