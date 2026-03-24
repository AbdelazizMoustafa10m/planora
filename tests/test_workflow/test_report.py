from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

import pytest

from planora.core.events import AgentResult, PhaseResult, PhaseStatus
from planora.core.workspace import WorkspaceManager
from planora.prompts.contracts import AuditSeverity, AuditVerdict
from planora.workflow.report import (
    _count_findings,
    _count_severities,
    _extract_verdict,
    _format_cost,
    _format_duration,
    _format_file_size,
    _format_tokens,
    generate_plan_report,
)

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("td", "expected"),
    [
        (None, "—"),
        (timedelta(seconds=0), "0s"),
        (timedelta(seconds=45), "45s"),
        (timedelta(seconds=90), "1m 30s"),
        (timedelta(minutes=5, seconds=0), "5m 0s"),
        (timedelta(seconds=-5), "0s"),  # negative clamped to 0
    ],
)
def test_format_duration_various_inputs(td: timedelta | None, expected: str) -> None:
    assert _format_duration(td) == expected


@pytest.mark.parametrize(
    ("cost", "expected"),
    [
        (None, "N/A"),
        (Decimal("0"), "$0.0000"),
        (Decimal("0.0123"), "$0.0123"),
        (Decimal("1.5"), "$1.5000"),
    ],
)
def test_format_cost_formats_decimal_or_none(cost: Decimal | None, expected: str) -> None:
    assert _format_cost(cost) == expected


@pytest.mark.parametrize(
    ("size_bytes", "expected"),
    [
        (0, "0 B"),
        (512, "512 B"),
        (1024, "1.0 KB"),
        (2048, "2.0 KB"),
        (1_048_576, "1.0 MB"),
        (2_097_152, "2.0 MB"),
    ],
)
def test_format_file_size_uses_correct_unit(size_bytes: int, expected: str) -> None:
    assert _format_file_size(size_bytes) == expected


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (None, "—"),
        ({"input_tokens": 100, "output_tokens": 50}, "100/50"),
        ({"in": 200, "out": 30}, "200/30"),
        ({"input_tokens": 0, "output_tokens": 0}, "0/0"),
    ],
)
def test_format_tokens_various_shapes(
    usage: dict[str, int] | None, expected: str
) -> None:
    assert _format_tokens(usage) == expected


# ---------------------------------------------------------------------------
# _extract_verdict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("## Verdict\nAPPROVE\n", AuditVerdict.APPROVE.value),
        ("## Verdict\nNEEDS_REVISION\n", AuditVerdict.NEEDS_REVISION.value),
        ("## Verdict\nMAJOR_CONCERNS\n", AuditVerdict.MAJOR_CONCERNS.value),
        ("## Verdict\n\nAPPROVE\n", AuditVerdict.APPROVE.value),  # blank line before value
        ("no verdict here", "UNKNOWN"),
        ("## Summary\nsome stuff\n", "UNKNOWN"),
        ("## Verdict\nunrecognised-value\n", "UNKNOWN"),
    ],
)
def test_extract_verdict_various_audit_content(content: str, expected: str) -> None:
    assert _extract_verdict(content) == expected


# ---------------------------------------------------------------------------
# _count_findings
# ---------------------------------------------------------------------------


def test_count_findings_counts_h3_headings_in_findings_section() -> None:
    content = (
        "## Findings\n"
        "### [HIGH] Missing error handling\n"
        "### [MEDIUM] No tests\n"
        "## Strengths\n"
        "### This should not count\n"
    )

    assert _count_findings(content) == 2


def test_count_findings_returns_zero_when_no_findings_section() -> None:
    assert _count_findings("## Summary\nAll good.") == 0


def test_count_findings_returns_zero_for_empty_findings() -> None:
    content = "## Findings\nNo findings.\n## Strengths\n"

    assert _count_findings(content) == 0


# ---------------------------------------------------------------------------
# _count_severities
# ---------------------------------------------------------------------------


def test_count_severities_returns_zero_dict_for_empty_content() -> None:
    result = _count_severities("")

    expected_keys = {s.value for s in AuditSeverity if s != AuditSeverity.SUGGESTION}
    assert set(result.keys()) == expected_keys
    assert all(v == 0 for v in result.values())


def test_count_severities_counts_bracketed_labels() -> None:
    content = (
        "### [CRITICAL] Data breach\n"
        "### [HIGH] Missing validation\n"
        "### [HIGH] Another high\n"
        "### [MEDIUM] Minor issue\n"
    )

    result = _count_severities(content)

    assert result[AuditSeverity.CRITICAL.value] == 1
    assert result[AuditSeverity.HIGH.value] == 2
    assert result[AuditSeverity.MEDIUM.value] == 1
    assert result[AuditSeverity.LOW.value] == 0


def test_count_severities_does_not_count_suggestion_label() -> None:
    content = "[SUGGESTION] Use a better name\n"

    result = _count_severities(content)

    assert AuditSeverity.SUGGESTION.value not in result


# ---------------------------------------------------------------------------
# generate_plan_report (integration: uses workspace I/O)
# ---------------------------------------------------------------------------


def _make_agent_result(
    name: str,
    tmp_path: Path,
    *,
    exit_code: int = 0,
    output_empty: bool = False,
    cost_usd: Decimal | None = None,
) -> AgentResult:
    output_path = tmp_path / f"{name}.md"
    return AgentResult(
        agent_name=name,
        output_path=output_path,
        stream_path=output_path.with_suffix(".stream"),
        log_path=output_path.with_suffix(".log"),
        exit_code=exit_code,
        duration=timedelta(seconds=5),
        output_empty=output_empty,
        cost_usd=cost_usd,
    )


def _make_plan_result(phases: list[PhaseResult], tmp_path: Path):
    from planora.core.events import PlanResult  # noqa: PLC0415

    return PlanResult(
        phases=phases,
        final_plan_path=tmp_path / "final-plan.md",
        report_path=None,
        archive_path=None,
        total_duration=timedelta(seconds=60),
        total_cost_usd=Decimal("0.0500"),
        agent_results={},
        success=True,
    )


def test_generate_plan_report_writes_plan_report_md(tmp_path: Path) -> None:
    ws = WorkspaceManager(tmp_path)
    ws.ensure_dirs()
    plan_result = _make_plan_result([], tmp_path)

    report_path = generate_plan_report(
        workspace=ws,
        plan_result=plan_result,
        planner="claude",
        auditors=[],
        audit_rounds=0,
    )

    assert report_path == ws.workspace_dir / "plan-report.md"
    assert report_path.exists()


def test_generate_plan_report_includes_pipeline_config_section(tmp_path: Path) -> None:
    ws = WorkspaceManager(tmp_path)
    ws.ensure_dirs()
    plan_result = _make_plan_result([], tmp_path)

    report_path = generate_plan_report(
        workspace=ws,
        plan_result=plan_result,
        planner="claude",
        auditors=["gemini", "codex"],
        audit_rounds=1,
        max_concurrency=3,
    )

    content = report_path.read_text(encoding="utf-8")
    assert "Pipeline Configuration" in content
    assert "claude" in content
    assert "gemini, codex" in content
    assert "3" in content  # concurrency


def test_generate_plan_report_shows_no_audit_output_when_agent_has_no_file(
    tmp_path: Path,
) -> None:
    ws = WorkspaceManager(tmp_path)
    ws.ensure_dirs()
    plan_result = _make_plan_result(
        [PhaseResult(name="audit", status=PhaseStatus.DONE)], tmp_path
    )

    report_path = generate_plan_report(
        workspace=ws,
        plan_result=plan_result,
        planner="claude",
        auditors=["gemini"],
        audit_rounds=1,
    )

    content = report_path.read_text(encoding="utf-8")
    assert "gemini" in content
    assert "FAILED (no output)" in content


def test_generate_plan_report_parses_verdict_from_workspace_audit_file(
    tmp_path: Path,
) -> None:
    ws = WorkspaceManager(tmp_path)
    ws.ensure_dirs()
    ws.write_file(
        "audit-gemini.md",
        "## Verdict\nAPPROVE\n\n## Findings\nNo findings.\n",
    )
    plan_result = _make_plan_result(
        [PhaseResult(name="audit", status=PhaseStatus.DONE)], tmp_path
    )

    report_path = generate_plan_report(
        workspace=ws,
        plan_result=plan_result,
        planner="claude",
        auditors=["gemini"],
        audit_rounds=1,
    )

    content = report_path.read_text(encoding="utf-8")
    assert "APPROVE" in content


def test_generate_plan_report_includes_files_created_section(tmp_path: Path) -> None:
    ws = WorkspaceManager(tmp_path)
    ws.ensure_dirs()
    ws.write_file("initial-plan.md", "# Plan\n")
    plan_result = _make_plan_result([], tmp_path)

    report_path = generate_plan_report(
        workspace=ws,
        plan_result=plan_result,
        planner="claude",
        auditors=[],
        audit_rounds=0,
    )

    content = report_path.read_text(encoding="utf-8")
    assert "Files Created" in content
    assert "initial-plan.md" in content


def test_generate_plan_report_includes_archive_path(tmp_path: Path) -> None:
    ws = WorkspaceManager(tmp_path)
    ws.ensure_dirs()
    plan_result = _make_plan_result([], tmp_path)

    report_path = generate_plan_report(
        workspace=ws,
        plan_result=plan_result,
        planner="claude",
        auditors=[],
        audit_rounds=0,
    )

    content = report_path.read_text(encoding="utf-8")
    assert "Archive" in content
