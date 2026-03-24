from __future__ import annotations

import re
from datetime import datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

from planora.prompts.contracts import AuditSeverity, AuditVerdict

if TYPE_CHECKING:
    from pathlib import Path

    from planora.core.events import AgentResult, PhaseResult, PlanResult
    from planora.core.workspace import WorkspaceManager

# ---------------------------------------------------------------------------
# Pure helper functions — no I/O, fully testable in isolation
# ---------------------------------------------------------------------------

_VERDICT_PATTERN = re.compile(
    r"^##\s+Verdict\s*$",
    re.MULTILINE,
)
_FINDINGS_SECTION_PATTERN = re.compile(
    r"^##\s+Findings\s*$",
    re.MULTILINE,
)
_HEADING3_PATTERN = re.compile(r"^###\s+", re.MULTILINE)
_SEVERITY_LABELS = "|".join(
    re.escape(s.value) for s in AuditSeverity if s != AuditSeverity.SUGGESTION
)
_SEVERITY_PATTERN = re.compile(r"\[(" + _SEVERITY_LABELS + r")\]")

_UNKNOWN_VERDICT = "UNKNOWN"


def _extract_verdict(audit_content: str) -> str:
    """Extract APPROVE/NEEDS_REVISION/MAJOR_CONCERNS from an audit file.

    Matches the ``## Verdict`` heading and reads the first non-empty line that
    follows it.  Returns ``"UNKNOWN"`` when no recognisable verdict is found.
    """
    match = _VERDICT_PATTERN.search(audit_content)
    if not match:
        return _UNKNOWN_VERDICT

    after = audit_content[match.end() :]
    for line in after.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        for verdict in AuditVerdict:
            if verdict.value in stripped:
                return verdict.value
        # First non-empty line found but no known verdict value — stop searching
        break

    return _UNKNOWN_VERDICT


def _count_findings(audit_content: str) -> int:
    """Count ``### `` headings within the Findings section of an audit file.

    Returns the count of third-level headings that appear after the
    ``## Findings`` heading and before the next ``## `` section (or EOF).
    """
    section_match = _FINDINGS_SECTION_PATTERN.search(audit_content)
    if not section_match:
        return 0

    findings_body = audit_content[section_match.end() :]
    # Truncate at the next top-level or second-level heading
    next_section = re.search(r"^##\s+", findings_body, re.MULTILINE)
    if next_section:
        findings_body = findings_body[: next_section.start()]

    return len(_HEADING3_PATTERN.findall(findings_body))


def _count_severities(audit_content: str) -> dict[str, int]:
    """Count ``[CRITICAL]``, ``[HIGH]``, ``[MEDIUM]``, ``[LOW]`` occurrences.

    Returns a dict mapping each severity label to its occurrence count.
    Keys are always present, value is 0 when no match found.
    """
    counts: dict[str, int] = {s.value: 0 for s in AuditSeverity if s != AuditSeverity.SUGGESTION}
    for m in _SEVERITY_PATTERN.finditer(audit_content):
        label = m.group(1)
        counts[label] = counts.get(label, 0) + 1
    return counts


def _format_duration(td: timedelta | None) -> str:
    """Format a timedelta as ``Xm Ys`` (e.g. ``"2m 30s"``).

    Returns ``"—"`` when *td* is ``None``.  Sub-minute durations render as
    ``"Xs"`` only.
    """
    if td is None:
        return "—"
    total_seconds = int(td.total_seconds())
    if total_seconds < 0:
        total_seconds = 0
    minutes, seconds = divmod(total_seconds, 60)
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _format_cost(cost: Decimal | None) -> str:
    """Format a cost as ``"$X.XXXX"`` or ``"N/A"`` when *cost* is ``None``."""
    if cost is None:
        return "N/A"
    return f"${cost:.4f}"


def _format_file_size(size_bytes: int) -> str:
    """Format a file size in bytes as a human-readable string.

    Uses KB (1 024 bytes) or MB (1 048 576 bytes) as appropriate; falls back
    to plain bytes for very small files.
    """
    if size_bytes >= 1_048_576:
        return f"{size_bytes / 1_048_576:.1f} MB"
    if size_bytes >= 1_024:
        return f"{size_bytes / 1_024:.1f} KB"
    return f"{size_bytes} B"


# ---------------------------------------------------------------------------
# Internal section builders — pure string assembly, no I/O
# ---------------------------------------------------------------------------


def _build_pipeline_config_section(
    planner: str,
    auditors: list[str],
    audit_rounds: int,
    max_concurrency: int,
    planner_model: str = "",
    auditor_models: dict[str, str] | None = None,
) -> str:
    planner_cell = f"{planner} ({planner_model})" if planner_model else planner
    if auditors:
        if auditor_models:
            auditor_parts = [
                f"{a} ({auditor_models[a]})" if a in auditor_models else a for a in auditors
            ]
        else:
            auditor_parts = list(auditors)
        auditor_list = ", ".join(auditor_parts)
    else:
        auditor_list = "none"
    return (
        "## Pipeline Configuration\n\n"
        "| Setting          | Value                          |\n"
        "|------------------|--------------------------------|\n"
        f"| Planner          | {planner_cell}                 |\n"
        f"| Auditors         | {auditor_list}                 |\n"
        f"| Audit Rounds     | {audit_rounds}                 |\n"
        f"| Concurrency      | {max_concurrency}              |\n"
    )


def _phase_display_label(phase_key: str) -> str:
    """Map internal phase keys to human-facing labels for the report."""
    if phase_key == "plan":
        return "Plan"
    if phase_key == "report":
        return "Report"
    if phase_key == "audit":
        return "Audit R1"
    if phase_key == "refine":
        return "Refine R1"
    if phase_key.startswith("audit-r"):
        return f"Audit R{phase_key.removeprefix('audit-r')}"
    if phase_key.startswith("refine-r"):
        return f"Refine R{phase_key.removeprefix('refine-r')}"
    return phase_key.replace("-", " ").title()


def _build_phase_summary_section(phases: list[PhaseResult], planner: str) -> str:
    _cols = "| Phase       | Status    | Duration | Agent(s)        | Cost     | Output File |"
    _sep = "|-------------|-----------|----------|-----------------|----------|-------------|"
    header_row = _cols + "\n" + _sep + "\n"
    rows = ["## Phase Summary\n\n" + header_row]
    for phase in phases:
        agents_in_phase = (
            ", ".join(ar.agent_name for ar in phase.agent_results)
            if phase.agent_results
            else planner
        )
        output_file = phase.output_files[0].name if phase.output_files else "—"
        label = _phase_display_label(phase.name)
        rows.append(
            f"| {label:<11} | {phase.status.value:<9} | {_format_duration(phase.duration):<8} "
            f"| {agents_in_phase:<15} | {_format_cost(phase.cost_usd):<8} | {output_file:<20} |\n"
        )
    # Report row is rendered from actual data if present in phases, otherwise as placeholder
    report_phase = next((p for p in phases if p.name == "report"), None)
    if report_phase is None:
        rows.append(
            f"| {'Report':<11} | {'done':<9} | {'—':<8} "
            f"| {'—':<15} | {'—':<8} | {'plan-report.md':<20} |\n"
        )
    return "".join(rows)


def _build_audit_results_section(
    workspace: WorkspaceManager,
    auditors: list[str],
    audit_rounds: int,
    phases: list[PhaseResult],
    auditor_models: dict[str, str] | None = None,
) -> str:
    lines: list[str] = ["## Audit Results\n"]

    for round_num in range(1, audit_rounds + 1):
        # Determine which phase corresponds to this audit round
        phase_name = f"audit-r{round_num}" if round_num > 1 else "audit"
        phase = next((p for p in phases if p.name == phase_name), None)

        if phase is not None:
            succeeded, total = _audit_phase_counts(phase)
            lines.append(f"\n### Round {round_num}: {succeeded}/{total} auditors completed\n\n")
        else:
            lines.append(f"\n### Round {round_num}\n\n")

        for agent in auditors:
            file_key = f"audit-{agent}.md" if round_num == 1 else f"audit-{agent}-r{round_num}.md"
            content = workspace.read_file(file_key)

            model = auditor_models.get(agent, "") if auditor_models else ""
            agent_heading = f"{agent} ({model})" if model else agent

            if content is None or not content.strip():
                lines.append(f"#### {agent_heading} — FAILED (no output)\n\n")
                continue

            verdict = _extract_verdict(content)
            finding_count = _count_findings(content)
            severities = _count_severities(content)

            severity_parts = [
                f"{count} {label}" for label, count in severities.items() if count > 0
            ]
            severity_str = ", ".join(severity_parts) if severity_parts else "none"

            lines.append(f"#### {agent_heading}\n\n")
            lines.append(f"- **Verdict:** {verdict}\n")
            lines.append(f"- **Findings:** {finding_count}\n")
            lines.append(f"- **Severities:** {severity_str}\n\n")

    return "".join(lines)


def _audit_phase_counts(phase: PhaseResult) -> tuple[int, int]:
    """Return (succeeded, total) agent counts for an audit phase."""
    total = len(phase.agent_results)
    succeeded = sum(1 for ar in phase.agent_results if not ar.output_empty and ar.exit_code == 0)
    return succeeded, total


def _build_files_created_section(workspace: WorkspaceManager) -> str:
    lines: list[str] = ["## Files Created\n\n"]
    try:
        file_entries = sorted(
            (f for f in workspace.workspace_dir.iterdir() if f.is_file()),
            key=lambda f: f.name,
        )
    except OSError:
        lines.append("_(workspace directory unavailable)_\n")
        return "".join(lines)

    if not file_entries:
        lines.append("_(no files)_\n")
        return "".join(lines)

    for file_path in file_entries:
        try:
            size = file_path.stat().st_size
            size_str = _format_file_size(size)
        except OSError:
            size_str = "unknown"
        lines.append(f"- `{file_path.name}` — {size_str}\n")

    return "".join(lines)


def _build_agent_cost_section(agent_results: dict[str, list[AgentResult]]) -> str:
    rows: list[str] = [
        "## Agent Cost Breakdown\n\n"
        "| Agent      | Phase      | Cost       | Tokens (in/out) | Duration |\n"
        "|------------|------------|------------|-----------------|----------|\n"
    ]
    total_cost: Decimal = Decimal("0")
    has_any_cost = False

    for phase_name, results in agent_results.items():
        for ar in results:
            if ar.cost_usd is not None:
                total_cost += ar.cost_usd
                has_any_cost = True

            token_str = _format_tokens(ar.token_usage)
            rows.append(
                f"| {ar.agent_name:<10} | {phase_name:<10} | {_format_cost(ar.cost_usd):<10} "
                f"| {token_str:<15} | {_format_duration(ar.duration):<8} |\n"
            )

    total_cost_display = f"**${total_cost:.4f}**" if has_any_cost else "**N/A**"
    rows.append(
        f"| **Total**  |            | {total_cost_display:<10} |                 |          |\n"
    )

    return "".join(rows)


def _format_tokens(token_usage: dict[str, int] | None) -> str:
    """Format token usage as ``"in/out"`` or ``"—"`` when unavailable."""
    if token_usage is None:
        return "—"
    in_tokens = token_usage.get("input_tokens", token_usage.get("in", 0))
    out_tokens = token_usage.get("output_tokens", token_usage.get("out", 0))
    return f"{in_tokens}/{out_tokens}"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_plan_report(
    workspace: WorkspaceManager,
    plan_result: PlanResult,
    planner: str,
    auditors: list[str],
    audit_rounds: int,
    max_concurrency: int = 1,
    planner_model: str = "",
    auditor_models: dict[str, str] | None = None,
) -> Path:
    """Generate ``plan-report.md`` and write it to the workspace.

    Reads audit files from the workspace, assembles a structured Markdown
    report covering pipeline config, phase summary, audit results, files
    created, and agent cost breakdown, then writes ``plan-report.md`` via
    ``workspace.write_file``.

    Args:
        workspace:        Active :class:`WorkspaceManager` for reading audit
                          files and writing the report.
        plan_result:      Aggregated result from the pipeline run.
        planner:          Name of the planning agent (e.g. ``"claude"``).
        auditors:         Ordered list of auditor agent names.
        audit_rounds:     Number of audit/refine rounds that were configured.
        max_concurrency:  Maximum number of agents run in parallel (default 1).
        planner_model:    Model identifier for the planner agent (e.g.
                          ``"claude-opus-4-6"``).  Empty string when unknown.
        auditor_models:   Mapping of auditor agent name to its model identifier.
                          ``None`` or absent keys render without model suffix.

    Returns:
        :class:`Path` pointing to the written ``plan-report.md`` inside the
        workspace directory.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    duration_str = _format_duration(plan_result.total_duration)
    cost_str = _format_cost(plan_result.total_cost_usd)

    header = (
        "# Plan Report\n\n"
        f"**Generated:** {timestamp}\n"
        f"**Duration:** {duration_str}\n"
        f"**Total Cost:** {cost_str}\n\n"
    )

    pipeline_section = _build_pipeline_config_section(
        planner=planner,
        auditors=auditors,
        audit_rounds=audit_rounds,
        max_concurrency=max_concurrency,
        planner_model=planner_model,
        auditor_models=auditor_models,
    )

    phase_section = _build_phase_summary_section(
        phases=plan_result.phases,
        planner=planner,
    )

    audit_section = _build_audit_results_section(
        workspace=workspace,
        auditors=auditors,
        audit_rounds=audit_rounds,
        phases=plan_result.phases,
        auditor_models=auditor_models,
    )

    files_section = _build_files_created_section(workspace)

    cost_section = _build_agent_cost_section(plan_result.agent_results)

    archive_section = f"## Archive\n\nAll outputs archived to: `{workspace.archive_dir}`\n"

    content = (
        "\n\n".join(
            [
                header.rstrip(),
                pipeline_section.rstrip(),
                phase_section.rstrip(),
                audit_section.rstrip(),
                files_section.rstrip(),
                cost_section.rstrip(),
                archive_section.rstrip(),
            ]
        )
        + "\n"
    )

    return workspace.write_file("plan-report.md", content)
