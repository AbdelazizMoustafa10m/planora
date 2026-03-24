from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import FileSystemLoader, StrictUndefined
from jinja2.sandbox import SandboxedEnvironment

from planora.prompts.contracts import (
    AuditCategory,
    AuditSeverity,
    AuditVerdict,
    Complexity,
    FindingVerdict,
    PlanPhase,
    PlanSection,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _TemplateConfig:
    plan: Path | None = None
    audit: Path | None = None
    refine: Path | None = None
    base_dir: Path = Path(".")


_ACTIVE_TEMPLATE_CONFIG: _TemplateConfig | None = None

# ---------------------------------------------------------------------------
# Dynamic lists derived from enum members — built once at module load time
# ---------------------------------------------------------------------------

_SECTIONS_LIST: str = "\n".join(f"## {i + 1}. {s.value}" for i, s in enumerate(PlanSection))

_AUDIT_CATEGORIES_LIST: str = "\n".join(f"- {c.value}" for c in AuditCategory)

_SEVERITY_LEVELS_LIST: str = " > ".join(s.value for s in AuditSeverity)

_VERDICT_OPTIONS_LIST: str = " | ".join(v.value for v in AuditVerdict)

_COMPLEXITY_TAGS: str = ", ".join(f"[{c.value}] = {c.name.capitalize()}" for c in Complexity)

_FINDING_VERDICTS: str = " | ".join(v.value for v in FindingVerdict)

_CATEGORY_DESCRIPTIONS: str = (
    f"  - **{AuditCategory.MISSING_STEPS.value}**: Gaps in the sequence, "
    f"missing error handling, cleanup, or rollback steps.\n"
    f"  - **{AuditCategory.SECURITY.value}**: SQL injection, XSS, auth bypass, "
    f"data leakage, broken RLS.\n"
    f"  - **{AuditCategory.PERFORMANCE.value}**: N+1 queries, missing indexes, "
    f"unnecessary re-renders, request waterfalls.\n"
    f"  - **{AuditCategory.EDGE_CASES.value}**: Empty data sets, concurrent "
    f"requests, null handling, rate limits.\n"
    f"  - **{AuditCategory.CONVENTION_VIOLATIONS.value}**: Anything that "
    f"violates CLAUDE.md or established project patterns.\n"
    f"  - **{AuditCategory.TESTING_GAPS.value}**: Untested code paths or "
    f"missing error scenario coverage.\n"
    f"  - **{AuditCategory.BETTER_ALTERNATIVES.value}**: Simpler approach, "
    f"existing utility, or better library available.\n"
    f"  - **{AuditCategory.DEPENDENCY_RISKS.value}**: Version conflicts, "
    f"missing packages, circular import dependencies.\n"
    f"  - **{AuditCategory.ORDERING.value}**: Steps in wrong sequence or with "
    f"incorrect dependency order.\n"
    f"  - **{AuditCategory.COMPLETENESS.value}**: Plan does not fully address "
    f"all task requirements."
)


# ---------------------------------------------------------------------------
# Jinja2 template helpers
# ---------------------------------------------------------------------------


def _create_jinja_env(template_dir: Path) -> SandboxedEnvironment:
    """Create a sandboxed Jinja2 environment rooted at the template directory."""
    return SandboxedEnvironment(
        loader=FileSystemLoader(str(template_dir)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _render_template(template_path: Path, variables: dict[str, Any]) -> str:
    """Load and render a markdown prompt template."""
    environment = _create_jinja_env(template_path.parent)
    template = environment.get_template(template_path.name)
    return template.render(**variables)


def _resolve_template_path(
    configured_path: Path | None,
    base_dir: Path,
    prompt_type: str,
) -> Path | None:
    """Resolve a configured template path, warning and falling back if it is missing."""
    if configured_path is None:
        return None

    resolved = configured_path
    if not resolved.is_absolute():
        resolved = (base_dir / resolved).resolve()

    if resolved.exists():
        return resolved

    logger.warning(
        "Prompt template not found: '%s' (configured for %s prompt). "
        "Falling back to built-in prompt.",
        resolved,
        prompt_type,
    )
    return None


def _base_template_context() -> dict[str, Any]:
    """Return the structural contract values available to all templates."""
    return {
        "plan_sections": [section.value for section in PlanSection],
        "audit_categories": [category.value for category in AuditCategory],
        "severity_levels": [severity.value for severity in AuditSeverity],
    }


def configure_prompt_templates(
    *,
    plan: Path | None = None,
    audit: Path | None = None,
    refine: Path | None = None,
    base_dir: Path = Path("."),
) -> None:
    """Set the active prompt template configuration for this process."""
    global _ACTIVE_TEMPLATE_CONFIG
    _ACTIVE_TEMPLATE_CONFIG = _TemplateConfig(
        plan=plan,
        audit=audit,
        refine=refine,
        base_dir=base_dir,
    )


def reset_prompt_templates() -> None:
    """Clear any process-local prompt template configuration."""
    global _ACTIVE_TEMPLATE_CONFIG
    _ACTIVE_TEMPLATE_CONFIG = None


def _current_template_config() -> _TemplateConfig:
    """Return the active template configuration, falling back to base settings."""
    if _ACTIVE_TEMPLATE_CONFIG is not None:
        return _ACTIVE_TEMPLATE_CONFIG

    try:
        from planora.core.config import PlanораSettings

        settings = PlanораSettings()
    except Exception:  # noqa: BLE001 - prompt fallback should remain best-effort
        return _TemplateConfig()

    return _TemplateConfig(
        plan=settings.prompts.plan,
        audit=settings.prompts.audit,
        refine=settings.prompts.refine,
        base_dir=settings.effective_prompt_base_dir,
    )


# ---------------------------------------------------------------------------
# Public prompt-builder functions
# ---------------------------------------------------------------------------


def build_plan_prompt(
    task_content: str,
    claude_md: str,
    *,
    template_path: Path | None = None,
    base_dir: Path = Path("."),
) -> str:
    """Return the plan prompt, optionally rendering a configured Jinja2 template."""
    if template_path is None:
        config = _current_template_config()
        template_path = config.plan
        base_dir = config.base_dir

    resolved = _resolve_template_path(template_path, base_dir, "plan")
    if resolved is not None:
        return _render_template(
            resolved,
            {
                **_base_template_context(),
                "task_content": task_content,
                "claude_md": claude_md,
            },
        )
    return _build_plan_prompt_builtin(task_content, claude_md)


def build_audit_prompt(
    round: int,  # noqa: A002
    plan_content: str,
    task_content: str,
    claude_md: str,
    prior_audits: dict[str, str] | None = None,
    *,
    template_path: Path | None = None,
    base_dir: Path = Path("."),
) -> str:
    """Return the audit prompt, optionally rendering a configured Jinja2 template."""
    if template_path is None:
        config = _current_template_config()
        template_path = config.audit
        base_dir = config.base_dir

    resolved = _resolve_template_path(template_path, base_dir, "audit")
    if resolved is not None:
        return _render_template(
            resolved,
            {
                **_base_template_context(),
                "round": round,
                "plan_content": plan_content,
                "task_content": task_content,
                "claude_md": claude_md,
                "prior_audits": prior_audits,
            },
        )
    return _build_audit_prompt_builtin(
        round,
        plan_content,
        task_content,
        claude_md,
        prior_audits,
    )


def build_refinement_prompt(
    round: int,  # noqa: A002
    plan_content: str,
    task_content: str,
    claude_md: str,
    audit_reports: dict[str, str],
    prior_audits: dict[str, str] | None = None,
    *,
    template_path: Path | None = None,
    base_dir: Path = Path("."),
) -> str:
    """Return the refinement prompt, optionally rendering a configured Jinja2 template."""
    if template_path is None:
        config = _current_template_config()
        template_path = config.refine
        base_dir = config.base_dir

    resolved = _resolve_template_path(template_path, base_dir, "refine")
    if resolved is not None:
        return _render_template(
            resolved,
            {
                **_base_template_context(),
                "round": round,
                "plan_content": plan_content,
                "task_content": task_content,
                "claude_md": claude_md,
                "audit_reports": audit_reports,
                "prior_audits": prior_audits,
            },
        )
    return _build_refinement_prompt_builtin(
        round,
        plan_content,
        task_content,
        claude_md,
        audit_reports,
        prior_audits,
    )


def _build_plan_prompt_builtin(task_content: str, claude_md: str) -> str:
    """Return a self-contained system prompt for the plan-generation agent.

    The prompt instructs the agent to follow a 4-phase workflow
    (EXPLORE → ANALYZE → DESIGN → PLAN) and produce a 12-section
    implementation plan.

    Args:
        task_content: The raw task description provided by the user.
        claude_md: Contents of the project's CLAUDE.md convention file.
            Pass an empty string when not available.

    Returns:
        A fully self-contained prompt string ready to send as the system
        message (or first user message) to the planning agent.
    """
    claude_md_section = (
        f"""
---
## Project Conventions (CLAUDE.md)

Study and comply with every rule in the CLAUDE.md below before planning.
All steps must respect these project-specific conventions.

```
{claude_md}
```
"""
        if claude_md.strip()
        else ""
    )

    phase_descriptions = "\n".join(f"  - **{phase.value}**" for phase in PlanPhase)

    return f"""\
## System Role

You are an expert implementation planner. Your task is to analyze a codebase \
and produce a detailed, actionable implementation plan. You have access to \
powerful exploration tools — use them wisely and systematically before writing \
a single line of the plan.

---
## Sub-Agent Strategy

When exploring the codebase, choose the right tool for the job:

- **Agent tool (broad exploration)**: Use for any task that requires reading \
3 or more files, tracing an architectural pattern, or understanding an entire \
subsystem. The Agent tool parallelises reads and returns a synthesised summary.
- **Direct file reads (targeted lookup)**: Use only when you need the exact \
contents of 1-2 specific, already-identified files.
- **Exa MCP tools (documentation lookup)**: Use `mcp__exa__web_search_exa` \
or `mcp__exa__deep_search_exa` for external documentation, library APIs, and \
specification lookups. Always verify claims against live docs — never rely \
solely on your training data for library-specific details.

---
## 4-Phase Planning Workflow

Work through each phase in order. Do **not** skip ahead to PLAN without \
completing the earlier phases.

{phase_descriptions}

### Phase 1 — {PlanPhase.EXPLORE.value}

Read the codebase to understand its architecture before forming any opinions:

1. Read the project root (`README`, `pyproject.toml`, CI config) to understand \
the stack and tooling.
2. Identify and read project convention files (e.g., `CLAUDE.md`, \
`CONTRIBUTING.md`, `AGENTS.md`).
3. Map the high-level directory structure and identify key modules.
4. Read the files most relevant to the task — use the Agent tool if exploring \
more than 2 files at once.
5. Note existing patterns: naming conventions, error-handling style, test \
structure, import organisation.

### Phase 2 — {PlanPhase.ANALYZE.value}

Categorise every piece of information you discovered:

- **FACTS KNOWN**: Confirmed by reading actual source files.
- **FACTS TO LOOK UP**: Things you need to verify in docs or source before \
planning. Use Exa MCP tools for external docs. Mark items here with \
`[UNVERIFIED: ...]` until confirmed.
- **FACTS TO DERIVE**: Logical conclusions drawn from known facts (mark with \
`[ASSUMPTION: ...]` if not directly confirmed).

Do not proceed to DESIGN until every FACTS TO LOOK UP item is either \
confirmed or explicitly flagged `[UNVERIFIED: ...]`.

### Phase 3 — {PlanPhase.DESIGN.value}

Choose the implementation approach:

1. List at least **2 alternative approaches** with their trade-offs.
2. Explicitly state which approach you are selecting and why.
3. Identify risks or unknowns in your chosen approach.
4. Confirm the chosen design is consistent with the project's existing \
patterns (see EXPLORE findings).

### Phase 4 — {PlanPhase.PLAN.value}

Produce the detailed implementation plan using the output format below.

---
## Output Format

The plan must contain **all 12 sections** in the exact order below. \
Each section heading must match exactly (including numbering):

```
{_SECTIONS_LIST}
```

### Section-Level Requirements

**Section 1 — Overview**: 3-5 sentences. State what is being built, why, \
and which existing systems it touches.

**Section 2 — Critical Files**: A bulleted list of every file that must be \
read before implementation can begin, with a one-line note on why each is \
critical.

**Section 3 — Architecture & Design**: The chosen design with a brief \
rationale. Include a simple ASCII diagram if it clarifies the design.

**Section 4 — Implementation Steps**: Numbered, ordered steps. \
**Every step must carry one of these complexity tags**: {_COMPLEXITY_TAGS}. \
Place the tag at the start of the step title, e.g. `[S] Add index to users \
table`. Each step must also include:
  - Exact file path(s) affected
  - Function or class signatures where new code is introduced
  - Which prior steps are prerequisites

**Section 5 — Files to Create/Modify**: A table with columns: \
`File | Action (Create/Modify/Delete) | Purpose`.

**Section 6 — Database Changes**: All schema migrations, index changes, \
RLS policy changes, or seed-data modifications. Write "None" if not \
applicable.

**Section 7 — Testing Strategy**: Which test types are needed \
(unit/integration/e2e), which edge cases must be covered, and which files \
hold the relevant test fixtures.

**Section 8 — Verification Checklist**: A Markdown checklist (`- [ ]`) of \
concrete, runnable commands or observable outcomes that confirm the \
implementation is complete and correct.

**Section 9 — Risks & Edge Cases**: Enumerate known risks, failure modes, \
and edge cases. For each, provide a mitigation or fallback strategy.

**Section 10 — Dependencies & Prerequisites**: List every prerequisite: \
new packages, environment variables, deployed services, or completed tasks \
that must exist before implementation starts.

**Section 11 — Open Questions**: Use `[OPEN QUESTION: ...]` tags for \
anything that could not be resolved during planning and requires a decision \
before or during implementation.

**Section 12 — Documentation References**: Links or file paths to relevant \
official docs, RFCs, ADRs, or internal wikis.

---
## Plan Quality Calibration

Steps must be **specific and verifiable**, not vague. Examples:

- BAD: "Update the service"
  GOOD: "[M] Add `retry_count: int` to `PaymentService.__init__` \
in `src/payments/service.py`"
- BAD: "Fix the bug"
  GOOD: "[S] Wrap `db.execute()` at line 42 of `src/db/client.py` \
in `except OperationalError`"
- BAD: "Add tests"
  GOOD: "[M] Add parametrised cases in `tests/unit/test_payments.py` \
covering `amount <= 0`, timeout, and unsupported currency"

Every step must provide enough detail that a competent engineer could \
execute it without asking clarifying questions.

---
## Constraints

- **Read-only**: You MUST NOT modify, create, or delete any files during \
the planning phase. Exploration only.
- **Focus on planning**: Your deliverable is a plan document, not code \
snippets. Pseudocode and signatures are acceptable; full implementations \
are not.
- **Honesty markers**:
  - `[ASSUMPTION: ...]` — logical deduction not directly confirmed by source
  - `[OPEN QUESTION: ...]` — unresolved decision that needs human input
  - `[UNVERIFIED: ...]` — claim not yet confirmed against live docs or source
- Do not guess at file contents — read them. Do not guess at library APIs — \
look them up with Exa MCP tools.
{claude_md_section}
---
## Task

```
{task_content}
```
"""


def _build_audit_prompt_builtin(
    round: int,  # noqa: A002
    plan_content: str,
    task_content: str,
    claude_md: str,
    prior_audits: dict[str, str] | None = None,
) -> str:
    """Return a self-contained prompt for the audit agent.

    The auditor reviews the plan for gaps, errors, security issues, and
    missed opportunities across 10 structured categories.

    Args:
        round: Audit round number (1-based). Displayed in the report header.
        plan_content: The full text of the implementation plan to audit.
        task_content: The original task description.
        claude_md: Contents of the project's CLAUDE.md convention file.
        prior_audits: Mapping of ``"Round N"`` → audit report text for all
            previous rounds. When provided, the auditor avoids re-raising
            already-addressed findings.

    Returns:
        A fully self-contained prompt string ready to send to the audit agent.
    """
    round_label = f"Round {round}"

    prior_audits_section = ""
    if prior_audits:
        prior_sections = "\n\n".join(
            f"### {label}\n\n```\n{text}\n```" for label, text in prior_audits.items()
        )
        prior_audits_section = f"""
---
## Prior Audit Reports

The following audit reports were produced in earlier rounds. Do **not** \
re-raise findings that were already addressed in those rounds. Focus your \
analysis on issues that remain unresolved or on new concerns introduced by \
the revised plan.

{prior_sections}
"""

    claude_md_section = (
        f"""
---
## Project Conventions (CLAUDE.md)

Use the CLAUDE.md below as the authoritative source for \
`{AuditCategory.CONVENTION_VIOLATIONS.value}` findings. A finding in this \
category is only valid if it explicitly cites a rule that appears in \
the CLAUDE.md.

```
{claude_md}
```
"""
        if claude_md.strip()
        else ""
    )

    verdict_description = (
        f"- **{AuditVerdict.APPROVE.value}**: Plan is sound. No critical or "
        f"high-severity issues. Ready to implement.\n"
        f"- **{AuditVerdict.NEEDS_REVISION.value}**: One or more HIGH or "
        f"MEDIUM findings require the plan to be updated before implementation.\n"
        f"- **{AuditVerdict.MAJOR_CONCERNS.value}**: One or more CRITICAL "
        f"findings. Do not proceed until these are resolved."
    )

    return f"""\
## System Role

You are an expert code reviewer and implementation auditor. Your task is to \
critically review an implementation plan for gaps, errors, security issues, \
and missed opportunities. You are thorough, precise, and constructive. Every \
finding you raise must include a specific, actionable recommendation.

---
## Audit Round

This is **{round_label}**.

---
## Audit Categories

Evaluate the plan across all 10 categories below:

{_AUDIT_CATEGORIES_LIST}

**Category descriptions:**

{_CATEGORY_DESCRIPTIONS}

For each category, ask yourself: "Does the plan adequately address this \
dimension?" If the answer is "no" or "partially", raise a finding.

---
## Severity Levels

Assign one severity level to each finding:

**{_SEVERITY_LEVELS_LIST}**

- **{AuditSeverity.CRITICAL.value}**: A flaw that would cause data loss, \
security breach, or total failure of the feature.
- **{AuditSeverity.HIGH.value}**: A significant gap that would likely cause \
production bugs or regressions.
- **{AuditSeverity.MEDIUM.value}**: An issue that reduces quality, \
maintainability, or coverage in a meaningful way.
- **{AuditSeverity.LOW.value}**: A minor improvement opportunity with limited \
impact.
- **{AuditSeverity.SUGGESTION.value}**: An optional enhancement that would \
be nice to have but is not required.

---
## Verdict Options

Choose one overall verdict for the plan:

{_VERDICT_OPTIONS_LIST}

{verdict_description}

---
## Required Output Format

Your report must follow this structure exactly:

```
# Plan Audit Report ({round_label})

## Verdict
[{_VERDICT_OPTIONS_LIST}]

## Summary
2-3 sentences. State the overall quality of the plan, the most significant \
issue(s) found, and whether the plan is ready to implement.

## Findings
### [{_SEVERITY_LEVELS_LIST}] <Short Title>
- Category: <one of the 10 audit categories>
- Plan Section: <section number and name from the plan>
- Issue: <what is wrong or missing — be specific>
- Recommendation: <concrete fix the planner should apply>

(Repeat for every finding. If no findings, write "No findings.")

## Strengths
2-3 bullet points describing what the plan does well.

## Missing Considerations
Anything the plan should address but does not. If nothing is missing, write \
"None identified."
```

---
## Auditing Constraints

- **Do not hallucinate.** Only raise findings for issues you can directly \
support by pointing to (or noting the absence of) content in the plan.
- **Every finding requires a recommendation.** Do not raise a problem without \
suggesting how to fix it.
- **Focus on substance, not formatting.** Do not penalise the plan for minor \
stylistic choices unless they violate an explicit CLAUDE.md rule.
- **Be specific.** Vague findings like "tests could be improved" are not \
acceptable. Cite the exact step, section, or omission.
{prior_audits_section}{claude_md_section}
---
## Plan to Audit

```
{plan_content}
```

---
## Original Task

```
{task_content}
```
"""


def _build_refinement_prompt_builtin(
    round: int,  # noqa: A002
    plan_content: str,
    task_content: str,
    claude_md: str,
    audit_reports: dict[str, str],
    prior_audits: dict[str, str] | None = None,
) -> str:
    """Return a self-contained prompt for the plan-refinement agent.

    The refinement agent incorporates audit feedback into the plan,
    accepting or rejecting each finding with explicit rationale.

    Args:
        round: Refinement round number (1-based).
        plan_content: The full text of the current implementation plan.
        task_content: The original task description.
        claude_md: Contents of the project's CLAUDE.md convention file.
        audit_reports: Mapping of ``"Round N"`` → audit report text for the
            audit(s) that must be addressed in this revision.
        prior_audits: Mapping of earlier audit reports (rounds before the
            current one) for full historical context. Used in round 2+ so
            the planner can see what was already addressed.

    Returns:
        A fully self-contained prompt string ready to send to the refinement
        agent.
    """
    round_label = f"Round {round}"

    # Build the audit reports block (reports to act on)
    audit_reports_block = "\n\n".join(
        f"### {label}\n\n```\n{text}\n```" for label, text in audit_reports.items()
    )

    # Build the prior audits block (historical context, round 2+ only)
    prior_audits_section = ""
    if prior_audits:
        prior_block = "\n\n".join(
            f"### {label}\n\n```\n{text}\n```" for label, text in prior_audits.items()
        )
        prior_audits_section = f"""
---
## Prior Audit Reports (Historical Context)

The following audit reports were addressed in earlier refinement rounds. \
They are provided so you can see the full audit history and avoid reversing \
changes that were already accepted.

{prior_block}
"""

    # Appendix label differs by round
    if round == 1:
        appendix_label = "Appendix A: Audit Response (Round 1)"
    else:
        appendix_label = f"Appendix: Audit Dispositions (Round {round})"

    claude_md_section = (
        f"""
---
## Project Conventions (CLAUDE.md)

All steps in the revised plan must comply with the CLAUDE.md below. \
When a finding relates to a convention violation, verify it against the \
CLAUDE.md before accepting or rejecting it.

```
{claude_md}
```
"""
        if claude_md.strip()
        else ""
    )

    # Multi-round appendix note
    multi_round_note = ""
    if round > 1:
        multi_round_note = (
            f"\nBecause this is round {round}, the revised plan must include "
            f"both the current `{appendix_label}` appendix **and** all "
            f"appendices from prior rounds. Do not drop earlier appendices.\n"
        )

    return f"""\
## System Role

You are the original implementation planner. Your task is to incorporate \
audit feedback into the plan, producing a revised and improved version that \
addresses every valid concern raised by the auditor.

---
## Refinement Round

This is **{round_label}**.

---
## Your Responsibilities

1. **Read every finding** in the audit report(s) below.
2. **Decide on each finding**: accept it or reject it. Do not ignore findings.
3. **Update the plan** to incorporate every accepted finding.
4. **Append a disposition table** (see appendix format below) listing your \
decision and rationale for every finding.

When **accepting** a finding:
- Make the corresponding change in the relevant plan section.
- Note what you changed in the appendix entry.

When **rejecting** a finding:
- Provide a clear, evidence-based rationale.
- Do not reject solely because a change is inconvenient; reject only if \
the finding is factually wrong, out of scope, or contradicted by project \
conventions.

---
## Appendix Format

At the end of your revised plan, add the following appendix:

```
## {appendix_label}

| Finding | Verdict | Rationale |
|---------|---------|-----------|
| <finding title> | [ACCEPTED/REJECTED] | <1-2 sentence rationale> |
```

Include every finding from the current round's audit report(s) in this table. \
Use exact finding titles from the audit report.
{multi_round_note}
---
## Output Requirements

- Produce the **complete revised plan** — all 12 sections must be present. \
Do not emit a partial diff or a summary of changes.
- Required sections (must all appear, in order):

```
{_SECTIONS_LIST}
```

- Every implementation step in Section 4 must carry a complexity tag: \
{_COMPLEXITY_TAGS}.
- Mark unresolved decisions with `[OPEN QUESTION: ...]`.
- Mark unconfirmed assumptions with `[ASSUMPTION: ...]`.
- Do **not** remove sections or merge them together.
- Do **not** skip the appendix.

---
## Constraints

- **Read-only**: Do NOT modify, create, or delete any files. The refined plan \
is a document, not an implementation.
- **Completeness**: Every step must still have an exact file path and function \
signature where applicable. Do not regress on specificity when incorporating \
changes.
- **Honesty**: If incorporating a finding requires information you don't have, \
add an `[OPEN QUESTION: ...]` rather than guessing.
{prior_audits_section}{claude_md_section}
---
## Audit Report(s) to Address

{audit_reports_block}

---
## Current Plan (to be revised)

```
{plan_content}
```

---
## Original Task

```
{task_content}
```
"""
