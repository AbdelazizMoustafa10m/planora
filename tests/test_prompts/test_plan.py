from __future__ import annotations

from planora.prompts.contracts import (
    AuditCategory,
    AuditSeverity,
    AuditVerdict,
    Complexity,
    PlanPhase,
    PlanSection,
)
from planora.prompts.plan import (
    _AUDIT_CATEGORIES_LIST,
    _COMPLEXITY_TAGS,
    _SECTIONS_LIST,
    _SEVERITY_LEVELS_LIST,
    _VERDICT_OPTIONS_LIST,
    _base_template_context,
    _build_audit_prompt_builtin,
    _build_plan_prompt_builtin,
    _build_refinement_prompt_builtin,
    build_audit_prompt,
    build_plan_prompt,
    build_refinement_prompt,
    configure_prompt_templates,
    reset_prompt_templates,
)

# ---------------------------------------------------------------------------
# Contract constant structure
# ---------------------------------------------------------------------------


def test_sections_list_contains_all_12_sections() -> None:
    lines = _SECTIONS_LIST.strip().splitlines()

    assert len(lines) == len(PlanSection)
    for i, section in enumerate(PlanSection):
        assert section.value in lines[i]
        assert str(i + 1) in lines[i]


def test_audit_categories_list_contains_all_10_categories() -> None:
    for category in AuditCategory:
        assert category.value in _AUDIT_CATEGORIES_LIST


def test_severity_levels_list_joins_all_non_suggestion_severities() -> None:
    for severity in AuditSeverity:
        if severity == AuditSeverity.SUGGESTION:
            continue
        assert severity.value in _SEVERITY_LEVELS_LIST


def test_verdict_options_list_contains_all_verdicts() -> None:
    for verdict in AuditVerdict:
        assert verdict.value in _VERDICT_OPTIONS_LIST


def test_complexity_tags_contains_all_complexity_values() -> None:
    for complexity in Complexity:
        assert complexity.value in _COMPLEXITY_TAGS


def test_base_template_context_keys() -> None:
    ctx = _base_template_context()

    assert set(ctx.keys()) == {"plan_sections", "audit_categories", "severity_levels"}
    assert ctx["plan_sections"] == [s.value for s in PlanSection]
    assert ctx["audit_categories"] == [c.value for c in AuditCategory]
    assert ctx["severity_levels"] == [s.value for s in AuditSeverity]


# ---------------------------------------------------------------------------
# build_plan_prompt_builtin — section structure
# ---------------------------------------------------------------------------


def test_build_plan_prompt_includes_all_12_section_headings() -> None:
    prompt = _build_plan_prompt_builtin("Add tests", "")

    for section in PlanSection:
        assert section.value in prompt


def test_build_plan_prompt_includes_4_phase_workflow_headings() -> None:
    prompt = _build_plan_prompt_builtin("Add tests", "")

    for phase in PlanPhase:
        assert phase.value in prompt


def test_build_plan_prompt_embeds_task_content() -> None:
    task = "Implement the payment gateway"
    prompt = _build_plan_prompt_builtin(task, "")

    assert task in prompt


def test_build_plan_prompt_with_claude_md_includes_conventions_section() -> None:
    prompt = _build_plan_prompt_builtin("Add tests", "Use ruff for linting")

    assert "Project Conventions (CLAUDE.md)" in prompt
    assert "Use ruff for linting" in prompt


def test_build_plan_prompt_without_claude_md_omits_conventions_section() -> None:
    prompt = _build_plan_prompt_builtin("Add tests", "")

    assert "Project Conventions (CLAUDE.md)" not in prompt


def test_build_plan_prompt_without_claude_md_whitespace_omits_conventions_section() -> None:
    prompt = _build_plan_prompt_builtin("Add tests", "   \n  ")

    assert "Project Conventions (CLAUDE.md)" not in prompt


def test_build_plan_prompt_includes_complexity_tags() -> None:
    prompt = _build_plan_prompt_builtin("Add tests", "")

    for complexity in Complexity:
        assert f"[{complexity.value}]" in prompt


# ---------------------------------------------------------------------------
# build_audit_prompt_builtin — section structure
# ---------------------------------------------------------------------------


def test_build_audit_prompt_includes_round_label() -> None:
    prompt = _build_audit_prompt_builtin(3, "plan content", "task", "")

    assert "Round 3" in prompt


def test_build_audit_prompt_includes_all_audit_categories() -> None:
    prompt = _build_audit_prompt_builtin(1, "plan content", "task", "")

    for category in AuditCategory:
        assert category.value in prompt


def test_build_audit_prompt_includes_all_severities() -> None:
    prompt = _build_audit_prompt_builtin(1, "plan content", "task", "")

    for severity in AuditSeverity:
        if severity == AuditSeverity.SUGGESTION:
            continue
        assert severity.value in prompt


def test_build_audit_prompt_includes_verdict_options() -> None:
    prompt = _build_audit_prompt_builtin(1, "plan content", "task", "")

    for verdict in AuditVerdict:
        assert verdict.value in prompt


def test_build_audit_prompt_embeds_plan_and_task_content() -> None:
    prompt = _build_audit_prompt_builtin(1, "# Implementation plan", "Add auth", "")

    assert "# Implementation plan" in prompt
    assert "Add auth" in prompt


def test_build_audit_prompt_with_prior_audits_includes_prior_section() -> None:
    prior = {"Round 1": "Round 1 audit report content"}
    prompt = _build_audit_prompt_builtin(2, "plan", "task", "", prior_audits=prior)

    assert "Prior Audit Reports" in prompt
    assert "Round 1 audit report content" in prompt


def test_build_audit_prompt_without_prior_audits_omits_prior_section() -> None:
    prompt = _build_audit_prompt_builtin(1, "plan", "task", "", prior_audits=None)

    assert "Prior Audit Reports" not in prompt


def test_build_audit_prompt_with_claude_md_includes_conventions_section() -> None:
    prompt = _build_audit_prompt_builtin(1, "plan", "task", "Use type hints everywhere")

    assert "Project Conventions (CLAUDE.md)" in prompt
    assert "Use type hints everywhere" in prompt


# ---------------------------------------------------------------------------
# build_refinement_prompt_builtin — appendix label
# ---------------------------------------------------------------------------


def test_build_refinement_prompt_round_1_uses_appendix_a_label() -> None:
    prompt = _build_refinement_prompt_builtin(
        1, "plan", "task", "", {"Round 1": "audit report"}
    )

    assert "Appendix A: Audit Response (Round 1)" in prompt


def test_build_refinement_prompt_round_2_uses_generic_appendix_label() -> None:
    prompt = _build_refinement_prompt_builtin(
        2, "plan", "task", "", {"Round 2": "audit report"}
    )

    assert "Appendix: Audit Dispositions (Round 2)" in prompt


def test_build_refinement_prompt_includes_all_12_section_headings() -> None:
    prompt = _build_refinement_prompt_builtin(
        1, "plan", "task", "", {"Round 1": "audit report"}
    )

    for section in PlanSection:
        assert section.value in prompt


def test_build_refinement_prompt_round_2_includes_multi_round_note() -> None:
    prompt = _build_refinement_prompt_builtin(
        2, "plan", "task", "", {"Round 2": "audit report"}
    )

    assert "round 2" in prompt.lower()
    assert "prior rounds" in prompt.lower()


def test_build_refinement_prompt_with_prior_audits_includes_history() -> None:
    prior = {"Round 1": "first round report"}
    prompt = _build_refinement_prompt_builtin(
        2, "plan", "task", "", {"Round 2": "second audit"}, prior_audits=prior
    )

    assert "Prior Audit Reports (Historical Context)" in prompt
    assert "first round report" in prompt


def test_build_refinement_prompt_embeds_audit_reports_block() -> None:
    prompt = _build_refinement_prompt_builtin(
        1, "plan", "task", "", {"Round 1": "specific audit finding"}
    )

    assert "specific audit finding" in prompt


# ---------------------------------------------------------------------------
# Public API — configure_prompt_templates / reset / fallback
# ---------------------------------------------------------------------------


def test_build_plan_prompt_falls_back_to_builtin_when_no_template(
    tmp_path,
) -> None:
    reset_prompt_templates()
    configure_prompt_templates(plan=None)

    prompt = build_plan_prompt("Add tests", "")

    for section in PlanSection:
        assert section.value in prompt


def test_build_audit_prompt_falls_back_to_builtin_when_no_template() -> None:
    reset_prompt_templates()
    configure_prompt_templates(audit=None)

    prompt = build_audit_prompt(
        round=1,
        plan_content="# Plan",
        task_content="Add tests",
        claude_md="",
    )

    assert "Round 1" in prompt
    for category in AuditCategory:
        assert category.value in prompt


def test_build_refinement_prompt_falls_back_to_builtin_when_no_template() -> None:
    reset_prompt_templates()
    configure_prompt_templates(refine=None)

    prompt = build_refinement_prompt(
        round=1,
        plan_content="# Plan",
        task_content="Add tests",
        claude_md="",
        audit_reports={"Round 1": "report"},
    )

    assert "Appendix A: Audit Response (Round 1)" in prompt


def test_build_plan_prompt_uses_custom_template(tmp_path) -> None:
    template_file = tmp_path / "plan.j2"
    template_file.write_text("TASK={{ task_content }}", encoding="utf-8")
    reset_prompt_templates()

    prompt = build_plan_prompt(
        "My task",
        "",
        template_path=template_file,
        base_dir=tmp_path,
    )

    assert prompt == "TASK=My task"


def test_build_plan_prompt_warns_and_falls_back_for_missing_template(
    tmp_path, caplog
) -> None:
    import logging

    missing = tmp_path / "no-such.j2"
    reset_prompt_templates()

    with caplog.at_level(logging.WARNING):
        prompt = build_plan_prompt("My task", "", template_path=missing, base_dir=tmp_path)

    assert "not found" in caplog.text.lower()
    # Fell back to builtin — should contain plan sections
    for section in PlanSection:
        assert section.value in prompt


def test_reset_prompt_templates_clears_active_config() -> None:
    configure_prompt_templates(plan=None)
    reset_prompt_templates()

    from planora.prompts.plan import _ACTIVE_TEMPLATE_CONFIG  # noqa: PLC0415

    assert _ACTIVE_TEMPLATE_CONFIG is None
