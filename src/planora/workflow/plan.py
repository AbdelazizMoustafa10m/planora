from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

from planora.core.events import (
    AgentResult,
    PhaseResult,
    PhaseStatus,
    PlanResult,
)
from planora.prompts.plan import (
    build_audit_prompt,
    build_plan_prompt,
    build_refinement_prompt,
)
from planora.workflow.engine import PhaseRunner
from planora.workflow.report import generate_plan_report

if TYPE_CHECKING:
    from planora.agents.registry import AgentRegistry
    from planora.agents.runner import AgentRunner
    from planora.core.events import UICallback
    from planora.core.workspace import WorkspaceManager

logger = logging.getLogger(__name__)


class PlanWorkflow:
    """Multi-phase implementation planning workflow.

    Orchestrates: plan → (audit → refine) * N rounds → report/archive.
    Delegates execution to PhaseRunner; owns phase sequencing and error policy.
    """

    def __init__(
        self,
        *,
        workspace: WorkspaceManager,
        registry: AgentRegistry,
        runner: AgentRunner,
        ui: UICallback,
        planner: str = "claude",
        auditors: list[str] | None = None,
        audit_rounds: int = 1,
        max_concurrency: int = 3,
        dry_run: bool = False,
    ) -> None:
        self._workspace = workspace
        self._registry = registry
        self._ui = ui
        self._planner = planner
        self._auditors: list[str] = auditors if auditors is not None else []
        self._audit_rounds = audit_rounds
        self._max_concurrency = max_concurrency
        self._dry_run = dry_run
        self._phase_runner = PhaseRunner(runner, ui, max_concurrency)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self, task_input: str | Path) -> PlanResult:
        """Execute the full planning pipeline and return aggregated results.

        Pipeline:
        1. Preflight — validate agents, create workspace, save task-input.md
        2. Read CLAUDE.md from project root (if exists)
        3. Phase plan — produce initial-plan.md
        4. For each round (1..audit_rounds):
           a. Phase audit — parallel per auditor
           b. Phase refine — incorporate audit feedback
        5. Phase report — generate plan-report.md, archive workspace
        6. Return PlanResult

        Error policy (§15a.4):
        - Plan fails → abort immediately (success=False)
        - Some auditors fail → refine with available audits
        - All auditors fail → skip refinement, warn
        - Refine fails → copy initial-plan.md as final-plan.md, warn
        - Report fails → non-fatal (report_path=None)
        """
        started_at = datetime.now()

        task_content = self._read_task_input(task_input)
        self._preflight(task_content)

        claude_md = self._read_claude_md()

        phases: list[PhaseResult] = []
        agent_results: dict[str, list[AgentResult]] = {}
        report_path: Path | None = None
        archive_path: Path | None = None
        final_plan_path: Path | None = None

        # Emit initial pipeline status
        self._emit_pipeline_status(phases)

        # ----------------------------------------------------------------
        # Phase: plan
        # ----------------------------------------------------------------
        plan_phase = await self._run_plan_phase(task_content, claude_md)
        phases.append(plan_phase)
        agent_results["plan"] = list(plan_phase.agent_results)
        self._emit_pipeline_status(phases)

        if plan_phase.status == PhaseStatus.FAILED:
            return self._build_result(
                phases=phases,
                agent_results=agent_results,
                final_plan_path=None,
                report_path=None,
                archive_path=None,
                started_at=started_at,
                success=False,
            )

        # ----------------------------------------------------------------
        # Audit / refine rounds
        # ----------------------------------------------------------------
        # Keep track of accumulated audit content for prompt context
        prior_audits: dict[str, str] = {}

        for round_num in range(1, self._audit_rounds + 1):
            # Read the plan content to audit (latest available)
            plan_content = self._workspace.read_file("final-plan.md")
            if plan_content is None:
                plan_content = self._workspace.read_file("initial-plan.md") or ""

            # a) Audit phase
            audit_phase_name = "audit" if round_num == 1 else f"audit-r{round_num}"
            audit_phase = await self._run_audit_phase(
                round_num=round_num,
                phase_name=audit_phase_name,
                plan_content=plan_content,
                task_content=task_content,
                claude_md=claude_md,
                prior_audits=prior_audits if prior_audits else None,
            )
            phases.append(audit_phase)
            agent_results[audit_phase_name] = list(audit_phase.agent_results)
            self._emit_pipeline_status(phases)

            # Collect non-empty audit results for refinement
            audit_reports = self._collect_audit_reports(round_num, audit_phase)

            if not audit_reports:
                self._ui.on_log(
                    "warning",
                    f"All auditors failed in round {round_num}. Skipping refinement.",
                )
                # Accumulate empty round in prior_audits for next round context
                continue

            # b) Refine phase
            refine_phase_name = "refine" if round_num == 1 else f"refine-r{round_num}"
            refine_phase = await self._run_refine_phase(
                round_num=round_num,
                phase_name=refine_phase_name,
                plan_content=plan_content,
                task_content=task_content,
                claude_md=claude_md,
                audit_reports=audit_reports,
                prior_audits=prior_audits if prior_audits else None,
            )
            phases.append(refine_phase)
            agent_results[refine_phase_name] = list(refine_phase.agent_results)
            self._emit_pipeline_status(phases)

            if refine_phase.status == PhaseStatus.FAILED:
                self._ui.on_log(
                    "warning",
                    f"Refinement failed in round {round_num}. "
                    "Using previous plan content as final.",
                )
                # Copy previous plan to final-plan.md so downstream always has it
                self._copy_to_final_plan(plan_content)

            # Accumulate current round audits into prior_audits for next round
            prior_audits[f"Round {round_num}"] = "\n\n".join(audit_reports.values())

        # ----------------------------------------------------------------
        # Determine final plan path
        # ----------------------------------------------------------------
        final_plan_path = self._resolve_final_plan_path()

        # ----------------------------------------------------------------
        # Phase: report (non-fatal)
        # ----------------------------------------------------------------
        report_phase_result, report_path, archive_path = self._run_report_phase(
            phases=phases,
            agent_results=agent_results,
            final_plan_path=final_plan_path,
            started_at=started_at,
        )
        if report_phase_result is not None:
            phases.append(report_phase_result)

        return self._build_result(
            phases=phases,
            agent_results=agent_results,
            final_plan_path=final_plan_path,
            report_path=report_path,
            archive_path=archive_path,
            started_at=started_at,
            success=True,
        )

    # ------------------------------------------------------------------
    # Private — setup helpers
    # ------------------------------------------------------------------

    def _preflight(self, task_content: str) -> None:
        """Validate agents, create workspace, save task-input.md."""
        all_agents = [self._planner, *self._auditors]
        missing = self._registry.validate(all_agents)
        if missing:
            raise ValueError(
                f"Agents not available (binary not on PATH): {', '.join(missing)}"
            )

        self._workspace.set_task_slug(task_content)
        self._workspace.ensure_dirs(reuse=False)
        self._workspace.write_file("task-input.md", task_content)

    @staticmethod
    def _read_task_input(task_input: str | Path) -> str:
        """Return task content as a string, reading from file if a Path is given."""
        if isinstance(task_input, Path):
            return task_input.read_text(encoding="utf-8")
        return task_input

    def _read_claude_md(self) -> str:
        """Read CLAUDE.md from the project root. Returns empty string if absent."""
        claude_md_path = self._workspace.workspace_dir.parent / "CLAUDE.md"
        if not claude_md_path.exists():
            return ""
        return claude_md_path.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Private — phase runners
    # ------------------------------------------------------------------

    async def _run_plan_phase(
        self,
        task_content: str,
        claude_md: str,
    ) -> PhaseResult:
        """Run the initial planning phase; produces initial-plan.md."""
        prompt = build_plan_prompt(task_content, claude_md)
        output_path = self._workspace.workspace_dir / "initial-plan.md"
        planner_config = self._registry.get(self._planner)
        return await self._phase_runner.run_phase(
            "plan",
            planner_config,
            prompt,
            output_path,
            self._dry_run,
        )

    async def _run_audit_phase(
        self,
        round_num: int,
        phase_name: str,
        plan_content: str,
        task_content: str,
        claude_md: str,
        prior_audits: dict[str, str] | None,
    ) -> PhaseResult:
        """Run parallel audit phase for the given round."""
        if not self._auditors:
            return PhaseResult(
                name=phase_name,
                status=PhaseStatus.SKIPPED,
            )

        prompt = build_audit_prompt(
            round=round_num,
            plan_content=plan_content,
            task_content=task_content,
            claude_md=claude_md,
            prior_audits=prior_audits,
        )

        agents = [
            (
                self._registry.get(auditor),
                prompt,
                self._audit_output_path(auditor, round_num),
            )
            for auditor in self._auditors
        ]

        return await self._phase_runner.run_parallel(phase_name, agents, self._dry_run)

    async def _run_refine_phase(
        self,
        round_num: int,
        phase_name: str,
        plan_content: str,
        task_content: str,
        claude_md: str,
        audit_reports: dict[str, str],
        prior_audits: dict[str, str] | None,
    ) -> PhaseResult:
        """Run the refinement phase incorporating audit feedback."""
        prompt = build_refinement_prompt(
            round=round_num,
            plan_content=plan_content,
            task_content=task_content,
            claude_md=claude_md,
            audit_reports=audit_reports,
            prior_audits=prior_audits,
        )
        output_path = self._workspace.workspace_dir / "final-plan.md"
        planner_config = self._registry.get(self._planner)
        return await self._phase_runner.run_phase(
            phase_name,
            planner_config,
            prompt,
            output_path,
            self._dry_run,
        )

    def _run_report_phase(
        self,
        phases: list[PhaseResult],
        agent_results: dict[str, list[AgentResult]],
        final_plan_path: Path | None,
        started_at: datetime,
    ) -> tuple[PhaseResult | None, Path | None, Path | None]:
        """Generate report and archive. Non-fatal — returns (phase, report_path, archive_path)."""
        # Build an intermediate PlanResult for the report generator
        intermediate = self._build_result(
            phases=phases,
            agent_results=agent_results,
            final_plan_path=final_plan_path,
            report_path=None,
            archive_path=None,
            started_at=started_at,
            success=True,
        )

        report_path: Path | None = None
        archive_path: Path | None = None

        try:
            report_path = generate_plan_report(
                workspace=self._workspace,
                plan_result=intermediate,
                planner=self._planner,
                auditors=self._auditors,
                audit_rounds=self._audit_rounds,
                max_concurrency=self._max_concurrency,
            )
            archive_path = self._workspace.archive()
        except OSError as exc:
            self._ui.on_log("warning", f"Report/archive step failed (non-fatal): {exc}")
            return None, None, None

        report_phase = PhaseResult(
            name="report",
            status=PhaseStatus.DONE,
            output_files=[report_path] if report_path else [],
        )
        return report_phase, report_path, archive_path

    # ------------------------------------------------------------------
    # Private — audit helpers
    # ------------------------------------------------------------------

    def _audit_output_path(self, auditor: str, round_num: int) -> Path:
        """Return the workspace path for a given auditor and round."""
        filename = (
            f"audit-{auditor}.md"
            if round_num == 1
            else f"audit-{auditor}-r{round_num}.md"
        )
        return self._workspace.workspace_dir / filename

    def _collect_audit_reports(
        self,
        round_num: int,
        audit_phase: PhaseResult,
    ) -> dict[str, str]:
        """Return non-empty audit content keyed by auditor name for this round."""
        reports: dict[str, str] = {}
        for agent_result in audit_phase.agent_results:
            if agent_result.output_empty or agent_result.exit_code != 0:
                continue
            file_key = (
                f"audit-{agent_result.agent_name}.md"
                if round_num == 1
                else f"audit-{agent_result.agent_name}-r{round_num}.md"
            )
            content = self._workspace.read_file(file_key)
            if content and content.strip():
                reports[agent_result.agent_name] = content
        return reports

    def _copy_to_final_plan(self, plan_content: str) -> None:
        """Write plan_content to final-plan.md (fallback when refine fails)."""
        self._workspace.write_file("final-plan.md", plan_content)

    def _resolve_final_plan_path(self) -> Path | None:
        """Return path to final-plan.md if it exists, else initial-plan.md."""
        final = self._workspace.workspace_dir / "final-plan.md"
        if final.exists():
            return final
        initial = self._workspace.workspace_dir / "initial-plan.md"
        if initial.exists():
            return initial
        return None

    # ------------------------------------------------------------------
    # Private — result construction
    # ------------------------------------------------------------------

    @staticmethod
    def _sum_phase_costs(phases: list[PhaseResult]) -> Decimal | None:
        """Return total cost across all phases, or None if no cost data."""
        costs = [p.cost_usd for p in phases if p.cost_usd is not None]
        if not costs:
            return None
        return sum(costs, Decimal(0))

    @staticmethod
    def _build_result(
        phases: list[PhaseResult],
        agent_results: dict[str, list[AgentResult]],
        final_plan_path: Path | None,
        report_path: Path | None,
        archive_path: Path | None,
        started_at: datetime,
        success: bool,
    ) -> PlanResult:
        total_cost = PlanWorkflow._sum_phase_costs(phases)
        return PlanResult(
            phases=phases,
            final_plan_path=final_plan_path,
            report_path=report_path,
            archive_path=archive_path,
            total_duration=datetime.now() - started_at,
            total_cost_usd=total_cost,
            agent_results=agent_results,
            success=success,
        )

    # ------------------------------------------------------------------
    # Private — UI helpers
    # ------------------------------------------------------------------

    def _emit_pipeline_status(self, phases: list[PhaseResult]) -> None:
        """Notify the UI of current phase statuses."""
        statuses: dict[str, PhaseStatus] = {p.name: p.status for p in phases}
        self._ui.on_pipeline_update(statuses)
