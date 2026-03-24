from __future__ import annotations

import asyncio
import contextlib
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
from planora.workflow.engine import PhaseRunner, WorkflowControl
from planora.workflow.report import generate_plan_report

if TYPE_CHECKING:
    from planora.agents.registry import AgentRegistry
    from planora.agents.runner import AgentRunner
    from planora.core.config import PlanораSettings
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
        skip_planning: bool = False,
        skip_refinement: bool = False,
        reuse_workspace: bool = False,
        completed_rounds: set[int] | None = None,
        snapshot_interval: float | None = None,
        stall_check_interval: float = 5.0,
        control: WorkflowControl | None = None,
        plan_template_path: Path | None = None,
        audit_template_path: Path | None = None,
        refine_template_path: Path | None = None,
        prompt_base_dir: Path = Path("."),
        settings: PlanораSettings | None = None,
    ) -> None:
        from planora.observability.hooks import ClaudeHooksManager
        from planora.observability.telemetry import PlanoraTelemetry

        self._workspace = workspace
        self._registry = registry
        self._ui = ui
        self._planner = planner
        self._auditors: list[str] = auditors if auditors is not None else []
        self._audit_rounds = audit_rounds
        self._max_concurrency = max_concurrency
        self._dry_run = dry_run
        self._skip_planning = skip_planning
        self._skip_refinement = skip_refinement
        self._reuse_workspace = reuse_workspace
        self._completed_rounds: set[int] = (
            completed_rounds if completed_rounds is not None else set()
        )
        self._plan_template_path = plan_template_path
        self._audit_template_path = audit_template_path
        self._refine_template_path = refine_template_path
        self._prompt_base_dir = prompt_base_dir
        self._control = control
        self._telemetry = PlanoraTelemetry(settings) if settings is not None else None
        _hooks_manager = (
            ClaudeHooksManager(workspace.workspace_dir.parent) if settings is not None else None
        )
        self._phase_runner = PhaseRunner(
            runner,
            ui,
            max_concurrency,
            snapshot_interval=snapshot_interval,
            stall_check_interval=stall_check_interval,
            control=control,
            telemetry=self._telemetry,
            hooks_manager=_hooks_manager,
        )
        self._started_at: datetime | None = None
        self._task_content: str | None = None
        self._claude_md = ""
        self._phases: list[PhaseResult] = []
        self._agent_results: dict[str, list[AgentResult]] = {}
        self._prior_audits: dict[str, str] = {}
        self._round_audit_reports: dict[int, dict[str, str]] = {}
        self._report_path: Path | None = None
        self._archive_path: Path | None = None
        self._final_plan_path: Path | None = None

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

        task_content = await self._read_task_input(task_input)
        await self._preflight(task_content)
        await self._areset_run_state(task_content=task_content, started_at=started_at)

        pipeline_ctx = (
            self._telemetry.pipeline_span(self._workspace._task_slug)
            if self._telemetry is not None
            else contextlib.nullcontext()
        )
        with pipeline_ctx:
            phases = self._phases
            agent_results = self._agent_results

            # Emit initial pipeline status
            self._emit_pipeline_status(phases)

            # ----------------------------------------------------------------
            # Phase: plan
            # ----------------------------------------------------------------
            if self._skip_planning:
                plan_phase = self._build_skipped_plan_phase()
                self._ui.on_log(
                    "info",
                    "Skipping planning and reusing existing initial-plan.md.",
                )
            else:
                plan_phase = await self.phase_plan()
            self._record_phase_result(plan_phase)
            self._emit_pipeline_status(phases)

            if plan_phase.status == PhaseStatus.FAILED:
                return self._build_result(
                    phases=phases,
                    agent_results=agent_results,
                    final_plan_path=None,
                    report_path=None,
                    archive_path=None,
                    started_at=self._require_started_at(),
                    success=False,
                )

            # Pause checkpoint -- honoured at each inter-phase boundary
            if self._control is not None:
                await self._control.wait_if_paused()

            # ----------------------------------------------------------------
            # Audit / refine rounds
            # ----------------------------------------------------------------
            for round_num in range(1, self._audit_rounds + 1):
                if round_num in self._completed_rounds:
                    self._ui.on_log(
                        "info",
                        f"Skipping completed round {round_num}.",
                    )
                    continue

                audit_phase = await self.phase_audit(round_num)
                self._record_phase_result(audit_phase)
                self._emit_pipeline_status(phases)

                if self._control is not None:
                    await self._control.wait_if_paused()

                # Collect non-empty audit results for refinement
                audit_reports = dict(self._round_audit_reports.get(round_num, {}))

                if not audit_reports:
                    self._ui.on_log(
                        "warning",
                        f"All auditors failed in round {round_num}. Skipping refinement.",
                    )
                    continue

                # b) Refine phase
                plan_content = await self._current_plan_content()
                refine_phase = await self.phase_refine(round_num)
                self._record_phase_result(refine_phase)
                self._emit_pipeline_status(phases)

                if self._control is not None:
                    await self._control.wait_if_paused()

                if refine_phase.status == PhaseStatus.FAILED:
                    self._ui.on_log(
                        "warning",
                        f"Refinement failed in round {round_num}. "
                        "Using previous plan content as final.",
                    )
                    # Copy previous plan to final-plan.md so downstream always has it
                    await self._copy_to_final_plan(plan_content)

                # Accumulate current round audits into prior_audits for next round
                self._prior_audits[f"Round {round_num}"] = "\n\n".join(audit_reports.values())

            # ----------------------------------------------------------------
            # Determine final plan path
            # ----------------------------------------------------------------
            self._final_plan_path = self._resolve_final_plan_path()

            # ----------------------------------------------------------------
            # Phase: report (non-fatal)
            # ----------------------------------------------------------------
            report_phase_result = await self.phase_report()
            if report_phase_result is not None:
                phases.append(report_phase_result)

            return self._build_result(
                phases=phases,
                agent_results=agent_results,
                final_plan_path=self._final_plan_path,
                report_path=self._report_path,
                archive_path=self._archive_path,
                started_at=self._require_started_at(),
                success=True,
            )

    async def phase_plan(self) -> PhaseResult:
        """Run the initial planning phase; produces initial-plan.md."""
        prompt = build_plan_prompt(
            self._require_task_content(),
            self._claude_md,
            template_path=self._plan_template_path,
            base_dir=self._prompt_base_dir,
        )
        output_path = self._workspace.workspace_dir / "initial-plan.md"
        planner_config = self._registry.get(self._planner)
        return await self._phase_runner.run_phase(
            "plan",
            planner_config,
            prompt,
            output_path,
            self._dry_run,
        )

    async def phase_audit(self, round: int) -> PhaseResult:
        """Run one audit round against the latest available plan."""
        if not self._auditors:
            return PhaseResult(
                name=self._audit_phase_name(round),
                status=PhaseStatus.SKIPPED,
            )

        prompt = build_audit_prompt(
            round=round,
            plan_content=await self._current_plan_content(),
            task_content=self._require_task_content(),
            claude_md=self._claude_md,
            prior_audits=self._prior_audits or None,
            template_path=self._audit_template_path,
            base_dir=self._prompt_base_dir,
        )

        phase_result = await self._run_parallel_audits(
            self._auditors,
            prompt,
            round,
        )
        self._round_audit_reports[round] = await self._collect_audit_reports(round, phase_result)
        return phase_result

    async def phase_refine(self, round: int) -> PhaseResult:
        """Run one refinement round using the current round's audit outputs."""
        phase_name = self._refine_phase_name(round)
        audit_reports = self._round_audit_reports.get(round, {})
        if not audit_reports:
            return PhaseResult(name=phase_name, status=PhaseStatus.SKIPPED)

        prompt = build_refinement_prompt(
            round=round,
            plan_content=await self._current_plan_content(),
            task_content=self._require_task_content(),
            claude_md=self._claude_md,
            audit_reports=audit_reports,
            prior_audits=self._prior_audits or None,
            template_path=self._refine_template_path,
            base_dir=self._prompt_base_dir,
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

    async def phase_report(self) -> PhaseResult | None:
        """Generate the report and archive for the current run state."""
        report_phase_result, report_path, archive_path = await self._run_report_phase(
            phases=self._phases,
            agent_results=self._agent_results,
            final_plan_path=self._final_plan_path,
            started_at=self._require_started_at(),
        )
        self._report_path = report_path
        self._archive_path = archive_path
        return report_phase_result

    # ------------------------------------------------------------------
    # Private — setup helpers
    # ------------------------------------------------------------------

    async def _preflight(self, task_content: str) -> None:
        """Validate agents, create workspace, save task-input.md."""
        all_agents = [self._planner, *self._auditors]
        missing = self._registry.validate(all_agents)
        if missing:
            raise ValueError(f"Agents not available (binary not on PATH): {', '.join(missing)}")

        self._workspace.set_task_slug(task_content)
        self._workspace.ensure_dirs(reuse=self._reuse_workspace or self._skip_planning)
        if self._skip_planning:
            initial_plan_path = self._workspace.workspace_dir / "initial-plan.md"
            if not initial_plan_path.exists() or initial_plan_path.stat().st_size == 0:
                raise ValueError(
                    "skip_planning requires an existing non-empty initial-plan.md "
                    "in .plan-workspace/."
                )
        await self._workspace.awrite_file("task-input.md", task_content)

    @staticmethod
    async def _read_task_input(task_input: str | Path) -> str:
        """Return task content as a string, reading from file via thread pool if a Path is given."""
        if isinstance(task_input, Path):
            return await asyncio.to_thread(task_input.read_text, encoding="utf-8")
        return task_input

    async def _read_claude_md(self) -> str:
        """Read CLAUDE.md from the project root. Returns empty string if absent."""
        claude_md_path = self._workspace.workspace_dir.parent / "CLAUDE.md"
        if not claude_md_path.exists():
            return ""
        return await asyncio.to_thread(claude_md_path.read_text, encoding="utf-8")

    # ------------------------------------------------------------------
    # Private — phase runners
    # ------------------------------------------------------------------

    async def _run_parallel_audits(
        self,
        agents: list[str],
        prompt: str,
        round: int,
    ) -> PhaseResult:
        """Run multiple auditors in parallel for a single audit round."""
        phase_name = self._audit_phase_name(round)
        work_items = [
            (
                self._registry.get(auditor),
                prompt,
                self._audit_output_path(auditor, round),
            )
            for auditor in agents
        ]
        return await self._phase_runner.run_parallel(
            phase_name,
            work_items,
            self._dry_run,
        )

    async def _run_report_phase(
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
        filename = f"audit-{auditor}.md" if round_num == 1 else f"audit-{auditor}-r{round_num}.md"
        return self._workspace.workspace_dir / filename

    async def _collect_audit_reports(
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
            file_content = await self._workspace.aread_file(file_key)
            if file_content and file_content.strip():
                reports[agent_result.agent_name] = file_content
        return reports

    async def _copy_to_final_plan(self, plan_content: str) -> None:
        """Write plan_content to final-plan.md (fallback when refine fails)."""
        await self._workspace.awrite_file("final-plan.md", plan_content)

    async def _current_plan_content(self) -> str:
        """Return the latest available plan text for audit/refine inputs."""
        plan_content = await self._workspace.aread_file("final-plan.md")
        if plan_content is not None:
            return plan_content
        return await self._workspace.aread_file("initial-plan.md") or ""

    def _resolve_final_plan_path(self) -> Path | None:
        """Return path to final-plan.md if it exists and is non-empty, else initial-plan.md."""
        final = self._workspace.workspace_dir / "final-plan.md"
        if final.exists() and final.stat().st_size > 0:
            return final
        initial = self._workspace.workspace_dir / "initial-plan.md"
        if initial.exists() and initial.stat().st_size > 0:
            return initial
        return None

    # ------------------------------------------------------------------
    # Private — result construction
    # ------------------------------------------------------------------

    async def _areset_run_state(self, *, task_content: str, started_at: datetime) -> None:
        """Reset per-run state before entering the workflow pipeline."""
        self._started_at = started_at
        self._task_content = task_content
        self._claude_md = await self._read_claude_md()
        self._phases = []
        self._agent_results = {}
        self._prior_audits = {}
        self._round_audit_reports = {}
        self._report_path = None
        self._archive_path = None
        self._final_plan_path = None

    def _build_skipped_plan_phase(self) -> PhaseResult:
        """Return a skipped plan phase that preserves the existing initial plan."""
        initial_plan_path = self._workspace.workspace_dir / "initial-plan.md"
        return PhaseResult(
            name="plan",
            status=PhaseStatus.SKIPPED,
            output_files=[initial_plan_path],
        )

    def _record_phase_result(self, phase: PhaseResult) -> None:
        """Append a phase result and mirror its agent outputs in the run summary."""
        self._phases.append(phase)
        self._agent_results[phase.name] = list(phase.agent_results)

    @staticmethod
    def _audit_phase_name(round_num: int) -> str:
        """Return the audit phase name for the given round."""
        return "audit" if round_num == 1 else f"audit-r{round_num}"

    @staticmethod
    def _refine_phase_name(round_num: int) -> str:
        """Return the refine phase name for the given round."""
        return "refine" if round_num == 1 else f"refine-r{round_num}"

    def _require_started_at(self) -> datetime:
        """Return the run start time or raise if the workflow is uninitialized."""
        if self._started_at is None:
            raise RuntimeError("PlanWorkflow has not been initialized for a run.")
        return self._started_at

    def _require_task_content(self) -> str:
        """Return the task content or raise if the workflow is uninitialized."""
        if self._task_content is None:
            raise RuntimeError("PlanWorkflow has not been initialized for a run.")
        return self._task_content

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
