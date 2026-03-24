from __future__ import annotations

import asyncio
import sys
from pathlib import Path  # noqa: TC003
from typing import Annotated

import typer
from click.core import ParameterSource
from rich.console import Console
from rich.prompt import Prompt

from planora.cli.app import plan_app
from planora.core.config import PlanораSettings

console = Console()


def parse_auditor_csv(csv_input: str) -> list[str]:
    """Split on comma, strip whitespace, remove empty, deduplicate preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for name in csv_input.split(","):
        name = name.strip()
        if name and name not in seen:
            seen.add(name)
            result.append(name)
    return result


def _require_existing_initial_plan(root: Path) -> None:
    """Exit with a clear error when skip-planning has no reusable plan file."""
    initial_plan = root / ".plan-workspace" / "initial-plan.md"
    if initial_plan.exists() and initial_plan.stat().st_size > 0:
        return

    console.print(
        "[red]--skip-planning requires an existing non-empty .plan-workspace/initial-plan.md[/red]"
    )
    raise typer.Exit(1)


def _detect_completed_rounds(
    workspace_dir: Path,
    auditors: list[str],
    total_rounds: int,
) -> set[int]:
    """Return the set of round numbers whose audit outputs are all present and non-empty.

    A round is considered complete when every auditor in the list has produced a
    non-empty audit file for that round.  The final-plan.md produced by refinement
    is checked for round 1; for subsequent rounds the refined plan overwrites the
    same file so we only test audit file presence.
    """
    completed: set[int] = set()
    if not auditors:
        return completed
    for round_num in range(1, total_rounds + 1):
        audit_files_present = all(
            _audit_file_nonempty(workspace_dir, auditor, round_num) for auditor in auditors
        )
        if audit_files_present:
            completed.add(round_num)
    return completed


def _audit_file_nonempty(workspace_dir: Path, auditor: str, round_num: int) -> bool:
    """Return True when the audit output file for (auditor, round) is present and non-empty."""
    filename = f"audit-{auditor}.md" if round_num == 1 else f"audit-{auditor}-r{round_num}.md"
    path = workspace_dir / filename
    return path.exists() and path.stat().st_size > 0


def _option_was_supplied(ctx: typer.Context, name: str) -> bool:
    """Return True when an option came from the command line instead of its default."""
    return ctx.get_parameter_source(name) != ParameterSource.DEFAULT


def _resolve_input_mode(
    task: str | None,
    task_file: Path | None,
    interactive: bool,
    tui: bool,
) -> str:
    """Determine input mode: 'run', 'wizard', or 'tui'.

    Resolution order:
    1. --tui flag → 'tui' (if TTY), else fall through
    2. --interactive / -i → 'wizard'
    3. No task and no task-file → auto-detect from TTY state
    4. Otherwise → 'run'
    """
    if tui:
        if not sys.stdin.isatty():
            console.print("[yellow]Warning:[/] --tui ignored in non-TTY context")
        else:
            return "tui"

    if interactive:
        return "wizard"

    if task is None and task_file is None:
        if sys.stdin.isatty() and sys.stdout.isatty():
            return "wizard"
        if not sys.stdin.isatty():
            return "run"  # piped stdin
        console.print("[red]Error:[/] No task provided and not in interactive terminal")
        raise typer.Exit(1)

    return "run"


@plan_app.command("run")
def plan_run(
    ctx: typer.Context,
    task: Annotated[str | None, typer.Argument(help="Task description")] = None,
    task_file: Annotated[Path | None, typer.Option(help="Read task from file")] = None,
    planner: Annotated[str, typer.Option(help="Planner agent")] = "claude",
    auditors: Annotated[str, typer.Option(help="Comma-separated auditor list")] = "gemini,codex",
    audit_rounds: Annotated[
        int, typer.Option(help="Audit+refine cycles (1 or 2)", min=1, max=2)
    ] = 1,
    concurrency: Annotated[int, typer.Option(help="Max parallel auditors", min=1)] = 3,
    skip_planning: Annotated[bool, typer.Option(help="Reuse existing initial-plan.md")] = False,
    skip_audit: Annotated[bool, typer.Option(help="Skip audit phases")] = False,
    skip_refinement: Annotated[bool, typer.Option(help="Skip refinement phases")] = False,
    dry_run: Annotated[bool, typer.Option(help="Show commands without executing")] = False,
    interactive: Annotated[bool, typer.Option("-i", help="Launch interactive wizard")] = False,
    tui: Annotated[bool, typer.Option(help="Launch TUI dashboard")] = False,
    output_format: Annotated[
        str, typer.Option(help="Output format: text (default), events (JSONL on stderr)")
    ] = "text",
    project_root: Annotated[
        Path | None, typer.Option(help="Override project root detection")
    ] = None,
    stall_timeout: Annotated[
        float, typer.Option(help="Override normal stall threshold (seconds)")
    ] = 300.0,
    deep_timeout: Annotated[
        float, typer.Option(help="Override deep tool stall threshold (seconds)")
    ] = 600.0,
    profile: Annotated[
        str | None,
        typer.Option(help="Activate named profile from planora.toml"),
    ] = None,
    config: Annotated[
        list[str] | None,
        typer.Option(help="Override config key=value (TOML syntax, dot notation)"),
    ] = None,
) -> None:
    """Run multi-agent implementation planning."""

    # 1. Resolve input mode
    mode = _resolve_input_mode(task, task_file, interactive, tui)

    # 2. Wizard mode: prompt for task interactively
    if mode == "wizard":
        task = Prompt.ask("[bold]Enter task description[/bold]")
        if not task:
            console.print("[red]No task provided[/red]")
            raise typer.Exit(1)

    # 3. Resolve task content from the first available source
    task_content: str
    if task is not None:
        task_content = task
    elif task_file is not None:
        try:
            task_content = task_file.read_text(encoding="utf-8")
        except OSError as exc:
            console.print(f"[red]Error reading task file: {exc}[/red]")
            raise typer.Exit(1) from exc
    elif not sys.stdin.isatty():
        task_content = sys.stdin.read()
    else:
        console.print("[red]No task input provided[/red]")
        raise typer.Exit(1)

    if not task_content.strip():
        console.print("[red]Task content is empty[/red]")
        raise typer.Exit(1)

    # 4. Load and resolve configuration layers
    try:
        settings = PlanораSettings()
        if profile is not None:
            settings = settings.with_profile(profile)
        if config:
            settings = settings.with_config_overrides(config)
        cli_config_overrides: list[str] = []
        if _option_was_supplied(ctx, "stall_timeout"):
            cli_config_overrides.append(f"observability.stall_timeout={stall_timeout}")
        if _option_was_supplied(ctx, "deep_timeout"):
            cli_config_overrides.append(f"observability.deep_tool_timeout={deep_timeout}")
        if cli_config_overrides:
            settings = settings.with_config_overrides(cli_config_overrides)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    resolved_planner = (
        planner if _option_was_supplied(ctx, "planner") else settings.effective_planner
    )
    resolved_auditors = (
        parse_auditor_csv(auditors)
        if _option_was_supplied(ctx, "auditors")
        else settings.effective_auditors
    )
    resolved_audit_rounds = (
        audit_rounds
        if _option_was_supplied(ctx, "audit_rounds")
        else settings.effective_audit_rounds
    )
    resolved_concurrency = (
        concurrency if _option_was_supplied(ctx, "concurrency") else settings.effective_concurrency
    )

    root_setting = (
        project_root.resolve()
        if _option_was_supplied(ctx, "project_root") and project_root is not None
        else settings.effective_project_root
    )
    root = root_setting

    # 5. Parse and apply skip flags
    auditor_list = list(resolved_auditors)
    if skip_audit:
        auditor_list = []
    effective_rounds = resolved_audit_rounds
    if skip_planning:
        _require_existing_initial_plan(root)

    # 6. Configure prompt templates and shared dependencies
    from planora.agents.registry import AgentRegistry
    from planora.agents.runner import AgentRunner
    from planora.prompts.plan import configure_prompt_templates

    configure_prompt_templates(
        plan=settings.prompts.plan,
        audit=settings.prompts.audit,
        refine=settings.prompts.refine,
        base_dir=settings.effective_prompt_base_dir,
    )
    registry = AgentRegistry.from_settings(settings)
    runner = AgentRunner()

    # 7. Fail fast: validate planner binary before creating workspace
    missing = registry.validate([resolved_planner])
    if missing:
        console.print(f"[red]Planner '{resolved_planner}' not available (binary not on PATH)[/red]")
        raise typer.Exit(1)

    # 8. TUI mode — optional extra; fall back gracefully
    if mode == "tui":
        try:
            from planora.tui.app import PlanoraTUI

            PlanoraTUI(
                task_input=task_content,
                planner=resolved_planner,
                auditors=auditor_list,
                audit_rounds=effective_rounds,
                max_concurrency=resolved_concurrency,
                project_root=root,
                registry=registry,
                runner=runner,
                plan_template_path=settings.prompts.plan,
                audit_template_path=settings.prompts.audit,
                refine_template_path=settings.prompts.refine,
                prompt_base_dir=settings.effective_prompt_base_dir,
            ).run()
            return
        except ImportError:
            console.print(
                "[yellow]Warning:[/] TUI mode requires the 'tui' extra. "
                "Install with: uv pip install 'planora[tui]'"
            )
            console.print("Falling back to CLI mode.")

    # 9. Lazy imports — keep module-level startup fast, avoid circular imports
    from planora.cli.callbacks import CLICallback, EventsOutputCallback
    from planora.core.events import PhaseStatus
    from planora.core.workspace import WorkspaceManager
    from planora.workflow.plan import PlanWorkflow

    # 10. Choose UI callback based on output format
    ui: CLICallback | EventsOutputCallback
    ui = EventsOutputCallback() if output_format == "events" else CLICallback()

    # 11. Build and run the workflow
    workspace = WorkspaceManager(root, reports_dir=Path(settings.effective_reports_dir))
    workflow = PlanWorkflow(
        workspace=workspace,
        registry=registry,
        runner=runner,
        ui=ui,
        planner=resolved_planner,
        auditors=auditor_list,
        audit_rounds=effective_rounds,
        max_concurrency=resolved_concurrency,
        dry_run=dry_run,
        skip_planning=skip_planning,
        skip_refinement=skip_refinement,
        reuse_workspace=skip_planning,
        snapshot_interval=settings.effective_cli_status_interval,
        stall_check_interval=settings.effective_monitor_interval,
        plan_template_path=settings.prompts.plan,
        audit_template_path=settings.prompts.audit,
        refine_template_path=settings.prompts.refine,
        prompt_base_dir=settings.effective_prompt_base_dir,
        settings=settings,
    )

    result = asyncio.run(workflow.run(task_content))

    # 11. Print summary
    if result.success:
        console.print("\n[bold green]Planning complete![/bold green]")
        if result.final_plan_path:
            console.print(f"  Plan: {result.final_plan_path}")
        if result.archive_path:
            console.print(f"  Archive: {result.archive_path}")
        return

    console.print("\n[bold red]Planning failed[/bold red]")
    for phase in result.phases:
        if phase.status == PhaseStatus.FAILED:
            console.print(f"  Failed phase: {phase.name}")
            if phase.error:
                console.print(f"  Error: {phase.error}")
    raise typer.Exit(1)


@plan_app.command("wizard")
def plan_wizard(
    planner: Annotated[str, typer.Option(help="Planner agent")] = "claude",
    auditors: Annotated[str, typer.Option(help="Comma-separated auditor list")] = "gemini,codex",
    audit_rounds: Annotated[
        int, typer.Option(help="Audit+refine cycles (1 or 2)", min=1, max=2)
    ] = 1,
    concurrency: Annotated[int, typer.Option(help="Max parallel auditors", min=1)] = 3,
    dry_run: Annotated[bool, typer.Option(help="Show commands without executing")] = False,
    output_format: Annotated[
        str, typer.Option(help="Output format: text (default), events (JSONL on stderr)")
    ] = "text",
    project_root: Annotated[
        Path | None, typer.Option(help="Override project root detection")
    ] = None,
    profile: Annotated[
        str | None,
        typer.Option(help="Activate named profile from planora.toml"),
    ] = None,
    config: Annotated[
        list[str] | None,
        typer.Option(help="Override config key=value (TOML syntax, dot notation)"),
    ] = None,
) -> None:
    """Launch the interactive task wizard and run the planning workflow."""
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        console.print("[red]Error:[/] wizard requires an interactive terminal")
        raise typer.Exit(1)

    task = Prompt.ask("[bold]Enter task description[/bold]")
    if not task.strip():
        console.print("[red]No task provided[/red]")
        raise typer.Exit(1)

    try:
        settings = PlanораSettings()
        if profile is not None:
            settings = settings.with_profile(profile)
        if config:
            settings = settings.with_config_overrides(config)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    resolved_planner = planner
    resolved_auditors = parse_auditor_csv(auditors)
    root = project_root.resolve() if project_root is not None else settings.effective_project_root

    from planora.agents.registry import AgentRegistry
    from planora.agents.runner import AgentRunner
    from planora.cli.callbacks import CLICallback, EventsOutputCallback
    from planora.core.events import PhaseStatus
    from planora.core.workspace import WorkspaceManager
    from planora.prompts.plan import configure_prompt_templates
    from planora.workflow.plan import PlanWorkflow

    configure_prompt_templates(
        plan=settings.prompts.plan,
        audit=settings.prompts.audit,
        refine=settings.prompts.refine,
        base_dir=settings.effective_prompt_base_dir,
    )
    registry = AgentRegistry.from_settings(settings)
    runner = AgentRunner()

    missing = registry.validate([resolved_planner])
    if missing:
        console.print(f"[red]Planner '{resolved_planner}' not available (binary not on PATH)[/red]")
        raise typer.Exit(1)

    ui: CLICallback | EventsOutputCallback
    ui = EventsOutputCallback() if output_format == "events" else CLICallback()

    workspace = WorkspaceManager(root, reports_dir=Path(settings.effective_reports_dir))
    workflow = PlanWorkflow(
        workspace=workspace,
        registry=registry,
        runner=runner,
        ui=ui,
        planner=resolved_planner,
        auditors=resolved_auditors,
        audit_rounds=audit_rounds,
        max_concurrency=concurrency,
        dry_run=dry_run,
        snapshot_interval=settings.effective_cli_status_interval,
        stall_check_interval=settings.effective_monitor_interval,
        plan_template_path=settings.prompts.plan,
        audit_template_path=settings.prompts.audit,
        refine_template_path=settings.prompts.refine,
        prompt_base_dir=settings.effective_prompt_base_dir,
        settings=settings,
    )

    result = asyncio.run(workflow.run(task))

    if result.success:
        console.print("\n[bold green]Planning complete![/bold green]")
        if result.final_plan_path:
            console.print(f"  Plan: {result.final_plan_path}")
        if result.archive_path:
            console.print(f"  Archive: {result.archive_path}")
        return

    console.print("\n[bold red]Planning failed[/bold red]")
    for phase in result.phases:
        if phase.status == PhaseStatus.FAILED:
            console.print(f"  Failed phase: {phase.name}")
            if phase.error:
                console.print(f"  Error: {phase.error}")
    raise typer.Exit(1)


@plan_app.command("resume")
def plan_resume(
    project_root: Annotated[Path | None, typer.Option(help="Project root")] = None,
    tui: Annotated[bool, typer.Option(help="Launch TUI dashboard")] = False,
    output_format: Annotated[
        str, typer.Option(help="Output format: text (default), events (JSONL on stderr)")
    ] = "text",
) -> None:
    """Resume an interrupted plan run from the existing workspace."""
    _ = tui  # tui flag reserved for future use

    settings = PlanораSettings()
    root = project_root.resolve() if project_root is not None else settings.effective_project_root
    workspace_dir = root / ".plan-workspace"

    # 1. Workspace must exist
    if not workspace_dir.exists():
        console.print("[red]No workspace to resume from (.plan-workspace/ not found)[/red]")
        raise typer.Exit(1)

    # 2. task-input.md is required — no task, nothing to resume
    task_file = workspace_dir / "task-input.md"
    if not task_file.exists():
        console.print("[red]Missing task-input.md in workspace[/red]")
        raise typer.Exit(1)

    task_content = task_file.read_text(encoding="utf-8")

    # 3. Inspect completed state
    initial_plan = workspace_dir / "initial-plan.md"
    final_plan = workspace_dir / "final-plan.md"
    report = workspace_dir / "plan-report.md"

    has_initial = initial_plan.exists() and initial_plan.stat().st_size > 0
    has_final = final_plan.exists() and final_plan.stat().st_size > 0
    has_report = report.exists() and report.stat().st_size > 0

    if has_report:
        console.print("[green]Run already complete. All files present.[/green]")
        return

    # 4. Infer auditor list from existing audit-*.md filenames
    auditor_names: list[str] = []
    for audit_file in sorted(workspace_dir.glob("audit-*.md")):
        stem = audit_file.stem.removeprefix("audit-")
        # Strip round suffix (e.g. "-r2") leaving just the agent name
        if "-r" in stem:
            stem = stem.rsplit("-r", 1)[0]
        if stem and stem not in auditor_names:
            auditor_names.append(stem)

    # 5. Determine what needs to be (re-)run
    if not has_initial:
        console.print("Restarting from plan phase...")
    elif not has_final:
        console.print("Restarting from audit/refine phase...")
    else:
        console.print("Restarting from report phase...")

    # 6. Lazy imports
    from planora.agents.registry import AgentRegistry
    from planora.agents.runner import AgentRunner
    from planora.cli.callbacks import CLICallback, EventsOutputCallback
    from planora.core.workspace import WorkspaceManager
    from planora.prompts.plan import configure_prompt_templates
    from planora.workflow.plan import PlanWorkflow

    # 7. Effective workflow parameters derived from state detection
    effective_auditors = auditor_names if auditor_names else settings.effective_auditors
    has_r2_audits = any(workspace_dir.glob("audit-*-r2.md"))
    effective_rounds = 2 if has_r2_audits else 1

    # Detect which audit+refine rounds are fully complete so the workflow
    # can skip them rather than re-running or wiping their outputs.
    completed_rounds = _detect_completed_rounds(workspace_dir, effective_auditors, effective_rounds)

    configure_prompt_templates(
        plan=settings.prompts.plan,
        audit=settings.prompts.audit,
        refine=settings.prompts.refine,
        base_dir=settings.effective_prompt_base_dir,
    )

    # 8. Choose UI callback
    ui: CLICallback | EventsOutputCallback
    ui = EventsOutputCallback() if output_format == "events" else CLICallback()

    # 9. Run workflow against the existing workspace; always reuse=True so
    #    partial outputs produced in a previous run are never wiped.
    workspace = WorkspaceManager(root, reports_dir=Path(settings.effective_reports_dir))
    workflow = PlanWorkflow(
        workspace=workspace,
        registry=AgentRegistry.from_settings(settings),
        runner=AgentRunner(),
        ui=ui,
        planner=settings.effective_planner,
        auditors=effective_auditors,
        audit_rounds=effective_rounds,
        dry_run=False,
        skip_planning=has_initial,
        reuse_workspace=True,
        completed_rounds=completed_rounds,
        snapshot_interval=settings.effective_cli_status_interval,
        stall_check_interval=settings.effective_monitor_interval,
        plan_template_path=settings.prompts.plan,
        audit_template_path=settings.prompts.audit,
        refine_template_path=settings.prompts.refine,
        prompt_base_dir=settings.effective_prompt_base_dir,
        settings=settings,
    )

    result = asyncio.run(workflow.run(task_content))

    if result.success:
        console.print("\n[bold green]Resume complete![/bold green]")
        if result.final_plan_path:
            console.print(f"  Plan: {result.final_plan_path}")
        if result.archive_path:
            console.print(f"  Archive: {result.archive_path}")
        return

    console.print("\n[bold red]Resume failed[/bold red]")
    raise typer.Exit(1)
