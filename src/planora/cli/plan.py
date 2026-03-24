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
    task_file: Annotated[
        Path | None, typer.Option(help="Read task from file")
    ] = None,
    planner: Annotated[str, typer.Option(help="Planner agent")] = "claude",
    auditors: Annotated[
        str, typer.Option(help="Comma-separated auditor list")
    ] = "gemini,codex",
    audit_rounds: Annotated[
        int, typer.Option(help="Audit+refine cycles (1 or 2)", min=1, max=2)
    ] = 1,
    concurrency: Annotated[
        int, typer.Option(help="Max parallel auditors", min=1)
    ] = 3,
    skip_planning: Annotated[
        bool, typer.Option(help="Reuse existing initial-plan.md")
    ] = False,
    skip_audit: Annotated[bool, typer.Option(help="Skip audit phases")] = False,
    skip_refinement: Annotated[
        bool, typer.Option(help="Skip refinement phases")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option(help="Show commands without executing")
    ] = False,
    interactive: Annotated[
        bool, typer.Option("-i", help="Launch interactive wizard")
    ] = False,
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
    _ = skip_planning

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
            cli_config_overrides.append(
                f"observability.stall_timeout={stall_timeout}"
            )
        if _option_was_supplied(ctx, "deep_timeout"):
            cli_config_overrides.append(
                f"observability.deep_tool_timeout={deep_timeout}"
            )
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
        concurrency
        if _option_was_supplied(ctx, "concurrency")
        else settings.effective_concurrency
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
    effective_rounds = 0 if skip_refinement else resolved_audit_rounds

    from planora.prompts.plan import configure_prompt_templates

    configure_prompt_templates(
        plan=settings.prompts.plan,
        audit=settings.prompts.audit,
        refine=settings.prompts.refine,
        base_dir=settings.effective_prompt_base_dir,
    )

    # 6. TUI mode — optional extra; fall back gracefully
    if mode == "tui":
        try:
            from planora.tui.app import PlanoraTUI

            PlanoraTUI(
                task_input=task_content,
                planner=resolved_planner,
                auditors=auditor_list,
                audit_rounds=effective_rounds,
                max_concurrency=resolved_concurrency,
            ).run()
            return
        except ImportError:
            console.print(
                "[yellow]Warning:[/] TUI mode requires the 'tui' extra. "
                "Install with: uv pip install 'planora[tui]'"
            )
            console.print("Falling back to CLI mode.")

    # 7. Lazy imports — keep module-level startup fast, avoid circular imports
    from planora.agents.registry import AgentRegistry
    from planora.agents.runner import AgentRunner
    from planora.cli.callbacks import CLICallback, EventsOutputCallback
    from planora.core.events import PhaseStatus
    from planora.core.workspace import WorkspaceManager
    from planora.workflow.plan import PlanWorkflow

    registry = AgentRegistry.from_settings(settings)

    # 8. Fail fast: validate planner binary before creating workspace
    missing = registry.validate([resolved_planner])
    if missing:
        console.print(
            f"[red]Planner '{resolved_planner}' not available (binary not on PATH)[/red]"
        )
        raise typer.Exit(1)

    # 9. Choose UI callback based on output format
    ui: CLICallback | EventsOutputCallback
    ui = EventsOutputCallback() if output_format == "events" else CLICallback()

    # 10. Build and run the workflow
    workspace = WorkspaceManager(root)
    workflow = PlanWorkflow(
        workspace=workspace,
        registry=registry,
        runner=AgentRunner(),
        ui=ui,
        planner=resolved_planner,
        auditors=auditor_list,
        audit_rounds=effective_rounds,
        max_concurrency=resolved_concurrency,
        dry_run=dry_run,
        plan_template_path=settings.prompts.plan,
        audit_template_path=settings.prompts.audit,
        refine_template_path=settings.prompts.refine,
        prompt_base_dir=settings.effective_prompt_base_dir,
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


@plan_app.command("resume")
def plan_resume(
    project_root: Annotated[
        Path | None, typer.Option(help="Project root")
    ] = None,
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
        console.print(
            "[red]No workspace to resume from (.plan-workspace/ not found)[/red]"
        )
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

    # If final plan exists, auditing is already done — skip audit+refine
    if has_final:
        effective_auditors = []
        effective_rounds = 0

    configure_prompt_templates(
        plan=settings.prompts.plan,
        audit=settings.prompts.audit,
        refine=settings.prompts.refine,
        base_dir=settings.effective_prompt_base_dir,
    )

    # 8. Choose UI callback
    ui: CLICallback | EventsOutputCallback
    ui = EventsOutputCallback() if output_format == "events" else CLICallback()

    # 9. Run workflow — preflight will wipe and recreate workspace,
    #    which is acceptable: resume value is reading the saved task and
    #    re-running from the detected phase.
    workspace = WorkspaceManager(root)
    workflow = PlanWorkflow(
        workspace=workspace,
        registry=AgentRegistry.from_settings(settings),
        runner=AgentRunner(),
        ui=ui,
        planner=settings.effective_planner,
        auditors=effective_auditors,
        audit_rounds=effective_rounds,
        dry_run=False,
        plan_template_path=settings.prompts.plan,
        audit_template_path=settings.prompts.audit,
        refine_template_path=settings.prompts.refine,
        prompt_base_dir=settings.effective_prompt_base_dir,
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
