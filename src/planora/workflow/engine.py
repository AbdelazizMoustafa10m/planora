from __future__ import annotations

import asyncio
import signal
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path  # noqa: TC003 — used at runtime in path operations

from planora.agents.registry import AgentConfig, AgentMode
from planora.agents.runner import (
    AgentRunner,  # noqa: TC001 — stored as instance attribute, called at runtime
)
from planora.core.events import AgentResult, PhaseResult, PhaseStatus, UICallback


class PhaseRunner:
    """
    Generic phase runner. PlanWorkflow delegates execution to this class.

    Owns: sequential single-agent execution, parallel multi-agent execution
    with Semaphore-based concurrency control, and SIGINT/SIGTERM signal handling.
    """

    def __init__(
        self,
        runner: AgentRunner,
        ui: UICallback,
        max_concurrency: int = 3,
        snapshot_interval: float | None = None,
    ) -> None:
        self._runner = runner
        self._ui = ui
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._active_processes: list[asyncio.subprocess.Process] = []
        self._shutting_down = False
        self._snapshot_interval = snapshot_interval

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def _install_signal_handlers(self) -> None:
        """Install SIGINT/SIGTERM handlers on the running event loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._initiate_shutdown)

    def _initiate_shutdown(self) -> None:
        """
        Graceful shutdown on first signal; force-kill on second.

        First signal:
        1. Sets _shutting_down to prevent new phases from starting.
        2. Sends SIGTERM to all active subprocesses.
        3. Schedules _force_kill_all after a 5-second grace period.

        Second signal: immediately kills all active subprocesses.
        """
        if self._shutting_down:
            for proc in self._active_processes:
                proc.kill()
            return
        self._shutting_down = True
        for proc in self._active_processes:
            proc.terminate()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.call_later(5.0, self._force_kill_all)

    def _force_kill_all(self) -> None:
        """Kill any subprocess still running after the grace period."""
        for proc in self._active_processes:
            if proc.returncode is None:
                proc.kill()

    # ------------------------------------------------------------------
    # Public execution methods
    # ------------------------------------------------------------------

    async def run_phase(
        self,
        name: str,
        agent: AgentConfig,
        prompt: str,
        output_path: Path,
        dry_run: bool = False,
    ) -> PhaseResult:
        """
        Execute a single-agent phase (e.g. plan, refine).

        Returns FAILED immediately if a shutdown is in progress.
        """
        self._install_signal_handlers()

        if self._shutting_down:
            return _failed_phase(name, "Shutdown in progress")

        self._ui.on_phase_start(name, name)
        self._ui.on_agent_start(agent.name, name)

        started_at = datetime.now()
        result = await self._runner.run(
            agent=agent,
            prompt=prompt,
            output_path=output_path,
            mode=AgentMode.PLAN,
            dry_run=dry_run,
            on_event=lambda ev: self._ui.dispatch_agent_event(agent.name, ev),
            on_snapshot=self._ui.on_snapshot,
            snapshot_interval=self._snapshot_interval,
            on_process_start=self._register_process,
            on_process_end=self._unregister_process,
        )

        self._ui.on_agent_end(agent.name, result)

        duration = datetime.now() - started_at
        is_success = result.exit_code == 0 and (dry_run or not result.output_empty)
        status = PhaseStatus.DONE if is_success else PhaseStatus.FAILED
        output_files = [result.output_path] if not result.output_empty else []
        cost_usd = result.cost_usd

        phase_result = PhaseResult(
            name=name,
            status=status,
            duration=duration,
            output_files=output_files,
            agent_results=[result],
            error=result.error if status == PhaseStatus.FAILED else None,
            cost_usd=cost_usd,
        )
        self._ui.on_phase_end(name, phase_result)
        return phase_result

    async def run_parallel(
        self,
        name: str,
        agents: list[tuple[AgentConfig, str, Path]],
        dry_run: bool = False,
    ) -> PhaseResult:
        """
        Execute multiple agents concurrently with Semaphore-based concurrency control.

        All agents run regardless of individual failures. Exceptions from
        asyncio.gather are normalised into synthetic AgentResult values with
        exit_code=1 so callers always receive a homogeneous list of AgentResult.

        Returns FAILED immediately if a shutdown is in progress.
        """
        self._install_signal_handlers()

        if self._shutting_down:
            return _failed_phase(name, "Shutdown in progress")

        self._ui.on_phase_start(name, name)

        started_at = datetime.now()

        coroutines = [
            self._run_with_semaphore(cfg, prompt, output_path, dry_run)
            for cfg, prompt, output_path in agents
        ]
        raw_results: list[AgentResult | BaseException] = await asyncio.gather(
            *coroutines,
            return_exceptions=True,
        )

        agent_results: list[AgentResult] = []
        for (cfg, _prompt, output_path), outcome in zip(agents, raw_results, strict=True):
            if isinstance(outcome, BaseException):
                synthetic = AgentResult(
                    agent_name=cfg.name,
                    output_path=output_path,
                    stream_path=output_path.with_suffix(".stream"),
                    log_path=output_path.with_suffix(".log"),
                    exit_code=1,
                    duration=timedelta(0),
                    output_empty=True,
                    error=str(outcome),
                )
                agent_results.append(synthetic)
                self._ui.on_agent_end(cfg.name, synthetic)
            else:
                agent_results.append(outcome)
                self._ui.on_agent_end(cfg.name, outcome)

        duration = datetime.now() - started_at

        any_failed = any(
            r.exit_code != 0 or (not dry_run and r.output_empty) for r in agent_results
        )
        status = PhaseStatus.FAILED if any_failed else PhaseStatus.DONE

        output_files = [r.output_path for r in agent_results if not r.output_empty]

        costs = [r.cost_usd for r in agent_results if r.cost_usd is not None]
        cost_usd: Decimal | None = sum(costs, Decimal(0)) if costs else None

        errors = [r.error for r in agent_results if r.error]
        error_msg = "; ".join(errors) if errors else None

        phase_result = PhaseResult(
            name=name,
            status=status,
            duration=duration,
            output_files=output_files,
            agent_results=agent_results,
            error=error_msg,
            cost_usd=cost_usd,
        )
        self._ui.on_phase_end(name, phase_result)
        return phase_result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _run_with_semaphore(
        self,
        agent: AgentConfig,
        prompt: str,
        output_path: Path,
        dry_run: bool,
    ) -> AgentResult:
        """Acquire the semaphore then run the agent, notifying the UI at start."""
        async with self._semaphore:
            self._ui.on_agent_start(agent.name, agent.name)
            return await self._runner.run(
                agent=agent,
                prompt=prompt,
                output_path=output_path,
                mode=AgentMode.PLAN,
                dry_run=dry_run,
                on_event=lambda ev: self._ui.dispatch_agent_event(agent.name, ev),
                on_snapshot=self._ui.on_snapshot,
                snapshot_interval=self._snapshot_interval,
                on_process_start=self._register_process,
                on_process_end=self._unregister_process,
            )

    def _register_process(self, proc: asyncio.subprocess.Process) -> None:
        """Track a live subprocess so shutdown signals can reach it."""
        self._active_processes.append(proc)

    def _unregister_process(self, proc: asyncio.subprocess.Process) -> None:
        """Stop tracking a subprocess once it has exited or been cancelled."""
        if proc in self._active_processes:
            self._active_processes.remove(proc)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _failed_phase(name: str, reason: str) -> PhaseResult:
    """Return a PhaseResult with FAILED status and no timing."""
    return PhaseResult(
        name=name,
        status=PhaseStatus.FAILED,
        error=reason,
    )
