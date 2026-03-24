from __future__ import annotations

import importlib
import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator
    from decimal import Decimal

    from planora.core.config import PlanораSettings
    from planora.core.events import ToolExecution

logger = logging.getLogger(__name__)


class PlanoraTelemetry:
    """Optional OpenTelemetry integration with graceful degradation."""

    def __init__(self, settings: PlanораSettings) -> None:
        self._enabled = settings.effective_telemetry_enabled
        self._trace_api: Any | None = None
        self._tracer: Any | None = None

        if not self._enabled:
            return

        try:
            self._trace_api = importlib.import_module("opentelemetry.trace")
            resource_module = importlib.import_module("opentelemetry.sdk.resources")
            sdk_trace_module = importlib.import_module("opentelemetry.sdk.trace")

            resource = resource_module.Resource.create(
                {"service.name": settings.effective_telemetry_service_name}
            )
            provider = sdk_trace_module.TracerProvider(resource=resource)

            endpoint = settings.effective_telemetry_otlp_endpoint
            if endpoint:
                try:
                    exporter_path = (
                        "opentelemetry.exporter.otlp.proto.grpc.trace_exporter"
                        if settings.effective_telemetry_otlp_protocol == "grpc"
                        else "opentelemetry.exporter.otlp.proto.http.trace_exporter"
                    )
                    exporter_module = importlib.import_module(exporter_path)
                    export_module = importlib.import_module("opentelemetry.sdk.trace.export")
                    exporter = exporter_module.OTLPSpanExporter(endpoint=endpoint)
                    processor = export_module.BatchSpanProcessor(exporter)
                    provider.add_span_processor(processor)
                except ImportError:
                    logger.warning(
                        "OTLP exporter packages not installed. "
                        "Telemetry will run without exporting spans."
                    )

            self._trace_api.set_tracer_provider(provider)
            self._tracer = self._trace_api.get_tracer("planora")
        except ImportError:
            logger.debug("OpenTelemetry packages not installed. Telemetry disabled.")
            self._enabled = False
            self._trace_api = None
            self._tracer = None

    @contextmanager
    def pipeline_span(self, task_slug: str) -> Iterator[Any | None]:
        """Create a root span for the entire planning pipeline."""
        if not self._enabled or self._tracer is None:
            yield None
            return

        attributes = {
            "planora.pipeline": "plan",
            "planora.task_slug": task_slug,
        }
        with self._tracer.start_as_current_span(
            "planora.pipeline",
            attributes=attributes,
        ) as span:
            yield span

    @contextmanager
    def phase_span(self, phase: str, agent: str | None = None) -> Iterator[Any | None]:
        """Create a workflow phase span or yield `None` when telemetry is disabled."""
        if not self._enabled or self._tracer is None:
            yield None
            return

        attributes = {
            "planora.phase": phase,
            "planora.agent": agent or "",
        }
        with self._tracer.start_as_current_span(
            f"planora.phase.{phase}",
            attributes=attributes,
        ) as span:
            yield span

    def tool_span(self, agent: str, tool: ToolExecution) -> Any | None:
        """Create a child span for a tool invocation."""
        if not self._enabled or self._tracer is None:
            return None

        return self._tracer.start_span(
            f"planora.tool.{tool.name}",
            attributes={
                "planora.agent": agent,
                "planora.tool.name": tool.name,
                "planora.tool.detail": tool.detail or "",
                "planora.tool.id": tool.tool_id,
            },
        )

    def record_cost(self, agent: str, cost_usd: Decimal) -> None:
        """Record cost information on the current span if telemetry is active."""
        if not self._enabled or self._trace_api is None:
            return

        try:
            span = self._trace_api.get_current_span()
            if span is None or not getattr(span, "is_recording", lambda: True)():
                return
            span.set_attribute("planora.agent", agent)
            span.set_attribute(f"planora.cost.{agent}", float(cost_usd))
            span.set_attribute("planora.cost.total", float(cost_usd))
        except Exception:  # noqa: BLE001
            logger.debug("Failed to record telemetry cost for agent %s", agent)
