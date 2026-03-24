from __future__ import annotations

import builtins
import sys
import types
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

import pytest

from planora.core.events import ToolExecution
from planora.observability.telemetry import PlanoraTelemetry

if TYPE_CHECKING:
    from collections.abc import Generator


class _FakeSpan:
    def __init__(self, name: str, attributes: dict[str, object] | None = None) -> None:
        self.name = name
        self.attributes = dict(attributes or {})

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value


class _BrokenSpan(_FakeSpan):
    def set_attribute(self, key: str, value: object) -> None:
        del key, value
        raise RuntimeError("cannot set attributes")


class _FakeTraceModule(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("opentelemetry.trace")
        self.current_span: _FakeSpan | None = None
        self.provider = None
        self.tracer = _FakeTracer(self)

    def set_tracer_provider(self, provider) -> None:
        self.provider = provider

    def get_tracer(self, name: str, version: str | None = None):
        self.tracer.name = name
        self.tracer.version = version or ""
        return self.tracer

    def get_current_span(self):
        return self.current_span


class _FakeTracer:
    def __init__(self, trace_module: _FakeTraceModule) -> None:
        self._trace_module = trace_module
        self.current_spans: list[_FakeSpan] = []
        self.child_spans: list[_FakeSpan] = []
        self.name = ""
        self.version = ""

    def start_as_current_span(self, name: str, attributes: dict[str, object] | None = None):
        span = _FakeSpan(name, attributes)
        self.current_spans.append(span)
        trace_module = self._trace_module

        class _ContextManager:
            def __enter__(self_nonlocal):
                self_nonlocal._previous = trace_module.current_span
                trace_module.current_span = span
                return span

            def __exit__(self_nonlocal, exc_type, exc, tb) -> bool:
                del exc_type, exc, tb
                trace_module.current_span = self_nonlocal._previous
                return False

        return _ContextManager()

    def start_span(self, name: str, attributes: dict[str, object] | None = None):
        span = _FakeSpan(name, attributes)
        self.child_spans.append(span)
        return span


class _FakeResource:
    @classmethod
    def create(cls, attributes: dict[str, object]) -> dict[str, object]:
        return attributes


class _FakeTracerProvider:
    def __init__(self, resource) -> None:
        self.resource = resource
        self.processors = []

    def add_span_processor(self, processor) -> None:
        self.processors.append(processor)


class _FakeBatchSpanProcessor:
    def __init__(self, exporter) -> None:
        self.exporter = exporter


class _FakeOTLPSpanExporter:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint


def _install_fake_opentelemetry(
    monkeypatch: pytest.MonkeyPatch,
    *,
    with_exporter: bool,
) -> _FakeTraceModule:
    trace_module = _FakeTraceModule()

    opentelemetry_pkg = types.ModuleType("opentelemetry")
    opentelemetry_pkg.trace = trace_module
    sdk_pkg = types.ModuleType("opentelemetry.sdk")
    resources_mod = types.ModuleType("opentelemetry.sdk.resources")
    resources_mod.Resource = _FakeResource
    trace_sdk_mod = types.ModuleType("opentelemetry.sdk.trace")
    trace_sdk_mod.TracerProvider = _FakeTracerProvider
    export_mod = types.ModuleType("opentelemetry.sdk.trace.export")
    export_mod.BatchSpanProcessor = _FakeBatchSpanProcessor

    modules = {
        "opentelemetry": opentelemetry_pkg,
        "opentelemetry.trace": trace_module,
        "opentelemetry.sdk": sdk_pkg,
        "opentelemetry.sdk.resources": resources_mod,
        "opentelemetry.sdk.trace": trace_sdk_mod,
        "opentelemetry.sdk.trace.export": export_mod,
        "opentelemetry.exporter": types.ModuleType("opentelemetry.exporter"),
        "opentelemetry.exporter.otlp": types.ModuleType("opentelemetry.exporter.otlp"),
        "opentelemetry.exporter.otlp.proto": types.ModuleType("opentelemetry.exporter.otlp.proto"),
        "opentelemetry.exporter.otlp.proto.grpc": types.ModuleType(
            "opentelemetry.exporter.otlp.proto.grpc"
        ),
        "opentelemetry.exporter.otlp.proto.http": types.ModuleType(
            "opentelemetry.exporter.otlp.proto.http"
        ),
    }
    package_names = {
        "opentelemetry",
        "opentelemetry.sdk",
        "opentelemetry.exporter",
        "opentelemetry.exporter.otlp",
        "opentelemetry.exporter.otlp.proto",
        "opentelemetry.exporter.otlp.proto.grpc",
        "opentelemetry.exporter.otlp.proto.http",
    }
    for name in package_names:
        modules[name].__path__ = []  # type: ignore[attr-defined]
    if with_exporter:
        grpc_mod = types.ModuleType("opentelemetry.exporter.otlp.proto.grpc.trace_exporter")
        grpc_mod.OTLPSpanExporter = _FakeOTLPSpanExporter
        modules["opentelemetry.exporter.otlp.proto.grpc.trace_exporter"] = grpc_mod

    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    return trace_module


@pytest.fixture
def fake_otel_with_exporter(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[_FakeTraceModule, None, None]:
    """Install fake OpenTelemetry modules including the OTLP exporter, then clean up."""
    trace_module = _install_fake_opentelemetry(monkeypatch, with_exporter=True)
    yield trace_module


@pytest.fixture
def fake_otel_without_exporter(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[_FakeTraceModule, None, None]:
    """Install fake OpenTelemetry modules without the OTLP exporter, then clean up."""
    trace_module = _install_fake_opentelemetry(monkeypatch, with_exporter=False)
    yield trace_module


def test_telemetry_disabled_is_a_no_op(settings) -> None:
    telemetry = PlanoraTelemetry(settings.model_copy(update={"telemetry_enabled": False}))
    tool = ToolExecution(
        tool_id="tool-1",
        name="Read",
        friendly_name="Read file",
        started_at=datetime.now(),
    )

    with telemetry.phase_span("plan") as span:
        assert span is None

    assert telemetry.tool_span("claude", tool) is None
    telemetry.record_cost("claude", Decimal("1.23"))


def test_telemetry_enabled_without_packages_disables_gracefully(monkeypatch, settings) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("opentelemetry"):
            raise ImportError("missing otel dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    telemetry = PlanoraTelemetry(settings.model_copy(update={"telemetry_enabled": True}))

    assert telemetry._enabled is False


def test_telemetry_builds_spans_when_fake_otel_is_available(
    fake_otel_with_exporter: _FakeTraceModule, settings
) -> None:
    trace_module = fake_otel_with_exporter
    telemetry = PlanoraTelemetry(
        settings.model_copy(
            update={
                "telemetry_enabled": True,
                "telemetry_service_name": "planora-tests",
                "telemetry_otlp_endpoint": "http://collector:4317",
            }
        )
    )
    tool = ToolExecution(
        tool_id="tool-9",
        name="Read",
        friendly_name="Read file",
        detail="README.md",
        started_at=datetime.now(),
    )

    with telemetry.phase_span("plan", agent="claude") as span:
        telemetry.record_cost("claude", Decimal("2.50"))

    child_span = telemetry.tool_span("claude", tool)

    assert telemetry._enabled is True
    assert trace_module.provider.resource == {"service.name": "planora-tests"}
    assert len(trace_module.provider.processors) == 1
    assert span.attributes == {
        "planora.phase": "plan",
        "planora.agent": "claude",
        "planora.cost.claude": 2.5,
        "planora.cost.total": 2.5,
    }
    assert child_span.attributes == {
        "planora.agent": "claude",
        "planora.tool.name": "Read",
        "planora.tool.detail": "README.md",
        "planora.tool.id": "tool-9",
    }


def test_telemetry_warns_when_exporter_module_is_missing(
    fake_otel_without_exporter: _FakeTraceModule, caplog, settings
) -> None:
    with caplog.at_level("WARNING"):
        telemetry = PlanoraTelemetry(
            settings.model_copy(
                update={
                    "telemetry_enabled": True,
                    "telemetry_otlp_endpoint": "http://collector:4317",
                }
            )
        )

    assert telemetry._enabled is True
    assert "Telemetry will run without exporting spans" in caplog.text


def test_record_cost_swallows_span_errors(
    fake_otel_with_exporter: _FakeTraceModule, caplog, settings
) -> None:
    trace_module = fake_otel_with_exporter
    telemetry = PlanoraTelemetry(settings.model_copy(update={"telemetry_enabled": True}))
    trace_module.current_span = _BrokenSpan("broken")

    with caplog.at_level("DEBUG"):
        telemetry.record_cost("claude", Decimal("1.00"))

    assert "Failed to record telemetry cost for agent claude" in caplog.text
