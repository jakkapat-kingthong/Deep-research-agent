from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

_initialized = False


def setup_tracing() -> None:
    global _initialized
    if _initialized:
        return

    # ตั้งชื่อ Service เพื่อให้รู้ว่า Log มาจากแอปไหน
    resource = Resource.create({SERVICE_NAME: "deep-research-agent"})
    provider = TracerProvider(resource=resource)

    # สั่งให้พ่น Log การทำงานออกมาที่ Console (Terminal)
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    _initialized = True


tracer = trace.get_tracer("deep_research")
