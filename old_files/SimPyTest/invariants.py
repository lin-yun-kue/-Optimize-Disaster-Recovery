from __future__ import annotations

from typing import Any


def _is_negative(value: Any) -> bool:
    """Return True when a numeric value is negative."""
    return isinstance(value, (int, float)) and value < 0


def validate_engine_invariants(engine: Any, run_succeeded: bool) -> list[str]:
    """
    Validate core simulation invariants for one engine run.

    Returns:
        List of human-readable invariant violations. Empty means pass.
    """
    violations: list[str] = []
    summary = engine.get_summary()

    created = int(summary.get("disasters_created", 0))
    resolved = int(summary.get("disasters_resolved", 0))

    if created < 0 or resolved < 0:
        violations.append(f"Negative disaster counts: created={created}, resolved={resolved}")
    if resolved > created:
        violations.append(f"Resolved exceeds created: resolved={resolved}, created={created}")
    if run_succeeded and created != resolved:
        violations.append(f"Successful run ended with unresolved disasters: created={created}, resolved={resolved}")

    non_negative_fields = [
        "non_idle_time",
        "total_operating_cost",
        "total_fuel_cost",
        "total_spent",
        "total_resource_hours",
        "total_dispatch_delay",
        "dispatch_delay_events",
        "avg_dispatch_delay",
        "max_dispatch_delay",
        "avg_response_time",
        "avg_resolution_time",
    ]
    for field in non_negative_fields:
        value = summary.get(field, 0.0)
        if _is_negative(value):
            violations.append(f"Negative summary value: {field}={value}")

    dispatch_events = int(summary.get("dispatch_delay_events", 0))
    avg_dispatch_delay = float(summary.get("avg_dispatch_delay", 0.0))
    max_dispatch_delay = float(summary.get("max_dispatch_delay", 0.0))
    if dispatch_events == 0 and (avg_dispatch_delay != 0.0 or max_dispatch_delay != 0.0):
        violations.append(
            "Dispatch delay stats inconsistent when no delay events occurred: "
            f"avg={avg_dispatch_delay}, max={max_dispatch_delay}"
        )
    if dispatch_events > 0 and avg_dispatch_delay > max_dispatch_delay:
        violations.append(
            f"Dispatch delay stats inconsistent: avg_dispatch_delay={avg_dispatch_delay} > max_dispatch_delay={max_dispatch_delay}"
        )

    metrics = engine.metrics
    resolved_metrics_count = sum(1 for m in metrics.disaster_metrics.values() if m.get("end_time") is not None)
    if resolved_metrics_count != metrics.total_disasters_resolved:
        violations.append(
            "Resolved disaster aggregate mismatch (possible duplicate resolve accounting): "
            f"aggregate={metrics.total_disasters_resolved}, unique_resolved={resolved_metrics_count}"
        )

    for disaster_id, metric in metrics.disaster_metrics.items():
        start_time = float(metric.get("start_time", 0.0))
        end_time = metric.get("end_time")
        response_time = float(metric.get("response_time", 0.0))
        resolution_time = float(metric.get("resolution_time", 0.0))

        if response_time < 0:
            violations.append(f"Negative response time for disaster {disaster_id}: {response_time}")
        if resolution_time < 0:
            violations.append(f"Negative resolution time for disaster {disaster_id}: {resolution_time}")

        if end_time is not None:
            end_time_float = float(end_time)
            if end_time_float < start_time:
                violations.append(
                    f"Disaster {disaster_id} has end time before start time: start={start_time}, end={end_time_float}"
                )

    return violations
