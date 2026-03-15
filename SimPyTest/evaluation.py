from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from SimPyTest.engine import SimPySimulationEngine

EVALUATION_PROTOCOL_VERSION = "v1-2026-03-12"


@dataclass(frozen=True)
class SimulationSummary:
    non_idle_time: float
    total_drive_time: float
    total_operating_cost: float
    total_fuel_cost: float
    total_spent: float
    total_resource_hours: float
    total_dispatch_delay: float
    dispatch_delay_events: int
    avg_dispatch_delay: float
    max_dispatch_delay: float
    disasters_created: int
    disasters_resolved: int
    resolution_rate: float
    avg_response_time: float
    avg_resolution_time: float
    total_weighted_closure_hours: float


class KPIBundle(TypedDict):
    resolution_rate: float
    avg_response_time_min: float
    avg_resolution_time_min: float
    total_spent: float
    total_weighted_closure_hours: float


def build_simulation_summary(engine: SimPySimulationEngine) -> SimulationSummary:
    total_operating_cost = 0.0
    total_fuel_cost = 0.0
    total_hours = 0.0

    for node in engine.resource_nodes:
        for inventory in node.inventory.values():
            for resource in inventory.items:
                total_operating_cost += resource.accumulated_cost
                total_hours += resource.total_hours_operated
                total_fuel_cost += resource.fuel_consumption_rate * resource.total_hours_operated * 4.50

    metrics_summary = engine.metrics.get_summary()
    total_spent = total_operating_cost + total_fuel_cost
    return SimulationSummary(
        non_idle_time=engine.non_idle_time,
        total_drive_time=engine.total_drive_time,
        total_operating_cost=total_operating_cost,
        total_fuel_cost=total_fuel_cost,
        total_spent=total_spent,
        total_resource_hours=total_hours,
        total_dispatch_delay=engine.total_dispatch_delay,
        dispatch_delay_events=engine.dispatch_delay_events,
        avg_dispatch_delay=(engine.total_dispatch_delay / engine.dispatch_delay_events if engine.dispatch_delay_events > 0 else 0.0),
        max_dispatch_delay=engine.max_dispatch_delay,
        disasters_created=int(metrics_summary.get("total_disasters_created", 0)),
        disasters_resolved=int(metrics_summary.get("total_disasters_resolved", 0)),
        resolution_rate=float(metrics_summary.get("resolution_rate", 0.0)),
        avg_response_time=float(metrics_summary.get("avg_response_time", 0.0)),
        avg_resolution_time=float(metrics_summary.get("avg_resolution_time", 0.0)),
        total_weighted_closure_hours=float(metrics_summary.get("total_weighted_closure_hours", 0.0)),
    )


def compute_kpi_bundle(summary: SimulationSummary) -> KPIBundle:
    return {
        "resolution_rate": float(summary.resolution_rate),
        "avg_response_time_min": float(summary.avg_response_time),
        "avg_resolution_time_min": float(summary.avg_resolution_time),
        "total_spent": float(summary.total_spent),
        "total_weighted_closure_hours": float(summary.total_weighted_closure_hours),
    }


def compute_objective_score(summary: SimulationSummary) -> float:
    """
    Compute the canonical objective score for benchmark comparison.

    Higher is better.

    The score is the negative of two directly interpretable costs:
    population impact cost plus non-idle simulation time in hours.
    """
    population_cost = float(summary.total_weighted_closure_hours)
    simulation_time_cost = float(summary.non_idle_time / 60.0)
    monetary_cost = float(summary.total_spent)
    return -(population_cost + simulation_time_cost + monetary_cost)
