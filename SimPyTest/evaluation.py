from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict, TYPE_CHECKING

if TYPE_CHECKING:
    from SimPyTest.engine import SimPySimulationEngine


@dataclass(frozen=True)
class SimulationSummary:
    terminal_outcome: str | None
    time_with_disasters: float
    total_drive_time: float
    total_operating_cost: float
    total_fuel_cost: float
    total_spent: float
    total_resource_hours: float
    disasters_created: int
    disasters_resolved: int
    resolution_rate: float
    avg_response_time: float
    avg_resolution_time: float
    total_weighted_closure_hours: float


class ResearchMetricBundle(TypedDict):
    resolution_rate: float
    time_with_disasters: float
    avg_response_time_min: float
    avg_resolution_time_min: float
    total_spent: float
    total_weighted_closure_hours: float
    total_drive_time: float
    total_resource_hours: float


class PrimaryMetricBundle(TypedDict):
    time_with_disasters: float
    avg_response_time_min: float
    total_spent: float


KPIBundle = ResearchMetricBundle


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
        terminal_outcome=engine.last_terminal_outcome,
        time_with_disasters=engine.time_with_disasters,
        total_drive_time=engine.total_drive_time,
        total_operating_cost=total_operating_cost,
        total_fuel_cost=total_fuel_cost,
        total_spent=total_spent,
        total_resource_hours=total_hours,
        disasters_created=int(metrics_summary.get("total_disasters_created", 0)),
        disasters_resolved=int(metrics_summary.get("total_disasters_resolved", 0)),
        resolution_rate=float(metrics_summary.get("resolution_rate", 0.0)),
        avg_response_time=float(metrics_summary.get("avg_response_time", 0.0)),
        avg_resolution_time=float(metrics_summary.get("avg_resolution_time", 0.0)),
        total_weighted_closure_hours=float(metrics_summary.get("total_weighted_closure_hours", 0.0)),
    )


def compute_research_metric_bundle(summary: SimulationSummary) -> ResearchMetricBundle:
    return {
        "resolution_rate": float(summary.resolution_rate),
        "time_with_disasters": float(summary.time_with_disasters),
        "avg_response_time_min": float(summary.avg_response_time),
        "avg_resolution_time_min": float(summary.avg_resolution_time),
        "total_spent": float(summary.total_spent),
        "total_weighted_closure_hours": float(summary.total_weighted_closure_hours),
        "total_drive_time": float(summary.total_drive_time),
        "total_resource_hours": float(summary.total_resource_hours),
    }


def compute_primary_metric_bundle(summary: SimulationSummary) -> PrimaryMetricBundle:
    return {
        "time_with_disasters": float(summary.time_with_disasters),
        "avg_response_time_min": float(summary.avg_response_time),
        "total_spent": float(summary.total_spent),
    }


def compute_kpi_bundle(summary: SimulationSummary) -> KPIBundle:
    return compute_research_metric_bundle(summary)


def compute_training_objective_score(summary: SimulationSummary) -> float:
    """
    Compute the training-only scalar objective used for PPO/checkpoint selection.

    Higher is better.

    This is intentionally not the research comparison rule. It is a compact
    optimization target that:
    - heavily penalizes unresolved disasters
    - prefers less time with active disasters
    - prefers lower closure impact
    - lightly prefers lower spending
    """
    terminal_failure_penalty = 0.0 if summary.terminal_outcome == "SUCCESS" else 1_000_000.0
    unresolved_disasters = max(0, int(summary.disasters_created - summary.disasters_resolved))
    unresolved_penalty = float(unresolved_disasters) * 100_000.0
    closure_cost = float(summary.total_weighted_closure_hours)
    time_cost_hours = float(summary.time_with_disasters / 60.0)
    spending_cost_thousands = float(summary.total_spent / 1000.0)
    return -(terminal_failure_penalty + unresolved_penalty + closure_cost + time_cost_hours + spending_cost_thousands)


def compute_objective_score(summary: SimulationSummary) -> float:
    return compute_training_objective_score(summary)
