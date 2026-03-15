from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

from SimPyTest.engine import SimPySimulationEngine
from SimPyTest.evaluation import build_simulation_summary
from SimPyTest.simulation import Resource, ResourceType


class DecisionLogRecord(TypedDict):
    sim_time: float
    resource_id: int
    resource_type: str
    resource_location: list[float]
    global_state_summary: dict[str, float | int]
    visible_candidate_ids: list[int]
    valid_actions: list[int]
    selected_action: int
    selected_action_kind: str
    selected_disaster_id: int | None
    invalid_action: bool


class EpisodeLogRecord(TypedDict):
    seed: int
    scenario_name: str
    controller_name: str
    terminal_label: str
    success: bool
    total_sim_time: float
    non_idle_time: float
    total_spent: float
    population_affected: int
    weighted_closure_metric: float
    avg_response_time: float
    avg_resolution_time: float
    decision_count: int
    invalid_action_count: int


@dataclass
class DispatchEpisodeLogger:
    controller_name: str
    scenario_name: str
    seed: int
    decisions: list[DecisionLogRecord] = field(default_factory=list)
    episode: EpisodeLogRecord | None = None

    def log_decision(self, record: DecisionLogRecord) -> None:
        self.decisions.append(record)

    def finalize(self, record: EpisodeLogRecord) -> None:
        self.episode = record

    def to_payload(self) -> dict[str, Any]:
        return {
            "controller_name": self.controller_name,
            "scenario_name": self.scenario_name,
            "seed": self.seed,
            "episode": self.episode,
            "decisions": self.decisions,
        }

    def write_artifacts(self, output_dir: str | Path) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        summary_path = path / "episode_summary.json"
        decisions_path = path / "decision_log.jsonl"

        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_payload(), handle, indent=2, sort_keys=True)

        with decisions_path.open("w", encoding="utf-8") as handle:
            for record in self.decisions:
                handle.write(json.dumps(record, sort_keys=True))
                handle.write("\n")


def build_global_state_summary(engine: SimPySimulationEngine | None) -> dict[str, float | int]:
    if engine is None:
        return {
            "sim_time": 0.0,
            "active_disasters": 0,
            "idle_trucks": 0,
            "idle_excavators": 0,
            "idle_snowplows": 0,
            "idle_assessment_vehicles": 0,
        }

    idle_resources = engine.idle_resources
    return {
        "sim_time": float(engine.env.now),
        "active_disasters": len(engine.disaster_store.items),
        "idle_trucks": len(idle_resources.inventory[ResourceType.TRUCK].items),
        "idle_excavators": len(idle_resources.inventory[ResourceType.EXCAVATOR].items),
        "idle_snowplows": len(idle_resources.inventory[ResourceType.SNOWPLOW].items),
        "idle_assessment_vehicles": len(idle_resources.inventory[ResourceType.ASSESSMENT_VEHICLE].items),
    }


def build_decision_log_record(
    *,
    engine: SimPySimulationEngine | None,
    current_resource: Resource,
    current_candidate_ids: list[int],
    valid_actions: list[int],
    action: int,
    action_kind: str,
    selected_disaster_id: int | None,
    invalid_action: bool,
) -> DecisionLogRecord:
    return {
        "sim_time": float(engine.env.now if engine is not None else 0.0),
        "resource_id": current_resource.id,
        "resource_type": current_resource.resource_type.name,
        "resource_location": [float(current_resource.location[0]), float(current_resource.location[1])],
        "global_state_summary": build_global_state_summary(engine),
        "visible_candidate_ids": current_candidate_ids,
        "valid_actions": valid_actions,
        "selected_action": int(action),
        "selected_action_kind": action_kind,
        "selected_disaster_id": selected_disaster_id,
        "invalid_action": invalid_action,
    }


def finalize_episode_log(
    *,
    logger: DispatchEpisodeLogger | None,
    engine: SimPySimulationEngine | None,
    episode_seed: int,
    scenario_name: str,
    controller_name: str,
    invalid_action_count: int,
) -> None:
    if logger is None or engine is None:
        return

    metrics = engine.metrics.get_per_disaster_stats()
    population_affected = sum(int(record["population_affected"] or 0) for record in metrics)
    summary = build_simulation_summary(engine)
    terminal_outcome = engine.last_terminal_outcome or SimPySimulationEngine.TERMINAL_FAIL_INVALID_STATE
    logger.finalize(
        {
            "seed": episode_seed,
            "scenario_name": scenario_name,
            "controller_name": controller_name,
            "terminal_label": terminal_outcome,
            "success": terminal_outcome == SimPySimulationEngine.TERMINAL_SUCCESS,
            "total_sim_time": float(engine.env.now),
            "non_idle_time": float(summary.non_idle_time),
            "total_spent": float(summary.total_spent),
            "population_affected": population_affected,
            "weighted_closure_metric": float(summary.total_weighted_closure_hours),
            "avg_response_time": float(summary.avg_response_time),
            "avg_resolution_time": float(summary.avg_resolution_time),
            "decision_count": len(logger.decisions),
            "invalid_action_count": invalid_action_count,
        }
    )
