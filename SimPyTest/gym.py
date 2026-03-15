from __future__ import annotations

from collections.abc import Generator
from typing import Any, Literal, TypedDict, cast

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces
from simpy.core import EmptySchedule
from simpy.events import Event
from typing_extensions import override

from SimPyTest.calendar import Season
from SimPyTest.evaluation import EVALUATION_PROTOCOL_VERSION, SimulationSummary, build_simulation_summary, compute_objective_score
from SimPyTest.gym_logging import DispatchEpisodeLogger, build_decision_log_record, finalize_episode_log
from SimPyTest.real_world_params import travel_minutes_from_distance
from SimPyTest.scenario_types import ScenarioConfig
from .engine import SimPySimulationEngine
from .policies import Policy
from .simulation import Disaster, Resource, ResourceType

OBSERVATION_VERSION = "v3"
ACTION_VERSION = "v2"
REWARD_VERSION = "v1"

CURRENT_RESOURCE_FEATURES = 7
GLOBAL_STATE_FEATURES = 10
DISASTER_FEATURES = 14

ActType = int
SortOptions = Literal["nearest", "furthest", "random", "most_progress", "least_progress"]


class ObsType(TypedDict):
    current_resource: npt.NDArray[np.float32]
    global_state: npt.NDArray[np.float32]
    candidate_disasters: npt.NDArray[np.float32]
    valid_actions: npt.NDArray[np.int8]


class InfoType(TypedDict):
    sim_time: float
    active_disasters: int
    season: str
    weather_factor: float
    calendar_date: str | None
    total_spent: float
    objective_score: float
    reward_components: dict[str, float]
    summary: SimulationSummary
    terminal_outcome: str | None
    is_success: bool
    is_failure: bool
    is_truncated: bool
    invalid_action: bool
    selected_action: int | None
    selected_action_kind: str | None
    visible_candidate_ids: list[int]
    observation_version: str
    action_version: str
    reward_version: str
    evaluation_version: str
    invalid_action_count: int
    invalid_action_remaps: int
    valid_action_count: int
    requested_action: int | None
    executed_action: int | None


class DisasterResponseGym(gym.Env[ObsType, ActType]):
    # metadata = {"render_modes": []}

    def __init__(
        self,
        max_visible_disasters: int,
        sorting_strategy: SortOptions,
        scenario_config: ScenarioConfig,
        *,
        controller_name: str = "ppo",
        scenario_name: str = "custom",
    ):
        super().__init__()
        self.scenario_config: ScenarioConfig = scenario_config
        self.max_slots: int = max_visible_disasters
        self.sorting_strategy: SortOptions = sorting_strategy
        self.controller_name: str = controller_name
        self.scenario_name: str = scenario_name

        self.action_space: spaces.Space[ActType] = spaces.Discrete(self.max_slots)
        self.observation_space: spaces.Space[ObsType] = cast(
            spaces.Space[ObsType],
            spaces.Dict(
                {
                    "current_resource": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(CURRENT_RESOURCE_FEATURES,),
                        dtype=np.float32,
                    ),
                    "global_state": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(GLOBAL_STATE_FEATURES,),
                        dtype=np.float32,
                    ),
                    "candidate_disasters": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(self.max_slots, DISASTER_FEATURES),
                        dtype=np.float32,
                    ),
                    "valid_actions": spaces.Box(
                        low=0,
                        high=1,
                        shape=(self.max_slots,),
                        dtype=np.int8,
                    ),
                }
            ),
        )

        self.engine: SimPySimulationEngine | None = None
        self.current_resource: Resource | None = None
        self.current_candidates: list[Disaster] = []
        self.current_action: int | None = None
        self.decision_needed: bool = False

        self._episode_seed: int = 0
        self._invalid_action_count: int = 0
        self._invalid_action_remaps: int = 0
        self._last_invalid_action: bool = False
        self._last_selected_action: int | None = None
        self._last_requested_action: int | None = None
        self._last_executed_action: int | None = None
        self._last_action_kind: str | None = None
        self._prev_objective_score: float | None = None
        self._prev_active_disasters: int | None = None
        self._prev_total_percent_remaining: float | None = None
        self.logger: DispatchEpisodeLogger | None = None

    def _clamp_unit(self, value: float) -> float:
        return float(min(max(value, 0.0), 1.0))

    def _scenario_max_resource_count(self, value: int | tuple[int, int]) -> int:
        if isinstance(value, tuple):
            return int(value[1])
        return int(value)

    def _max_total_resources(self) -> int:
        counts = self.scenario_config.resource_counts
        return max(
            1,
            self._scenario_max_resource_count(counts.trucks)
            + self._scenario_max_resource_count(counts.excavators)
            + self._scenario_max_resource_count(counts.snowplows)
            + self._scenario_max_resource_count(counts.assessment_vehicles),
        )

    def _max_distance_miles(self) -> float:
        cfg = self.scenario_config.distance_model
        return max(1.0, float(cfg.spawn_distance_range[1]) * float(cfg.non_gis_distance_unit_miles) * 3.0)

    def _max_travel_minutes(self) -> float:
        slowest_speed = min(float(resource_type.specs["speed"]) for resource_type in ResourceType)
        return max(1.0, travel_minutes_from_distance(self._max_distance_miles(), slowest_speed))

    def _max_active_disasters(self) -> int:
        return max(1, int(self.scenario_config.seasonal_spawn.target_events_range[1]))

    def _max_closure_minutes(self) -> float:
        priors = self.scenario_config.operational_priors.closure_minutes_range_by_disaster
        return max(60.0, max(float(high) for _key, (_low, high) in priors.items()))

    def _current_summary(self) -> SimulationSummary:
        if self.engine is None:
            raise RuntimeError("Environment not initialized")
        return build_simulation_summary(self.engine)

    def _total_percent_remaining(self) -> float:
        if self.engine is None:
            return 0.0
        return float(sum(disaster.percent_remaining() for disaster in self.engine.disaster_store.items))

    def _valid_actions(self) -> npt.NDArray[np.int8]:
        valid_actions = np.zeros(self.max_slots, dtype=np.int8)
        valid_actions[: len(self.current_candidates)] = 1
        return valid_actions

    def action_masks(self) -> npt.NDArray[np.bool_]:
        return self._valid_actions().astype(np.bool_)

    def _zero_obs(self) -> ObsType:
        return {
            "current_resource": np.zeros(CURRENT_RESOURCE_FEATURES, dtype=np.float32),
            "global_state": np.zeros(GLOBAL_STATE_FEATURES, dtype=np.float32),
            "candidate_disasters": np.zeros((self.max_slots, DISASTER_FEATURES), dtype=np.float32),
            "valid_actions": np.zeros(self.max_slots, dtype=np.int8),
        }

    def _normalize_resource_count(self, count: int, resource_type: ResourceType) -> float:
        counts = self.scenario_config.resource_counts
        max_counts = {
            ResourceType.TRUCK: self._scenario_max_resource_count(counts.trucks),
            ResourceType.EXCAVATOR: self._scenario_max_resource_count(counts.excavators),
            ResourceType.SNOWPLOW: self._scenario_max_resource_count(counts.snowplows),
            ResourceType.ASSESSMENT_VEHICLE: self._scenario_max_resource_count(counts.assessment_vehicles),
        }
        return self._clamp_unit(count / max(1, max_counts[resource_type]))

    def _normalize_location(self, loc: tuple[float, float]) -> tuple[float, float]:
        max_coord = max(1.0, float(self.scenario_config.distance_model.spawn_distance_range[1]))
        return (
            self._clamp_unit(abs(loc[0]) / max_coord),
            self._clamp_unit(abs(loc[1]) / max_coord),
        )

    def _resource_sort_distance(self, resource: Resource, disaster: Disaster) -> float:
        if self.engine is None:
            return 0.0
        return float(self.engine.get_distance(resource, disaster))

    def _sort_candidates(self, resource: Resource, candidates: list[Disaster]) -> list[Disaster]:
        if self.engine is None:
            return []

        if self.sorting_strategy == "nearest":
            return sorted(candidates, key=lambda d: (self._resource_sort_distance(resource, d), d.created_time, d.id))
        if self.sorting_strategy == "furthest":
            return sorted(candidates, key=lambda d: (-self._resource_sort_distance(resource, d), d.created_time, d.id))
        if self.sorting_strategy == "random":
            ordered = list(candidates)
            self.engine.rng.shuffle(ordered)
            return sorted(ordered, key=lambda d: d.id)
        if self.sorting_strategy == "most_progress":
            return sorted(candidates, key=lambda d: (-(1.0 - d.percent_remaining()), d.created_time, d.id))
        return sorted(candidates, key=lambda d: (d.percent_remaining(), d.created_time, d.id))

    def _get_actionable_disasters(self, resource: Resource) -> list[Disaster]:
        if self.engine is None:
            return []
        actionable = [disaster for disaster in self.engine.disaster_store.items if resource.resource_type in disaster.needed_resources()]
        return self._sort_candidates(resource, actionable)[: self.max_slots]

    def _build_current_resource_features(self) -> npt.NDArray[np.float32]:
        features = np.zeros(CURRENT_RESOURCE_FEATURES, dtype=np.float32)
        if self.current_resource is None:
            return features

        resource_id_norm = self._clamp_unit(self.current_resource.id / self._max_total_resources())
        x_norm, y_norm = self._normalize_location(self.current_resource.location)
        features[0] = resource_id_norm
        features[1 + self.current_resource.resource_type.value] = 1.0
        features[-2] = x_norm
        features[-1] = y_norm
        return features

    def _build_global_state_features(self) -> npt.NDArray[np.float32]:
        features = np.zeros(GLOBAL_STATE_FEATURES, dtype=np.float32)
        if self.engine is None:
            return features

        season_map = {Season.WINTER: 0, Season.SPRING: 1, Season.SUMMER: 2, Season.FALL: 3}
        season_idx = season_map.get(self.engine.calendar.get_season(), 0)
        active_disasters = len(self.engine.disaster_store.items)
        idle_resources = self.engine.idle_resources

        cursor = 0
        features[cursor] = self._clamp_unit(self.engine.env.now / self.engine.MAX_SIM_TIME)
        cursor += 1
        features[cursor + season_idx] = 1.0
        cursor += 4
        features[cursor] = self._clamp_unit(active_disasters / self._max_active_disasters())
        cursor += 1
        for resource_type in ResourceType:
            count = len(idle_resources.inventory[resource_type].items)
            features[cursor] = self._normalize_resource_count(count, resource_type)
            cursor += 1
        return features

    def _build_candidate_feature_matrix(self) -> npt.NDArray[np.float32]:
        features = np.zeros((self.max_slots, DISASTER_FEATURES), dtype=np.float32)
        if self.engine is None or self.current_resource is None:
            return features

        max_closure_minutes = self._max_closure_minutes()
        max_travel_minutes = self._max_travel_minutes()

        for idx, disaster in enumerate(self.current_candidates):
            row = np.zeros(DISASTER_FEATURES, dtype=np.float32)
            row[0] = self._clamp_unit(disaster.id / 10_000.0)
            row[1 + disaster.one_hot_index] = 1.0
            row[5] = self._clamp_unit(disaster.get_scale())
            row[6] = self._clamp_unit(1.0 - disaster.percent_remaining())
            row[7] = self._clamp_unit((self.engine.env.now - disaster.created_time) / self.engine.MAX_SIM_TIME)
            distance = self.engine.get_distance(self.current_resource, disaster)
            speed = float(self.current_resource.resource_type.specs["speed"])
            row[8] = self._clamp_unit(travel_minutes_from_distance(distance, speed) / max_travel_minutes)
            cursor = 9
            for resource_type in ResourceType:
                row[cursor] = self._normalize_resource_count(len(disaster.roster[resource_type]), resource_type)
                cursor += 1
            row[cursor] = self._clamp_unit(disaster.estimated_total_closure_minutes / max_closure_minutes)
            features[idx] = row

        return features

    def _get_obs(self) -> ObsType:
        return {
            "current_resource": self._build_current_resource_features(),
            "global_state": self._build_global_state_features(),
            "candidate_disasters": self._build_candidate_feature_matrix(),
            "valid_actions": self._valid_actions(),
        }

    def _get_info(self) -> InfoType:
        summary: SimulationSummary = self._current_summary() if self.engine is not None else SimulationSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0)
        season_str = "unknown"
        calendar_date = None
        terminal_outcome = self.engine.last_terminal_outcome if self.engine is not None else None
        if self.engine is not None:
            season_str = self.engine.calendar.get_season().name.lower()
            calendar_date = str(self.engine.calendar)

        return {
            "sim_time": self.engine.env.now if self.engine is not None else 0.0,
            "active_disasters": len(self.engine.disaster_store.items) if self.engine is not None else 0,
            "season": season_str,
            "weather_factor": 0.0,
            "calendar_date": calendar_date,
            "total_spent": summary.total_spent,
            "objective_score": compute_objective_score(summary),
            "reward_components": {},
            "summary": summary,
            "terminal_outcome": terminal_outcome,
            "is_success": terminal_outcome == SimPySimulationEngine.TERMINAL_SUCCESS,
            "is_failure": terminal_outcome
            in {
                SimPySimulationEngine.TERMINAL_FAIL_DEADLOCK,
                SimPySimulationEngine.TERMINAL_FAIL_INVALID_STATE,
                SimPySimulationEngine.TERMINAL_FAIL_TIMEOUT,
            },
            "is_truncated": terminal_outcome == SimPySimulationEngine.TERMINAL_FAIL_TIMEOUT,
            "invalid_action": self._last_invalid_action,
            "selected_action": self._last_selected_action,
            "selected_action_kind": self._last_action_kind,
            "visible_candidate_ids": [disaster.id for disaster in self.current_candidates],
            "observation_version": OBSERVATION_VERSION,
            "action_version": ACTION_VERSION,
            "reward_version": REWARD_VERSION,
            "evaluation_version": EVALUATION_PROTOCOL_VERSION,
            "invalid_action_count": self._invalid_action_count,
            "invalid_action_remaps": self._invalid_action_remaps,
            "valid_action_count": len(self.current_candidates),
            "requested_action": self._last_requested_action,
            "executed_action": self._last_executed_action,
        }

    def _resolve_action(self, action: int) -> tuple[str, Disaster | None, bool]:
        if 0 <= action < len(self.current_candidates):
            return "DISPATCH", self.current_candidates[action], False
        return "INVALID", None, True

    def _remap_invalid_action(self) -> tuple[int, Disaster | None]:
        valid_indices = np.flatnonzero(self._valid_actions() == 1)
        if len(valid_indices) == 0:
            return 0, None
        remapped = int(valid_indices[0])
        return remapped, self.current_candidates[remapped]

    def _calculate_reward(
        self,
        *,
        previous_time: float,
        current_time: float,
        terminal_outcome: str | None,
        invalid_action: bool,
        immediate_action_bonus: float,
    ) -> tuple[float, dict[str, float]]:
        if self.engine is None:
            return 0.0, {}

        summary = build_simulation_summary(self.engine)
        objective_score = compute_objective_score(summary)
        reward = 0.0
        components = {
            "dispatch_bonus": 0.0,
            "objective_delta": 0.0,
            "resolution_reward": 0.0,
            "time_penalty": 0.0,
            "invalid_action_penalty": 0.0,
            "terminal_bonus": 0.0,
            "remaining_disaster_penalty": 0.0,
            "progress_reward": 0.0,
        }

        reward += immediate_action_bonus
        components["dispatch_bonus"] = immediate_action_bonus

        if self._prev_objective_score is not None:
            objective_delta = (objective_score - self._prev_objective_score) * 0.1
            reward += objective_delta
            components["objective_delta"] = objective_delta

        active_disasters = len(self.engine.disaster_store.items)
        if self._prev_active_disasters is not None and self._prev_active_disasters > active_disasters:
            resolution_reward = float((self._prev_active_disasters - active_disasters) * 25.0)
            reward += resolution_reward
            components["resolution_reward"] = resolution_reward

        total_percent_remaining = self._total_percent_remaining()
        if self._prev_total_percent_remaining is not None:
            progress_reward = max(0.0, self._prev_total_percent_remaining - total_percent_remaining) * 75.0
            reward += progress_reward
            components["progress_reward"] = progress_reward

        elapsed = max(0.0, current_time - previous_time)
        time_penalty = elapsed * 0.0005
        reward -= time_penalty
        components["time_penalty"] = -time_penalty

        if invalid_action:
            reward -= 5.0
            components["invalid_action_penalty"] = -5.0

        terminal_bonus = 0.0
        if terminal_outcome == SimPySimulationEngine.TERMINAL_SUCCESS:
            terminal_bonus = 750.0
        elif terminal_outcome == SimPySimulationEngine.TERMINAL_FAIL_TIMEOUT:
            terminal_bonus = -250.0
        elif terminal_outcome in {
            SimPySimulationEngine.TERMINAL_FAIL_DEADLOCK,
            SimPySimulationEngine.TERMINAL_FAIL_INVALID_STATE,
        }:
            terminal_bonus = -250.0

        remaining_disaster_penalty = float(active_disasters) * -100.0 if terminal_outcome is not None and active_disasters > 0 else 0.0
        reward += terminal_bonus
        reward += remaining_disaster_penalty
        components["terminal_bonus"] = terminal_bonus
        components["remaining_disaster_penalty"] = remaining_disaster_penalty

        self._prev_objective_score = objective_score
        self._prev_active_disasters = active_disasters
        self._prev_total_percent_remaining = total_percent_remaining
        return float(reward), components

    def _score_action(self, action_kind: str, selected_disaster: Disaster | None, invalid_action: bool) -> float:
        if self.engine is None or self.current_resource is None:
            return 0.0

        if invalid_action:
            return -1.5

        if selected_disaster is None:
            return -1.5

        distance = self.engine.get_distance(self.current_resource, selected_disaster)
        travel_minutes = travel_minutes_from_distance(distance, float(self.current_resource.resource_type.specs["speed"]))
        travel_score = 1.0 - min(max(travel_minutes / self._max_travel_minutes(), 0.0), 1.0)
        urgency_score = min(max(selected_disaster.estimated_total_closure_minutes / self._max_closure_minutes(), 0.0), 1.0)
        remaining_score = min(max(selected_disaster.percent_remaining(), 0.0), 1.0)

        partner_bonus = 0.5
        for resource_type in selected_disaster.needed_resources():
            if resource_type == self.current_resource.resource_type:
                continue
            if len(selected_disaster.roster[resource_type]) == 0:
                partner_bonus -= 0.75

        same_type_penalty = 0.25 * self._normalize_resource_count(
            len(selected_disaster.roster[self.current_resource.resource_type]),
            self.current_resource.resource_type,
        )
        return 2.5 * urgency_score + 1.5 * travel_score + 1.0 * remaining_score + partner_bonus - same_type_penalty

    def _advance_to_decision_or_terminal(self) -> None:
        if self.engine is None:
            return

        try:
            while not self.decision_needed and self.engine.last_terminal_outcome is None:
                self.engine.advance_to_next_event()
        except EmptySchedule:
            self.engine.last_terminal_outcome = self.engine.infer_terminal_outcome(schedule_exhausted=True)
        except Exception as exc:
            self.engine.last_terminal_error = repr(exc)
            self.engine.last_terminal_outcome = SimPySimulationEngine.TERMINAL_FAIL_INVALID_STATE

        if self.engine.last_terminal_outcome is not None:
            self.decision_needed = False
            self.current_resource = None
            self.current_candidates = []

    @override
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[ObsType, InfoType]:  # pyright: ignore[reportIncompatibleMethodOverride]
        super().reset(seed=seed)
        if seed is None:
            seed = int(np.random.randint(0, 2**31 - 1))

        self._episode_seed = seed
        gym_policy = Policy("gym_driver", lambda r, ds, env: ds[0] if ds else None)
        self.engine = SimPySimulationEngine(policy=gym_policy, seed=seed, scenario_config=self.scenario_config)
        self.engine.initialize_world()
        self.engine.run_in_gym(self.loop)

        self.current_resource = None
        self.current_candidates = []
        self.current_action = None
        self.decision_needed = False
        self._invalid_action_count = 0
        self._invalid_action_remaps = 0
        self._last_invalid_action = False
        self._last_selected_action = None
        self._last_requested_action = None
        self._last_executed_action = None
        self._last_action_kind = None
        self._prev_objective_score = compute_objective_score(build_simulation_summary(self.engine))
        self._prev_active_disasters = len(self.engine.disaster_store.items)
        self._prev_total_percent_remaining = self._total_percent_remaining()
        self.logger = DispatchEpisodeLogger(
            controller_name=self.controller_name,
            scenario_name=self.scenario_name,
            seed=seed,
        )

        self._advance_to_decision_or_terminal()
        if self.engine.last_terminal_outcome is not None:
            finalize_episode_log(
                logger=self.logger,
                engine=self.engine,
                episode_seed=self._episode_seed,
                scenario_name=self.scenario_name,
                controller_name=self.controller_name,
                invalid_action_count=self._invalid_action_count,
            )
            return self._zero_obs(), self._get_info()
        return self._get_obs(), self._get_info()

    @override
    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, InfoType]:  # pyright: ignore[reportIncompatibleMethodOverride]
        if self.engine is None:
            raise RuntimeError("Environment must be reset before calling step().")
        if not self.decision_needed or self.current_resource is None:
            raise RuntimeError("Environment is not currently waiting for a dispatch decision.")

        previous_time = self.engine.env.now
        requested_action = int(action)
        selected_action = requested_action
        action_kind, selected_disaster, invalid_action = self._resolve_action(selected_action)
        if invalid_action:
            selected_action, selected_disaster = self._remap_invalid_action()
            if selected_disaster is not None:
                action_kind = "REMAP_INVALID"
        immediate_action_bonus = self._score_action(action_kind, selected_disaster, invalid_action)

        if self.logger is not None:
            self.logger.log_decision(
                build_decision_log_record(
                    engine=self.engine,
                    current_resource=self.current_resource,
                    current_candidate_ids=[disaster.id for disaster in self.current_candidates],
                    valid_actions=self._valid_actions().astype(int).tolist(),
                    action=selected_action,
                    action_kind=action_kind,
                    selected_disaster_id=selected_disaster.id if selected_disaster is not None else None,
                    invalid_action=invalid_action,
                )
            )

        if invalid_action:
            self._invalid_action_count += 1
            if selected_disaster is not None:
                self._invalid_action_remaps += 1

        self._last_invalid_action = invalid_action
        self._last_selected_action = selected_action
        self._last_requested_action = requested_action
        self._last_executed_action = selected_action
        self._last_action_kind = action_kind
        self.current_action = selected_action
        self.decision_needed = False

        self._advance_to_decision_or_terminal()

        terminal_outcome = self.engine.last_terminal_outcome
        terminated = terminal_outcome in {
            SimPySimulationEngine.TERMINAL_SUCCESS,
            SimPySimulationEngine.TERMINAL_FAIL_DEADLOCK,
            SimPySimulationEngine.TERMINAL_FAIL_INVALID_STATE,
        }
        truncated = terminal_outcome == SimPySimulationEngine.TERMINAL_FAIL_TIMEOUT

        obs = self._zero_obs() if (terminated or truncated) else self._get_obs()
        reward, reward_components = self._calculate_reward(
            previous_time=previous_time,
            current_time=self.engine.env.now,
            terminal_outcome=terminal_outcome,
            invalid_action=invalid_action,
            immediate_action_bonus=immediate_action_bonus,
        )
        info = self._get_info()
        info["reward_components"] = reward_components

        if terminated or truncated:
            finalize_episode_log(
                logger=self.logger,
                engine=self.engine,
                episode_seed=self._episode_seed,
                scenario_name=self.scenario_name,
                controller_name=self.controller_name,
                invalid_action_count=self._invalid_action_count,
            )

        return obs, reward, terminated, truncated, info

    def loop(self) -> Generator[Event, Resource, None]:
        if self.engine is None:
            raise RuntimeError("Environment must be reset before calling step().")

        while True:
            yield self.engine.env.process(self.engine.disaster_store.wait_for_any())
            resource = yield self.engine.env.process(self.engine.idle_resources.get_resource_for_disasters(list(self.engine.disaster_store.items)))
            actionable = self._get_actionable_disasters(resource)

            if not actionable:
                resource.assigned_node = self.engine.idle_resources
                self.engine.idle_resources.mark_resource_available(resource)
                yield self.engine.env.timeout(0)
                continue

            self.current_resource = resource
            self.current_candidates = actionable
            self.decision_needed = True
            yield self.engine.env.timeout(0)

            if self.current_action is None:
                raise RuntimeError("No action was provided for the pending dispatch decision.")

            action_kind, selected_disaster, invalid_action = self._resolve_action(self.current_action)
            if invalid_action or selected_disaster is None:
                resource.assigned_node = self.engine.idle_resources
                self.engine.idle_resources.mark_resource_available(resource)
            else:
                if self.engine.branch_decision is None:
                    self.engine.branch_decision = selected_disaster.id
                self.engine.decision_log.append(selected_disaster.id)
                self.engine.decisions_made += 1
                selected_disaster.transfer_resource(resource)

            self.current_action = None
            self.current_resource = None
            self.current_candidates = []

    def update_scenario_config(self, new_config: ScenarioConfig) -> None:
        self.scenario_config = new_config
