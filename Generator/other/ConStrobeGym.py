"""
ConStrobeGym.py — Gymnasium environment that drives a ConStrobe discrete-event
simulation for inference with trained ML/RL dispatch policies.

Drop-in replacement for SimPyTest.gym.DisasterResponseGym.
The observation space, action space, and ObsType dict are identical so that
pre-trained PPO and MLP checkpoints can be used without modification.

Typical usage
-------------
::
    from Generator.ConStrobeGym import ConStrobeGym
    from SimPyTest.scenario_types import ScenarioConfig
    from SimPyTest.benchmark_catalog import EASY

    env = ConStrobeGym(
        max_visible_disasters=8,
        scenario_config=EASY,
        exe_path=r"C:\\Program Files\\constrobe\\constrobe\\constrobe.exe",
    )
    obs, info = env.reset(seed=42)
    while True:
        action, _ = model.predict(obs, action_masks=obs["valid_actions"].astype(bool))
        obs, reward, terminated, truncated, info = env.step(int(action))
        if terminated or truncated:
            break
    env.close()
"""

from __future__ import annotations

import os
from typing import Any, cast

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces
from typing_extensions import override

from SimPyTest.scenario_types import ScenarioConfig
from SimPyTest.simulation import ResourceType
from SimPyTest.gis_utils import (
    CLATSOP_LOCAL_COORD_MAX,
    meters_to_miles,
    travel_minutes_from_distance,
)
from SimPyTest.gym import (
    CURRENT_RESOURCE_FEATURES,
    DISASTER_FEATURES,
    GLOBAL_STATE_FEATURES,
    ObsType,
    InfoType,
    ActType,
)
from SimPyTest.evaluation import (
    SimulationSummary,
    compute_training_objective_score,
    compute_research_metric_bundle,
    ResearchMetricBundle,
)
from SimPyTest.engine import SimulationRNG

from .ConStrobeProcessBridge import ConStrobeProcessBridge, ObsPayload
from .ConStrobeJSTRXBuilder import ConStrobeJSTRXBuilder, MAX_SIM_TIME

# ---------------------------------------------------------------------------
# Constants (must stay in sync with SimPyTest.gym)
# ---------------------------------------------------------------------------

_COORD_MAX_MILES: float = max(1.0, meters_to_miles(CLATSOP_LOCAL_COORD_MAX))
_MAX_ACTIVE_DISASTERS: int = 25
_DEFAULT_EXE_PATH: str = r"C:\Program Files\constrobe\constrobe\constrobe.exe"

# Index positions within ObsPayload.values produced by ConStrobeJSTRXBuilder
_IDX_SIM_TIME: int = ConStrobeJSTRXBuilder.OBS_IDX_SIM_TIME
_IDX_TRUCK: int = ConStrobeJSTRXBuilder.OBS_IDX_TRUCK_COUNT
_IDX_EXCAV: int = ConStrobeJSTRXBuilder.OBS_IDX_EXCAV_COUNT
_IDX_ACTIVE_DIS: int = ConStrobeJSTRXBuilder.OBS_IDX_ACTIVE_DISASTERS
_IDX_SEASON: int = ConStrobeJSTRXBuilder.OBS_IDX_SEASON
_DIS_START: int = ConStrobeJSTRXBuilder.OBS_IDX_DISASTER_FEATURES_START
_DIS_FIELDS: int = ConStrobeJSTRXBuilder.DISASTER_FIELDS_PER_SLOT


# ---------------------------------------------------------------------------
# ConStrobeGym
# ---------------------------------------------------------------------------


class ConStrobeGym(gym.Env[ObsType, ActType]):
    """
    Gymnasium environment that drives constrobe.exe for policy inference.

    Parameters
    ----------
    max_visible_disasters : int
        Number of candidate disaster slots (must match the trained policy).
    scenario_config : ScenarioConfig
        The same config used when training.
    exe_path : str
        Path to the ConStrobe executable.
    output_dir : str
        Directory for generated .jstrx files.
    scenario_name, controller_name : str
        Labels for logging / info dicts.
    obs_timeout : float
        Seconds to wait for an OBS post before treating the step as truncated.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        max_visible_disasters: int,
        scenario_config: ScenarioConfig,
        exe_path: str = _DEFAULT_EXE_PATH,
        output_dir: str = "generated",
        *,
        scenario_name: str = "constrobe",
        controller_name: str = "ppo",
        obs_timeout: float = 60.0,
    ) -> None:
        super().__init__()
        self.max_slots = max_visible_disasters
        self.scenario_config = scenario_config
        self._exe_path = exe_path
        self._output_dir = output_dir
        self.scenario_name = scenario_name
        self.controller_name = controller_name
        self._obs_timeout = obs_timeout

        # ---- Spaces (identical to DisasterResponseGym) ----
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

        # ---- Runtime state ----
        self._bridge: ConStrobeProcessBridge | None = None
        self._builder: ConStrobeJSTRXBuilder | None = None
        self._current_payload: ObsPayload | None = None
        self._episode_terminated: bool = False
        self._episode_truncated: bool = False
        self._episode_seed: int = 0
        self._step_count: int = 0
        self._invalid_action_count: int = 0
        self._prev_objective_score: float | None = None

        # Running episode metrics (approximated from payload deltas)
        self._disasters_created: int = 0
        self._disasters_resolved: int = 0
        self._last_active_disasters: int = 0
        self._total_sim_time_with_disasters: float = 0.0
        self._last_sim_time: float = 0.0

        # Resource count maxima for normalisation
        self._max_trucks: int = self._scenario_max_resource_count(scenario_config.resource_counts.trucks)
        self._max_excavators: int = self._scenario_max_resource_count(scenario_config.resource_counts.excavators)

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    @override
    def reset(self, *, seed: int, options: dict[str, Any] | None = None) -> tuple[ObsType, InfoType]:
        super().reset()
        self._episode_seed = seed

        # Tear down previous episode
        self._close_bridge()

        # Build JSTRX file for this seed
        self._builder = ConStrobeJSTRXBuilder(self.scenario_config, seed=seed, max_slots=self.max_slots)
        self._builder.build_model()
        jstrx_path = self._builder.write(self._output_dir)

        # Spawn subprocess
        generator = self._builder._graph
        assert generator is not None, "build_model() must be called before write()"
        self._bridge = ConStrobeProcessBridge(
            self._exe_path,
            generator,
            obs_timeout=self._obs_timeout,
        )

        # Inject the blocking GET function into the DecisionNode
        dn = self._builder.decision_node
        if dn is not None:
            dn.set_get_fn(self._bridge.blocking_get_fn)

        # Load and start
        self._bridge.load_jstrx(jstrx_path)
        self._bridge.reset_model()
        self._bridge.set_animate(False)

        # Reset episode state
        self._episode_terminated = False
        self._episode_truncated = False
        self._step_count = 0
        self._invalid_action_count = 0
        self._prev_objective_score = None
        self._disasters_created = 0
        self._disasters_resolved = 0
        self._last_active_disasters = 0
        self._total_sim_time_with_disasters = 0.0
        self._last_sim_time = 0.0

        # Start the run; wait for the first decision point
        self._bridge.start_run()
        payload = self._bridge.wait_for_obs()

        if payload is None:
            # Episode ended immediately (degenerate scenario)
            self._episode_terminated = True
            return self._zero_obs(), self._get_info(payload=None)

        self._current_payload = payload
        self._update_running_metrics(payload)
        self._prev_objective_score = self._approx_objective_score()

        return self._build_obs(payload), self._get_info(payload=payload)

    @override
    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, InfoType]:
        if self._bridge is None or self._current_payload is None:
            raise RuntimeError("ConStrobeGym.step() called before reset() or after episode end.")
        if self._episode_terminated or self._episode_truncated:
            raise RuntimeError("ConStrobeGym.step() called on a finished episode. Call reset() first.")

        action = int(action)

        # Validate action against valid mask
        valid = self._valid_actions(self._current_payload)
        invalid_action = bool(valid[action] == 0)
        if invalid_action:
            self._invalid_action_count += 1
            # Remap to first valid action
            valid_indices = np.flatnonzero(valid == 1)
            if len(valid_indices) > 0:
                action = int(valid_indices[0])

        # Deliver action → unblocks the GET in the reader thread
        self._bridge.deliver_action(action)
        self._step_count += 1

        # Check for immediate episode end (ConStrobe may finish before the
        # next OBS is emitted if this was the last disaster)
        done = self._bridge.wait_for_done(timeout=0.05)

        if done:
            # Collect results and terminate
            self._bridge.run_model(blocking=False)  # trigger GETRESULTS
            self._episode_terminated = True
            reward = self._calculate_reward(terminal=True, invalid=invalid_action)
            return (
                self._zero_obs(),
                reward,
                True,
                False,
                self._get_info(payload=None),
            )

        # Wait for next observation
        next_payload = self._bridge.wait_for_obs()

        if next_payload is None:
            # Timed out or episode ended
            done = self._bridge.wait_for_done(timeout=0.5)
            if done:
                self._episode_terminated = True
                reward = self._calculate_reward(terminal=True, invalid=invalid_action)
                return (
                    self._zero_obs(),
                    reward,
                    True,
                    False,
                    self._get_info(payload=None),
                )
            # True timeout — treat as truncation
            self._episode_truncated = True
            return (
                self._zero_obs(),
                self._calculate_reward(terminal=False, invalid=invalid_action),
                False,
                True,
                self._get_info(payload=None),
            )

        self._current_payload = next_payload
        self._update_running_metrics(next_payload)
        reward = self._calculate_reward(terminal=False, invalid=invalid_action)
        obs = self._build_obs(next_payload)
        return obs, reward, False, False, self._get_info(payload=next_payload)

    @override
    def close(self) -> None:
        self._close_bridge()
        super().close()

    def action_masks(self) -> npt.NDArray[np.bool_]:
        """SB3-MaskablePPO interface."""
        if self._current_payload is None:
            return np.zeros(self.max_slots, dtype=np.bool_)
        return self._valid_actions(self._current_payload).astype(np.bool_)

    def update_scenario_config(self, new_config: ScenarioConfig) -> None:
        """Hot-swap scenario config (takes effect on next reset())."""
        self.scenario_config = new_config
        self._max_trucks = self._scenario_max_resource_count(new_config.resource_counts.trucks)
        self._max_excavators = self._scenario_max_resource_count(new_config.resource_counts.excavators)

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def _build_obs(self, payload: ObsPayload) -> ObsType:
        return {
            "current_resource": self._build_current_resource_features(payload),
            "global_state": self._build_global_state_features(payload),
            "candidate_disasters": self._build_candidate_feature_matrix(payload),
            "valid_actions": self._valid_actions(payload),
        }

    def _build_current_resource_features(self, payload: ObsPayload) -> npt.NDArray[np.float32]:
        """
        7-element vector: [resource_id_norm, type_truck, type_excav,
                           ?, ?, x_norm, y_norm]

        ConStrobe does not track which specific resource triggered the decision
        (the entity is a token, not an object with an ID).  We populate what
        we can from the payload and zero the rest.  The trained policy is
        robust to zeroed unused features.
        """
        feats = np.zeros(CURRENT_RESOURCE_FEATURES, dtype=np.float32)
        # Truck count in payload; if trucks are available the triggering
        # resource is more likely a truck (heuristic: dominant type).
        truck_count = payload.get_float(_IDX_TRUCK)
        excav_count = payload.get_float(_IDX_EXCAV)
        total = max(1, self._max_trucks + self._max_excavators)

        if truck_count > 0:
            feats[1] = 1.0  # type_truck one-hot
        elif excav_count > 0:
            feats[2] = 1.0  # type_excav one-hot
        # feats[0] = resource_id_norm — 0 (unknown)
        # feats[-2], feats[-1] = x, y — 0 (resource location not tracked)
        return feats

    def _build_global_state_features(self, payload: ObsPayload) -> npt.NDArray[np.float32]:
        """
        10-element vector: [time_norm, season_0..3, active_disasters_norm,
                            truck_idle_norm, excav_idle_norm]
        """
        feats = np.zeros(GLOBAL_STATE_FEATURES, dtype=np.float32)
        sim_time = payload.get_float(_IDX_SIM_TIME)
        season_idx = int(payload.get_float(_IDX_SEASON))
        active = payload.get_float(_IDX_ACTIVE_DIS)
        trucks = payload.get_float(_IDX_TRUCK)
        excavs = payload.get_float(_IDX_EXCAV)

        feats[0] = self._clamp(sim_time / MAX_SIM_TIME)
        season_idx = max(0, min(3, season_idx))
        feats[1 + season_idx] = 1.0
        feats[5] = self._clamp(active / _MAX_ACTIVE_DISASTERS)
        feats[6] = self._clamp(trucks / max(1, self._max_trucks))
        feats[7] = self._clamp(excavs / max(1, self._max_excavators))
        # feats[8], feats[9] spare — zero
        return feats

    def _build_candidate_feature_matrix(self, payload: ObsPayload) -> npt.NDArray[np.float32]:
        """
        (max_slots × DISASTER_FEATURES) matrix.
        DISASTER_FEATURES = 14 (matches DisasterResponseGym).
        """
        matrix = np.zeros((self.max_slots, DISASTER_FEATURES), dtype=np.float32)
        sim_time = payload.get_float(_IDX_SIM_TIME)

        for slot in range(self.max_slots):
            base = _DIS_START + slot * _DIS_FIELDS
            dis_id = payload.get_float(base + 0)
            dis_type = payload.get_float(base + 1)  # 0=landslide, 1=wildfire
            scale = payload.get_float(base + 2)
            pct_remaining = payload.get_float(base + 3)
            age_norm = payload.get_float(base + 4)
            x_norm = payload.get_float(base + 5)
            y_norm = payload.get_float(base + 6)

            if scale == 0.0 and pct_remaining == 0.0:
                continue  # empty slot

            row = matrix[slot]
            row[0] = self._clamp(dis_id / 10_000.0)
            # one-hot for disaster type (indices 1–4, same as SimPy gym)
            type_idx = int(dis_type)
            if 0 <= type_idx < 4:
                row[1 + type_idx] = 1.0
            row[5] = self._clamp(scale)
            row[6] = self._clamp(1.0 - pct_remaining)
            row[7] = self._clamp(age_norm)
            row[8] = self._clamp(
                travel_minutes_from_distance(
                    x_norm * _COORD_MAX_MILES,
                    EXCAVATOR_SPEED_MPH := ResourceType.EXCAVATOR.specs["speed"],
                )
                / self._max_travel_minutes()
            )
            row[9] = x_norm  # truck roster size placeholder → x_norm
            row[10] = y_norm  # excav roster size placeholder → y_norm
            # rows 11–13 spare

        return matrix

    def _valid_actions(self, payload: ObsPayload) -> npt.NDArray[np.int8]:
        mask = np.zeros(self.max_slots, dtype=np.int8)
        for slot in range(self.max_slots):
            base = _DIS_START + slot * _DIS_FIELDS
            scale = payload.get_float(base + 2)
            pct = payload.get_float(base + 3)
            if scale > 0.0 or pct > 0.0:
                mask[slot] = 1
        return mask

    def _zero_obs(self) -> ObsType:
        return {
            "current_resource": np.zeros(CURRENT_RESOURCE_FEATURES, dtype=np.float32),
            "global_state": np.zeros(GLOBAL_STATE_FEATURES, dtype=np.float32),
            "candidate_disasters": np.zeros((self.max_slots, DISASTER_FEATURES), dtype=np.float32),
            "valid_actions": np.zeros(self.max_slots, dtype=np.int8),
        }

    # ------------------------------------------------------------------
    # Reward & info
    # ------------------------------------------------------------------

    def _calculate_reward(self, terminal: bool, invalid: bool) -> float:
        current_score = self._approx_objective_score()
        prev = self._prev_objective_score or current_score
        delta = current_score - prev
        reward = delta / 1_000.0
        if invalid:
            reward -= 2.0
        self._prev_objective_score = current_score
        return float(reward)

    def _approx_objective_score(self) -> float:
        """
        Approximate the training objective score from running episode metrics.
        Uses the same formula as compute_training_objective_score() but
        sourced from the in-model counters rather than SimPy objects.
        """
        unresolved = max(0, self._disasters_created - self._disasters_resolved)
        unresolved_penalty = float(unresolved) * 100_000.0
        time_cost_hours = self._total_sim_time_with_disasters / 60.0
        # No spending cost without per-resource tracking
        return -(unresolved_penalty + time_cost_hours)

    def _get_info(self, payload: ObsPayload | None) -> InfoType:
        sim_time = payload.get_float(_IDX_SIM_TIME) if payload else self._last_sim_time
        active = payload.get_float(_IDX_ACTIVE_DIS) if payload else 0.0
        season_idx = payload.get_float(_IDX_SEASON) if payload else 0.0
        season_names = ["winter", "spring", "summer", "fall"]
        season_str = season_names[max(0, min(3, int(season_idx)))]

        terminal_outcome: str | None = None
        if self._episode_terminated:
            terminal_outcome = "SUCCESS" if active == 0.0 else "FAIL_TIMEOUT"
        elif self._episode_truncated:
            terminal_outcome = "FAIL_TIMEOUT"

        summary = SimulationSummary(
            terminal_outcome=terminal_outcome,
            time_with_disasters=self._total_sim_time_with_disasters,
            total_drive_time=0.0,
            total_operating_cost=0.0,
            total_fuel_cost=0.0,
            total_spent=0.0,
            total_resource_hours=0.0,
            disasters_created=self._disasters_created,
            disasters_resolved=self._disasters_resolved,
            resolution_rate=(self._disasters_resolved / max(1, self._disasters_created)),
            avg_response_time=0.0,
            avg_resolution_time=0.0,
            total_weighted_closure_hours=0.0,
        )
        obj_score = compute_training_objective_score(summary)

        return {
            "sim_time": sim_time,
            "active_disasters": int(active),
            "season": season_str,
            "calendar_date": None,
            "total_spent": 0.0,
            "research_metrics": compute_research_metric_bundle(summary),
            "training_objective_score": obj_score,
            "objective_score": obj_score,
            "reward_components": {},
            "summary": summary,
            "terminal_outcome": terminal_outcome,
            "is_success": terminal_outcome == "SUCCESS",
            "is_failure": terminal_outcome in {"FAIL_DEADLOCK", "FAIL_INVALID_STATE", "FAIL_TIMEOUT"},
            "is_truncated": terminal_outcome == "FAIL_TIMEOUT",
            "invalid_action": False,
            "selected_action": None,
            "selected_action_kind": None,
            "visible_candidate_ids": [],
            "observation_version": "constrobe_v1",
            "action_version": "v3",
            "reward_version": "constrobe_v1",
            "invalid_action_count": self._invalid_action_count,
            "invalid_action_remaps": 0,
            "valid_action_count": int(sum(self._valid_actions(payload)) if payload else 0),
            "requested_action": None,
            "executed_action": None,
        }

    # ------------------------------------------------------------------
    # Running metric bookkeeping
    # ------------------------------------------------------------------

    def _update_running_metrics(self, payload: ObsPayload) -> None:
        sim_time = payload.get_float(_IDX_SIM_TIME)
        active = payload.get_float(_IDX_ACTIVE_DIS)

        # Estimate time-with-disasters using trapezoid rule between steps
        dt = max(0.0, sim_time - self._last_sim_time)
        if self._last_active_disasters > 0:
            self._total_sim_time_with_disasters += dt

        # Infer disaster creation / resolution from changes in counter
        if active > self._last_active_disasters:
            self._disasters_created += int(active - self._last_active_disasters)
        elif active < self._last_active_disasters:
            self._disasters_resolved += int(self._last_active_disasters - active)

        self._last_sim_time = sim_time
        self._last_active_disasters = int(active)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _close_bridge(self) -> None:
        if self._bridge is not None:
            try:
                self._bridge.close()
                self._bridge.cleanup()
            except Exception:
                pass
            self._bridge = None

    @staticmethod
    def _clamp(v: float) -> float:
        return float(min(max(v, 0.0), 1.0))

    def _max_travel_minutes(self) -> float:
        slowest = min(float(rt.specs["speed"]) for rt in ResourceType)
        return max(1.0, travel_minutes_from_distance(_COORD_MAX_MILES, slowest))

    @staticmethod
    def _scenario_max_resource_count(value: int | tuple[int, int]) -> int:
        if isinstance(value, tuple):
            return int(value[1])
        return int(value)
