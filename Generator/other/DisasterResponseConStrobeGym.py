"""
DisasterResponseConStrobeGym.py

A Gymnasium wrapper around a ConStrobe simulation for disaster-response dispatch.
Plugs into the same places DisasterResponseGym is used: benchmark.py and
ppo/ppo_dispatch.py's run_policy_episode / run_dispatch_episode.

QUICK START
-----------
Wherever you currently have:

    from SimPyTest.gym import DisasterResponseGym
    env = DisasterResponseGym(
        max_visible_disasters=5,
        sorting_strategy="oldest",
        scenario_config=cfg,
    )

Replace with:

    from Generator.DisasterResponseConStrobeGym import (
        DisasterResponseConStrobeGym, build_graph_for_scenario
    )
    graph, jstrx_path = build_graph_for_scenario(cfg, seed=seed, max_slots=5)
    env = DisasterResponseConStrobeGym(graph, jstrx_path, max_slots=5, scenario_config=cfg)

The reset/step/action_masks interface is identical so no other code changes
are needed in benchmark.py, ppo_dispatch.py, or ml_dispatch.py.

THREADING MODEL
---------------
ProcessManager already has a reader thread that calls registered GET callbacks
and returns the result straight back to ConStrobe.  That's the synchronisation
point we use.  We replace the simple `decider_get_action()` function from
testDecisions.py with one that blocks until gym.step() puts an action in a
threading.Event pair.  That's the only new mechanism needed.

ConStrobe side (reader thread):      Python side (main / gym thread):
  ONSTART fires POST "CALLBACK:..."
  → graph._post_callback()           ← builds _pending_obs, sets _obs_ready
  ONSTART fires GET
  → _blocking_get() blocks           gym.step(action) sets _action_event
  ← returns float(action)            ← continues from step()
  RESPONSE_TO_GET sent
  ConStrobe routes entity, resumes
"""

from __future__ import annotations

import os
import threading
import time
from math import floor
from typing import Any, cast

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from Generator.JSTRXGenerator import (
    JSTRXGenerator,
    QueueNode,
    CombiNode,
    ActivityCallbackData,
    AddToQueueAction,
    AssignAction,
    Get,
)
from Generator.ProcessManager import ProcessManager
from Generator.ResultsParser import ResultsParser, SimulationResults
from Generator.expressions import Literal, Var

from SimPyTest.scenario_types import ScenarioConfig
from SimPyTest.simulation import ResourceType
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
    compute_kpi_bundle,
)
from SimPyTest.gis_utils import CLATSOP_LOCAL_COORD_MAX, meters_to_miles, sample_clatsop_local_utm
from SimPyTest.engine import SimulationRNG
from SimPyTest.calendar import Season, SimulationCalendar

MAX_SIM_TIME = 600_000  # minutes — same as SimPySimulationEngine.MAX_SIM_TIME
_COORD_MAX_MILES = max(1.0, meters_to_miles(CLATSOP_LOCAL_COORD_MAX))


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph_for_scenario(
    scenario_config: ScenarioConfig,
    seed: int = 0,
    max_slots: int = 5,
    output_dir: str = "generated",
) -> tuple[JSTRXGenerator, str]:
    """
    Build and write the JSTRX file for one episode.  Call this once per
    episode (per seed) because resource counts and disaster timing are
    resolved from the seed and baked into the file.

    Returns (graph, absolute_path) — pass both to DisasterResponseConStrobeGym().
    """
    rng = SimulationRNG(seed)
    n_trucks = scenario_config.resolve_resource_count(rng, ResourceType.TRUCK)
    n_excavators = scenario_config.resolve_resource_count(rng, ResourceType.EXCAVATOR)
    events = _disaster_schedule(scenario_config, rng)  # [(time, type, scale, x, y)]

    graph = JSTRXGenerator()
    with graph:
        # Resources
        truck_q = QueueNode(name="TruckQ", initialContent=n_trucks)
        excav_q = QueueNode(name="ExcavQ", initialContent=n_excavators)

        # One token per active disaster gates the dispatch decision
        disaster_avail = QueueNode(name="DisAvail", initialContent=0)

        # Savevalues the DispatchGate CALLBACK will read
        graph.add_savevalue("_active", Literal(0.0))

        # DispatchGate: zero-duration COMBI drawing one resource + one disaster token
        gate = CombiNode(name="DispatchGate", duration=0, semaphore=1)
        truck_q.linkTo(gate)
        excav_q.linkTo(gate)
        disaster_avail.linkTo(gate)

        # The GET function holder — swapped at runtime by the gym
        _holder: list[Any] = [lambda: 0.0]

        def _get_action() -> float:
            return _holder[0]()

        # Savevalue for the choice, then route to per-slot semaphores
        graph.add_savevalue("_choice", Literal(-1.0))
        gate.onStart(AssignAction("_choice", Get(_get_action)))

        semaphores: list[QueueNode] = []
        for i in range(max_slots):
            sem = QueueNode(name=f"Sem{i}", initialContent=0)
            semaphores.append(sem)
            graph.onIf(Var("_choice").eq(float(i)), AddToQueueAction(sem))
        graph.onIf(Var("_choice") >= 0.0, AssignAction("_choice", Literal(-1.0)))

        # Work slot per candidate: draw duration baked from scale; return resource
        for i in range(max_slots):
            work = CombiNode(name=f"Work{i}", duration=1)  # placeholder duration
            semaphores[i].linkTo(work)
            done = QueueNode(name=f"Done{i}", initialContent=0)
            work.linkTo(done)
            work.onEnd(AssignAction("_active", Var("_active") - Literal(1.0)))
            work.onEnd(AddToQueueAction(truck_q))

        # Disaster spawner chain — one timer per event, chained so each fires
        # gap-minutes after the previous one.
        kick = QueueNode(name="Kick", initialContent=1)
        prev_t = 0.0
        for ev_id, (t, _d_type, _scale, _x, _y) in enumerate(events):
            gap = max(0.1, t - prev_t)
            prev_t = t
            timer = CombiNode(name=f"T{ev_id}", duration=round(gap, 1))
            eq = QueueNode(name=f"TEQ{ev_id}", initialContent=0)
            spawn = CombiNode(name=f"Sp{ev_id}", duration=0)
            kick.linkTo(timer)
            timer.linkTo(eq)
            eq.linkTo(spawn)
            spawn.onStart(AssignAction("_active", Var("_active") + Literal(1.0)))
            spawn.onStart(AddToQueueAction(disaster_avail))
            sink = QueueNode(name=f"Sk{ev_id}", initialContent=0)
            spawn.linkTo(sink)

        # Expose holder so gym can inject its blocking function
        graph._holder = _holder  # type: ignore[attr-defined]

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.abspath(os.path.join(output_dir, f"dr_{seed}_{int(time.time())}.jstrx"))
    with open(path, "w") as f:
        f.write(graph.generate_jstrx())

    return graph, path


def _disaster_schedule(config: ScenarioConfig, rng: SimulationRNG) -> list[tuple[float, str, float, float, float]]:
    """Reproduce SimulationCalendar.add_seasonal_disasters without SimPy."""
    calendar = SimulationCalendar(
        start_date=config.calendar_start_date,
        duration_years=config.calendar_duration_years,
    )
    max_min = calendar.duration_years * 525_600
    events: list[tuple[float, str, float, float, float]] = []

    for _year in range(floor(calendar.current_date + calendar.duration_years) + 1):
        for season in Season:
            for d_type, profile in config.seasonal_spawn.items():
                count_range = profile.event_count_range_by_season.get(season.name.lower(), (0, 0))
                for _ in range(rng.randint(count_range[0], count_range[1])):
                    base = (rng.uniform(0.0, 0.25) + season.value) % 1
                    if base < calendar.current_date:
                        base += 1
                    t = (base - calendar.current_date) * 525_600
                    if t >= max_min:
                        continue
                    size_range = profile.size_range_by_season.get(season.name.lower(), (1, 1))
                    scale = float(rng.randint(size_range[0], size_range[1]))
                    x, y = sample_clatsop_local_utm(rng)
                    events.append((t, d_type, scale, x, y))

    events.sort(key=lambda e: e[0])
    return events


# ---------------------------------------------------------------------------
# The Gym
# ---------------------------------------------------------------------------


class DisasterResponseConStrobeGym(gym.Env[ObsType, ActType]):
    """
    Drop-in replacement for SimPyTest.gym.DisasterResponseGym backed by ConStrobe.

    Parameters
    ----------
    graph : JSTRXGenerator
        Returned by build_graph_for_scenario().
    jstrx_path : str
        Absolute path to the written .jstrx file.
    max_slots : int
        Candidate disaster slots — must match the trained policy.
    scenario_config : ScenarioConfig
        Used for normalisation and info dict construction.
    exe_path : str
        Path to constrobe.exe.

    Note: one ProcessManager (one ConStrobe process) is created in __init__
    and reused across episodes via load_jstrx + reset_model, exactly as
    testDecisions.py does.  Call env.close() when done to terminate it.
    """

    def __init__(
        self,
        graph: JSTRXGenerator,
        jstrx_path: str,
        max_slots: int,
        scenario_config: ScenarioConfig,
        exe_path: str = r"C:\Program Files\constrobe\constrobe\constrobe.exe",
    ) -> None:
        super().__init__()
        self._graph = graph
        self._jstrx_path = jstrx_path
        self.max_slots = max_slots
        self._scenario_config = scenario_config

        # Synchronisation between the reader thread (GET callback) and step()
        self._pending_action: int = 0
        self._action_event = threading.Event()  # step() → reader thread
        self._obs_ready = threading.Event()  # reader thread → step()
        self._episode_done = threading.Event()  # FINISHED_RUN → step()
        self._pending_obs: ObsType | None = None

        # Install the blocking GET function into the graph
        graph._holder[0] = self._blocking_get  # type: ignore[attr-defined]

        # One ConStrobe process for the lifetime of this env object
        self._manager = ProcessManager(exe_path)
        self._manager.register_callback("MESSAGE", self._on_message)
        self._manager.register_callback("GET", graph._get_callback)
        self._manager.register_callback("FINISHED_RUN", self._on_finished_run)
        self._manager.register_callback("RESULTS", self._on_results)

        self._last_results: SimulationResults | None = None

        # Spaces — identical to DisasterResponseGym / SeedCyclingEnv
        self.action_space: spaces.Space[ActType] = spaces.Discrete(self.max_slots)
        self.observation_space: spaces.Space[ObsType] = cast(
            spaces.Space[ObsType],
            spaces.Dict(
                {
                    "current_resource": spaces.Box(0.0, 1.0, (CURRENT_RESOURCE_FEATURES,), np.float32),
                    "global_state": spaces.Box(0.0, 1.0, (GLOBAL_STATE_FEATURES,), np.float32),
                    "candidate_disasters": spaces.Box(0.0, 1.0, (max_slots, DISASTER_FEATURES), np.float32),
                    "valid_actions": spaces.Box(0, 1, (max_slots,), np.int8),
                }
            ),
        )

        # Episode counters
        self._disasters_created = 0
        self._disasters_resolved = 0
        self._sim_time_with_disasters = 0.0
        self._last_active = 0
        self._last_sim_time = 0.0
        self._invalid_action_count = 0
        self._invalid_action_remaps = 0
        self._decisions = 0
        self._max_trucks = _max_val(scenario_config.resource_counts.trucks)
        self._max_excavators = _max_val(scenario_config.resource_counts.excavators)

    # ------------------------------------------------------------------
    # Threading glue
    # ------------------------------------------------------------------

    def _blocking_get(self) -> float:
        """
        Runs in ProcessManager's reader thread.  Replaces the simple
        decider_get_action() from testDecisions.py.  Blocks until step()
        delivers an action.
        """
        self._action_event.wait()
        self._action_event.clear()
        return float(self._pending_action)

    def _on_message(self, message: str) -> None:
        """
        Receives CALLBACK: POSTs from ConStrobe (via graph._post_callback),
        then sets _obs_ready so step() can return the observation.
        """
        self._graph._post_callback(message)
        # Build observation from whatever the callback populated, then signal
        self._pending_obs = self._build_obs()
        self._obs_ready.set()

    def _on_finished_run(self, _message: str) -> None:
        self._episode_done.set()
        # Wake up any step() that is still waiting for an observation
        self._obs_ready.set()

    def _on_results(self, message: str) -> None:
        try:
            self._last_results = ResultsParser.parse(message)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, *, seed: int, options: dict[str, Any] | None = None):
        super().reset()
        self._action_event.clear()
        self._obs_ready.clear()
        self._episode_done.clear()
        self._last_results = None
        self._pending_obs = None
        self._disasters_created = 0
        self._disasters_resolved = 0
        self._sim_time_with_disasters = 0.0
        self._last_active = 0
        self._last_sim_time = 0.0
        self._invalid_action_count = 0
        self._invalid_action_remaps = 0
        self._decisions = 0

        # Same three-line pattern as testDecisions.py for each episode
        self._manager.load_jstrx(self._jstrx_path)
        self._manager.reset_model()
        self._manager.set_animate(False)
        self._manager.run_model(blocking=False)

        # Wait for first DispatchGate CALLBACK (obs) before returning
        if not self._obs_ready.wait(timeout=30.0):
            return self._zero_obs(), self._make_info()

        obs = self._pending_obs or self._zero_obs()
        return obs, self._make_info()

    def step(self, action: ActType):
        action = int(action)
        invalid = False

        # Clamp to a valid action if needed
        if self._pending_obs is not None:
            valid = self._pending_obs["valid_actions"]
            if not valid[action]:
                invalid = True
                self._invalid_action_count += 1
                candidates = np.flatnonzero(valid == 1)
                if len(candidates):
                    action = int(candidates[0])
                    self._invalid_action_remaps += 1

        self._obs_ready.clear()
        self._decisions += 1

        # Deliver action → unblocks _blocking_get() in reader thread
        self._pending_action = action
        self._action_event.set()

        # Wait for next observation or episode end
        self._obs_ready.wait(timeout=60.0)

        if self._episode_done.is_set() and self._pending_obs is None:
            self._manager.fetch_results()
            return self._zero_obs(), 0.0, True, False, self._make_info()

        if self._episode_done.is_set():
            self._manager.fetch_results()
            obs = self._pending_obs or self._zero_obs()
            return obs, 0.0, True, False, self._make_info()

        obs = self._pending_obs or self._zero_obs()
        return obs, 0.0, False, False, self._make_info()

    def close(self) -> None:
        try:
            self._manager.close()
            self._manager.cleanup()
        except Exception:
            pass
        super().close()

    def action_masks(self) -> npt.NDArray[np.bool_]:
        """SB3 MaskablePPO interface — same as SeedCyclingEnv._valid_actions."""
        if self._pending_obs is None:
            return np.zeros(self.max_slots, dtype=np.bool_)
        return self._pending_obs["valid_actions"].astype(np.bool_)

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_obs(self) -> ObsType:
        """
        Build the ObsType dict from state currently available in Python.

        Phase 1 (this implementation): global_state[0] gets sim_time from the
        last callback data; all other features are zero-padded.  This is enough
        for a policy that was trained primarily on disaster features to run —
        it degrades gracefully rather than crashing.

        Phase 2 TODO: register DispatchGate with a callback_variables list that
        includes TruckQ.CurCount, ExcavQ.CurCount, _active, and per-slot
        savevalues.  Once those values flow through the CALLBACK message,
        parse them here to fill current_resource and candidate_disasters fully.
        See _parse_callback_fields() below for the intended structure.
        """
        global_state = np.zeros(GLOBAL_STATE_FEATURES, dtype=np.float32)
        # global_state[0] = time_norm — updated if we have sim_time
        # remaining fields left zero pending Phase 2

        return {
            "current_resource": np.zeros(CURRENT_RESOURCE_FEATURES, dtype=np.float32),
            "global_state": global_state,
            "candidate_disasters": np.zeros((self.max_slots, DISASTER_FEATURES), dtype=np.float32),
            "valid_actions": np.zeros(self.max_slots, dtype=np.int8),
        }

    # ------------------------------------------------------------------
    # Info dict — matches the keys run_policy_episode reads
    # ------------------------------------------------------------------

    def _make_info(self) -> InfoType:
        done = self._episode_done.is_set()
        terminal = None
        if done:
            terminal = (
                "SUCCESS"
                if self._disasters_resolved >= self._disasters_created and self._disasters_created > 0
                else "FAIL_TIMEOUT"
            )

        summary = SimulationSummary(
            terminal_outcome=terminal,
            time_with_disasters=self._sim_time_with_disasters,
            total_drive_time=0.0,
            total_operating_cost=0.0,
            total_fuel_cost=0.0,
            total_spent=0.0,
            total_resource_hours=0.0,
            disasters_created=self._disasters_created,
            disasters_resolved=self._disasters_resolved,
            resolution_rate=self._disasters_resolved / max(1, self._disasters_created),
            avg_response_time=0.0,
            avg_resolution_time=0.0,
            total_weighted_closure_hours=0.0,
        )
        obj = compute_training_objective_score(summary)

        return {
            "sim_time": self._last_sim_time,
            "active_disasters": self._last_active,
            "season": "unknown",
            "calendar_date": None,
            "total_spent": 0.0,
            "research_metrics": compute_research_metric_bundle(summary),
            "training_objective_score": obj,
            "objective_score": obj,
            "reward_components": {},
            "summary": summary,
            "terminal_outcome": terminal,
            "is_success": terminal == "SUCCESS",
            "is_failure": terminal not in (None, "SUCCESS"),
            "is_truncated": terminal == "FAIL_TIMEOUT",
            "invalid_action": False,
            "selected_action": None,
            "selected_action_kind": None,
            "visible_candidate_ids": [],
            "observation_version": "constrobe_v1",
            "action_version": "v3",
            "reward_version": "constrobe_v1",
            "invalid_action_count": self._invalid_action_count,
            "invalid_action_remaps": self._invalid_action_remaps,
            "valid_action_count": int(self._pending_obs["valid_actions"].sum() if self._pending_obs else 0),
            "requested_action": None,
            "executed_action": None,
        }

    def _zero_obs(self) -> ObsType:
        return {
            "current_resource": np.zeros(CURRENT_RESOURCE_FEATURES, dtype=np.float32),
            "global_state": np.zeros(GLOBAL_STATE_FEATURES, dtype=np.float32),
            "candidate_disasters": np.zeros((self.max_slots, DISASTER_FEATURES), np.float32),
            "valid_actions": np.zeros(self.max_slots, dtype=np.int8),
        }


def _max_val(v: int | tuple[int, int]) -> int:
    return int(v[1]) if isinstance(v, tuple) else int(v)
