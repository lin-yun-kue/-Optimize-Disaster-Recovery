"""
ConStrobeJSTRXBuilder.py — Builds JSTRX model files for the disaster-response
domain from a SimPyTest ScenarioConfig.

The generated model mirrors the SimPyTest simulation topology:
  - Resource queues (one per ResourceType) pre-loaded with resource tokens.
  - A full seasonal disaster calendar pre-computed from the seed and baked
    in as timed ADDTOQUEUE events.
  - A DecisionNode that gates every dispatch decision behind a GET round-trip.
  - Per-disaster work COMBI nodes sized by scale.
  - Resource return paths back to the appropriate queue.
  - An active-disaster counter savevalue used in the ENDCOND expression.

Usage
-----
::
    path = ConStrobeJSTRXBuilder.build(
        scenario_config=cfg,
        seed=42,
        max_slots=8,
        output_dir="generated/",
    )
    # path is the absolute path to the written .jstrx file
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from math import floor
from typing import NamedTuple

from .JSTRXGenerator import (
    CURRENT_GRAPH,
    JSTRXGenerator,
    QueueNode,
    CombiNode,
    AddToQueueAction,
    AssignAction,
    Get,
)
from .decision_node import DecisionNode, DisasterActivityNode
from .expressions import Literal, Var

# We import from SimPyTest only to reuse the RNG and calendar logic.
# No SimPy simulation objects are created.
from SimPyTest.engine import SimulationRNG
from SimPyTest.calendar import Season, SimulationCalendar
from SimPyTest.scenario_types import ScenarioConfig
from SimPyTest.simulation import ResourceType
from SimPyTest.gis_utils import (
    CLATSOP_LOCAL_COORD_MAX,
    sample_clatsop_local_utm,
    meters_to_miles,
    travel_minutes_from_distance,
)

# ---------------------------------------------------------------------------
# Constants matching SimPyTest values
# ---------------------------------------------------------------------------

MAX_SIM_TIME: int = 600_000  # minutes (same as SimPySimulationEngine.MAX_SIM_TIME)
TRUCK_SPEED_MPH: float = ResourceType.TRUCK.specs["speed"]  # 23.4
EXCAVATOR_SPEED_MPH: float = ResourceType.EXCAVATOR.specs["speed"]  # 11.6
TRUCK_CAPACITY_CY: float = ResourceType.TRUCK.specs["capacity"]  # 10  cubic yards
EXCAVATOR_WORK_RATE_CY_HR: float = ResourceType.EXCAVATOR.specs.get("work_rate", 10.0)
# Work rate in cubic yards per minute
EXCAVATOR_WORK_RATE_CY_MIN: float = EXCAVATOR_WORK_RATE_CY_HR / 60.0

# Normalisation denominator for coordinate features
_COORD_MAX_MILES: float = max(1.0, meters_to_miles(CLATSOP_LOCAL_COORD_MAX))


# ---------------------------------------------------------------------------
# Internal schedule record
# ---------------------------------------------------------------------------


class _DisasterEvent(NamedTuple):
    spawn_time_minutes: float  # minutes from start of simulation
    disaster_type: str  # "landslide" | "wildfire_debris"
    scale_cubic_yards: float  # total work
    x_utm: float  # raw UTM x (metres, not normalised)
    y_utm: float  # raw UTM y


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class ConStrobeJSTRXBuilder:
    """
    Generates a JSTRX model for disaster-response dispatch.

    Instantiate, call ``build_model()``, then ``write()`` to get a file path.
    The static ``build()`` convenience method does all three in one call.
    """

    def __init__(
        self,
        scenario_config: ScenarioConfig,
        seed: int = 0,
        max_slots: int = 8,
    ) -> None:
        self.config = scenario_config
        self.seed = seed
        self.max_slots = max_slots
        self._rng = SimulationRNG(seed)
        self._graph: JSTRXGenerator | None = None
        self._decision_node: DecisionNode | None = None
        self._disaster_events: list[_DisasterEvent] = []
        self._disaster_nodes: list[DisasterActivityNode] = []
        # Computed resource counts (resolved from config ranges)
        self._n_trucks: int = 0
        self._n_excavators: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        scenario_config: ScenarioConfig,
        seed: int = 0,
        max_slots: int = 8,
        output_dir: str = "generated",
    ) -> str:
        """
        Build and write a JSTRX file.  Returns the absolute file path.
        """
        builder = cls(scenario_config, seed, max_slots)
        builder.build_model()
        return builder.write(output_dir)

    def build_model(self) -> JSTRXGenerator:
        """
        Construct the full JSTRX model graph.  Returns the generator so callers
        can inspect or further extend the model before writing.
        """
        self._resolve_resources()
        self._compute_disaster_schedule()

        self._graph = JSTRXGenerator()
        with self._graph:
            self._build_graph()

        return self._graph

    def write(self, output_dir: str = "generated") -> str:
        """
        Write the generated JSTRX to disk.  Returns the absolute file path.
        Must call ``build_model()`` first.
        """
        if self._graph is None:
            raise RuntimeError("Call build_model() before write().")

        os.makedirs(output_dir, exist_ok=True)
        filename = f"disaster_response_{self.seed}_{int(time.time())}.jstrx"
        path = os.path.abspath(os.path.join(output_dir, filename))
        with open(path, "w") as f:
            f.write(self._graph.generate_jstrx())
        return path

    @property
    def decision_node(self) -> DecisionNode | None:
        """The DecisionNode created during build_model(), if any."""
        return self._decision_node

    @property
    def disaster_nodes(self) -> list[DisasterActivityNode]:
        """All disaster activity nodes in spawn order."""
        return list(self._disaster_nodes)

    # ------------------------------------------------------------------
    # Step 1: Resolve resource counts
    # ------------------------------------------------------------------

    def _resolve_resources(self) -> None:
        self._n_trucks = self.config.resolve_resource_count(self._rng, ResourceType.TRUCK)
        self._n_excavators = self.config.resolve_resource_count(self._rng, ResourceType.EXCAVATOR)

    # ------------------------------------------------------------------
    # Step 2: Pre-compute disaster schedule (mirrors calendar.py logic)
    # ------------------------------------------------------------------

    def _compute_disaster_schedule(self) -> None:
        """
        Reproduce add_seasonal_disasters() without SimPy, storing plain
        (_spawn_time_minutes, disaster_type, scale, x, y) tuples.
        """
        calendar = SimulationCalendar(
            start_date=self.config.calendar_start_date,
            duration_years=self.config.calendar_duration_years,
        )

        max_sim_minutes = calendar.duration_years * 525_600
        sim_year_span = floor(calendar.current_date + calendar.duration_years) + 1
        profiles = self.config.seasonal_spawn

        events: list[_DisasterEvent] = []

        for _year in range(sim_year_span):
            for season_enum in Season:
                for disaster_type, profile in profiles.items():
                    count_range = profile.event_count_range_by_season.get(season_enum.name.lower(), (0, 0))
                    disaster_count = self._rng.randint(count_range[0], count_range[1])
                    for _ in range(disaster_count):
                        base_time = (self._rng.uniform(0.0, 0.25) + season_enum.value) % 1
                        if base_time < calendar.current_date:
                            base_time += 1
                        event_time = (base_time - calendar.current_date) * 525_600

                        if event_time >= max_sim_minutes:
                            continue

                        size_range = profile.size_range_by_season.get(season_enum.name.lower(), (1, 1))
                        scale = float(self._rng.randint(size_range[0], size_range[1]))
                        x, y = sample_clatsop_local_utm(self._rng)
                        events.append(
                            _DisasterEvent(
                                spawn_time_minutes=event_time,
                                disaster_type=disaster_type,
                                scale_cubic_yards=scale,
                                x_utm=x,
                                y_utm=y,
                            )
                        )

        events.sort(key=lambda e: e.spawn_time_minutes)
        self._disaster_events = events

    # ------------------------------------------------------------------
    # Step 3: Build the JSTRX graph
    # ------------------------------------------------------------------

    def _build_graph(self) -> None:
        """
        Constructs all nodes and links inside the active JSTRXGenerator context.
        Called from within the ``with self._graph:`` block.
        """
        g = CURRENT_GRAPH.get()
        assert g is not None

        # ---- Savevalues ------------------------------------------------
        # Active disaster counter (decremented when a work activity ends)
        g.add_savevalue("_active_disasters", Literal(0.0))
        # Season index (0=winter … 3=fall) updated at each decision point
        g.add_savevalue("_season_idx", Literal(0.0))

        # ---- Resource queues -------------------------------------------
        truck_queue = QueueNode(name="TruckQueue", initialContent=self._n_trucks)
        excavator_queue = QueueNode(name="ExcavatorQueue", initialContent=self._n_excavators)

        # ---- Disaster-available signal queue ---------------------------
        # One token per active disaster.  The dispatch COMBI draws from this
        # to ensure it only fires when at least one disaster exists.
        disaster_avail_queue = QueueNode(name="DisasterAvail", initialContent=0)

        # ---- Observation variable list for DecisionNode ----------------
        # Order MUST match ConStrobeGym._build_obs() field index constants.
        obs_vars = [
            "SimTime",
            "TruckQueue.CurCount",
            "ExcavatorQueue.CurCount",
            "_active_disasters",
            "_season_idx",
        ]
        # Append per-disaster metadata savevalue names (id, type, scale,
        # pct_remaining, age, x, y) for each slot up to max_slots.
        # We use positional savevalues indexed by slot rather than by disaster
        # to keep the obs vector fixed-size regardless of which disasters are
        # currently active.  These are updated by the BEFOREDRAWS callback on
        # the DecisionNode's COMBI.
        for slot_idx in range(self.max_slots):
            for field_name in [
                f"_d{slot_idx}_id",
                f"_d{slot_idx}_type",
                f"_d{slot_idx}_scale",
                f"_d{slot_idx}_pct",
                f"_d{slot_idx}_age",
                f"_d{slot_idx}_x",
                f"_d{slot_idx}_y",
            ]:
                g.add_savevalue(field_name, Literal(0.0))
                obs_vars.append(field_name)

        # ---- DecisionNode (dispatch gate) ------------------------------
        self._decision_node = DecisionNode(
            name="DispatchDecision",
            max_candidates=self.max_slots,
            observation_variable_names=obs_vars,
        )
        decision_combi = self._decision_node.combi

        # The resource token arrives from TruckQueue or ExcavatorQueue.
        # A combined intermediate queue lets either resource type trigger a
        # dispatch decision through the same gate.
        resource_ready_queue = QueueNode(name="ResourceReady", initialContent=0)
        truck_queue.linkTo(decision_combi)
        excavator_queue.linkTo(decision_combi)
        # Disaster availability gate (draws 1 token per dispatch)
        disaster_avail_queue.linkTo(decision_combi)

        # ---- Disaster work activities & return paths -------------------
        # One work COMBI per disaster event (baked at model-build time).
        # The semaphore for slot i → DisasterActivity for the i-th active
        # disaster (in spawn order, wrapping around after max_slots).
        # NOTE: for the initial implementation we cap at max_slots concurrent
        # disasters.  A production version would use a dynamic FORK but that
        # requires ConStrobe features we avoid here.

        disaster_return_queues: list[tuple[QueueNode, QueueNode]] = []
        # One truck-return and excavator-return queue per disaster slot

        for slot_idx in range(self.max_slots):
            sem_queue = self._decision_node.semaphores[slot_idx]

            # Disaster work node (placeholder; duration set per actual event)
            work_combi = CombiNode(name=f"WorkSlot{slot_idx}", duration=0)

            # Return queues feed resources back to the typed pools
            truck_ret = QueueNode(name=f"TruckRet{slot_idx}", initialContent=0)
            excav_ret = QueueNode(name=f"ExcavRet{slot_idx}", initialContent=0)
            disaster_return_queues.append((truck_ret, excav_ret))

            # Entity flow: semaphore → work → return queue
            sem_queue.linkTo(work_combi)
            work_combi.linkTo(truck_ret)

            # Work-end actions: decrement counter, release disaster availability
            # token, return resources to their typed queues.
            work_combi.onEnd(
                AssignAction(
                    "_active_disasters",
                    Var("_active_disasters") - Literal(1.0),
                )
            )

            # Truck return: ADDTOQUEUE TruckQueue
            work_combi.onEnd(AddToQueueAction(truck_queue))
            # Excavator return: ADDTOQUEUE ExcavatorQueue
            work_combi.onEnd(AddToQueueAction(excavator_queue))

        # ---- Disaster spawner nodes ------------------------------------
        # Each event gets a zero-duration NORMAL node whose start is gated by
        # a timer-fired ADDTOQUEUE.  The timer is simulated by an upstream
        # NORMAL node with the event's arrival time as its duration.
        #
        # Topology per event:
        #   TimerQueue_N (initial=0) → TimerActivity_N (dur=spawn_time) →
        #   SpawnQueue_N → [updates _active_disasters, ADDTOQUEUE DisasterAvail]
        #
        # The AAAStartTrigger (built-in) fires at t=0 to kick the timer chain.
        # We chain all timers off of AAAStartTrigger via a shared kick queue.

        kick_queue = QueueNode(name="KickQueue", initialContent=1)

        # Sort events and assign monotonic IDs
        for disaster_id, event in enumerate(self._disaster_events):
            x_norm = min(max(abs(event.x_utm) / _COORD_MAX_MILES, 0.0), 1.0)
            y_norm = min(max(abs(event.y_utm) / _COORD_MAX_MILES, 0.0), 1.0)

            dis_node = DisasterActivityNode(
                disaster_id=disaster_id,
                disaster_type=event.disaster_type,
                scale_cubic_yards=event.scale_cubic_yards,
                work_rate_per_minute=EXCAVATOR_WORK_RATE_CY_MIN,
                x_norm=x_norm,
                y_norm=y_norm,
                name=f"DisSpwn{disaster_id}",
            )
            self._disaster_nodes.append(dis_node)

            # JSTRX topology: QueueNode may only link to CombiNode.
            # Timer chain: KickQueue → TimerCombi (dur=spawn_time) → SpawnCombi (dur=0)
            #
            # TimerCombi acts as the delay (its duration = minutes until spawn).
            # SpawnCombi increments counters and enqueues the disaster token.
            timer_combi = CombiNode(
                name=f"DisTmr{disaster_id}",
                duration=round(event.spawn_time_minutes, 1),
            )
            # Intermediate queue required between two COMBI nodes
            timer_exit_queue = QueueNode(name=f"DisTEQ{disaster_id}", initialContent=0)
            spawn_combi = CombiNode(name=f"DisSpwn{disaster_id}", duration=0)

            # Wire: kick_queue → timer_combi → timer_exit_queue → spawn_combi
            kick_queue.linkTo(timer_combi)
            timer_combi.linkTo(timer_exit_queue)
            timer_exit_queue.linkTo(spawn_combi)

            # SpawnCombi ONSTART actions:
            #   1. Increment active disaster count
            spawn_combi.onStart(
                AssignAction(
                    "_active_disasters",
                    Var("_active_disasters") + Literal(1.0),
                )
            )
            #   2. Add 1 availability token so DispatchDecision can fire
            spawn_combi.onStart(AddToQueueAction(disaster_avail_queue))

            # SpawnCombi output (entity is discarded into a sink — it has served
            # its purpose by updating the counter and availability queue)
            done_sink = QueueNode(name=f"DisDone{disaster_id}", initialContent=0)
            spawn_combi.linkTo(done_sink)

        # ---- ENDCOND ---------------------------------------------------
        # Handled in generate_jstrx via the default "ENDCOND SimTime>=600000;"
        # We override via the generator's _code_block for a compound condition.
        # If ConStrobe supports OR in ENDCOND we use it; otherwise we rely
        # solely on the max-time condition and the gym detects success via
        # _active_disasters == 0 check in each observation.
        g.setCode(f"ENDCOND SimTime>={MAX_SIM_TIME};")

    # ------------------------------------------------------------------
    # Helper: normalise UTM coordinate to [0, 1]
    # ------------------------------------------------------------------

    @staticmethod
    def _norm_coord(utm_metres: float) -> float:
        return min(max(abs(utm_metres) / _COORD_MAX_MILES, 0.0), 1.0)

    # ------------------------------------------------------------------
    # Observation index constants (mirrors gym.py constants)
    # ------------------------------------------------------------------

    OBS_IDX_SIM_TIME: int = 0
    OBS_IDX_TRUCK_COUNT: int = 1
    OBS_IDX_EXCAV_COUNT: int = 2
    OBS_IDX_ACTIVE_DISASTERS: int = 3
    OBS_IDX_SEASON: int = 4
    OBS_IDX_DISASTER_FEATURES_START: int = 5
    DISASTER_FIELDS_PER_SLOT: int = 7  # id, type, scale, pct, age, x, y

    @classmethod
    def disaster_obs_index(cls, slot: int, field: int) -> int:
        """
        Return the flat index into the ObsPayload.values list for a given
        (slot, field) pair in the per-disaster section.

        Fields are: 0=id, 1=type_oh, 2=scale, 3=pct_remaining,
                    4=age_norm, 5=x_norm, 6=y_norm
        """
        return cls.OBS_IDX_DISASTER_FEATURES_START + slot * cls.DISASTER_FIELDS_PER_SLOT + field
