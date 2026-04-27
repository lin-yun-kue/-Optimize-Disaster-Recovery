from __future__ import annotations
import math
from collections import deque
from SimPyTest.evaluation import build_simulation_summary

import random
from collections.abc import Generator
from typing import TYPE_CHECKING, Callable, Deque, Literal, TypeVar

import simpy
from simpy.core import EmptySchedule
from simpy.events import Process, Timeout

from SimPyTest.scenario_types import ScenarioConfig
from .calendar import SimulationCalendar, add_seasonal_disasters
from .metrics_tracker import SimulationMetrics
from .simulation import Disaster, DisasterStore, IdleResources, Resource, ResourceNode, Depot, DumpSite, ResourceType
from SimPyTest.gis_utils import DEPOTS, LANDFILLS, sample_clatsop_local_utm, meters_to_miles
from SimPyTest.gis_utils import CRS_UTM, get_road_distance

if TYPE_CHECKING:
    from .gis_utils import GISConfig
    import networkx as nx
    from .policies import Policy
    from .visualization import EngineVisualizer


T = TypeVar("T")


class SimulationRNG:
    """Custom RNG that can be cloned with exact state preservation (wraps random.Random)."""

    seed: int
    _rng: random.Random

    def __init__(self, seed: int = 0):
        self.seed = seed
        self._rng = random.Random(seed)

    def random(self) -> float:
        return self._rng.random()

    def randint(self, a: int, b: int) -> int:
        return self._rng.randint(a, b)

    def uniform(self, a: float, b: float) -> float:
        return self._rng.uniform(a, b)

    def lognormvariate(self, mu: float, sigma: float) -> float:
        return self._rng.lognormvariate(mu, sigma)

    def expovariate(self, lambd: float) -> float:
        return self._rng.expovariate(lambd)

    def choice(self, seq: list[T]) -> T:
        return self._rng.choice(seq)

    def shuffle(self, seq: list[T]) -> None:
        self._rng.shuffle(seq)

    def setstate(self, state: tuple[int, int, int, int, int, int]):
        self._rng.setstate(state)

    def getstate(self):
        return self._rng.getstate()

    def reseed(self, seed: int):
        self.seed = seed
        self._rng = random.Random(seed)

    def clone(self) -> "SimulationRNG":
        new = SimulationRNG(self.seed)
        new._rng.setstate(self._rng.getstate())
        return new


# ============================================================================
# MARK: Engine
# ============================================================================
class SimPySimulationEngine:
    """
    Simulation engine wrapper for your disaster-resource SimPy simulation.
    """

    MAX_SIM_TIME: int = 600_000
    TERMINAL_SUCCESS: Literal["SUCCESS"] = "SUCCESS"
    TERMINAL_FAIL_DEADLOCK: Literal["FAIL_DEADLOCK"] = "FAIL_DEADLOCK"
    TERMINAL_FAIL_TIMEOUT: Literal["FAIL_TIMEOUT"] = "FAIL_TIMEOUT"
    TERMINAL_FAIL_INVALID_STATE: Literal["FAIL_INVALID_STATE"] = "FAIL_INVALID_STATE"

    def __init__(
        self,
        *,
        policy: Policy,
        scenario_config: ScenarioConfig,
        seed: int = 0,
        track_metrics: bool = True,
        visualizer: EngineVisualizer | None = None,
    ):
        self.policy: Policy = policy
        self.seed: int = seed
        self.rng: SimulationRNG = SimulationRNG(seed)
        self.track_metrics: bool = track_metrics

        self.scenario_config: ScenarioConfig = scenario_config

        # SimPy environment and domain objects
        self.env: simpy.Environment = simpy.Environment()
        self.idle_resources: IdleResources = IdleResources(self)
        self.disaster_store: DisasterStore = DisasterStore(self.env)
        self.resource_nodes: list[ResourceNode] = []

        # GIS support
        self.road_graph: "nx.Graph[tuple[float, float]] | None" = None
        self.gis_config: GISConfig | None = scenario_config.gis_config
        self.calendar: SimulationCalendar = SimulationCalendar(start_date=scenario_config.calendar_start_date, duration_years=scenario_config.calendar_duration_years)
        self.metrics: SimulationMetrics = SimulationMetrics()
        self.total_spent: float = 0.0
        self.time_with_disasters: float = 0.0
        self.total_drive_time: float = 0.0
        self.tournament_decisions: list[tuple[float, str]] = []
        self.decision_log: list[int] = []
        self.replay_buffer: Deque[int] = deque()
        self.branch_decision: int | None = None
        self.decisions_made: int = 0
        self.max_decisions: int | None = None
        self.stop_before_policy_decision: bool = False
        self.pending_decision_resource: Resource | None = None
        self._prev_disaster_count: int = 0
        self._last_calendar_update: float = 0.0
        self._last_visualizer_update_time: float | None = None
        self.disasters_process: simpy.Process | None = None
        self._main_loop_process: simpy.Process | None = None
        self.last_terminal_outcome: str | None = None
        self.last_terminal_error: str | None = None

        self.visualizer: EngineVisualizer | None = visualizer

    # ----------------------------------------------------------------------------
    # MARK: Run Control
    # ----------------------------------------------------------------------------
    def run(self, max_decisions: int | None = None) -> bool:
        self.max_decisions = max_decisions
        self.decisions_made = 0
        self.disasters_process = self.env.process(self.add_disasters())
        self._main_loop_process = self.env.process(self.loop())
        self._initialize_runtime_tracking()

        while True:
            try:
                self.advance_to_next_event()
            except EmptySchedule:
                self.last_terminal_outcome = self.infer_terminal_outcome(schedule_exhausted=True)
                break
            except Exception as e:
                self.last_terminal_error = repr(e)
                self.last_terminal_outcome = self.TERMINAL_FAIL_INVALID_STATE
                print(f"   [!] Policy {self.policy.name} failed at {self.env.now}: {e}")
                break

            if self.last_terminal_outcome is not None:
                break

        if self.visualizer is not None:
            self._update_visualizer(force=True)
            self.visualizer.close()

        return self.last_terminal_outcome == self.TERMINAL_SUCCESS

    def run_in_gym(self, gym_loop: Callable[[], Generator[simpy.Event, Resource, None]]):
        """Run the simulation in a Gym environment."""

        self.disasters_process = self.env.process(self.add_disasters())
        self._main_loop_process = self.env.process(gym_loop())
        self._initialize_runtime_tracking()

    def _initialize_runtime_tracking(self) -> None:
        self._prev_disaster_count = len(self.disaster_store.items)
        self._last_calendar_update = self.env.now
        self.last_terminal_outcome = None
        self.last_terminal_error = None
        self.pending_decision_resource = None
        self._last_visualizer_update_time = None

    def _update_visualizer(self, force: bool = False) -> None:
        if self.visualizer is None:
            return
        if not force and self._last_visualizer_update_time is not None:
            if self.env.now - self._last_visualizer_update_time < 100.0:
                return
        self.visualizer.update(self.visualizer.snapshot())
        self._last_visualizer_update_time = self.env.now

    def advance_to_next_event(self) -> None:
        previous_time = self.env.now
        self.env.step()
        elapsed = max(0.0, self.env.now - previous_time)

        if elapsed > 0:
            self.calendar.advance_time_minutes(elapsed)
            self._last_calendar_update = self.env.now

        current_disasters = len(self.disaster_store.items)
        disaster_count_changed = current_disasters != self._prev_disaster_count
        if self._prev_disaster_count > 0:
            self.time_with_disasters += elapsed
        self._prev_disaster_count = current_disasters

        if self.visualizer is not None:
            self._update_visualizer(force=disaster_count_changed)

        terminal_outcome = self.infer_terminal_outcome(schedule_exhausted=False)
        if terminal_outcome is not None:
            self.last_terminal_outcome = terminal_outcome

    def infer_terminal_outcome(self, schedule_exhausted: bool) -> str | None:
        if self.env.now > self.MAX_SIM_TIME:
            return self.TERMINAL_FAIL_TIMEOUT

        if self.disasters_process is not None and self.disasters_process.triggered and len(self.disaster_store.items) == 0:
            return self.TERMINAL_SUCCESS

        if not schedule_exhausted:
            return None

        if len(self.disaster_store.items) > 0:
            return self.TERMINAL_FAIL_DEADLOCK

        return self.TERMINAL_FAIL_INVALID_STATE

    # ----------------------------------------------------------------------------
    # MARK: Simulation Processes
    # ----------------------------------------------------------------------------

    def loop(self) -> Generator[Process | Timeout, Resource, bool | None]:
        """
        Waits for ANY resource to become available anywhere,
        then asks the policy where to send it.
        """
        while True:
            yield self.env.process(self.disaster_store.wait_for_any())
            resource = yield self.env.process(self.idle_resources.get_resource_for_disasters(list(self.disaster_store.items)))

            if not self.disaster_store.items:
                resource.assigned_node = self.idle_resources
                self.idle_resources.mark_resource_available(resource)
                continue

            actionable = [disaster for disaster in self.disaster_store.items if resource.resource_type in disaster.required_resources]
            if not actionable:
                resource.assigned_node = self.idle_resources
                self.idle_resources.mark_resource_available(resource)
                continue

            if self.replay_buffer:
                target_id = self.replay_buffer.popleft()
                candidates = [d for d in actionable if d.id == target_id]
                if not candidates:
                    resource.assigned_node = self.idle_resources
                    self.idle_resources.mark_resource_available(resource)
                    raise RuntimeError("Replay buffer contains invalid disaster ID")
                target_disaster = candidates[0]
            else:
                if len(actionable) > 1 and self.stop_before_policy_decision:
                    self.pending_decision_resource = resource
                    return None

                target_disaster = self.policy.func(resource, actionable, self.env)
                if target_disaster is None:
                    resource.assigned_node = self.idle_resources
                    self.idle_resources.mark_resource_available(resource)
                    continue
                if self.branch_decision is None:
                    self.branch_decision = target_disaster.id
                self.decision_log.append(target_disaster.id)
                self.decisions_made += 1

            target_disaster.transfer_resource(resource)
            if not self.replay_buffer and self.max_decisions is not None and self.decisions_made >= self.max_decisions:
                return len(self.disaster_store.items) == 0

    # ----------------------------------------------------------------------------
    # MARK: Decision Loop
    # ----------------------------------------------------------------------------

    def get_distance(self, r1: Resource, r2: ResourceNode) -> float:
        if self.road_graph is not None:
            dist_meters = get_road_distance(self.road_graph, r1.location, r2.location, self.gis_config)
            dist = meters_to_miles(dist_meters)
            if dist != float("inf"):
                return dist
        euclidean_units_meters = math.hypot(r1.location[0] - r2.location[0], r1.location[1] - r2.location[1])
        return meters_to_miles(euclidean_units_meters)

    def get_nearest_node(self, resource: Resource, node_type: type[ResourceNode]) -> ResourceNode | None:
        nodes = [node for node in self.resource_nodes if isinstance(node, node_type)]
        if not nodes:
            return None

        return min(nodes, key=lambda node: self.get_distance(resource, node))

    def generate_disaster_locations(self, num_locations: int) -> list[tuple[float, float]]:
        locations: list[tuple[float, float]] = []
        if self.gis_config is not None and self.gis_config.roads_gdf is not None and self.road_graph is not None:
            sampled = self.gis_config.roads_gdf.sample(n=num_locations, random_state=self.seed)
            sampled = sampled.to_crs(CRS_UTM)
            centroids = sampled.geometry.centroid
            spatial_index = self.gis_config.get_spatial_index()
            for centroid in centroids:
                x, y = centroid.xy
                locations.append(spatial_index.get_nearest_node(x[0], y[0]))
            return locations

        return [sample_clatsop_local_utm(self.rng) for _ in range(num_locations)]

    def add_disasters(self) -> Generator[simpy.Event, object, None]:
        yield from add_seasonal_disasters(self)

    def initialize_world(self):
        if self.gis_config is not None:
            self.road_graph = self.gis_config.load_road_network()
        self.init_nodes()
        self.spawn_resources()

    def init_nodes(self):
        depots = DEPOTS if self.gis_config is None else self.gis_config.depots
        landfills = LANDFILLS if self.gis_config is None else self.gis_config.landfills

        for depot in depots:
            self.resource_nodes.append(Depot(self, (depot["Longitude"], depot["Latitude"]), label=depot["Name"]))
        for landfill in landfills:
            self.resource_nodes.append(DumpSite(self, (landfill["Longitude"], landfill["Latitude"]), label=landfill["Name"]))

    def spawn_resources(self) -> None:
        spawn_plan = [
            (ResourceType.TRUCK, self.scenario_config.resolve_resource_count(self.rng, ResourceType.TRUCK)),
            (ResourceType.EXCAVATOR, self.scenario_config.resolve_resource_count(self.rng, ResourceType.EXCAVATOR)),
        ]
        resource_id = 0
        engine_depots = [depot for depot in self.resource_nodes if isinstance(depot, Depot)]

        for resource_type, count in spawn_plan:
            for _ in range(count):
                depot = self.rng.choice(engine_depots)
                resource = Resource(resource_id, resource_type, self)
                resource.home_depot = depot
                depot.transfer_resource(resource, True)
                resource_id += 1

    def summary(self):
        return build_simulation_summary(self)

    def record_disaster_created_metrics(self, disaster: Disaster) -> None:
        if not self.track_metrics:
            return
        self.metrics.record_disaster_created(
            disaster_id=disaster.id,
            disaster_type=type(disaster).__name__,
            sim_time=self.env.now,
            road_miles=None,
            population_affected=None,
            road_class="secondary",
            aadt=None,
            truck_pct=None,
            detour_penalty=1.0,
        )
