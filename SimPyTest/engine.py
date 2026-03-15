from __future__ import annotations
from SimPyTest.visualization import EngineVisualizer

import random
from collections.abc import Generator
from typing import TYPE_CHECKING, Callable, Literal, TypeVar

import networkx as nx
import simpy
from simpy.core import EmptySchedule
from simpy.events import Process, Timeout

from SimPyTest.disaster_keys import disaster_key_for_instance_name
from SimPyTest.scenario_types import ScenarioConfig
from .calendar import SimulationCalendar, add_seasonal_disasters
from .engine_world import generate_disaster_locations, get_distance, initialize_world
from .metrics_tracker import SimulationMetrics
from .simulation import Disaster, DisasterStore, DumpSite, IdleResources, Resource, ResourceNode, ResourceType

if TYPE_CHECKING:
    from .gis_utils import GISConfig
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

    MAX_SIM_TIME: int = 60_000
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
        visualizer: EngineVisualizer | None = None,
    ):
        self.policy: Policy = policy
        self.seed: int = seed
        self.rng: SimulationRNG = SimulationRNG(seed)
        self.decision_rng: SimulationRNG = SimulationRNG(seed + 99999)

        self.scenario_config: ScenarioConfig = scenario_config

        # SimPy environment and domain objects
        self.env: simpy.Environment = simpy.Environment()
        self.idle_resources: IdleResources = IdleResources(self)
        self.disaster_store: DisasterStore = DisasterStore(self.env)
        self.resource_nodes: list[ResourceNode] = []

        # GIS support
        self.road_graph: nx.Graph[tuple[float, float]] | None = None
        self.gis_config: GISConfig | None = scenario_config.gis_config
        self.calendar: SimulationCalendar = SimulationCalendar(
            start_date=scenario_config.calendar_start_date,
            duration_years=scenario_config.calendar_duration_years,
            seed=seed,
        )
        self.metrics: SimulationMetrics = SimulationMetrics()
        self.total_spent: float = 0.0
        self.non_idle_time: float = 0.0
        self.total_drive_time: float = 0.0
        self.tournament_decisions: list[tuple[float, str]] = []
        self.total_dispatch_delay: float = 0.0
        self.dispatch_delay_events: int = 0
        self.max_dispatch_delay: float = 0.0
        self.decision_log: list[int] = []
        self.replay_buffer: list[int] = []
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
            if self.env.now - self._last_visualizer_update_time < 20.0:
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
        if current_disasters > 0 or self._prev_disaster_count > 0:
            self.non_idle_time += elapsed
        self._prev_disaster_count = current_disasters

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

            actionable = [disaster for disaster in self.disaster_store.items if resource.resource_type in disaster.needed_resources()]
            if not actionable:
                resource.assigned_node = self.idle_resources
                self.idle_resources.mark_resource_available(resource)
                continue

            if self.replay_buffer:
                target_id = self.replay_buffer.pop(0)
                candidates = [d for d in actionable if d.id == target_id]
                if not candidates:
                    resource.assigned_node = self.idle_resources
                    self.idle_resources.mark_resource_available(resource)
                    continue
                target_disaster = candidates[0]
            else:
                if len(actionable) > 1 and self.stop_before_policy_decision:
                    self.pending_decision_resource = resource
                    return None

                target_disaster = actionable[0] if len(actionable) == 1 else self.policy.func(resource, actionable, self.env)
                if target_disaster is None:
                    resource.assigned_node = self.idle_resources
                    self.idle_resources.mark_resource_available(resource)
                    continue
                if self.branch_decision is None:
                    self.branch_decision = target_disaster.id
                self.decision_log.append(target_disaster.id)
                self.decisions_made += 1

            dispatch_delay = self._consume_dispatch_delay_minutes(target_disaster)
            if dispatch_delay > 0:
                self.metrics.record_dispatch_delay(target_disaster.id, dispatch_delay)
                self.total_dispatch_delay += dispatch_delay
                self.dispatch_delay_events += 1
                self.max_dispatch_delay = max(self.max_dispatch_delay, dispatch_delay)
                yield self.env.timeout(dispatch_delay)
                if not target_disaster.active or target_disaster not in self.disaster_store.items:
                    resource.assigned_node = self.idle_resources
                    self.idle_resources.mark_resource_available(resource)
                    continue

            target_disaster.transfer_resource(resource)
            if not self.replay_buffer and self.max_decisions is not None and self.decisions_made >= self.max_decisions:
                return len(self.disaster_store.items) == 0

    def _consume_dispatch_delay_minutes(self, disaster: Disaster) -> float:
        if disaster.dispatch_delay_applied:
            return 0.0
        disaster.dispatch_delay_applied = True
        return self._sample_dispatch_delay_minutes(disaster)

    def _sample_dispatch_delay_minutes(self, disaster: Disaster) -> float:
        priors = self.scenario_config.operational_priors.disaster_operational_priors
        disaster_key = disaster_key_for_instance_name(type(disaster).__name__) or ""
        sampled = priors.get(disaster_key, {}).get("dispatch_delay_minutes_range")
        low, high = (15.0, 90.0)
        if isinstance(sampled, tuple) and len(sampled) == 2:
            low, high = float(sampled[0]), float(sampled[1])
        if high <= 0 or high <= low:
            return 0.0

        delay = self.rng.uniform(low, high)
        if self.scenario_config.weather_model.enable_dispatch_scaling:
            weather_key = disaster_key_for_instance_name(type(disaster).__name__)
            if weather_key is not None:
                weather_factor = self.calendar.get_weather_factor(weather_key)
                multiplier = self.scenario_config.weather_model.storm_dispatch_delay_multiplier
                delay *= 1.0 + (multiplier - 1.0) * max(0.0, min(1.0, weather_factor))
        return float(delay)

    # ----------------------------------------------------------------------------
    # MARK: Decision Loop
    # ----------------------------------------------------------------------------

    def get_distance(self, r1: Resource, r2: ResourceNode) -> float:
        return get_distance(self, r1, r2)

    def generate_disaster_locations(self, num_locations: int) -> list[tuple[float, float]]:
        return generate_disaster_locations(self, num_locations)

    def add_disasters(self) -> Generator[simpy.Event, object, None]:
        dump_site = next(d for d in self.resource_nodes if isinstance(d, DumpSite))
        yield from add_seasonal_disasters(self, dump_site)

    def initialize_world(self):
        initialize_world(self)

    def record_disaster_created_metrics(self, disaster: Disaster) -> None:
        self.metrics.record_disaster_created(
            disaster_id=disaster.id,
            disaster_type=type(disaster).__name__,
            sim_time=self.env.now,
            road_miles=None,
            population_affected=None,
            road_class="secondary",
            aadt=None,
            truck_pct=None,
            detour_penalty=1.0 * disaster.vulnerability_index,
        )
