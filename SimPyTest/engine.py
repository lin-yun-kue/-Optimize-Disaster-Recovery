from __future__ import annotations
from simpy.events import Process
import simpy
import random
from typing import TypeVar, TYPE_CHECKING
from datetime import datetime
from dataclasses import dataclass, field
import math

try:
    from typing import Never
except ImportError:
    # Python < 3.10 compatibility
    from typing import NoReturn as Never
from collections.abc import Generator
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from math import cos, sin
from simpy.core import EmptySchedule

from SimPyTest.gis_utils import CRS_UTM, get_road_distance
from SimPyTest.multi_year_config import SiteCapacity
from SimPyTest.real_world_params import get_default_disaster_operational_priors, meters_to_miles
from .simulation import *
import networkx as nx
from shapely.geometry import Point

if TYPE_CHECKING:
    from .policies import Policy
    from .gis_utils import GISConfig


# ============================================================================
# MARK: RNG Wrapper
# ============================================================================


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


@dataclass
class ScenarioConfig:
    # Resource counts (can be a range for randomization)
    # Scaled for realistic speeds (truck: 23.4 mph, excavator: 11.6 mph)
    num_trucks: int | tuple[int, int] = (15, 25)
    num_excavators: int | tuple[int, int] = (10, 15)
    num_snowplows: int | tuple[int, int] = (0, 5)
    num_assessment_vehicles: int | tuple[int, int] = 0

    # Disaster counts - more landslides to create more decision points
    num_landslides: int | tuple[int, int] = (15, 25)
    landslide_size_range: tuple[int, int] = (200, 400)

    # Distance range for landslide spawning (in same units as speeds)
    landslide_distance_range: tuple[int, int] = (1800, 2200)

    # Calendar/Seasonality options (None = disabled/simple mode)
    calendar_start_date: datetime | None = None  # e.g., datetime(2024, 1, 1)
    calendar_duration_years: int = 1
    use_seasonal_disasters: bool = False  # Use OREGON_DISASTER_PATTERNS for spawning
    use_weather_modifier: bool = False  # Scale disaster size by weather
    seasonal_size_bucket_weights: dict[str, dict[str, float]] = field(default_factory=dict)
    disaster_closure_lognormal_priors: dict[str, dict[str, tuple[float, float]]] = field(
        default_factory=lambda: {
            "landslide": {
                "summer": (1.386, 1.715),  # median ~4h
                "default": (2.079, 1.715),  # median ~8h
            },
            "snow": {"default": (0.693, 1.398)},  # median ~2h, p90 ~12h
            "flood": {
                "summer": (2.890, 1.743),  # median ~18h
                "default": (3.584, 1.743),  # median ~36h
            },
            "wildfire_debris": {"default": (1.386, 1.939)},  # median ~4h
        }
    )
    snow_road_miles_range: tuple[float, float] = (2.0, 10.0)
    flood_cleanup_probability: float = 0.7
    flood_cleanup_amount_range: tuple[int, int] = (20, 200)
    flood_assessment_minutes_range: tuple[float, float] = (30.0, 90.0)
    flood_status_check_interval_minutes: float = 60.0

    # Budget/Cost options
    annual_budget: float = 10_000_000  # $10M default
    track_costs: bool = True  # Track resource operating costs
    time_variance: float = 0.0  # Stochastic variance (0.0 = deterministic)

    # Dispatch delay realism (sim time is minutes)
    use_dispatch_delay_priors: bool = False
    dispatch_delay_minutes_range: tuple[float, float] = (0.0, 0.0)
    storm_dispatch_delay_multiplier: float = 2.0

    # Time-model realism (sim time unit is real-world minutes)
    calendar_minutes_per_sim_minute: float = 1.0
    non_gis_distance_unit_miles: float = 0.01
    non_seasonal_spawn_interval_minutes_range: tuple[float, float] = (5.0, 60.0)
    seasonal_spawn_interval_minutes_range: tuple[float, float] = (60.0, 1440.0)
    snow_auto_resolve_multiplier: float = 3.0
    snow_work_interval_minutes: float = 60.0
    disaster_operational_priors: dict[str, dict[str, object]] = field(
        default_factory=get_default_disaster_operational_priors
    )

    # Optional callback for manual placement: (engine) -> None
    custom_setup_fn: Callable[[SimPySimulationEngine], None] | None = None

    # Optional GIS configuration
    gis_config: GISConfig | None = None

    def get_site_capacity(self, site_type: str) -> SiteCapacity:
        """Get capacity configuration for a site type."""
        return SiteCapacity(max_concurrent=10, efficiency_curve=lambda x: min(1.0, x / 10))


# ============================================================================
# MARK: Engine
# ============================================================================
class SimPySimulationEngine:
    """
    Simulation engine wrapper for your disaster-resource SimPy simulation.
    """

    MAX_SIM_TIME: int = 600_000

    def __init__(
        self,
        policy: Policy,
        seed: int = 0,
        live_plot: bool = False,
        scenario_config: ScenarioConfig | None = None,
    ):
        self.policy: Policy = policy
        self.seed: int = seed
        self.rng: SimulationRNG = SimulationRNG(seed)
        self.decision_rng: SimulationRNG = SimulationRNG(seed + 99999)
        self.live_plot: bool = live_plot

        if scenario_config is None:
            self.scenario_config: ScenarioConfig = ScenarioConfig()
        else:
            self.scenario_config = scenario_config

        # SimPy environment and domain objects
        self.env: simpy.Environment = simpy.Environment()
        self.idle_resources: IdleResources = IdleResources(self)
        self.disaster_store: DisasterStore = DisasterStore(self.env)
        # self.resources: List[Resource] = []
        self.resource_nodes: list[ResourceNode] = []

        # GIS support
        self.road_graph: nx.Graph[tuple[float, float]] | None = None
        self.gis_config: GISConfig | None = scenario_config.gis_config if scenario_config else None

        # Calendar/Seasonality support
        self.calendar: SimulationCalendar | None = None
        if scenario_config is not None and scenario_config.calendar_start_date is not None:
            from .calendar import SimulationCalendar

            self.calendar = SimulationCalendar(
                start_date=scenario_config.calendar_start_date,
                duration_years=scenario_config.calendar_duration_years,
            )

        # Budget tracking
        self.total_spent: float = 0.0
        self.budget_exhausted: bool = False

        # Metrics tracking
        from .metrics_tracker import SimulationMetrics

        self.metrics = SimulationMetrics()

        # result tracking
        self._time_points: list[float] = []
        self._known_disasters: dict[int, Disaster] = {}
        self._disaster_histories: dict[int, list[float]] = {}
        self.non_idle_time: float = 0.0
        self.tournament_decisions: list[tuple[float, str]] = []
        self.total_dispatch_delay: float = 0.0
        self.dispatch_delay_events: int = 0
        self.max_dispatch_delay: float = 0.0

        # replay
        self.decision_log: list[int] = []
        self.replay_buffer: list[int] = []
        self.branch_decision: int | None = None

        # Decision limiting for tree search tournament
        self.decisions_made: int = 0
        self.max_decisions: int | None = None
        self._prev_disaster_count: int = 0

        # whether the scheduled "add_disasters" is present and its process reference
        self.disasters_process: simpy.Process | None = None
        self._main_loop_process: simpy.Process | None = None

        self.fig: Figure | None = None
        self.axs: list[Axes] | None = None
        self._gis_edge_segments: list[tuple[tuple[float, float], tuple[float, float]]] | None = None
        self._gis_bounds: tuple[float, float, float, float] | None = None

        # print(f"Running {self.policy.name} with seed {self.seed}.")

    # ----------------------------------------------------------------------------
    # MARK: Run Control
    # ----------------------------------------------------------------------------
    def run(self, max_decisions: int | None = None):
        """
        Run to completion (EmptySchedule).

        Args:
            max_decisions: If set, stop after N decisions are made (for tree search).
        """
        self.max_decisions = max_decisions
        self.decisions_made = 0

        if self.live_plot and self.fig is None:
            self.fig, self.axs = self.setup_plot()

        self.disasters_process = self.env.process(self.add_disasters())
        self._main_loop_process = self.env.process(self.loop())

        simulation_succeeded = False
        last_idle_time = self.env.now
        self._prev_disaster_count = 0  # Track previous disaster count for proper time tracking
        last_calendar_update = self.env.now  # Track when we last updated the calendar

        while True:
            try:
                target_time = self.env.now + 1
                while self.env.now < target_time:
                    self.env.step()

                    # Advance calendar with simulation time (real-world minute mapping).
                    if self.calendar is not None and self.env.now - last_calendar_update >= 1.0:
                        minutes_passed = self.env.now - last_calendar_update
                        self.calendar.advance_time_minutes(
                            minutes_passed * self.scenario_config.calendar_minutes_per_sim_minute
                        )
                        last_calendar_update = self.env.now
            except EmptySchedule:
                # CASE 1: The schedule is empty.
                # Did we finish?
                if self.disasters_process and self.disasters_process.triggered and len(self.disaster_store.items) == 0:
                    simulation_succeeded = True
                else:
                    # We ran out of events but disasters remain -> FAILURE
                    simulation_succeeded = False
                break
            except Exception as e:
                print(f"   [!] Exception: {e}")
                simulation_succeeded = False
                print(f"   [!] Policy {self.policy.name} failed at {self.env.now}.")

                raise e

            # break if max time reached
            if self.env.now > self.MAX_SIM_TIME:
                simulation_succeeded = False
                print(f"   [!] Policy {self.policy.name} timed out at {self.env.now}.")
                break

            # collect idle time
            # FIX: Track previous disaster count to properly count time when disasters go from 1->0
            if not hasattr(self, "_prev_disaster_count"):
                self._prev_disaster_count = len(self.disaster_store.items)

            curr_disasters = len(self.disaster_store.items)
            if curr_disasters > 0 or self._prev_disaster_count > 0:
                self.non_idle_time += self.env.now - last_idle_time
            self._prev_disaster_count = curr_disasters
            last_idle_time = self.env.now

            # detect new disasters
            for ls in list(self.disaster_store.items):
                if ls.id not in self._known_disasters:
                    self._known_disasters[ls.id] = ls
                    self._disaster_histories[ls.id] = [0] * len(self._time_points)

            self._time_points.append(self.env.now)
            for ls_id, ls_obj in self._known_disasters.items():
                if isinstance(ls_obj, Landslide):
                    val = ls_obj.dirt.level
                else:
                    val = 0
                self._disaster_histories[ls_id].append(val)

            if self.live_plot and self.axs is not None:
                self.update_plot()

            # success condition
            if self.disasters_process.triggered and len(self.disaster_store.items) == 0:
                simulation_succeeded = True
                break

        if self.live_plot:
            plt.close(self.fig)

        return simulation_succeeded

    # ----------------------------------------------------------------------------
    # MARK: Run in Gym Environment
    # ----------------------------------------------------------------------------

    def run_in_gym(self, gym_loop: Callable[[], Generator[simpy.Event, Resource, Never]]):
        """Run the simulation in a Gym environment."""

        self.disasters_process = self.env.process(self.add_disasters())
        self._main_loop_process = self.env.process(gym_loop())

    # ----------------------------------------------------------------------------
    # MARK: Simulation Processes
    # ----------------------------------------------------------------------------

    def loop(self) -> Generator[Process, Resource, bool | None]:
        """
        The main orchestrator. It waits for ANY resource to become available
        at the depot, then asks the policy where to send it.
        """
        while True:
            resource = yield self.env.process(self.idle_resources.get_any_resource())
            yield self.env.process(self.disaster_store.wait_for_any())

            target_disaster = None

            # A disaster may resolve between wake-up and policy selection.
            # In that case keep the resource idle and wait for the next event.
            if len(self.disaster_store.items) == 0:
                resource.assigned_node = self.idle_resources
                self.idle_resources.mark_resource_available(resource)
                continue

            if len(self.replay_buffer) > 0:
                # Force the decision from history
                target_id = self.replay_buffer.pop(0)

                # Find the disaster object with this ID
                candidates = [d for d in self.disaster_store.items if d.id == target_id]

                # If disaster was already resolved (not in store), skip this decision
                if not candidates:
                    resource.assigned_node = self.idle_resources
                    self.idle_resources.mark_resource_available(resource)
                    continue

                target_disaster = candidates[0]
            else:
                # Check if budget is exhausted before asking policy
                if self.scenario_config.track_costs and self.total_spent >= self.scenario_config.annual_budget:
                    self.budget_exhausted = True

                # Ask the policy
                if len(self.disaster_store.items) == 1:
                    target_disaster = self.disaster_store.items[0]
                else:
                    target_disaster = self.policy.func(resource, self.disaster_store.items, self.env)

                # If policy returns None (e.g., budget exhausted), skip this dispatch
                if target_disaster is None:
                    resource.assigned_node = self.idle_resources
                    self.idle_resources.mark_resource_available(resource)
                    continue

                # If this was the VERY FIRST move after replay buffer emptied, capture it
                if self.branch_decision is None:
                    self.branch_decision = target_disaster.id

                # Record this decision for future replays
                self.decision_log.append(target_disaster.id)

                # Increment decision counter
                self.decisions_made += 1

            dispatch_delay = self._sample_dispatch_delay_minutes(target_disaster)
            if dispatch_delay > 0:
                self.total_dispatch_delay += dispatch_delay
                self.dispatch_delay_events += 1
                self.max_dispatch_delay = max(self.max_dispatch_delay, dispatch_delay)
                yield self.env.timeout(dispatch_delay)

                # Target may resolve during delay (especially short snow/flood cases)
                if (not getattr(target_disaster, "active", True)) or (target_disaster not in self.disaster_store.items):
                    resource.assigned_node = self.idle_resources
                    self.idle_resources.mark_resource_available(resource)
                    continue

            # Execute the chosen dispatch before enforcing the decision cutoff.
            # This keeps replay histories consistent with the simulated state.
            target_disaster.transfer_resource(resource)

            # Check if we've reached the decision limit (for tree search)
            if len(self.replay_buffer) == 0 and self.max_decisions is not None and self.decisions_made >= self.max_decisions:
                # Stop simulation here - return current state after the dispatch was applied.
                # (the caller will check disaster_store.items for partial state)
                return len(self.disaster_store.items) == 0

    def _sample_dispatch_delay_minutes(self, disaster: Disaster) -> float:
        """Sample a dispatch delay prior in simulation minutes."""
        cfg = self.scenario_config

        if cfg.use_dispatch_delay_priors:
            priors = cfg.disaster_operational_priors
            key_by_type = {
                "Landslide": "landslide",
                "WildfireDebris": "wildfire_debris",
                "FloodEvent": "flood",
                "SnowEvent": "snow",
            }
            pri = priors.get(key_by_type.get(type(disaster).__name__, ""), {})
            sampled = pri.get("dispatch_delay_minutes_range")
            if isinstance(sampled, tuple) and len(sampled) == 2:
                low, high = float(sampled[0]), float(sampled[1])
            else:
                low, high = (15.0, 90.0)
        else:
            low, high = cfg.dispatch_delay_minutes_range

        if high <= 0 or high <= low:
            return 0.0

        delay = self.rng.uniform(low, high)

        if self.calendar is not None:
            # Weather-driven backlog proxy: adverse weather increases dispatch latency.
            type_name = type(disaster).__name__
            weather_key = {
                "Landslide": "landslide",
                "WildfireDebris": "wildfire_debris",
                "FloodEvent": "flood",
                "SnowEvent": "snow",
            }.get(type_name)
            if weather_key is not None:
                weather_factor = self.calendar.get_weather_factor(weather_key)
                # Scale between 1x and storm_dispatch_delay_multiplier x
                delay *= 1.0 + (cfg.storm_dispatch_delay_multiplier - 1.0) * max(0.0, min(1.0, weather_factor))

        return float(delay)

    # ----------------------------------------------------------------------------
    # MARK: Decision Loop
    # ----------------------------------------------------------------------------

    def get_distance(self, r1: Resource, r2: ResourceNode) -> float:
        """Calculate distance in miles between a resource and a resource node."""

        if self.road_graph is not None:
            dist_meters = get_road_distance(self.road_graph, r1.location, r2.location, self.gis_config)
            dist = meters_to_miles(dist_meters)
            if dist == float("inf"):
                print(f"No path from {r1.location} to {r2.location}")
                # Fallback to Euclidean if no path found
                euclidean_units = math.hypot(r1.location[0] - r2.location[0], r1.location[1] - r2.location[1])
                return euclidean_units * self.scenario_config.non_gis_distance_unit_miles
            return dist
        else:
            euclidean_units = math.hypot(r1.location[0] - r2.location[0], r1.location[1] - r2.location[1])
            return euclidean_units * self.scenario_config.non_gis_distance_unit_miles

    # ----------------------------------------------------------------------------
    # MARK: Disaster Generator
    # ----------------------------------------------------------------------------

    def add_disasters(self):
        dump_site = [d for d in self.resource_nodes if isinstance(d, DumpSite)][0]

        def get_count(val: int | tuple[int, int]) -> int:
            return self.rng.randint(val[0], val[1]) if isinstance(val, tuple) else val

        # Use seasonal disaster patterns if enabled
        if self.scenario_config.use_seasonal_disasters and self.calendar is not None:
            yield from self._add_seasonal_disasters(dump_site)
        else:
            # Original behavior: spawn landslides
            num_to_spawn = get_count(self.scenario_config.num_landslides)
            low, high = self.scenario_config.landslide_size_range
            locations = self._generate_disaster_locations(num_to_spawn)
            spawn_low, spawn_high = self.scenario_config.non_seasonal_spawn_interval_minutes_range

            for loc in locations:
                landslide = Landslide(self, self.rng.randint(low, high), dump_site, location=loc)
                self.disaster_store.put(landslide)
                self._record_disaster_created_metrics(landslide)
                yield self.env.timeout(self.rng.uniform(spawn_low, spawn_high))

    def _add_seasonal_disasters(self, dump_site: DumpSite):
        """Add disasters using a weather/season-conditioned arrival process."""
        from .calendar import OREGON_DISASTER_PATTERNS

        # Calendar is guaranteed non-None when this is called
        calendar = self.calendar
        assert calendar is not None

        def get_count(val: int | tuple[int, int]) -> int:
            return self.rng.randint(val[0], val[1]) if isinstance(val, tuple) else val

        target_total_disasters = get_count(self.scenario_config.num_landslides)
        max_sim_minutes = (
            (calendar.end_date - calendar.start_date).total_seconds() / 60.0
        ) / max(0.001, self.scenario_config.calendar_minutes_per_sim_minute)
        min_gap, max_gap = self.scenario_config.seasonal_spawn_interval_minutes_range

        seasonal_disaster_classes: dict[str, type[Disaster]] = {
            cls.disaster_type: cls
            for cls in (Landslide, SnowEvent, WildfireDebris, FloodEvent)
            if cls.disaster_type is not None
        }

        spawned = 0
        while spawned < target_total_disasters and self.env.now < max_sim_minutes:
            season = calendar.get_season()

            # Build dynamic rates per day from seasonal pattern + optional weather modifier.
            rates: list[tuple[str, float]] = []
            for disaster_type, factors in OREGON_DISASTER_PATTERNS.items():
                multiplier = factors.seasonal_multiplier.get(season, 0.0)
                if multiplier <= 0.0:
                    continue

                rate = factors.base_rate * multiplier
                if self.scenario_config.use_weather_modifier:
                    weather = calendar.get_weather_factor(disaster_type)
                    rate *= 1.0 + max(0.0, min(1.5, weather))

                if rate > 0:
                    rates.append((disaster_type, rate))

            if not rates:
                yield self.env.timeout(60.0)
                continue

            total_rate_per_day = sum(rate for _, rate in rates)
            if total_rate_per_day <= 0:
                yield self.env.timeout(60.0)
                continue

            interarrival_days = self.rng.expovariate(total_rate_per_day)
            interarrival_minutes = interarrival_days * 24.0 * 60.0
            interarrival_minutes = max(min_gap, min(max_gap, interarrival_minutes))
            yield self.env.timeout(interarrival_minutes)
            if self.env.now >= max_sim_minutes:
                continue

            roll = self.rng.random() * total_rate_per_day
            cumulative = 0.0
            disaster_type = rates[0][0]
            for key, rate in rates:
                cumulative += rate
                if roll <= cumulative:
                    disaster_type = key
                    break

            disaster_cls = seasonal_disaster_classes.get(disaster_type)
            if disaster_cls is None:
                continue
            factors = OREGON_DISASTER_PATTERNS[disaster_type]
            size_dist = factors.size_distribution
            size_keys = list(size_dist.keys())
            weather_mod = 1.0
            if self.scenario_config.use_weather_modifier and disaster_cls.disaster_type is not None:
                weather_mod = calendar.get_weather_factor(disaster_cls.disaster_type)
                weather_mod = max(0.5, min(2.5, 0.5 + weather_mod))

            size_key = disaster_cls.pick_size_bucket(self.rng, size_keys, self.scenario_config)
            size_min, size_max = size_dist[size_key]
            sampled_min = int(size_min * weather_mod)
            sampled_max = int(size_max * weather_mod)
            sampled_range = (max(1, sampled_min), max(1, sampled_max))
            loc = self._generate_disaster_locations(1)[0]

            disaster = disaster_cls.spawn_from_seasonal(
                engine=self,
                dump_site=dump_site,
                location=loc,
                size_range=sampled_range,
                season=season,
            )
            self.disaster_store.put(disaster)
            self._record_disaster_created_metrics(disaster)
            spawned += 1

    def _generate_disaster_locations(self, num_locations: int) -> list[tuple[float, float]]:
        """Generate disaster locations, using GIS if available."""
        locations: list[tuple[float, float]] = []

        if self.gis_config is not None and self.gis_config.roads_gdf is not None and self.road_graph is not None:
            sampled = self.gis_config.roads_gdf.sample(n=num_locations, random_state=self.seed)
            sampled = sampled.to_crs(CRS_UTM)
            centroids = sampled.geometry.centroid
            spatial_index = self.gis_config.get_spatial_index()

            for centroid in centroids:
                x, y = centroid.x, centroid.y
                node = spatial_index.get_nearest_node(x, y)
                locations.append(node)
        else:
            dist_range = getattr(self.scenario_config, "landslide_distance_range", (50, 200))
            locations = [
                (self.rng.randint(dist_range[0], dist_range[1]), self.rng.randint(dist_range[0], dist_range[1]))
                for _ in range(num_locations)
            ]

        return locations

    def initialize_world(self):
        """Initialize the world with a randomized set of resources."""

        # Load GIS data if configured
        if self.gis_config is not None:
            self.road_graph = self.gis_config.load_road_network()

        if self.scenario_config.custom_setup_fn:
            self.scenario_config.custom_setup_fn(self)
        else:
            # Determine depot and dump site locations
            depot_loc = (0, 0)
            dump_loc = (0, 0)

            if self.gis_config is not None and len(self.gis_config.depots) > 0:
                depot = self.gis_config.depots[0]
                if "x" in depot and "y" in depot:
                    depot_loc = (depot["x"], depot["y"])

            if self.gis_config is not None and len(self.gis_config.landfills) > 0:
                landfill = self.gis_config.landfills[0]
                if "x" in landfill and "y" in landfill:
                    dump_loc = (landfill["x"], landfill["y"])

            depot = Depot(self, depot_loc)
            dump_site = DumpSite(self, dump_loc)

            self.resource_nodes.append(depot)
            self.resource_nodes.append(dump_site)

            def get_count(val: int | tuple[int, int]) -> int:
                return self.rng.randint(val[0], val[1]) if isinstance(val, tuple) else val

            spawn_plan = [
                (ResourceType.TRUCK, get_count(self.scenario_config.num_trucks)),
                (ResourceType.EXCAVATOR, get_count(self.scenario_config.num_excavators)),
                (ResourceType.SNOWPLOW, get_count(self.scenario_config.num_snowplows)),
                (ResourceType.ASSESSMENT_VEHICLE, get_count(self.scenario_config.num_assessment_vehicles)),
            ]

            rid = 0
            for r_type, count in spawn_plan:
                for _ in range(count):
                    r = Resource(rid, r_type, self)
                    depot.transfer_resource(r, True)
                    rid += 1

    def _record_disaster_created_metrics(self, disaster: Disaster) -> None:
        """Best-effort metric capture for disaster creation."""
        disaster_type = type(disaster).__name__
        road_miles = 0.0
        population_affected = 0

        try:
            from .metrics_tracker import PopulationImpact
            from .simulation import FloodEvent, Landslide, SnowEvent, WildfireDebris

            if isinstance(disaster, SnowEvent):
                road_miles = float(getattr(disaster, "road_miles_affected", 0.0))
                population_affected = PopulationImpact.estimate(
                    road_miles=road_miles,
                    road_type="secondary",
                    duration_hours=float(getattr(disaster, "severity_hours", 24.0)),
                )
            elif isinstance(disaster, FloodEvent):
                road_miles = max(0.5, float(getattr(disaster, "cleanup_amount", 0.0)) * 0.02)
                population_affected = PopulationImpact.estimate(
                    road_miles=road_miles,
                    road_type="secondary",
                    duration_hours=float(getattr(disaster, "duration_hours", 24.0)),
                )
            elif isinstance(disaster, (Landslide, WildfireDebris)):
                size = float(getattr(disaster, "initial_size", getattr(disaster, "initial_debris", 0.0)))
                road_miles = max(0.1, size * 0.01) if size > 0 else 0.0
                population_affected = PopulationImpact.estimate_from_disaster_size(size or 10.0, "secondary")
        except Exception:
            pass

        self.metrics.record_disaster_created(
            disaster_id=disaster.id,
            disaster_type=disaster_type,
            sim_time=self.env.now,
            road_miles=road_miles,
            population_affected=population_affected,
        )

    # ----------------------------------------------------------------------------
    # MARK: Results & Plotting helpers
    # ----------------------------------------------------------------------------
    def get_summary(self) -> dict[str, float]:
        """Return a summary dictionary of results similar to old engine get_results()."""
        # Calculate total costs from all resources
        total_operating_cost = 0.0
        total_fuel_cost = 0.0
        total_hours = 0.0

        for node in self.resource_nodes:
            if hasattr(node, "inventory"):
                for resource_type, inventory in node.inventory.items():
                    for resource in inventory.items:
                        if hasattr(resource, "accumulated_cost"):
                            total_operating_cost += resource.accumulated_cost
                            total_hours += resource.total_hours_operated
                            # Estimate fuel cost (simplified)
                            if hasattr(resource, "fuel_consumption_rate"):
                                fuel_gallons = resource.fuel_consumption_rate * resource.total_hours_operated
                                total_fuel_cost += fuel_gallons * 4.50  # Default fuel price

        self.total_spent = total_operating_cost + total_fuel_cost

        # Check if budget is exhausted
        if self.scenario_config.track_costs and self.total_spent >= self.scenario_config.annual_budget:
            self.budget_exhausted = True

        # Get metrics summary
        metrics_summary = self.metrics.get_summary()

        return {
            "non_idle_time": self.non_idle_time,
            "total_operating_cost": total_operating_cost,
            "total_fuel_cost": total_fuel_cost,
            "total_spent": self.total_spent,
            "budget": self.scenario_config.annual_budget,
            "budget_remaining": self.scenario_config.annual_budget - self.total_spent,
            "budget_utilization": (
                self.total_spent / self.scenario_config.annual_budget if self.scenario_config.annual_budget > 0 else 0
            ),
            "total_resource_hours": total_hours,
            "budget_exhausted": self.budget_exhausted,
            "total_dispatch_delay": self.total_dispatch_delay,
            "dispatch_delay_events": self.dispatch_delay_events,
            "avg_dispatch_delay": (
                self.total_dispatch_delay / self.dispatch_delay_events if self.dispatch_delay_events > 0 else 0
            ),
            "max_dispatch_delay": self.max_dispatch_delay,
            # Disaster metrics
            "disasters_created": metrics_summary.get("total_disasters_created", 0),
            "disasters_resolved": metrics_summary.get("total_disasters_resolved", 0),
            "resolution_rate": metrics_summary.get("resolution_rate", 0),
            "avg_response_time": metrics_summary.get("avg_response_time", 0),
            "avg_resolution_time": metrics_summary.get("avg_resolution_time", 0),
            "total_population_affected": metrics_summary.get("total_population_affected", 0),
        }

    def setup_plot(self) -> tuple[Figure, list[Axes]]:
        fig, axs = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={"height_ratios": [3, 1]})
        ax_map = axs[0]
        ax_map.set_aspect("equal")
        ax_map.set_xlabel("X")
        ax_map.set_ylabel("Y")
        ax_map.grid(True, alpha=0.2)

        if self.road_graph is not None and self.gis_config is not None:
            self._gis_edge_segments = [(u, v) for u, v in self.road_graph.edges()]
            xs = [pt[0] for pt in self.road_graph.nodes]
            ys = [pt[1] for pt in self.road_graph.nodes]
            if xs and ys:
                self._gis_bounds = (min(xs), max(xs), min(ys), max(ys))
        else:
            self._gis_edge_segments = None
            self._gis_bounds = None

        min_x, max_x, min_y, max_y = self._compute_map_bounds()
        ax_map.set_xlim(min_x, max_x)
        ax_map.set_ylim(min_y, max_y)

        axs[1].set_xlabel("Time")
        axs[1].set_ylabel("Dirt")
        axs[1].grid(True)
        plt.ion()
        return fig, axs

    def _compute_map_bounds(self) -> tuple[float, float, float, float]:
        x_points: list[float] = []
        y_points: list[float] = []

        for node in self.resource_nodes + [self.idle_resources] + self.disaster_store.items:
            x_points.append(node.location[0])
            y_points.append(node.location[1])

            for resources in node.roster.values():
                for resource in resources:
                    x_points.extend([resource.location[0], resource.prev_location[0]])
                    y_points.extend([resource.location[1], resource.prev_location[1]])

        if self._gis_bounds is not None:
            x_points.extend([self._gis_bounds[0], self._gis_bounds[1]])
            y_points.extend([self._gis_bounds[2], self._gis_bounds[3]])

        if not x_points or not y_points:
            return (-120, 120, -120, 120)

        min_x = min(x_points)
        max_x = max(x_points)
        min_y = min(y_points)
        max_y = max(y_points)

        x_span = max(max_x - min_x, 1.0)
        y_span = max(max_y - min_y, 1.0)
        x_margin = max(5.0, x_span * 0.05)
        y_margin = max(5.0, y_span * 0.05)
        return (min_x - x_margin, max_x + x_margin, min_y - y_margin, max_y + y_margin)

    def update_plot(self):
        if self.fig is None or self.axs is None:
            return
        ax_map, ax_dirt = self.axs
        ax_map.clear()
        ax_map.set_aspect("equal")
        ax_map.set_xlabel("X")
        ax_map.set_ylabel("Y")
        ax_map.grid(True, alpha=0.2)
        min_x, max_x, min_y, max_y = self._compute_map_bounds()
        ax_map.set_xlim(min_x, max_x)
        ax_map.set_ylim(min_y, max_y)

        if self._gis_edge_segments:
            roads = LineCollection(self._gis_edge_segments, colors="#b0b0b0", linewidths=0.3, alpha=0.5, zorder=1)
            ax_map.add_collection(roads)

        ax_map.set_title(f"Policy: {self.policy.name} | Seed: {self.seed} | Time: {self.env.now:.2f}")

        # Plot resource holders (depots, dump sites, idle node).
        for node in self.resource_nodes + [self.idle_resources]:
            ax_map.plot(node.location[0], node.location[1], node.render["color"], markersize=12, zorder=3)
            ax_map.text(node.location[0], node.location[1], f"{node.render['label']}-{node.id}", fontsize=8, zorder=4)

        # Plot disasters.
        for node in self.disaster_store.items:
            size = node.percent_remaining() * 100
            ax_map.plot(
                node.location[0],
                node.location[1],
                node.render["marker"],
                color=node.render["color"],
                markersize=10 + size,
                alpha=0.75,
                zorder=4,
            )
            ax_map.text(
                node.location[0],
                node.location[1],
                f"{node.render['label']}-{node.id}\n{int(size)}%",
                fontsize=8,
                ha="center",
                va="center",
                zorder=5,
            )

        # Plot all resources from each holder/disaster roster.
        for node in self.resource_nodes + [self.idle_resources] + self.disaster_store.items:
            for resource_type, resources in node.roster.items():
                for r in resources:
                    frac = r.id * (1 + math.sqrt(5)) / 2
                    loc1 = (cos(frac) * 10 + r.location[0], sin(frac) * 10 + r.location[1])
                    loc2 = (
                        cos(frac) * 10 + r.prev_location[0],
                        sin(frac) * 10 + r.prev_location[1],
                    )

                    time_frac = 1 if r.move_time == 0 else (self.env.now - r.move_start_time) / r.move_time
                    time_frac = max(0, time_frac)
                    time_frac = min(1, time_frac)

                    loc = (
                        loc1[0] * time_frac + loc2[0] * (1 - time_frac),
                        loc1[1] * time_frac + loc2[1] * (1 - time_frac),
                    )

                    marker = resource_type.render["marker"]
                    color = resource_type.render["color"]

                    ax_map.plot(loc[0], loc[1], marker=marker, color=color, markersize=8, zorder=6)
                    ax_map.text(loc[0] + 2, loc[1] + 2, f"{r.id}", color=color, fontsize=7, zorder=7)

        ax_dirt.clear()
        if self._disaster_histories and len(self._time_points) > 0:
            ids = sorted(self._disaster_histories.keys())
            y_data = [self._disaster_histories[i] for i in ids]
            labels = [f"L{i}" for i in ids]
            ax_dirt.stackplot(self._time_points, *y_data, labels=labels, alpha=0.8, step="post")
            ax_dirt.legend(loc="upper left", fontsize="small", framealpha=0.5)

        plt.draw()
        plt.pause(0.001)
