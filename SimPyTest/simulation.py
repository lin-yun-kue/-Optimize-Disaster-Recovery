from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Generator
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, ClassVar, TypedDict

try:
    from typing import Never
except ImportError:
    from typing import NoReturn as Never

import simpy
from simpy.events import AllOf, AnyOf, Process
from simpy.resources.container import ContainerGet
from simpy.resources.store import FilterStoreGet, StorePut
from typing_extensions import override

from SimPyTest.real_world_params import travel_minutes_from_distance
from SimPyTest.multi_year_config import ResourceClass, get_default_resource_costs

if TYPE_CHECKING:
    from .engine import ScenarioConfig, SimPySimulationEngine, SimulationRNG
    from .calendar import Season
    from typing import Optional

# ============================================================================
# MARK: Configuration & Enums
# ============================================================================


class ResourceRender(TypedDict):
    """Rendering options for a resource."""

    marker: str
    color: str


class ResourceType(Enum):
    TRUCK = (
        0,
        {"speed": 23.4, "capacity": 10, "load_time": 10, "dump_time": 10},
        ResourceRender({"marker": "^", "color": "blue"}),
    )
    EXCAVATOR = (
        1,
        {"speed": 11.6, "work_rate": 10},
        ResourceRender({"marker": "P", "color": "orange"}),
    )
    SNOWPLOW = (
        2,
        {"speed": 15.0, "work_rate": 5},
        ResourceRender({"marker": "s", "color": "cyan"}),
    )
    ASSESSMENT_VEHICLE = (
        3,
        {"speed": 35.0, "work_rate": 1},
        ResourceRender({"marker": "A", "color": "purple"}),
    )
    # FIRE_TRUCK = (
    #     auto(),
    #     {"speed": 3.0, "work_rate": 5},
    #     ResourceRender({"marker": "p", "color": "red"}),
    # )
    # AMBULANCE = (
    #     auto(),
    #     {"speed": 4.0, "capacity": 1, "load_time": 5},
    #     ResourceRender({"marker": "X", "color": "skyblue"}),
    # )

    def __init__(self, value: int, specs: dict[str, float], render: ResourceRender):
        self._value_ = value
        self.specs = specs
        self.render = render

    if TYPE_CHECKING:

        @property
        @override
        def value(self) -> int: ...


# ============================================================================
# MARK: Base Resource
# ============================================================================


@dataclass
class Resource:
    id: int
    resource_type: ResourceType
    engine: SimPySimulationEngine

    assigned_node: ResourceNode = field(init=False)

    drive_process: simpy.Process | None = field(default=None, repr=False)

    # Visualization State
    _location: tuple[float, float] = field(default=(0, 0), repr=False)
    prev_location: tuple[float, float] = (0, 0)
    _move_time: float = field(default=0, repr=False)
    move_start_time: float = 0

    # Resource constraints and costs
    home_depot: Optional["Depot"] = field(default=None, repr=False)  # Must return here to refuel
    fuel_level: float = field(default=100.0, repr=False)  # 0-100%
    fuel_capacity: float = field(default=100.0, repr=False)  # Gallons
    fuel_consumption_rate: float = field(default=5.0, repr=False)  # Gallons/hour
    hourly_operating_cost: float = field(default=75.0, repr=False)  # $/hour
    accumulated_cost: float = field(default=0.0, repr=False)  # Total cost accrued
    total_hours_operated: float = field(default=0.0, repr=False)  # Total operational hours

    def __post_init__(self):
        self.assigned_node = self.engine.idle_resources
        costs = get_default_resource_costs()
        class_map = {
            ResourceType.TRUCK: ResourceClass.TRUCK,
            ResourceType.EXCAVATOR: ResourceClass.EXCAVATOR,
            ResourceType.SNOWPLOW: ResourceClass.SNOWPLOW,
            ResourceType.ASSESSMENT_VEHICLE: ResourceClass.ASSESSMENT_VEHICLE,
        }
        rc = class_map.get(self.resource_type)
        if rc is not None and rc in costs:
            cfg = costs[rc]
            self.hourly_operating_cost = float(cfg.hourly_operating)
            self.fuel_consumption_rate = float(cfg.fuel_per_hour)

    def consume_fuel(self, hours: float) -> float:
        """Consume fuel for given hours of operation. Returns fuel consumed."""
        # Apply time variance from config if available
        variance = self.engine.scenario_config.time_variance

        actual_consumption = self.fuel_consumption_rate * hours * self.engine.rng.uniform(1 - variance, 1 + variance)
        self.fuel_level = max(0, self.fuel_level - actual_consumption)
        return actual_consumption

    def add_operating_cost(self, hours: float) -> float:
        """Add operating cost for given hours. Returns cost added."""
        cost = self.hourly_operating_cost * hours
        self.accumulated_cost += cost
        self.total_hours_operated += hours

        # Also update engine's total spent if tracking enabled
        if self.engine.scenario_config.track_costs:
            self.engine.total_spent += cost

        self.engine.metrics.record_resource_metrics(
            resource_id=self.id,
            resource_type=self.resource_type.name,
            hours_operated=hours,
            total_cost=cost,
        )

        return cost

    def needs_refuel(self, threshold: float = 20.0) -> bool:
        """Check if resource needs refueling (below threshold %)."""
        return self.fuel_level < (self.fuel_capacity * threshold / 100)

    def refuel(self) -> None:
        """Refuel to full capacity."""
        self.fuel_level = self.fuel_capacity

    def refuel_amount(self, gallons: float) -> None:
        """Add specific amount of fuel."""
        self.fuel_level = min(self.fuel_capacity, self.fuel_level + gallons)

    @property
    def location(self) -> tuple[float, float]:
        """The current location of the resource."""
        return self._location

    @location.setter
    def location(self, loc: tuple[float, float]):
        self.prev_location = self._location
        self._location = loc

    @property
    def move_time(self) -> float:
        """The time it takes to move the resource from its current location to
        its new location."""
        return self._move_time

    @move_time.setter
    def move_time(self, time: float):
        self._move_time = time
        self.move_start_time = self.engine.env.now

    @override
    def __hash__(self) -> int:
        return hash(self.id)

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, Resource) and self.id == other.id


# ============================================================================
# MARK: Base Resource Node
# ============================================================================


class ResourceNodeRender(TypedDict):
    """Rendering options for a resource node."""

    label: str
    color: str


class ResourceStore(simpy.FilterStore):
    items: list[Resource]


class ResourceNode(ABC):
    """
    A generic node in the graph.
    """

    engine: SimPySimulationEngine
    env: simpy.Environment
    id: int
    location: tuple[float, float]

    def __init__(self, engine: SimPySimulationEngine, location: tuple[float, float]):
        self.engine = engine
        self.env = engine.env
        self.id = engine.rng.randint(1, 10000)
        self.location = location

        # Physical Inventory: Resources currently ON SITE and AVAILABLE
        self.inventory: dict[ResourceType, ResourceStore] = defaultdict(lambda: ResourceStore(self.env))

        # Administrative Roster: Resources ASSIGNED to this node, but not AVAILABLE
        # Could be driving or performing a task
        self.roster: dict[ResourceType, set[Resource]] = defaultdict(set)

    @property
    @abstractmethod
    def render(self) -> ResourceNodeRender:
        """Rendering options for a resource node."""
        raise NotImplementedError

    def mark_resource_available(self, resource: Resource):
        """Logic when a resource physically arrives."""
        self.inventory[resource.resource_type].put(resource)

    def transfer_resource(self, resource: Resource, instant: bool = False):
        """
        Moves the resource from its current node to this node.
        """
        # 1. Roster Logic
        # Remove from old boss
        if resource.assigned_node and resource in resource.assigned_node.roster[resource.resource_type]:
            resource.assigned_node.roster[resource.resource_type].remove(resource)

        # 2. Physics Logic (Travel)
        if resource.drive_process is not None and not resource.drive_process.triggered:
            try:
                resource.drive_process.interrupt("reassigned")
            except Exception:
                pass

            # print(f"Assigning resource {resource} to")
        if instant:
            # resource.assigned_node = self
            # self.roster[resource.resource_type].add(resource)
            # resource.location = self.location
            # resource.move_time = 0
            # self.mark_resource_available(resource)
            # resource.drive_process = None
            for _ in self.drive_resource(resource):
                pass
        else:
            resource.drive_process = self.env.process(self.drive_resource(resource))

    def drive_resource(self, resource: Resource):
        """
        Moves resource from the resource's current node -> this node.
        This part handles the actual driving of the resource.
        """
        resource.assigned_node = self
        self.roster[resource.resource_type].add(resource)

        specs = resource.resource_type.specs

        start_loc = resource.location
        end_loc = self.location

        if start_loc == (0, 0):
            travel_time_minutes = 0.0
        else:
            # get_distance returns miles (GIS meters are converted by the engine)
            dist_miles = self.engine.get_distance(resource, self)
            travel_time_minutes = travel_minutes_from_distance(dist_miles, specs["speed"])

        # Snap destination to road graph if in GIS mode
        if self.engine.gis_config is not None and self.engine.road_graph is not None:
            spatial_index = self.engine.gis_config.get_spatial_index()
            snapped_loc = spatial_index.get_nearest_node(self.location[0], self.location[1])
            resource.location = snapped_loc
        else:
            resource.location = self.location
        resource.move_time = travel_time_minutes

        try:
            yield self.env.timeout(travel_time_minutes)
        except simpy.Interrupt as _e:
            loc1 = resource.location
            loc2 = resource.prev_location

            time_frac = 1 if resource.move_time == 0 else (self.env.now - resource.move_start_time) / resource.move_time
            time_frac = max(0, time_frac)
            time_frac = min(1, time_frac)

            loc = (
                loc1[0] * time_frac + loc2[0] * (1 - time_frac),
                loc1[1] * time_frac + loc2[1] * (1 - time_frac),
            )

            # Snap interrupted location to road graph if in GIS mode
            if self.engine.gis_config is not None and self.engine.road_graph is not None:
                spatial_index = self.engine.gis_config.get_spatial_index()
                loc = spatial_index.get_nearest_node(loc[0], loc[1])

            resource.location = loc
            resource.move_time = 0
            return

        self.mark_resource_available(resource)
        resource.drive_process = None


# ============================================================================
# MARK: Base Disaster
# ============================================================================


class DisasterRender(TypedDict):
    """Rendering options for a disaster."""

    label: str
    color: str
    marker: str


class Disaster(ResourceNode, ABC):
    """
    Base class for all disasters.
    Handles the logistics: resources arriving, tracking active rosters, and returning home.
    """

    one_hot_index: int

    active: bool
    created_time: float
    _first_response_recorded: bool
    assisting_resource_ids: set[int]
    disaster_type: ClassVar[str | None] = None
    seasonal_size_weights: ClassVar[dict[str, float]] = {}

    def __init__(
        self,
        engine: SimPySimulationEngine,
        location: tuple[float, float] | None = None,
    ):
        if location is None:
            # Random location if not specified
            loc = (engine.rng.randint(-100, 100), engine.rng.randint(-100, 100))
        else:
            loc = location
        super().__init__(engine, loc)

        self.active = True
        self.created_time = engine.env.now
        self._first_response_recorded = False
        self.assisting_resource_ids = set()

        # Rich event metadata (aligned with closure/dispatch event schemas).
        self.cause_category: str = "unknown"
        self.closure_type: str = "unspecified"
        self.evidence_tier: str = "tier_d"
        self.severity_index: float = 0.5
        self.detection_delay_minutes: float = 0.0
        self.assessment_minutes_estimate: float = 0.0
        self.admin_reopen_delay_minutes: float = 0.0
        self.estimated_total_closure_minutes: float = 0.0
        self.sampled_closure_duration_hours: float = 0.0
        self.start_timestamp_calendar: str | None = None
        if self.engine.calendar is not None:
            self.start_timestamp_calendar = self.engine.calendar.current_date.isoformat()

    def apply_operational_prior(self, prior_key: str, baseline_closure_minutes: float) -> None:
        """Attach disaster timing/category priors from scenario configuration."""
        priors = self.engine.scenario_config.disaster_operational_priors
        prior = priors.get(prior_key, {})

        self.cause_category = str(prior.get("cause_category", "unknown"))
        self.closure_type = str(prior.get("closure_type", "unspecified"))
        self.evidence_tier = str(prior.get("evidence_tier", "tier_d"))

        det_range = prior.get("detection_delay_minutes_range", (0.0, 0.0))
        if isinstance(det_range, tuple) and len(det_range) == 2:
            self.detection_delay_minutes = self.engine.rng.uniform(float(det_range[0]), float(det_range[1]))

        assess_range = prior.get("assessment_minutes_range", (0.0, 0.0))
        if isinstance(assess_range, tuple) and len(assess_range) == 2:
            self.assessment_minutes_estimate = self.engine.rng.uniform(float(assess_range[0]), float(assess_range[1]))

        reopen_range = prior.get("admin_reopen_delay_minutes_range", (0.0, 0.0))
        if isinstance(reopen_range, tuple) and len(reopen_range) == 2:
            self.admin_reopen_delay_minutes = self.engine.rng.uniform(float(reopen_range[0]), float(reopen_range[1]))

        severity_range = prior.get("severity_index_range", (0.4, 1.0))
        if isinstance(severity_range, tuple) and len(severity_range) == 2:
            self.severity_index = self.engine.rng.uniform(float(severity_range[0]), float(severity_range[1]))

        self.estimated_total_closure_minutes = max(
            0.0,
            baseline_closure_minutes
            + self.detection_delay_minutes
            + self.assessment_minutes_estimate
            + self.admin_reopen_delay_minutes,
        )

    def apply_closure_duration_prior(
        self,
        prior_key: str,
        season: "Season" | None,
        min_hours: float = 0.5,
        max_hours: float = 24.0 * 21,
    ) -> float:
        """Sample heavy-tailed closure duration prior for this disaster."""
        prior_cfg = self.engine.scenario_config.disaster_closure_lognormal_priors.get(prior_key, {})
        season_key = season.name.lower() if season is not None else "default"
        prior = prior_cfg.get(season_key, prior_cfg.get("default"))
        if prior is None:
            sampled = min_hours
        else:
            mu, sigma = prior
            sampled = self.engine.rng.lognormvariate(mu, sigma)
        sampled = max(min_hours, min(max_hours, sampled))
        self.sampled_closure_duration_hours = float(sampled)
        self.estimated_total_closure_minutes = max(
            self.estimated_total_closure_minutes, self.sampled_closure_duration_hours * 60.0
        )
        return self.sampled_closure_duration_hours

    @abstractmethod
    def needed_resources(self) -> list[ResourceType]:
        """Returns a list of resources needed to resolve the disaster."""
        raise NotImplementedError

    @abstractmethod
    def percent_remaining(self) -> float:
        """Returns the percentage of the disaster that is still active."""
        raise NotImplementedError

    @abstractmethod
    def get_scale(self) -> float:
        """Returns the scale of the disaster."""
        raise NotImplementedError

    @property
    @abstractmethod
    @override
    def render(self) -> DisasterRender:
        """Rendering options for a disaster."""
        raise NotImplementedError

    @override
    def mark_resource_available(self, resource: Resource):
        """Logic for when a resource actually reaches the coordinates."""
        # Check if the job is already done while they were driving
        if not self.active:
            self.engine.idle_resources.transfer_resource(resource)
            return

        self.assisting_resource_ids.add(resource.id)
        self.engine.metrics.record_resource_metrics(
            resource_id=resource.id,
            resource_type=resource.resource_type.name,
            hours_operated=0.0,
            total_cost=0.0,
        )
        self.engine.metrics.increment_disaster_assist(resource.id)
        if not self._first_response_recorded:
            self.engine.metrics.record_first_response(self.id, self.env.now)
            self._first_response_recorded = True

        super().mark_resource_available(resource)

    def resolve(self):
        self.active = False
        self.engine.metrics.record_disaster_resolved(
            disaster_id=self.id,
            sim_time=self.env.now,
            total_cost=0.0,
            resources_used=len(self.assisting_resource_ids),
        )
        yield self.engine.disaster_store.get(filter=lambda x: x == self)
        self.teardown()

    def teardown(self):
        """Clean up all resources when disaster is resolved."""
        # Drain all stores and send everyone home
        # If people are heading here we send them away anyway
        # for r_type, store in self.roster.items():
        #     for r in store:
        #         self.idle_resources.transfer_resource(r)

        # The above as a while loop to avoid Set changes during iteration
        for _r_type, store in self.roster.items():
            while len(store) > 0:
                self.engine.idle_resources.transfer_resource(store.pop())

    @classmethod
    def pick_size_bucket(
        cls,
        rng: "SimulationRNG",
        available_buckets: list[str],
        scenario_config: "ScenarioConfig",
    ) -> str:
        """Pick a size bucket using class defaults with optional scenario overrides."""
        if not available_buckets:
            raise ValueError("available_buckets must not be empty")

        if cls.disaster_type is None:
            return rng.choice(available_buckets)

        override_weights = scenario_config.seasonal_size_bucket_weights.get(cls.disaster_type, {})
        weights = override_weights or cls.seasonal_size_weights
        if not weights:
            return rng.choice(available_buckets)

        valid = [(bucket, weight) for bucket, weight in weights.items() if bucket in available_buckets and weight > 0]
        total_weight = sum(weight for _, weight in valid)
        if total_weight <= 0:
            return rng.choice(available_buckets)

        pick = rng.random() * total_weight
        cumulative = 0.0
        for bucket, weight in valid:
            cumulative += weight
            if pick <= cumulative:
                return bucket
        return rng.choice(available_buckets)

    @classmethod
    def spawn_from_seasonal(
        cls,
        engine: "SimPySimulationEngine",
        dump_site: "DumpSite",
        location: tuple[float, float],
        size_range: tuple[int, int],
        season: "Season",
    ) -> "Disaster":
        raise NotImplementedError(f"{cls.__name__} must implement spawn_from_seasonal")


# ============================================================================
# MARK: Resource Nodes / Buildings
# ============================================================================


class IdleResources(ResourceNode):
    def __init__(self, engine: SimPySimulationEngine):
        super().__init__(engine, location=(0, -10))

    @property
    @override
    def render(self) -> ResourceNodeRender:
        return {"color": "cs", "label": "Resource Node"}

    @override
    def drive_resource(self, resource: Resource, instant: bool = False):
        """Moves resource from the resource's current node -> Target."""
        resource.assigned_node = self
        self.roster[resource.resource_type].add(resource)

        yield self.env.timeout(0)

        self.mark_resource_available(resource)

    def get_any_resource(self) -> Generator[AnyOf, Resource, Resource]:
        """Returns a random resource from any of the stores."""
        # Since resources are in separate stores by type, we listen to all of them
        # and trigger when the first one becomes available.
        get_events: dict[ResourceType, FilterStoreGet] = {rt: self.inventory[rt].get() for rt in ResourceType}
        finished: dict[FilterStoreGet, Resource] = yield self.env.any_of(list(get_events.values()))  # pyright: ignore[reportAssignmentType]

        winner_event = self.engine.rng.choice(list(finished.keys()))
        resource: Resource = finished[winner_event]

        for rt, event in get_events.items():
            if event == winner_event:
                continue

            if event.triggered:
                unused_resource = event.value
                self.inventory[rt].put(unused_resource)
            else:
                event.cancel()

        return resource


class Depot(ResourceNode):
    def __init__(self, engine: SimPySimulationEngine, location: tuple[float, float]):
        super().__init__(engine, location=location)

    @property
    @override
    def render(self) -> ResourceNodeRender:
        return {"color": "ks", "label": "Depot"}

    @override
    def drive_resource(self, resource: Resource):
        self.engine.idle_resources.transfer_resource(resource, True)
        return super().drive_resource(resource)


# class Hospital(ResourceNode):
#     def __init__(self, engine: SimPySimulationEngine):
#         super().__init__(engine, location=SimulationConfig.HOSPITAL_LOCATION)
#         engine.env.process(self.process_loop())

#     @property
#     @override
#     def render(self) -> ResourceNodeRender:
#         return {"color": "ms", "label": "Hospital"}

#     def process_loop(self) -> Generator[FilterStoreGet | Timeout, Resource, Never]:
#         while True:
#             amb: Resource = yield self.inventory[ResourceType.AMBULANCE].get()
#             yield self.env.timeout(10)  # Unloading patient
#             self.engine.idle_resources.transfer_resource(amb)


class DumpSite(ResourceNode):
    """
    The Dump Site is a Node with capacity constraints.
    Trucks arrive here, wait for available dump slots, wait for 'dump_time',
    and then automatically return to idle resources.
    """

    def __init__(self, engine: SimPySimulationEngine, location: tuple[float, float], capacity: int = 2):
        super().__init__(engine, location=location)

        # Capacity management: simpy Store with fixed number of "slots"
        self.capacity = capacity
        self.dump_slots = simpy.Store(engine.env, capacity)
        # Initialize with available slots
        for i in range(capacity):
            self.dump_slots.put(i)

        self.active_dumps: int = 0
        self.total_dumps_completed: int = 0

        self.process: Process = engine.env.process(self.dump_loop())

    @property
    @override
    def render(self) -> ResourceNodeRender:
        return {"color": "ys", "label": "Dump Site"}

    def dump_loop(self):
        truck_specs = ResourceType.TRUCK.specs

        while True:
            # 1. Get Resources from Inventory (Must be physically here)
            truck: Resource = yield self.inventory[ResourceType.TRUCK].get()

            # 2. Wait for an available dump slot (capacity constraint)
            slot: int = yield self.dump_slots.get()
            self.active_dumps += 1

            # 3. Process the dump with time variance
            dump_time = truck_specs.get("dump_time", 10)  # Default 10 minutes

            # Apply time variance from config if available
            variance = self.engine.scenario_config.time_variance

            actual_dump_time = dump_time * self.engine.rng.uniform(1 - variance, 1 + variance)

            # Consume fuel and add operating cost
            truck.consume_fuel(actual_dump_time / 60)  # Convert to hours
            truck.add_operating_cost(actual_dump_time / 60)

            yield self.env.timeout(actual_dump_time)

            # 4. Release the slot
            self.dump_slots.put(slot)
            self.active_dumps -= 1
            self.total_dumps_completed += 1

            # 5. Send Truck back to idle
            self.engine.idle_resources.transfer_resource(truck)


# ============================================================================
# MARK: Disasters
# ============================================================================


class DisasterStore(simpy.FilterStore):
    items: list[Disaster]

    def wait_for_any(self) -> Generator[FilterStoreGet | StorePut, Disaster, None]:
        """Yields until any item is available."""
        if self.items:
            return

        d: Disaster = yield self.get()
        yield self.put(d)

    if TYPE_CHECKING:

        @override
        def get(self, filter: Callable[[Disaster], bool] = lambda item: True) -> FilterStoreGet: ...


class Landslide(Disaster):
    """
    Requires: EXCAVATOR + TRUCK.
    Work cannot happen without both.
    """

    one_hot_index: int = 0
    disaster_type: ClassVar[str] = "landslide"
    seasonal_size_weights: ClassVar[dict[str, float]] = {"small": 0.50, "medium": 0.30, "large": 0.15, "major": 0.05}

    dirt: simpy.Container
    initial_size: float
    dump_node: DumpSite
    process: simpy.Process

    def __init__(
        self,
        engine: SimPySimulationEngine,
        size: float,
        dump_node: DumpSite,
        location: tuple[float, float] | None = None,
    ):
        super().__init__(engine, location)

        self.dirt = simpy.Container(engine.env, init=size)
        self.initial_size = size

        self.dump_node = dump_node
        truck_specs = ResourceType.TRUCK.specs
        trips_estimate = max(1.0, size / max(1.0, truck_specs["capacity"]))
        baseline_minutes = trips_estimate * float(truck_specs["load_time"] + truck_specs["dump_time"])
        self.apply_operational_prior("landslide", baseline_minutes)

        self.process = engine.env.process(self.work_loop())

    @override
    def needed_resources(self) -> list[ResourceType]:
        return [ResourceType.EXCAVATOR, ResourceType.TRUCK]

    @override
    def percent_remaining(self):
        return self.dirt.level / self.initial_size

    @override
    def get_scale(self):
        return self.dirt.level / self.engine.scenario_config.landslide_size_range[1]

    @property
    @override
    def render(self) -> DisasterRender:
        return {"color": "brown", "label": "Landslide", "marker": "o"}

    def work_loop(self) -> Generator[FilterStoreGet | ContainerGet | AllOf | Process, Resource, None]:
        """The specific logic for clearing a landslide."""
        truck_specs = ResourceType.TRUCK.specs

        dump_trips: list[Process] = []

        while self.active and self.dirt.level > 0:
            # 1. Request resources

            excavator: Resource = yield self.inventory[ResourceType.EXCAVATOR].get()
            truck: Resource = yield self.inventory[ResourceType.TRUCK].get()

            # 2. Perform Work
            amount = min(truck_specs["capacity"], self.dirt.level)
            yield self.dirt.get(amount)

            def load(ls: Landslide, excavator: Resource, truck: Resource):

                # Simulate Loading Time
                yield ls.env.timeout(truck_specs["load_time"])

                # 3. Release Excavator immediately (stays on site)
                yield ls.inventory[ResourceType.EXCAVATOR].put(excavator)

                # 4. Send Truck to dump (Process handled in parallel so loop continues)
                ls.dump_node.transfer_resource(truck)

            dump_trips.append(self.env.process(load(self, excavator, truck)))

        # Wait for all dump trips to finish
        yield self.env.all_of(dump_trips)

        # Once loop breaks, teardown
        yield self.env.process(self.resolve())

    @classmethod
    def spawn_from_seasonal(
        cls,
        engine: "SimPySimulationEngine",
        dump_site: "DumpSite",
        location: tuple[float, float],
        size_range: tuple[int, int],
        season: "Season",  # noqa: ARG003
    ) -> "Landslide":
        size = engine.rng.randint(size_range[0], size_range[1])
        disaster = cls(engine, size, dump_site, location=location)
        disaster.apply_closure_duration_prior("landslide", season)
        return disaster


class SnowEvent(Disaster):
    """
    Snow/ice event blocking roads.
    Requires: SNOWPLOW resources.
    Can be active (plowing) or passive (waiting for melt).
    """

    one_hot_index: int = 1
    disaster_type: ClassVar[str] = "snow"
    seasonal_size_weights: ClassVar[dict[str, float]] = {"light": 0.65, "moderate": 0.25, "heavy": 0.10}

    def __init__(
        self,
        engine: SimPySimulationEngine,
        severity_hours: float,
        road_miles_affected: float,
        location: tuple[float, float] | None = None,
    ):
        super().__init__(engine, location)

        self.severity_hours = severity_hours
        self.road_miles_affected = road_miles_affected
        self.remaining_work = severity_hours * road_miles_affected
        self.initial_work = self.remaining_work

        self.severity_minutes = severity_hours * 60.0
        self.auto_resolve_minutes = self.severity_minutes * engine.scenario_config.snow_auto_resolve_multiplier
        self.plow_interval_minutes = engine.scenario_config.snow_work_interval_minutes
        self.apply_operational_prior("snow", self.auto_resolve_minutes)

        self.process = engine.env.process(self.work_loop())
        self.auto_resolve_process = engine.env.process(self.auto_resolve_loop())

    @override
    def needed_resources(self) -> list[ResourceType]:
        return [ResourceType.SNOWPLOW]

    @override
    def percent_remaining(self):
        return self.remaining_work / self.initial_work if self.initial_work > 0 else 0

    @override
    def get_scale(self):
        return self.road_miles_affected / 10

    @property
    @override
    def render(self) -> DisasterRender:
        return {"color": "lightblue", "label": "Snow", "marker": "*"}

    def work_loop(self):
        """Plow snow until cleared."""
        while self.active and self.remaining_work > 0:
            plow = yield self.inventory[ResourceType.SNOWPLOW].get()

            num_plows = len(self.roster[ResourceType.SNOWPLOW])
            base_rate = ResourceType.SNOWPLOW.specs["work_rate"]
            efficiency = 1.0

            work_per_hour = base_rate * efficiency
            work_hours = (
                min(self.plow_interval_minutes / 60.0, self.remaining_work / work_per_hour) if work_per_hour > 0 else 0
            )

            if work_hours > 0:
                plow.consume_fuel(work_hours)
                plow.add_operating_cost(work_hours)

                yield self.env.timeout(work_hours * 60)
                self.remaining_work -= work_per_hour * work_hours

            yield self.inventory[ResourceType.SNOWPLOW].put(plow)

        if self.active:
            yield self.env.process(self.resolve())

    def auto_resolve_loop(self):
        """Snow melts naturally over time."""
        yield self.env.timeout(self.auto_resolve_minutes)
        if self.active:
            self.remaining_work = 0

    @classmethod
    def spawn_from_seasonal(
        cls,
        engine: "SimPySimulationEngine",
        dump_site: "DumpSite",  # noqa: ARG003
        location: tuple[float, float],
        size_range: tuple[int, int],
        season: "Season",
    ) -> "SnowEvent":
        low, high = size_range
        prior_cfg = engine.scenario_config.disaster_closure_lognormal_priors.get("snow", {})
        season_key = season.name.lower()
        prior = prior_cfg.get(season_key, prior_cfg.get("default"))

        if prior is not None:
            mu, sigma = prior
            severity_hours = max(low, min(high, int(engine.rng.lognormvariate(mu, sigma))))
        else:
            severity_hours = engine.rng.randint(low, high)

        miles_low, miles_high = engine.scenario_config.snow_road_miles_range
        road_miles = engine.rng.uniform(miles_low, miles_high)
        disaster = cls(engine, severity_hours, road_miles, location=location)
        disaster.sampled_closure_duration_hours = float(severity_hours)
        return disaster


class WildfireDebris(Disaster):
    """
    Wildfire debris blocking roads (fallen trees, etc.).
    Requires: EXCAVATOR + TRUCK (similar to landslide but different seasonality).
    """

    one_hot_index: int = 2
    disaster_type: ClassVar[str] = "wildfire_debris"
    seasonal_size_weights: ClassVar[dict[str, float]] = {"small": 0.50, "medium": 0.35, "large": 0.15}

    def __init__(
        self,
        engine: SimPySimulationEngine,
        debris_amount: float,  # cubic yards
        dump_node: DumpSite,
        location: tuple[float, float] | None = None,
    ):
        super().__init__(engine, location)

        self.debris = simpy.Container(engine.env, init=debris_amount)
        self.initial_debris = debris_amount
        self.dump_node = dump_node
        truck_specs = ResourceType.TRUCK.specs
        trips_estimate = max(1.0, debris_amount / max(1.0, truck_specs["capacity"]))
        baseline_minutes = trips_estimate * float(truck_specs["load_time"] + truck_specs["dump_time"])
        self.apply_operational_prior("wildfire_debris", baseline_minutes)

        self.process = engine.env.process(self.work_loop())

    @override
    def needed_resources(self) -> list[ResourceType]:
        return [ResourceType.EXCAVATOR, ResourceType.TRUCK]

    @override
    def percent_remaining(self):
        return self.debris.level / self.initial_debris if self.initial_debris > 0 else 0

    @override
    def get_scale(self):
        return self.debris.level / 200  # Normalize to 200 cubic yards

    @property
    @override
    def render(self) -> DisasterRender:
        return {"color": "darkred", "label": "Fire Debris", "marker": "^"}

    def work_loop(self):
        """Clear debris - similar to landslide but with different timing."""
        truck_specs = ResourceType.TRUCK.specs
        dump_trips = []

        while self.active and self.debris.level > 0:
            # Get resources
            excavator = yield self.inventory[ResourceType.EXCAVATOR].get()
            truck = yield self.inventory[ResourceType.TRUCK].get()

            # Calculate work with capacity constraints
            num_excavators = len(self.roster[ResourceType.EXCAVATOR])
            efficiency = 1.0

            site_capacity = self.engine.scenario_config.get_site_capacity("wildfire_debris")
            efficiency = site_capacity.efficiency_curve(num_excavators)

            amount = min(truck_specs["capacity"] * efficiency, self.debris.level)
            yield self.debris.get(amount)

            def load_and_dump(debris_site, excavator, truck):
                # Loading time
                load_time = truck_specs["load_time"] / efficiency

                # Consume resources
                excavator.consume_fuel(load_time / 60)
                excavator.add_operating_cost(load_time / 60)

                yield debris_site.env.timeout(load_time)

                # Release excavator
                yield debris_site.inventory[ResourceType.EXCAVATOR].put(excavator)

                # Send truck to dump
                debris_site.dump_node.transfer_resource(truck)

            dump_trips.append(self.env.process(load_and_dump(self, excavator, truck)))

        # Wait for all trips
        yield self.env.all_of(dump_trips)

        if self.active:
            yield self.env.process(self.resolve())

    @classmethod
    def spawn_from_seasonal(
        cls,
        engine: "SimPySimulationEngine",
        dump_site: "DumpSite",
        location: tuple[float, float],
        size_range: tuple[int, int],
        season: "Season",  # noqa: ARG003
    ) -> "WildfireDebris":
        debris_amount = engine.rng.randint(size_range[0], size_range[1])
        disaster = cls(engine, debris_amount, dump_site, location=location)
        disaster.apply_closure_duration_prior("wildfire_debris", season)
        return disaster


class FloodEvent(Disaster):
    """
    Flooding event blocking roads.
    Requires: ASSESSMENT_VEHICLE (to assess) + EXCAVATOR/TRUCK (if repairs needed).
    Passive until water recedes, then may need cleanup.
    """

    one_hot_index: int = 3
    disaster_type: ClassVar[str] = "flood"
    seasonal_size_weights: ClassVar[dict[str, float]] = {"minor": 0.55, "moderate": 0.30, "major": 0.15}

    def __init__(
        self,
        engine: SimPySimulationEngine,
        duration_hours: float,
        cleanup_needed: bool,
        cleanup_amount: float,  # cubic yards if cleanup needed
        dump_node: DumpSite | None,
        location: tuple[float, float] | None = None,
    ):
        super().__init__(engine, location)

        self.duration_hours = duration_hours
        self.cleanup_needed = cleanup_needed
        self.cleanup_amount = cleanup_amount
        self.dump_node = dump_node
        self.duration_minutes = duration_hours * 60.0

        # Track progress
        self.assessed = False
        self.flood_resolved = False
        self.remaining_cleanup = cleanup_amount if cleanup_needed else 0
        self.initial_cleanup = cleanup_amount
        self.status_check_interval_minutes = engine.scenario_config.flood_status_check_interval_minutes
        self.apply_operational_prior("flood", self.duration_minutes)

        self.process = engine.env.process(self.work_loop())
        self.flood_process = engine.env.process(self.flood_duration_loop())

    @override
    def needed_resources(self) -> list[ResourceType]:
        if not self.assessed:
            return [ResourceType.ASSESSMENT_VEHICLE]
        elif self.cleanup_needed and self.remaining_cleanup > 0:
            return [ResourceType.EXCAVATOR, ResourceType.TRUCK]
        else:
            return []

    @override
    def percent_remaining(self):
        if not self.flood_resolved:
            return 1.0
        elif self.cleanup_needed:
            return self.remaining_cleanup / self.initial_cleanup if self.initial_cleanup > 0 else 0
        else:
            return 0.0

    @override
    def get_scale(self):
        if self.cleanup_needed:
            return self.cleanup_amount / 100
        return self.duration_hours / 72  # Normalize to 3-day flood

    @property
    @override
    def render(self) -> DisasterRender:
        if not self.flood_resolved:
            return {"color": "blue", "label": "Flooding", "marker": "v"}
        elif self.cleanup_needed and self.remaining_cleanup > 0:
            return {"color": "teal", "label": "Flood Cleanup", "marker": "v"}
        else:
            return {"color": "lightgreen", "label": "Flood Resolved", "marker": "v"}

    def flood_duration_loop(self):
        """Wait for flood to recede."""
        yield self.env.timeout(self.duration_minutes)
        self.flood_resolved = True

    def work_loop(self):
        """Handle flood response."""
        # Phase 1: Assessment (must happen during flood)
        if not self.assessed:
            # Wait for assessment vehicle
            vehicle = yield self.inventory[ResourceType.ASSESSMENT_VEHICLE].get()

            assess_low, assess_high = self.engine.scenario_config.flood_assessment_minutes_range
            assess_time = self.engine.rng.uniform(assess_low, assess_high)
            vehicle.consume_fuel(assess_time / 60)
            vehicle.add_operating_cost(assess_time / 60)

            yield self.env.timeout(assess_time)
            self.assessed = True

            # Return assessment vehicle
            yield self.inventory[ResourceType.ASSESSMENT_VEHICLE].put(vehicle)

        # Phase 2: Wait for flood to recede
        while not self.flood_resolved:
            yield self.env.timeout(self.status_check_interval_minutes)

        # Phase 3: Cleanup if needed
        if self.cleanup_needed and self.remaining_cleanup > 0 and self.dump_node:
            truck_specs = ResourceType.TRUCK.specs
            dump_trips = []

            while self.active and self.remaining_cleanup > 0:
                excavator = yield self.inventory[ResourceType.EXCAVATOR].get()
                truck = yield self.inventory[ResourceType.TRUCK].get()

                amount = min(truck_specs["capacity"], self.remaining_cleanup)
                self.remaining_cleanup -= amount

                def load_cleanup(flood_event, excavator, truck):
                    yield flood_event.env.timeout(truck_specs["load_time"])
                    yield flood_event.inventory[ResourceType.EXCAVATOR].put(excavator)
                    flood_event.dump_node.transfer_resource(truck)

                dump_trips.append(self.env.process(load_cleanup(self, excavator, truck)))

            yield self.env.all_of(dump_trips)

        if self.active:
            yield self.env.process(self.resolve())

    @classmethod
    def spawn_from_seasonal(
        cls,
        engine: "SimPySimulationEngine",
        dump_site: "DumpSite",
        location: tuple[float, float],
        size_range: tuple[int, int],
        season: "Season",
    ) -> "FloodEvent":
        low, high = size_range
        prior_cfg = engine.scenario_config.disaster_closure_lognormal_priors.get("flood", {})
        season_key = season.name.lower()
        prior = prior_cfg.get(season_key, prior_cfg.get("default"))

        if prior is not None:
            mu, sigma = prior
            duration_hours = max(low, min(high, int(engine.rng.lognormvariate(mu, sigma))))
        else:
            duration_hours = engine.rng.randint(low, high)
        duration_hours = max(1, duration_hours)

        cleanup_needed = engine.rng.random() < engine.scenario_config.flood_cleanup_probability
        amount_low, amount_high = engine.scenario_config.flood_cleanup_amount_range
        cleanup_amount = engine.rng.randint(amount_low, amount_high) if cleanup_needed else 0
        assessment_cfg = engine.scenario_config.num_assessment_vehicles
        assessment_capacity = assessment_cfg[1] if isinstance(assessment_cfg, tuple) else assessment_cfg

        disaster = cls(
            engine,
            duration_hours=duration_hours,
            cleanup_needed=cleanup_needed,
            cleanup_amount=cleanup_amount,
            dump_node=dump_site,
            location=location,
        )
        disaster.sampled_closure_duration_hours = float(duration_hours)
        if assessment_capacity <= 0:
            # If no assessment vehicles exist, skip assessment phase to avoid deadlock.
            disaster.assessed = True
        return disaster


# class StructureFire(Disaster):
#     """
#     Requires: FIRE_TRUCK (Parallel) and AMBULANCE (Parallel).
#     Firefighters fight fire, Medics help people. Independent.
#     """

#     one_hot_index: int = 1

#     def __init__(self, engine: SimPySimulationEngine, intensity: float, casualties: float, hospital_node: Hospital):
#         super().__init__(engine)
#         self.fire_intensity: Container = simpy.Container(engine.env, init=intensity)
#         self.casualties: Container = simpy.Container(engine.env, init=casualties)
#         self.initial_intensity: float = intensity
#         self.initial_casualties: float = casualties
#         self.hospital: Hospital = hospital_node

#         engine.env.process(self.firefight_loop())
#         engine.env.process(self.rescue_loop())

#     @override
#     def needed_resources(self):
#         return [ResourceType.FIRE_TRUCK, ResourceType.AMBULANCE]

#     @override
#     def percent_remaining(self):
#         return (self.fire_intensity.level + self.casualties.level) / (self.initial_intensity + self.initial_casualties)

#     @property
#     @override
#     def render(self) -> DisasterRender:
#         return {"color": "red", "label": "Structure Fire", "marker": "p"}

#     def check_done(self):
#         if self.fire_intensity.level <= 0 and self.casualties.level <= 0:
#             self.env.process(self.resolve())

#     def firefight_loop(self) -> Generator[FilterStoreGet | Timeout | ContainerGet | StorePut, Resource, None]:
#         while self.active and self.fire_intensity.level > 0:
#             ft: Resource = yield self.inventory[ResourceType.FIRE_TRUCK].get()

#             work_duration = 10
#             amount = ResourceType.FIRE_TRUCK.specs["work_rate"] * work_duration

#             yield self.env.timeout(work_duration)

#             yield self.fire_intensity.get(min(amount, self.fire_intensity.level))

#             yield self.inventory[ResourceType.FIRE_TRUCK].put(ft)

#         self.check_done()

#     def rescue_loop(self) -> Generator[FilterStoreGet | Timeout | ContainerGet, Resource, None]:
#         while self.active and self.casualties.level > 0:
#             amb: Resource = yield self.inventory[ResourceType.AMBULANCE].get()

#             yield self.env.timeout(5)  # Stabilize patient

#             yield self.casualties.get(1)
#             # Ambulance leaves to hospital
#             self.hospital.transfer_resource(amb)

#         self.check_done()


# class BuildingCollapse(Disaster):
#     """
#     Requires: EXCAVATOR + FIRE_TRUCK (Simultaneous) to find casualties.
#     AMBULANCE to remove casualty.
#     3 resource types needed.
#     """

#     one_hot_index: int = 2

#     def __init__(self, engine: SimPySimulationEngine, rubble_amount: float, hospital_node: Hospital):
#         super().__init__(engine)
#         self.rubble: Container = simpy.Container(engine.env, init=rubble_amount)
#         self.initial_rubble: float = rubble_amount
#         self.initial_casualties_trapped: int = math.ceil(rubble_amount / 10)
#         self.casualties_trapped: Container = simpy.Container(engine.env, init=self.initial_casualties_trapped)
#         self.hospital: Hospital = hospital_node

#         engine.env.process(self.clear_loop())
#         engine.env.process(self.rescue_loop())

#     @override
#     def needed_resources(self):
#         return [ResourceType.EXCAVATOR, ResourceType.FIRE_TRUCK, ResourceType.AMBULANCE]

#     @override
#     def percent_remaining(self):
#         return (self.rubble.level + self.casualties_trapped.level) / (
#             self.initial_rubble + self.initial_casualties_trapped
#         )

#     @property
#     @override
#     def render(self) -> DisasterRender:
#         return {"color": "gray", "label": "Building Collapse", "marker": "X"}

#     def check_done(self):
#         if self.rubble.level <= 0 and self.casualties_trapped.level <= 0:
#             self.env.process(self.resolve())

#     def clear_loop(self) -> Generator[FilterStoreGet | Timeout | ContainerGet | StorePut, Resource, None]:
#         while self.active and self.rubble.level > 0:
#             # Phase 1: Clear Rubble (Needs Excavator AND Fire Truck for safety)

#             exc: Resource = yield self.inventory[ResourceType.EXCAVATOR].get()
#             ft: Resource = yield self.inventory[ResourceType.FIRE_TRUCK].get()

#             yield self.env.timeout(15)  # Slow, careful work
#             amount = 10
#             yield self.rubble.get(min(amount, self.rubble.level))

#             yield self.inventory[ResourceType.EXCAVATOR].put(exc)
#             yield self.inventory[ResourceType.FIRE_TRUCK].put(ft)

#         self.check_done()

#     def rescue_loop(self) -> Generator[FilterStoreGet | Timeout | ContainerGet, Resource, None]:
#         while self.active and self.casualties_trapped.level > 0:
#             # Phase 2: Check for victims (Simplified: 1 unit of rubble cleared = chance of victim rescue available)

#             # We can only rescue if we have an ambulance
#             amb: Resource = yield self.inventory[ResourceType.AMBULANCE].get()
#             yield self.env.timeout(5)
#             yield self.casualties_trapped.get(1)
#             self.hospital.transfer_resource(amb)

#         self.check_done()
