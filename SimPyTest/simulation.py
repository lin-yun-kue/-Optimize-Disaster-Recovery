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
from simpy.events import AllOf, AnyOf, Event, Process
from simpy.resources.container import ContainerGet
from simpy.resources.store import FilterStoreGet, StorePut
from typing_extensions import override

from SimPyTest.real_world_params import travel_minutes_from_distance

if TYPE_CHECKING:
    from SimPyTest.scenario_types import ScenarioConfig
    from .engine import SimPySimulationEngine, SimulationRNG
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
        {"speed": 23.4, "capacity": 10, "load_time": 10, "dump_time": 10, "hourly_operating_cost": 75.0, "fuel_per_hour": 5.0},
        ResourceRender({"marker": "^", "color": "blue"}),
    )
    EXCAVATOR = (
        1,
        {"speed": 11.6, "work_rate": 10, "hourly_operating_cost": 150.0, "fuel_per_hour": 8.0},
        ResourceRender({"marker": "P", "color": "orange"}),
    )
    SNOWPLOW = (
        2,
        {"speed": 15.0, "work_rate": 5, "hourly_operating_cost": 100.0, "fuel_per_hour": 6.0},
        ResourceRender({"marker": "s", "color": "cyan"}),
    )
    ASSESSMENT_VEHICLE = (
        3,
        {"speed": 35.0, "work_rate": 1, "hourly_operating_cost": 60.0, "fuel_per_hour": 3.5},
        ResourceRender({"marker": "D", "color": "purple"}),
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
        self.hourly_operating_cost = float(self.resource_type.specs.get("hourly_operating_cost", self.hourly_operating_cost))
        self.fuel_consumption_rate = float(self.resource_type.specs.get("fuel_per_hour", self.fuel_consumption_rate))

    def consume_fuel(self, hours: float) -> float:
        """Consume fuel for given hours of operation. Returns fuel consumed."""
        # Apply time variance from config if available
        variance = self.engine.scenario_config.operational_priors.time_variance

        actual_consumption = self.fuel_consumption_rate * hours * self.engine.rng.uniform(1 - variance, 1 + variance)
        self.fuel_level = max(0, self.fuel_level - actual_consumption)
        return actual_consumption

    def add_operating_cost(self, hours: float) -> float:
        """Add operating cost for given hours. Returns cost added."""
        cost = self.hourly_operating_cost * hours
        self.accumulated_cost += cost
        self.total_hours_operated += hours

        # Cost is always tracked in core mode.
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

    def record_operating_minutes(self, minutes: float) -> None:
        if minutes <= 0:
            return
        hours = minutes / 60.0
        self.consume_fuel(hours)
        self.add_operating_cost(hours)

    def record_drive_minutes(self, minutes: float) -> None:
        if minutes <= 0:
            return
        self.engine.total_drive_time += minutes
        self.record_operating_minutes(minutes)

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
            elapsed_minutes = max(0.0, self.env.now - resource.move_start_time)
            resource.record_drive_minutes(elapsed_minutes)
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

        resource.record_drive_minutes(travel_time_minutes)
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
        self.vulnerability_index: float = 1.0
        self.detection_delay_minutes: float = 0.0
        self.assessment_minutes_estimate: float = 0.0
        self.admin_reopen_delay_minutes: float = 0.0
        self.estimated_total_closure_minutes: float = 0.0
        self.sampled_closure_duration_hours: float = 0.0
        self.dispatch_delay_applied: bool = False
        self.start_timestamp_calendar: str | None = None
        self.start_timestamp_calendar = self.engine.calendar.current_date.isoformat()

    def apply_operational_prior(self, prior_key: str, baseline_closure_minutes: float) -> None:
        """Attach disaster timing/category priors from scenario configuration."""
        priors = self.engine.scenario_config.operational_priors.disaster_operational_priors
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
            baseline_closure_minutes + self.detection_delay_minutes + self.assessment_minutes_estimate + self.admin_reopen_delay_minutes,
        )

    def apply_closure_duration_prior(
        self,
        prior_key: str,
        min_hours: float = 0.5,
        max_hours: float = 24.0 * 21,
    ) -> float:
        """Sample closure-duration metadata from simple per-disaster minute ranges."""
        minutes_low, minutes_high = self.engine.scenario_config.operational_priors.closure_minutes_range_by_disaster.get(
            prior_key,
            (min_hours * 60.0, max_hours * 60.0),
        )
        sampled_minutes = self.engine.rng.uniform(float(minutes_low), float(minutes_high))
        sampled = max(min_hours, min(max_hours, sampled_minutes / 60.0))
        self.sampled_closure_duration_hours = sampled
        self.estimated_total_closure_minutes = max(self.estimated_total_closure_minutes, self.sampled_closure_duration_hours * 60.0)
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
        self.cooldown_until: dict[int, float] = {}
        self._cooldown_processes: dict[int, Process] = {}

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

    @override
    def mark_resource_available(self, resource: Resource):
        cooldown_end = self.cooldown_until.get(resource.id)
        if cooldown_end is not None and cooldown_end > self.env.now:
            return

        self.cooldown_until.pop(resource.id, None)
        self._cooldown_processes.pop(resource.id, None)
        super().mark_resource_available(resource)

    def hold_resource(self, resource: Resource, duration_minutes: float) -> None:
        hold_duration = max(0.0, float(duration_minutes))
        cooldown_end = self.env.now + hold_duration
        self.cooldown_until[resource.id] = cooldown_end

        existing = self._cooldown_processes.get(resource.id)
        if existing is not None and not existing.triggered:
            existing.interrupt("hold_reset")

        if resource.assigned_node and resource in resource.assigned_node.roster[resource.resource_type]:
            resource.assigned_node.roster[resource.resource_type].remove(resource)

        resource.assigned_node = self
        self.roster[resource.resource_type].add(resource)
        self._cooldown_processes[resource.id] = self.env.process(self._release_after_cooldown(resource, cooldown_end))

    def _release_after_cooldown(self, resource: Resource, cooldown_end: float):
        while True:
            remaining = cooldown_end - self.env.now
            if remaining <= 0:
                break
            try:
                yield self.env.timeout(remaining)
            except simpy.Interrupt:
                cooldown_end = self.cooldown_until.get(resource.id, self.env.now)
                continue

        if self.cooldown_until.get(resource.id) != cooldown_end:
            return
        self.mark_resource_available(resource)

    def count_held_resources(self, resource_type: ResourceType | None = None) -> int:
        held_ids = {resource_id for resource_id, end_time in self.cooldown_until.items() if end_time > self.env.now}
        if resource_type is None:
            return len(held_ids)

        count = 0
        for resource in self.roster[resource_type]:
            if resource.id in held_ids:
                count += 1
        return count

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

    def get_resource_for_disasters(self, disasters: list[Disaster]) -> Generator[AnyOf, Resource, Resource]:
        eligible_types = [
            resource_type
            for resource_type in ResourceType
            if any(resource_type in disaster.needed_resources() for disaster in disasters)
        ]
        if not eligible_types:
            eligible_types = list(ResourceType)

        get_events: dict[ResourceType, FilterStoreGet] = {resource_type: self.inventory[resource_type].get() for resource_type in eligible_types}
        finished: dict[FilterStoreGet, Resource] = yield self.env.any_of(list(get_events.values()))  # pyright: ignore[reportAssignmentType]

        winner_event = self.engine.rng.choice(list(finished.keys()))
        resource = finished[winner_event]

        for resource_type, event in get_events.items():
            if event == winner_event:
                continue

            if event.triggered:
                self.inventory[resource_type].put(event.value)
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
            variance = self.engine.scenario_config.operational_priors.time_variance

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
    _available_event: Event

    def __init__(self, env: simpy.Environment):
        super().__init__(env)
        self._available_event = env.event()

    def wait_for_any(self) -> Generator[Event, object, None]:
        """Yields until any item is available."""
        if any(disaster.needed_resources() for disaster in self.items):
            return
        yield self._available_event

    def _sync_available_event(self) -> None:
        if any(disaster.needed_resources() for disaster in self.items):
            if not self._available_event.triggered:
                self._available_event.succeed()
            return
        if self._available_event.triggered:
            self._available_event = self._env.event()

    def refresh_available_event(self) -> None:
        self._sync_available_event()

    @override
    def put(self, item: Disaster) -> StorePut:
        event = super().put(item)
        if event.triggered:
            self._sync_available_event()
        else:
            event.callbacks.append(lambda _event: self._sync_available_event())
        return event

    @override
    def get(self, filter: Callable[[Disaster], bool] = lambda item: True) -> FilterStoreGet:
        event = super().get(filter)
        if event.triggered:
            self._sync_available_event()
        else:
            event.callbacks.append(lambda _event: self._sync_available_event())
        return event

    if TYPE_CHECKING:
        pass


class Landslide(Disaster):
    """
    Requires: EXCAVATOR + TRUCK.
    Work cannot happen without both.
    """

    one_hot_index: int = 0
    disaster_type: ClassVar[str] = "landslide"
    seasonal_size_weights: ClassVar[dict[str, float]] = {"small": 0.50, "medium": 0.30, "large": 0.15, "major": 0.05}
    size_bands: ClassVar[dict[str, tuple[int, int]]] = {
        "small": (0, 400),
        "medium": (401, 900),
        "large": (901, 2999),
        "major": (3000, 5000),
    }

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
        size_ranges = self.engine.scenario_config.seasonal_spawn.disasters.get("landslide")
        max_size = 1.0
        if size_ranges is not None:
            for rng in size_ranges.size_range_by_season.values():
                max_size = max(max_size, float(rng[1]))
        return self.dirt.level / max_size

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
                load_time = float(truck_specs["load_time"])
                excavator.record_operating_minutes(load_time)
                truck.record_operating_minutes(load_time)

                # Simulate Loading Time
                yield ls.env.timeout(load_time)

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
        size = cls.sample_size(engine, size_range)
        disaster = cls(engine, size, dump_site, location=location)
        disaster.apply_closure_duration_prior("landslide")
        return disaster

    @classmethod
    def sample_size(cls, engine: "SimPySimulationEngine", size_range: tuple[int, int]) -> int:
        low, high = size_range
        if high <= low:
            return low

        candidates: list[tuple[tuple[int, int], float]] = []
        for label in ("small", "medium", "large", "major"):
            band_low, band_high = cls.size_bands[label]
            bucket_low = max(low, band_low)
            bucket_high = min(high, band_high)
            if bucket_low <= bucket_high:
                candidates.append(((bucket_low, bucket_high), cls.seasonal_size_weights[label]))

        if not candidates:
            return engine.rng.randint(low, high)

        total_weight = sum(weight for _, weight in candidates)
        roll = engine.rng.random() * total_weight
        cumulative = 0.0
        selected_low, selected_high = candidates[0][0]
        for (bucket_low, bucket_high), weight in candidates:
            cumulative += weight
            if roll <= cumulative:
                selected_low, selected_high = bucket_low, bucket_high
                break

        return engine.rng.randint(selected_low, selected_high)


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
        self.route_prep_complete = False
        self.compaction_factor = engine.rng.uniform(1.1, 1.8)
        self.initial_route_prep_minutes = max(20.0, road_miles_affected * 8.0)
        self.remaining_work = severity_hours * road_miles_affected * self.compaction_factor
        self.initial_work = self.remaining_work

        self.severity_minutes = severity_hours * 60.0
        self.plow_interval_minutes = engine.scenario_config.operational_priors.snow_work_interval_minutes
        baseline_minutes = self.initial_route_prep_minutes + (self.remaining_work / max(1.0, ResourceType.SNOWPLOW.specs["work_rate"])) * 60.0
        self.apply_operational_prior("snow", baseline_minutes)

        self.process = engine.env.process(self.work_loop())

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
        """Inspect and clear snow through repeated plow passes."""
        while self.active and self.remaining_work > 0:
            plow = yield self.inventory[ResourceType.SNOWPLOW].get()

            if not self.route_prep_complete:
                prep_hours = self.initial_route_prep_minutes / 60.0
                plow.consume_fuel(prep_hours)
                plow.add_operating_cost(prep_hours)
                yield self.env.timeout(self.initial_route_prep_minutes)
                self.route_prep_complete = True
                self.remaining_work = max(0.0, self.remaining_work - self.road_miles_affected)

            if self.remaining_work > 0:
                work_per_hour = self._plow_work_rate()
                work_hours = min(self.plow_interval_minutes / 60.0, self.remaining_work / work_per_hour) if work_per_hour > 0 else 0.0
                if work_hours > 0:
                    plow.consume_fuel(work_hours)
                    plow.add_operating_cost(work_hours)
                    yield self.env.timeout(work_hours * 60.0)
                    self.remaining_work = max(0.0, self.remaining_work - work_per_hour * work_hours)

            yield self.inventory[ResourceType.SNOWPLOW].put(plow)

        if self.active:
            yield self.env.process(self.resolve())

    def _plow_work_rate(self) -> float:
        base_rate = float(ResourceType.SNOWPLOW.specs["work_rate"])
        temperature = float(self.engine.calendar.weather_state["temperature"])
        temperature_bonus = 0.0
        if temperature >= 10.0:
            temperature_bonus = 0.75
        elif temperature >= 5.0:
            temperature_bonus = 0.35
        elif temperature <= -5.0:
            temperature_bonus = -0.2

        return max(1.0, base_rate * (1.0 + temperature_bonus))

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
        severity_hours = engine.rng.randint(low, high)
        miles_low, miles_high = engine.scenario_config.operational_priors.snow_road_miles_range
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

            if num_excavators <= 0:
                efficiency = 0.0

            amount = min(truck_specs["capacity"] * efficiency, self.debris.level)
            yield self.debris.get(amount)

            def load_and_dump(debris_site, excavator, truck):
                # Loading time
                load_time = truck_specs["load_time"] / efficiency

                excavator.record_operating_minutes(load_time)
                truck.record_operating_minutes(load_time)

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
        disaster.apply_closure_duration_prior("wildfire_debris")
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
    STANDING_WATER_ONLY: ClassVar[str] = "standing_water_only"
    DEBRIS_CLEANUP: ClassVar[str] = "debris_cleanup"
    WASHOUT_REPAIR: ClassVar[str] = "washout_repair"

    def __init__(
        self,
        engine: SimPySimulationEngine,
        duration_hours: float,
        location: tuple[float, float] | None = None,
    ):
        super().__init__(engine, location)

        self.duration_hours = duration_hours
        self.duration_minutes = duration_hours * 60.0

        # Track progress
        self.assessed = False
        self.water_receded = False
        self.status_check_interval_minutes = engine.scenario_config.operational_priors.flood_status_check_interval_minutes
        self.work_interval_minutes = engine.scenario_config.operational_priors.flood_work_interval_minutes
        self.post_recede_job_type = self.STANDING_WATER_ONLY
        self.initial_work = 0.0
        self.remaining_work = 0.0
        self.reopen_minutes = 0.0
        self.apply_operational_prior("flood", self.duration_minutes)

        self.process = engine.env.process(self.work_loop())

    @override
    def needed_resources(self) -> list[ResourceType]:
        if not self.assessed:
            return [ResourceType.ASSESSMENT_VEHICLE]
        if self.water_receded and self.remaining_work > 0:
            return [ResourceType.EXCAVATOR, ResourceType.TRUCK]
        return []

    @override
    def percent_remaining(self):
        if self.water_receded and self.initial_work > 0:
            return self.remaining_work / self.initial_work
        if self.water_receded:
            return 0.0
        remaining_ratio = max(0.0, min(1.0, (self.duration_minutes - (self.env.now - self.created_time)) / max(self.duration_minutes, 1.0)))
        if self.initial_work > 0:
            return 0.5 + 0.5 * remaining_ratio
        return remaining_ratio

    @override
    def get_scale(self):
        return self.duration_hours / 72  # Normalize to 3-day flood

    @property
    @override
    def render(self) -> DisasterRender:
        if not self.water_receded:
            return {"color": "blue", "label": "Flooding", "marker": "v"}
        return {"color": "lightgreen", "label": "Flood Resolved", "marker": "v"}

    def work_loop(self):
        """Handle flood assessment, recession, and post-flood work."""
        if not self.assessed:
            vehicle = yield self.inventory[ResourceType.ASSESSMENT_VEHICLE].get()

            assess_time = self._assessment_cycle_minutes()
            assess_hours = assess_time / 60.0
            vehicle.consume_fuel(assess_hours)
            vehicle.add_operating_cost(assess_hours)
            yield self.env.timeout(assess_time)
            self.assessed = True
            self._plan_post_recede_work()
            self.engine.disaster_store.refresh_available_event()
            yield self.inventory[ResourceType.ASSESSMENT_VEHICLE].put(vehicle)

        while not self.water_receded:
            yield self.env.timeout(self.status_check_interval_minutes)
            self.water_receded = self.env.now - self.created_time >= self.duration_minutes
            self.engine.disaster_store.refresh_available_event()

        while self.active and self.remaining_work > 0:
            excavator = yield self.inventory[ResourceType.EXCAVATOR].get()
            truck = yield self.inventory[ResourceType.TRUCK].get()

            work_hours = self.work_interval_minutes / 60.0
            work_rate = min(
                float(ResourceType.EXCAVATOR.specs["work_rate"]),
                float(ResourceType.TRUCK.specs["capacity"]) / max(work_hours, 1e-6),
            ) * self._work_rate_multiplier()
            work_amount = min(self.remaining_work, work_rate * work_hours)

            if work_amount > 0:
                excavator.consume_fuel(work_hours)
                excavator.add_operating_cost(work_hours)
                truck.consume_fuel(work_hours)
                truck.add_operating_cost(work_hours)
                yield self.env.timeout(self.work_interval_minutes)
                self.remaining_work -= work_amount
                if self.remaining_work <= 0:
                    self.engine.disaster_store.refresh_available_event()

            yield self.inventory[ResourceType.EXCAVATOR].put(excavator)
            yield self.inventory[ResourceType.TRUCK].put(truck)

        if self.reopen_minutes > 0:
            yield self.env.timeout(self.reopen_minutes)

        if self.active:
            yield self.env.process(self.resolve())

    def _assessment_cycle_minutes(self) -> float:
        assess_low, assess_high = self.engine.scenario_config.operational_priors.flood_assessment_minutes_range
        return self.engine.rng.uniform(assess_low, assess_high)

    def _pick_post_recede_job_type(self) -> str:
        weights = dict(self.engine.scenario_config.operational_priors.flood_post_recede_job_probabilities)
        if self.severity_index >= 0.8:
            weights[self.WASHOUT_REPAIR] = weights.get(self.WASHOUT_REPAIR, 0.0) + 0.35
            weights[self.STANDING_WATER_ONLY] = max(0.0, weights.get(self.STANDING_WATER_ONLY, 0.0) - 0.15)
        elif self.severity_index <= 0.55:
            weights[self.STANDING_WATER_ONLY] = weights.get(self.STANDING_WATER_ONLY, 0.0) + 0.25
            weights[self.WASHOUT_REPAIR] = max(0.0, weights.get(self.WASHOUT_REPAIR, 0.0) - 0.1)

        total_weight = sum(weight for weight in weights.values() if weight > 0)
        if total_weight <= 0:
            return self.STANDING_WATER_ONLY

        roll = self.engine.rng.random() * total_weight
        cumulative = 0.0
        for job_type, weight in weights.items():
            if weight <= 0:
                continue
            cumulative += weight
            if roll <= cumulative:
                return job_type
        return self.STANDING_WATER_ONLY

    def _plan_post_recede_work(self) -> None:
        priors = self.engine.scenario_config.operational_priors
        self.post_recede_job_type = self._pick_post_recede_job_type()
        reopen_low, reopen_high = priors.flood_reopen_minutes_range
        self.reopen_minutes = self.engine.rng.uniform(reopen_low, reopen_high)

        if self.post_recede_job_type == self.DEBRIS_CLEANUP:
            low, high = priors.flood_debris_work_range
            self.initial_work = self.engine.rng.uniform(low, high)
        elif self.post_recede_job_type == self.WASHOUT_REPAIR:
            low, high = priors.flood_washout_work_range
            self.initial_work = self.engine.rng.uniform(low, high)
        else:
            self.initial_work = 0.0

        self.remaining_work = self.initial_work

    def _work_rate_multiplier(self) -> float:
        priors = self.engine.scenario_config.operational_priors
        if self.post_recede_job_type == self.WASHOUT_REPAIR:
            return priors.flood_washout_work_rate_multiplier
        return priors.flood_debris_work_rate_multiplier

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
        duration_hours = engine.rng.randint(low, high)
        duration_hours = max(1, duration_hours)

        disaster = cls(
            engine,
            duration_hours=duration_hours,
            location=location,
        )
        disaster.sampled_closure_duration_hours = float(duration_hours)
        assessment_capacity = engine.scenario_config.resource_counts.assessment_vehicles
        if isinstance(assessment_capacity, tuple):
            assessment_capacity = assessment_capacity[1]
        if assessment_capacity <= 0:
            # If no assessment vehicles exist, skip assessment phase to avoid deadlock.
            disaster.assessed = True
            disaster._plan_post_recede_work()
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
