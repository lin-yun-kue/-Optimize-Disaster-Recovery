from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Generator
from dataclasses import dataclass, field
from enum import Enum
from simpy.resources.store import FilterStoreGet
from simpy.resources.store import Store
from simpy.resources.container import Container
from simpy.events import Process
from simpy.resources.container import ContainerGet
from simpy.events import AllOf, Timeout
from typing import TYPE_CHECKING, Callable, ClassVar, TypedDict

import simpy
from simpy.events import AnyOf, Event
from simpy.resources.store import StorePut
from typing_extensions import override

from .gis_utils import travel_minutes_from_distance, CLATSOP_CENTER_UTM

if TYPE_CHECKING:
    from .engine import SimPySimulationEngine

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
        # Units: speed=mph, capacity=cubic yards, load_time=minutes, dump_time=minutes, hourly_operating_cost=dollars/hr, fuel_per_hour=gallons/hr (empty), fuel_per_hour_loaded=gallons/hr (loaded)
        {"speed": 23.4, "capacity": 10, "load_time": 10, "dump_time": 10, "hourly_operating_cost": 75.0, "fuel_per_hour": 5.5, "fuel_per_hour_loaded": 7.5, "fuel_capacity": 100.0},
        ResourceRender({"marker": "^", "color": "blue"}),
    )
    EXCAVATOR = (
        1,
        # Units: speed=mph, work_rate=cubic yards/hr, hourly_operating_cost=dollars/hr, fuel_per_hour=gallons/hr (work), fuel_per_hour_driving=gallons/hr (driving)
        {"speed": 11.6, "work_rate": 10, "hourly_operating_cost": 150.0, "fuel_per_hour": 8.0, "fuel_per_hour_driving": 5.5, "fuel_capacity": 80.0},
        ResourceRender({"marker": "P", "color": "orange"}),
    )

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
    home_depot: Depot | None = field(default=None, repr=False)  # Must return here to refuel
    fuel_level: float = field(default=100.0, repr=False)  # 0-100%
    fuel_capacity: float = field(default=100.0, repr=False)  # Gallons
    fuel_consumption_rate: float = field(default=5.0, repr=False)  # Gallons/hour (working)
    fuel_consumption_rate_driving: float = field(default=5.0, repr=False)  # Gallons/hour (driving)
    is_loaded: bool = field(default=False, repr=False)  # For trucks: True when carrying load
    hourly_operating_cost: float = field(default=75.0, repr=False)  # $/hour
    accumulated_cost: float = field(default=0.0, repr=False)  # Total cost accrued
    total_hours_operated: float = field(default=0.0, repr=False)  # Total operational hours

    def __post_init__(self):
        self.assigned_node = self.engine.idle_resources
        self.hourly_operating_cost = float(self.resource_type.specs.get("hourly_operating_cost", self.hourly_operating_cost))
        self.fuel_consumption_rate = float(self.resource_type.specs.get("fuel_per_hour", self.fuel_consumption_rate))
        self.fuel_consumption_rate_driving = float(self.resource_type.specs.get("fuel_per_hour_driving", self.fuel_consumption_rate))
        self.fuel_capacity = float(self.resource_type.specs.get("fuel_capacity", self.fuel_capacity))

    def consume_fuel(self, hours: float, is_driving: bool = False) -> float:
        """Consume fuel for given hours of operation. Returns fuel consumed.

        Args:
            hours: Hours of operation
            is_driving: If True, use driving fuel rate instead of working rate
        """
        if self.resource_type == ResourceType.TRUCK and self.is_loaded:
            rate = self.resource_type.specs.get("fuel_per_hour_loaded", self.fuel_consumption_rate)
        elif is_driving:
            rate = self.fuel_consumption_rate_driving
        else:
            rate = self.fuel_consumption_rate

        variance = self.engine.scenario_config.time_variance
        actual_consumption = rate * hours * self.engine.rng.uniform(1 - variance, 1 + variance)
        self.fuel_level = max(0, self.fuel_level - actual_consumption)
        return actual_consumption

    def add_operating_cost(self, hours: float) -> float:
        """Add operating cost for given hours. Returns cost added."""
        cost = self.hourly_operating_cost * hours
        self.accumulated_cost += cost
        self.total_hours_operated += hours
        self.engine.total_spent += cost
        if self.engine.track_metrics:
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

    def return_to_depot(self) -> None:
        """Send resource back to its home depot due to low fuel."""
        if self.home_depot is None:
            return

        current_node = self.assigned_node
        if current_node and current_node != self.home_depot:
            self.home_depot.transfer_resource(self)

    def check_fuel_and_return(self) -> bool:
        """Check fuel level, return to depot if needed. Returns True if returned."""
        if self.needs_refuel(threshold=20.0):
            self.return_to_depot()
            return True
        return False

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
        self.consume_fuel(minutes / 60.0, is_driving=True)
        self.add_operating_cost(minutes / 60.0)

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
    label: str

    def __init__(self, engine: SimPySimulationEngine, location: tuple[float, float], label: str = "Resource Node"):
        self.engine = engine
        self.env = engine.env
        self.id = engine.rng.randint(1, 10000)
        self.location = location
        self.label = label

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
        end_loc = self

        if start_loc == (0, 0):
            travel_time_minutes = 0.0
        else:
            # get_distance returns miles (GIS meters are converted by the engine)
            dist_miles = self.engine.get_distance(resource, end_loc)
            travel_time_minutes = travel_minutes_from_distance(dist_miles, specs["speed"])

        # Snap destination to road graph if in GIS mode
        if self.engine.gis_config is not None and self.engine.road_graph is not None:
            spatial_index = self.engine.gis_config.get_spatial_index()
            snapped_loc = spatial_index.get_nearest_node(end_loc.location[0], end_loc.location[1])
            resource.location = snapped_loc
        else:
            resource.location = end_loc.location
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
    hatch: str


class Disaster(ResourceNode, ABC):
    """
    Base class for all disasters.
    Handles the logistics: resources arriving, tracking active rosters, and returning home.
    """

    one_hot_index: int

    active: bool
    created_time: float
    _first_response_recorded: bool
    disaster_type: ClassVar[str] = "None"
    required_resources: ClassVar[tuple[ResourceType, ...]] = ()

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

        if self.engine.track_metrics:
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
        if self.engine.track_metrics:
            self.engine.metrics.record_disaster_resolved(
                disaster_id=self.id,
                sim_time=self.env.now,
                total_cost=0.0,
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
        location: tuple[float, float],
        size: int,
    ) -> "Disaster":
        raise NotImplementedError(f"{cls.__name__} must implement spawn_from_seasonal")


# ============================================================================
# MARK: Resource Nodes / Buildings
# ============================================================================


class IdleResources(ResourceNode):
    def __init__(self, engine: SimPySimulationEngine):
        super().__init__(engine, location=CLATSOP_CENTER_UTM)
        self.cooldown_until: dict[int, float] = {}
        self._cooldown_processes: dict[int, Process] = {}

    @property
    @override
    def render(self) -> ResourceNodeRender:
        return {"color": "cyan", "label": "Resource Node"}

    @override
    def drive_resource(self, resource: Resource, instant: bool = False):
        """Moves resource from the resource's current node to Idle Resource while driving location to its home depot."""
        resource.assigned_node = self
        self.roster[resource.resource_type].add(resource)

        yield self.env.timeout(0)

        self.mark_resource_available(resource)

        if not resource.home_depot:
            return

        specs = resource.resource_type.specs
        start_loc = resource.location
        end_loc = resource.home_depot

        if start_loc == (0, 0):
            travel_time_minutes = 0.0
        else:
            # get_distance returns miles (GIS meters are converted by the engine)
            dist_miles = self.engine.get_distance(resource, end_loc)
            travel_time_minutes = travel_minutes_from_distance(dist_miles, specs["speed"])

        # Snap destination to road graph if in GIS mode
        if self.engine.gis_config is not None and self.engine.road_graph is not None:
            spatial_index = self.engine.gis_config.get_spatial_index()
            snapped_loc = spatial_index.get_nearest_node(end_loc.location[0], end_loc.location[1])
            resource.location = snapped_loc
        else:
            resource.location = end_loc.location
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
        resource.drive_process = None

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
        need_trucks = False
        need_excavators = False
        for disaster in disasters:
            required = disaster.required_resources
            if ResourceType.TRUCK in required:
                need_trucks = True
            if ResourceType.EXCAVATOR in required:
                need_excavators = True
            if need_trucks and need_excavators:
                break

        if need_trucks and need_excavators:
            eligible_types = [ResourceType.TRUCK, ResourceType.EXCAVATOR]
        elif need_trucks:
            eligible_types = [ResourceType.TRUCK]
        elif need_excavators:
            eligible_types = [ResourceType.EXCAVATOR]
        else:
            eligible_types = [ResourceType.TRUCK, ResourceType.EXCAVATOR]

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
    def __init__(self, engine: SimPySimulationEngine, location: tuple[float, float], label: str = "Depot"):
        super().__init__(engine, location=location, label=label)

    @property
    @override
    def render(self) -> ResourceNodeRender:
        return {"color": "black", "label": self.label}

    @override
    def mark_resource_available(self, resource: Resource):
        resource.refuel()
        super().mark_resource_available(resource)

    @override
    def drive_resource(self, resource: Resource):
        self.engine.idle_resources.transfer_resource(resource, True)
        return super().drive_resource(resource)


class DumpSite(ResourceNode):
    """
    The Dump Site is a Node with capacity constraints.
    Trucks arrive here, wait for available dump slots, wait for 'dump_time',
    and then automatically return to idle resources.
    """

    def __init__(self, engine: SimPySimulationEngine, location: tuple[float, float], capacity: int = 2, label: str = "Dump Site"):
        super().__init__(engine, location=location, label=label)

        # Capacity management: simpy Store with fixed number of "slots"
        self.capacity: int = capacity
        self.dump_slots: Store = simpy.Store(engine.env, capacity)
        # Initialize with available slots
        for i in range(capacity):
            self.dump_slots.put(i)

        self.active_dumps: int = 0
        self.total_dumps_completed: int = 0

        self.process: Process = engine.env.process(self.dump_loop())

    @property
    @override
    def render(self) -> ResourceNodeRender:
        return {"color": "yellow", "label": self.label}

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

            # Mark truck as empty after dumping
            truck.is_loaded = False

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
        if self.items:
            return
        yield self._available_event

    def _sync_available_event(self) -> None:
        if self.items:
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
    required_resources: ClassVar[tuple[ResourceType, ResourceType]] = (ResourceType.EXCAVATOR, ResourceType.TRUCK)
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
        location: tuple[float, float] | None = None,
    ):
        super().__init__(engine, location)

        self.dirt = simpy.Container(engine.env, init=size)
        self.initial_size = size

        self.process = engine.env.process(self.work_loop())

    @override
    def percent_remaining(self):
        return self.dirt.level / self.initial_size

    @override
    def get_scale(self):
        size_ranges = self.engine.scenario_config.seasonal_spawn.get("landslide")
        max_size = 1.0
        if size_ranges is not None:
            for rng in size_ranges.size_range_by_season.values():
                max_size = max(max_size, float(rng[1]))
        return self.dirt.level / max_size

    @property
    @override
    def render(self) -> DisasterRender:
        return {"color": "brown", "label": "Landslide", "marker": "o", "hatch": "/"}

    def work_loop(self) -> Generator[FilterStoreGet | ContainerGet | AllOf | Process, Resource, None]:
        """The specific logic for clearing a landslide."""
        truck_specs = ResourceType.TRUCK.specs

        dump_trips: list[Process] = []

        while self.active and self.dirt.level > 0:
            # 1. Request resources
            excavator: Resource = yield self.inventory[ResourceType.EXCAVATOR].get()
            truck: Resource = yield self.inventory[ResourceType.TRUCK].get()

            if excavator.check_fuel_and_return():
                self.inventory[ResourceType.TRUCK].put(truck)
                continue
            if truck.check_fuel_and_return():
                self.inventory[ResourceType.EXCAVATOR].put(excavator)
                continue

            # 2. Perform Work
            amount = min(truck_specs["capacity"], self.dirt.level)
            yield self.dirt.get(amount)

            def load(ls: Landslide, excavator: Resource, truck: Resource):
                load_time = float(truck_specs["load_time"])
                excavator.record_operating_minutes(load_time)
                truck.record_operating_minutes(load_time)

                # Simulate Loading Time
                yield ls.env.timeout(load_time)

                # Mark truck as loaded before driving to dump
                truck.is_loaded = True

                # 3. Release Excavator immediately (stays on site)
                yield ls.inventory[ResourceType.EXCAVATOR].put(excavator)

                # 4. Send Truck to dump (Process handled in parallel so loop continues)
                dump_node = self.engine.get_nearest_node(truck, DumpSite)
                if dump_node is None:
                    raise RuntimeError("Failed to find dump node for landslide")

                dump_node.transfer_resource(truck)

            dump_trips.append(self.env.process(load(self, excavator, truck)))

        # Wait for all dump trips to finish
        yield self.env.all_of(dump_trips)

        # Once loop breaks, teardown
        yield self.env.process(self.resolve())

    @classmethod
    @override
    def spawn_from_seasonal(
        cls,
        engine: "SimPySimulationEngine",
        location: tuple[float, float],
        size: int,
    ) -> "Landslide":
        disaster = cls(engine, size, location=location)
        return disaster


class WildfireDebris(Disaster):
    """
    Wildfire debris blocking roads (fallen trees, etc.).
    Requires: EXCAVATOR + TRUCK (similar to landslide but different seasonality).
    """

    one_hot_index: int = 2
    disaster_type: ClassVar[str] = "wildfire_debris"
    required_resources: ClassVar[tuple[ResourceType, ResourceType]] = (ResourceType.EXCAVATOR, ResourceType.TRUCK)

    def __init__(
        self,
        engine: SimPySimulationEngine,
        debris_amount: float,  # cubic yards
        location: tuple[float, float] | None = None,
    ):
        super().__init__(engine, location)

        self.debris: Container = simpy.Container(engine.env, init=debris_amount)
        self.initial_debris: float = debris_amount
        self.max_debris: float = debris_amount * 5.0

        self.growth_rate: float = 50.0
        self.max_size = debris_amount * 5.0
        self.spawn_probability = 0.3
        self.spawn_radius = 5000.0
        self.spawn_size_factor = 0.4
        self.max_spawns = 5
        self.spawns_created = 0

        self.process: Process = engine.env.process(self.work_loop())
        self.growth_process: Process = engine.env.process(self.growth_loop())
        self.spawn_process: Process = engine.env.process(self.spawn_loop())

    @override
    def percent_remaining(self):
        return self.debris.level / self.initial_debris if self.initial_debris > 0 else 0

    @override
    def get_scale(self):
        size_ranges = self.engine.scenario_config.seasonal_spawn.get("wildfire_debris")
        max_size = 1.0
        if size_ranges is not None:
            for rng in size_ranges.size_range_by_season.values():
                max_size = max(max_size, float(rng[1]))
        return self.debris.level / max_size

    @property
    @override
    def render(self) -> DisasterRender:
        return {"color": "darkred", "label": "Fire Debris", "marker": "^", "hatch": "+"}

    def growth_loop(self) -> Generator[Timeout, None, None]:
        """Continuously grow debris over time until the fire is resolved."""
        while self.active:
            yield self.env.timeout(self.engine.rng.uniform(60.0, 60.0 * 3.0))
            if self.active and len(self.inventory[ResourceType.TRUCK].items) <= 0 and len(self.inventory[ResourceType.EXCAVATOR].items) <= 0 and self.debris.level < self.max_debris:
                growth_amount = self.growth_rate
                self.debris.put(growth_amount)

    def spawn_loop(self) -> Generator[Timeout, None, None]:
        """Spawn smaller fires nearby at random intervals."""
        while self.active and self.spawns_created < self.max_spawns:
            yield self.env.timeout(self.engine.rng.uniform(60.0, 60.0 * 6.0))
            if not self.active or self.spawns_created >= self.max_spawns:
                break

            if self.engine.rng.random() < self.spawn_probability:
                angle = self.engine.rng.uniform(0, 2 * math.pi)
                distance = self.engine.rng.uniform(self.spawn_radius / 2, self.spawn_radius)
                new_x = self.location[0] + distance * math.cos(angle)
                new_y = self.location[1] + distance * math.sin(angle)
                new_location = (new_x, new_y)

                new_size = self.initial_debris * self.spawn_size_factor
                new_fire = WildfireDebris(
                    self.engine,
                    new_size,
                    location=new_location,
                )
                new_fire.growth_rate = self.growth_rate * self.spawn_size_factor
                new_fire.spawn_probability = self.spawn_probability * 0.5
                new_fire.spawn_radius = self.spawn_radius * 1.2
                new_fire.spawn_size_factor = self.spawn_size_factor
                new_fire.max_spawns = 1

                self.engine.disaster_store.put(new_fire)
                self.engine.record_disaster_created_metrics(new_fire)
                self.spawns_created += 1

    def work_loop(self) -> Generator[FilterStoreGet | ContainerGet | AllOf | Process, Resource, None]:
        """Clear debris - similar to landslide but with different timing."""
        truck_specs = ResourceType.TRUCK.specs
        dump_trips: list[Process] = []

        while self.active and self.debris.level > 0:
            # Get resources
            excavator: Resource = yield self.inventory[ResourceType.EXCAVATOR].get()
            truck: Resource = yield self.inventory[ResourceType.TRUCK].get()

            if excavator.check_fuel_and_return():
                self.inventory[ResourceType.TRUCK].put(truck)
                continue
            if truck.check_fuel_and_return():
                self.inventory[ResourceType.EXCAVATOR].put(excavator)
                continue

            # Calculate work with capacity constraints
            num_excavators = len(self.roster[ResourceType.EXCAVATOR])
            efficiency = 1.0

            if num_excavators <= 0:
                efficiency = 0.0

            amount = min(truck_specs["capacity"] * efficiency, self.debris.level)
            yield self.debris.get(amount)

            def load_and_dump(debris_site: WildfireDebris, excavator: Resource, truck: Resource):
                # Loading time
                load_time = truck_specs["load_time"] / efficiency

                excavator.record_operating_minutes(load_time)
                truck.record_operating_minutes(load_time)

                yield debris_site.env.timeout(load_time)

                # Mark truck as loaded before driving to dump
                truck.is_loaded = True

                # Release excavator
                yield debris_site.inventory[ResourceType.EXCAVATOR].put(excavator)

                # Send truck to dump
                dump_node = self.engine.get_nearest_node(truck, DumpSite)
                if dump_node is None:
                    raise RuntimeError("Failed to find dump node for wildfire debris")
                dump_node.transfer_resource(truck)

            dump_trips.append(self.env.process(load_and_dump(self, excavator, truck)))

        # Wait for all trips
        yield self.env.all_of(dump_trips)

        if self.active:
            yield self.env.process(self.resolve())

    @classmethod
    @override
    def spawn_from_seasonal(
        cls,
        engine: "SimPySimulationEngine",
        location: tuple[float, float],
        size: int,
    ) -> "WildfireDebris":
        disaster = cls(engine, size, location=location)
        return disaster
