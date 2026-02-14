from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Generator
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, TypedDict

try:
    from typing import Never
except ImportError:
    from typing import NoReturn as Never

import simpy
from simpy.events import AllOf, AnyOf, Process
from simpy.resources.container import ContainerGet
from simpy.resources.store import FilterStoreGet, StorePut
from typing_extensions import override

if TYPE_CHECKING:
    from .engine import SimPySimulationEngine
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

    def consume_fuel(self, hours: float) -> float:
        """Consume fuel for given hours of operation. Returns fuel consumed."""
        # Apply time variance from config if available
        variance = 0.0
        if hasattr(self.engine, "scenario_config"):
            variance = getattr(self.engine.scenario_config, "time_variance", 0.0)

        actual_consumption = self.fuel_consumption_rate * hours * random.uniform(1 - variance, 1 + variance)
        self.fuel_level = max(0, self.fuel_level - actual_consumption)
        return actual_consumption

    def add_operating_cost(self, hours: float) -> float:
        """Add operating cost for given hours. Returns cost added."""
        cost = self.hourly_operating_cost * hours
        self.accumulated_cost += cost
        self.total_hours_operated += hours
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
            travel_time = 0
        else:
            # Use road graph distance if available, otherwise use Euclidean
            dist = self.engine.get_distance(resource, self)

            travel_time = dist / specs["speed"]

        # Snap destination to road graph if in GIS mode
        if self.engine.gis_config is not None and self.engine.road_graph is not None:
            spatial_index = self.engine.gis_config.get_spatial_index()
            snapped_loc = spatial_index.get_nearest_node(self.location[0], self.location[1])
            resource.location = snapped_loc
        else:
            resource.location = self.location
        resource.move_time = travel_time

        try:
            yield self.env.timeout(travel_time)
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

        super().mark_resource_available(resource)

    def resolve(self):
        self.active = False
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
        finished: dict[FilterStoreGet, Resource] = yield self.env.any_of(
            list(get_events.values())
        )  # pyright: ignore[reportAssignmentType]

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
            variance = 0.0
            if hasattr(self.engine, "scenario_config"):
                variance = getattr(self.engine.scenario_config, "time_variance", 0.0)

            actual_dump_time = dump_time * random.uniform(1 - variance, 1 + variance)

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
