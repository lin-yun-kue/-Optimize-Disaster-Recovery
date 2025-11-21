from __future__ import annotations
import simpy
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict
from typing import Dict, Set, List, Optional, Tuple, Any, TypedDict, TYPE_CHECKING
from abc import ABC, abstractmethod
import math

if TYPE_CHECKING:
    from engine import SimPySimulationEngine

# ============================================================================
# MARK: Configuration & Enums
# ============================================================================


class ResourceRender(TypedDict):
    """Rendering options for a resource."""

    marker: str
    color: str


class ResourceType(Enum):
    TRUCK = (
        auto(),
        {"speed": 2.0, "capacity": 10, "load_time": 10},
        ResourceRender({"marker": "^", "color": "blue"}),
    )
    EXCAVATOR = (
        auto(),
        {"speed": 1.0, "work_rate": 10},
        ResourceRender({"marker": "P", "color": "orange"}),
    )
    FIRE_TRUCK = (
        auto(),
        {"speed": 3.0, "work_rate": 5},
        ResourceRender({"marker": "p", "color": "red"}),
    )
    AMBULANCE = (
        auto(),
        {"speed": 4.0, "capacity": 1, "load_time": 5},
        ResourceRender({"marker": "X", "color": "skyblue"}),
    )

    specs: Dict[str, float]
    render: ResourceRender

    def __new__(cls, value: int, specs: Dict[str, float], render: ResourceRender):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.specs = specs
        obj.render = render
        return obj


class SimulationConfig:
    DROP_OFF_LOCATION = (75, 75)
    DEPOT_LOCATION = (0, 0)
    HOSPITAL_LOCATION = (-75, -75)

    # Ticks should be in minutes

    # Disaster Specifics
    LANDSLIDE_MIN_SIZE = 150
    LANDSLIDE_MAX_SIZE = 250
    NUM_STARTING_LANDSLIDES = 10
    NUM_LANDSLIDES = 15

    NUM_TRUCKS = 100
    NUM_EXCAVATORS = 15
    NUM_FIRE_TRUCKS = 4
    NUM_AMBULANCES = 5


# ============================================================================
# MARK: Base Resource
# ============================================================================


@dataclass
class Resource:
    id: int
    resource_type: ResourceType
    engine: SimPySimulationEngine

    assigned_node: ResourceNode = field(init=False)

    drive_process: Optional[simpy.Process] = field(default=None, repr=False)

    # Visualization State
    _location: tuple = field(default=SimulationConfig.DEPOT_LOCATION, repr=False)
    prev_location: tuple = SimulationConfig.DEPOT_LOCATION
    _move_time: float = field(default=0, repr=False)
    move_start_time: float = 0

    def __post_init__(self):
        self.assigned_node = self.engine.idle_resources

    @property
    def location(self) -> tuple:
        """The current location of the resource."""
        return self._location

    @location.setter
    def location(self, loc: tuple):
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

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other) -> bool:
        return isinstance(other, Resource) and self.id == other.id


# ============================================================================
# MARK: Base Resource Node
# ============================================================================


class ResourceNodeRender(TypedDict):
    """Rendering options for a resource node."""

    label: str
    color: str


class ResourceStore(simpy.FilterStore):
    items: List[Resource]


class ResourceNode(ABC):
    """
    A generic node in the graph.
    """

    def __init__(self, engine: SimPySimulationEngine, location: Tuple[float, float]):
        self.engine = engine
        self.env = engine.env
        self.id = random.randint(1, 10000)
        self.location = location

        # Physical Inventory: Resources currently ON SITE and AVAILABLE
        self.inventory: Dict[ResourceType, ResourceStore] = defaultdict(lambda: ResourceStore(self.env))

        # Administrative Roster: Resources ASSIGNED to this node, but not AVAILABLE
        # Could be driving or performing a task
        self.roster: Dict[ResourceType, Set[Resource]] = defaultdict(set)

    @property
    @abstractmethod
    def render(self) -> ResourceNodeRender:
        """Rendering options for a resource node."""
        raise NotImplementedError

    def mark_resource_available(self, resource: Resource):
        """Logic when a resource physically arrives."""
        self.inventory[resource.resource_type].put(resource)

    def transfer_resource(self, resource: Resource):
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
        dist = math.hypot(end_loc[0] - start_loc[0], end_loc[1] - start_loc[1])
        travel_time = dist / specs["speed"]

        resource.location = self.location
        resource.move_time = travel_time

        try:
            yield self.env.timeout(travel_time)
        except simpy.Interrupt as e:
            loc1 = resource.location
            loc2 = resource.prev_location

            time_frac = 1 if resource.move_time == 0 else (self.env.now - resource.move_start_time) / resource.move_time
            time_frac = max(0, time_frac)
            time_frac = min(1, time_frac)

            loc = (
                loc1[0] * time_frac + loc2[0] * (1 - time_frac),
                loc1[1] * time_frac + loc2[1] * (1 - time_frac),
            )

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


class Disaster(ResourceNode):
    """
    Base class for all disasters.
    Handles the logistics: resources arriving, tracking active rosters, and returning home.
    """

    def __init__(
        self,
        engine: SimPySimulationEngine,
    ):
        loc = (random.randint(-100, 100), random.randint(-100, 100))
        super().__init__(engine, loc)

        self.active = True

    @abstractmethod
    def needed_resources(self):
        """Returns a list of resources needed to resolve the disaster."""
        raise NotImplementedError

    @abstractmethod
    def percent_remaining(self):
        """Returns the percentage of the disaster that is still active."""
        raise NotImplementedError

    @property
    @abstractmethod
    def render(self) -> DisasterRender:
        """Rendering options for a disaster."""
        raise NotImplementedError

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
        for r_type, store in self.roster.items():
            while len(store) > 0:
                self.engine.idle_resources.transfer_resource(store.pop())


# ============================================================================
# MARK: Resource Nodes / Buildings
# ============================================================================


class IdleResources(ResourceNode):
    def __init__(self, engine: SimPySimulationEngine):
        super().__init__(engine, location=(0, -10))

    @property
    def render(self) -> ResourceNodeRender:
        return {"color": "cs", "label": "Resource Node"}

    def drive_resource(self, resource: Resource):
        """Moves resource from the resource's current node -> Target."""
        resource.assigned_node = self
        self.roster[resource.resource_type].add(resource)

        yield self.env.timeout(0)

        self.mark_resource_available(resource)

    def get_any_resource(self):
        """Returns a random resource from any of the stores."""
        # Since resources are in separate stores by type, we listen to all of them
        # and trigger when the first one becomes available.
        get_events = {rt: self.inventory[rt].get() for rt in ResourceType}
        finished = yield self.env.any_of(list(get_events.values()))

        winner_event = random.choice(list(finished.keys()))
        resource: Resource = winner_event.value

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
    def __init__(self, engine: SimPySimulationEngine):
        super().__init__(engine, location=SimulationConfig.DEPOT_LOCATION)

    @property
    def render(self) -> ResourceNodeRender:
        return {"color": "ks", "label": "Depot"}

    def drive_resource(self, resource: Resource):
        yield self.env.timeout(0)
        self.engine.idle_resources.transfer_resource(resource)


class Hospital(ResourceNode):
    def __init__(self, engine: SimPySimulationEngine):
        super().__init__(engine, location=SimulationConfig.HOSPITAL_LOCATION)
        engine.env.process(self.process_loop())

    @property
    def render(self) -> ResourceNodeRender:
        return {"color": "ms", "label": "Hospital"}

    def process_loop(self):
        while True:
            amb = yield self.inventory[ResourceType.AMBULANCE].get()
            yield self.env.timeout(10)  # Unloading patient
            self.engine.idle_resources.transfer_resource(amb)


class DumpSite(ResourceNode):
    """
    The Dump Site is a Node.
    Trucks arrive here, wait for 'dump_time', and then automatically
    return to their ASSIGNED node (the Landslide).
    """

    def __init__(self, engine: SimPySimulationEngine):
        super().__init__(engine, location=SimulationConfig.DROP_OFF_LOCATION)

        self.process = engine.env.process(self.dump_loop())

    @property
    def render(self) -> ResourceNodeRender:
        return {"color": "ys", "label": "Dump Site"}

    def dump_loop(self):
        truck_specs = ResourceType.TRUCK.specs

        while True:
            # 1. Get Resources from Inventory (Must be physically here)
            truck = yield self.inventory[ResourceType.TRUCK].get()

            # 2. Take dirt
            yield self.env.timeout(1)

            # 4. Send Truck to free space
            self.engine.idle_resources.transfer_resource(truck)


# ============================================================================
# MARK: Disasters
# ============================================================================


class DisasterStore(simpy.FilterStore):
    items: List[Disaster]

    def wait_for_any(self):
        """Yields until any item is available."""
        if self.items:
            return

        d = yield self.get()
        yield self.put(d)


class Landslide(Disaster):
    """
    Requires: EXCAVATOR + TRUCK.
    Work cannot happen without both.
    """

    def __init__(self, engine: SimPySimulationEngine, size: float, dump_node: DumpSite):
        super().__init__(engine)

        self.dirt = simpy.Container(engine.env, init=size)
        self.initial_size = size

        self.dump_node = dump_node

        self.process = engine.env.process(self.work_loop())

    def needed_resources(self):
        return [ResourceType.EXCAVATOR, ResourceType.TRUCK]

    def percent_remaining(self):
        return self.dirt.level / self.initial_size

    @property
    def render(self) -> DisasterRender:
        return {"color": "brown", "label": "Landslide", "marker": "o"}

    def work_loop(self):
        """The specific logic for clearing a landslide."""
        truck_specs = ResourceType.TRUCK.specs

        dump_trips = []

        while self.active and self.dirt.level > 0:
            # 1. Request resources

            excavator = yield self.inventory[ResourceType.EXCAVATOR].get()
            truck = yield self.inventory[ResourceType.TRUCK].get()

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


class StructureFire(Disaster):
    """
    Requires: FIRE_TRUCK (Parallel) and AMBULANCE (Parallel).
    Firefighters fight fire, Medics help people. Independent.
    """

    def __init__(self, engine: SimPySimulationEngine, intensity: float, casualties: float, hospital_node: Hospital):
        super().__init__(engine)
        self.fire_intensity = simpy.Container(engine.env, init=intensity)
        self.casualties = simpy.Container(engine.env, init=casualties)
        self.initial_intensity = intensity
        self.initial_casualties = casualties
        self.hospital = hospital_node

        engine.env.process(self.firefight_loop())
        engine.env.process(self.rescue_loop())

    def needed_resources(self):
        return [ResourceType.FIRE_TRUCK, ResourceType.AMBULANCE]

    def percent_remaining(self):
        return (self.fire_intensity.level + self.casualties.level) / (self.initial_intensity + self.initial_casualties)

    @property
    def render(self) -> DisasterRender:
        return {"color": "red", "label": "Structure Fire", "marker": "p"}

    def check_done(self):
        if self.fire_intensity.level <= 0 and self.casualties.level <= 0:
            self.env.process(self.resolve())

    def firefight_loop(self):
        while self.active and self.fire_intensity.level > 0:
            ft = yield self.inventory[ResourceType.FIRE_TRUCK].get()

            work_duration = 10
            amount = ResourceType.FIRE_TRUCK.specs["work_rate"] * work_duration

            yield self.env.timeout(work_duration)

            yield self.fire_intensity.get(min(amount, self.fire_intensity.level))

            yield self.inventory[ResourceType.FIRE_TRUCK].put(ft)

        self.check_done()

    def rescue_loop(self):
        while self.active and self.casualties.level > 0:
            amb = yield self.inventory[ResourceType.AMBULANCE].get()

            yield self.env.timeout(5)  # Stabilize patient

            yield self.casualties.get(1)
            # Ambulance leaves to hospital
            self.hospital.transfer_resource(amb)

        self.check_done()


class BuildingCollapse(Disaster):
    """
    Requires: EXCAVATOR + FIRE_TRUCK (Simultaneous) to find casualties.
    AMBULANCE to remove casualty.
    3 resource types needed.
    """

    def __init__(self, engine: SimPySimulationEngine, rubble_amount: float, hospital_node: Hospital):
        super().__init__(engine)
        self.rubble = simpy.Container(engine.env, init=rubble_amount)
        self.initial_rubble = rubble_amount
        self.initial_casualties_trapped = math.ceil(rubble_amount / 10)
        self.casualties_trapped = simpy.Container(engine.env, init=self.initial_casualties_trapped)
        self.hospital = hospital_node

        engine.env.process(self.clear_loop())
        event = engine.env.process(self.rescue_loop())

    def needed_resources(self):
        return [ResourceType.EXCAVATOR, ResourceType.FIRE_TRUCK, ResourceType.AMBULANCE]

    def percent_remaining(self):
        return (self.rubble.level + self.casualties_trapped.level) / (
            self.initial_rubble + self.initial_casualties_trapped
        )

    @property
    def render(self) -> DisasterRender:
        return {"color": "gray", "label": "Building Collapse", "marker": "X"}

    def check_done(self):
        if self.rubble.level <= 0 and self.casualties_trapped.level <= 0:
            self.env.process(self.resolve())

    def clear_loop(self):
        while self.active and self.rubble.level > 0:
            # Phase 1: Clear Rubble (Needs Excavator AND Fire Truck for safety)

            exc = yield self.inventory[ResourceType.EXCAVATOR].get()
            ft = yield self.inventory[ResourceType.FIRE_TRUCK].get()

            yield self.env.timeout(15)  # Slow, careful work
            amount = 10
            yield self.rubble.get(min(amount, self.rubble.level))

            yield self.inventory[ResourceType.EXCAVATOR].put(exc)
            yield self.inventory[ResourceType.FIRE_TRUCK].put(ft)

        self.check_done()

    def rescue_loop(self):
        while self.active and self.casualties_trapped.level > 0:
            # Phase 2: Check for victims (Simplified: 1 unit of rubble cleared = chance of victim rescue available)

            # We can only rescue if we have an ambulance
            amb = yield self.inventory[ResourceType.AMBULANCE].get()
            yield self.env.timeout(5)
            yield self.casualties_trapped.get(1)
            self.hospital.transfer_resource(amb)

        self.check_done()
