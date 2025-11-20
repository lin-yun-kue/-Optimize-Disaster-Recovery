from __future__ import annotations
import simpy
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict
from typing import Dict, Set, List, Optional, Tuple
import math

# ==========================================
# 1. Configuration & Enums
# ==========================================


class ResourceType(Enum):
    TRUCK = auto()
    EXCAVATOR = auto()
    # BULLDOZER = auto()


class SimulationConfig:
    DROP_OFF_LOCATION = (75, 75)
    DEPOT_LOCATION = (0, 0)

    # Ticks should be in minutes

    SPECS = {
        ResourceType.TRUCK: {
            "drive_speed": 2.0,  # Units per tick
            "capacity": 10,
            "load_time": 10,
            "dump_time": 5,
        },
        ResourceType.EXCAVATOR: {
            "drive_speed": 1.0,
            "work_rate": 10,  # Dirt per tick
        },
    }

    # Disaster Specifics
    LANDSLIDE_MIN_SIZE = 15
    LANDSLIDE_MAX_SIZE = 25
    NUM_STARTING_LANDSLIDES = 10
    NUM_LANDSLIDES = 15

    NUM_TRUCKS = 10
    NUM_EXCAVATORS = 2


@dataclass
class Resource:
    id: int
    resource_type: ResourceType
    env: simpy.Environment

    assigned_node: ResourceNode  # Who is my boss? (Roster)

    # Visualization State
    _location: tuple = field(default=SimulationConfig.DEPOT_LOCATION, repr=False)
    prev_location: tuple = SimulationConfig.DEPOT_LOCATION
    _move_time: float = field(default=0, repr=False)
    move_start_time: float = 0

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
        self.move_start_time = self.env.now

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other) -> bool:
        return isinstance(other, Resource) and self.id == other.id


# ==========================================
# 2. Base Classes
# ==========================================


class ResourceNode:
    """
    A generic node in the graph.
    """

    idle_resources: IdleResources
    disaster_store: DisasterStore

    def __init__(self, env: simpy.Environment, location: Tuple[float, float]):
        self.env = env
        self.id = random.randint(1, 1000)
        self.location = location

        # Physical Inventory: Resources currently ON SITE and AVAILABLE
        self.inventory: Dict[ResourceType, simpy.FilterStore] = defaultdict(lambda: simpy.FilterStore(env))

        # Administrative Roster: Resources ASSIGNED to this node, but not AVAILABLE
        # Could be driving or performing a task
        self.roster: Dict[ResourceType, Set[Resource]] = defaultdict(set)

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
        resource.env.process(self.drive_resource(resource))

    def drive_resource(self, resource: Resource):
        """
        Moves resource from the resource's current node -> this node.
        This part handles the actual driving of the resource.
        """
        resource.assigned_node = self
        self.roster[resource.resource_type].add(resource)

        specs = SimulationConfig.SPECS[resource.resource_type]
        dist = math.hypot(resource.location[0] - self.location[0], resource.location[1] - self.location[1])
        travel_time = dist / specs["drive_speed"]

        resource.location = self.location
        resource.move_time = travel_time

        yield resource.env.timeout(travel_time)

        self.mark_resource_available(resource)


class Disaster(ResourceNode):
    """
    Base class for all disasters.
    Handles the logistics: resources arriving, tracking active rosters, and returning home.
    """

    def __init__(
        self,
        env: simpy.Environment,
    ):
        loc = (random.randint(-100, 100), random.randint(-100, 100))
        super().__init__(env, loc)

    def is_resolved(self) -> bool:
        """Subclasses must implement this to tell us when the disaster is over."""
        raise NotImplementedError

    def mark_resource_available(self, resource: Resource):
        """Logic for when a resource actually reaches the coordinates."""
        # Check if the job is already done while they were driving
        if self.is_resolved():
            self.idle_resources.transfer_resource(resource)
            return

        super().mark_resource_available(resource)

    def instant_teardown(self):
        """Remove disaster from the global disaster store."""
        # Remove self from the global disaster store
        yield self.disaster_store.get(filter=lambda x: x == self)

    def teardown(self):
        """Clean up all resources when disaster is resolved."""
        # Drain all stores and send everyone home
        # We loop until roster is empty to catch people currently driving to us
        while any(len(s) > 0 for s in self.roster.values()):

            for r_type, store in self.inventory.items():
                if len(store.items) > 0:
                    res = yield store.get()
                    self.idle_resources.transfer_resource(res)

            # If people are in the roster but not in the store, they are driving/working.
            # Wait a tick and check again.
            # There is probably a better way to do this, but idk
            if any(len(s) > 0 for s in self.roster.values()):
                yield self.env.timeout(1)


# ==========================================
# 3. Concrete Implementations
# ==========================================


class IdleResources(ResourceNode):
    def __init__(self, env: simpy.Environment):
        super().__init__(env, location=(0, 0))

    def drive_resource(self, resource: Resource):
        """Moves resource from the resource's current node -> Target."""
        resource.assigned_node = self
        self.roster[resource.resource_type].add(resource)

        yield resource.env.timeout(0)

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
    def __init__(self, env):
        super().__init__(env, location=SimulationConfig.DEPOT_LOCATION)

    def drive_resource(self, resource: Resource):
        yield self.env.timeout(0)
        self.idle_resources.transfer_resource(resource)


class Landslide(Disaster):
    def __init__(self, env, size, dump_node: DumpSite):
        loc = (random.randint(-100, 100), random.randint(-100, 100))
        super().__init__(env)

        self.dirt = simpy.Container(env, init=size)
        self.initial_size = size

        self.dump_node = dump_node

        self.process = env.process(self.work_loop())

    def is_resolved(self):
        resolved = self.dirt.level <= 0
        if resolved:
            self.env.process(self.instant_teardown())
        return resolved

    def work_loop(self):
        """The specific logic for clearing a landslide."""
        truck_specs = SimulationConfig.SPECS[ResourceType.TRUCK]

        dump_trips = []

        while not self.is_resolved():
            # 1. Request resources using the generic stores

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
        yield self.env.process(self.teardown())


class DumpSite(ResourceNode):
    """
    The Dump Site is a Node.
    Trucks arrive here, wait for 'dump_time', and then automatically
    return to their ASSIGNED node (the Landslide).
    """

    def __init__(self, env):
        super().__init__(env, location=SimulationConfig.DROP_OFF_LOCATION)

        self.process = env.process(self.dump_loop())

    def dump_loop(self):
        truck_specs = SimulationConfig.SPECS[ResourceType.TRUCK]

        while True:
            # 1. Get Resources from Inventory (Must be physically here)
            truck = yield self.inventory[ResourceType.TRUCK].get()

            # 2. Take dirt
            yield self.env.timeout(truck_specs["load_time"])

            # 4. Send Truck to free space
            self.idle_resources.transfer_resource(truck)


class DisasterStore(simpy.FilterStore):

    def wait_for_any(self):
        """Yields until any item is available."""
        if self.items:
            return

        d = yield self.get()
        yield self.put(d)
