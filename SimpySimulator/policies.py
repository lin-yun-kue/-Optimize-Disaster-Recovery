import simpy
import random
from typing import List, Callable, TYPE_CHECKING
import math
from simulation import *

# ============================================================================
# MARK: Utils
# ============================================================================


def get_dist(r: Resource, node: ResourceNode):
    return math.hypot(r.location[0] - node.location[0], r.location[1] - node.location[1])


def is_useful(resource: Resource, disaster: Disaster):
    """Determines if a resource is actually needed at a disaster site."""
    return resource.resource_type in disaster.needed_resources()


def get_effective_dirt(landslide):
    """
    Calculates dirt remaining after accounting for all trucks currently
    en-route or working at the site.
    """
    current_dirt = landslide.dirt.level

    # Count trucks in the roster (both on-site and driving)
    num_trucks = len(landslide.roster[ResourceType.TRUCK])
    truck_capacity = ResourceType.TRUCK.specs["capacity"]

    potential_removal = num_trucks * truck_capacity
    return current_dirt - potential_removal


def get_severity(disaster: Disaster) -> float:
    """Returns a normalized 'size' of the disaster."""
    if isinstance(disaster, Landslide):
        return disaster.dirt.level
    elif isinstance(disaster, StructureFire):
        return disaster.fire_intensity.level + (disaster.casualties.level * 10)
    elif isinstance(disaster, BuildingCollapse):
        return disaster.rubble.level + (disaster.casualties_trapped.level * 10)
    return 0


def get_partner_count(disaster: Disaster, resource: Resource) -> int:
    """
    If I am a dependent resource (e.g. Truck), how many of my 'Bosses' (e.g. Excavator)
    are at the site?
    """
    rt = resource.resource_type
    roster = disaster.roster

    if rt == ResourceType.TRUCK:
        return len(roster[ResourceType.EXCAVATOR])
    elif rt == ResourceType.AMBULANCE:
        # Ambulances don't strictly rely on others, but they need events to happen
        return 1
    elif rt == ResourceType.EXCAVATOR:
        # At a collapse, Excavator needs a Fire Truck partner
        if isinstance(disaster, BuildingCollapse):
            return len(roster[ResourceType.FIRE_TRUCK])
    elif rt == ResourceType.FIRE_TRUCK:
        # At a collapse, Fire Truck needs an Excavator partner
        if isinstance(disaster, BuildingCollapse):
            return len(roster[ResourceType.EXCAVATOR])

    return 1  # Default: I don't need a partner


# ============================================================================
# MARK: Random Policy
# ============================================================================


def random_policy(resource: Resource, disasters: DisasterStore, env: simpy.Environment):
    """
    Randomly assigns a resource to any available disaster.
    """

    valid_targets = [d for d in disasters.items if is_useful(resource, d)]

    if not valid_targets:
        # TODO: This should just return / do nothing, but also needs to return the resource to the idle list, and also not cause and infinite loop where no time advances
        valid_targets = disasters.items

    target = random.choice(valid_targets)
    disaster: Disaster = yield disasters.get(filter=lambda x: x == target)
    disasters.put(disaster)
    disaster.transfer_resource(resource)


# ============================================================================
# MARK: First Priority Policy
# ============================================================================


def first_priority_policy(resource: Resource, disasters: DisasterStore, env: simpy.Environment):
    """
    Prioritizes the disaster with the MOST total resources currently assigned.
    """

    # Sort by total roster count (Trucks + Excavators)
    def get_total_resources(ls: Disaster):
        return sum(len(roster_set) for roster_set in ls.roster.values())

    sorted_disasters = sorted(disasters.items, key=get_total_resources)

    # Pick the one with the most resources, sorted Ascending, last in list
    target = sorted_disasters[-1]
    disaster: Landslide = yield disasters.get(filter=lambda x: x == target)

    disasters.put(disaster)
    disaster.transfer_resource(resource)


# ============================================================================
# MARK: Split Excavator Policy
# ============================================================================


def split_excavator_policy(resource: Resource, disasters: DisasterStore, env: simpy.Environment):
    """
    Evenly splits Excavators.
    Splits Trucks based on where Excavators are located.
    """

    candidates = [d for d in disasters.items if is_useful(resource, d)]
    if not candidates:
        candidates = disasters.items

    rt = resource.resource_type
    target = None

    # 1. Heavy Machinery Logic
    if rt in [ResourceType.EXCAVATOR, ResourceType.FIRE_TRUCK]:
        # Go to the site with the FEWEST of my type
        target = min(candidates, key=lambda x: len(x.roster[rt]))

    # 2. Support Logic
    else:
        # Go to the site with the MOST partners (e.g. Truck looks for Excavators)
        # Tie-breaker: Fewest of my own type (load balancing)
        target = max(candidates, key=lambda x: (get_partner_count(x, resource), -len(x.roster[rt])))

    disaster = yield disasters.get(filter=lambda x: x == target)
    disasters.put(disaster)
    disaster.transfer_resource(resource)


# ============================================================================
# MARK: Closest Neighbor Policy
# ============================================================================


def closest_neighbor_policy(resource: Resource, disasters: DisasterStore, env: simpy.Environment):
    """
    Minimizes travel time.
    Trucks only go to the nearest site that actually has an Excavator.
    """

    candidates = [d for d in disasters.items if is_useful(resource, d)]
    if not candidates:
        candidates = disasters.items

    target = min(candidates, key=lambda d: get_dist(resource, d))

    disaster = yield disasters.get(filter=lambda x: x == target)
    disasters.put(disaster)
    disaster.transfer_resource(resource)


# ============================================================================
# MARK: Smallest Job First Policy
# ============================================================================


def smallest_job_first_policy(resource: Resource, disasters: DisasterStore, env: simpy.Environment):
    """
    Prioritizes the landslide with the least amount of dirt remaining.
    Clears disasters off the map one by one.
    """
    # Sort by current dirt level (ascending)
    candidates = [d for d in disasters.items if is_useful(resource, d)]
    if not candidates:
        candidates = disasters.items

    # Sort by severity (ascending)
    target = min(candidates, key=get_severity)

    disaster = yield disasters.get(filter=lambda x: x == target)
    disasters.put(disaster)
    disaster.transfer_resource(resource)


# ============================================================================
# MARK: Balanced Ratio Policy
# ============================================================================


def balanced_ratio_policy(resource: Resource, disasters: DisasterStore, env: simpy.Environment):
    """
    Attempts to maintain an ideal Truck-to-Excavator ratio to minimize queuing.
    """

    candidates = [d for d in disasters.items if is_useful(resource, d)]
    if not candidates:
        candidates = disasters.items

    rt = resource.resource_type

    def get_ratio_score(d):
        # We want to go to the LOWEST score

        if rt == ResourceType.TRUCK:
            excs = len(d.roster[ResourceType.EXCAVATOR])
            if excs == 0:
                return 1000  # Penalty
            return len(d.roster[ResourceType.TRUCK]) / excs

        elif rt == ResourceType.AMBULANCE:
            # Ratio: Ambulances per Casualty
            casualties = 0
            if isinstance(d, StructureFire):
                casualties = d.casualties.level
            elif isinstance(d, BuildingCollapse):
                casualties = d.casualties_trapped.level

            if casualties == 0:
                return 1000
            return len(d.roster[ResourceType.AMBULANCE]) / casualties

        elif rt in [ResourceType.EXCAVATOR, ResourceType.FIRE_TRUCK]:
            # Heavy machinery just spreads out
            return len(d.roster[rt])

        return 0

    target = min(candidates, key=get_ratio_score)

    disaster = yield disasters.get(filter=lambda x: x == target)
    disasters.put(disaster)
    disaster.transfer_resource(resource)


# ============================================================================
# MARK: Smart Split Policy
# ============================================================================


def smart_split_policy(resource: Resource, disasters: DisasterStore, env: simpy.Environment):
    """
    - Heavy units split evenly.
    - Support units go where partners are.
    - CRITICAL: Do not send resources if the inbound fleet is already sufficient to clear the job.
    """

    candidates = [d for d in disasters.items if is_useful(resource, d)]
    if not candidates:
        candidates = disasters.items

    rt = resource.resource_type

    # 1. Filter out "Already Solved" sites (Effective work <= 0)
    viable_sites = []
    for d in candidates:
        severity = get_severity(d)

        # Calculate incoming capacity
        inbound_count = len(d.roster[rt])
        capacity_per_unit = 0

        if rt == ResourceType.TRUCK:
            capacity_per_unit = ResourceType.TRUCK.specs["capacity"]
        elif rt == ResourceType.AMBULANCE:
            capacity_per_unit = 1
        else:
            # Heavy machinery doesn't have "capacity" per se, they work over time.
            # So we always consider them viable unless severity is 0
            viable_sites.append(d)
            continue

        potential_work = inbound_count * capacity_per_unit
        if severity - potential_work > 0:
            viable_sites.append(d)

    # Fallback if all sites are "booked"
    if not viable_sites:
        viable_sites = candidates

    # 2. Apply Split Logic on Viable Sites
    target = None
    if rt in [ResourceType.EXCAVATOR, ResourceType.FIRE_TRUCK]:
        target = min(viable_sites, key=lambda x: len(x.roster[rt]))
    else:
        # Support units go to viable site with fewest of my type
        target = min(viable_sites, key=lambda x: len(x.roster[rt]))

    disaster = yield disasters.get(filter=lambda x: x == target)
    disasters.put(disaster)
    disaster.transfer_resource(resource)


# ============================================================================
# MARK: Cost Function Policy
# ============================================================================


def cost_function_policy(resource: Resource, disasters: DisasterStore, env: simpy.Environment):
    """
    Calculates Cost: Distance + Queue + Completion Penalty.
    """
    candidates = [d for d in disasters.items if is_useful(resource, d)]
    if not candidates:
        candidates = disasters.items

    def calculate_score(d):
        # 1. Drive Time
        dist = get_dist(resource, d)
        speed = resource.resource_type.specs["speed"]
        drive_cost = dist / speed

        # 2. Queue Cost (Waiting for partners)
        queue_penalty = 0
        partners = get_partner_count(d, resource)

        if partners == 0:
            queue_penalty = 500  # High penalty for having no partner on site
        else:
            my_type_count = len(d.roster[resource.resource_type])
            ratio = my_type_count / partners
            queue_penalty = ratio * 10

        return drive_cost + queue_penalty

    target = min(candidates, key=calculate_score)

    disaster = yield disasters.get(filter=lambda x: x == target)
    disasters.put(disaster)
    disaster.transfer_resource(resource)


# ============================================================================
# MARK: Gravity Policy
# ============================================================================


def gravity_policy(resource: Resource, disasters: DisasterStore, env: simpy.Environment):
    """
    High Pull = High Severity + High Priority.
    Low Pull = High Distance + High Queue.
    """
    candidates = [d for d in disasters.items if is_useful(resource, d)]
    if not candidates:
        candidates = disasters.items

    def get_score(d):
        # Numerator: Urgency
        severity = get_severity(d)

        # Priority Multiplier (Lives > Dirt)
        priority = 1.0
        if isinstance(d, (StructureFire, BuildingCollapse)):
            priority = 5.0  # Saving lives is 5x more important than dirt

        urgency = severity * priority

        # Denominator: Friction
        dist = get_dist(resource, d) + 1

        my_count = len(d.roster[resource.resource_type])
        friction = dist + (my_count * 10)

        return urgency / friction

    target = max(candidates, key=get_score)

    disaster = yield disasters.get(filter=lambda x: x == target)
    disasters.put(disaster)
    disaster.transfer_resource(resource)


# ============================================================================
# MARK: Chain Gang Policy
# ============================================================================


def chain_gang_policy(resource: Resource, disasters: DisasterStore, env: simpy.Environment):
    """
    Anchors (Exc/Fire) -> Largest Jobs.
    Runners (Truck/Amb) -> Closest Active Anchor.
    """
    candidates = [d for d in disasters.items if is_useful(resource, d)]
    if not candidates:
        candidates = disasters.items

    rt = resource.resource_type
    target = None

    # ANCHORS: Go to biggest mess and stay there
    if rt in [ResourceType.EXCAVATOR, ResourceType.FIRE_TRUCK]:
        target = max(candidates, key=get_severity)

    # RUNNERS: Go to closest site that has an Anchor
    else:
        # Find sites that have the partners we need
        active_sites = [d for d in candidates if get_partner_count(d, resource) > 0]

        if active_sites:
            target = min(active_sites, key=lambda d: get_dist(resource, d))
        else:
            # No active sites? Go to biggest job to wait for anchors
            target = max(candidates, key=get_severity)

    disaster = yield disasters.get(filter=lambda x: x == target)
    disasters.put(disaster)
    disaster.transfer_resource(resource)


# ============================================================================
# MARK: Policy Map
# ============================================================================


@dataclass
class Policy:
    name: str
    func: Callable


POLICIES: List[Policy] = [
    Policy("random", random_policy),
    Policy("first_priority", first_priority_policy),
    Policy("split_excavator", split_excavator_policy),
    Policy("closest_neighbor", closest_neighbor_policy),
    Policy("smallest_job_first", smallest_job_first_policy),
    Policy("balanced_ratio", balanced_ratio_policy),
    Policy("smart_split", smart_split_policy),
    Policy("cost_function", cost_function_policy),
    Policy("gravity", gravity_policy),
    Policy("chain_gang", chain_gang_policy),
]
