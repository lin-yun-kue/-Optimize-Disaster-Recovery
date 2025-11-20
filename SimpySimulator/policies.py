from simulation import (
    SimulationConfig,
    Resource,
    Landslide,
    DisasterStore,
    ResourceType,
)
import simpy
import random
from typing import List
import math


def random_policy(resource: Resource, disasters: DisasterStore, env: simpy.Environment):
    """
    Randomly assigns a resource to any available disaster.
    """

    target = random.choice(disasters.items)
    disaster: Landslide = yield disasters.get(filter=lambda x: x == target)

    disasters.put(disaster)
    disaster.transfer_resource(resource)


def first_priority_policy(resource: Resource, disasters: DisasterStore, env: simpy.Environment):
    """
    Prioritizes the disaster with the MOST total resources currently assigned.
    """

    # Sort by total roster count (Trucks + Excavators)
    def get_total_resources(ls: Landslide):
        return sum(len(roster_set) for roster_set in ls.roster.values())

    sorted_disasters = sorted(disasters.items, key=get_total_resources)

    # Pick the one with the most resources, sorted Ascending, last in list
    target = sorted_disasters[-1]
    disaster: Landslide = yield disasters.get(filter=lambda x: x == target)

    disasters.put(disaster)
    disaster.transfer_resource(resource)


def split_excavator_policy(resource: Resource, disasters: DisasterStore, env: simpy.Environment):
    """
    Evenly splits Excavators.
    Splits Trucks based on where Excavators are located.
    """

    target = None

    if resource.resource_type == ResourceType.EXCAVATOR:
        # Sort by number of excavators in roster (ascending)
        sorted_disasters = sorted(disasters.items, key=lambda x: len(x.roster[ResourceType.EXCAVATOR]))
        target = sorted_disasters[0]  # Pick the one with fewest excavators

    elif resource.resource_type == ResourceType.TRUCK:
        # Heuristic:
        # 1. Penalize sites with 0 Excavators (High value so they go to end of sort)
        # 2. Otherwise sort by number of trucks (Low value goes to start of sort)
        def truck_heuristic(ls: Landslide):
            exc_count = len(ls.roster[ResourceType.EXCAVATOR])
            truck_count = len(ls.roster[ResourceType.TRUCK])

            if exc_count == 0:
                return 1000000  # Last priority
            return truck_count  # Prioritize keeping truck counts low

        sorted_disasters = sorted(disasters.items, key=truck_heuristic)
        target = sorted_disasters[0]

    disaster: Landslide = yield disasters.get(filter=lambda x: x == target)

    disasters.put(disaster)
    disaster.transfer_resource(resource)


def closest_neighbor_policy(resource: Resource, disasters: DisasterStore, env: simpy.Environment):
    """
    Minimizes travel time.
    Trucks only go to the nearest site that actually has an Excavator.
    """

    candidates = disasters.items
    target = None

    # Helper to calculate distance
    def get_dist(r, ls):
        return math.hypot(r.location[0] - ls.location[0], r.location[1] - ls.location[1])

    if resource.resource_type == ResourceType.EXCAVATOR:
        # Excavators just go to the closest pile of dirt
        target = min(candidates, key=lambda ls: get_dist(resource, ls))

    elif resource.resource_type == ResourceType.TRUCK:
        # Trucks filter for sites that have (or will have) an excavator
        valid_sites = [ls for ls in candidates if len(ls.roster[ResourceType.EXCAVATOR]) > 0]

        if valid_sites:
            target = min(valid_sites, key=lambda ls: get_dist(resource, ls))
        else:
            # If no sites have excavators, fall back to closest site (prepare for arrival)
            target = min(candidates, key=lambda ls: get_dist(resource, ls))

    # Execute
    disaster = yield disasters.get(filter=lambda x: x == target)
    disasters.put(disaster)
    disaster.transfer_resource(resource)


def smallest_job_first_policy(resource: Resource, disasters: DisasterStore, env: simpy.Environment):
    """
    Prioritizes the landslide with the least amount of dirt remaining.
    Clears disasters off the map one by one.
    """
    # Sort by current dirt level (ascending)
    sorted_disasters = sorted(disasters.items, key=lambda x: x.dirt.level)

    # Simply pick the smallest one
    target = sorted_disasters[0]

    disaster = yield disasters.get(filter=lambda x: x == target)
    disasters.put(disaster)
    disaster.transfer_resource(resource)


def balanced_ratio_policy(resource: Resource, disasters: DisasterStore, env: simpy.Environment):
    """
    Attempts to maintain an ideal Truck-to-Excavator ratio to minimize queuing.
    """

    target = None
    candidates = disasters.items

    if resource.resource_type == ResourceType.EXCAVATOR:
        # Excavators go to the site with the LEAST excavators (spread the bottleneck)
        # Tie-breaker: Most dirt
        target = min(
            candidates,
            key=lambda ls: (len(ls.roster[ResourceType.EXCAVATOR]), -ls.dirt.level),
        )

    elif resource.resource_type == ResourceType.TRUCK:
        # 1. Calculate the "Score" for each site.
        # Score = (Current Trucks) / (Current Excavators)
        # We want to send the truck to the site with the LOWEST ratio.

        def get_ratio_score(ls):
            exc_count = len(ls.roster[ResourceType.EXCAVATOR])
            trk_count = len(ls.roster[ResourceType.TRUCK])

            if exc_count == 0:
                # High penalty, don't send trucks to empty sites unless necessary
                return 1000

            return trk_count / exc_count

        target = min(candidates, key=get_ratio_score)

    disaster = yield disasters.get(filter=lambda x: x == target)
    disasters.put(disaster)
    disaster.transfer_resource(resource)


def get_effective_dirt(landslide):
    """
    Calculates dirt remaining after accounting for all trucks currently
    en-route or working at the site.
    """
    current_dirt = landslide.dirt.level

    # Count trucks in the roster (both on-site and driving)
    num_trucks = len(landslide.roster[ResourceType.TRUCK])
    truck_capacity = SimulationConfig.SPECS[ResourceType.TRUCK]["capacity"]

    potential_removal = num_trucks * truck_capacity
    return current_dirt - potential_removal


def smart_split_policy(resource: Resource, disasters: DisasterStore, env: simpy.Environment):
    """
    1. Splits Excavators evenly.
    2. Distributes Trucks to sites with Excavators.
    3. CRITICAL: Stops sending Trucks if the inbound fleet is already enough to clear the pile.
    """

    candidates = disasters.items
    target = None

    if resource.resource_type == ResourceType.EXCAVATOR:
        # Standard Split Logic
        target = min(candidates, key=lambda x: len(x.roster[ResourceType.EXCAVATOR]))

    elif resource.resource_type == ResourceType.TRUCK:
        # Filter out sites that are effectively done
        # We only want sites where (Effective Dirt > 0) AND (Excavators > 0)

        viable_sites = []
        for ls in candidates:
            has_excavator = len(ls.roster[ResourceType.EXCAVATOR]) > 0
            needs_help = get_effective_dirt(ls) > 0

            if has_excavator and needs_help:
                viable_sites.append(ls)

        if not viable_sites:
            # If all viable sites are full, or no excavators are set up yet:
            # Fallback 1: Go to any site with an excavator (even if overbooked, better than idle)
            # Fallback 2: Go to the site with the most dirt (standard fallback)
            with_exc = [ls for ls in candidates if len(ls.roster[ResourceType.EXCAVATOR]) > 0]
            if with_exc:
                target = min(with_exc, key=lambda x: len(x.roster[ResourceType.TRUCK]))
            else:
                target = max(candidates, key=lambda x: x.dirt.level)
        else:
            # Of the viable sites, send to the one with the fewest trucks
            # (Load Balancing)
            target = min(viable_sites, key=lambda x: len(x.roster[ResourceType.TRUCK]))

    disaster = yield disasters.get(filter=lambda x: x == target)
    disasters.put(disaster)
    disaster.transfer_resource(resource)


def cost_function_policy(resource: Resource, disasters: DisasterStore, env: simpy.Environment):
    candidates = disasters.items

    # Specs
    t_specs = SimulationConfig.SPECS[ResourceType.TRUCK]
    e_specs = SimulationConfig.SPECS[ResourceType.EXCAVATOR]

    def calculate_score(ls):
        # 1. Distance Cost
        dist = math.hypot(resource.location[0] - ls.location[0], resource.location[1] - ls.location[1])

        speed = t_specs["drive_speed"] if resource.resource_type == ResourceType.TRUCK else e_specs["drive_speed"]
        drive_time = dist / speed

        # 2. Queue Cost (Only applies to trucks joining a line)
        queue_penalty = 0
        if resource.resource_type == ResourceType.TRUCK:
            num_exc = len(ls.roster[ResourceType.EXCAVATOR])
            num_trk = len(ls.roster[ResourceType.TRUCK])

            if num_exc == 0:
                queue_penalty = 10000  # Huge penalty for no excavator
            else:
                # Rough estimate: How many loads before me?
                # If there are 5 trucks and 1 excavator, I wait for 5 loads.
                loads_ahead = num_trk / num_exc
                queue_penalty = loads_ahead * t_specs["load_time"]

        # 3. Completion Penalty (Lookahead)
        completion_penalty = 0
        if get_effective_dirt(ls) <= 0:
            completion_penalty = 5000  # Don't go to finished sites

        return drive_time + queue_penalty + completion_penalty

    # Pick the site with the lowest Cost Score
    target = min(candidates, key=calculate_score)

    disaster = yield disasters.get(filter=lambda x: x == target)
    disasters.put(disaster)
    disaster.transfer_resource(resource)


def gravity_policy(resource: Resource, disasters: DisasterStore, env: simpy.Environment):
    """
    Gravitational Pull:
    - High Dirt = High Pull
    - High Distance = Low Pull
    - High Queue = Low Pull
    """

    candidates = disasters.items
    t_specs = SimulationConfig.SPECS[ResourceType.TRUCK]

    def get_score(ls):
        # 1. Distance Factor
        dist = math.hypot(resource.location[0] - ls.location[0], resource.location[1] - ls.location[1])
        if dist == 0:
            dist = 0.1  # Avoid division by zero

        # 2. Urgency Factor (Dirt Remaining)
        urgency = get_effective_dirt(ls)
        if urgency <= 0:
            return -float("inf")  # Do not go

        # 3. Queue Factor
        # We penalize sites that already have too many trucks per excavator
        exc_count = len(ls.roster[ResourceType.EXCAVATOR])
        trk_count = len(ls.roster[ResourceType.TRUCK])

        if resource.resource_type == ResourceType.TRUCK:
            if exc_count == 0:
                return -float("inf")  # Dead end
            ratio = trk_count / exc_count
            # If ratio is > 5, the score drops massively
            queue_penalty = ratio * 20
        else:
            # For excavators, we want to go where there are FEW excavators
            queue_penalty = exc_count * 50

        # THE FORMULA: (Urgency) / (Distance + Penalty)
        score = urgency / (dist + queue_penalty)
        return score

    # We want the HIGHEST score
    target = max(candidates, key=get_score)

    disaster = yield disasters.get(filter=lambda x: x == target)
    disasters.put(disaster)
    disaster.transfer_resource(resource)


def chain_gang_policy(resource: Resource, disasters: DisasterStore, env: simpy.Environment):
    """
    OPTIMIZED FOR TRUCK SCARCITY.

    1. Excavators: Anchors. They go to the LARGEST pile and stay there until it dies.
       They do not care about splitting evenly. They care about volume.

    2. Trucks: Runners. They go to the CLOSEST active Excavator.
       They ignore queue size because we know Excavators are starving.
    """

    candidates = disasters.items
    target = None

    if resource.resource_type == ResourceType.EXCAVATOR:
        # Anchor Logic: Go to the site with the MOST dirt and stay there.
        # This minimizes the time the "Loading Dock" is closed due to travel.
        target = max(candidates, key=lambda x: x.dirt.level)

    elif resource.resource_type == ResourceType.TRUCK:
        # Runner Logic: Find active Excavators.
        # We only care about sites that HAVE an excavator and HAVE dirt.
        active_sites = [
            ls for ls in candidates if len(ls.roster[ResourceType.EXCAVATOR]) > 0 and get_effective_dirt(ls) > 0
        ]

        if not active_sites:
            # Fallback: If no excavators are set up, go to the largest potential job
            target = max(candidates, key=lambda x: x.dirt.level)
        else:
            # Go to the CLOSEST active site.
            # Distance is the only enemy when Trucks are the bottleneck.
            def get_dist(ls):
                return math.hypot(
                    resource.location[0] - ls.location[0],
                    resource.location[1] - ls.location[1],
                )

            target = min(active_sites, key=get_dist)

    disaster = yield disasters.get(filter=lambda x: x == target)
    disasters.put(disaster)
    disaster.transfer_resource(resource)
