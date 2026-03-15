from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import simpy

from .calendar import Season
from .policies_tournament import (
    run_tournament_policy,
    set_tournament_depth,
    clear_tournament_cache,
)
from .real_world_params import travel_minutes_from_distance
from .simulation import Disaster, Landslide, Resource, ResourceNode, ResourceType, SnowEvent


@dataclass
class Policy:
    name: str
    func: Callable[[Resource, list[Disaster], simpy.Environment], Disaster | None]


def get_dist(r: Resource, node: ResourceNode) -> float:
    if r.engine.road_graph is not None:
        return r.engine.get_distance(r, node)
    return math.hypot(r.location[0] - node.location[0], r.location[1] - node.location[1])


def is_useful(resource: Resource, disaster: Disaster) -> bool:
    return resource.resource_type in disaster.needed_resources()


def get_severity(disaster: Disaster) -> float:
    if isinstance(disaster, Landslide):
        return disaster.dirt.level
    return float(disaster.get_scale())


def get_partner_count(disaster: Disaster, resource: Resource) -> int:
    """
    If I am a dependent resource (e.g. Truck), how many of my 'Bosses' (e.g. Excavator)
    are at the site?
    """
    rt = resource.resource_type
    roster = disaster.roster

    if rt == ResourceType.TRUCK:
        return len(roster[ResourceType.EXCAVATOR])

    return 1


# ============================================================================
# MARK: Random Policy
# ============================================================================


def random_policy(resource: Resource, disasters: list[Disaster], _env: simpy.Environment) -> Disaster:
    """
    Randomly assigns a resource to any available disaster.
    """

    valid_targets = [d for d in disasters if is_useful(resource, d)]

    if not valid_targets:
        valid_targets = disasters

    target = resource.engine.decision_rng.choice(valid_targets)
    return target


# ============================================================================
# MARK: First Priority Policy
# ============================================================================


def first_priority_policy(_resource: Resource, disasters: list[Disaster], _env: simpy.Environment) -> Disaster:
    """
    Prioritizes the disaster with the MOST total resources currently assigned.
    """

    def get_total_resources(ls: Disaster):
        return sum(len(roster_set) for roster_set in ls.roster.values())

    sorted_disasters = sorted(disasters, key=get_total_resources)

    target = sorted_disasters[-1]
    return target


# ============================================================================
# MARK: Split Excavator Policy
# ============================================================================


def split_excavator_policy(resource: Resource, disasters: list[Disaster], _env: simpy.Environment) -> Disaster:
    """
    Evenly splits Excavators.
    Splits Trucks based on where Excavators are located.
    """

    candidates = [d for d in disasters if is_useful(resource, d)] or disasters

    rt = resource.resource_type
    target = None

    if rt in [ResourceType.EXCAVATOR]:  # [,ResourceType.FIRE_TRUCK]
        target = min(candidates, key=lambda x: len(x.roster[rt]))
    else:
        target = max(candidates, key=lambda x: (get_partner_count(x, resource), -len(x.roster[rt])))

    return target


# ============================================================================
# MARK: Smallest Job First Policy
# ============================================================================


def smallest_job_first_policy(resource: Resource, disasters: list[Disaster], _env: simpy.Environment) -> Disaster:
    """
    Prioritizes the landslide with the least amount of dirt remaining.
    Clears disasters off the map one by one.
    """
    candidates = [d for d in disasters if is_useful(resource, d)] or disasters

    # Sort by severity (ascending)
    target = min(candidates, key=get_severity)

    return target


# ============================================================================
# MARK: Balanced Ratio Policy
# ============================================================================


def balanced_ratio_policy(resource: Resource, disasters: list[Disaster], _env: simpy.Environment) -> Disaster:
    candidates = [d for d in disasters if is_useful(resource, d)] or disasters
    rt = resource.resource_type

    def score(d: Disaster) -> float:
        if rt == ResourceType.TRUCK:
            excs = len(d.roster[ResourceType.EXCAVATOR])
            if excs == 0:
                return 1000.0
            return len(d.roster[ResourceType.TRUCK]) / excs
        if rt == ResourceType.EXCAVATOR:
            return float(len(d.roster[rt]))
        return 0.0

    return min(candidates, key=score)


# ============================================================================
# MARK: Smart Split Policy
# ============================================================================


def smart_split_policy(resource: Resource, disasters: list[Disaster], _env: simpy.Environment) -> Disaster:
    candidates = [d for d in disasters if is_useful(resource, d)] or disasters
    rt = resource.resource_type

    viable_sites: list[Disaster] = []
    for d in candidates:
        severity = get_severity(d)
        inbound_count = len(d.roster[rt])
        if rt == ResourceType.TRUCK:
            potential_removal = inbound_count * ResourceType.TRUCK.specs["capacity"]
            if severity - potential_removal > 0:
                viable_sites.append(d)
        else:
            viable_sites.append(d)

    viable_sites = viable_sites or candidates
    if rt == ResourceType.EXCAVATOR:
        return min(viable_sites, key=lambda x: len(x.roster[rt]))
    return min(viable_sites, key=lambda x: len(x.roster[rt]))


# ============================================================================
# MARK: Cost Function Policy
# ============================================================================


def cost_function_policy(resource: Resource, disasters: list[Disaster], _env: simpy.Environment) -> Disaster:
    candidates = [d for d in disasters if is_useful(resource, d)] or disasters

    def calculate_score(d: Disaster) -> float:
        dist = get_dist(resource, d)
        speed = resource.resource_type.specs["speed"]
        drive_cost = travel_minutes_from_distance(dist, speed)

        partners = get_partner_count(d, resource)
        if partners == 0:
            queue_penalty = 500.0
        else:
            my_type_count = len(d.roster[resource.resource_type])
            queue_penalty = (my_type_count / partners) * 10.0
        return drive_cost + queue_penalty

    return min(candidates, key=calculate_score)


def get_seasonal_priority(disaster: Disaster) -> tuple[int, float]:
    engine = disaster.engine
    priority = 100
    season = engine.calendar.get_season() if engine.calendar is not None else None

    if isinstance(disaster, SnowEvent):
        if season == Season.WINTER:
            priority = 1
        elif season == Season.FALL:
            priority = 3
        else:
            priority = 10
    elif isinstance(disaster, Landslide):
        if season in [Season.WINTER, Season.SPRING]:
            priority = 2
        elif season == Season.FALL:
            priority = 4
        else:
            priority = 8
    else:
        priority = 5

    return (priority, get_severity(disaster))


def _has_required_partners(disaster: Disaster, resource: Resource) -> bool:
    needed = disaster.needed_resources()
    if resource.resource_type not in needed:
        return True

    for rt in needed:
        if rt == resource.resource_type:
            continue
        if len(disaster.roster.get(rt, [])) > 0:
            return True
    return False


def seasonal_priority_policy(resource: Resource, disasters: list[Disaster], _env: simpy.Environment) -> Disaster:
    candidates = [d for d in disasters if is_useful(resource, d)] or disasters
    candidates_with_partners = [d for d in candidates if _has_required_partners(d, resource)]
    if candidates_with_partners:
        candidates = candidates_with_partners
    return sorted(candidates, key=get_seasonal_priority)[0]


def tournament_policy_func(resource: Resource, disasters: list[Disaster], env: simpy.Environment) -> Disaster | None:
    return run_tournament_policy(resource, disasters, env, POLICIES)


POLICIES: list[Policy] = [
    # Policy("random", random_policy),  # 0%
    Policy("first_priority", first_priority_policy),
    Policy("split_excavator", split_excavator_policy),
    # Policy("smallest_job_first", smallest_job_first_policy),
    Policy("balanced_ratio", balanced_ratio_policy),
    Policy("smart_split", smart_split_policy),
    # Policy("cost_function", cost_function_policy),
    Policy("seasonal_priority", seasonal_priority_policy),
]

TOURNAMENT_POLICY = Policy("tournament", tournament_policy_func)
BENCHMARK_POLICIES: list[Policy] = [*POLICIES, TOURNAMENT_POLICY]
