import simpy
from typing import Callable
import math
import multiprocessing
import pickle
import time
from .real_world_params import travel_minutes_from_distance
from .simulation import *
from .engine import SimPySimulationEngine, ScenarioConfig

# Import calendar for seasonal policies
from .calendar import Season


# ============================================================================
# MARK: Tournament Configuration
# ============================================================================


class TournamentConfig:
    """Configuration for the tree search tournament policy."""

    depth: int = 1  # 1 = run to completion, 2+ = look ahead N decisions


TOURNAMENT_DEPTH = TournamentConfig()

DISABLE_TOURNAMENT_MULTIPROCESSING = True
TOURNAMENT_DEBUG = False
TOURNAMENT_POLICY_WHITELIST: set[str] | None = None


def set_tournament_depth(depth: int):
    """Set the tree search depth for tournament policy."""
    TOURNAMENT_DEPTH.depth = depth


def set_tournament_debug(enabled: bool):
    """Enable/disable tournament debug logging."""
    global TOURNAMENT_DEBUG
    TOURNAMENT_DEBUG = enabled


def set_tournament_candidate_policies(policy_names: list[str] | None):
    """Restrict tournament search to the given policy names (excluding tournament itself).

    Pass None to use all available non-tournament policies.
    """
    global TOURNAMENT_POLICY_WHITELIST
    TOURNAMENT_POLICY_WHITELIST = set(policy_names) if policy_names is not None else None


@dataclass
class Policy:
    name: str
    func: Callable[[Resource, list[Disaster], simpy.Environment], Disaster | None]


# ============================================================================
# MARK: Utils
# ============================================================================


def get_dist(r: Resource, node: ResourceNode):
    """Get distance between resource and node, using GIS-aware distance if available."""
    # Use engine's GIS-aware distance calculation if engine is available and has road graph
    if hasattr(r, "engine") and r.engine and r.engine.road_graph is not None:
        return r.engine.get_distance(r, node)
    # Fallback to Euclidean distance
    return math.hypot(r.location[0] - node.location[0], r.location[1] - node.location[1])


def is_useful(resource: Resource, disaster: Disaster):
    """Determines if a resource is actually needed at a disaster site."""
    return resource.resource_type in disaster.needed_resources()


def get_effective_dirt(landslide: Landslide) -> float:
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
    # elif isinstance(disaster, StructureFire):
    #     return disaster.fire_intensity.level + (disaster.casualties.level * 10)
    # elif isinstance(disaster, BuildingCollapse):
    #     return disaster.rubble.level + (disaster.casualties_trapped.level * 10)
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
    # elif rt == ResourceType.AMBULANCE:
    #     # Ambulances don't strictly rely on others, but they need events to happen
    #     return 1
    # elif rt == ResourceType.EXCAVATOR:
    #     # At a collapse, Excavator needs a Fire Truck partner
    #     if isinstance(disaster, BuildingCollapse):
    #         return len(roster[ResourceType.FIRE_TRUCK])
    # elif rt == ResourceType.FIRE_TRUCK:
    #     # At a collapse, Fire Truck needs an Excavator partner
    #     if isinstance(disaster, BuildingCollapse):
    #         return len(roster[ResourceType.EXCAVATOR])

    return 1  # Default: I don't need a partner


# ============================================================================
# MARK: Random Policy
# ============================================================================


def random_policy(resource: Resource, disasters: list[Disaster], _env: simpy.Environment) -> Disaster:
    """
    Randomly assigns a resource to any available disaster.
    """

    valid_targets = [d for d in disasters if is_useful(resource, d)]

    if not valid_targets:
        # TODO: This should just return / do nothing, but also needs to return the resource to the idle list, and also not cause and infinite loop where no time advances
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

    # Sort by total roster count (Trucks + Excavators)
    def get_total_resources(ls: Disaster):
        return sum(len(roster_set) for roster_set in ls.roster.values())

    sorted_disasters = sorted(disasters, key=get_total_resources)

    # Pick the one with the most resources, sorted Ascending, last in list
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

    candidates = [d for d in disasters if is_useful(resource, d)]
    if not candidates:
        candidates = disasters

    rt = resource.resource_type
    target = None

    # 1. Heavy Machinery Logic
    if rt in [ResourceType.EXCAVATOR]:  # [,ResourceType.FIRE_TRUCK]
        # Go to the site with the FEWEST of my type
        target = min(candidates, key=lambda x: len(x.roster[rt]))

    # 2. Support Logic
    else:
        # Go to the site with the MOST partners (e.g. Truck looks for Excavators)
        # Tie-breaker: Fewest of my own type (load balancing)
        target = max(candidates, key=lambda x: (get_partner_count(x, resource), -len(x.roster[rt])))

    return target


# ============================================================================
# MARK: Closest Neighbor Policy
# ============================================================================


def closest_neighbor_policy(resource: Resource, disasters: list[Disaster], _env: simpy.Environment) -> Disaster:
    """
    Minimizes travel time.
    Trucks only go to the nearest site that actually has an Excavator.
    """

    candidates = [d for d in disasters if is_useful(resource, d)]
    if not candidates:
        candidates = disasters

    target = min(candidates, key=lambda d: get_dist(resource, d))

    return target


# ============================================================================
# MARK: Smallest Job First Policy
# ============================================================================


def smallest_job_first_policy(resource: Resource, disasters: list[Disaster], _env: simpy.Environment) -> Disaster:
    """
    Prioritizes the landslide with the least amount of dirt remaining.
    Clears disasters off the map one by one.
    """
    # Sort by current dirt level (ascending)
    candidates = [d for d in disasters if is_useful(resource, d)]
    if not candidates:
        candidates = disasters

    # Sort by severity (ascending)
    target = min(candidates, key=get_severity)

    return target


# ============================================================================
# MARK: Balanced Ratio Policy
# ============================================================================


def balanced_ratio_policy(resource: Resource, disasters: list[Disaster], _env: simpy.Environment) -> Disaster:
    """
    Attempts to maintain an ideal Truck-to-Excavator ratio to minimize queuing.
    """

    candidates = [d for d in disasters if is_useful(resource, d)]
    if not candidates:
        candidates = disasters

    rt = resource.resource_type

    def get_ratio_score(d: Disaster):
        # We want to go to the LOWEST score

        if rt == ResourceType.TRUCK:
            excs = len(d.roster[ResourceType.EXCAVATOR])
            if excs == 0:
                return 1000  # Penalty
            return len(d.roster[ResourceType.TRUCK]) / excs

        # elif rt == ResourceType.AMBULANCE:
        #     # Ratio: Ambulances per Casualty
        #     casualties = 0
        #     if isinstance(d, StructureFire):
        #         casualties = d.casualties.level
        #     elif isinstance(d, BuildingCollapse):
        #         casualties = d.casualties_trapped.level

        #     if casualties == 0:
        #         return 1000
        #     return len(d.roster[ResourceType.AMBULANCE]) / casualties

        elif rt in [ResourceType.EXCAVATOR]:  # [,ResourceType.FIRE_TRUCK]:
            # Heavy machinery just spreads out
            return len(d.roster[rt])

        return 0

    target = min(candidates, key=get_ratio_score)

    return target


# ============================================================================
# MARK: Smart Split Policy
# ============================================================================


def smart_split_policy(resource: Resource, disasters: list[Disaster], _env: simpy.Environment) -> Disaster:
    """
    - Heavy units split evenly.
    - Support units go where partners are.
    - CRITICAL: Do not send resources if the inbound fleet is already sufficient to clear the job.
    """

    candidates = [d for d in disasters if is_useful(resource, d)]
    if not candidates:
        candidates = disasters

    rt = resource.resource_type

    # 1. Filter out "Already Solved" sites (Effective work <= 0)
    viable_sites: list[Disaster] = []
    for d in candidates:
        severity = get_severity(d)

        # Calculate incoming capacity
        inbound_count = len(d.roster[rt])
        capacity_per_unit = 0

        if rt == ResourceType.TRUCK:
            capacity_per_unit = ResourceType.TRUCK.specs["capacity"]
        # elif rt == ResourceType.AMBULANCE:
        #     capacity_per_unit = 1
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
    if rt in [ResourceType.EXCAVATOR]:  # [,ResourceType.FIRE_TRUCK]:
        target = min(viable_sites, key=lambda x: len(x.roster[rt]))
    else:
        # Support units go to viable site with fewest of my type
        target = min(viable_sites, key=lambda x: len(x.roster[rt]))

    return target


# ============================================================================
# MARK: Cost Function Policy
# ============================================================================


def cost_function_policy(resource: Resource, disasters: list[Disaster], _env: simpy.Environment) -> Disaster:
    """
    Calculates Cost: Distance + Queue + Completion Penalty.
    """
    candidates = [d for d in disasters if is_useful(resource, d)]
    if not candidates:
        candidates = disasters

    def calculate_score(d: Disaster):
        # 1. Drive Time
        dist = get_dist(resource, d)
        speed = resource.resource_type.specs["speed"]
        drive_cost = travel_minutes_from_distance(dist, speed)

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

    return target


# ============================================================================
# MARK: Gravity Policy
# ============================================================================


def gravity_policy(resource: Resource, disasters: list[Disaster], _env: simpy.Environment) -> Disaster:
    """
    High Pull = High Severity + High Priority.
    Low Pull = High Distance + High Queue.
    """
    candidates = [d for d in disasters if is_useful(resource, d)]
    if not candidates:
        candidates = disasters

    def get_score(d: Disaster):
        # Numerator: Urgency
        severity = get_severity(d)

        # Priority Multiplier (Lives > Dirt)
        priority = 1.0
        # if isinstance(d, (StructureFire, BuildingCollapse)):
        #     priority = 5.0  # Saving lives is 5x more important than dirt

        urgency = severity * priority

        # Denominator: Friction
        dist = get_dist(resource, d) + 1

        my_count = len(d.roster[resource.resource_type])
        friction = dist + (my_count * 10)

        return urgency / friction

    target = max(candidates, key=get_score)

    return target


# ============================================================================
# MARK: Chain Gang Policy
# ============================================================================


def chain_gang_policy(resource: Resource, disasters: list[Disaster], _env: simpy.Environment) -> Disaster:
    """
    Anchors (Exc/Fire) -> Largest Jobs.
    Runners (Truck/Amb) -> Closest Active Anchor.
    """
    candidates = [d for d in disasters if is_useful(resource, d)]
    if not candidates:
        candidates = disasters

    rt = resource.resource_type
    target = None

    # ANCHORS: Go to biggest mess and stay there
    if rt in [ResourceType.EXCAVATOR]:  # [,ResourceType.FIRE_TRUCK]:
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

    return target


# ============================================================================
# MARK: Tournament Policy
# ============================================================================


class PolicyResult(TypedDict):
    success: bool
    time: float
    policy: str
    move_id: int | None
    partial_history: list[int]
    sim_time: float
    decisions_made: int
    disasters_remaining: int
    branch_decision: int | None


ROLLOUT_CACHE: dict[tuple[str, int, tuple[int, ...], int | None], PolicyResult] = {}


def clear_tournament_cache():
    """Clear cached rollout evaluations used by tournament search."""
    ROLLOUT_CACHE.clear()


def _score_policy_result(result: PolicyResult) -> float:
    """Lower is better. Prefer success, then faster total time."""
    if result["success"]:
        return result["time"]
    return 100000 + result["disasters_remaining"] * 100 + result["time"]


def _get_tournament_candidates() -> list[Policy]:
    """Return tournament candidate policies after applying any whitelist."""
    candidates = [p for p in POLICIES if not p.name.startswith("tournament")]
    if TOURNAMENT_POLICY_WHITELIST is None:
        return candidates
    return [p for p in candidates if p.name in TOURNAMENT_POLICY_WHITELIST]


def evaluate_policy_worker(
    policy: Policy,
    seed: int,
    history: list[int],
    max_decisions: int | None = None,
    scenario_config: ScenarioConfig | None = None,
) -> PolicyResult:
    cache_key = (policy.name, seed, tuple(history), max_decisions)
    cached = ROLLOUT_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    try:
        sim_fork = SimPySimulationEngine(policy, seed, live_plot=False, scenario_config=scenario_config)
        sim_fork.initialize_world()

        sim_fork.replay_buffer = list(history)

        sim_fork.run(max_decisions=max_decisions)

        time_taken = sim_fork.get_summary()["non_idle_time"]

        result: PolicyResult = {
            "success": len(sim_fork.disaster_store.items) == 0,
            "time": time_taken,
            "policy": policy.name,
            "move_id": sim_fork.branch_decision,
            # Return the full replay history represented by this fork:
            # prior replayed moves + newly generated moves.
            "partial_history": list(history) + list(sim_fork.decision_log),
            "sim_time": sim_fork.env.now,
            "decisions_made": sim_fork.decisions_made,
            "disasters_remaining": len(sim_fork.disaster_store.items),
            "branch_decision": sim_fork.branch_decision,
        }
        ROLLOUT_CACHE[cache_key] = dict(result)
        return result
    except Exception as e:
        print(f"Exception: {e}")
        raise e


def evaluate_policy_tree(
    policy: Policy,
    seed: int,
    history: list[int],
    depth: int,
    scenario_config: ScenarioConfig | None = None,
) -> PolicyResult:
    """
    Tree search evaluation.

    For depth=1: Run to completion (standard evaluation)
    For depth>1:
        1. Run policy for `depth` decisions
        2. From that state, try ALL policies to completion
        3. Return best overall result

    This gives true look-ahead: "If I use policy X for the next N decisions,
    which policy should I use after that to finish fastest?"
    """
    if depth == 1:
        return evaluate_policy_worker(policy, seed, history, max_decisions=None, scenario_config=scenario_config)

    # Step 1: Run the candidate policy for `depth` decisions
    result_depth = evaluate_policy_worker(policy, seed, history, max_decisions=depth, scenario_config=scenario_config)

    # If no branch decision made or failed completely, return
    if result_depth["move_id"] is None:
        return result_depth

    # Step 2: From the state after depth decisions, try ALL policies to completion
    extended_history = list(result_depth["partial_history"])

    # Get all candidate policies (excluding tournament)
    candidates = _get_tournament_candidates()
    if not candidates:
        return result_depth

    best_final_result = None
    best_final_score = float("inf")

    # Run each policy from this state to completion SEQUENTIALLY
    # (can't use multiprocessing inside multiprocessing)
    for sub_policy in candidates:
        final_result = evaluate_policy_worker(
            sub_policy,
            seed,
            extended_history,
            max_decisions=None,
            scenario_config=scenario_config,
        )

        # Score: prefer success, then faster time
        score = _score_policy_result(final_result)

        if score < best_final_score:
            best_final_score = score
            best_final_result = final_result

    # Return the best scored rollout, but preserve the ROOT move/policy for the
    # tournament's current decision.
    if best_final_result:
        result = dict(best_final_result)
        result["move_id"] = result_depth["move_id"]
        result["branch_decision"] = result_depth["branch_decision"]
        result["policy"] = policy.name
        result["partial_history"] = list(result_depth["partial_history"])
        return result

    # Fallback to the depth result if no final results
    return result_depth


def evaluate_policy_recursive_tree(
    policy: Policy,
    seed: int,
    history: list[int],
    depth: int,
    scenario_config: ScenarioConfig | None = None,
) -> PolicyResult:
    """
    Recursive policy tree search matching tournament replan semantics.

    Depth semantics:
      depth=1: run the candidate policy to completion
      depth=2: run candidate policy for 1 decision, then all policies to completion
      depth=3: run candidate policy for 1 decision, then all policies for 1 decision,
               then all policies to completion
    """
    if depth <= 1:
        return evaluate_policy_worker(policy, seed, history, max_decisions=None, scenario_config=scenario_config)

    # Take exactly one policy-driven decision at this level, then recurse.
    result_step = evaluate_policy_worker(policy, seed, history, max_decisions=1, scenario_config=scenario_config)

    if result_step["move_id"] is None:
        return result_step

    extended_history = list(result_step["partial_history"])
    candidates = _get_tournament_candidates()
    if not candidates:
        return result_step

    best_child_result = None
    best_child_score = float("inf")

    for sub_policy in candidates:
        child_result = evaluate_policy_recursive_tree(
            sub_policy,
            seed,
            extended_history,
            depth - 1,
            scenario_config=scenario_config,
        )
        score = _score_policy_result(child_result)
        if score < best_child_score:
            best_child_score = score
            best_child_result = child_result

    if best_child_result:
        result = dict(best_child_result)
        # Preserve the root move of this branch because the live tournament will
        # only execute one step before re-evaluating.
        result["move_id"] = result_step["move_id"]
        result["branch_decision"] = result_step["branch_decision"]
        result["policy"] = policy.name
        result["partial_history"] = list(result_step["partial_history"])
        return result

    return result_step


def tournament_policy_func(resource: Resource, disasters: list[Disaster], env: simpy.Environment) -> Disaster | None:
    """
    The Meta-Policy (Tree Search Version).

    Args:
        depth: Number of decisions to look ahead.
               depth=1: Run each policy to completion (original behavior)
               depth>1: For each policy, run for N decisions, then try all policies to completion

    Configuration:
        Use TOURNAMENT_DEPTH variable to set depth before running.
    """
    global TOURNAMENT_DEPTH
    depth = getattr(TOURNAMENT_DEPTH, "depth", 1)
    global TOURNAMENT_DEBUG

    master_engine = resource.engine
    current_history = list(master_engine.decision_log)
    master_seed = master_engine.seed
    fork_scenario_config: ScenarioConfig | None = None
    try:
        pickle.dumps(master_engine.scenario_config)
        fork_scenario_config = master_engine.scenario_config
    except Exception:
        # Some configs (e.g., GIS objects / spatial indices) may not be picklable.
        # Fall back to default config rather than crashing the tournament.
        fork_scenario_config = None

    candidates = _get_tournament_candidates()
    if not candidates:
        raise Exception("Tournament candidate set is empty. Configure at least one non-tournament policy.")

    global DISABLE_TOURNAMENT_MULTIPROCESSING
    use_multiprocessing = not DISABLE_TOURNAMENT_MULTIPROCESSING
    t0 = time.perf_counter()
    if len(current_history) == 0:
        clear_tournament_cache()

    if TOURNAMENT_DEBUG:
        print(f"[tournament] t={env.now:.2f} depth={depth} history={len(current_history)} " f"disasters={len(disasters)} mp={use_multiprocessing}")

    if depth == 1:
        tasks = [(p, master_seed, current_history, None, fork_scenario_config) for p in candidates]

        if use_multiprocessing:
            with multiprocessing.Pool(processes=len(candidates)) as pool:
                results = pool.starmap(evaluate_policy_worker, tasks)
        else:
            results = [evaluate_policy_worker(*task) for task in tasks]
    else:
        tasks = [(p, master_seed, current_history, depth, fork_scenario_config) for p in candidates]

        if use_multiprocessing:
            with multiprocessing.Pool(processes=len(candidates)) as pool:
                results = pool.starmap(evaluate_policy_tree, tasks)
        else:
            results = [evaluate_policy_tree(*task) for task in tasks]

    best_result = None
    best_score = float("inf")

    for result in results:
        score = _score_policy_result(result)
        if score < best_score:
            best_score = score
            best_result = result

    if best_result is None:
        raise Exception("Tournament produced no policy results.")

    # Record winner
    master_engine.tournament_decisions.append((env.now, best_result["policy"]))

    if TOURNAMENT_DEBUG:
        elapsed = time.perf_counter() - t0
        successes = sum(1 for r in results if r["success"])
        print(
            f"[tournament] chose={best_result['policy']} move={best_result['move_id']} "
            f"successes={successes}/{len(results)} score={best_score:.2f} "
            f"wall={elapsed:.2f}s cache={len(ROLLOUT_CACHE)}"
        )

    # Find the disaster object in the Master's store that matches the ID
    targets = [d for d in disasters if d.id == best_result["move_id"]]
    if not targets:
        if TOURNAMENT_DEBUG:
            print(f"[tournament] move_id={best_result['move_id']} not in current disasters " f"{[d.id for d in disasters]}")
        # Fallback: re-ask the root winning policy on the current live state.
        fallback_policy = next((p for p in POLICIES if p.name == best_result["policy"]), None)
        if fallback_policy is not None and fallback_policy.name != "tournament":
            return fallback_policy.func(resource, disasters, env)
        return None
    target = targets[0]
    return target


def tournament_recursive_policy_func(resource: Resource, disasters: list[Disaster], env: simpy.Environment) -> Disaster | None:
    """
    Meta-policy with recursive one-step branching.

    depth=1: all policies to completion
    depth=2: all policies one step, then all policies to completion
    depth=3: all policies one step, then all policies one step, then all policies to completion
    """
    global TOURNAMENT_DEPTH
    depth = getattr(TOURNAMENT_DEPTH, "depth", 1)
    global TOURNAMENT_DEBUG

    master_engine = resource.engine
    current_history = list(master_engine.decision_log)
    master_seed = master_engine.seed
    fork_scenario_config: ScenarioConfig | None = None
    try:
        pickle.dumps(master_engine.scenario_config)
        fork_scenario_config = master_engine.scenario_config
    except Exception:
        fork_scenario_config = None

    candidates = _get_tournament_candidates()
    if not candidates:
        raise Exception("Tournament candidate set is empty. Configure at least one non-tournament policy.")

    global DISABLE_TOURNAMENT_MULTIPROCESSING
    use_multiprocessing = not DISABLE_TOURNAMENT_MULTIPROCESSING
    t0 = time.perf_counter()
    if len(current_history) == 0:
        clear_tournament_cache()

    if TOURNAMENT_DEBUG:
        print(f"[tournament_recursive] t={env.now:.2f} depth={depth} history={len(current_history)} " f"disasters={len(disasters)} mp={use_multiprocessing}")

    if depth == 1:
        tasks = [(p, master_seed, current_history, None, fork_scenario_config) for p in candidates]
        if use_multiprocessing:
            with multiprocessing.Pool(processes=len(candidates)) as pool:
                results = pool.starmap(evaluate_policy_worker, tasks)
        else:
            results = [evaluate_policy_worker(*task) for task in tasks]
    else:
        tasks = [(p, master_seed, current_history, depth, fork_scenario_config) for p in candidates]
        if use_multiprocessing:
            with multiprocessing.Pool(processes=len(candidates)) as pool:
                results = pool.starmap(evaluate_policy_recursive_tree, tasks)
        else:
            results = [evaluate_policy_recursive_tree(*task) for task in tasks]

    best_result = None
    best_score = float("inf")
    for result in results:
        score = _score_policy_result(result)
        if score < best_score:
            best_score = score
            best_result = result

    if best_result is None:
        raise Exception("Recursive tournament produced no policy results.")

    master_engine.tournament_decisions.append((env.now, f"{best_result['policy']} (recursive)"))

    if TOURNAMENT_DEBUG:
        elapsed = time.perf_counter() - t0
        successes = sum(1 for r in results if r["success"])
        print(
            f"[tournament_recursive] chose={best_result['policy']} move={best_result['move_id']} "
            f"successes={successes}/{len(results)} score={best_score:.2f} "
            f"wall={elapsed:.2f}s cache={len(ROLLOUT_CACHE)}"
        )

    targets = [d for d in disasters if d.id == best_result["move_id"]]
    if not targets:
        if TOURNAMENT_DEBUG:
            print(f"[tournament_recursive] move_id={best_result['move_id']} not in current disasters " f"{[d.id for d in disasters]}")
        fallback_policy = next((p for p in POLICIES if p.name == best_result["policy"]), None)
        if fallback_policy is not None and not fallback_policy.name.startswith("tournament"):
            return fallback_policy.func(resource, disasters, env)
        return None

    return targets[0]


# ============================================================================
# MARK: Seasonal-Aware Policies
# ============================================================================


def get_seasonal_priority(disaster: Disaster) -> tuple[int, float]:
    """Get priority score based on current season and disaster type.

    Returns (priority_level, severity) - lower priority_level = more urgent.
    """
    from .calendar import Season

    engine = disaster.engine
    priority = 100  # Default priority

    # Get season if calendar is available
    season = None
    if engine.calendar is not None:
        season = engine.calendar.get_season()

    # Snow events: highest priority in winter (they auto-resolve but can worsen)
    if isinstance(disaster, SnowEvent):
        if season == Season.WINTER:
            priority = 1  # Urgent - snow is falling now
        elif season == Season.FALL:
            priority = 3  # Getting close to winter
        else:
            priority = 10  # Low priority in off-season

    # Landslides: highest in wet seasons (winter, spring, fall)
    elif isinstance(disaster, Landslide):
        if season in [Season.WINTER, Season.SPRING]:
            priority = 2  # High priority - wet season
        elif season == Season.FALL:
            priority = 4  # Coming wet season
        else:
            priority = 8  # Summer - dry, less likely

    # Unknown disaster types get medium priority
    else:
        priority = 5

    severity = get_severity(disaster)
    return (priority, severity)


def _has_required_partners(disaster: Disaster, resource: Resource) -> bool:
    """Check if the disaster has (or will have) the required partner resources.

    For a Landslide (needs EXCAVATOR + TRUCK):
    - If we're dispatching a TRUCK, check if there's an EXCAVATOR already there
    - If we're dispatching an EXCAVATOR, check if there's a TRUCK already there
    """
    needed = disaster.needed_resources()

    # If this resource type is not needed, return True (any disaster is valid)
    if resource.resource_type not in needed:
        return True

    # Check partner resources
    for rt in needed:
        if rt == resource.resource_type:
            continue  # Skip self

        # Check if partner resource is already at the site
        if len(disaster.roster.get(rt, [])) > 0:
            return True

    # No partner resources available
    return False


def seasonal_priority_policy(resource: Resource, disasters: list[Disaster], _env: simpy.Environment) -> Disaster:
    """
    Prioritizes disasters based on current season.
    Winter: Snow events first (they worsen over time if not cleared)
    Wet seasons: Landslides first
    Summer: Wildfire debris first

    CRITICAL: Only picks disasters where partner resources are available.
    """
    candidates = [d for d in disasters if is_useful(resource, d)]
    if not candidates:
        candidates = disasters

    # Filter to only disasters with required partners available
    # This prevents sending a truck to a landslide with no excavator
    candidates_with_partners = [d for d in candidates if _has_required_partners(d, resource)]
    if candidates_with_partners:
        candidates = candidates_with_partners
    # If no candidates have partners, fall back to all (let it fail naturally)

    # Sort by seasonal priority (primary) and severity (secondary)
    sorted_disasters = sorted(candidates, key=get_seasonal_priority)

    return sorted_disasters[0]


def weather_aware_policy(resource: Resource, disasters: list[Disaster], _env: simpy.Environment) -> Disaster:
    """
    Uses weather conditions to prioritize.
    High rain: landslides/floods first (will worsen)
    Low temp: snow first
    High drought: wildfire debris first

    CRITICAL: Only picks disasters where partner resources are available.
    """
    candidates = [d for d in disasters if is_useful(resource, d)]
    if not candidates:
        candidates = disasters

    # Filter to only disasters with required partners available
    candidates_with_partners = [d for d in candidates if _has_required_partners(d, resource)]
    if candidates_with_partners:
        candidates = candidates_with_partners

    engine = resource.engine

    # Get weather factors
    weather_score = 0.0
    if engine.calendar is not None:
        rain_factor = engine.calendar.get_weather_factor("landslide")
        snow_factor = engine.calendar.get_weather_factor("snow")
        fire_factor = engine.calendar.get_weather_factor("wildfire_debris")

        # Calculate weather urgency score for each disaster
        def get_weather_urgency(d: Disaster) -> float:
            urgency = 0.0
            disaster_name = type(d).__name__
            if disaster_name == "Landslide":
                urgency = rain_factor * 100
            elif disaster_name == "SnowEvent":
                urgency = snow_factor * 100
            elif disaster_name == "WildfireDebris":
                urgency = fire_factor * 100
            else:
                urgency = 50  # Default
            return urgency + get_severity(d)

        sorted_disasters = sorted(candidates, key=get_weather_urgency, reverse=True)
        return sorted_disasters[0]

    # Fallback if no calendar
    return closest_neighbor_policy(resource, disasters, _env)


def resource_season_match_policy(resource: Resource, disasters: list[Disaster], _env: simpy.Environment) -> Disaster:
    """
    Matches resource type to seasonal conditions.
    Snowplows: only in winter (or don't dispatch at all in summer)
    Excavators/Trucks: preferred in dry seasons

    CRITICAL: Only picks disasters where partner resources are available.
    """
    candidates = [d for d in disasters if is_useful(resource, d)]
    if not candidates:
        candidates = disasters

    # Filter to only disasters with required partners available
    candidates_with_partners = [d for d in candidates if _has_required_partners(d, resource)]
    if candidates_with_partners:
        candidates = candidates_with_partners

    engine = resource.engine

    # Check if we have calendar and what season it is
    season = None
    if engine.calendar is not None:
        season = engine.calendar.get_season()

    # Snowplows: only useful in winter/fall
    if resource.resource_type == ResourceType.SNOWPLOW:
        snow_candidates = [d for d in candidates if isinstance(d, SnowEvent)]
        if snow_candidates:
            candidates = snow_candidates
        elif season in [Season.SUMMER, Season.SPRING]:
            # Don't dispatch snowplows in summer - they're wasted
            return candidates[0] if candidates else disasters[0]

    # Excavators/Trucks: prefer non-snow in winter
    elif resource.resource_type in [ResourceType.EXCAVATOR, ResourceType.TRUCK]:
        non_snow = [d for d in candidates if not isinstance(d, SnowEvent)]
        if non_snow and season == Season.WINTER:
            candidates = non_snow

    # Sort by severity among filtered candidates
    return min(candidates, key=get_severity)


# ============================================================================
# MARK: Metrics-Aware Policies
# ============================================================================


def cost_efficient_policy(resource: Resource, disasters: list[Disaster], _env: simpy.Environment) -> Disaster:
    """
    Prioritizes disasters that can be resolved quickly with minimal cost.
    Uses engine metrics to estimate efficiency.

    CRITICAL: Only picks disasters where partner resources are available.
    """
    candidates = [d for d in disasters if is_useful(resource, d)]
    if not candidates:
        candidates = disasters

    # Filter to only disasters with required partners available
    candidates_with_partners = [d for d in candidates if _has_required_partners(d, resource)]
    if candidates_with_partners:
        candidates = candidates_with_partners

    engine = resource.engine

    def get_cost_efficiency(d: Disaster) -> float:
        """Calculate cost efficiency: lower severity = faster resolution = less cost."""
        severity = get_severity(d)
        # Higher severity = more work = more cost = worse efficiency
        # We want LOW severity (quick jobs) first
        return severity

    # Sort by cost efficiency (lowest severity first = most cost-efficient)
    sorted_candidates = sorted(candidates, key=get_cost_efficiency)
    return sorted_candidates[0]


def population_impact_policy(resource: Resource, disasters: list[Disaster], _env: simpy.Environment) -> Disaster:
    """
    Prioritizes disasters that affect more people.
    Uses road type and severity to estimate population impact.

    CRITICAL: Only picks disasters where partner resources are available.
    """
    candidates = [d for d in disasters if is_useful(resource, d)]
    if not candidates:
        candidates = disasters

    # Filter to only disasters with required partners available
    candidates_with_partners = [d for d in candidates if _has_required_partners(d, resource)]
    if candidates_with_partners:
        candidates = candidates_with_partners

    from .metrics_tracker import PopulationImpact

    def get_population_impact(d: Disaster) -> int:
        """Estimate population impact for a disaster."""
        size = get_severity(d)
        # Default to secondary road if unknown
        return PopulationImpact.estimate_from_disaster_size(size, "secondary")

    # Sort by population impact (highest first)
    sorted_candidates = sorted(candidates, key=get_population_impact, reverse=True)
    return sorted_candidates[0]


def budget_aware_policy(resource: Resource, disasters: list[Disaster], _env: simpy.Environment) -> Disaster | None:
    """
    Prioritizes based on current budget utilization.
    When budget is low, prioritizes small jobs to stretch resources.
    When budget is high, can take on larger jobs.
    When budget is exhausted, skips low-severity disasters.

    CRITICAL: Only picks disasters where partner resources are available.
    """
    candidates = [d for d in disasters if is_useful(resource, d)]
    if not candidates:
        candidates = disasters

    # Filter to only disasters with required partners available
    candidates_with_partners = [d for d in candidates if _has_required_partners(d, resource)]
    if candidates_with_partners:
        candidates = candidates_with_partners

    engine = resource.engine

    # Get current budget utilization
    budget_util = 0.0
    budget_exhausted = False
    if hasattr(engine, "scenario_config") and engine.scenario_config.track_costs:
        summary = engine.get_summary()
        budget_util = summary.get("budget_utilization", 0.0)
        budget_exhausted = summary.get("budget_exhausted", False)

    def get_adjusted_priority(d: Disaster) -> float:
        """Adjust priority based on budget situation."""
        severity = get_severity(d)

        # If budget is exhausted, skip low-severity disasters
        if budget_exhausted:
            # Only respond to high-severity disasters
            if severity < 0.7:
                return float("inf")  # Skip these
            return severity * 0.5  # Deprioritize everything when exhausted

        # If budget is running low (>80%), prioritize small jobs
        if budget_util > 0.8:
            # Heavily favor small jobs
            return severity * 2
        elif budget_util > 0.5:
            # Moderately favor small jobs
            return severity * 1.5
        else:
            # Normal priority
            return severity

    sorted_candidates = sorted(candidates, key=get_adjusted_priority)

    # If budget is exhausted, skip first candidate if it was marked to skip
    if budget_exhausted:
        for d in sorted_candidates:
            if get_severity(d) >= 0.7:
                return d
        return None  # No disasters worth responding to

    return sorted_candidates[0]


def resource_efficiency_policy(resource: Resource, disasters: list[Disaster], _env: simpy.Environment) -> Disaster:
    """
    Prioritizes matching the right resource type to the right job.
    Trucks go to jobs that need trucking, excavators to excavation.

    CRITICAL: Only picks disasters where partner resources are available.
    """
    candidates = [d for d in disasters if is_useful(resource, d)]
    if not candidates:
        candidates = disasters

    # Filter to only disasters with required partners available
    candidates_with_partners = [d for d in candidates if _has_required_partners(d, resource)]
    if candidates_with_partners:
        candidates = candidates_with_partners

    def get_resource_fit(d: Disaster) -> float:
        """How well this resource type fits the job."""
        needed = d.needed_resources()

        # Perfect match: this exact resource type is needed
        if resource.resource_type in needed:
            # Additional factor: how many of this type already at site?
            current_count = len(d.roster.get(resource.resource_type, []))
            # Fewer of our type = more need for us = better fit
            return 1.0 / (1.0 + current_count)

        # Not needed - low priority
        return 0.0

    # Sort by resource fit (best match first)
    sorted_candidates = sorted(candidates, key=get_resource_fit, reverse=True)
    return sorted_candidates[0]


# ============================================================================
# MARK: Policy Map
# ============================================================================


POLICIES: list[Policy] = [
    # Policy("random", random_policy),
    Policy("first_priority", first_priority_policy),
    Policy("split_excavator", split_excavator_policy),
    # Policy("closest_neighbor", closest_neighbor_policy),
    Policy("smallest_job_first", smallest_job_first_policy),
    Policy("balanced_ratio", balanced_ratio_policy),
    Policy("smart_split", smart_split_policy),
    Policy("cost_function", cost_function_policy),
    # Policy("gravity", gravity_policy),
    Policy("chain_gang", chain_gang_policy),
    Policy("tournament", tournament_policy_func),
    Policy("tournament_recursive", tournament_recursive_policy_func),
    Policy("seasonal_priority", seasonal_priority_policy),
    # Policy("weather_aware", weather_aware_policy),
    Policy("resource_season_match", resource_season_match_policy),
    # New metrics-aware policies
    Policy("cost_efficient", cost_efficient_policy),
    Policy("population_impact", population_impact_policy),
    Policy("budget_aware", budget_aware_policy),
    Policy("resource_efficiency", resource_efficiency_policy),
]
