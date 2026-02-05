import time
import simpy
from typing import List, Callable, TYPE_CHECKING
import math
from .simulation import *
from .engine import SimPySimulationEngine
import multiprocessing


@dataclass
class Policy:
    name: str
    func: Callable[[Resource, list[Disaster], simpy.Environment], Disaster]


# ============================================================================
# MARK: Utils
# ============================================================================


def get_dist(r: Resource, node: ResourceNode):
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


def random_policy(resource: Resource, disasters: list[Disaster], env: simpy.Environment) -> Disaster:
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


def first_priority_policy(resource: Resource, disasters: list[Disaster], env: simpy.Environment) -> Disaster:
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


def split_excavator_policy(resource: Resource, disasters: list[Disaster], env: simpy.Environment) -> Disaster:
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


def closest_neighbor_policy(resource: Resource, disasters: list[Disaster], env: simpy.Environment) -> Disaster:
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


def smallest_job_first_policy(resource: Resource, disasters: list[Disaster], env: simpy.Environment) -> Disaster:
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


def balanced_ratio_policy(resource: Resource, disasters: list[Disaster], env: simpy.Environment) -> Disaster:
    """
    Attempts to maintain an ideal Truck-to-Excavator ratio to minimize queuing.
    """

    candidates = [d for d in disasters if is_useful(resource, d)]
    if not candidates:
        candidates = disasters

    rt = resource.resource_type

    def get_ratio_score(d):
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


def smart_split_policy(resource: Resource, disasters: list[Disaster], env: simpy.Environment) -> Disaster:
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
    viable_sites = []
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


def cost_function_policy(resource: Resource, disasters: list[Disaster], env: simpy.Environment) -> Disaster:
    """
    Calculates Cost: Distance + Queue + Completion Penalty.
    """
    candidates = [d for d in disasters if is_useful(resource, d)]
    if not candidates:
        candidates = disasters

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

    return target


# ============================================================================
# MARK: Gravity Policy
# ============================================================================


def gravity_policy(resource: Resource, disasters: list[Disaster], env: simpy.Environment) -> Disaster:
    """
    High Pull = High Severity + High Priority.
    Low Pull = High Distance + High Queue.
    """
    candidates = [d for d in disasters if is_useful(resource, d)]
    if not candidates:
        candidates = disasters

    def get_score(d):
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


def chain_gang_policy(resource: Resource, disasters: list[Disaster], env: simpy.Environment) -> Disaster:
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


def evaluate_policy_worker(policy: Policy, seed: int, history: list[int]) -> PolicyResult:
    # Test every policy
    try:
        # 1. Create Fresh Engine
        sim_fork = SimPySimulationEngine(policy, seed, live_plot=False)
        sim_fork.initialize_world()  # Populate world exactly as master was populated

        # 2. Load History for Replay
        sim_fork.replay_buffer = list(history)

        # 3. Run to completion
        # The loop will consume replay_buffer, then capture the NEXT decision in sim_fork.branch_decision
        success = sim_fork.run()

        time_taken = sim_fork.get_summary()["non_idle_time"]

        return {
            "success": success,
            "time": time_taken,
            "policy": policy.name,
            "move_id": sim_fork.branch_decision,
        }
    except Exception as e:
        print(f"Exception: {e}")
        raise e


def tournament_policy_func(resource: Resource, disasters: list[Disaster], env: simpy.Environment) -> Disaster:
    """
    The Meta-Policy (Replay Version).
    1. Instantiate N new simulation engines (one per candidate policy).
    2. Feed them the `decision_log` from the Master simulation.
    3. Run them. They will fast-forward replay history, then switch to candidate policy.
    4. Compare completion times.
    5. Extract the specific move the winner made at the branching point.
    """
    master_engine = resource.engine

    # Get current history from Master
    current_history = list(master_engine.decision_log)
    master_seed = master_engine.seed

    candidates = [p for p in POLICIES if p.name != "tournament"]

    tasks = [(p, master_seed, current_history) for p in candidates]

    with multiprocessing.Pool(processes=len(candidates)) as pool:
        results = pool.starmap(evaluate_policy_worker, tasks)

    best_result = None
    min_time = float("inf")

    for result in results:
        if result["success"] and result["time"] < min_time:
            min_time = result["time"]
            best_result = result

    if best_result is None:
        raise Exception("No policy was able to complete the simulation.")

    # Record winner
    master_engine._tournament_decisions.append((env.now, best_result["policy"]))

    # Find the disaster object in the Master's store that matches the ID
    target = [d for d in disasters if d.id == best_result["move_id"]][0]
    return target

# ============================================================================
# MARK: KSwitchPolicy
# ============================================================================

class KSwitchPolicy:
    """
    分岔後：
      - 前 K 次 decision 用 first_policy
      - 第 K+1 次開始都用 rest_policy
    K=1 => 下一次 decision 用 first_policy，後面都 rest_policy
    """
    def __init__(self, first_policy: Policy, rest_policy: Policy, k: int):
        if k < 0:
            raise ValueError("k must be >= 0")
        self.first_policy = first_policy
        self.rest_policy = rest_policy
        self.k = k

        self.name = f"switch({first_policy.name}->{rest_policy.name},k={k})"

        self._post_branch_decision_count = 0
        self.first_move_id = None  # 分岔後第1次 decision 的結果（用來回主模擬）
        self.func = self._func

    def _func(self, resource, disasters, env):
        self._post_branch_decision_count += 1

        if self._post_branch_decision_count <= self.k:
            chosen = self.first_policy.func(resource, disasters, env)
        else:
            chosen = self.rest_policy.func(resource, disasters, env)

        # 記錄分岔後第 1 次 decision 的選擇（重要：回主模擬用）
        if self._post_branch_decision_count == 1:
            # 這裡假設 policy 回傳的是 Disaster 物件，且有 .id
            self.first_move_id = chosen.id

        return chosen
    
def evaluate_policy_worker_switch(first_policy: Policy, rest_policy: Policy, k: int, seed: int, history: list[int]) -> PolicyResult:
    try:
        # 1) 用 wrapper policy 包起來（engine 仍然只吃到「一個 policy」）
        switch_policy = KSwitchPolicy(first_policy, rest_policy, k)

        # 2) Create Fresh Engine
        sim_fork = SimPySimulationEngine(switch_policy, seed, live_plot=False)
        sim_fork.initialize_world()

        # 3) Load History for Replay
        sim_fork.replay_buffer = list(history)

        # 4) Run to completion
        success = sim_fork.run()

        time_taken = sim_fork.get_summary()["non_idle_time"]

        # 分岔後第 1 次 decision 的 move_id（K=1 就是你要的「下一步」）
        move_id = switch_policy.first_move_id

        # 保底：如果你想兼容 engine 的 branch_decision 行為，也可以 fallback
        if move_id is None:
            move_id = sim_fork.branch_decision

        return {
            "success": success,
            "time": time_taken,
            "policy": switch_policy.name,   # 記錄是 switch(...)
            "move_id": move_id,
        }
    except Exception as e:
        print(f"Exception: {e}")
        raise

def k_tournament_policy_func(resource: Resource, disasters: list[Disaster], env: simpy.Environment) -> Disaster:
    """
    The Meta-Policy (Replay Version).
    1. Instantiate N new simulation engines (one per candidate policy).
    2. Feed them the `decision_log` from the Master simulation.
    3. Run them. They will fast-forward replay history, then switch to candidate policy.
    4. Compare completion times.
    5. Extract the specific move the winner made at the branching point.
    """
    master_engine = resource.engine

    # Get current history from Master
    current_history = list(master_engine.decision_log)
    master_seed = master_engine.seed

    exclude_policy = ["tournament", "k-tournament"]
    candidates = [p for p in POLICIES if p.name not in exclude_policy]

    k=1
    tasks = []
    for first in candidates:
        for rest in candidates:
            tasks.append((first, rest, k, master_seed, current_history))


    with multiprocessing.Pool(processes=len(candidates)) as pool:
        results = pool.starmap(evaluate_policy_worker_switch, tasks)

    best_result = None
    min_time = float("inf")

    for result in results:
        if result["success"] and result["time"] < min_time:
            min_time = result["time"]
            best_result = result

    if best_result is None:
        raise Exception("No policy was able to complete the simulation.")

    # Record winner
    master_engine._tournament_decisions.append((env.now, best_result["policy"]))

    # Find the disaster object in the Master's store that matches the ID
    target = [d for d in disasters if d.id == best_result["move_id"]][0]
    return target

# ============================================================================
# MARK: Policy Map
# ============================================================================


POLICIES: list[Policy] = [
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
    Policy("tournament", tournament_policy_func),
    Policy("k-tournament", k_tournament_policy_func),
]
