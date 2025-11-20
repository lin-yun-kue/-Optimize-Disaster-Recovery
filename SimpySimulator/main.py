from __future__ import annotations
import simpy
import random
import statistics
from typing import List, Dict, TypedDict, DefaultDict
import matplotlib.pyplot as plt
from math import pi as PI, cos, sin
from simpy.core import EmptySchedule
from argparse import ArgumentParser
from collections import defaultdict

from simulation import *

from policies import (
    random_policy,
    first_priority_policy,
    split_excavator_policy,
    closest_neighbor_policy,
    smallest_job_first_policy,
    balanced_ratio_policy,
    smart_split_policy,
    cost_function_policy,
    gravity_policy,
    chain_gang_policy,
)

# --- Configuration ---
POLICY_MAP = {
    "random": random_policy,
    "first_priority": first_priority_policy,
    "split_excavator": split_excavator_policy,
    "closest_neighbor": closest_neighbor_policy,
    "smallest_job_first": smallest_job_first_policy,
    "balanced_ratio": balanced_ratio_policy,
    "smart_split": smart_split_policy,
    "cost_function": cost_function_policy,
    "gravity": gravity_policy,
    "chain_gang": chain_gang_policy,
}

MAX_SIM_TIME = 20_000


# --- Simulation Processes ---
def loop(env: simpy.Environment, idle_resources: IdleResources, disaster_store: DisasterStore, policy):
    """
    The main orchestrator. It waits for ANY resource to become available
    at the depot, then asks the policy where to send it.
    """
    while True:
        resource = yield env.process(idle_resources.get_any_resource())
        yield env.process(disaster_store.wait_for_any())
        yield env.process(policy(resource, disaster_store, env))


def add_disasters(env: simpy.Environment, disaster_store: DisasterStore, dump_site: DumpSite):
    min_size = SimulationConfig.LANDSLIDE_MIN_SIZE
    max_size = SimulationConfig.LANDSLIDE_MAX_SIZE

    initial_delay = random.randint(10, 20)
    yield env.timeout(initial_delay)

    for i in range(SimulationConfig.NUM_STARTING_LANDSLIDES):
        landslide_size = random.randint(min_size, max_size)
        landslide = Landslide(env, landslide_size, dump_site)
        disaster_store.put(landslide)

    for i in range(SimulationConfig.NUM_LANDSLIDES):
        landslide_size = random.randint(min_size, max_size)
        landslide = Landslide(env, landslide_size, dump_site)
        disaster_store.put(landslide)

        delay = random.randint(100, 500)
        if i < SimulationConfig.NUM_LANDSLIDES - 1:
            yield env.timeout(delay)


# --- Visualization Helper Functions ---
def setup_plot():
    fig, axs = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={"height_ratios": [3, 1]})
    axs[0].set_xlim(-120, 120)
    axs[0].set_ylim(-120, 120)
    axs[0].set_aspect("equal")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Dirt")
    axs[1].grid(True)
    plt.ion()
    return fig, axs


def update_plot(
    axs,
    sim_time,
    time_data,
    dirt_data,
    depot,
    disaster_store,
    trucks,
    excavators,
    policy_name,
    seed,
    env,
):
    ax_map, ax_dirt = axs
    ax_map.clear()
    ax_map.set_xlim(-120, 120)
    ax_map.set_ylim(-120, 120)
    ax_map.set_title(f"Policy: {policy_name} | Seed: {seed} | Time: {sim_time:.2f}")

    # Plot Depot
    ax_map.plot(depot.location[0], depot.location[1], "gs", markersize=15, label="Depot")

    # Plot Landslides
    all_landslides: List[Landslide] = disaster_store.items
    for ls in all_landslides:
        ratio = ls.dirt.level / ls.initial_size
        ax_map.plot(ls.location[0], ls.location[1], "ro", markersize=10 + 20 * ratio, alpha=0.7)
        ax_map.text(
            ls.location[0],
            ls.location[1],
            f"L{ls.id}\n{ls.dirt.level}",
            ha="center",
            va="center",
            fontsize=9,
        )

    # Plot Resources
    for i, r in enumerate(trucks + excavators):
        frac = i / len(trucks) * PI * 2
        loc1 = (cos(frac) * 10 + r.location[0], sin(frac) * 10 + r.location[1])
        loc2 = (
            cos(frac) * 10 + r.prev_location[0],
            sin(frac) * 10 + r.prev_location[1],
        )

        time_frac = min(
            1,
            max(
                0,
                (1 if r.move_time == 0 else (env.now - r.move_start_time) / r.move_time),
            ),
        )

        loc = (
            loc1[0] * time_frac + loc2[0] * (1 - time_frac),
            loc1[1] * time_frac + loc2[1] * (1 - time_frac),
        )

        if r.resource_type == ResourceType.TRUCK:
            marker, color, label = "^", "blue", "T"
        else:
            marker, color, label = "P", "orange", "E"

        ax_map.plot(loc[0], loc[1], marker=marker, color=color, markersize=8)
        ax_map.text(loc[0] + 2, loc[1] + 2, f"{i}", color=color, fontsize=8)

    ax_dirt.clear()
    if dirt_data and len(time_data) > 0:
        ids = sorted(dirt_data.keys())
        y_data = [dirt_data[i] for i in ids]
        labels = [f"L{i}" for i in ids]
        ax_dirt.stackplot(time_data, *y_data, labels=labels, alpha=0.8, step="post")
        ax_dirt.legend(loc="upper left", fontsize="small", framealpha=0.5)

    plt.draw()
    plt.pause(0.001)


# --- Main Execution Function ---
def run_simulation(policy_name, policy_func, seed_value, live_plot=False):
    print(f"--- Running Policy {policy_name}, Seed {seed_value} ---")

    # 1. RESET RANDOMNESS
    # Important: We set the seed here so that for a specific seed_value (e.g. 42),
    # the landslides appear at the exact same time and size for every policy.
    random.seed(seed_value)

    # 2. SETUP ENVIRONMENT
    env = simpy.Environment()
    disaster_store = DisasterStore(env)
    ResourceNode.disaster_store = disaster_store
    idle_resources = IdleResources(env)
    ResourceNode.idle_resources = idle_resources
    depot = Depot(env)

    dump_site = DumpSite(env)

    all_trucks = [Resource(i, ResourceType.TRUCK, env, depot) for i in range(SimulationConfig.NUM_TRUCKS)]
    all_excavators = [
        Resource(i + 100, ResourceType.EXCAVATOR, env, depot) for i in range(SimulationConfig.NUM_EXCAVATORS)
    ]

    for r in all_excavators + all_trucks:
        depot.transfer_resource(r)

    # 3. SETUP PROCESSES
    disasters_event = env.process(add_disasters(env, disaster_store, dump_site))
    env.process(loop(env, idle_resources, disaster_store, policy_func))

    # 4. SETUP VISUALIZATION (Optional)
    fig, axs = (None, None)
    if live_plot:
        fig, axs = setup_plot()

    time_points = []
    known_landslides: Dict[int, Landslide] = {}
    landslide_histories: Dict[int, List[float]] = {}

    # 5. RUN LOOP
    simulation_succeeded = False

    while True:
        try:
            target_time = env.now + 1
            # while env.now < target_time:
            #     env.step()
            env.run(until=target_time)
        except EmptySchedule:
            # CASE 1: The schedule is empty.
            # Did we finish?
            if disasters_event.triggered and len(disaster_store.items) == 0:
                simulation_succeeded = True
            else:
                # We ran out of events but disasters remain -> FAILURE
                simulation_succeeded = False
                if live_plot:
                    print(f"   [!] Policy {policy_name} stalled at {env.now} with items remaining.")
            break
        except Exception as e:
            print(f"   [!] Exception: {e}")
            simulation_succeeded = False
            if live_plot:
                print(f"   [!] Policy {policy_name} failed at {env.now}.")
            break

        # CASE 2: Safety Cutout (Infinite Loops)
        if env.now > MAX_SIM_TIME:
            simulation_succeeded = False
            if live_plot:
                print(f"   [!] Policy {policy_name} timed out at {env.now}.")
            break

        # Data Collection
        for ls in disaster_store.items:
            if ls.id not in known_landslides:
                known_landslides[ls.id] = ls
                landslide_histories[ls.id] = [0] * len(time_points)

        time_points.append(env.now)
        for ls_id, ls_obj in known_landslides.items():
            val = ls_obj.dirt.level if ls_obj in disaster_store.items else 0
            landslide_histories[ls_id].append(val)

        if live_plot:
            update_plot(
                axs,
                env.now,
                time_points,
                landslide_histories,
                depot,
                disaster_store,
                all_trucks,
                all_excavators,
                policy_name,
                seed_value,
                env,
            )

        # CASE 3: Success during loop
        if disasters_event.triggered and len(disaster_store.items) == 0:
            simulation_succeeded = True
            break

    if live_plot:
        plt.close(fig)

    return simulation_succeeded, env.now


class PolicyResult(TypedDict):
    success: List[float]
    fail: int


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Show live plot for each policy")
    parser.add_argument("--seeds", type=int, default=3, help="Number of different seeds to test")
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Policy to run (random, first_priority, split_excavator, closest_neighbor, smallest_job_first, balanced_ratio, smart_split, cost_function, gravity)",
    )
    args = parser.parse_args()

    print(f"\n--- STARTING COMPARISON ({args.seeds} Seeds) ---\n")

    # Store results as: {"policy_name": {"success": [times...], "fail": count}}
    aggregated_results: DefaultDict[str, PolicyResult] = defaultdict(lambda: {"success": [], "fail": 0})

    for seed in range(args.seeds):
        print(f"--- {seed} ---")
        if args.policy is not None:
            func = POLICY_MAP[args.policy]
            success, duration = run_simulation(args.policy, func, seed_value=seed, live_plot=args.live)
            if success:
                aggregated_results[args.policy]["success"].append(duration)
            else:
                aggregated_results[args.policy]["fail"] += 1
        else:
            for name, func in POLICY_MAP.items():
                success, duration = run_simulation(name, func, seed_value=seed, live_plot=args.live)

                if success:
                    aggregated_results[name]["success"].append(duration)
                else:
                    aggregated_results[name]["fail"] += 1

    print("\n" + "=" * 85)
    print(f"{'POLICY':<20} | {'SUCCESS %':<10} | {'AVG TIME':<10} | {'STDEV':<10} | {'MIN':<8} | {'MAX':<8}")
    print("-" * 85)

    # Calculate statistics
    final_stats = []
    for name, data in aggregated_results.items():
        success_times = data["success"]
        fail_count = data["fail"]
        total_runs = len(success_times) + fail_count

        success_rate = (len(success_times) / total_runs) * 100 if total_runs > 0 else 0

        if success_times:
            avg = statistics.mean(success_times)
            stdev = statistics.stdev(success_times) if len(success_times) > 1 else 0.0
            mn = min(success_times)
            mx = max(success_times)
        else:
            avg = float("inf")  # Sort failures to bottom
            stdev = 0.0
            mn = 0
            mx = 0

        final_stats.append((name, success_rate, avg, stdev, mn, mx))

    # Sort by Success Rate (Desc), then Avg Time (Asc)
    final_stats.sort(key=lambda x: (-x[1], x[2]))

    for name, rate, avg, stdev, mn, mx in final_stats:
        avg_str = f"{avg:.2f}" if avg != float("inf") else "N/A"
        stdev_str = f"{stdev:.2f}" if avg != float("inf") else "N/A"
        mn_str = f"{mn:.0f}" if avg != float("inf") else "N/A"
        mx_str = f"{mx:.0f}" if avg != float("inf") else "N/A"

        print(f"{name:<20} | {rate:<9.1f}% | {avg_str:<10} | {stdev_str:<10} | {mn_str:<8} | {mx_str:<8}")
    print("=" * 85 + "\n")
