from __future__ import annotations
import simpy
import random
import statistics
from typing import List, Dict, TypedDict, DefaultDict, TYPE_CHECKING
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from math import pi as PI, cos, sin
from simpy.core import EmptySchedule
from argparse import ArgumentParser
from collections import defaultdict
from engine import SimPySimulationEngine
from simulation import *


# --- Main Execution Function ---
def setup_simulation(engine: SimPySimulationEngine):
    depot = Depot(engine)
    dump_site = DumpSite(engine)

    engine.resource_nodes.append(depot)
    engine.resource_nodes.append(dump_site)

    all_resources = []
    rid = 0
    for _ in range(SimulationConfig.NUM_TRUCKS):
        all_resources.append(Resource(rid, ResourceType.TRUCK, engine))
        rid += 1
    for _ in range(SimulationConfig.NUM_EXCAVATORS):
        all_resources.append(Resource(rid, ResourceType.EXCAVATOR, engine))
        rid += 1
    for _ in range(SimulationConfig.NUM_FIRE_TRUCKS):
        all_resources.append(Resource(rid, ResourceType.FIRE_TRUCK, engine))
        rid += 1
    for _ in range(SimulationConfig.NUM_AMBULANCES):
        all_resources.append(Resource(rid, ResourceType.AMBULANCE, engine))
        rid += 1

    for r in all_resources:
        depot.transfer_resource(r)


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

    from policies import POLICIES

    for seed in range(args.seeds):
        print(f"--- {seed} ---")
        if args.policy is not None:
            policy = [p for p in POLICIES if p.name == args.policy][0]
            # success, duration = run_simulation(policy, seed_value=seed, live_plot=args.live)
            engine = SimPySimulationEngine(policy=policy, seed=seed, live_plot=args.live)
            setup_simulation(engine)
            success = engine.run()
            duration = engine.get_summary()["non_idle_time"]
            if success:
                aggregated_results[args.policy]["success"].append(duration)
            else:
                aggregated_results[args.policy]["fail"] += 1
        else:
            for policy in POLICIES:
                engine = SimPySimulationEngine(policy=policy, seed=seed, live_plot=args.live)
                setup_simulation(engine)
                success = engine.run()
                duration = engine.get_summary()["non_idle_time"]

                if success:
                    aggregated_results[policy.name]["success"].append(duration)
                else:
                    aggregated_results[policy.name]["fail"] += 1

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
