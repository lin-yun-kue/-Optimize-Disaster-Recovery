"""
Benchmark file that tests all policies and PPO model on shared seeds.
Tests both with and without GIS for comparison.
"""

from __future__ import annotations
import json
import statistics
import time
from typing import Any, TypedDict
from collections import defaultdict
from argparse import ArgumentParser
import numpy as np

from stable_baselines3 import PPO
from SimPyTest.gym import DisasterResponseEnv
from SimPyTest.engine import SimPySimulationEngine, ScenarioConfig
from SimPyTest.invariants import validate_engine_invariants
from SimPyTest.policies import POLICIES, set_tournament_depth
from SimPyTest.gis_utils import Depot, GISConfig, Landfill, build_road_graph, load_roads, load_and_prune_roads


class PolicyResult(TypedDict):
    success: list[float]
    success_wall: list[float]
    fail: int
    invariant_fail: int


depots: list[Depot] = [
    {
        "Longitude": -123.92616052570274,
        "Latitude": 46.16262226957932,
        "Name": "ODOT Warrenton",
        "numExc": 1,
        "numTrucks": 1,
        "color": "green",
    },
    {
        "Longitude": -123.92250095724509,
        "Latitude": 45.984013191819535,
        "Name": "ODOT Seaside",
        "numExc": 1,
        "numTrucks": 2,
        "color": "yellow",
    },
    {
        "Longitude": -123.81788435312336,
        "Latitude": 45.91011185257814,
        "Name": "ODOT Necanium",
        "numExc": 1,
        "numTrucks": 3,
        "color": "purple",
    },
]

landfills: list[Landfill] = [
    {"Label": "A", "Longitude": -123.70105856996436, "Latitude": 45.91375387223106, "Name": "Random Landfill"},
    {"Label": "B", "Longitude": -123.80828890576535, "Latitude": 46.17804487993376, "Name": "Astoria Recology"},
    {
        "Label": "C",
        "Longitude": -123.90783452733942,
        "Latitude": 45.95217222679762,
        "Name": "Seaside Knife River Quarry",
    },
]

ROAD_SHAPEFILE = "maps/tl_2024_41007_roads/tl_2024_41007_roads.shp"

STANDARD_SEED_SETS: dict[str, list[int]] = {
    "smoke": [0, 1, 2],
    "standard": list(range(10)),
    "extended": list(range(20)),
}

STANDARD_BENCHMARK_SUITES: dict[str, list[dict[str, Any]]] = {
    "real_world_core": [
        {"name": "clatsop_winter_no_gis", "difficulty": "clatsop_winter_ops", "compare_gis": False},
        {"name": "clatsop_summer_no_gis", "difficulty": "clatsop_summer_ops", "compare_gis": False},
        {"name": "storm_stress_no_gis", "difficulty": "clatsop_storm_stress", "compare_gis": False},
    ],
    "real_world_gis_compare": [
        {"name": "clatsop_winter_compare_gis", "difficulty": "clatsop_winter_ops", "compare_gis": True},
        {"name": "clatsop_summer_compare_gis", "difficulty": "clatsop_summer_ops", "compare_gis": True},
    ],
}


def create_gis_config():
    """Create GIS configuration with pruned road network for faster simulation."""
    try:
        # Load and prune road network while maintaining connectivity
        # Uses major roads (I=Interstate, U=US Highway, S=State) but ensures all depots
        # and landfills remain connected to the network
        roads_gdf = load_and_prune_roads(ROAD_SHAPEFILE, depots=depots, landfills=landfills, enabled_types=["I", "U", "S"])
        road_graph = build_road_graph(roads_gdf)
        return GISConfig(roads_gdf=roads_gdf, road_graph=road_graph, depots=depots, landfills=landfills)
    except Exception as e:
        print(f"Warning: Could not load GIS data: {e}")
        return None


def create_scenario_config(difficulty="medium", gis_config=None):
    """Create scenario configuration based on difficulty.

    Difficulty levels:
    - easy/medium/hard: legacy non-seasonal profiles
    - clatsop_winter_ops: realistic winter coastal operations
    - clatsop_summer_ops: realistic summer wildfire/debris mix
    - clatsop_storm_stress: severe storm season with tighter budget
    - everything_bagel: full-feature stress profile
    """
    from datetime import datetime

    if difficulty == "easy":
        return ScenarioConfig(
            num_trucks=(20, 30),
            num_excavators=(12, 16),
            num_landslides=(1, 2),
            landslide_size_range=(100, 150),
            landslide_distance_range=(500, 1000),
            gis_config=gis_config,
        )
    elif difficulty == "medium":
        return ScenarioConfig(
            num_trucks=(15, 25),
            num_excavators=(8, 12),
            num_landslides=(8, 15),
            landslide_size_range=(200, 400),
            landslide_distance_range=(1000, 2000),
            gis_config=gis_config,
        )
    elif difficulty == "hard":
        return ScenarioConfig(
            num_trucks=(10, 15),
            num_excavators=(5, 8),
            num_landslides=(15, 25),
            landslide_size_range=(200, 400),
            landslide_distance_range=(1000, 2000),
            gis_config=gis_config,
        )
    elif difficulty == "clatsop_winter_ops":
        # Clatsop-oriented winter operations: snow/flood/slide mix, weather + dispatch priors
        return ScenarioConfig(
            num_trucks=(14, 22),
            num_excavators=(7, 11),
            num_snowplows=(3, 7),
            num_assessment_vehicles=(1, 2),
            num_landslides=(10, 18),
            landslide_size_range=(200, 1200),
            landslide_distance_range=(900, 2400),
            calendar_start_date=datetime(2024, 1, 1),
            calendar_duration_years=1,
            use_seasonal_disasters=True,
            use_weather_modifier=True,
            use_dispatch_delay_priors=True,
            seasonal_spawn_interval_minutes_range=(60.0, 720.0),
            track_costs=True,
            annual_budget=10_000_000,
            time_variance=0.1,
            gis_config=gis_config,
        )
    elif difficulty == "clatsop_summer_ops":
        # Clatsop-oriented summer operations: wildfire debris dominant with occasional flooding
        return ScenarioConfig(
            num_trucks=(12, 20),
            num_excavators=(6, 10),
            num_snowplows=(0, 1),
            num_assessment_vehicles=(1, 2),
            num_landslides=(9, 16),
            landslide_size_range=(150, 900),
            landslide_distance_range=(900, 2400),
            calendar_start_date=datetime(2024, 7, 1),
            calendar_duration_years=1,
            use_seasonal_disasters=True,
            use_weather_modifier=True,
            use_dispatch_delay_priors=True,
            seasonal_spawn_interval_minutes_range=(90.0, 1440.0),
            flood_assessment_minutes_range=(30.0, 90.0),
            flood_status_check_interval_minutes=45.0,
            track_costs=True,
            annual_budget=9_000_000,
            time_variance=0.1,
            gis_config=gis_config,
        )
    elif difficulty == "clatsop_storm_stress":
        # Severe storm season: tighter budget, heavier event load, larger variability
        return ScenarioConfig(
            num_trucks=(10, 16),
            num_excavators=(5, 9),
            num_snowplows=(2, 5),
            num_assessment_vehicles=(1, 2),
            num_landslides=(14, 24),
            landslide_size_range=(300, 2500),
            landslide_distance_range=(900, 2600),
            calendar_start_date=datetime(2024, 11, 1),
            calendar_duration_years=1,
            use_seasonal_disasters=True,
            use_weather_modifier=True,
            use_dispatch_delay_priors=True,
            seasonal_spawn_interval_minutes_range=(45.0, 540.0),
            track_costs=True,
            annual_budget=4_500_000,
            time_variance=0.15,
            gis_config=gis_config,
        )
    elif difficulty == "everything_bagel":
        # Full simulation: calendar, seasonal disasters, weather, budget tracking, more resources/disasters
        return ScenarioConfig(
            num_trucks=(25, 35),
            num_excavators=(15, 20),
            num_snowplows=(5, 10),
            num_assessment_vehicles=(2, 4),
            num_landslides=(20, 30),
            landslide_size_range=(300, 600),  # Larger disasters
            landslide_distance_range=(800, 2500),  # Wider geographic spread
            calendar_start_date=datetime(2024, 1, 1),
            calendar_duration_years=1,
            use_seasonal_disasters=True,
            use_weather_modifier=True,
            use_dispatch_delay_priors=True,
            track_costs=True,
            annual_budget=15_000_000,  # $15M budget
            time_variance=0.15,  # 15% stochastic variance
            gis_config=gis_config,
        )
    else:
        return ScenarioConfig(gis_config=gis_config)


def test_policy_on_seed(
    policy,
    seed: int,
    scenario_config: ScenarioConfig,
    live_plot: bool = False,
    check_invariants: bool = False,
) -> tuple[bool, float, float, list[str]]:
    """Test a single policy on a specific seed."""
    t0 = time.perf_counter()
    engine = SimPySimulationEngine(policy=policy, seed=seed, live_plot=live_plot, scenario_config=scenario_config)
    engine.initialize_world()
    success = engine.run()
    sim_duration = engine.get_summary()["non_idle_time"]
    wall_duration = time.perf_counter() - t0
    invariant_violations = validate_engine_invariants(engine, success) if check_invariants else []
    return success, sim_duration, wall_duration, invariant_violations


class ActionMaskedPPO:
    """
    Wrapper class that handles action masking for PPO model predictions.
    Copied from ppo.py to ensure consistency.
    """

    def __init__(self, model):
        self.model = model

    def predict(self, observation, deterministic=False):
        """
        Predict action with proper action masking.
        """
        # Extract valid_actions mask from observation
        valid_actions = observation["valid_actions"]

        # Get raw action predictions from model
        action, states = self.model.predict(observation, deterministic=deterministic)
        action = int(action)

        # Check if action is valid, if not choose first valid action
        if action < len(valid_actions) and valid_actions[action] == 1:
            return action, states
        else:
            # Action is invalid, choose first valid action
            valid_indices = np.where(valid_actions == 1)[0]
            if len(valid_indices) > 0:
                return valid_indices[0], states
            else:
                # No valid actions, return 0 as fallback
                return 0, states


def test_ppo_on_seed(
    model_path: str, seed: int, scenario_config: ScenarioConfig, max_visible_disasters: int = 5
) -> tuple[bool, float, float]:
    """Test PPO model on a specific seed with action masking."""
    t0 = time.perf_counter()
    env = DisasterResponseEnv(
        max_visible_disasters=max_visible_disasters,
        sorting_strategy="nearest",
        scenario_config=scenario_config,
    )

    model = PPO.load(model_path)
    # Wrap with action masking - this is crucial!
    masked_model = ActionMaskedPPO(model)

    obs, info = env.reset(seed=seed)
    episode_reward = 0
    step_count = 0

    while True:
        # Use masked model to ensure only valid actions are chosen
        action, _states = masked_model.predict(obs, deterministic=True)
        action = int(action)

        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        step_count += 1

        if terminated or truncated:
            success = terminated  # Terminated = success, truncated = failure
            # Use simulation time as duration metric
            sim_duration = info.get("sim_time", 0)
            wall_duration = time.perf_counter() - t0
            return success, sim_duration, wall_duration

        # Safety limit
        if step_count > 10000:
            return False, info.get("sim_time", 0), time.perf_counter() - t0


def resolve_seed_values(seeds: int, seed_set: str | None) -> list[int]:
    """Resolve benchmark seeds from a count or a named standard seed set."""
    if seed_set is None:
        return list(range(seeds))
    if seed_set not in STANDARD_SEED_SETS:
        raise ValueError(f"Unknown seed set '{seed_set}'. Available: {sorted(STANDARD_SEED_SETS)}")
    return list(STANDARD_SEED_SETS[seed_set])


def summarize_results(results: defaultdict[str, PolicyResult]) -> dict[str, dict[str, float | int | None]]:
    """Machine-readable summary for result serialization."""
    summaries: dict[str, dict[str, float | int | None]] = {}
    for name, data in results.items():
        success_times = data["success"]
        success_wall_times = data.get("success_wall", [])
        fail_count = data["fail"]
        total_runs = len(success_times) + fail_count
        success_rate = (len(success_times) / total_runs) * 100 if total_runs > 0 else 0.0
        summaries[name] = {
            "runs": total_runs,
            "successes": len(success_times),
            "failures": fail_count,
            "invariant_failures": data.get("invariant_fail", 0),
            "success_rate": success_rate,
            "avg_sim_time": statistics.mean(success_times) if success_times else None,
            "avg_wall_time_s": statistics.mean(success_wall_times) if success_wall_times else None,
            "sim_stdev": statistics.stdev(success_times) if len(success_times) > 1 else (0.0 if len(success_times) == 1 else None),
            "sim_min": min(success_times) if success_times else None,
            "sim_max": max(success_times) if success_times else None,
        }
    return summaries


def run_benchmark_single(
    gis_enabled: bool,
    seed_values: list[int],
    difficulty: str,
    ppo_model_path: str | None,
    live_plot: bool,
    tournament_only: bool = False,
    check_invariants: bool = False,
) -> defaultdict[str, PolicyResult]:
    """
    Run benchmark for a single configuration (GIS or no GIS).

    Args:
        gis_enabled: Whether to use GIS
        seed_values: Explicit list of seeds to test
        difficulty: Difficulty level
        ppo_model_path: Path to PPO model
        live_plot: Show live plots
        tournament_only: If True, run ONLY the tournament policy

    Returns:
        Dictionary of results
    """
    config_name = "GIS" if gis_enabled else "No GIS"
    print(f"\n{'='*85}")
    print(f"BENCHMARK: Testing with {config_name}")
    print(f"Seeds: {seed_values}, Difficulty: {difficulty}")
    if tournament_only:
        print("Mode: TOURNAMENT ONLY")
    print(f"{'='*85}\n")

    # Create scenario config
    if gis_enabled:
        gis_config = create_gis_config()
        if gis_config is None:
            print(f"ERROR: Could not load GIS config. Skipping {config_name} tests.")
            return defaultdict(lambda: {"success": [], "success_wall": [], "fail": 0, "invariant_fail": 0})
    else:
        gis_config = None

    scenario_config = create_scenario_config(difficulty, gis_config)
    print(
        "Scenario config summary: "
        f"seasonal={scenario_config.use_seasonal_disasters}, "
        f"weather={scenario_config.use_weather_modifier}, "
        f"dispatch_priors={scenario_config.use_dispatch_delay_priors}, "
        f"budget=${scenario_config.annual_budget:,.0f}, "
        f"time_variance={scenario_config.time_variance}, "
        f"calendar_start={scenario_config.calendar_start_date}, "
        f"spawn_interval_min={scenario_config.seasonal_spawn_interval_minutes_range}"
    )

    # Store results
    aggregated_results: defaultdict[str, PolicyResult] = defaultdict(
        lambda: {"success": [], "success_wall": [], "fail": 0, "invariant_fail": 0}
    )
    tournament_policy = [p for p in POLICIES if p.name == "tournament"]

    # Test policies on each seed
    for seed in seed_values:
        print(f"\n--- SEED {seed} ---")

        if tournament_only:
            # Run ONLY tournament policy
            if tournament_policy:
                policy = tournament_policy[0]
                success, sim_duration, wall_duration, invariant_violations = test_policy_on_seed(
                    policy,
                    seed,
                    scenario_config,
                    live_plot,
                    check_invariants=check_invariants,
                )

                if success:
                    aggregated_results[policy.name]["success"].append(sim_duration)
                    aggregated_results[policy.name]["success_wall"].append(wall_duration)
                else:
                    aggregated_results[policy.name]["fail"] += 1
                if invariant_violations:
                    aggregated_results[policy.name]["invariant_fail"] += 1

                print(
                    f"  {policy.name:<20} | {'SUCCESS' if success else 'FAIL':<7} | "
                    f"Sim: {sim_duration:.2f} | Wall: {wall_duration:.2f}s"
                    + (f" | INV_FAIL({len(invariant_violations)})" if invariant_violations else "")
                )
                for violation in invariant_violations:
                    print(f"    - invariant: {violation}")
        else:
            # Test each policy
            for policy in POLICIES:
                if policy.name.startswith("tournament"):
                    continue

                success, sim_duration, wall_duration, invariant_violations = test_policy_on_seed(
                    policy,
                    seed,
                    scenario_config,
                    live_plot,
                    check_invariants=check_invariants,
                )

                if success:
                    aggregated_results[policy.name]["success"].append(sim_duration)
                    aggregated_results[policy.name]["success_wall"].append(wall_duration)
                else:
                    aggregated_results[policy.name]["fail"] += 1
                if invariant_violations:
                    aggregated_results[policy.name]["invariant_fail"] += 1

                print(
                    f"  {policy.name:<20} | {'SUCCESS' if success else 'FAIL':<7} | "
                    f"Sim: {sim_duration:.2f} | Wall: {wall_duration:.2f}s"
                    + (f" | INV_FAIL({len(invariant_violations)})" if invariant_violations else "")
                )
                for violation in invariant_violations:
                    print(f"    - invariant: {violation}")

        # Test PPO if model provided
        if ppo_model_path and not tournament_only:
            try:
                success, sim_duration, wall_duration = test_ppo_on_seed(ppo_model_path, seed, scenario_config)

                if success:
                    aggregated_results["PPO"]["success"].append(sim_duration)
                    aggregated_results["PPO"]["success_wall"].append(wall_duration)
                else:
                    aggregated_results["PPO"]["fail"] += 1

                print(
                    f"  {'PPO':<20} | {'SUCCESS' if success else 'FAIL':<7} | "
                    f"Sim: {sim_duration:.2f} | Wall: {wall_duration:.2f}s"
                )
            except Exception as e:
                print(f"  {'PPO':<20} | ERROR: {e}")
                aggregated_results["PPO"]["fail"] += 1

    return aggregated_results


def print_results_table(results: defaultdict[str, PolicyResult], title: str = "RESULTS"):
    """Print formatted results table."""
    print(f"\n{title}")
    print("=" * 129)
    print(
        f"{'APPROACH':<20} | {'SUCCESS %':<10} | {'AVG SIM':<10} | {'AVG WALL':<10} | "
        f"{'SIM STDEV':<10} | {'SIM MIN':<8} | {'SIM MAX':<8} | {'INV FAIL':<8}"
    )
    print("-" * 129)

    # Calculate statistics
    final_stats: list[tuple[str, float, float, float, float, float, float, int]] = []
    for name, data in results.items():
        success_times = data["success"]
        success_wall_times = data.get("success_wall", [])
        fail_count = data["fail"]
        invariant_fail = data.get("invariant_fail", 0)
        total_runs = len(success_times) + fail_count

        success_rate = (len(success_times) / total_runs) * 100 if total_runs > 0 else 0

        if success_times:
            avg = statistics.mean(success_times)
            avg_wall = statistics.mean(success_wall_times) if success_wall_times else float("inf")
            stdev = statistics.stdev(success_times) if len(success_times) > 1 else 0.0
            mn = min(success_times)
            mx = max(success_times)
        else:
            avg = float("inf")
            avg_wall = float("inf")
            stdev = 0.0
            mn = 0
            mx = 0

        final_stats.append((name, success_rate, avg, avg_wall, stdev, mn, mx, invariant_fail))

    # Sort by Success Rate (Desc), then Avg Time (Asc)
    final_stats.sort(key=lambda x: (-x[1], x[2]))

    for name, rate, avg, avg_wall, stdev, mn, mx, invariant_fail in final_stats:
        avg_str = f"{avg:.2f}" if avg != float("inf") else "N/A"
        avg_wall_str = f"{avg_wall:.2f}s" if avg_wall != float("inf") else "N/A"
        stdev_str = f"{stdev:.2f}" if avg != float("inf") else "N/A"
        mn_str = f"{mn:.0f}" if avg != float("inf") else "N/A"
        mx_str = f"{mx:.0f}" if avg != float("inf") else "N/A"

        print(
            f"{name:<20} | {rate:<9.1f}% | {avg_str:<10} | {avg_wall_str:<10} | "
            f"{stdev_str:<10} | {mn_str:<8} | {mx_str:<8} | {invariant_fail:<8}"
        )

    print("=" * 129 + "\n")


def print_comparison_table(results_no_gis: defaultdict[str, PolicyResult], results_gis: defaultdict[str, PolicyResult]):
    """Print side-by-side comparison of GIS vs No GIS."""
    print("\n" + "=" * 120)
    print("COMPARISON: GIS vs No GIS")
    print("=" * 120)
    print(f"{'APPROACH':<20} | {'NO GIS':<35} | {'GIS':<35} | {'DIFFERENCE':<25}")
    print(f"{'':20} | {'Success % | Avg Time':<35} | {'Success % | Avg Time':<35} | {'Success | Time':<25}")
    print("-" * 120)

    # Get all unique approach names
    all_names = set(results_no_gis.keys()) | set(results_gis.keys())

    for name in sorted(all_names):
        # No GIS stats
        no_gis_data = results_no_gis.get(name, {"success": [], "success_wall": [], "fail": 0})
        no_gis_success = no_gis_data["success"]
        no_gis_fail = no_gis_data["fail"]
        no_gis_total = len(no_gis_success) + no_gis_fail
        no_gis_rate = (len(no_gis_success) / no_gis_total * 100) if no_gis_total > 0 else 0
        no_gis_avg = statistics.mean(no_gis_success) if no_gis_success else float("inf")
        no_gis_str = f"{no_gis_rate:>6.1f}% | {no_gis_avg:>8.1f}" if no_gis_success else f"{no_gis_rate:>6.1f}% | {'N/A':>8}"

        # GIS stats
        gis_data = results_gis.get(name, {"success": [], "success_wall": [], "fail": 0})
        gis_success = gis_data["success"]
        gis_fail = gis_data["fail"]
        gis_total = len(gis_success) + gis_fail
        gis_rate = (len(gis_success) / gis_total * 100) if gis_total > 0 else 0
        gis_avg = statistics.mean(gis_success) if gis_success else float("inf")
        gis_str = f"{gis_rate:>6.1f}% | {gis_avg:>8.1f}" if gis_success else f"{gis_rate:>6.1f}% | {'N/A':>8}"

        # Differences
        rate_diff = gis_rate - no_gis_rate
        if no_gis_success and gis_success:
            time_diff = gis_avg - no_gis_avg
            time_diff_pct = (time_diff / no_gis_avg * 100) if no_gis_avg != 0 else 0
            diff_str = f"{rate_diff:>+6.1f}% | {time_diff:>+7.1f} ({time_diff_pct:>+.1f}%)"
        else:
            diff_str = f"{rate_diff:>+6.1f}% | {'N/A':>15}"

        print(f"{name:<20} | {no_gis_str:<35} | {gis_str:<35} | {diff_str:<25}")

    print("=" * 120 + "\n")


def run_benchmark(
    seeds: int = 5,
    seed_set: str | None = None,
    difficulty: str = "clatsop_winter_ops",
    ppo_model_path: str | None = None,
    live_plot: bool = False,
    compare_gis: bool = False,
    tournament_only: bool = False,
    check_invariants: bool = False,
):
    """
    Run benchmark comparing all policies and PPO on shared seeds.

    Args:
        seeds: Number of different seeds to test
        difficulty: Scenario profile name
        ppo_model_path: Path to trained PPO model (if None, skips PPO testing)
        live_plot: Whether to show live plots for policies
        compare_gis: Whether to test both GIS and non-GIS configurations
        tournament_only: If True, run ONLY the tournament policy (skip all other policies)
    """
    seed_values = resolve_seed_values(seeds, seed_set)

    print(f"\n{'#'*85}")
    print(f"#{'':83}#")
    if tournament_only:
        print(f"#{'TOURNAMENT POLICY BENCHMARK':^83}#")
    else:
        print(f"#{'COMPREHENSIVE BENCHMARK':^83}#")
    print(f"#{'':83}#")
    print(f"{'#'*85}\n")

    # Always run without GIS
    results_no_gis = run_benchmark_single(
        gis_enabled=False,
        seed_values=seed_values,
        difficulty=difficulty,
        ppo_model_path=ppo_model_path,
        live_plot=live_plot,
        tournament_only=tournament_only,
        check_invariants=check_invariants,
    )

    print_results_table(results_no_gis, "RESULTS: NO GIS")

    # Optionally run with GIS
    results_gis = None
    if compare_gis:
        results_gis = run_benchmark_single(
            gis_enabled=True,
            seed_values=seed_values,
            difficulty=difficulty,
            ppo_model_path=ppo_model_path,
            live_plot=live_plot,
            tournament_only=tournament_only,
            check_invariants=check_invariants,
        )

        print_results_table(results_gis, "RESULTS: GIS")

        # Print comparison
        print_comparison_table(results_no_gis, results_gis)

    return results_no_gis, results_gis


def run_standard_suite(
    suite_name: str,
    seed_set: str = "standard",
    ppo_model_path: str | None = None,
    tournament_only: bool = False,
    check_invariants: bool = False,
) -> dict[str, Any]:
    """Run a named benchmark suite with standardized scenarios/seeds."""
    if suite_name not in STANDARD_BENCHMARK_SUITES:
        raise ValueError(f"Unknown suite '{suite_name}'. Available: {sorted(STANDARD_BENCHMARK_SUITES)}")

    seed_values = resolve_seed_values(seeds=0, seed_set=seed_set)
    suite_runs: list[dict[str, Any]] = []

    for spec in STANDARD_BENCHMARK_SUITES[suite_name]:
        print(f"\nRunning suite scenario: {spec['name']}")
        results_no_gis, results_gis = run_benchmark(
            seeds=len(seed_values),
            seed_set=seed_set,
            difficulty=spec["difficulty"],
            ppo_model_path=ppo_model_path,
            live_plot=False,
            compare_gis=bool(spec.get("compare_gis", False)),
            tournament_only=tournament_only,
            check_invariants=check_invariants,
        )
        suite_runs.append(
            {
                "name": spec["name"],
                "difficulty": spec["difficulty"],
                "compare_gis": bool(spec.get("compare_gis", False)),
                "seed_set": seed_set,
                "seed_values": seed_values,
                "results_no_gis": summarize_results(results_no_gis),
                "results_gis": summarize_results(results_gis) if results_gis is not None else None,
            }
        )

    return {
        "suite": suite_name,
        "seed_set": seed_set,
        "seed_values": seed_values,
        "runs": suite_runs,
    }


def write_results_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"Wrote benchmark results to {path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Benchmark policies and PPO on shared seeds")
    parser.add_argument("--seeds", type=int, default=5, help="Number of different seeds to test")
    parser.add_argument(
        "--seed-set",
        type=str,
        default=None,
        choices=list(STANDARD_SEED_SETS.keys()),
        help="Use a standardized seed set instead of --seeds (smoke/standard/extended)",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="clatsop_winter_ops",
        choices=[
            "easy",
            "medium",
            "hard",
            "clatsop_winter_ops",
            "clatsop_summer_ops",
            "clatsop_storm_stress",
            "everything_bagel",
        ],
        help=(
            "Scenario profile: legacy easy/medium/hard, realistic clatsop_winter_ops/clatsop_summer_ops/"
            "clatsop_storm_stress, or everything_bagel"
        ),
    )
    parser.add_argument("--ppo-model", type=str, default=None, help="Path to trained PPO model (e.g., model.zip)")
    parser.add_argument("--live", action="store_true", help="Show live plot for policy simulations")
    parser.add_argument("--compare-gis", action="store_true", help="Test both GIS and non-GIS configurations for comparison")
    parser.add_argument(
        "--everything-bagel",
        action="store_true",
        help="Run full simulation with all features enabled: calendar, seasonal disasters, weather, budget, more disasters",
    )
    parser.add_argument("--tournament-only", action="store_true", help="Run ONLY the tournament policy (skip all other policies)")
    parser.add_argument(
        "--check-invariants",
        action="store_true",
        help="Validate simulation invariants on each policy run and report violations",
    )
    parser.add_argument("--tournament-depth", type=int, default=1, help="Tree search depth for tournament policy (1=run to completion, 2+=look ahead N decisions)")
    parser.add_argument(
        "--suite",
        type=str,
        default=None,
        choices=list(STANDARD_BENCHMARK_SUITES.keys()),
        help="Run a standardized benchmark suite (overrides --difficulty/--compare-gis single-run mode)",
    )
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to write machine-readable benchmark results")

    args = parser.parse_args()

    # If --everything-bagel is set, override difficulty
    if args.everything_bagel:
        args.difficulty = "everything_bagel"

    # Set tournament depth before running
    if args.tournament_depth > 1:
        print(f"Using tree search with depth {args.tournament_depth}")
    set_tournament_depth(args.tournament_depth)

    if args.suite is not None:
        payload = run_standard_suite(
            suite_name=args.suite,
            seed_set=args.seed_set or "standard",
            ppo_model_path=args.ppo_model,
            tournament_only=args.tournament_only,
            check_invariants=args.check_invariants,
        )
        if args.output_json:
            write_results_json(args.output_json, payload)
    else:
        results_no_gis, results_gis = run_benchmark(
            seeds=args.seeds,
            seed_set=args.seed_set,
            difficulty=args.difficulty,
            ppo_model_path=args.ppo_model,
            live_plot=args.live,
            compare_gis=args.compare_gis,
            tournament_only=args.tournament_only,
            check_invariants=args.check_invariants,
        )
        if args.output_json:
            write_results_json(
                args.output_json,
                {
                    "mode": "single",
                    "difficulty": args.difficulty,
                    "seed_set": args.seed_set,
                    "seed_values": resolve_seed_values(args.seeds, args.seed_set),
                    "compare_gis": args.compare_gis,
                    "tournament_only": args.tournament_only,
                    "results_no_gis": summarize_results(results_no_gis),
                    "results_gis": summarize_results(results_gis) if results_gis is not None else None,
                },
            )
