"""
Real-world-oriented example usage for the disaster response simulation.

This script demonstrates calibrated scenario presets and how to configure
key model options (time units, seasonal/weather behavior, dispatch delays,
cost tracking, and GIS routing).
"""

from __future__ import annotations

import os
from dataclasses import fields, is_dataclass
from datetime import datetime
from typing import Any

from SimPyTest.engine import ScenarioConfig, SimPySimulationEngine
from SimPyTest.gis_utils import Depot, GISConfig, Landfill, build_road_graph, load_and_prune_roads
from SimPyTest.gym import DisasterResponseEnv
from SimPyTest.policies import POLICIES


# ---------------------------------------------------------------------------
# Regional setup (Clatsop-focused placeholder depots/landfills)
# ---------------------------------------------------------------------------

depots: list[Depot] = [
    {
        "Longitude": -123.92616052570274,
        "Latitude": 46.16262226957932,
        "Name": "ODOT Warrenton",
        "numExc": 5,
        "numTrucks": 50,
        "color": "green",
    },
    {
        "Longitude": -123.92250095724509,
        "Latitude": 45.984013191819535,
        "Name": "ODOT Seaside",
        "numExc": 3,
        "numTrucks": 40,
        "color": "yellow",
    },
    {
        "Longitude": -123.81788435312336,
        "Latitude": 45.91011185257814,
        "Name": "ODOT Necanium",
        "numExc": 4,
        "numTrucks": 30,
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


def get_policy(name: str):
    return next(p for p in POLICIES if p.name == name)


def maybe_build_gis_config() -> GISConfig | None:
    if not os.path.exists(ROAD_SHAPEFILE):
        return None
    roads_gdf = load_and_prune_roads(ROAD_SHAPEFILE, depots=depots, landfills=landfills, enabled_types=["I", "U", "S"])
    road_graph = build_road_graph(roads_gdf)
    return GISConfig(roads_gdf=roads_gdf, road_graph=road_graph, depots=depots, landfills=landfills)


def config_to_dict(cfg: ScenarioConfig) -> dict[str, Any]:
    if not is_dataclass(cfg):
        return {"repr": repr(cfg)}
    return {f.name: getattr(cfg, f.name) for f in fields(cfg)}


def print_config(cfg: ScenarioConfig, title: str = "ScenarioConfig") -> None:
    print(f"\n{title}:")
    cfg_data = config_to_dict(cfg)

    focus_keys = [
        "num_trucks",
        "num_excavators",
        "num_snowplows",
        "num_assessment_vehicles",
        "num_landslides",
        "landslide_size_range",
        "calendar_start_date",
        "calendar_duration_years",
        "use_seasonal_disasters",
        "use_weather_modifier",
        "use_dispatch_delay_priors",
        "calendar_minutes_per_sim_minute",
        "non_seasonal_spawn_interval_minutes_range",
        "seasonal_spawn_interval_minutes_range",
        "track_costs",
        "annual_budget",
        "time_variance",
    ]
    for key in focus_keys:
        print(f"  {key}: {cfg_data.get(key)}")


def run_engine_example(name: str, policy_name: str, cfg: ScenarioConfig, seed: int = 42) -> None:
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    print_config(cfg)

    engine = SimPySimulationEngine(policy=get_policy(policy_name), seed=seed, scenario_config=cfg)
    engine.initialize_world()
    success = engine.run()
    summary = engine.get_summary()

    print(f"\nPolicy: {policy_name}")
    print(f"Simulation {'succeeded' if success else 'failed'}")
    print(f"Non-idle minutes: {summary['non_idle_time']:.2f}")
    print(f"Disasters created/resolved: {summary['disasters_created']}/{summary['disasters_resolved']}")
    print(f"Avg response minutes: {summary['avg_response_time']:.2f}")
    print(f"Dispatch delay avg/max minutes: {summary['avg_dispatch_delay']:.2f}/{summary['max_dispatch_delay']:.2f}")
    print(f"Budget spent/utilization: ${summary['total_spent']:,.0f} / {summary['budget_utilization']:.1%}")


# ---------------------------------------------------------------------------
# Realistic scenario examples
# ---------------------------------------------------------------------------

def example_1_winter_operations_no_gis() -> None:
    cfg = ScenarioConfig(
        num_trucks=(14, 20),
        num_excavators=(7, 10),
        num_snowplows=(3, 6),
        num_assessment_vehicles=(1, 2),
        num_landslides=(10, 16),
        landslide_size_range=(200, 900),
        calendar_start_date=datetime(2024, 1, 1),
        calendar_duration_years=1,
        use_seasonal_disasters=True,
        use_weather_modifier=True,
        use_dispatch_delay_priors=True,
        calendar_minutes_per_sim_minute=1.0,
        seasonal_spawn_interval_minutes_range=(60.0, 720.0),
        snow_auto_resolve_multiplier=3.0,
        track_costs=True,
        annual_budget=9_000_000,
        time_variance=0.1,
    )
    run_engine_example("Example 1: Winter Clatsop Operations (No GIS)", "seasonal_priority", cfg)


def example_2_summer_wildfire_and_flood_no_gis() -> None:
    cfg = ScenarioConfig(
        num_trucks=(12, 18),
        num_excavators=(6, 10),
        num_snowplows=(0, 1),
        num_assessment_vehicles=(1, 2),
        num_landslides=(9, 15),
        landslide_size_range=(150, 700),
        calendar_start_date=datetime(2024, 7, 1),
        calendar_duration_years=1,
        use_seasonal_disasters=True,
        use_weather_modifier=True,
        use_dispatch_delay_priors=True,
        seasonal_spawn_interval_minutes_range=(90.0, 1440.0),
        flood_assessment_minutes_range=(30.0, 90.0),
        flood_status_check_interval_minutes=45.0,
        track_costs=True,
        annual_budget=8_000_000,
        time_variance=0.1,
    )
    run_engine_example("Example 2: Summer Wildfire/Flood Mix (No GIS)", "weather_aware", cfg)


def example_3_realistic_gis_winter() -> None:
    gis_config = maybe_build_gis_config()
    if gis_config is None:
        print("\nSkipping Example 3 (GIS): shapefile not found.")
        return

    cfg = ScenarioConfig(
        num_trucks=(16, 24),
        num_excavators=(8, 12),
        num_snowplows=(3, 6),
        num_assessment_vehicles=(1, 2),
        num_landslides=(10, 18),
        landslide_size_range=(250, 1200),
        calendar_start_date=datetime(2024, 1, 1),
        calendar_duration_years=1,
        use_seasonal_disasters=True,
        use_weather_modifier=True,
        use_dispatch_delay_priors=True,
        seasonal_spawn_interval_minutes_range=(60.0, 720.0),
        track_costs=True,
        annual_budget=12_000_000,
        gis_config=gis_config,
    )
    run_engine_example("Example 3: Realistic Winter Scenario with GIS Routing", "resource_efficiency", cfg)


def example_4_budget_stress_test() -> None:
    cfg = ScenarioConfig(
        num_trucks=(10, 14),
        num_excavators=(5, 8),
        num_snowplows=(2, 4),
        num_assessment_vehicles=(1, 1),
        num_landslides=(14, 20),
        landslide_size_range=(300, 1800),
        calendar_start_date=datetime(2024, 11, 1),
        calendar_duration_years=1,
        use_seasonal_disasters=True,
        use_weather_modifier=True,
        use_dispatch_delay_priors=True,
        seasonal_spawn_interval_minutes_range=(45.0, 540.0),
        track_costs=True,
        annual_budget=3_500_000,
        time_variance=0.15,
    )
    run_engine_example("Example 4: Budget-Constrained Storm Season Stress Test", "budget_aware", cfg)


def example_5_gym_realistic_configuration() -> None:
    print("\n" + "=" * 80)
    print("Example 5: Gym Environment with Realistic Configuration")
    print("=" * 80)

    cfg = ScenarioConfig(
        num_trucks=(12, 18),
        num_excavators=(6, 10),
        num_snowplows=(2, 4),
        num_assessment_vehicles=(1, 2),
        num_landslides=(8, 12),
        landslide_size_range=(200, 1000),
        calendar_start_date=datetime(2024, 1, 1),
        calendar_duration_years=1,
        use_seasonal_disasters=True,
        use_weather_modifier=True,
        use_dispatch_delay_priors=True,
        track_costs=True,
        annual_budget=7_500_000,
    )
    print_config(cfg)

    env = DisasterResponseEnv(max_visible_disasters=5, sorting_strategy="nearest", scenario_config=cfg)
    obs, info = env.reset(seed=42)

    print("\nObservation keys:", list(obs.keys()))
    print("Info keys:", list(info.keys()))

    for i in range(8):
        valid_actions = [j for j, v in enumerate(obs["valid_actions"]) if v == 1]
        if not valid_actions:
            print("No valid actions available.")
            break
        action = valid_actions[0]
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"Step {i + 1}: action={action} reward={reward:.2f} "
            f"active={info['active_disasters']} sim_time={info['sim_time']:.1f} "
            f"budget_util={info['budget_utilization']:.1%}"
        )
        if terminated or truncated:
            break


def example_6_configuration_surface_demo() -> None:
    """Show how to override advanced priors/custom options in a single config."""
    cfg = ScenarioConfig(
        num_trucks=(14, 20),
        num_excavators=(7, 10),
        num_snowplows=(2, 5),
        num_assessment_vehicles=(1, 2),
        num_landslides=(10, 14),
        calendar_start_date=datetime(2024, 1, 1),
        calendar_duration_years=1,
        use_seasonal_disasters=True,
        use_weather_modifier=True,
        use_dispatch_delay_priors=True,
        calendar_minutes_per_sim_minute=1.0,
        non_gis_distance_unit_miles=0.01,
        non_seasonal_spawn_interval_minutes_range=(10.0, 80.0),
        seasonal_spawn_interval_minutes_range=(60.0, 720.0),
        snow_auto_resolve_multiplier=2.5,
        snow_work_interval_minutes=30.0,
        flood_assessment_minutes_range=(25.0, 80.0),
        flood_status_check_interval_minutes=30.0,
        storm_dispatch_delay_multiplier=2.2,
        track_costs=True,
        annual_budget=10_000_000,
    )

    # Example custom prior tuning (Tier C assumptions; replace when measured data arrives)
    cfg.disaster_operational_priors["landslide"]["dispatch_delay_minutes_range"] = (15.0, 110.0)
    cfg.disaster_operational_priors["flood"]["assessment_minutes_range"] = (35.0, 100.0)

    print("\n" + "=" * 80)
    print("Example 6: Full Configuration Surface (Advanced Overrides)")
    print("=" * 80)
    print_config(cfg, "Advanced ScenarioConfig")
    print("\nCustom prior overrides:")
    print("  landslide.dispatch_delay_minutes_range:", cfg.disaster_operational_priors["landslide"]["dispatch_delay_minutes_range"])
    print("  flood.assessment_minutes_range:", cfg.disaster_operational_priors["flood"]["assessment_minutes_range"])


def run_all_examples() -> None:
    print("\n" + "=" * 80)
    print("Real-World-Calibrated Simulation Examples")
    print("=" * 80)

    example_1_winter_operations_no_gis()
    example_2_summer_wildfire_and_flood_no_gis()
    example_3_realistic_gis_winter()
    example_4_budget_stress_test()
    example_5_gym_realistic_configuration()
    example_6_configuration_surface_demo()

    print("\n" + "=" * 80)
    print("All examples completed")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_all_examples()
