"""
Benchmark file that tests all policies and PPO model on shared seeds.
Tests both with and without GIS for comparison.
"""

from __future__ import annotations
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict
from collections import defaultdict
from argparse import ArgumentParser

from SimPyTest import SimPySimulationEngine
from SimPyTest.clatsop_spatial import CLATSOP_LOCAL_COORD_MAX
from SimPyTest.gym import DisasterResponseGym
from SimPyTest.scenario_types import ScenarioConfig
from SimPyTest.visualization import EngineVisualizer

# from SimPyTest.benchmark_runner import run_policy_episode, run_ppo_episode
from SimPyTest.evaluation import EVALUATION_PROTOCOL_VERSION, KPIBundle, build_simulation_summary, compute_kpi_bundle, compute_objective_score
from SimPyTest.policies import POLICIES, TOURNAMENT_POLICY, Policy, set_tournament_depth
from scripts.training.ppo.ppo_dispatch import EpisodeResult as PPOEpisodeResult
from scripts.training.ppo.ppo_dispatch import load_model, run_policy_episode as run_ppo_policy_episode, select_device
from SimPyTest.gis_utils import Depot, GISConfig, Landfill, build_road_graph, load_and_prune_roads
from SimPyTest.real_world_params import meters_to_miles
from SimPyTest.scenario_types import (
    DistanceModelConfig,
    OperationalPriorsConfig,
    ResourceCounts,
    SeasonalDisasterConfig,
    SeasonalSpawnConfig,
    WeatherModelConfig,
)

if TYPE_CHECKING:
    from scripts.training.mlp.ml_dispatch import TrainedDispatchPolicy


class PolicyResult(TypedDict):
    success: list[float]
    success_wall: list[float]
    objective: list[float]
    kpis: list[KPIBundle]
    fail: int


class PPOModelSpec(TypedDict):
    label: str
    path: str
    metadata: dict[str, Any]


class DispatchModelSpec(TypedDict):
    label: str
    path: str
    metadata: dict[str, Any]


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


class Suite(TypedDict):
    preset: str
    compare_gis: bool


STANDARD_BENCHMARK_SUITES: dict[str, Suite] = {
    "clatsop_winter_no_gis": {"preset": "clatsop_winter_ops", "compare_gis": False},
    "clatsop_summer_no_gis": {"preset": "clatsop_summer_ops", "compare_gis": False},
    "storm_stress_no_gis": {"preset": "clatsop_storm_stress", "compare_gis": False},
    "landslide_curriculum_no_gis": {"preset": "clatsop_landslide_curriculum", "compare_gis": False},
    "landslide_ops_no_gis": {"preset": "clatsop_landslide_ops", "compare_gis": False},
    "single_landslide_micro_no_gis": {"preset": "single_landslide_micro", "compare_gis": False},
    "clatsop_winter_compare_gis": {"preset": "clatsop_winter_ops", "compare_gis": True},
    "clatsop_summer_compare_gis": {"preset": "clatsop_summer_ops", "compare_gis": True},
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


def create_scenario_config(preset: str = "clatsop_winter_ops", gis_config: GISConfig | None = None) -> ScenarioConfig:
    """Create scenario configuration based on preset.

    Supported preset levels:
    - clatsop_winter_ops: winter coastal operations
    - clatsop_summer_ops: summer wildfire/debris mix
    - clatsop_storm_stress: severe storm season with heavier event pressure
    - clatsop_landslide_curriculum: easy landslide-only curriculum for PPO debugging
    - clatsop_landslide_curriculum_stage2: slightly harder landslide-only curriculum stage
    - clatsop_landslide_ops: landslide-only operations with trucks and excavators
    - single_landslide_micro: one tiny deterministic landslide with fixed resources
    """
    from datetime import datetime

    # Default seasonal profiles now live with benchmark profile constructors.
    base_profiles = {
        "landslide": SeasonalDisasterConfig(
            event_count_range_by_season={"winter": (3, 8), "spring": (2, 6), "summer": (0, 2), "fall": (2, 6)},
            size_range_by_season={"winter": (200, 5000), "spring": (100, 3500), "summer": (50, 1200), "fall": (100, 4000)},
        ),
        "snow": SeasonalDisasterConfig(
            event_count_range_by_season={"winter": (2, 8), "spring": (0, 2), "summer": (0, 0), "fall": (0, 1)},
            size_range_by_season={"winter": (2, 24), "spring": (1, 8), "summer": (1, 2), "fall": (1, 6)},
        ),
        "wildfire_debris": SeasonalDisasterConfig(
            event_count_range_by_season={"winter": (0, 0), "spring": (0, 2), "summer": (2, 7), "fall": (1, 5)},
            size_range_by_season={"winter": (20, 100), "spring": (50, 300), "summer": (100, 1500), "fall": (80, 1200)},
        ),
        "flood": SeasonalDisasterConfig(
            event_count_range_by_season={"winter": (2, 6), "spring": (1, 4), "summer": (0, 1), "fall": (1, 3)},
            size_range_by_season={"winter": (24, 240), "spring": (24, 168), "summer": (12, 72), "fall": (24, 168)},
        ),
    }

    if preset == "clatsop_winter_ops":
        return ScenarioConfig(
            resource_counts=ResourceCounts(trucks=(14, 22), excavators=(7, 11), snowplows=(3, 7), assessment_vehicles=(1, 2)),
            seasonal_spawn=SeasonalSpawnConfig(
                target_events_range=(10, 18),
                interarrival_minutes_range=(60.0, 720.0),
                disasters=base_profiles,
            ),
            weather_model=WeatherModelConfig(
                enable_spawn_scaling=True,
                enable_dispatch_scaling=True,
                use_vulnerability_weighting=False,
                storm_dispatch_delay_multiplier=2.0,
            ),
            operational_priors=OperationalPriorsConfig(
                time_variance=0.1,
                use_dispatch_delay_priors=True,
            ),
            distance_model=DistanceModelConfig(
                non_gis_distance_unit_miles=meters_to_miles(1.0),
                spawn_distance_range=(0, CLATSOP_LOCAL_COORD_MAX),
            ),
            calendar_start_date=datetime(2024, 1, 1),
            calendar_duration_years=1,
            gis_config=gis_config,
        )
    if preset == "clatsop_summer_ops":
        summer_profiles = dict(base_profiles)
        summer_profiles["wildfire_debris"] = SeasonalDisasterConfig(
            event_count_range_by_season={"winter": (0, 0), "spring": (1, 3), "summer": (4, 10), "fall": (2, 6)},
            size_range_by_season={"winter": (20, 120), "spring": (80, 400), "summer": (120, 1500), "fall": (100, 1200)},
        )
        return ScenarioConfig(
            resource_counts=ResourceCounts(trucks=(12, 20), excavators=(6, 10), snowplows=(0, 1), assessment_vehicles=(1, 2)),
            seasonal_spawn=SeasonalSpawnConfig(
                target_events_range=(9, 16),
                interarrival_minutes_range=(90.0, 1440.0),
                disasters=summer_profiles,
            ),
            weather_model=WeatherModelConfig(
                enable_spawn_scaling=True,
                enable_dispatch_scaling=True,
                use_vulnerability_weighting=False,
                storm_dispatch_delay_multiplier=1.6,
            ),
            operational_priors=OperationalPriorsConfig(
                time_variance=0.1,
                use_dispatch_delay_priors=True,
                flood_assessment_minutes_range=(30.0, 90.0),
                flood_status_check_interval_minutes=45.0,
            ),
            distance_model=DistanceModelConfig(
                non_gis_distance_unit_miles=meters_to_miles(1.0),
                spawn_distance_range=(0, CLATSOP_LOCAL_COORD_MAX),
            ),
            calendar_start_date=datetime(2024, 7, 1),
            calendar_duration_years=1,
            gis_config=gis_config,
        )
    if preset == "clatsop_storm_stress":
        stress_profiles = dict(base_profiles)
        stress_profiles["landslide"] = SeasonalDisasterConfig(
            event_count_range_by_season={"winter": (6, 12), "spring": (4, 8), "summer": (1, 3), "fall": (4, 10)},
            size_range_by_season={"winter": (700, 7000), "spring": (400, 5000), "summer": (200, 2500), "fall": (500, 6000)},
        )
        return ScenarioConfig(
            resource_counts=ResourceCounts(trucks=(10, 16), excavators=(5, 9), snowplows=(2, 5), assessment_vehicles=(1, 2)),
            seasonal_spawn=SeasonalSpawnConfig(
                target_events_range=(14, 24),
                interarrival_minutes_range=(45.0, 540.0),
                disasters=stress_profiles,
            ),
            weather_model=WeatherModelConfig(
                enable_spawn_scaling=True,
                enable_dispatch_scaling=True,
                use_vulnerability_weighting=False,
                storm_dispatch_delay_multiplier=2.3,
            ),
            operational_priors=OperationalPriorsConfig(
                time_variance=0.15,
                use_dispatch_delay_priors=True,
            ),
            distance_model=DistanceModelConfig(
                non_gis_distance_unit_miles=meters_to_miles(1.0),
                spawn_distance_range=(0, CLATSOP_LOCAL_COORD_MAX),
            ),
            calendar_start_date=datetime(2024, 11, 1),
            calendar_duration_years=1,
            gis_config=gis_config,
        )
    if preset == "clatsop_landslide_curriculum":
        zero_events = {"winter": (0, 0), "spring": (0, 0), "summer": (0, 0), "fall": (0, 0)}
        landslide_only_profiles = {
            "landslide": SeasonalDisasterConfig(
                event_count_range_by_season={"winter": (1, 1), "spring": (1, 1), "summer": (1, 1), "fall": (1, 1)},
                size_range_by_season={"winter": (120, 450), "spring": (120, 420), "summer": (100, 320), "fall": (120, 440)},
            ),
            "snow": SeasonalDisasterConfig(event_count_range_by_season=zero_events, size_range_by_season={"winter": (1, 1), "spring": (1, 1), "summer": (1, 1), "fall": (1, 1)}),
            "wildfire_debris": SeasonalDisasterConfig(event_count_range_by_season=zero_events, size_range_by_season={"winter": (1, 1), "spring": (1, 1), "summer": (1, 1), "fall": (1, 1)}),
            "flood": SeasonalDisasterConfig(event_count_range_by_season=zero_events, size_range_by_season={"winter": (1, 1), "spring": (1, 1), "summer": (1, 1), "fall": (1, 1)}),
        }
        return ScenarioConfig(
            resource_counts=ResourceCounts(trucks=6, excavators=3, snowplows=0, assessment_vehicles=0),
            seasonal_spawn=SeasonalSpawnConfig(
                target_events_range=(1, 1),
                interarrival_minutes_range=(300.0, 600.0),
                disasters=landslide_only_profiles,
            ),
            weather_model=WeatherModelConfig(
                enable_spawn_scaling=False,
                enable_dispatch_scaling=False,
                use_vulnerability_weighting=False,
                storm_dispatch_delay_multiplier=1.0,
            ),
            operational_priors=OperationalPriorsConfig(
                time_variance=0.0,
                use_dispatch_delay_priors=False,
            ),
            distance_model=DistanceModelConfig(
                non_gis_distance_unit_miles=meters_to_miles(1.0),
                spawn_distance_range=(0, int(CLATSOP_LOCAL_COORD_MAX * 0.15)),
            ),
            calendar_start_date=datetime(2024, 1, 1),
            calendar_duration_years=1,
            gis_config=gis_config,
        )
    if preset == "clatsop_landslide_curriculum_stage2":
        zero_events = {"winter": (0, 0), "spring": (0, 0), "summer": (0, 0), "fall": (0, 0)}
        landslide_only_profiles = {
            "landslide": SeasonalDisasterConfig(
                event_count_range_by_season={"winter": (2, 2), "spring": (2, 2), "summer": (1, 1), "fall": (2, 2)},
                size_range_by_season={"winter": (150, 650), "spring": (150, 620), "summer": (120, 360), "fall": (150, 640)},
            ),
            "snow": SeasonalDisasterConfig(event_count_range_by_season=zero_events, size_range_by_season={"winter": (1, 1), "spring": (1, 1), "summer": (1, 1), "fall": (1, 1)}),
            "wildfire_debris": SeasonalDisasterConfig(event_count_range_by_season=zero_events, size_range_by_season={"winter": (1, 1), "spring": (1, 1), "summer": (1, 1), "fall": (1, 1)}),
            "flood": SeasonalDisasterConfig(event_count_range_by_season=zero_events, size_range_by_season={"winter": (1, 1), "spring": (1, 1), "summer": (1, 1), "fall": (1, 1)}),
        }
        return ScenarioConfig(
            resource_counts=ResourceCounts(trucks=10, excavators=5, snowplows=0, assessment_vehicles=0),
            seasonal_spawn=SeasonalSpawnConfig(
                target_events_range=(2, 2),
                interarrival_minutes_range=(240.0, 720.0),
                disasters=landslide_only_profiles,
            ),
            weather_model=WeatherModelConfig(
                enable_spawn_scaling=False,
                enable_dispatch_scaling=False,
                use_vulnerability_weighting=False,
                storm_dispatch_delay_multiplier=1.0,
            ),
            operational_priors=OperationalPriorsConfig(
                time_variance=0.0,
                use_dispatch_delay_priors=False,
            ),
            distance_model=DistanceModelConfig(
                non_gis_distance_unit_miles=meters_to_miles(1.0),
                spawn_distance_range=(0, int(CLATSOP_LOCAL_COORD_MAX * 0.2)),
            ),
            calendar_start_date=datetime(2024, 1, 1),
            calendar_duration_years=1,
            gis_config=gis_config,
        )
    if preset == "clatsop_landslide_ops":
        zero_events = {"winter": (0, 0), "spring": (0, 0), "summer": (0, 0), "fall": (0, 0)}
        landslide_only_profiles = {
            "landslide": SeasonalDisasterConfig(
                event_count_range_by_season={"winter": (5, 10), "spring": (3, 7), "summer": (1, 3), "fall": (4, 8)},
                size_range_by_season={"winter": (300, 5000), "spring": (200, 4000), "summer": (100, 1800), "fall": (250, 4500)},
            ),
            "snow": SeasonalDisasterConfig(event_count_range_by_season=zero_events, size_range_by_season={"winter": (1, 1), "spring": (1, 1), "summer": (1, 1), "fall": (1, 1)}),
            "wildfire_debris": SeasonalDisasterConfig(event_count_range_by_season=zero_events, size_range_by_season={"winter": (1, 1), "spring": (1, 1), "summer": (1, 1), "fall": (1, 1)}),
            "flood": SeasonalDisasterConfig(event_count_range_by_season=zero_events, size_range_by_season={"winter": (1, 1), "spring": (1, 1), "summer": (1, 1), "fall": (1, 1)}),
        }
        return ScenarioConfig(
            resource_counts=ResourceCounts(trucks=(12, 18), excavators=(6, 10), snowplows=0, assessment_vehicles=0),
            seasonal_spawn=SeasonalSpawnConfig(
                target_events_range=(8, 16),
                interarrival_minutes_range=(90.0, 900.0),
                disasters=landslide_only_profiles,
            ),
            weather_model=WeatherModelConfig(
                enable_spawn_scaling=True,
                enable_dispatch_scaling=True,
                use_vulnerability_weighting=False,
                storm_dispatch_delay_multiplier=1.8,
            ),
            operational_priors=OperationalPriorsConfig(
                time_variance=0.1,
                use_dispatch_delay_priors=True,
            ),
            distance_model=DistanceModelConfig(
                non_gis_distance_unit_miles=meters_to_miles(1.0),
                spawn_distance_range=(0, CLATSOP_LOCAL_COORD_MAX),
            ),
            calendar_start_date=datetime(2024, 1, 1),
            calendar_duration_years=1,
            gis_config=gis_config,
        )
    if preset == "single_landslide_micro":
        zero_events = {"winter": (0, 0), "spring": (0, 0), "summer": (0, 0), "fall": (0, 0)}
        micro_profiles = {
            "landslide": SeasonalDisasterConfig(
                event_count_range_by_season={"winter": (1, 1), "spring": (0, 0), "summer": (0, 0), "fall": (0, 0)},
                size_range_by_season={"winter": (200, 200), "spring": (200, 200), "summer": (200, 200), "fall": (200, 200)},
            ),
            "snow": SeasonalDisasterConfig(event_count_range_by_season=zero_events, size_range_by_season={"winter": (1, 1), "spring": (1, 1), "summer": (1, 1), "fall": (1, 1)}),
            "wildfire_debris": SeasonalDisasterConfig(event_count_range_by_season=zero_events, size_range_by_season={"winter": (1, 1), "spring": (1, 1), "summer": (1, 1), "fall": (1, 1)}),
            "flood": SeasonalDisasterConfig(event_count_range_by_season=zero_events, size_range_by_season={"winter": (1, 1), "spring": (1, 1), "summer": (1, 1), "fall": (1, 1)}),
        }
        return ScenarioConfig(
            resource_counts=ResourceCounts(trucks=10, excavators=2, snowplows=0, assessment_vehicles=0),
            seasonal_spawn=SeasonalSpawnConfig(
                target_events_range=(1, 1),
                interarrival_minutes_range=(1.0, 1.0),
                disasters=micro_profiles,
            ),
            weather_model=WeatherModelConfig(
                enable_spawn_scaling=False,
                enable_dispatch_scaling=False,
                use_vulnerability_weighting=False,
                storm_dispatch_delay_multiplier=1.0,
            ),
            operational_priors=OperationalPriorsConfig(
                time_variance=0.0,
                use_dispatch_delay_priors=False,
            ),
            distance_model=DistanceModelConfig(
                non_gis_distance_unit_miles=meters_to_miles(1.0),
                spawn_distance_range=(0, 0),
            ),
            calendar_start_date=datetime(2024, 1, 1),
            calendar_duration_years=1,
            gis_config=gis_config,
        )

    raise ValueError(f"Unknown preset '{preset}'.")


def summarize_results(results: defaultdict[str, PolicyResult]) -> dict[str, dict[str, Any]]:
    """Machine-readable summary for result serialization."""
    summaries: dict[str, dict[str, Any]] = {}

    for name, data in results.items():
        success_times = data["success"]
        success_wall_times = data.get("success_wall", [])
        objective_scores = data.get("objective", [])
        kpi_records = data.get("kpis", [])
        fail_count = data["fail"]
        total_runs = len(success_times) + fail_count
        success_rate = (len(success_times) / total_runs) * 100 if total_runs > 0 else 0.0
        kpi_means: dict[str, float] = {}
        if kpi_records:
            kpi_keys = sorted(kpi_records[0].keys())
            for key in kpi_keys:
                kpi_means[key] = statistics.mean([record[key] for record in kpi_records])

        summaries[name] = {
            "runs": total_runs,
            "successes": len(success_times),
            "failures": fail_count,
            "success_rate": success_rate,
            "avg_sim_time": statistics.mean(success_times) if success_times else None,
            "avg_wall_time_s": statistics.mean(success_wall_times) if success_wall_times else None,
            "avg_objective_score": statistics.mean(objective_scores) if objective_scores else None,
            "objective_min": min(objective_scores) if objective_scores else None,
            "objective_max": max(objective_scores) if objective_scores else None,
            "kpi_mean": kpi_means if kpi_means else None,
            "sim_min": min(success_times) if success_times else None,
            "sim_max": max(success_times) if success_times else None,
        }
    return summaries


def run_policy_episode(policy: Policy, seed: int, scenario_config: ScenarioConfig, live_plot: bool) -> tuple[bool, float, float, float, KPIBundle]:
    t0 = time.perf_counter()
    engine = SimPySimulationEngine(policy=policy, seed=seed, scenario_config=scenario_config)
    if live_plot:
        engine.visualizer = EngineVisualizer(engine)
    engine.initialize_world()
    success = engine.run()
    summary = build_simulation_summary(engine)
    return success, summary.non_idle_time, time.perf_counter() - t0, compute_objective_score(summary), compute_kpi_bundle(summary)


def default_ppo_label(model_path: Path, metadata: dict[str, Any]) -> str:
    scenario_name = metadata.get("scenario_name")
    if scenario_name:
        return f"ppo_{scenario_name}"
    return f"ppo_{model_path.stem}"


def load_ppo_model_spec(model_path: str | None) -> PPOModelSpec | None:
    if model_path is None:
        return None
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"PPO model not found: {path}")
    metadata_path = path.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return {
        "label": default_ppo_label(path, metadata),
        "path": str(path.resolve()),
        "metadata": metadata,
    }


def default_dispatch_label(model_path: Path, metadata: dict[str, Any]) -> str:
    scenario_name = metadata.get("scenario_name") or metadata.get("difficulty")
    if scenario_name:
        return f"dispatch_ml_{scenario_name}"
    if model_path.stem == "dispatch_model":
        return f"dispatch_ml_{model_path.parent.name}"
    return f"dispatch_ml_{model_path.stem}"


def load_dispatch_model_spec(model_path: str | None) -> DispatchModelSpec | None:
    if model_path is None:
        return None
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Dispatch model not found: {path}")
    metadata_path = path.parent / "dispatch_model_meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return {
        "label": default_dispatch_label(path, metadata),
        "path": str(path.resolve()),
        "metadata": metadata,
    }


def run_ppo_episode(model: object, seed: int) -> tuple[bool, float, float, float, KPIBundle]:
    episode: PPOEpisodeResult = run_ppo_policy_episode(model, seed, True, "ppo_benchmark")
    return (
        episode["success"],
        episode["sim_time"],
        episode["wall_time_s"],
        episode["objective_score"],
        episode["kpis"],
    )


def run_dispatch_episode(
    model: "TrainedDispatchPolicy",
    seed: int,
    scenario_config: ScenarioConfig,
    scenario_name: str,
) -> tuple[bool, float, float, float, KPIBundle]:
    t0 = time.perf_counter()
    dispatch_env = DisasterResponseGym(
        max_visible_disasters=model.max_visible_disasters,
        sorting_strategy="nearest",
        scenario_config=scenario_config,
        controller_name="dispatch_ml_benchmark",
        scenario_name=scenario_name,
    )
    observation, info = dispatch_env.reset(seed=seed)
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = dispatch_env.step(action)
        _ = reward

    summary = info["summary"]
    return (
        bool(info["is_success"]),
        summary.non_idle_time,
        time.perf_counter() - t0,
        float(info["objective_score"]),
        compute_kpi_bundle(summary),
    )


def approach_column_width(names: list[str] | set[str]) -> int:
    return max([len("APPROACH"), *(len(name) for name in names)], default=len("APPROACH"))


def run_benchmark_single(
    gis_enabled: bool,
    seed_values: list[int],
    preset: str,
    policies: list[str] | None = None,
    ex_policies: list[str] | None = None,
    live_plot: bool = False,
    ppo_model_path: str | None = None,
    mlp_model_path: str | None = None,
) -> defaultdict[str, PolicyResult]:
    """
    Run benchmark for a single configuration (GIS or no GIS).

    Args:
        gis_enabled: Whether to use GIS
        seed_values: Explicit list of seeds to test
        preset: preset level
        ppo_model_path: Path to PPO model
        live_plot: Show live plots
        tournament_only: If True, run ONLY the tournament policy

    Returns:
        Dictionary of results
    """
    config_name = "GIS" if gis_enabled else "No GIS"

    # Create scenario config
    if gis_enabled:
        gis_config = create_gis_config()
        if gis_config is None:
            print(f"ERROR: Could not load GIS config. Skipping {config_name} tests.")
            return defaultdict(
                lambda: {
                    "success": [],
                    "success_wall": [],
                    "objective": [],
                    "kpis": [],
                    "fail": 0,
                }
            )
    else:
        gis_config = None

    scenario_config = create_scenario_config(preset, gis_config)
    print(
        "Scenario config summary: "
        "seasonal=True, "
        f"weather_spawn={scenario_config.weather_model.enable_spawn_scaling}, "
        f"weather_dispatch={scenario_config.weather_model.enable_dispatch_scaling}, "
        f"dispatch_priors={scenario_config.operational_priors.use_dispatch_delay_priors}, "
        f"time_variance={scenario_config.operational_priors.time_variance}, "
        f"calendar_start={scenario_config.calendar_start_date}, "
        f"spawn_interval_min={scenario_config.seasonal_spawn.interarrival_minutes_range}"
    )

    # Store results
    aggregated_results: defaultdict[str, PolicyResult] = defaultdict(
        lambda: {
            "success": [],
            "success_wall": [],
            "objective": [],
            "kpis": [],
            "fail": 0,
        }
    )
    run_policies = []
    if policies is None:
        run_policies = POLICIES + [TOURNAMENT_POLICY]
    else:
        run_policies = [p for p in POLICIES if p.name in policies]
        if TOURNAMENT_POLICY.name in policies:
            run_policies.append(TOURNAMENT_POLICY)

    if ex_policies is not None:
        run_policies = [p for p in run_policies if p.name not in ex_policies]

    run_names = [policy.name for policy in run_policies]
    ppo_spec = load_ppo_model_spec(ppo_model_path)
    ppo_model = load_model(ppo_spec["path"], select_device()) if ppo_spec is not None else None
    dispatch_spec = load_dispatch_model_spec(mlp_model_path)
    if dispatch_spec is not None:
        from scripts.training.mlp.ml_dispatch import TrainedDispatchPolicy

        dispatch_model = TrainedDispatchPolicy.load(dispatch_spec["path"], device=select_device())
    else:
        dispatch_model = None
    if ppo_spec is not None:
        run_names.append(ppo_spec["label"])
    if dispatch_spec is not None:
        run_names.append(dispatch_spec["label"])
    name_width = approach_column_width(run_names)

    print(f"\n{'='*85}")
    print(f"BENCHMARK: Testing with {config_name}")
    print(f"Seeds: {seed_values}, preset: {preset}, policies: {policies}")
    print(f"{'='*85}\n")

    # Test policies on each seed
    for seed in seed_values:
        print(f"\n--- SEED {seed} ---")

        for policy in run_policies:
            print(f"  {policy.name:<{name_width}} | ", end="", flush=True)
            success, sim_duration, wall_duration, objective_score, kpi_bundle = run_policy_episode(policy, seed, scenario_config, live_plot)

            if success:
                aggregated_results[policy.name]["success"].append(sim_duration)
                aggregated_results[policy.name]["success_wall"].append(wall_duration)
            else:
                aggregated_results[policy.name]["fail"] += 1
            aggregated_results[policy.name]["objective"].append(objective_score)
            aggregated_results[policy.name]["kpis"].append(kpi_bundle)

            print(
                f"{'SUCCESS' if success else 'FAIL':<7} | "
                f"Sim: {sim_duration:<10.2f} | "
                f"Obj: {objective_score:<10.2f} | "
                f"Cost: {kpi_bundle['total_spent']:<12.2f} | "
                f"Wall: {wall_duration:<10.2f}s"
            )

        if ppo_spec is not None and ppo_model is not None:
            print(f"  {ppo_spec['label']:<{name_width}} | ", end="")
            success, sim_duration, wall_duration, objective_score, kpi_bundle = run_ppo_episode(ppo_model, seed)

            if success:
                aggregated_results[ppo_spec["label"]]["success"].append(sim_duration)
                aggregated_results[ppo_spec["label"]]["success_wall"].append(wall_duration)
            else:
                aggregated_results[ppo_spec["label"]]["fail"] += 1
            aggregated_results[ppo_spec["label"]]["objective"].append(objective_score)
            aggregated_results[ppo_spec["label"]]["kpis"].append(kpi_bundle)

            print(
                f"{'SUCCESS' if success else 'FAIL':<7} | "
                f"Sim: {sim_duration:<10.2f} | "
                f"Obj: {objective_score:<10.2f} | "
                f"Cost: {kpi_bundle['total_spent']:<12.2f} | "
                f"Wall: {wall_duration:<10.2f}s"
            )

        if dispatch_spec is not None and dispatch_model is not None:
            print(f"  {dispatch_spec['label']:<{name_width}} | ", end="")
            success, sim_duration, wall_duration, objective_score, kpi_bundle = run_dispatch_episode(dispatch_model, seed, scenario_config, preset)

            if success:
                aggregated_results[dispatch_spec["label"]]["success"].append(sim_duration)
                aggregated_results[dispatch_spec["label"]]["success_wall"].append(wall_duration)
            else:
                aggregated_results[dispatch_spec["label"]]["fail"] += 1
            aggregated_results[dispatch_spec["label"]]["objective"].append(objective_score)
            aggregated_results[dispatch_spec["label"]]["kpis"].append(kpi_bundle)

            print(
                f"{'SUCCESS' if success else 'FAIL':<7} | "
                f"Sim: {sim_duration:<10.2f} | "
                f"Obj: {objective_score:<10.2f} | "
                f"Cost: {kpi_bundle['total_spent']:<12.2f} | "
                f"Wall: {wall_duration:<10.2f}s"
            )

    return aggregated_results


def print_results_table(results: defaultdict[str, PolicyResult], title: str = "RESULTS"):
    """Print formatted results table."""
    name_width = approach_column_width(list(results.keys()))
    header = f"{'APPROACH':<{name_width}} | {'SUCCESS %':<10} | {'AVG OBJ':<8} | {'AVG SIM':<12} | " f"{'AVG COST':<12} | {'AVG WALL':<10} | {'SIM MIN':<8} | {'SIM MAX':<8}"
    print(f"\n{title}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    # Calculate statistics
    final_stats: list[tuple[str, float, float | None, float, float | None, float, float, float]] = []
    for name, data in results.items():
        success_times = data["success"]
        success_wall_times = data.get("success_wall", [])
        objective_scores = data.get("objective", [])
        kpi_records = data.get("kpis", [])
        fail_count = data["fail"]
        total_runs = len(success_times) + fail_count

        success_rate = (len(success_times) / total_runs) * 100 if total_runs > 0 else 0

        if success_times:
            avg = statistics.mean(success_times)
            avg_wall = statistics.mean(success_wall_times) if success_wall_times else float("inf")
            mn = min(success_times)
            mx = max(success_times)
        else:
            avg = float("inf")
            avg_wall = float("inf")
            mn = 0
            mx = 0

        avg_objective = statistics.mean(objective_scores) if objective_scores else None
        avg_cost = statistics.mean(record["total_spent"] for record in kpi_records) if kpi_records else None
        final_stats.append((name, success_rate, avg_objective, avg, avg_cost, avg_wall, mn, mx))

    final_stats.sort(key=lambda x: (-x[1], -(x[2] if x[2] is not None else float("-inf")), x[3]))

    for name, rate, avg_objective, avg, avg_cost, avg_wall, mn, mx in final_stats:
        avg_objective_str = f"{avg_objective:.2f}" if avg_objective is not None else "N/A"
        avg_str = f"{avg:.2f}" if avg != float("inf") else "N/A"
        avg_cost_str = f"{avg_cost:.2f}" if avg_cost is not None else "N/A"
        avg_wall_str = f"{avg_wall:.2f}s" if avg_wall != float("inf") else "N/A"
        mn_str = f"{mn:.0f}" if avg != float("inf") else "N/A"
        mx_str = f"{mx:.0f}" if avg != float("inf") else "N/A"

        print(f"{name:<{name_width}} | {rate:<9.1f}% | {avg_objective_str:<8} | {avg_str:<12} | " f"{avg_cost_str:<12} | {avg_wall_str:<10} | {mn_str:<8} | {mx_str:<8}")

    print("=" * len(header) + "\n")


def print_comparison_table(results_no_gis: defaultdict[str, PolicyResult], results_gis: defaultdict[str, PolicyResult]):
    """Print side-by-side comparison of GIS vs No GIS."""
    all_names = set(results_no_gis.keys()) | set(results_gis.keys())
    name_width = approach_column_width(all_names)
    header = f"{'APPROACH':<{name_width}} | {'NO GIS':<35} | {'GIS':<35} | {'DIFFERENCE':<25}"
    subheader = f"{'':<{name_width}} | {'Success % | Avg Time':<35} | {'Success % | Avg Time':<35} | {'Success | Time':<25}"
    print("\n" + "=" * len(header))
    print("COMPARISON: GIS vs No GIS")
    print("=" * len(header))
    print(header)
    print(subheader)
    print("-" * len(header))

    for name in sorted(all_names):
        # No GIS stats
        no_gis_data = results_no_gis.get(name, {"success": [], "success_wall": [], "objective": [], "kpis": [], "fail": 0})
        no_gis_success = no_gis_data["success"]
        no_gis_fail = no_gis_data["fail"]
        no_gis_total = len(no_gis_success) + no_gis_fail
        no_gis_rate = (len(no_gis_success) / no_gis_total * 100) if no_gis_total > 0 else 0
        no_gis_avg = statistics.mean(no_gis_success) if no_gis_success else float("inf")
        no_gis_str = f"{no_gis_rate:>6.1f}% | {no_gis_avg:>8.1f}" if no_gis_success else f"{no_gis_rate:>6.1f}% | {'N/A':>8}"

        # GIS stats
        gis_data = results_gis.get(name, {"success": [], "success_wall": [], "objective": [], "kpis": [], "fail": 0})
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

        print(f"{name:<{name_width}} | {no_gis_str:<35} | {gis_str:<35} | {diff_str:<25}")

    print("=" * len(header) + "\n")


def run_benchmark(
    seeds: list[int],
    preset: str = "clatsop_winter_ops",
    compare_gis: bool = False,
    policies: list[str] | None = None,
    ex_policies: list[str] | None = None,
    live_plot: bool = False,
    ppo_model_path: str | None = None,
    mlp_model_path: str | None = None,
):
    """
    Run benchmark comparing all policies and PPO on shared seeds.

    Args:
        seeds: Number of different seeds to test
        preset: Scenario profile name
        ppo_model_path: Path to trained PPO model (if None, skips PPO testing)
        live_plot: Whether to show live plots for policies
        compare_gis: Whether to test both GIS and non-GIS configurations
        tournament_only: If True, run ONLY the tournament policy (skip all other policies)
    """

    print(f"\n{'#'*85}")
    print(f"#{'':83}#")
    print(f"#{'BENCHMARK':^83}#")
    print(f"#{'':83}#")
    print(f"{'#'*85}\n")

    # Always run without GIS
    results_no_gis = run_benchmark_single(
        gis_enabled=False,
        seed_values=seeds,
        preset=preset,
        policies=policies,
        ex_policies=ex_policies,
        live_plot=live_plot,
        ppo_model_path=ppo_model_path,
        mlp_model_path=mlp_model_path,
    )

    print_results_table(results_no_gis, "RESULTS: NO GIS")

    # Optionally run with GIS
    results_gis = None
    if compare_gis:
        results_gis = run_benchmark_single(
            gis_enabled=True,
            seed_values=seeds,
            preset=preset,
            policies=policies,
            ex_policies=ex_policies,
            live_plot=live_plot,
            ppo_model_path=ppo_model_path,
            mlp_model_path=mlp_model_path,
        )

        print_results_table(results_gis, "RESULTS: GIS")

        # Print comparison
        print_comparison_table(results_no_gis, results_gis)

    return results_no_gis, results_gis


def write_results_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"Wrote benchmark results to {path}")


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Benchmark policies and PPO on shared seeds")
    parser.add_argument("--seeds", type=str, default="5", help="Either a seed count (e.g. 5) or comma-separated seeds (e.g. 1,3,5)")
    parser.add_argument(
        "--preset",
        type=str,
        default="clatsop_winter_ops",
        choices=[
            "clatsop_winter_ops",
            "clatsop_summer_ops",
            "clatsop_storm_stress",
            "clatsop_landslide_curriculum",
            "clatsop_landslide_curriculum_stage2",
            "clatsop_landslide_ops",
            "single_landslide_micro",
        ],
        help="Scenario profile: clatsop_winter_ops, clatsop_summer_ops, clatsop_storm_stress, clatsop_landslide_curriculum, clatsop_landslide_curriculum_stage2, clatsop_landslide_ops, or single_landslide_micro",
    )
    parser.add_argument("--compare-gis", action="store_true", help="Test both GIS and non-GIS configurations for comparison")
    parser.add_argument("--policies", type=str, default=None, help="Comma-separated list of policies to test")
    parser.add_argument("--ex-policies", type=str, default=None, help="Comma-separated list of policies to exclude from testing")
    parser.add_argument("--tournament-depth", type=int, default=1, help="Tree search depth for tournament policy (1=run to completion, 2+=look ahead N decisions)")
    parser.add_argument(
        "--suite",
        type=str,
        default=None,
        choices=list(STANDARD_BENCHMARK_SUITES.keys()),
        help="Run a standardized benchmark suite (overrides --preset/--compare-gis single-run mode)",
    )
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to write machine-readable benchmark results")
    parser.add_argument("--live-plot", action="store_true", help="Show live plots")
    parser.add_argument("--ppo-model-path", type=str, default=None, help="Optional PPO checkpoint to include in benchmark output")
    parser.add_argument("--mlp-model-path", type=str, default=None, help="Optional MLP dispatch checkpoint to include in benchmark output")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    raw_seeds = str(args.seeds).strip()
    if "," in raw_seeds:
        seeds = [int(seed.strip()) for seed in raw_seeds.split(",") if seed.strip()]
    else:
        seeds = list(range(int(raw_seeds)))

    if args.tournament_depth > 1:
        print(f"Using tree search with depth {args.tournament_depth}")
    set_tournament_depth(args.tournament_depth)

    if args.suite is not None:
        suite = STANDARD_BENCHMARK_SUITES[args.suite]
        args.preset = suite["preset"]
        args.compare_gis = bool(suite.get("compare_gis", False))

    results_no_gis, results_gis = run_benchmark(
        seeds=seeds,
        preset=args.preset,
        compare_gis=args.compare_gis,
        policies=args.policies.split(",") if args.policies else None,
        ex_policies=args.ex_policies.split(",") if args.ex_policies else None,
        live_plot=args.live_plot,
        ppo_model_path=args.ppo_model_path,
        mlp_model_path=args.mlp_model_path,
    )
    if args.output_json:
        write_results_json(
            args.output_json,
            {
                "benchmark_version": "v1",
                "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "mode": "single",
                "preset": args.preset,
                "seeds": seeds,
                "compare_gis": args.compare_gis,
                "policies": args.policies,
                "ppo_model_path": args.ppo_model_path,
                "mlp_model_path": args.mlp_model_path,
                "results_no_gis": summarize_results(results_no_gis),
                "results_gis": summarize_results(results_gis) if results_gis is not None else None,
            },
        )
