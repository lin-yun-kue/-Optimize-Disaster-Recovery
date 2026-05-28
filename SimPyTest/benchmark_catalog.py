from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

from SimPyTest.calendar import Season
from SimPyTest.gis_utils import GISConfig
from SimPyTest.scenario_types import (
    ResourceCounts,
    ScenarioConfig,
    SeasonalDisasterConfig,
)

SCENARIO_CATALOG_VERSION = "v2"
OBJECTIVE_VERSION = "v2"


@dataclass(frozen=True)
class BenchmarkScenarioSpec:
    factory: Callable[[GISConfig | None], ScenarioConfig]


@dataclass(frozen=True)
class SeedSetSpec:
    name: str
    description: str
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class BenchmarkSuiteSpec:
    scenario_names: tuple[str, ...]
    seeds: tuple[int, ...]
    gis_enabled: bool


DIFFICULTY_LEVELS: dict[str, dict[str, dict[str, tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]]] = {
    "landslide": {
        "event_count_range_by_season": {
            "winter": ((5, 10), (9, 16), (14, 24)),
            "spring": ((2, 8), (4, 12), (11, 16)),
            "summer": ((0, 2), (0, 4), (2, 7)),
            "fall": ((2, 8), (5, 15), (9, 22)),
        },
        "size_range_by_season": {
            "winter": ((150, 3000), (200, 5000), (700, 7000)),
            # "winter": ((150, 3000), (200, 800), (700, 7000)),
            "spring": ((100, 2200), (100, 3500), (400, 5000)),
            "summer": ((50, 800), (50, 1200), (200, 2500)),
            "fall": ((100, 2500), (100, 4000), (500, 6000)),
        },
    },
    "wildfire_debris": {
        "event_count_range_by_season": {
            "winter": ((0, 0), (0, 0), (0, 1)),
            "spring": ((0, 1), (0, 2), (3, 5)),
            "summer": ((1, 5), (4, 10), (7, 14)),
            "fall": ((1, 3), (1, 5), (3, 8)),
        },
        "size_range_by_season": {
            "winter": ((1, 1), (20, 100), (20, 120)),
            "spring": ((20, 120), (50, 300), (80, 400)),
            "summer": ((80, 900), (100, 1500), (120, 1500)),
            "fall": ((60, 700), (80, 1200), (100, 1200)),
        },
    },
}

SEASONS = ("winter", "spring", "summer", "fall")


def _extract_for_difficulty(difficulty_level: int) -> ScenarioConfig:
    seasonal_spawn = {
        disaster: SeasonalDisasterConfig(**{key: {season: data[key][season][difficulty_level] for season in SEASONS} for key in ("event_count_range_by_season", "size_range_by_season")})
        for disaster, data in DIFFICULTY_LEVELS.items()
    }

    return ScenarioConfig(
        resource_counts=ResourceCounts(trucks=(14, 22), excavators=(7, 11)),
        seasonal_spawn=seasonal_spawn,
        time_variance=0.1,
        calendar_start_date=0.0,
        calendar_duration_years=1,
        gis_config=None,
    )


def _seasonal_counts(
    *,
    winter: tuple[int, int] = (0, 0),
    spring: tuple[int, int] = (0, 0),
    summer: tuple[int, int] = (0, 0),
    fall: tuple[int, int] = (0, 0),
) -> dict[str, tuple[int, int]]:
    return {
        "winter": winter,
        "spring": spring,
        "summer": summer,
        "fall": fall,
    }


def _single_disaster_curriculum_config(
    *,
    trucks: int,
    excavators: int,
    landslide_count: tuple[int, int],
    landslide_size: tuple[int, int],
    wildfire_count: tuple[int, int] = (0, 0),
    wildfire_size: tuple[int, int] = (1, 1),
    time_variance: float,
    calendar_start_date: float = 0.92,
    calendar_duration_years: float = 0.25,
) -> ScenarioConfig:
    return ScenarioConfig(
        resource_counts=ResourceCounts(trucks=trucks, excavators=excavators),
        seasonal_spawn={
            "landslide": SeasonalDisasterConfig(
                event_count_range_by_season=_seasonal_counts(winter=landslide_count),
                size_range_by_season=_seasonal_counts(winter=landslide_size, spring=landslide_size, summer=landslide_size, fall=landslide_size),
            ),
            "wildfire_debris": SeasonalDisasterConfig(
                event_count_range_by_season=_seasonal_counts(
                    winter=wildfire_count,
                    spring=wildfire_count,
                    summer=wildfire_count,
                    fall=wildfire_count,
                ),
                size_range_by_season=_seasonal_counts(
                    winter=wildfire_size,
                    spring=wildfire_size,
                    summer=wildfire_size,
                    fall=wildfire_size,
                ),
            ),
        },
        time_variance=time_variance,
        calendar_start_date=calendar_start_date,
        calendar_duration_years=calendar_duration_years,
        gis_config=None,
    )


def _year_round_variants(scenario: ScenarioConfig, num: int = 5) -> list[ScenarioConfig]:
    ordered_seasons = sorted(Season, key=lambda s: s.value)
    re: list[ScenarioConfig] = []
    for i in range(num):
        copy = deepcopy(scenario)
        start_value = ordered_seasons[i].value
        if start_value < 0:
            start_value += 1
        copy.calendar_start_date = start_value
        copy.calendar_duration_years = 1.0 / num
        re.append(copy)
    return re


def _add_gis_config(scenario: ScenarioConfig, gis_config: GISConfig | None) -> ScenarioConfig:
    if gis_config is None:
        return scenario
    return ScenarioConfig(
        resource_counts=scenario.resource_counts,
        seasonal_spawn=scenario.seasonal_spawn,
        time_variance=scenario.time_variance,
        calendar_start_date=scenario.calendar_start_date,
        calendar_duration_years=scenario.calendar_duration_years,
        gis_config=gis_config,
    )


def _make_benchmark_specs() -> dict[str, BenchmarkScenarioSpec]:
    specs: dict[str, BenchmarkScenarioSpec] = {}
    ordered_seasons = sorted(Season, key=lambda s: s.value)
    season_index_map = {s.name.lower(): i for i, s in enumerate(ordered_seasons)}

    for di, difficulty in enumerate(["easy", "medium", "hard"]):
        sen = _extract_for_difficulty(di)
        years = _year_round_variants(sen, 4)
        for si, season in enumerate(SEASONS):
            season_idx = season_index_map[season]
            specs[f"{difficulty}-{season}"] = BenchmarkScenarioSpec(
                factory=lambda gis_config, yrs=years, idx=season_idx: _add_gis_config(yrs[idx], gis_config),
            )

    specs["clatsop_landslide_curriculum"] = BenchmarkScenarioSpec(
        factory=lambda gis_config: _add_gis_config(
            _single_disaster_curriculum_config(
                trucks=6,
                excavators=3,
                landslide_count=(1, 1),
                landslide_size=(200, 900),
                time_variance=0.0,
            ),
            gis_config,
        ),
    )
    specs["clatsop_landslide_curriculum_stage2"] = BenchmarkScenarioSpec(
        factory=lambda gis_config: _add_gis_config(
            _single_disaster_curriculum_config(
                trucks=10,
                excavators=5,
                landslide_count=(2, 2),
                landslide_size=(250, 1200),
                time_variance=0.0,
            ),
            gis_config,
        ),
    )
    specs["clatsop_landslide_ops"] = BenchmarkScenarioSpec(
        factory=lambda gis_config: _add_gis_config(
            _single_disaster_curriculum_config(
                trucks=12,
                excavators=6,
                landslide_count=(3, 4),
                landslide_size=(300, 1800),
                time_variance=0.1,
                calendar_duration_years=0.4,
            ),
            gis_config,
        ),
    )
    specs["single_landslide_micro"] = BenchmarkScenarioSpec(
        factory=lambda gis_config: _add_gis_config(
            _single_disaster_curriculum_config(
                trucks=4,
                excavators=2,
                landslide_count=(1, 1),
                landslide_size=(100, 250),
                time_variance=0.0,
                calendar_duration_years=0.1,
            ),
            gis_config,
        ),
    )

    # Active documentation and training still refer to these Clatsop names.
    specs["clatsop_winter_ops"] = specs["medium-winter"]
    specs["clatsop_summer_ops"] = specs["medium-summer"]
    specs["clatsop_storm_stress"] = specs["hard-fall"]

    return specs


SCENARIO_SPECS: dict[str, BenchmarkScenarioSpec] = _make_benchmark_specs()


BENCHMARK_SUITES: dict[str, BenchmarkSuiteSpec] = {
    "quick": BenchmarkSuiteSpec(
        scenario_names=("easy-winter", "easy-summer"),
        seeds=(0, 1, 2),
        gis_enabled=False,
    ),
    "full": BenchmarkSuiteSpec(
        scenario_names=("medium-winter", "medium-spring", "medium-summer", "medium-fall", "hard-winter", "hard-spring", "hard-summer", "hard-fall"),
        seeds=tuple(range(200, 220)),
        gis_enabled=False,
    ),
    "gis": BenchmarkSuiteSpec(
        scenario_names=("medium-winter", "medium-spring", "medium-summer", "medium-fall"),
        seeds=tuple(range(200, 220)),
        gis_enabled=True,
    ),
}


def create_scenario_config(name: str, gis_config: GISConfig | None) -> ScenarioConfig:
    try:
        spec = SCENARIO_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown preset '{name}'.") from exc
    return spec.factory(gis_config)
