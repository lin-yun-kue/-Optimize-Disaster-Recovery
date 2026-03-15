from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, TYPE_CHECKING

from SimPyTest.real_world_params import get_default_disaster_operational_priors

if TYPE_CHECKING:
    from SimPyTest.engine import SimulationRNG


class _IntRangeSampler(Protocol):
    def randint(self, a: int, b: int) -> int: ...


@dataclass
class ResourceCounts:
    trucks: int | tuple[int, int]
    excavators: int | tuple[int, int]
    snowplows: int | tuple[int, int]
    assessment_vehicles: int | tuple[int, int]

    def resolve(self, rng: _IntRangeSampler, value: int | tuple[int, int]) -> int:
        if isinstance(value, tuple):
            return rng.randint(value[0], value[1])
        return value


@dataclass
class SeasonalDisasterConfig:
    event_count_range_by_season: dict[str, tuple[int, int]]
    size_range_by_season: dict[str, tuple[int, int]]


@dataclass
class SeasonalSpawnConfig:
    target_events_range: tuple[int, int]
    interarrival_minutes_range: tuple[float, float]
    disasters: dict[str, SeasonalDisasterConfig]


@dataclass
class WeatherModelConfig:
    enable_spawn_scaling: bool
    enable_dispatch_scaling: bool
    use_vulnerability_weighting: bool
    vulnerability_multipliers: dict[str, float] = field(default_factory=dict)
    # Spawn/weather scaling clamps.
    max_rate_weather_boost: float = 1.5
    size_weather_scale_min: float = 0.5
    size_weather_scale_max: float = 2.5
    # Dispatch/weather scaling.
    storm_dispatch_delay_multiplier: float = 2.0


@dataclass
class OperationalPriorsConfig:
    time_variance: float
    use_dispatch_delay_priors: bool = True
    disaster_operational_priors: dict[str, dict[str, object]] = field(default_factory=get_default_disaster_operational_priors)
    closure_minutes_range_by_disaster: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "landslide": (120.0, 10_080.0),
            "snow": (60.0, 1_440.0),
            "wildfire_debris": (120.0, 7_200.0),
            "flood": (360.0, 20_160.0),
        }
    )
    snow_road_miles_range: tuple[float, float] = (2.0, 10.0)
    snow_auto_resolve_multiplier: float = 3.0
    snow_work_interval_minutes: float = 60.0
    flood_assessment_minutes_range: tuple[float, float] = (30.0, 90.0)
    flood_status_check_interval_minutes: float = 60.0
    flood_post_recede_job_probabilities: dict[str, float] = field(
        default_factory=lambda: {
            "standing_water_only": 0.4,
            "debris_cleanup": 0.4,
            "washout_repair": 0.2,
        }
    )
    flood_debris_work_range: tuple[float, float] = (20.0, 120.0)
    flood_washout_work_range: tuple[float, float] = (120.0, 600.0)
    flood_work_interval_minutes: float = 60.0
    flood_debris_work_rate_multiplier: float = 1.0
    flood_washout_work_rate_multiplier: float = 0.45
    flood_reopen_minutes_range: tuple[float, float] = (15.0, 60.0)


@dataclass
class DistanceModelConfig:
    non_gis_distance_unit_miles: float
    spawn_distance_range: tuple[int, int]


@dataclass
class ScenarioConfig:
    resource_counts: ResourceCounts
    seasonal_spawn: SeasonalSpawnConfig
    weather_model: WeatherModelConfig
    operational_priors: OperationalPriorsConfig
    distance_model: DistanceModelConfig
    calendar_start_date: datetime
    calendar_duration_years: int
    gis_config: Any | None = None

    def get_vulnerability_multiplier(self, disaster_type: str) -> float:
        return max(0.0, float(self.weather_model.vulnerability_multipliers.get(disaster_type, 1.0)))

    def resolve_resource_count(self, rng: SimulationRNG, resource_type: str) -> int:
        lookup: dict[str, int | tuple[int, int]] = {
            "truck": self.resource_counts.trucks,
            "excavator": self.resource_counts.excavators,
            "snowplow": self.resource_counts.snowplows,
            "assessment_vehicle": self.resource_counts.assessment_vehicles,
        }
        value = lookup[resource_type]
        return self.resource_counts.resolve(rng, value)
