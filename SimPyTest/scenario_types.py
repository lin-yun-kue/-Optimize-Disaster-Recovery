from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from SimPyTest.simulation import ResourceType

if TYPE_CHECKING:
    from SimPyTest.engine import SimulationRNG


@dataclass
class ResourceCounts:
    trucks: int | tuple[int, int]
    excavators: int | tuple[int, int]

    def resolve(self, rng: SimulationRNG, value: int | tuple[int, int]) -> int:
        if isinstance(value, tuple):
            return rng.randint(value[0], value[1])
        return value


@dataclass
class SeasonalDisasterConfig:
    event_count_range_by_season: dict[str, tuple[int, int]]
    size_range_by_season: dict[str, tuple[int, int]]


@dataclass
class ScenarioConfig:
    resource_counts: ResourceCounts
    seasonal_spawn: dict[str, SeasonalDisasterConfig]
    time_variance: float
    calendar_start_date: float
    calendar_duration_years: float
    gis_config: Any | None = None

    def resolve_resource_count(self, rng: SimulationRNG, resource_type: ResourceType) -> int:
        lookup: dict[ResourceType, int | tuple[int, int]] = {
            ResourceType.TRUCK: self.resource_counts.trucks,
            ResourceType.EXCAVATOR: self.resource_counts.excavators,
        }
        value = lookup[resource_type]
        return self.resource_counts.resolve(rng, value)
