"""
Calendar and seasonal system for multi-year disaster simulation.
Handles time progression, seasonal probabilities
"""

from __future__ import annotations
from collections.abc import Generator
from enum import Enum
from math import floor

import simpy
from SimPyTest.simulation import Disaster, Landslide, WildfireDebris

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from SimPyTest.engine import SimPySimulationEngine


class Season(Enum):
    """Seasons of the year."""

    WINTER = -0.09
    SPRING = 0.16
    SUMMER = 0.41
    FALL = 0.66


class SimulationCalendar:
    """
    Calendar system that manages time progression and seasonal patterns.
    start_date: float Fraction of year to start at
    duration_years: float Number of years to go after start_date
    """

    def __init__(self, start_date: float, duration_years: float):
        self.current_date: float = start_date
        self.end_date: float = start_date + duration_years
        self.duration_years: float = duration_years

    def advance_time_minutes(self, minutes: float) -> float:
        """Advance the calendar by specified real-world minutes."""
        self.current_date += minutes / 525600
        self.current_date %= 1
        return self.current_date

    def get_season(self) -> Season:
        """
        Get current season.

        Winter: 0.91 -> 0.16
        Spring: 0.16 -> 0.41
        Summer: 0.41 -> 0.66
        Fall: 0.66 -> 0.91
        """

        if 0.16 < self.current_date <= 0.41:
            return Season.SPRING
        elif 0.41 < self.current_date <= 0.66:
            return Season.SUMMER
        elif 0.66 < self.current_date <= 0.91:
            return Season.FALL
        else:
            return Season.WINTER

    def get_year_progress(self) -> float:
        """Get progress through current year (0-1)."""
        return self.current_date


# ============================================================================
# MARK: Seasonal Spawn
# ============================================================================


def add_seasonal_disasters(engine: SimPySimulationEngine) -> Generator[simpy.Event, object, None]:
    calendar = engine.calendar
    rng = engine.rng

    max_sim_minutes = calendar.duration_years * 525600
    sim_year_span = floor(calendar.current_date + calendar.duration_years) + 1

    disaster_classes: dict[str, type[Disaster]] = {cls.disaster_type: cls for cls in (Landslide, WildfireDebris)}
    profiles = engine.scenario_config.seasonal_spawn

    event_times: list[tuple[float, str, int]] = []
    cluster_probability = 0.5
    cluster_size_range = (2, 3)
    cluster_gap_minutes = (10, 90)

    for year in range(sim_year_span):
        for season_enum in Season:
            for disaster_type, profile in profiles.items():
                count_range = profile.event_count_range_by_season.get(season_enum.name.lower(), (0, 0))
                disaster_count = rng.randint(count_range[0], count_range[1])
                event_offsets: list[float] = []
                remaining = disaster_count

                while remaining > 0:
                    use_cluster = remaining >= cluster_size_range[0] and rng.random() < cluster_probability
                    cluster_size = rng.randint(cluster_size_range[0], cluster_size_range[1]) if use_cluster else 1
                    cluster_size = min(cluster_size, remaining)

                    anchor_offset = rng.uniform(0.0, 0.25)
                    event_offsets.append(anchor_offset)

                    accumulated_gap = 0.0
                    for _ in range(cluster_size - 1):
                        accumulated_gap += rng.uniform(cluster_gap_minutes[0], cluster_gap_minutes[1]) / 525600
                        clustered_offset = min(anchor_offset + accumulated_gap, 0.25 - 1e-6)
                        event_offsets.append(clustered_offset)

                    remaining -= cluster_size



                for offset in event_offsets:
                    # Year time in season
                    base_time = (offset + season_enum.value) % 1
                    if base_time < calendar.current_date:
                        base_time += 1
                    event_time = (base_time - calendar.current_date) * 525600

                    event_size = profile.size_range_by_season.get(season_enum.name.lower(), (1, 1))
                    event_size = rng.randint(event_size[0], event_size[1])

                    event_times.append((event_time, disaster_type, event_size))

    event_times.sort(key=lambda x: x[0])

    for event in event_times:
        scheduled_time = event[0]
        disaster_type = event[1]
        event_size = event[2]

        if scheduled_time < engine.env.now:
            continue

        if scheduled_time > engine.env.now:
            yield engine.env.timeout(scheduled_time - engine.env.now)
        if engine.env.now >= max_sim_minutes:
            break

        disaster_cls = disaster_classes[disaster_type]
        profile = profiles[disaster_type]

        location = engine.generate_disaster_locations(1)[0]
        disaster = disaster_cls.spawn_from_seasonal(
            engine=engine,
            location=location,
            size=event_size,
        )
        engine.disaster_store.put(disaster)
        engine.record_disaster_created_metrics(disaster)
