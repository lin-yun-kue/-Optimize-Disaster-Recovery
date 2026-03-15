"""
Calendar and seasonal system for multi-year disaster simulation.
Handles time progression, seasonal probabilities, and weather patterns.
"""

from __future__ import annotations
from collections.abc import Generator
from datetime import datetime, timedelta
from enum import Enum, auto
import random
from typing import TYPE_CHECKING

import simpy
from SimPyTest.simulation import Disaster, DumpSite, FloodEvent, Landslide, ResourceType, SnowEvent, WildfireDebris

if TYPE_CHECKING:
    from SimPyTest.engine import SimPySimulationEngine


class Season(Enum):
    """Seasons of the year."""

    WINTER = auto()
    SPRING = auto()
    SUMMER = auto()
    FALL = auto()


class SimulationCalendar:
    """
    Calendar system that manages time progression and seasonal patterns.
    """

    def __init__(self, start_date: datetime, duration_years: int, seed: int = 0):
        self.start_date = start_date
        self.current_date = start_date
        self.end_date = start_date + timedelta(days=365 * duration_years)
        self.duration_years = duration_years
        self._rng = random.Random(seed)

        # Weather state (persistent across days)
        self.weather_state = {
            "rain_intensity": 0.0,  # 0-1, affects landslide/flood
            "temperature": 15.0,  # Celsius, affects snow/ice
            "wind_speed": 0.0,  # Affects wildfire spread
            "drought_index": 0.0,  # 0-1, affects wildfire risk
        }

    def advance_time(self, hours: float) -> datetime:
        """Backward-compatible hour-based advancement."""
        return self.advance_time_minutes(hours * 60.0)

    def advance_time_minutes(self, minutes: float) -> datetime:
        """Advance the calendar by specified real-world minutes."""
        self.current_date += timedelta(minutes=minutes)
        self._update_weather()
        return self.current_date

    def _update_weather(self):
        """Update weather state based on season and random variation."""
        season = self.get_season()

        # Seasonal baselines
        if season == Season.WINTER:
            base_temp = 5.0
            base_rain = 0.4
            base_drought = 0.1
        elif season == Season.SPRING:
            base_temp = 12.0
            base_rain = 0.5
            base_drought = 0.2
        elif season == Season.SUMMER:
            base_temp = 22.0
            base_rain = 0.1
            base_drought = 0.6
        else:  # FALL
            base_temp = 15.0
            base_rain = 0.6
            base_drought = 0.3

        # Random walk with mean reversion
        noise = self._rng.gauss(0, 0.1)
        self.weather_state["temperature"] = 0.9 * self.weather_state["temperature"] + 0.1 * base_temp + noise * 5
        self.weather_state["rain_intensity"] = max(0, min(1, 0.8 * self.weather_state["rain_intensity"] + 0.2 * base_rain + noise))
        self.weather_state["drought_index"] = max(0, min(1, 0.9 * self.weather_state["drought_index"] + 0.1 * base_drought + noise * 0.1))
        self.weather_state["wind_speed"] = max(0, self._rng.gauss(15, 10))  # km/h

    def get_season(self) -> Season:
        """Get current season."""
        month = self.current_date.month
        if month in [12, 1, 2]:
            return Season.WINTER
        elif month in [3, 4, 5]:
            return Season.SPRING
        elif month in [6, 7, 8]:
            return Season.SUMMER
        else:
            return Season.FALL

    def get_year_progress(self) -> float:
        """Get progress through current year (0-1)."""
        year_start = datetime(self.current_date.year, 1, 1)
        year_end = datetime(self.current_date.year + 1, 1, 1)
        return (self.current_date - year_start).total_seconds() / (year_end - year_start).total_seconds()

    def is_complete(self) -> bool:
        """Check if simulation duration is complete."""
        return self.current_date >= self.end_date

    def get_weather_factor(self, disaster_type: str) -> float:
        """Get weather modifier for a specific disaster type."""
        if disaster_type == "landslide":
            return self.weather_state["rain_intensity"]
        elif disaster_type == "snow":
            # High chance when temp < 2°C
            if self.weather_state["temperature"] < 2:
                return 1.0
            elif self.weather_state["temperature"] < 5:
                return 0.5
            else:
                return 0.0
        elif disaster_type in ("wildfire_debris", "wildfire"):
            return self.weather_state["drought_index"] * (1 + self.weather_state["wind_speed"] / 50)
        elif disaster_type == "flood":
            return self.weather_state["rain_intensity"]
        return 0.0

    def get_simulated_hours(self) -> float:
        """Get total simulated hours elapsed."""
        return (self.current_date - self.start_date).total_seconds() / 3600

    def get_year(self) -> int:
        """Get current simulation year (1-indexed)."""
        return (self.current_date - self.start_date).days // 365 + 1

    def __str__(self) -> str:
        return f"{self.current_date.strftime('%Y-%m-%d %H:%M')} (Year {self.get_year()})"


# ============================================================================
# MARK: Seasonal Spawn
# ============================================================================


def add_seasonal_disasters(engine: SimPySimulationEngine, dump_site: DumpSite) -> Generator[simpy.Event, object, None]:
    calendar = engine.calendar
    target_low, target_high = engine.scenario_config.seasonal_spawn.target_events_range
    target_total_disasters = engine.rng.randint(target_low, target_high)
    max_sim_minutes = (calendar.end_date - calendar.start_date).total_seconds() / 60.0
    min_gap, max_gap = engine.scenario_config.seasonal_spawn.interarrival_minutes_range

    seasonal_disaster_classes: dict[str, type[Disaster]] = {cls.disaster_type: cls for cls in (Landslide, SnowEvent, WildfireDebris, FloodEvent)}
    profiles = engine.scenario_config.seasonal_spawn.disasters

    event_times: list[float] = []
    event_time = 0.0
    while len(event_times) < target_total_disasters and event_time < max_sim_minutes:
        event_time += engine.rng.uniform(min_gap, max_gap)
        if event_time < max_sim_minutes:
            event_times.append(event_time)

    for scheduled_time in event_times:
        if scheduled_time > engine.env.now:
            yield engine.env.timeout(scheduled_time - engine.env.now)
        if engine.env.now >= max_sim_minutes:
            break

        season_enum = calendar.get_season()
        season = season_enum.name.lower()
        weighted_candidates: list[tuple[str, float]] = []
        for disaster_type, profile in profiles.items():
            count_range = profile.event_count_range_by_season.get(season, (0, 0))
            base_weight = max(0.0, (float(count_range[0]) + float(count_range[1])) / 2.0)
            if base_weight <= 0:
                continue
            weight = base_weight
            if engine.scenario_config.weather_model.enable_spawn_scaling:
                weather = calendar.get_weather_factor(disaster_type)
                max_boost = engine.scenario_config.weather_model.max_rate_weather_boost
                weight *= 1.0 + max(0.0, min(max_boost, weather))
            if engine.scenario_config.weather_model.use_vulnerability_weighting:
                weight *= engine.scenario_config.get_vulnerability_multiplier(disaster_type)
            if weight > 0:
                weighted_candidates.append((disaster_type, weight))

        if not weighted_candidates:
            continue

        total_weight = sum(weight for _, weight in weighted_candidates)
        roll = engine.rng.random() * total_weight
        disaster_type = weighted_candidates[0][0]
        cumulative = 0.0
        for key, weight in weighted_candidates:
            cumulative += weight
            if roll <= cumulative:
                disaster_type = key
                break

        disaster_cls = seasonal_disaster_classes[disaster_type]
        profile = profiles[disaster_type]
        size_range = profile.size_range_by_season.get(season, (1, 1))
        weather_mod = 1.0
        if engine.scenario_config.weather_model.enable_spawn_scaling and disaster_cls.disaster_type is not None:
            weather_mod = calendar.get_weather_factor(disaster_cls.disaster_type)
            low = engine.scenario_config.weather_model.size_weather_scale_min
            high = engine.scenario_config.weather_model.size_weather_scale_max
            weather_mod = max(low, min(high, 0.5 + weather_mod))

        sampled_range = (
            max(1, int(size_range[0] * weather_mod)),
            max(1, int(size_range[1] * weather_mod)),
        )
        location = engine.generate_disaster_locations(1)[0]
        disaster = disaster_cls.spawn_from_seasonal(
            engine=engine,
            dump_site=dump_site,
            location=location,
            size_range=sampled_range,
            season=season_enum,
        )
        if engine.scenario_config.weather_model.use_vulnerability_weighting and disaster_cls.disaster_type is not None:
            disaster.vulnerability_index = engine.scenario_config.get_vulnerability_multiplier(disaster_cls.disaster_type)
        engine.disaster_store.put(disaster)
        engine.record_disaster_created_metrics(disaster)
