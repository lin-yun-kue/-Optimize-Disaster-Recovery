"""
Calendar and seasonal system for multi-year disaster simulation.
Handles time progression, seasonal probabilities, and weather patterns.
"""

from __future__ import annotations
from datetime import datetime, timedelta
from enum import Enum, auto
from dataclasses import dataclass
from typing import Callable
import random


class Season(Enum):
    """Seasons of the year."""
    WINTER = auto()
    SPRING = auto()
    SUMMER = auto()
    FALL = auto()


@dataclass
class SeasonalFactors:
    """Factors that vary by season for a disaster type."""
    base_rate: float  # Base probability per day
    seasonal_multiplier: dict[Season, float]  # Multiplier by season
    size_distribution: dict[str, tuple[float, float]]  # (min, max) for different sizes
    
    def get_rate_for_date(self, date: datetime, weather_noise: float = 0.0) -> float:
        """Get the adjusted rate for a specific date."""
        season = self._get_season(date)
        base = self.base_rate * self.seasonal_multiplier.get(season, 1.0)
        # Add weather noise (gaussian, typically ±20%)
        return max(0, base * (1 + weather_noise))
    
    @staticmethod
    def _get_season(date: datetime) -> Season:
        month = date.month
        if month in [12, 1, 2]:
            return Season.WINTER
        elif month in [3, 4, 5]:
            return Season.SPRING
        elif month in [6, 7, 8]:
            return Season.SUMMER
        else:
            return Season.FALL


class SimulationCalendar:
    """
    Calendar system that manages time progression and seasonal patterns.
    """
    
    def __init__(self, start_date: datetime, duration_years: int):
        self.start_date = start_date
        self.current_date = start_date
        self.end_date = start_date + timedelta(days=365 * duration_years)
        self.duration_years = duration_years
        
        # Simulation speed: 1 sim minute = X real seconds
        # For 5-year sims, we'll run at accelerated time
        self.time_acceleration = 60  # 1 real second = 1 sim hour by default
        
        # Weather state (persistent across days)
        self.weather_state = {
            "rain_intensity": 0.0,  # 0-1, affects landslide/flood
            "temperature": 15.0,    # Celsius, affects snow/ice
            "wind_speed": 0.0,      # Affects wildfire spread
            "drought_index": 0.0,   # 0-1, affects wildfire risk
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
        noise = random.gauss(0, 0.1)
        self.weather_state["temperature"] = 0.9 * self.weather_state["temperature"] + 0.1 * base_temp + noise * 5
        self.weather_state["rain_intensity"] = max(0, min(1, 0.8 * self.weather_state["rain_intensity"] + 0.2 * base_rain + noise))
        self.weather_state["drought_index"] = max(0, min(1, 0.9 * self.weather_state["drought_index"] + 0.1 * base_drought + noise * 0.1))
        self.weather_state["wind_speed"] = max(0, random.gauss(15, 10))  # km/h
    
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


# Pre-defined realistic seasonal patterns for Oregon
# These are synthetic but based on Oregon's climate patterns
OREGON_DISASTER_PATTERNS = {
    "landslide": SeasonalFactors(
        # Clatsop is rain-dominant with frequent slide/flood closures in wet season.
        # Base rate remains moderate but winter/fall are emphasized.
        base_rate=0.035,
        seasonal_multiplier={
            Season.WINTER: 1.2,
            Season.SPRING: 0.8,
            Season.SUMMER: 0.15,
            Season.FALL: 1.0,
        },
        size_distribution={
            # Debris volume priors (cubic yards proxy). Includes rare "major"
            # events informed by Clatsop hazard-plan examples (thousands of yd^3).
            "small": (50, 200),
            "medium": (200, 800),
            "large": (800, 2500),
            "major": (3000, 5000),
        }
    ),
    "snow": SeasonalFactors(
        # Coastal Clatsop is mild; snow closures exist but are less frequent than
        # in inland/mountain districts. Keep short-duration events common in winter.
        base_rate=0.05,
        seasonal_multiplier={
            Season.WINTER: 1.0,
            Season.SPRING: 0.2,
            Season.SUMMER: 0.0,
            Season.FALL: 0.05,
        },
        size_distribution={
            "light": (1, 4),     # closure/restriction hours
            "moderate": (4, 12),
            "heavy": (12, 24),
        }
    ),
    "wildfire_debris": SeasonalFactors(
        base_rate=0.015,
        seasonal_multiplier={
            Season.WINTER: 0.0,
            Season.SPRING: 0.1,
            Season.SUMMER: 1.0,
            Season.FALL: 0.7,
        },
        size_distribution={
            "small": (20, 100),   # cubic yards of debris
            "medium": (100, 400),
            "large": (400, 1200),
        }
    ),
    "flood": SeasonalFactors(
        base_rate=0.045,
        seasonal_multiplier={
            Season.WINTER: 1.1,
            Season.SPRING: 0.8,
            Season.SUMMER: 0.1,
            Season.FALL: 0.5,
        },
        size_distribution={
            "minor": (12, 48),   # hours of closure
            "moderate": (48, 168),
            "major": (168, 336),  # multi-day/week closures
        }
    ),
}
