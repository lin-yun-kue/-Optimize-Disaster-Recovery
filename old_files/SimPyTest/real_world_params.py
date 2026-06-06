"""
Real-world grounded defaults and unit conversion helpers.

These parameters are intentionally conservative Tier C priors sourced from
the local research package in `research/real_world_data/`.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

METERS_PER_MILE = 1609.344
MINUTES_PER_HOUR = 60.0


def meters_to_miles(distance_meters: float) -> float:
    return distance_meters / METERS_PER_MILE


def travel_minutes_from_distance(distance_miles: float, speed_mph: float) -> float:
    if speed_mph <= 0:
        return 0.0
    return (distance_miles / speed_mph) * MINUTES_PER_HOUR


DEFAULT_DISASTER_OPERATIONAL_PRIORS: dict[str, dict[str, Any]] = {
    "landslide": {
        "cause_category": "geologic",
        "closure_type": "full_or_partial",
        "evidence_tier": "tier_c",
        "detection_delay_minutes_range": (10.0, 45.0),
        "dispatch_delay_minutes_range": (20.0, 120.0),
        "assessment_minutes_range": (20.0, 60.0),
        "admin_reopen_delay_minutes_range": (15.0, 120.0),
        "severity_index_range": (0.45, 1.0),
    },
    "snow": {
        "cause_category": "winter_weather",
        "closure_type": "restriction_or_closure",
        "evidence_tier": "tier_c",
        "detection_delay_minutes_range": (5.0, 20.0),
        "dispatch_delay_minutes_range": (10.0, 60.0),
        "assessment_minutes_range": (10.0, 30.0),
        "admin_reopen_delay_minutes_range": (5.0, 60.0),
        "severity_index_range": (0.25, 0.9),
    },
    "wildfire_debris": {
        "cause_category": "wildfire",
        "closure_type": "debris_blockage",
        "evidence_tier": "tier_c",
        "detection_delay_minutes_range": (10.0, 40.0),
        "dispatch_delay_minutes_range": (20.0, 90.0),
        "assessment_minutes_range": (15.0, 45.0),
        "admin_reopen_delay_minutes_range": (15.0, 180.0),
        "severity_index_range": (0.35, 1.0),
    },
    "flood": {
        "cause_category": "hydrologic",
        "closure_type": "flood_closure",
        "evidence_tier": "tier_c",
        "detection_delay_minutes_range": (10.0, 50.0),
        "dispatch_delay_minutes_range": (30.0, 180.0),
        "assessment_minutes_range": (30.0, 90.0),
        "admin_reopen_delay_minutes_range": (20.0, 180.0),
        "severity_index_range": (0.4, 1.0),
    },
}

DEFAULT_POPULATION_IMPACT_PRIORS: dict[str, Any] = {
    # Early Clatsop-oriented exposure priors (Tier B/C blend from local research).
    "daily_traffic": {
        "interstate": 50_000,
        "highway": 12_000,
        "secondary": 1_500,
        "local": 500,
    },
    "vehicle_occupancy": 1.5,
    "truck_impact_gamma": 0.35,
    "default_truck_pct": 0.08,
    # Keep size->impact conversion conservative.
    "size_to_road_miles": 0.01,
    "size_to_duration_hours": 0.08,
}


def get_default_disaster_operational_priors() -> dict[str, dict[str, Any]]:
    return deepcopy(DEFAULT_DISASTER_OPERATIONAL_PRIORS)


def get_default_population_impact_priors() -> dict[str, Any]:
    return deepcopy(DEFAULT_POPULATION_IMPACT_PRIORS)
