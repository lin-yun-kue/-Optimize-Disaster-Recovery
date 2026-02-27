"""
Multi-year simulation configuration system.
Defines budgets, resource costs, and scenario parameters.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable
from enum import Enum
import random

from .calendar import OREGON_DISASTER_PATTERNS


class ResourceClass(Enum):
    """Resource classification for budget and operational planning."""
    EXCAVATOR = "excavator"
    TRUCK = "truck"
    SNOWPLOW = "snowplow"
    CRANE = "crane"
    ASSESSMENT_VEHICLE = "assessment_vehicle"


@dataclass
class ResourceCosts:
    """Cost structure for a resource type."""
    purchase_cost: float
    hourly_operating: float
    fuel_per_hour: float
    fuel_cost_per_gallon: float = 4.50
    
    @property
    def total_hourly_cost(self) -> float:
        """Total hourly cost including fuel."""
        return self.hourly_operating + (self.fuel_per_hour * self.fuel_cost_per_gallon)


@dataclass
class SiteCapacity:
    """Capacity configuration for work sites."""
    max_concurrent: int
    efficiency_curve: Callable[[int], float]
    
    @staticmethod
    def linear_decline(max_cap: int) -> Callable[[int], float]:
        """Linear efficiency decline after max capacity."""
        def efficiency(n: int) -> float:
            if n <= max_cap:
                return 1.0
            return 1.0 + (n - max_cap) * 0.5 / max_cap
        return efficiency
    
    @staticmethod
    def logarithmic(max_cap: int) -> Callable[[int], float]:
        """Logarithmic efficiency - diminishing returns kick in quickly."""
        def efficiency(n: int) -> float:
            if n <= max_cap:
                return 1.0
            return 1.0 + 0.3 * (n - max_cap) / max_cap
        return efficiency


@dataclass
class BudgetConfig:
    """Budget configuration for a single year."""
    annual_budget: float
    
    year_multiplier: float = 1.0
    
    @property
    def effective_budget(self) -> float:
        return self.annual_budget * self.year_multiplier


# Default resource costs
DEFAULT_RESOURCE_COSTS: dict[ResourceClass, ResourceCosts] = {
    # Tier C priors until internal ODOT cost tables are integrated.
    ResourceClass.EXCAVATOR: ResourceCosts(
        purchase_cost=250_000,
        hourly_operating=150,
        fuel_per_hour=8.0,
    ),
    ResourceClass.TRUCK: ResourceCosts(
        purchase_cost=120_000,
        hourly_operating=75,
        fuel_per_hour=5.0,
    ),
    ResourceClass.SNOWPLOW: ResourceCosts(
        purchase_cost=180_000,
        hourly_operating=100,
        fuel_per_hour=6.0,
    ),
    ResourceClass.ASSESSMENT_VEHICLE: ResourceCosts(
        purchase_cost=95_000,
        hourly_operating=60,
        fuel_per_hour=3.5,
    ),
}

# Default site capacities
DEFAULT_SITE_CAPACITIES: dict[str, SiteCapacity] = {
    "landslide": SiteCapacity(
        max_concurrent=4,
        efficiency_curve=SiteCapacity.logarithmic(4)
    ),
    "snow": SiteCapacity(
        max_concurrent=2,
        efficiency_curve=SiteCapacity.linear_decline(2)
    ),
    "wildfire_debris": SiteCapacity(
        max_concurrent=6,
        efficiency_curve=SiteCapacity.logarithmic(6)
    ),
    "flood": SiteCapacity(
        max_concurrent=3,
        efficiency_curve=SiteCapacity.linear_decline(3)
    ),
}


def get_default_resource_costs() -> dict[ResourceClass, ResourceCosts]:
    """Get default resource costs."""
    return DEFAULT_RESOURCE_COSTS.copy()


def get_default_site_capacities() -> dict[str, SiteCapacity]:
    """Get default site capacities."""
    return DEFAULT_SITE_CAPACITIES.copy()
