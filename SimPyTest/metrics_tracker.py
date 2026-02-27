"""
Metrics tracking for simulation and RL training.
Provides comprehensive metrics for analysis and reward shaping.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import TypedDict
from .real_world_params import get_default_population_impact_priors


class DisasterMetrics(TypedDict):
    """Metrics for a single disaster event."""
    disaster_id: int
    disaster_type: str
    start_time: float
    end_time: float | None
    response_time: float  # Time from creation to first resource arrival
    resolution_time: float  # Time from creation to resolution
    total_cost: float
    resources_used: int
    population_affected: int
    road_miles_affected: float


class ResourceMetrics(TypedDict):
    """Metrics for a single resource."""
    resource_id: int
    resource_type: str
    total_hours_operated: float
    total_cost: float
    disasters_assisted: int
    travel_time: float


class SimulationMetrics:
    """
    Tracks comprehensive metrics throughout simulation.
    Integrates with engine for automatic tracking.
    """
    
    def __init__(self):
        self.disaster_metrics: dict[int, DisasterMetrics] = {}
        self.resource_metrics: dict[int, ResourceMetrics] = {}
        
        # Timing tracking
        self._disaster_start_times: dict[int, float] = {}
        self._disaster_first_response: dict[int, float] = {}
        
        # Aggregates
        self.total_disasters_created: int = 0
        self.total_disasters_resolved: int = 0
        self.total_cost: float = 0.0
        self.total_response_time: float = 0.0
        self.total_resolution_time: float = 0.0
        self.total_population_affected: int = 0
        
    def record_disaster_created(
        self,
        disaster_id: int,
        disaster_type: str,
        sim_time: float,
        road_miles: float = 0.0,
        population_affected: int = 0,
    ):
        """Record when a disaster is created."""
        self._disaster_start_times[disaster_id] = sim_time
        self.disaster_metrics[disaster_id] = {
            "disaster_id": disaster_id,
            "disaster_type": disaster_type,
            "start_time": sim_time,
            "end_time": None,
            "response_time": 0.0,
            "resolution_time": 0.0,
            "total_cost": 0.0,
            "resources_used": 0,
            "population_affected": population_affected,
            "road_miles_affected": road_miles,
        }
        self.total_disasters_created += 1
        
    def record_first_response(self, disaster_id: int, sim_time: float):
        """Record when first resource arrives at disaster."""
        if disaster_id in self._disaster_start_times:
            start = self._disaster_start_times[disaster_id]
            self._disaster_first_response[disaster_id] = sim_time
            if disaster_id in self.disaster_metrics:
                self.disaster_metrics[disaster_id]["response_time"] = sim_time - start
                
    def record_disaster_resolved(
        self,
        disaster_id: int,
        sim_time: float,
        total_cost: float,
        resources_used: int,
    ):
        """Record when a disaster is resolved."""
        if disaster_id in self._disaster_start_times:
            start = self._disaster_start_times[disaster_id]
            resolution_time = sim_time - start
            
            if disaster_id in self.disaster_metrics:
                m = self.disaster_metrics[disaster_id]
                m["end_time"] = sim_time
                m["resolution_time"] = resolution_time
                m["total_cost"] = total_cost
                m["resources_used"] = resources_used
                
                # Update aggregates
                self.total_disasters_resolved += 1
                self.total_cost += total_cost
                self.total_response_time += m["response_time"]
                self.total_resolution_time += resolution_time
                self.total_population_affected += m["population_affected"]
                
    def record_resource_metrics(
        self,
        resource_id: int,
        resource_type: str,
        hours_operated: float,
        total_cost: float,
    ):
        """Record resource usage metrics."""
        if resource_id not in self.resource_metrics:
            self.resource_metrics[resource_id] = {
                "resource_id": resource_id,
                "resource_type": resource_type,
                "total_hours_operated": 0.0,
                "total_cost": 0.0,
                "disasters_assisted": 0,
                "travel_time": 0.0,
            }
        
        m = self.resource_metrics[resource_id]
        m["total_hours_operated"] += hours_operated
        m["total_cost"] += total_cost
        
    def increment_disaster_assist(self, resource_id: int):
        """Increment disasters assisted count for a resource."""
        if resource_id in self.resource_metrics:
            self.resource_metrics[resource_id]["disasters_assisted"] += 1
            
    def get_summary(self) -> dict:
        """Get summary statistics."""
        resolved = self.total_disasters_resolved
        created = self.total_disasters_created
        
        return {
            "total_disasters_created": created,
            "total_disasters_resolved": resolved,
            "resolution_rate": resolved / created if created > 0 else 0,
            "total_cost": self.total_cost,
            "avg_response_time": self.total_response_time / resolved if resolved > 0 else 0,
            "avg_resolution_time": self.total_resolution_time / resolved if resolved > 0 else 0,
            "total_population_affected": self.total_population_affected,
            "resource_count": len(self.resource_metrics),
        }
        
    def get_per_disaster_stats(self) -> list[DisasterMetrics]:
        """Get per-disaster metrics."""
        return list(self.disaster_metrics.values())
        
    def get_per_resource_stats(self) -> list[ResourceMetrics]:
        """Get per-resource metrics."""
        return list(self.resource_metrics.values())


class PopulationImpact:
    """Calculate population impact from road closures."""

    _PRIORS = get_default_population_impact_priors()
    DAILY_TRAFFIC = _PRIORS["daily_traffic"]
    VEHICLE_OCCUPANCY = float(_PRIORS["vehicle_occupancy"])
    TRUCK_IMPACT_GAMMA = float(_PRIORS["truck_impact_gamma"])
    DEFAULT_TRUCK_PCT = float(_PRIORS["default_truck_pct"])
    SIZE_TO_ROAD_MILES = float(_PRIORS["size_to_road_miles"])
    SIZE_TO_DURATION_HOURS = float(_PRIORS["size_to_duration_hours"])
    
    @staticmethod
    def estimate(
        road_miles: float,
        road_type: str = "secondary",
        duration_hours: float = 24.0,
        alternative_routes: int = 0,
        truck_pct: float | None = None,
        seasonal_factor: float = 1.0,
    ) -> int:
        """Estimate population affected by road closure.
        
        Args:
            road_miles: Miles of road affected
            road_type: "interstate", "highway", "secondary", or "local"
            duration_hours: Duration of closure
            alternative_routes: Number of viable alternative routes
        
        Returns:
            Estimated number of people affected
        """
        base_volume = PopulationImpact.DAILY_TRAFFIC.get(road_type, PopulationImpact.DAILY_TRAFFIC["local"])

        # Detour factor: fewer alternatives = more impact
        detour_multiplier = 1.0 / (1 + alternative_routes * 0.5)
        truck_share = PopulationImpact.DEFAULT_TRUCK_PCT if truck_pct is None else max(0.0, min(1.0, truck_pct))
        truck_multiplier = 1.0 + PopulationImpact.TRUCK_IMPACT_GAMMA * truck_share

        # Calculate affected vehicles
        vehicles_affected = base_volume * seasonal_factor * truck_multiplier * detour_multiplier * (duration_hours / 24)

        people_affected = int(vehicles_affected * PopulationImpact.VEHICLE_OCCUPANCY)
        
        return people_affected
    
    @staticmethod
    def estimate_from_disaster_size(size: float, road_type: str = "secondary") -> int:
        """Estimate population from disaster size.
        
        Args:
            size: Disaster size (e.g., cubic yards for landslide)
            road_type: Type of road affected
        
        Returns:
            Estimated population affected
        """
        miles_affected = max(0.1, size * PopulationImpact.SIZE_TO_ROAD_MILES)
        hours_affected = max(1.0, size * PopulationImpact.SIZE_TO_DURATION_HOURS)
        
        return PopulationImpact.estimate(
            road_miles=miles_affected,
            road_type=road_type,
            duration_hours=hours_affected,
        )
