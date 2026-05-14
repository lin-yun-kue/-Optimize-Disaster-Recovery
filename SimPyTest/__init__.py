from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "SimPySimulationEngine",
    "ScenarioConfig",
    "ResourceType",
    "Resource",
    "Disaster",
    "Landslide",
    "IdleResources",
    "Depot",
    "DumpSite",
    "DisasterStore",
    "GISConfig",
    "load_roads",
    "build_road_graph",
    "get_road_distance",
]

_EXPORT_MAP: dict[str, tuple[str, str]] = {
    "SimPySimulationEngine": (".engine", "SimPySimulationEngine"),
    "ScenarioConfig": (".scenario_types", "ScenarioConfig"),
    "ResourceType": (".simulation", "ResourceType"),
    "Resource": (".simulation", "Resource"),
    "Disaster": (".simulation", "Disaster"),
    "Landslide": (".simulation", "Landslide"),
    "IdleResources": (".simulation", "IdleResources"),
    "Depot": (".simulation", "Depot"),
    "DumpSite": (".simulation", "DumpSite"),
    "DisasterStore": (".simulation", "DisasterStore"),
    "GISConfig": (".gis_utils", "GISConfig"),
    "load_roads": (".gis_utils", "load_roads"),
    "build_road_graph": (".gis_utils", "build_road_graph"),
    "get_road_distance": (".gis_utils", "get_road_distance"),
}


def __getattr__(name: str) -> Any:
    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
