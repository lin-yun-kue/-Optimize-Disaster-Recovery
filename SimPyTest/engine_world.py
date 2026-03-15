from __future__ import annotations

import math
from typing import TYPE_CHECKING

from SimPyTest.clatsop_spatial import (
    default_non_gis_depot_location,
    default_non_gis_dump_site_location,
    sample_clatsop_local_utm,
)
from SimPyTest.gis_utils import CRS_UTM, get_road_distance
from SimPyTest.real_world_params import meters_to_miles
from .simulation import Depot, DumpSite, Resource, ResourceType, ResourceNode

if TYPE_CHECKING:
    from SimPyTest.engine import SimPySimulationEngine


def get_distance(engine: SimPySimulationEngine, r1: Resource, r2: ResourceNode) -> float:
    if engine.road_graph is not None:
        dist_meters = get_road_distance(engine.road_graph, r1.location, r2.location, engine.gis_config)
        dist = meters_to_miles(dist_meters)
        if dist != float("inf"):
            return dist
    euclidean_units = math.hypot(r1.location[0] - r2.location[0], r1.location[1] - r2.location[1])
    return euclidean_units * engine.scenario_config.distance_model.non_gis_distance_unit_miles


def generate_disaster_locations(engine: SimPySimulationEngine, num_locations: int) -> list[tuple[float, float]]:
    locations: list[tuple[float, float]] = []
    if engine.gis_config is not None and engine.gis_config.roads_gdf is not None and engine.road_graph is not None:
        sampled = engine.gis_config.roads_gdf.sample(n=num_locations, random_state=engine.seed)
        sampled = sampled.to_crs(CRS_UTM)
        centroids = sampled.geometry.centroid
        spatial_index = engine.gis_config.get_spatial_index()
        for centroid in centroids:
            x, y = centroid.xy
            locations.append(spatial_index.get_nearest_node(x[0], y[0]))
        return locations

    return [sample_clatsop_local_utm(engine.rng) for _ in range(num_locations)]


def initialize_world(engine: SimPySimulationEngine) -> None:
    if engine.gis_config is not None:
        engine.road_graph = engine.gis_config.load_road_network()
    depot, _dump_site = init_nodes(engine)
    spawn_resources(engine, depot)


def init_nodes(engine: SimPySimulationEngine) -> tuple[Depot, DumpSite]:
    depot_loc = default_non_gis_depot_location()
    dump_loc = default_non_gis_dump_site_location()

    if engine.gis_config is not None and engine.gis_config.depots:
        depot = engine.gis_config.depots[0]
        if "x" in depot and "y" in depot:
            depot_loc = (depot["x"], depot["y"])

    if engine.gis_config is not None and engine.gis_config.landfills:
        landfill = engine.gis_config.landfills[0]
        if "x" in landfill and "y" in landfill:
            dump_loc = (landfill["x"], landfill["y"])

    depot_node = Depot(engine, depot_loc)
    dump_site = DumpSite(engine, dump_loc)
    engine.resource_nodes.extend([depot_node, dump_site])
    return depot_node, dump_site


def spawn_resources(engine: SimPySimulationEngine, depot: Depot) -> None:
    spawn_plan = [
        (ResourceType.TRUCK, engine.scenario_config.resolve_resource_count(engine.rng, "truck")),
        (ResourceType.EXCAVATOR, engine.scenario_config.resolve_resource_count(engine.rng, "excavator")),
        (ResourceType.SNOWPLOW, engine.scenario_config.resolve_resource_count(engine.rng, "snowplow")),
        (ResourceType.ASSESSMENT_VEHICLE, engine.scenario_config.resolve_resource_count(engine.rng, "assessment_vehicle")),
    ]
    resource_id = 0
    for resource_type, count in spawn_plan:
        for _ in range(count):
            depot.transfer_resource(Resource(resource_id, resource_type, engine), True)
            resource_id += 1
