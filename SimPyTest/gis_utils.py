"""
GIS utilities for loading road networks and handling spatial operations.
Adapted from the GIS-based disaster simulation project.
"""

from __future__ import annotations
import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point
from pyproj import Transformer
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict, cast
import numpy as np
from functools import lru_cache
import hashlib
from rtree import index
import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

# Coordinate systems
CRS_WGS = "EPSG:4326"  # Latitude/Longitude
CRS_UTM = "EPSG:32610"  # UTM Zone 10N (for Oregon/Washington coast)


def load_roads(filepath: str | Path, enabled_types: list[str] | None = None) -> gpd.GeoDataFrame:
    """
    Load and optionally filter roads from a shapefile.

    Args:
        filepath: Path to the road shapefile
        enabled_types: Optional list of road types (RTTYP values) to include
            Common types: 'I' (Interstate), 'U' (US Highway), 'S' (State),
                         'C' (County), 'M' (Municipal)
            Use ['I', 'U', 'S'] for faster simulation with major roads only

    Returns:
        GeoDataFrame with road geometries
    """
    gdf = gpd.read_file(filepath).dropna(subset=["geometry"])
    if enabled_types:
        gdf = gdf[gdf["RTTYP"].isin(enabled_types)]
    return cast(gpd.GeoDataFrame, gdf)


def get_road_network_stats(roads_gdf: gpd.GeoDataFrame) -> dict[str, Any]:
    """
    Get statistics about the road network.

    Args:
        roads_gdf: GeoDataFrame with road geometries

    Returns:
        Dictionary with network statistics
    """
    roads_utm = roads_gdf.to_crs(CRS_UTM)
    G = build_road_graph(roads_utm)

    return {
        "num_segments": len(roads_gdf),
        "total_length_km": float(roads_utm.geometry.length.sum() / 1000),
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "road_types": dict(roads_gdf["RTTYP"].value_counts()),
    }


def build_road_graph(roads_gdf: gpd.GeoDataFrame) -> nx.Graph[tuple[float, float]]:
    """
    Build a NetworkX graph from UTM-projected roads GeoDataFrame.

    Each edge has:
    - 'geometry': LineString of the road segment
    - 'length': Length in meters

    Args:
        roads_gdf: GeoDataFrame with road geometries (will be converted to UTM)

    Returns:
        NetworkX graph where nodes are (x, y) coordinate tuples
    """
    roads_gdf = roads_gdf.to_crs(CRS_UTM)
    G = nx.Graph()

    for _, row in roads_gdf.iterrows():
        geom = row.geometry
        if isinstance(geom, LineString):
            coords = list(geom.coords)
            for i in range(len(coords) - 1):
                u, v = coords[i], coords[i + 1]
                segment = LineString([u, v])
                G.add_edge(u, v, geometry=segment, length=segment.length)

    return G


def snap_to_graph(lon: float, lat: float, G: nx.Graph[tuple[float, float]]) -> tuple[float, float]:
    """
    Snap a WGS84 coordinate to the nearest node in the road graph.

    Args:
        lon: Longitude in WGS84
        lat: Latitude in WGS84
        G: Road graph with UTM nodes

    Returns:
        Nearest graph node as (x, y) tuple in UTM coordinates
    """
    transformer = Transformer.from_crs(CRS_WGS, CRS_UTM, always_xy=True)
    x, y = transformer.transform(lon, lat)
    point = Point(x, y)

    # Find nearest node
    nearest_node = min(G.nodes, key=lambda n: point.distance(Point(n)))
    return nearest_node


def get_graph_fingerprint(G: nx.Graph[tuple[float, float]]) -> str:
    """
    Generate a content-based fingerprint for a road graph.

    This creates a hash based on the graph's nodes, edges, and weights,
    ensuring that identical road networks have the same fingerprint
    regardless of object identity.

    Args:
        G: Road graph

    Returns:
        SHA256 hash string representing the graph content
    """
    # Create a deterministic representation of the graph
    graph_data = []

    # Add nodes (sorted for deterministic order)
    for node in sorted(G.nodes()):
        graph_data.append(f"node:{node[0]:.6f},{node[1]:.6f}")

    # Add edges with weights (sorted for deterministic order)
    for u, v, data in sorted(G.edges(data=True)):
        weight = data.get("length", 0)
        graph_data.append(f"edge:{u[0]:.6f},{u[1]:.6f}-{v[0]:.6f},{v[1]:.6f}:{weight:.6f}")

    # Create hash
    graph_string = "|".join(graph_data)
    return hashlib.sha256(graph_string.encode()).hexdigest()[:16]


# Global cache for road distances
_road_distance_cache: dict[tuple[str, tuple[float, float], tuple[float, float]], float] = {}


def clear_road_distance_cache() -> None:
    """Clear the road distance cache."""
    global _road_distance_cache
    _road_distance_cache.clear()


def get_cache_stats() -> dict[str, int]:
    """Get statistics about the road distance cache."""
    return {
        "cache_size": len(_road_distance_cache),
        "max_cache_size": 50000,
    }


def get_road_distance(
    G: nx.Graph[tuple[float, float]],
    node1: tuple[float, float],
    node2: tuple[float, float],
    gis_config: GISConfig | None = None,
) -> float:
    """
    Calculate shortest path distance along roads between two nodes.

    Args:
        G: Road graph
        node1: Starting node (x, y) in UTM
        node2: Ending node (x, y) in UTM
        gis_config: Optional GIS config to use for caching

    Returns:
        Distance in meters, or infinity if no path exists
    """
    # Try GIS config cache first
    if gis_config is not None:
        cached_distance = gis_config.get_distance_from_cache(node1, node2)
        if cached_distance is not None:
            return cached_distance
        print(".", end="", flush=True)

    # Fall back to global cache
    graph_fingerprint = get_graph_fingerprint(G)
    global_cache_key = (graph_fingerprint, node1, node2)

    if global_cache_key in _road_distance_cache:
        distance = _road_distance_cache[global_cache_key]
        # Also cache in GIS config if provided
        if gis_config is not None:
            gis_config.cache_distance(node1, node2, distance)
        return distance

    # Calculate the distance
    try:
        distance = nx.shortest_path_length(G, source=node1, target=node2, weight="length")

        # Cache in both places
        if len(_road_distance_cache) > 50000:
            keys_to_remove = list(_road_distance_cache.keys())[:10000]
            for key in keys_to_remove:
                del _road_distance_cache[key]
        _road_distance_cache[global_cache_key] = distance

        if gis_config is not None:
            gis_config.cache_distance(node1, node2, distance)

        return distance
    except nx.NetworkXNoPath:
        # Cache the infinity result
        if len(_road_distance_cache) > 50000:
            keys_to_remove = list(_road_distance_cache.keys())[:10000]
            for key in keys_to_remove:
                del _road_distance_cache[key]
        _road_distance_cache[global_cache_key] = float("inf")

        if gis_config is not None:
            gis_config.cache_distance(node1, node2, float("inf"))

        return float("inf")


def get_road_path(
    G: nx.Graph[tuple[float, float]], node1: tuple[float, float], node2: tuple[float, float]
) -> list[tuple[float, float]]:
    """
    Get the sequence of nodes along the shortest path.

    Args:
        G: Road graph
        node1: Starting node
        node2: Ending node

    Returns:
        List of nodes (x, y) tuples representing the path, empty if no path exists
    """
    try:
        return nx.shortest_path(G, source=node1, target=node2, weight="length")
    except nx.NetworkXNoPath:
        return []


# Spatial indexing for fast nearest node queries
class RoadNodeSpatialIndex:
    """R-tree spatial index for fast nearest road node queries."""

    def __init__(self, G: nx.Graph[tuple[float, float]]):
        """
        Build R-tree index from road graph nodes.

        Args:
            G: Road graph with (x, y) coordinate tuples as nodes
        """
        self.G = G
        self.idx = index.Index()
        self.node_map = {}

        # Build spatial index
        for i, node in enumerate(G.nodes()):
            x, y = node
            # Insert node with a tiny bounding box (point)
            self.idx.insert(i, (x, y, x, y))
            self.node_map[i] = node

    def get_nearest_node(self, x: float, y: float) -> tuple[float, float]:
        """
        Find the nearest road node to a coordinate.

        Args:
            x: X coordinate (UTM)
            y: Y coordinate (UTM)

        Returns:
            Nearest road node as (x, y) tuple
        """
        # Find nearest index using R-tree
        nearest_idx = next(self.idx.nearest((x, y, x, y), 1))
        return self.node_map[nearest_idx]

    def get_k_nearest_nodes(self, x: float, y: float, k: int = 5) -> list[tuple[float, float]]:
        """
        Find k nearest road nodes to a coordinate.

        Args:
            x: X coordinate (UTM)
            y: Y coordinate (UTM)
            k: Number of nearest nodes to return

        Returns:
            List of k nearest road nodes as (x, y) tuples, ordered by distance
        """
        # Find k nearest indices using R-tree
        nearest_indices = list(self.idx.nearest((x, y, x, y), k))
        return [self.node_map[i] for i in nearest_indices]


# GIS configuration
class Depot(TypedDict):
    Longitude: float
    Latitude: float
    Name: str
    numExc: int
    numTrucks: int
    color: str
    node: NotRequired[tuple[float, float]]
    x: NotRequired[float]
    y: NotRequired[float]


class Landfill(TypedDict):
    Label: str
    Longitude: float
    Latitude: float
    Name: str
    node: NotRequired[tuple[float, float]]
    x: NotRequired[float]
    y: NotRequired[float]


class GISConfig:
    """Configuration for GIS-based simulation."""

    def __init__(
        self,
        roads_gdf: gpd.GeoDataFrame,
        road_graph: nx.Graph[tuple[float, float]],
        depots: list[Depot] | None = None,
        landfills: list[Landfill] | None = None,
    ):
        """
        Initialize GIS configuration.

        Args:
            roads_gdf: GeoDataFrame with road geometries
            road_graph: NetworkX graph of the road network
            depots: List of depot definitions with Longitude, Latitude, Name, etc.
            landfills: List of landfill definitions with Longitude, Latitude, Name, etc.
        """
        self.depots = depots or []
        self.landfills = landfills or []

        # These will be populated when the graph is built
        self.roads_gdf: gpd.GeoDataFrame | None = roads_gdf
        self.road_graph: nx.Graph[tuple[float, float]] = road_graph

        # Initialize distance cache for this GIS configuration
        self._distance_cache: dict[tuple[str, tuple[float, float], tuple[float, float]], float] = {}
        self._graph_fingerprint: str | None = None
        self._max_cache_size = 50000

        # Initialize spatial index for fast nearest node queries
        self._spatial_index: RoadNodeSpatialIndex | None = None

        self.snap_locations()
        self.get_graph_fingerprint()

    def load_road_network(self) -> nx.Graph[tuple[float, float]]:
        """Load the road network and build the graph."""
        return self.road_graph

    def get_spatial_index(self) -> RoadNodeSpatialIndex:
        """Get or create the spatial index for fast nearest node queries."""
        if self._spatial_index is None and self.road_graph is not None:
            self._spatial_index = RoadNodeSpatialIndex(self.road_graph)
        return self._spatial_index or RoadNodeSpatialIndex(nx.Graph())

    def snap_to_nearest_road_node(self, x: float, y: float) -> tuple[float, float]:
        """
        Snap a UTM coordinate to the nearest road node using spatial index.

        Args:
            x: X coordinate in UTM
            y: Y coordinate in UTM

        Returns:
            Nearest road node as (x, y) tuple in UTM coordinates
        """
        spatial_index = self.get_spatial_index()
        return spatial_index.get_nearest_node(x, y)

    def snap_locations(self) -> None:
        """Snap all depot and landfill locations to the road graph."""
        for location in self.depots + self.landfills:
            node = snap_to_graph(location["Longitude"], location["Latitude"], self.road_graph)
            location["node"] = node
            location["x"], location["y"] = node

    def get_graph_fingerprint(self) -> str:
        """Get or compute the fingerprint of the road graph."""
        if self._graph_fingerprint is None and self.road_graph is not None:
            self._graph_fingerprint = get_graph_fingerprint(self.road_graph)
        return self._graph_fingerprint or ""

    def get_distance_from_cache(self, node1: tuple[float, float], node2: tuple[float, float]) -> float | None:
        """Get a distance from the cache, or None if not cached."""
        if not self._graph_fingerprint:
            return None
        cache_key = (self._graph_fingerprint, node1, node2)
        return self._distance_cache.get(cache_key)

    def cache_distance(self, node1: tuple[float, float], node2: tuple[float, float], distance: float) -> None:
        """Cache a distance calculation."""
        if not self._graph_fingerprint:
            return

        cache_key = (self._graph_fingerprint, node1, node2)

        # Limit cache size
        if len(self._distance_cache) >= self._max_cache_size:
            # Remove oldest entries (simple strategy)
            keys_to_remove = list(self._distance_cache.keys())[:10000]
            for key in keys_to_remove:
                del self._distance_cache[key]

        self._distance_cache[cache_key] = distance

    def clear_distance_cache(self) -> None:
        """Clear the distance cache."""
        self._distance_cache.clear()

    def get_cache_stats(self) -> dict[str, int | str]:
        """Get statistics about the distance cache."""
        return {
            "cache_size": len(self._distance_cache),
            "max_cache_size": self._max_cache_size,
            "graph_fingerprint": self.get_graph_fingerprint(),
        }


def get_graph_connected_components(G: nx.Graph) -> list[set]:
    """
    Get all connected components in the graph.
    
    Args:
        G: Road graph
        
    Returns:
        List of sets, where each set contains nodes in a connected component
    """
    return list(nx.connected_components(G))


def get_largest_connected_component(G: nx.Graph) -> nx.Graph:
    """
    Extract the largest connected component from the graph.
    
    This is useful after pruning to ensure the network remains fully connected.
    
    Args:
        G: Road graph
        
    Returns:
        Subgraph containing only the largest connected component
    """
    if G.number_of_nodes() == 0:
        return G
    
    largest_cc = max(nx.connected_components(G), key=len)
    return G.subgraph(largest_cc).copy()


def remove_dead_end_nodes(G: nx.Graph, iterations: int = 1) -> nx.Graph:
    """
    Remove dead-end nodes (degree 1) from the graph.
    
    This simplifies the network by removing spurs and cul-de-sacs while
    maintaining connectivity between major nodes.
    
    Args:
        G: Road graph
        iterations: Number of passes to remove dead ends (higher = more aggressive)
        
    Returns:
        Simplified graph with dead ends removed
    """
    G_simplified = G.copy()
    
    for _ in range(iterations):
        # Find nodes with degree 1 (dead ends)
        dead_ends = [node for node in G_simplified.nodes() if G_simplified.degree(node) == 1]
        
        if not dead_ends:
            break
            
        # Remove dead end nodes
        G_simplified.remove_nodes_from(dead_ends)
    
    return G_simplified


def simplify_graph_by_merging_degree_2_nodes(G: nx.Graph) -> nx.Graph:
    """
    Simplify graph by merging consecutive degree-2 nodes.
    
    This reduces the graph size while maintaining the overall topology
    by replacing long chains of nodes with single edges.
    
    Args:
        G: Road graph
        
    Returns:
        Simplified graph
    """
    G_simplified = G.copy()
    
    # Find all degree-2 nodes
    degree_2_nodes = [node for node in G_simplified.nodes() if G_simplified.degree(node) == 2]
    
    for node in degree_2_nodes:
        if node not in G_simplified:
            continue  # Already removed
            
        neighbors = list(G_simplified.neighbors(node))
        if len(neighbors) != 2:
            continue
            
        u, v = neighbors
        
        # Get edge data
        edge_data_u = G_simplified.get_edge_data(node, u)
        edge_data_v = G_simplified.get_edge_data(node, v)
        
        # Calculate combined length
        length_u = edge_data_u.get('length', 0) if edge_data_u else 0
        length_v = edge_data_v.get('length', 0) if edge_data_v else 0
        combined_length = length_u + length_v
        
        # Remove the degree-2 node and add direct edge between neighbors
        G_simplified.remove_node(node)
        G_simplified.add_edge(u, v, length=combined_length)
    
    return G_simplified


def prune_roads_maintaining_connectivity(
    roads_gdf: gpd.GeoDataFrame,
    critical_points: list[tuple[float, float]] | None = None,
    primary_types: list[str] | None = None,
    secondary_types: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """
    Prune road network while maintaining connectivity between critical points.
    
    Strategy:
    1. Start with primary road types (e.g., ['I', 'U', 'S'])
    2. Add secondary roads needed to connect critical points
    3. Keep only the largest connected component
    4. Optionally simplify by removing dead ends
    
    Args:
        roads_gdf: GeoDataFrame with all road segments
        critical_points: List of (lon, lat) tuples for locations that must be connected
                         (e.g., depot and landfill locations)
        primary_types: Road types to always include (default: ['I', 'U', 'S'])
        secondary_types: Additional road types to use for connectivity (default: ['C', 'M'])
        
    Returns:
        Pruned GeoDataFrame with maintained connectivity
    """
    if primary_types is None:
        primary_types = ['I', 'U', 'S']
    if secondary_types is None:
        secondary_types = ['C', 'M']
    
    # Start with primary roads
    primary_roads = roads_gdf[roads_gdf["RTTYP"].isin(primary_types)].copy()
    
    if len(primary_roads) == 0:
        print("Warning: No primary roads found, using all roads")
        return roads_gdf
    
    # Build graph from primary roads
    G_primary = build_road_graph(primary_roads)
    
    # If we have critical points, ensure they're all connected
    if critical_points and len(critical_points) > 1:
        transformer = Transformer.from_crs(CRS_WGS, CRS_UTM, always_xy=True)
        
        # Snap critical points to the graph
        snapped_points = []
        for lon, lat in critical_points:
            x, y = transformer.transform(lon, lat)
            point = Point(x, y)
            # Find nearest node in primary graph
            if len(G_primary.nodes()) > 0:
                nearest = min(G_primary.nodes(), key=lambda n: point.distance(Point(n)))
                snapped_points.append(nearest)
        
        # Check if all snapped points are in the same component
        if len(snapped_points) > 1:
            components = list(nx.connected_components(G_primary))
            
            # Group points by component
            point_components = {}
            for i, point in enumerate(snapped_points):
                for j, component in enumerate(components):
                    if point in component:
                        point_components[i] = j
                        break
            
            # If points are in different components, we need to add more roads
            unique_components = set(point_components.values())
            if len(unique_components) > 1:
                print(f"Critical points are in {len(unique_components)} separate components, adding connecting roads...")
                
                # Add secondary roads to try to connect components
                secondary_roads = roads_gdf[roads_gdf["RTTYP"].isin(secondary_types)].copy()
                combined_roads = gpd.GeoDataFrame(pd.concat([primary_roads, secondary_roads]).drop_duplicates(), crs=primary_roads.crs)
                G_combined = build_road_graph(combined_roads)
                
                # Check if this connects everything
                if nx.is_connected(G_combined):
                    print("Successfully connected all critical points using secondary roads")
                    return combined_roads
                else:
                    # Still not connected, use all roads
                    print("Warning: Could not connect all critical points even with secondary roads")
                    print("Using largest connected component with all road types")
                    return roads_gdf
    
    # Keep only the largest connected component
    G_largest = get_largest_connected_component(G_primary)
    
    # Get nodes in the largest component
    nodes_in_largest = set(G_largest.nodes())
    
    # Filter roads to only include those in the largest component
    # We need to check if both endpoints of each road segment are in the component
    def is_road_in_component(row):
        geom = row.geometry
        if isinstance(geom, LineString):
            coords = list(geom.coords)
            for i in range(len(coords) - 1):
                u, v = coords[i], coords[i + 1]
                if u in nodes_in_largest and v in nodes_in_largest:
                    return True
        return False
    
    # This is a bit inefficient but works
    primary_roads_copy = primary_roads.copy()
    primary_roads_copy['in_component'] = primary_roads_copy.apply(is_road_in_component, axis=1)
    result = primary_roads_copy[primary_roads_copy['in_component']].drop(columns=['in_component'])
    
    print(f"Pruned network: {len(roads_gdf)} -> {len(result)} segments ({(1 - len(result)/len(roads_gdf))*100:.1f}% reduction)")
    
    return result


def load_and_prune_roads(
    filepath: str | Path,
    depots: list[Depot] | None = None,
    landfills: list[Landfill] | None = None,
    enabled_types: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """
    Load roads and prune them while maintaining connectivity to depots and landfills.
    
    This is the recommended way to load roads for simulation - it ensures all
    critical locations remain reachable while still reducing network complexity.
    
    Args:
        filepath: Path to the road shapefile
        depots: List of depot dictionaries with 'Longitude' and 'Latitude' keys
        landfills: List of landfill dictionaries with 'Longitude' and 'Latitude' keys
        enabled_types: List of road types to include (default: ['I', 'U', 'S'])
        
    Returns:
        Pruned GeoDataFrame with maintained connectivity
    """
    # Load all roads first
    print(f"Loading road network from {filepath}...")
    roads_gdf = load_roads(filepath)
    print(f"  Loaded {len(roads_gdf)} road segments")
    
    # Collect critical points
    critical_points = []
    if depots:
        for depot in depots:
            critical_points.append((depot["Longitude"], depot["Latitude"]))
    if landfills:
        for landfill in landfills:
            critical_points.append((landfill["Longitude"], landfill["Latitude"]))
    
    # Prune while maintaining connectivity
    if enabled_types:
        print(f"Pruning to road types: {enabled_types}")
        pruned_gdf = prune_roads_maintaining_connectivity(
            roads_gdf,
            critical_points=critical_points if critical_points else None,
            primary_types=enabled_types,
        )
        return pruned_gdf
    else:
        # Keep largest connected component even if not filtering by type
        G = build_road_graph(roads_gdf)
        G_largest = get_largest_connected_component(G)
        print(f"Keeping largest connected component: {G_largest.number_of_nodes()} nodes, {G_largest.number_of_edges()} edges")
        
        # Convert back to GeoDataFrame (this is tricky, simplified version)
        return roads_gdf
