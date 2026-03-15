"""
Network statistics analysis script for ODOT road network.
Provides scale reference and region analysis for simulation setup.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
import geopandas as gpd
from shapely.geometry import LineString
import networkx as nx
from collections import defaultdict

# Constants from gis_utils
CRS_WGS = "EPSG:4326"
CRS_UTM = "EPSG:32610"


def load_roads(filepath, enabled_types=None):
    """Load roads from shapefile."""
    gdf = gpd.read_file(filepath).dropna(subset=["geometry"])
    if enabled_types:
        gdf = gdf[gdf["RTTYP"].isin(enabled_types)]
    return gdf


def build_road_graph(roads_gdf):
    """Build NetworkX graph from roads."""
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


def get_road_network_stats(roads_gdf):
    """Get basic network statistics."""
    roads_utm = roads_gdf.to_crs(CRS_UTM)
    G = build_road_graph(roads_utm)
    
    return {
        "num_segments": len(roads_gdf),
        "total_length_km": float(roads_utm.geometry.length.sum() / 1000),
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "road_types": dict(roads_gdf["RTTYP"].value_counts()),
    }


def get_graph_connected_components(G):
    """Get connected components."""
    return list(nx.connected_components(G))


def get_largest_connected_component(G):
    """Get largest connected component."""
    if G.number_of_nodes() == 0:
        return G
    largest_cc = max(nx.connected_components(G), key=len)
    return G.subgraph(largest_cc).copy()


def analyze_road_segments_by_type(roads_gdf) -> dict[str, Any]:
    """Analyze road segments by type and length."""
    roads_utm = roads_gdf.to_crs(CRS_UTM)
    
    stats_by_type = defaultdict(lambda: {"count": 0, "total_length_km": 0.0})
    
    for _, row in roads_utm.iterrows():
        road_type = row.get("RTTYP", "Unknown")
        length_km = row.geometry.length / 1000
        stats_by_type[road_type]["count"] += 1
        stats_by_type[road_type]["total_length_km"] += length_km
    
    return dict(stats_by_type)


def analyze_network_connectivity(G: nx.Graph) -> dict[str, Any]:
    """Analyze network connectivity and components."""
    components = get_graph_connected_components(G)
    largest_cc = get_largest_connected_component(G)
    
    component_sizes = [len(c) for c in components]
    
    return {
        "total_components": len(components),
        "largest_component_nodes": largest_cc.number_of_nodes(),
        "largest_component_edges": largest_cc.number_of_edges(),
        "largest_component_pct": (largest_cc.number_of_nodes() / G.number_of_nodes()) * 100,
        "component_sizes": sorted(component_sizes, reverse=True)[:10],  # Top 10
    }


def analyze_node_degree_distribution(G: nx.Graph) -> dict[str, Any]:
    """Analyze the degree distribution of nodes."""
    degrees = [G.degree(n) for n in G.nodes()]
    
    if not degrees:
        return {
            "min_degree": 0,
            "max_degree": 0,
            "avg_degree": 0.0,
            "median_degree": 0.0,
            "intersections_3plus": 0,
            "dead_ends": 0,
        }
    
    sorted_degrees = sorted(degrees)
    n = len(sorted_degrees)
    median = sorted_degrees[n // 2] if n % 2 == 1 else (sorted_degrees[n // 2 - 1] + sorted_degrees[n // 2]) / 2
    
    return {
        "min_degree": min(degrees),
        "max_degree": max(degrees),
        "avg_degree": sum(degrees) / len(degrees),
        "median_degree": median,
        "intersections_3plus": sum(1 for d in degrees if d >= 3),
        "dead_ends": sum(1 for d in degrees if d == 1),
    }


def create_synthetic_regions(roads_gdf, num_regions: int = 5) -> list[dict]:
    """
    Create synthetic regions based on road network density.
    In the future, this would use real ODOT district boundaries.
    """
    roads_utm = roads_gdf.to_crs(CRS_UTM)
    
    # Calculate bounding box
    bounds = roads_utm.total_bounds  # [minx, miny, maxx, maxy]
    
    # Divide into grid-based regions
    regions = []
    
    # For simplicity, divide into N horizontal strips
    y_range = bounds[3] - bounds[1]
    strip_height = y_range / num_regions
    
    for i in range(num_regions):
        y_min = bounds[1] + i * strip_height
        y_max = bounds[1] + (i + 1) * strip_height
        
        # Count roads in this region
        region_roads = roads_utm[
            (roads_utm.geometry.centroid.y >= y_min) &
            (roads_utm.geometry.centroid.y < y_max)
        ]
        
        total_length = region_roads.geometry.length.sum() / 1000
        
        regions.append({
            "id": f"region_{i+1}",
            "name": f"Region {i+1} (North to South)",
            "bounds": (bounds[0], y_min, bounds[2], y_max),
            "road_segments": len(region_roads),
            "total_length_km": total_length,
            "road_types": dict(region_roads["RTTYP"].value_counts()) if len(region_roads) > 0 else {},
        })
    
    return regions


def print_network_stats(filepath: str | Path, enabled_types: list[str] | None = None):
    """Print comprehensive network statistics."""
    print("=" * 80)
    print("ODOT ROAD NETWORK ANALYSIS")
    print("=" * 80)
    print(f"\nFile: {filepath}")
    print(f"Road types included: {enabled_types if enabled_types else 'ALL'}")
    print("\n" + "-" * 80)
    
    # Load roads
    print("\nLoading road network...")
    roads_gdf = load_roads(filepath, enabled_types)
    
    # Basic stats
    print("\n" + "=" * 40)
    print("BASIC STATISTICS")
    print("=" * 40)
    print(f"Total road segments: {len(roads_gdf):,}")
    
    # Build graph
    print("\nBuilding road graph...")
    roads_utm = roads_gdf.to_crs(CRS_UTM)
    G = build_road_graph(roads_utm)
    
    # Network stats
    stats = get_road_network_stats(roads_gdf)
    
    print(f"\nTotal road length: {stats['total_length_km']:,.2f} km ({stats['total_length_km']/1.60934:,.2f} miles)")
    print(f"Number of nodes (intersections): {stats['num_nodes']:,}")
    print(f"Number of edges (road segments): {stats['num_edges']:,}")
    
    # Road type breakdown
    print("\n" + "=" * 40)
    print("ROAD TYPE BREAKDOWN")
    print("=" * 40)
    type_stats = analyze_road_segments_by_type(roads_gdf)
    
    road_type_names = {
        "I": "Interstate",
        "U": "US Highway", 
        "S": "State Highway",
        "C": "County Road",
        "M": "Municipal Road",
        "O": "Other",
    }
    
    for road_type, data in sorted(type_stats.items(), key=lambda x: x[1]["total_length_km"], reverse=True):
        name = road_type_names.get(road_type, road_type)
        print(f"\n{name} ({road_type}):")
        print(f"  Segments: {data['count']:,}")
        print(f"  Total length: {data['total_length_km']:,.2f} km ({data['total_length_km']/1.60934:,.2f} miles)")
    
    # Connectivity analysis
    print("\n" + "=" * 40)
    print("CONNECTIVITY ANALYSIS")
    print("=" * 40)
    conn_stats = analyze_network_connectivity(G)
    print(f"Connected components: {conn_stats['total_components']}")
    print(f"Largest component: {conn_stats['largest_component_nodes']:,} nodes ({conn_stats['largest_component_pct']:.1f}%)")
    print(f"Top 10 component sizes: {conn_stats['component_sizes']}")
    
    # Node degree distribution
    print("\n" + "=" * 40)
    print("NODE DEGREE DISTRIBUTION")
    print("=" * 40)
    degree_stats = analyze_node_degree_distribution(G)
    print(f"Average degree: {degree_stats['avg_degree']:.2f}")
    print(f"Median degree: {degree_stats['median_degree']:.1f}")
    print(f"Min/Max degree: {degree_stats['min_degree']}/{degree_stats['max_degree']}")
    print(f"Intersections (degree >= 3): {degree_stats['intersections_3plus']:,}")
    print(f"Dead ends (degree = 1): {degree_stats['dead_ends']:,}")
    
    # Synthetic regions
    print("\n" + "=" * 40)
    print("SYNTHETIC REGIONS (North to South)")
    print("=" * 40)
    print("Note: These are synthetic regions for simulation.")
    print("Replace with real ODOT district boundaries when available.\n")
    
    regions = create_synthetic_regions(roads_gdf, num_regions=5)
    for region in regions:
        print(f"\n{region['name']}:")
        print(f"  Road segments: {region['road_segments']:,}")
        print(f"  Total length: {region['total_length_km']:,.2f} km")
        if region['road_types']:
            print(f"  Road types: {dict(list(region['road_types'].items())[:3])}...")  # Show top 3
    
    # Scale reference
    print("\n" + "=" * 40)
    print("SCALE REFERENCE FOR SIMULATION")
    print("=" * 40)
    total_miles = stats['total_length_km'] / 1.60934
    
    print(f"\nNetwork scale: {total_miles:,.0f} miles of road")
    print(f"If simulating 5 years of operations:")
    print(f"  - Average events per year: ~{int(total_miles / 100)}-{int(total_miles / 50)} (1 per 50-100 miles)")
    print(f"  - If each event takes 2-8 hours to resolve...")
    print(f"  - Total event-hours per year: ~{int(total_miles / 100 * 5)}-{int(total_miles / 50 * 8)} hours")
    
    print("\n" + "=" * 80)
    print("END OF ANALYSIS")
    print("=" * 80)
    
    return {
        "total_segments": len(roads_gdf),
        "total_miles": total_miles,
        "total_nodes": stats['num_nodes'],
        "regions": regions,
        "road_types": type_stats,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze ODOT road network for simulation scaling")
    parser.add_argument("--roads", type=str, required=True, help="Path to road shapefile")
    parser.add_argument("--types", nargs="+", default=None, 
                       help="Road types to include (e.g., I U S). Default: all")
    
    args = parser.parse_args()
    
    stats = print_network_stats(args.roads, args.types)
