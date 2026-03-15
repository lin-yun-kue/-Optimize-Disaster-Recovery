from __future__ import annotations

from typing import Protocol

from pyproj import Transformer

from .gis_utils import CRS_UTM, CRS_WGS

# Derived on 2026-03-12 from maps/tl_2024_41007_roads/tl_2024_41007_roads.shp.
CLATSOP_ROAD_BOUNDS_UTM = (
    421772.53,
    5069105.82,
    472174.56,
    5120526.74,
)
CLATSOP_ROAD_SPAN_METERS = (
    CLATSOP_ROAD_BOUNDS_UTM[2] - CLATSOP_ROAD_BOUNDS_UTM[0],
    CLATSOP_ROAD_BOUNDS_UTM[3] - CLATSOP_ROAD_BOUNDS_UTM[1],
)
CLATSOP_LOCAL_COORD_MAX = int(max(CLATSOP_ROAD_SPAN_METERS))

DEFAULT_DEPOT_LON_LAT = (-123.92616052570274, 46.16262226957932)
DEFAULT_DUMP_SITE_LON_LAT = (-123.70105856996436, 45.91375387223106)

_WGS_TO_UTM = Transformer.from_crs(CRS_WGS, CRS_UTM, always_xy=True)


class _CoordinateSampler(Protocol):
    def uniform(self, a: float, b: float) -> float: ...


def lon_lat_to_clatsop_local_utm(lon: float, lat: float) -> tuple[float, float]:
    x, y = _WGS_TO_UTM.transform(lon, lat)
    return (x - CLATSOP_ROAD_BOUNDS_UTM[0], y - CLATSOP_ROAD_BOUNDS_UTM[1])


def default_non_gis_depot_location() -> tuple[float, float]:
    return lon_lat_to_clatsop_local_utm(*DEFAULT_DEPOT_LON_LAT)


def default_non_gis_dump_site_location() -> tuple[float, float]:
    return lon_lat_to_clatsop_local_utm(*DEFAULT_DUMP_SITE_LON_LAT)


def sample_clatsop_local_utm(rng: _CoordinateSampler) -> tuple[float, float]:
    return (
        rng.uniform(0.0, CLATSOP_ROAD_SPAN_METERS[0]),
        rng.uniform(0.0, CLATSOP_ROAD_SPAN_METERS[1]),
    )
