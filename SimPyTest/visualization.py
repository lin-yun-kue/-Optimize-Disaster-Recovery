from __future__ import annotations

import math
from dataclasses import dataclass, field
from math import cos, sin
from time import perf_counter
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.figure import Figure
from matplotlib.text import Text
from SimPyTest.gis_utils import CLATSOP_ROAD_BOUNDS_UTM
from SimPyTest.simulation import WildfireDebris

if TYPE_CHECKING:
    from SimPyTest.engine import SimPySimulationEngine
    from SimPyTest.simulation import Landslide
else:
    from SimPyTest.simulation import Landslide


@dataclass(frozen=True)
class NodeSnapshot:
    id: int
    location: tuple[float, float]
    color: str
    label: str


@dataclass(frozen=True)
class DisasterSnapshot:
    id: int
    location: tuple[float, float]
    color: str
    marker: str
    label: str
    hatch: str
    percent_remaining: float
    size: float


@dataclass(frozen=True)
class ResourceSnapshot:
    id: int
    # interpolated display position (already computed by snapshot())
    display_location: tuple[float, float]
    marker: str
    color: str
    label: str


@dataclass(frozen=True)
class EngineSnapshot:
    policy_name: str
    seed: int
    env_now: float
    nodes: list[NodeSnapshot]
    disasters: list[DisasterSnapshot]
    resources: list[ResourceSnapshot]


def _interp_resource_location(resource, env_now: float) -> tuple[float, float]:
    """Return the interpolated display position for a moving resource."""
    frac = resource.id * (1 + math.sqrt(5)) / 2
    offset_x = cos(frac) * 10
    offset_y = sin(frac) * 10

    loc1 = (resource.location[0] + offset_x, resource.location[1] + offset_y)
    loc2 = (resource.prev_location[0] + offset_x, resource.prev_location[1] + offset_y)

    t = 1.0 if resource.move_time == 0 else (env_now - resource.move_start_time) / resource.move_time
    t = max(0.0, min(1.0, t))
    return (
        loc1[0] * t + loc2[0] * (1 - t),
        loc1[1] * t + loc2[1] * (1 - t),
    )


def _update_text_pool(
    ax: Axes,
    pool: list[Text],
    positions: list[tuple[float, float]],
    texts: list[str],
    *,
    offset: tuple[float, float] = (0.0, 0.0),
    fontsize: int = 8,
    ha: str = "left",
    va: str = "bottom",
    zorder: int = 5,
    colors: list[str] | None = None,
) -> None:
    """
    Reuse Text artists from *pool*, growing or hiding as needed.
    Mutates *pool* in-place.
    """
    n = len(positions)
    for i in range(max(n, len(pool))):
        if i < n:
            x = positions[i][0] + offset[0]
            y = positions[i][1] + offset[1]
            if i < len(pool):
                t = pool[i]
                t.set_position((x, y))
                t.set_text(texts[i])
                t.set_visible(True)
                if colors:
                    t.set_color(colors[i])
            else:
                kw: dict[str, Any] = dict(fontsize=fontsize, ha=ha, va=va, zorder=zorder)
                if colors:
                    kw["color"] = colors[i]
                pool.append(ax.text(x, y, texts[i], **kw))
        elif i < len(pool):
            pool[i].set_visible(False)


@dataclass
class _MarkerGroup:
    """Owns a single scatter PathCollection for one marker symbol."""

    scatter: PathCollection
    # parallel arrays — rebuilt each frame from the current data
    positions: list[tuple[float, float]] = field(default_factory=list)
    colors: list[str] = field(default_factory=list)
    sizes: list[float] = field(default_factory=list)

    def push(self, pos: tuple[float, float], color: str, size: float) -> None:
        self.positions.append(pos)
        self.colors.append(color)
        self.sizes.append(size)

    def flush(self) -> None:
        """Apply accumulated data to the scatter artist, then clear buffers."""
        if self.positions:
            self.scatter.set_offsets(np.array(self.positions))
            self.scatter.set_facecolor(self.colors)
            self.scatter.set_sizes(self.sizes)
        else:
            self.scatter.set_offsets(np.zeros((0, 2)))
            self.scatter.set_facecolor([])
            self.scatter.set_sizes([])
        self.positions.clear()
        self.colors.clear()
        self.sizes.clear()


def _get_or_create_marker_group(
    groups: dict[str, _MarkerGroup],
    ax: Axes,
    marker: str,
    zorder: int,
    base_size: float,
) -> _MarkerGroup:
    if marker not in groups:
        sc = ax.scatter([], [], marker=marker, facecolors=[], s=base_size, zorder=zorder, edgecolors="black", linewidths=0.5)
        groups[marker] = _MarkerGroup(scatter=sc)
    return groups[marker]


class EngineVisualizer:
    SHAPEFILE_PATH: str = "/Users/aidan/Documents/School/AI/Capstone/maps/" "tl_2024_41007_roads/tl_2024_41007_roads.shp"

    def __init__(self, engine: SimPySimulationEngine) -> None:
        self.engine = engine

        # figure / axes
        self.fig: Figure | None = None
        self.axs: tuple[Axes, Axes, Axes] | None = None  # (map, dirt, info)

        # --- static road geometry (loaded once) ---
        self._road_collection: LineCollection | None = None

        # --- node artists ---
        self._node_scatter: PathCollection | None = None
        self._node_label_pool: list[Text] = []

        # --- disaster artists  (one scatter per marker type) ---
        self._disaster_groups: dict[str, _MarkerGroup] = {}
        self._disaster_label_pool: list[Text] = []

        # --- resource artists  (one scatter per marker type) ---
        self._resource_groups: dict[str, _MarkerGroup] = {}
        self._resource_label_pool: list[Text] = []

        # --- dirt / history chart ---
        self._time_points: list[float] = []
        self._disaster_size: dict[int, list[float]] = {}  # id -> per-tick dirt
        self._disaster_hatch: dict[int, str] = {}  # id -> hatch
        self._disaster_type_label: dict[int, str] = {}  # id -> short label
        self._dirt_drawn_ids: tuple[int, ...] = ()
        self._dirt_drawn_len: int = 0

        # --- info panel ---
        self._info_cache: str = ""
        self._initial_resource_counts: dict[str, int] | None = None

        # --- fps ---
        self._last_wall_time: float | None = None
        self._fps: float = 0.0
        self._frame: int = 0

    def _load_roads(self, ax: Axes) -> None:
        """Load shapefile into a LineCollection and add it to ax (once)."""
        try:
            gdf = gpd.read_file(self.SHAPEFILE_PATH)
            if gdf.crs and str(gdf.crs) != "EPSG:32610":
                gdf = gdf.to_crs("EPSG:32610")
            segs = []
            for geom in gdf.geometry:
                if geom is None:
                    continue
                lines = [geom] if geom.geom_type == "LineString" else list(geom.geoms)
                for line in lines:
                    coords = list(line.coords)
                    segs.extend(zip(coords[:-1], coords[1:]))
            if segs:
                self._road_collection = LineCollection(segs, colors="#b0b0b0", linewidths=0.3, alpha=0.5, zorder=1)
                ax.add_collection(self._road_collection)
        except Exception as exc:
            print(f"Warning: could not load shapefile: {exc}")

    def _init_artists(self) -> None:
        """
        Create the figure, axes, and every reusable artist.
        Called once, lazily, from update().
        """
        fig = plt.figure(figsize=(16, 10), constrained_layout=True)
        grid = fig.add_gridspec(2, 2, width_ratios=[2.4, 1.2], height_ratios=[3.0, 1.2])
        ax_map = fig.add_subplot(grid[0, 0])
        ax_dirt = fig.add_subplot(grid[1, 0])
        ax_info = fig.add_subplot(grid[:, 1])

        ax_map.set_aspect("equal")
        ax_map.set_xlabel("X")
        ax_map.set_ylabel("Y")
        ax_map.grid(True, alpha=0.2)
        min_x, min_y, max_x, max_y = CLATSOP_ROAD_BOUNDS_UTM
        ax_map.set_xlim(min_x, max_x)
        ax_map.set_ylim(min_y, max_y)

        self._load_roads(ax_map)

        # Node scatter — nodes don't change marker type, so a single scatter works.
        # Use facecolors= (not c=) so the artist is in direct-color mode from the start,
        # which means set_facecolors() with named strings will always work.
        self._node_scatter = ax_map.scatter([], [], facecolors=[], s=144, zorder=3, edgecolors="black", linewidths=0.5)

        ax_dirt.set_xlabel("Time")
        ax_dirt.set_ylabel("Dirt")
        ax_dirt.grid(True)

        ax_info.set_axis_off()
        ax_info.set_title("Debug State", loc="left")

        self.fig = fig
        self.axs = (ax_map, ax_dirt, ax_info)
        plt.ion()

    def snapshot(self) -> EngineSnapshot:
        env_now = self.engine.env.now

        nodes = [
            NodeSnapshot(
                id=node.id,
                location=node.location,
                color=node.render["color"],
                label=node.render["label"],
            )
            for node in self.engine.resource_nodes + [self.engine.idle_resources]
        ]

        disasters: list[DisasterSnapshot] = []
        for d in self.engine.disaster_store.items:
            size = d.dirt.level if isinstance(d, Landslide) else d.debris.level if isinstance(d, WildfireDebris) else 0.0
            disasters.append(
                DisasterSnapshot(
                    id=d.id,
                    location=d.location,
                    color=d.render["color"],
                    marker=d.render["marker"],
                    label=d.render["label"],
                    hatch=d.render["hatch"],
                    percent_remaining=d.percent_remaining(),
                    size=size,
                )
            )

        resources: list[ResourceSnapshot] = []
        all_nodes = self.engine.resource_nodes + [self.engine.idle_resources] + self.engine.disaster_store.items
        for node in all_nodes:
            for rtype, rlist in node.roster.items():
                for r in rlist:
                    resources.append(
                        ResourceSnapshot(
                            id=r.id,
                            display_location=_interp_resource_location(r, env_now),
                            marker=rtype.render["marker"],
                            color=rtype.render["color"],
                            label=str(r.id),
                        )
                    )

        return EngineSnapshot(
            policy_name=self.engine.policy.name,
            seed=self.engine.seed,
            env_now=env_now,
            nodes=nodes,
            disasters=disasters,
            resources=resources,
        )

    def _record_history(self, snapshot: EngineSnapshot) -> None:
        # Register new disasters (backfill zeros for ticks they didn't exist)
        for d in snapshot.disasters:
            if d.id not in self._disaster_size:
                self._disaster_size[d.id] = [0.0] * len(self._time_points)
                self._disaster_type_label[d.id] = d.label
                self._disaster_hatch[d.id] = d.hatch

        self._time_points.append(snapshot.env_now)

        # Append current dirt for every known disaster (0 if not active this tick)
        active = {d.id: d.size for d in snapshot.disasters}
        for did, history in self._disaster_size.items():
            history.append(active.get(did, 0.0))

    def _update_nodes(self, ax: Axes, nodes: list[NodeSnapshot]) -> None:
        if not nodes and self._node_scatter:
            self._node_scatter.set_offsets(np.zeros((0, 2)))
            self._node_scatter.set_facecolor([])
            _update_text_pool(ax, self._node_label_pool, [], [])
            return

        positions = [n.location for n in nodes]
        colors = [n.color for n in nodes]
        texts = [f"{n.label}-{n.id}" for n in nodes]

        if self._node_scatter:
            self._node_scatter.set_offsets(np.array(positions))
            self._node_scatter.set_facecolor(colors)
        _update_text_pool(ax, self._node_label_pool, positions, texts, fontsize=8, zorder=4)

    def _update_disasters(self, ax: Axes, disasters: list[DisasterSnapshot]) -> None:
        # Reset all groups' buffers
        for g in self._disaster_groups.values():
            g.positions.clear()
            g.colors.clear()
            g.sizes.clear()

        positions: list[tuple[float, float]] = []
        texts: list[str] = []

        for d in disasters:
            size = (10 + d.percent_remaining * 100) ** 1.5
            group = _get_or_create_marker_group(self._disaster_groups, ax, d.marker, zorder=4, base_size=64)
            group.push(d.location, d.color, size)
            positions.append(d.location)
            texts.append(f"{d.label}-{d.id}\n{int(d.percent_remaining * 100)}%")

        for g in self._disaster_groups.values():
            g.flush()

        _update_text_pool(ax, self._disaster_label_pool, positions, texts, fontsize=8, ha="center", va="center", zorder=5)

    def _update_resources(self, ax: Axes, resources: list[ResourceSnapshot]) -> None:
        # Reset all groups' buffers
        for g in self._resource_groups.values():
            g.positions.clear()
            g.colors.clear()
            g.sizes.clear()

        positions: list[tuple[float, float]] = []
        texts: list[str] = []
        colors: list[str] = []

        for r in resources:
            group = _get_or_create_marker_group(self._resource_groups, ax, r.marker, zorder=6, base_size=64)
            group.push(r.display_location, r.color, 64)
            positions.append(r.display_location)
            texts.append(r.label)
            colors.append(r.color)

        for g in self._resource_groups.values():
            g.flush()

        _update_text_pool(ax, self._resource_label_pool, positions, texts, offset=(2, 2), fontsize=7, zorder=7, colors=colors)

    def _update_dirt_chart(self, ax: Axes) -> None:
        if not self._disaster_size or len(self._time_points) < 2:
            return

        ids = tuple(sorted(self._disaster_size))
        n = len(self._time_points)

        y_data: list[list[float]] = []
        labels: list[str] = []
        hatch: list[str] = []
        for i in ids:
            h = self._disaster_size[i]
            y_data.append(h)
            labels.append(f"{self._disaster_type_label.get(i, 'D')}{i}")
            hatch.append(self._disaster_hatch[i])

        ax.clear()
        ax.set_xlabel("Time")
        ax.set_ylabel("Dirt")
        ax.grid(True)

        ax.stackplot(self._time_points, *y_data, labels=labels, hatch=hatch, alpha=0.8, step="post")
        # ax.legend(loc="upper left", fontsize="small", framealpha=0.5)

        self._dirt_drawn_ids = ids
        self._dirt_drawn_len = n

    def _update_info_panel(self, ax: Axes) -> None:
        text = self._build_info_text()
        if text == self._info_cache:
            return
        ax.clear()
        ax.set_axis_off()
        ax.set_title("Debug State", loc="left")
        ax.text(0.0, 1.0, text, transform=ax.transAxes, va="top", ha="left", fontsize=9, family="monospace")
        self._info_cache = text

    def _capture_initial_resource_counts(self) -> dict[str, int]:
        seen: dict[str, set[int]] = {}
        all_nodes = self.engine.resource_nodes + [self.engine.idle_resources] + self.engine.disaster_store.items
        for node in all_nodes:
            for rtype, store in node.inventory.items():
                ids = seen.setdefault(rtype.name, set())
                for r in store.items:
                    ids.add(r.id)
            for rtype, rlist in node.roster.items():
                ids = seen.setdefault(rtype.name, set())
                for r in rlist:
                    ids.add(r.id)
        return {name: len(ids) for name, ids in sorted(seen.items())}

    def _build_info_text(self) -> str:
        if self._initial_resource_counts is None:
            self._initial_resource_counts = self._capture_initial_resource_counts()

        from typing import cast, Callable

        summary = self.engine.metrics.get_summary()
        cal = self.engine.calendar
        season = cal.get_season().name.title()

        # Active disaster counts
        active_counts: dict[str, int] = {}
        for d in self.engine.disaster_store.items:
            active_counts[d.render["label"]] = active_counts.get(d.render["label"], 0) + 1
        active_mix = ", ".join(f"{k}:{v}" for k, v in sorted(active_counts.items())) or "none"

        # Resolved disaster counts
        resolved_counts: dict[str, int] = {}
        for m in self.engine.metrics.disaster_metrics.values():
            if m["end_time"] is not None:
                lbl = str(m["disaster_type"])
                resolved_counts[lbl] = resolved_counts.get(lbl, 0) + 1
        resolved_mix = ", ".join(f"{k}:{v}" for k, v in sorted(resolved_counts.items())) or "none"

        resource_summary = ", ".join(f"{k}:{v}" for k, v in self._initial_resource_counts.items()) or "none"

        lines: list[str] = [
            f"Calendar: {cal.get_year_progress() * 365.25:.1f} days into year",
            f"Season:   {season}",
            f"Progress: {cal.get_year_progress() * 100:5.1f}%",
            f"Decisions: {self.engine.decisions_made}",
            f"Created/resolved: {summary['total_disasters_created']}/{summary['total_disasters_resolved']}",
            f"Active now: {len(self.engine.disaster_store.items)}",
            f"Active mix: {active_mix}",
            f"Resolved mix: {resolved_mix}",
            "",
            "Starting Config:",
            f"  resources: {resource_summary}",
        ]

        # Spawn weights
        profiles = self.engine.scenario_config.seasonal_spawn
        season_key = cal.get_season().name.lower()
        lines.append("")
        lines.append("Spawn Weights:")
        for dtype in sorted(profiles):
            p = profiles[dtype]
            cnt = p.event_count_range_by_season.get(season_key, (0, 0))
            sz = p.size_range_by_season.get(season_key, (0, 0))
            w = (cnt[0] + cnt[1]) / 2.0
            lines.append(f"  {dtype:<15} w={w:>4.1f} cnt={cnt} size={sz}")

        # Active disasters detail
        lines.append("")
        if not self.engine.disaster_store.items:
            lines.append("Active Disasters: none")
        else:
            lines.append("Active Disasters:")
            for d in sorted(self.engine.disaster_store.items, key=lambda x: x.id):
                assigned = sum(len(r) for r in d.roster.values())
                age = self.engine.env.now - d.created_time
                lines.append(f"  #{d.id:<5} {d.render['label']:<12} " f"rem={d.percent_remaining() * 100:>5.1f}%" f" age={age:>6.1f}m res={assigned} size={d.get_scale():.3f}")

        return "\n".join(lines)

    def update(self, snapshot: EngineSnapshot) -> None:
        if self.fig is None:
            self._init_artists()

        assert self.axs is not None
        ax_map, ax_dirt, ax_info = self.axs

        # FPS
        now = perf_counter()
        if self._last_wall_time is not None:
            dt = now - self._last_wall_time
            if dt > 0:
                inst = 1.0 / dt
                self._fps = inst if self._fps == 0 else self._fps * 0.9 + inst * 0.1
        self._last_wall_time = now
        self._frame += 1

        # Title (throttled)
        if self._frame % 10 == 0:
            ax_map.set_title(f"Policy: {snapshot.policy_name} | Seed: {snapshot.seed} | " f"Time: {snapshot.env_now:.2f} | FPS: {self._fps:.1f}")

        self._record_history(snapshot)
        self._update_nodes(ax_map, snapshot.nodes)
        self._update_disasters(ax_map, snapshot.disasters)
        self._update_resources(ax_map, snapshot.resources)
        self._update_dirt_chart(ax_dirt)
        self._update_info_panel(ax_info)

        plt.draw()
        plt.pause(0.001)

    def close(self) -> None:
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
