from __future__ import annotations

import math
from dataclasses import dataclass
from math import cos, sin
from time import perf_counter
from typing import TYPE_CHECKING, Callable, cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from SimPyTest.simulation import Landslide
    from SimPyTest.engine import SimPySimulationEngine
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
    percent_remaining: float
    dirt_level: float


@dataclass(frozen=True)
class ResourceSnapshot:
    id: int
    location: tuple[float, float]
    prev_location: tuple[float, float]
    move_time: float
    move_start_time: float
    marker: str
    color: str


@dataclass(frozen=True)
class EngineSnapshot:
    policy_name: str
    seed: int
    env_now: float
    nodes: list[NodeSnapshot]
    disasters: list[DisasterSnapshot]
    resources: list[ResourceSnapshot]


class EngineVisualizer:
    def __init__(self, engine: SimPySimulationEngine):
        self.engine: SimPySimulationEngine = engine
        self.fig: Figure | None = None
        self.axs: list[Axes] | None = None
        self._gis_edge_segments: list[tuple[tuple[float, float], tuple[float, float]]] | None = None
        self._gis_bounds: tuple[float, float, float, float] | None = None
        self._time_points: list[float] = []
        self._disaster_histories: dict[int, list[float]] = {}
        self._known_disaster_labels: dict[int, str] = {}
        self._last_render_wall_time: float | None = None
        self._fps_estimate: float = 0.0
        self._initial_resource_counts: dict[str, int] | None = None

    def _capture_initial_resource_counts(self) -> dict[str, int]:
        counts: dict[str, set[int]] = {}
        nodes = self.engine.resource_nodes + [self.engine.idle_resources] + self.engine.disaster_store.items
        for node in nodes:
            for resource_type, store in node.inventory.items():
                ids = counts.setdefault(resource_type.name, set())
                for resource in store.items:
                    ids.add(resource.id)
            for resource_type, resources in node.roster.items():
                ids = counts.setdefault(resource_type.name, set())
                for resource in resources:
                    ids.add(resource.id)
        return {name: len(ids) for name, ids in sorted(counts.items())}

    def _starting_config_lines(self) -> list[str]:
        if self._initial_resource_counts is None:
            self._initial_resource_counts = self._capture_initial_resource_counts()

        lines = ["Starting Config:"]
        resource_summary = ", ".join(f"{name}:{count}" for name, count in self._initial_resource_counts.items())
        lines.append(f"  resources: {resource_summary or 'none'}")
        lines.append(
            "  disasters: " f"target_events={self.engine.scenario_config.seasonal_spawn.target_events_range} " f"interarrival={self.engine.scenario_config.seasonal_spawn.interarrival_minutes_range}"
        )

        for disaster_type in sorted(self.engine.scenario_config.seasonal_spawn.disasters):
            profile = self.engine.scenario_config.seasonal_spawn.disasters[disaster_type]
            lines.append(f"  {disaster_type}:")
            countsline = "    counts:"
            for season, value in sorted(profile.event_count_range_by_season.items()):
                countsline += f" {season}={value}"
            lines.append(countsline)
            sizesline = "    sizes:"
            for season, value in sorted(profile.size_range_by_season.items()):
                sizesline += f" {season}={value}"
            lines.append(sizesline)
        return lines

    def setup(self) -> tuple[Figure, list[Axes]]:
        fig = plt.figure(figsize=(16, 10), constrained_layout=True)
        grid = fig.add_gridspec(2, 2, width_ratios=[2.4, 1.2], height_ratios=[3.0, 1.2])
        ax_map = fig.add_subplot(grid[0, 0])
        ax_dirt = fig.add_subplot(grid[1, 0])
        ax_info = fig.add_subplot(grid[:, 1])
        ax_map.set_aspect("equal")
        ax_map.set_xlabel("X")
        ax_map.set_ylabel("Y")
        ax_map.grid(True, alpha=0.2)

        if self.engine.road_graph is not None and self.engine.gis_config is not None:
            self._gis_edge_segments = [(u, v) for u, v in self.engine.road_graph.edges()]
            xs = [pt[0] for pt in self.engine.road_graph.nodes]
            ys = [pt[1] for pt in self.engine.road_graph.nodes]
            if xs and ys:
                self._gis_bounds = (min(xs), max(xs), min(ys), max(ys))
        else:
            self._gis_edge_segments = None
            self._gis_bounds = None

        min_x, max_x, min_y, max_y = self._compute_map_bounds()
        ax_map.set_xlim(min_x, max_x)
        ax_map.set_ylim(min_y, max_y)
        ax_dirt.set_xlabel("Time")
        ax_dirt.set_ylabel("Dirt")
        ax_dirt.grid(True)
        ax_info.set_axis_off()
        ax_info.set_title("Debug State", loc="left")
        plt.ion()

        self.fig = fig
        self.axs = [ax_map, ax_dirt, ax_info]
        return fig, self.axs

    def _compute_map_bounds(self, snapshot: EngineSnapshot | None = None) -> tuple[float, float, float, float]:
        x_points: list[float] = []
        y_points: list[float] = []
        if snapshot is None:
            for node in self.engine.resource_nodes + [self.engine.idle_resources] + self.engine.disaster_store.items:
                x_points.append(node.location[0])
                y_points.append(node.location[1])
                for resources in node.roster.values():
                    for resource in resources:
                        x_points.extend([resource.location[0], resource.prev_location[0]])
                        y_points.extend([resource.location[1], resource.prev_location[1]])
        else:
            for node in snapshot.nodes:
                x_points.append(node.location[0])
                y_points.append(node.location[1])
            for disaster in snapshot.disasters:
                x_points.append(disaster.location[0])
                y_points.append(disaster.location[1])
            for resource in snapshot.resources:
                x_points.extend([resource.location[0], resource.prev_location[0]])
                y_points.extend([resource.location[1], resource.prev_location[1]])

        if self._gis_bounds is not None:
            x_points.extend([self._gis_bounds[0], self._gis_bounds[1]])
            y_points.extend([self._gis_bounds[2], self._gis_bounds[3]])

        if not x_points or not y_points:
            return (-120, 120, -120, 120)

        min_x = min(x_points)
        max_x = max(x_points)
        min_y = min(y_points)
        max_y = max(y_points)
        x_span = max(max_x - min_x, 1.0)
        y_span = max(max_y - min_y, 1.0)
        x_margin = max(5.0, x_span * 0.05)
        y_margin = max(5.0, y_span * 0.05)
        return (min_x - x_margin, max_x + x_margin, min_y - y_margin, max_y + y_margin)

    def snapshot(self) -> EngineSnapshot:
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
        for disaster in self.engine.disaster_store.items:
            dirt_level = disaster.dirt.level if isinstance(disaster, Landslide) else 0.0
            disasters.append(
                DisasterSnapshot(
                    id=disaster.id,
                    location=disaster.location,
                    color=disaster.render["color"],
                    marker=disaster.render["marker"],
                    label=disaster.render["label"],
                    percent_remaining=disaster.percent_remaining(),
                    dirt_level=dirt_level,
                )
            )

        resources: list[ResourceSnapshot] = []
        for node in self.engine.resource_nodes + [self.engine.idle_resources] + self.engine.disaster_store.items:
            for resource_type, node_resources in node.roster.items():
                for resource in node_resources:
                    resources.append(
                        ResourceSnapshot(
                            id=resource.id,
                            location=resource.location,
                            prev_location=resource.prev_location,
                            move_time=resource.move_time,
                            move_start_time=resource.move_start_time,
                            marker=resource_type.render["marker"],
                            color=resource_type.render["color"],
                        )
                    )

        return EngineSnapshot(
            policy_name=self.engine.policy.name,
            seed=self.engine.seed,
            env_now=self.engine.env.now,
            nodes=nodes,
            disasters=disasters,
            resources=resources,
        )

    def _record_history(self, snapshot: EngineSnapshot) -> None:
        for disaster in snapshot.disasters:
            if disaster.id not in self._disaster_histories:
                self._disaster_histories[disaster.id] = [0.0] * len(self._time_points)
                self._known_disaster_labels[disaster.id] = disaster.label

        self._time_points.append(snapshot.env_now)
        for history in self._disaster_histories.values():
            history.append(0.0)

        for disaster in snapshot.disasters:
            self._disaster_histories[disaster.id][-1] = disaster.dirt_level

    def _format_active_counts(self) -> str:
        counts: dict[str, int] = {}
        for disaster in self.engine.disaster_store.items:
            label = disaster.render["label"]
            counts[label] = counts.get(label, 0) + 1
        if not counts:
            return "none"
        return ", ".join(f"{label}:{counts[label]}" for label in sorted(counts))

    def _format_resolved_counts(self) -> str:
        counts: dict[str, int] = {}
        for metrics in self.engine.metrics.disaster_metrics.values():
            if metrics["end_time"] is None:
                continue
            label = str(metrics["disaster_type"])
            counts[label] = counts.get(label, 0) + 1
        if not counts:
            return "none"
        return ", ".join(f"{label}:{counts[label]}" for label in sorted(counts))

    def _spawn_debug_lines(self) -> list[str]:
        season = self.engine.calendar.get_season().name.lower()
        profiles = self.engine.scenario_config.seasonal_spawn.disasters
        lines = ["Spawn Weights:"]
        for disaster_type in sorted(profiles):
            profile = profiles[disaster_type]
            count_low, count_high = profile.event_count_range_by_season.get(season, (0, 0))
            base_weight = max(0.0, (float(count_low) + float(count_high)) / 2.0)
            weather = self.engine.calendar.get_weather_factor(disaster_type)
            weight = base_weight
            if self.engine.scenario_config.weather_model.enable_spawn_scaling:
                max_boost = self.engine.scenario_config.weather_model.max_rate_weather_boost
                weight *= 1.0 + max(0.0, min(max_boost, weather))
            if self.engine.scenario_config.weather_model.use_vulnerability_weighting:
                weight *= self.engine.scenario_config.get_vulnerability_multiplier(disaster_type)
            size_low, size_high = profile.size_range_by_season.get(season, (0, 0))
            lines.append(f"  {disaster_type:<15} w={weight:>4.1f} " f"cnt=({count_low},{count_high}) size=({size_low},{size_high}) wx={weather:.2f}")
        return lines

    def _active_disaster_lines(self) -> list[str]:
        lines = ["Active Disasters:"]
        if not self.engine.disaster_store.items:
            return ["Active Disasters: none"]

        for disaster in sorted(self.engine.disaster_store.items, key=lambda item: item.id):
            assigned = sum(len(resources) for resources in disaster.roster.values())
            age = self.engine.env.now - disaster.created_time
            lines.append(f"  #{disaster.id:<5} {disaster.render['label']:<12} " f"rem={disaster.percent_remaining()*100:>5.1f}% age={age:>6.1f}m res={assigned}")
        # if len(self.engine.disaster_store.items) > 8:
        #     lines.append(f"  ... {len(self.engine.disaster_store.items) - 8} more")
        return lines

    def _debug_panel_text(self) -> str:
        weather = self.engine.calendar.weather_state
        get_summary = cast(Callable[[], dict[str, float | int]], self.engine.metrics.get_summary)
        summary = get_summary()
        season = self.engine.calendar.get_season().name.title()
        lines = [
            f"Calendar: {self.engine.calendar.current_date.strftime('%Y-%m-%d %H:%M')}",
            f"Season:   {season}",
            f"Year:     {self.engine.calendar.get_year()}  Progress: {self.engine.calendar.get_year_progress()*100:5.1f}%",
            f"Decisions: {self.engine.decisions_made}",
            f"Weather:  temp={weather['temperature']:5.1f}C  rain={weather['rain_intensity']:.2f}",
            f"          wind={weather['wind_speed']:5.1f}  drought={weather['drought_index']:.2f}",
            f"Disasters created/resolved: {summary['total_disasters_created']}/{summary['total_disasters_resolved']}",
            f"Active now: {len(self.engine.disaster_store.items)}",
            f"Active mix: {self._format_active_counts()}",
            f"Resolved mix: {self._format_resolved_counts()}",
            f"Spawn gap min: {self.engine.scenario_config.seasonal_spawn.interarrival_minutes_range}",
            "",
            *self._starting_config_lines(),
            "",
            *self._spawn_debug_lines(),
            "",
            *self._active_disaster_lines(),
        ]
        return "\n".join(lines)

    def update(self, snapshot: EngineSnapshot) -> None:
        render_started = perf_counter()
        if self.fig is None or self.axs is None:
            self.setup()
        if self.axs is None:
            return
        self._record_history(snapshot)

        ax_map, ax_dirt, ax_info = self.axs
        ax_map.clear()
        ax_map.set_aspect("equal")
        ax_map.set_xlabel("X")
        ax_map.set_ylabel("Y")
        ax_map.grid(True, alpha=0.2)
        min_x, max_x, min_y, max_y = self._compute_map_bounds(snapshot)
        ax_map.set_xlim(min_x, max_x)
        ax_map.set_ylim(min_y, max_y)

        if self._gis_edge_segments:
            roads = LineCollection(self._gis_edge_segments, colors="#b0b0b0", linewidths=0.3, alpha=0.5, zorder=1)
            ax_map.add_collection(roads)

        if self._last_render_wall_time is not None:
            elapsed = render_started - self._last_render_wall_time
            if elapsed > 0:
                instantaneous_fps = 1.0 / elapsed
                if self._fps_estimate == 0.0:
                    self._fps_estimate = instantaneous_fps
                else:
                    self._fps_estimate = self._fps_estimate * 0.9 + instantaneous_fps * 0.1
        self._last_render_wall_time = render_started

        ax_map.set_title(f"Policy: {snapshot.policy_name} | Seed: {snapshot.seed} | " f"Time: {snapshot.env_now:.2f} | FPS: {self._fps_estimate:.1f}")

        for node in snapshot.nodes:
            ax_map.plot(node.location[0], node.location[1], node.color, markersize=12, zorder=3)
            ax_map.text(node.location[0], node.location[1], f"{node.label}-{node.id}", fontsize=8, zorder=4)

        for disaster in snapshot.disasters:
            size = disaster.percent_remaining * 100
            ax_map.plot(disaster.location[0], disaster.location[1], disaster.marker, color=disaster.color, markersize=10 + size, alpha=0.75, zorder=4)
            ax_map.text(disaster.location[0], disaster.location[1], f"{disaster.label}-{disaster.id}\n{int(size)}%", fontsize=8, ha="center", va="center", zorder=5)

        for resource in snapshot.resources:
            frac = resource.id * (1 + math.sqrt(5)) / 2
            loc1 = (cos(frac) * 10 + resource.location[0], sin(frac) * 10 + resource.location[1])
            loc2 = (cos(frac) * 10 + resource.prev_location[0], sin(frac) * 10 + resource.prev_location[1])
            time_frac = 1.0 if resource.move_time == 0 else (snapshot.env_now - resource.move_start_time) / resource.move_time
            time_frac = max(0.0, min(1.0, time_frac))
            loc = (
                loc1[0] * time_frac + loc2[0] * (1 - time_frac),
                loc1[1] * time_frac + loc2[1] * (1 - time_frac),
            )
            ax_map.plot(loc[0], loc[1], marker=resource.marker, color=resource.color, markersize=8, zorder=6)
            ax_map.text(loc[0] + 2, loc[1] + 2, f"{resource.id}", color=resource.color, fontsize=7, zorder=7)

        ax_dirt.clear()
        ax_dirt.set_xlabel("Time")
        ax_dirt.set_ylabel("Dirt")
        ax_dirt.grid(True)
        if self._disaster_histories and self._time_points:
            ids = sorted(self._disaster_histories.keys())
            y_data = [self._disaster_histories[i] for i in ids]
            labels = [f"{self._known_disaster_labels.get(i, 'D')}{i}" for i in ids]
            ax_dirt.stackplot(self._time_points, *y_data, labels=labels, alpha=0.8, step="post")
            ax_dirt.legend(loc="upper left", fontsize="small", framealpha=0.5)

        ax_info.clear()
        ax_info.set_axis_off()
        ax_info.set_title("Debug State", loc="left")
        ax_info.text(
            0.0,
            1.0,
            self._debug_panel_text(),
            transform=ax_info.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            family="monospace",
        )

        plt.draw()
        plt.pause(0.001)

    def close(self) -> None:
        if self.fig is not None:
            plt.close(self.fig)
