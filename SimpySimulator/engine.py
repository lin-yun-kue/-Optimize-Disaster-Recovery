from __future__ import annotations
from simpy.events import Process
import simpy
import random
from typing import Final, Never, TypeVar, TYPE_CHECKING
from collections.abc import Generator
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from math import cos, sin
from simpy.core import EmptySchedule
from .simulation import *

if TYPE_CHECKING:
    from .policies import Policy


# ============================================================================
# MARK: RNG Wrapper
# ============================================================================


T = TypeVar("T")


class SimulationRNG:
    """Custom RNG that can be cloned with exact state preservation (wraps random.Random)."""

    seed: int
    _rng: random.Random

    def __init__(self, seed: int = 0):
        self.seed = seed
        self._rng = random.Random(seed)

    def random(self) -> float:
        return self._rng.random()

    def randint(self, a: int, b: int) -> int:
        return self._rng.randint(a, b)

    def uniform(self, a: float, b: float) -> float:
        return self._rng.uniform(a, b)

    def choice(self, seq: list[T]) -> T:
        return self._rng.choice(seq)

    def shuffle(self, seq: list[T]) -> None:
        self._rng.shuffle(seq)

    def setstate(self, state: tuple[int, int, int, int, int, int]):
        self._rng.setstate(state)

    def getstate(self):
        return self._rng.getstate()

    def reseed(self, seed: int):
        self.seed = seed
        self._rng = random.Random(seed)

    def clone(self) -> "SimulationRNG":
        new = SimulationRNG(self.seed)
        new._rng.setstate(self._rng.getstate())
        return new


@dataclass
class ScenarioConfig:
    # Resource counts (can be a range for randomization)
    num_trucks: int | tuple[int, int] = 50
    num_excavators: int | tuple[int, int] = 10

    # Disaster counts
    num_landslides: int | tuple[int, int] = (10, 15)
    landslide_size_range: tuple[int, int] = (150, 250)

    # Optional callback for manual placement: (engine) -> None
    custom_setup_fn: Callable[[SimPySimulationEngine], None] | None = None


# ============================================================================
# MARK: Engine
# ============================================================================
class SimPySimulationEngine:
    """
    Simulation engine wrapper for your disaster-resource SimPy simulation.
    """

    MAX_SIM_TIME: int = 20_000

    def __init__(
        self,
        policy: Policy,
        seed: int = 0,
        live_plot: bool = False,
        scenario_config: ScenarioConfig | None = None,
    ):
        self.policy: Policy = policy
        self.seed: int = seed
        self.rng: SimulationRNG = SimulationRNG(seed)
        self.decision_rng: SimulationRNG = SimulationRNG(seed + 99999)
        self.live_plot: bool = live_plot

        if scenario_config is None:
            self.scenario_config: ScenarioConfig = ScenarioConfig()
        else:
            self.scenario_config = scenario_config

        # SimPy environment and domain objects
        self.env: simpy.Environment = simpy.Environment()
        self.idle_resources: IdleResources = IdleResources(self)
        self.disaster_store: DisasterStore = DisasterStore(self.env)
        # self.resources: List[Resource] = []
        self.resource_nodes: list[ResourceNode] = []

        # result tracking
        self._time_points: list[float] = []
        self._known_disasters: dict[int, Disaster] = {}
        self._disaster_histories: dict[int, list[float]] = {}
        self.non_idle_time: float = 0.0
        self._tournament_decisions: list[tuple[float, str]] = []

        # replay
        self.decision_log: list[int] = []
        self.replay_buffer: list[int] = []
        self.branch_decision: int | None = None

        # whether the scheduled "add_disasters" is present and its process reference
        self.disasters_process: simpy.Process | None = None
        self._main_loop_process: simpy.Process | None = None

        self.fig: Figure | None = None
        self.axs: list[Axes] | None = None

        print(f"Running {self.policy.name} with seed {self.seed}.")

    # ----------------------------------------------------------------------------
    # MARK: Run Control
    # ----------------------------------------------------------------------------
    def run(self):
        """
        Run to completion (EmptySchedule).
        """
        if self.live_plot and self.fig is None:
            self.fig, self.axs = self.setup_plot()

        self.disasters_process = self.env.process(self.add_disasters())
        self._main_loop_process = self.env.process(self.loop())

        simulation_succeeded = False
        last_idle_time = self.env.now

        while True:
            try:
                target_time = self.env.now + 1
                while self.env.now < target_time:
                    self.env.step()
            except EmptySchedule:
                # CASE 1: The schedule is empty.
                # Did we finish?
                if self.disasters_process and self.disasters_process.triggered and len(self.disaster_store.items) == 0:
                    simulation_succeeded = True
                else:
                    # We ran out of events but disasters remain -> FAILURE
                    simulation_succeeded = False
                break
            except Exception as e:
                print(f"   [!] Exception: {e}")
                simulation_succeeded = False
                print(f"   [!] Policy {self.policy.name} failed at {self.env.now}.")

                raise e

            # break if max time reached
            if self.env.now > self.MAX_SIM_TIME:
                simulation_succeeded = False
                print(f"   [!] Policy {self.policy.name} timed out at {self.env.now}.")
                break

            # collect idle time
            if self.disaster_store.items:
                self.non_idle_time += self.env.now - last_idle_time
            last_idle_time = self.env.now

            # detect new disasters
            for ls in list(self.disaster_store.items):
                if ls.id not in self._known_disasters:
                    self._known_disasters[ls.id] = ls
                    self._disaster_histories[ls.id] = [0] * len(self._time_points)

            self._time_points.append(self.env.now)
            for ls_id, ls_obj in self._known_disasters.items():
                if isinstance(ls_obj, Landslide):
                    val = ls_obj.dirt.level
                else:
                    val = 0
                self._disaster_histories[ls_id].append(val)

            if self.live_plot and self.axs is not None:
                self.update_plot()

            # success condition
            if self.disasters_process.triggered and len(self.disaster_store.items) == 0:
                simulation_succeeded = True
                break

        if self.live_plot:
            plt.close(self.fig)

        return simulation_succeeded

    # ----------------------------------------------------------------------------
    # MARK: Run in Gym Environment
    # ----------------------------------------------------------------------------

    def run_in_gym(self, gym_loop: Callable[[], Generator[simpy.Event, Resource, Never]]):
        """Run the simulation in a Gym environment."""

        self.disasters_process = self.env.process(self.add_disasters())
        self._main_loop_process = self.env.process(gym_loop())

    # ----------------------------------------------------------------------------
    # MARK: Simulation Processes
    # ----------------------------------------------------------------------------

    def loop(self) -> Generator[Process, Resource, Never]:
        """
        The main orchestrator. It waits for ANY resource to become available
        at the depot, then asks the policy where to send it.
        """
        while True:
            resource = yield self.env.process(self.idle_resources.get_any_resource())
            yield self.env.process(self.disaster_store.wait_for_any())

            target_disaster = None

            if len(self.replay_buffer) > 0:
                # Force the decision from history
                target_id = self.replay_buffer.pop(0)

                # Find the disaster object with this ID
                candidates = [d for d in self.disaster_store.items if d.id == target_id]
                target_disaster = candidates[0]
            else:
                # Ask the policy
                if len(self.disaster_store.items) == 1:
                    target_disaster = self.disaster_store.items[0]
                else:
                    target_disaster = self.policy.func(resource, self.disaster_store.items, self.env)

                # If this was the VERY FIRST move after replay buffer emptied, capture it
                if self.branch_decision is None:
                    self.branch_decision = target_disaster.id

                # Record this decision for future replays
                self.decision_log.append(target_disaster.id)

            # disaster = self.policy.func(resource, self.disaster_store.items, self.env)
            target_disaster.transfer_resource(resource)

    # ----------------------------------------------------------------------------
    # MARK: Decision Loop
    # ----------------------------------------------------------------------------

    def get_distance(self, r1: Resource, r2: ResourceNode) -> float:
        return math.hypot(r1.location[0] - r2.location[0], r1.location[1] - r2.location[1])

    # ----------------------------------------------------------------------------
    # MARK: Disaster Generator
    # ----------------------------------------------------------------------------

    def add_disasters(self):
        dump_site = [d for d in self.resource_nodes if isinstance(d, DumpSite)][0]

        def get_count(val: int | tuple[int, int]) -> int:
            return self.rng.randint(val[0], val[1]) if isinstance(val, tuple) else val

        num_to_spawn = get_count(self.scenario_config.num_landslides)
        low, high = self.scenario_config.landslide_size_range

        for _ in range(num_to_spawn):
            landslide = Landslide(self, self.rng.randint(low, high), dump_site)
            self.disaster_store.put(landslide)

            # Optional: Add a small delay between spawns
            yield self.env.timeout(self.rng.uniform(0, 10))

    def initialize_world(self):
        """Initialize the world with a randomized set of resources."""

        if self.scenario_config.custom_setup_fn:
            self.scenario_config.custom_setup_fn(self)
        else:
            depot = Depot(self)
            dump_site = DumpSite(self)

            self.resource_nodes.append(depot)
            self.resource_nodes.append(dump_site)

            def get_count(val: int | tuple[int, int]) -> int:
                return self.rng.randint(val[0], val[1]) if isinstance(val, tuple) else val

            spawn_plan = [
                (ResourceType.TRUCK, get_count(self.scenario_config.num_trucks)),
                (ResourceType.EXCAVATOR, get_count(self.scenario_config.num_excavators)),
            ]

            rid = 0
            for r_type, count in spawn_plan:
                for _ in range(count):
                    r = Resource(rid, r_type, self)
                    depot.transfer_resource(r)
                    rid += 1

    # ----------------------------------------------------------------------------
    # MARK: Results & Plotting helpers
    # ----------------------------------------------------------------------------
    def get_summary(self) -> dict[str, float]:
        """Return a summary dictionary of results similar to old engine get_results()."""
        # res = {
        #     "sim_time": self.env.now,
        #     "non_idle_time": self.non_idle_time,
        #     "tournament_decisions": self.tournament_decisions,
        #     "num_known_disasters": len(self.known_disasters),
        #     "time_points": self.time_points,
        #     "disaster_histories": self.disaster_histories,
        # }
        # return res
        return {
            "non_idle_time": self.non_idle_time,
        }

    def setup_plot(self) -> tuple[Figure, list[Axes]]:
        fig, axs = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={"height_ratios": [3, 1]})
        axs[0].set_xlim(-120, 120)
        axs[0].set_ylim(-120, 120)
        axs[0].set_aspect("equal")
        axs[1].set_xlabel("Time")
        axs[1].set_ylabel("Dirt")
        axs[1].grid(True)
        plt.ion()
        return fig, axs

    def update_plot(self):
        if self.fig is None or self.axs is None:
            return
        ax_map, ax_dirt = self.axs
        ax_map.clear()
        ax_map.set_xlim(-120, 120)
        ax_map.set_ylim(-120, 120)
        ax_map.set_title(f"Policy: {self.policy.name} | Seed: {self.seed} | Time: {self.env.now:.2f}")

        for node in self.resource_nodes + [self.idle_resources] + self.disaster_store.items:
            if isinstance(node, Disaster):
                size = node.percent_remaining() * 100
                ax_map.plot(
                    node.location[0],
                    node.location[1],
                    node.render["marker"],
                    color=node.render["color"],
                    markersize=10 + size,
                    alpha=0.7,
                )
                ax_map.text(
                    node.location[0],
                    node.location[1],
                    f"{node.render['label']}{node.id}\n{int(size)}",
                    fontsize=9,
                    ha="center",
                    va="center",
                )
            else:
                ax_map.plot(
                    node.location[0], node.location[1], node.render["color"], markersize=12, label=node.render["label"]
                )
                ax_map.text(node.location[0], node.location[1], f"{node.render['label']}-{node.id}", fontsize=8)

            # Plot Resources
            for resource_type, resources in node.roster.items():
                for r in resources:
                    frac = r.id * (1 + math.sqrt(5)) / 2
                    loc1 = (cos(frac) * 10 + r.location[0], sin(frac) * 10 + r.location[1])
                    loc2 = (
                        cos(frac) * 10 + r.prev_location[0],
                        sin(frac) * 10 + r.prev_location[1],
                    )

                    time_frac = 1 if r.move_time == 0 else (self.env.now - r.move_start_time) / r.move_time
                    time_frac = max(0, time_frac)
                    time_frac = min(1, time_frac)

                    loc = (
                        loc1[0] * time_frac + loc2[0] * (1 - time_frac),
                        loc1[1] * time_frac + loc2[1] * (1 - time_frac),
                    )

                    marker = resource_type.render["marker"]
                    color = resource_type.render["color"]

                    ax_map.plot(loc[0], loc[1], marker=marker, color=color, markersize=8)
                    ax_map.text(loc[0] + 2, loc[1] + 2, f"{r.id}", color=color, fontsize=8)

        ax_dirt.clear()
        if self._disaster_histories and len(self._time_points) > 0:
            ids = sorted(self._disaster_histories.keys())
            y_data = [self._disaster_histories[i] for i in ids]
            labels = [f"L{i}" for i in ids]
            ax_dirt.stackplot(self._time_points, *y_data, labels=labels, alpha=0.8, step="post")
            ax_dirt.legend(loc="upper left", fontsize="small", framealpha=0.5)

        plt.draw()
        plt.pause(0.001)
