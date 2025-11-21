from __future__ import annotations
import simpy
import random
import statistics
from typing import List, Dict, TypedDict, DefaultDict
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from math import pi as PI, cos, sin
from simpy.core import EmptySchedule
from argparse import ArgumentParser
from collections import defaultdict
import simpy
import random
import copy
import dill
import statistics
from typing import Any, Callable, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from dataclasses import dataclass, field
from simulation import *
from policies import Policy


# ============================================================================
# MARK: RNG Wrapper
# ============================================================================


class SimulationRNG:
    """Custom RNG that can be cloned with exact state preservation (wraps random.Random)."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        self._rng = random.Random(seed)

    def random(self) -> float:
        return self._rng.random()

    def randint(self, a: int, b: int) -> int:
        return self._rng.randint(a, b)

    def uniform(self, a: float, b: float) -> float:
        return self._rng.uniform(a, b)

    def choice(self, seq):
        return self._rng.choice(seq)

    def setstate(self, state):
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


# ============================================================================
# MARK: Engine
# ============================================================================
class SimPySimulationEngine:
    """
    Simulation engine wrapper for your disaster-resource SimPy simulation.

    Responsibilities:
      - create the simpy.Environment and domain objects (IdleResources, Depot, DumpSite, DisasterStore, resources)
      - provide initialize(), step(), run(), get_results()
      - clone(), save(filepath), load(filepath)
      - simple logging and action_log to help build deterministic replay if needed

    Usage:
      engine = SimPySimulationEngine(policy_func=my_policy, seed=42, config=my_config)
      engine.initialize()          # build nodes/resources/processes (but don't advance)
      engine.run(max_time=1000)    # run until end condition / events exhausted
      engine.clone()               # best-effort clone via dill
    """

    MAX_SIM_TIME = 20_000

    def __init__(
        self,
        policy: Policy,
        seed: int = 0,
        live_plot: bool = False,
    ):
        self.policy = policy
        self.seed = seed
        self.rng = SimulationRNG(seed)
        self.live_plot = live_plot

        # SimPy environment and domain objects
        self.env: simpy.Environment = simpy.Environment()
        self.idle_resources = IdleResources(self)
        self.disaster_store = DisasterStore(self.env)
        # self.resources: List[Resource] = []
        self.resource_nodes: List[ResourceNode] = []

        # result tracking
        self.time_points: List[float] = []
        self.known_disasters: Dict[int, Any] = {}
        self.disaster_histories: Dict[int, List[float]] = {}
        self.non_idle_time = 0.0

        # whether the scheduled "add_disasters" is present and its process reference
        self._disasters_process: Optional[simpy.Process] = None
        self._main_loop_process: Optional[simpy.Process] = None

        if self.live_plot:
            self.fig, self.axs = self.setup_plot()

        print(f"Running {self.policy.name} with seed {self.seed} and plotting {self.live_plot}.")

    # ----------------------------------------------------------------------------
    # MARK: Run Control
    # ----------------------------------------------------------------------------
    def run(self):
        """
        Run to completion (EmptySchedule).
        """

        self._disasters_process = self.env.process(self.add_disasters())
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
                if (
                    self._disasters_process
                    and self._disasters_process.triggered
                    and len(self.disaster_store.items) == 0
                ):
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
                if ls.id not in self.known_disasters:
                    self.known_disasters[ls.id] = ls
                    self.disaster_histories[ls.id] = [0] * len(self.time_points)

            self.time_points.append(self.env.now)
            for ls_id, ls_obj in self.known_disasters.items():
                val = ls_obj.dirt.level if ls_obj in self.disaster_store.items else 0
                self.disaster_histories[ls_id].append(val)

            if self.live_plot and self.axs is not None:
                self.update_plot()

            # success condition
            if (
                self._disasters_process is not None
                and self._disasters_process.triggered
                and len(self.disaster_store.items) == 0
            ):
                simulation_succeeded = True
                break

        if self.live_plot:
            plt.close(self.fig)

        return simulation_succeeded

    # ----------------------------------------------------------------------------
    # MARK: Simulation Processes
    # ----------------------------------------------------------------------------

    def loop(self):
        """
        The main orchestrator. It waits for ANY resource to become available
        at the depot, then asks the policy where to send it.
        """
        while True:
            resource = yield self.env.process(self.idle_resources.get_any_resource())
            yield self.env.process(self.disaster_store.wait_for_any())
            yield self.env.process(self.policy.func(resource, self.disaster_store, self.env))

    # ----------------------------------------------------------------------------
    # MARK: Disaster Generator
    # ----------------------------------------------------------------------------

    def add_disasters(self):
        min_size = SimulationConfig.LANDSLIDE_MIN_SIZE
        max_size = SimulationConfig.LANDSLIDE_MAX_SIZE

        initial_delay = random.randint(10, 20)
        yield self.env.timeout(initial_delay)

        dump_site = [d for d in self.resource_nodes if isinstance(d, DumpSite)][0]

        for i in range(SimulationConfig.NUM_STARTING_LANDSLIDES):
            landslide_size = random.randint(min_size, max_size)
            landslide = Landslide(self, landslide_size, dump_site)
            self.disaster_store.put(landslide)

        for i in range(SimulationConfig.NUM_LANDSLIDES):
            landslide_size = random.randint(min_size, max_size)
            landslide = Landslide(self, landslide_size, dump_site)
            self.disaster_store.put(landslide)

            delay = random.randint(100, 500)
            if i < SimulationConfig.NUM_LANDSLIDES - 1:
                yield self.env.timeout(delay)

    # ----------------------------------------------------------------------------
    # MARK: Results & Plotting helpers
    # ----------------------------------------------------------------------------
    def get_summary(self) -> Dict[str, Any]:
        """Return a summary dictionary of results similar to old engine get_results()."""
        res = {
            "sim_time": self.env.now,
            "non_idle_time": self.non_idle_time,
            "num_known_disasters": len(self.known_disasters),
            "time_points": self.time_points,
            "disaster_histories": self.disaster_histories,
        }
        return res

    def setup_plot(self) -> Tuple[Figure, List[Axes]]:
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
        if self.disaster_histories and len(self.time_points) > 0:
            ids = sorted(self.disaster_histories.keys())
            y_data = [self.disaster_histories[i] for i in ids]
            labels = [f"L{i}" for i in ids]
            ax_dirt.stackplot(self.time_points, *y_data, labels=labels, alpha=0.8, step="post")
            ax_dirt.legend(loc="upper left", fontsize="small", framealpha=0.5)

        plt.draw()
        plt.pause(0.001)

    # ----------------------------------------------------------------------------
    # MARK: Clone / Save / Load
    # ----------------------------------------------------------------------------
    def clone(self) -> "SimPySimulationEngine":
        """
        Attempt to create an exact copy of the engine using dill.
        """
        try:
            # Use dill to deep-copy entire engine
            cloned = dill.loads(dill.dumps(self))
            return cloned
        except Exception as e:
            # Provide a helpful error and a fallback idea
            raise RuntimeError("Cloning via dill failed. Original error: " + repr(e)) from e

    def save(self, filepath: str):
        """Persist engine to disk (best-effort using dill)."""
        with open(filepath, "wb") as f:
            dill.dump(self, f)

    @staticmethod
    def load(filepath: str) -> "SimPySimulationEngine":
        with open(filepath, "rb") as f:
            return dill.load(f)


# # ---------------------------------------------------------------------
# # Example adapter: replace your `run_simulation` function to use engine
# # ---------------------------------------------------------------------
# def example_run_with_engine(policy_name: str, policy_func: Callable, seed_value: int, live_plot=False):
#     engine = SimPySimulationEngine(
#         policy_func=policy_func,
#         seed=seed_value,
#         live_plot=live_plot,
#         config={
#             # optionally mirror SimulationConfig constants used
#             "NUM_TRUCKS": SimulationConfig.NUM_TRUCKS,
#             "NUM_EXCAVATORS": SimulationConfig.NUM_EXCAVATORS,
#             "NUM_FIRE_TRUCKS": SimulationConfig.NUM_FIRE_TRUCKS,
#             "NUM_AMBULANCES": SimulationConfig.NUM_AMBULANCES,
#         },
#     )
#     # env/objects already created by constructor; if you disabled that, call engine._create_env_and_objects()
#     success = engine.run(max_time=MAX_SIM_TIME)
#     return success, engine.non_idle_time
