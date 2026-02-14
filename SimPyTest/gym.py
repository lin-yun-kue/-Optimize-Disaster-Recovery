from __future__ import annotations
from simpy.core import EmptySchedule
from .simulation import Disaster, Resource, ResourceType, Landslide

from simpy.events import Event
from typing import Literal, TypedDict, cast
from typing import Never
from collections.abc import Generator
from typing_extensions import override

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces
from .engine import ScenarioConfig, SimPySimulationEngine
from .policies import Policy, is_useful


# Define generic types for Gymnasium
# ObsType = dict[str, np.ndarray | int]
class ObsType(TypedDict):
    current_resource_type: int
    visible_disasters: npt.NDArray[np.float32]  # Shape: (max_slots, features)
    valid_actions: npt.NDArray[np.int8]  # Shape: (max_slots,)
    idle_trucks: tuple[float]  # Number of idle trucks normalized
    idle_excavators: tuple[float]  # Number of idle excavators normalized
    # avg_disaster_progress: float  # Average progress across all disasters


ActType = int

SortOptions = Literal["nearest", "furthest", "random", "most_progress", "least_progress"]


class InfoType(TypedDict):
    sim_time: float
    active_disasters: int


class DisasterResponseEnv(gym.Env[ObsType, ActType]):
    """
    Gymnasium environment for Disaster Response.

    max_visible_disasters: The maximum number of disasters that can be visible at once.
    sorting_strategy: The sorting strategy to use when multiple disasters are visible.
    """

    def __init__(
        self, max_visible_disasters: int, sorting_strategy: SortOptions, scenario_config: ScenarioConfig | None = None
    ):
        super().__init__()
        self.scenario_config: ScenarioConfig = scenario_config or ScenarioConfig()
        self.max_slots: int = max_visible_disasters
        self.sorting_strategy: str = sorting_strategy

        # Action: Index of the disaster to assign the current resource to
        self.action_space: spaces.Space[ActType] = spaces.Discrete(self.max_slots)

        self.num_resource_types: int = len(ResourceType)
        self.num_disaster_types: int = 1  # Just Landslide for now

        # Disaster Features:
        # 1. TypeID
        # 2. Progress (0-1)
        # 3. Total Dirt
        # 4. Distance
        # 5. Num Trucks Assigned
        # 6. Num Excavators Assigned
        self.num_disaster_features: int = self.num_disaster_types + 5

        # Observation:
        # 1. Current Resource Type as an index
        # 2. Visible Disasters as a matrix of [TypeID, Progress, Total Dirt, Distance, Num Trucks Assigned, Num Excavators Assigned]
        # 3. Additional resource availability info
        self.observation_space: spaces.Space[ObsType] = cast(
            spaces.Space[ObsType],
            spaces.Dict(
                {
                    "current_resource_type": spaces.Discrete(self.num_resource_types),
                    "visible_disasters": spaces.Box(
                        low=0,
                        high=1,
                        shape=(self.max_slots, self.num_disaster_features),
                        dtype=np.float32,
                    ),
                    "valid_actions": spaces.Box(low=0, high=1, shape=(self.max_slots,), dtype=np.int8),
                    "idle_trucks": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                    "idle_excavators": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                    # "avg_disaster_progress": spaces.Box(low=0, high=1, shape=(), dtype=np.float32),
                }
            ),
        )

        # Mapping from visible disaster index to disaster ID
        self.visible_disaster_mapping: dict[int, int] = {}

        # SimPy stuff
        self.engine: SimPySimulationEngine | None = None
        self.current_resource: Resource | None = None
        self.current_action: int | None = None
        self.decision_needed: bool = False

    def _get_obs(self) -> ObsType:
        """Constructs the observation vector from simulation state."""
        if self.engine is None:
            raise Exception("Environment must be reset before calling step().")
        if self.current_resource is None:
            raise Exception("The current resource is none. This should never happen.")

        disasters = self.engine.disaster_store.items
        candidates = [d for d in disasters if is_useful(self.current_resource, d)]

        def distance_sort(d: Disaster):
            if self.engine is None or self.current_resource is None:
                return float("inf")
            return self.engine.get_distance(self.current_resource, d)

        if self.sorting_strategy == "nearest":
            candidates.sort(key=distance_sort, reverse=True)
        elif self.sorting_strategy == "furthest":
            candidates.sort(key=distance_sort, reverse=False)
        elif self.sorting_strategy == "random":
            self.engine.rng.shuffle(candidates)
        elif self.sorting_strategy == "most_progress":
            candidates.sort(key=lambda d: d.percent_remaining(), reverse=True)
        elif self.sorting_strategy == "least_progress":
            candidates.sort(key=lambda d: d.percent_remaining(), reverse=False)
        else:
            raise Exception(f"Invalid sorting strategy: {self.sorting_strategy}")

        candidates = candidates[: self.max_slots]

        features = np.zeros((self.max_slots, self.num_disaster_features), dtype=np.float32)
        valid_actions = np.zeros(self.max_slots, dtype=np.int8)
        temp_visible_disaster_mapping = {}

        for i, d in enumerate(candidates):
            temp_visible_disaster_mapping[i] = d.id

            # Disaster Type
            type_one_hot = np.zeros(self.num_disaster_types, dtype=np.float32)
            type_one_hot[d.one_hot_index] = 1.0

            # Progress
            progress = d.percent_remaining()

            # Total Dirt
            scale = d.get_scale()

            # Distance
            dist = self.engine.get_distance(self.current_resource, d)

            # Trucks
            trucks = len(d.roster[ResourceType.TRUCK])

            # Excavators
            excavators = len(d.roster[ResourceType.EXCAVATOR])

            features[i] = np.concatenate((type_one_hot, [progress, scale, dist, trucks, excavators]))
            valid_actions[i] = 1.0

        self.visible_disaster_mapping = temp_visible_disaster_mapping

        # Calculate additional observation features
        roster = self.engine.idle_resources.roster
        idle_trucks = len(roster[ResourceType.TRUCK])
        idle_excavators = len(roster[ResourceType.EXCAVATOR])

        # Normalize by max possible resources (use upper bounds from scenario config)
        max_trucks = (
            self.engine.scenario_config.num_trucks[1]
            if isinstance(self.engine.scenario_config.num_trucks, tuple)
            else self.engine.scenario_config.num_trucks
        )
        max_excavators = (
            self.engine.scenario_config.num_excavators[1]
            if isinstance(self.engine.scenario_config.num_excavators, tuple)
            else self.engine.scenario_config.num_excavators
        )

        idle_trucks_norm = min(idle_trucks / max_trucks, 1.0) if max_trucks > 0 else 0.0
        idle_excavators_norm = min(idle_excavators / max_excavators, 1.0) if max_excavators > 0 else 0.0

        # Calculate average disaster progress
        all_disasters = self.engine.disaster_store.items
        if len(all_disasters) > 0:
            avg_progress = sum(d.percent_remaining() for d in all_disasters) / len(all_disasters)
        else:
            avg_progress = 0.0

        return {
            "current_resource_type": self.current_resource.resource_type.value,
            "visible_disasters": features,
            "valid_actions": valid_actions,
            "idle_trucks": (float(idle_trucks_norm),),
            "idle_excavators": (float(idle_excavators_norm),),
            # "avg_disaster_progress": float(avg_progress),
        }

    def _get_info(self) -> InfoType:
        return {
            "sim_time": self.engine.env.now if self.engine else 0,
            "active_disasters": len(self.engine.disaster_store.items) if self.engine else 0,
        }

    def _calculate_reward(
        self,
        last_time: float,
        current_time: float,
        terminated: bool,
        truncated: bool,
        is_valid_action: bool,
    ) -> float:
        """
        Calculate comprehensive reward based on progress, coordination, and efficiency.
        """
        if self.engine is None:
            return 0.0

        reward = 0.0

        # Terminal condition rewards
        if terminated:
            time_bonus = max(0, 5000 - current_time) / 50  # Up to +100 bonus for fast completion
            reward += 100.0 + time_bonus
        elif truncated:
            # Failure: heavy penalty
            reward -= 100.0

        # Invalid action penalty
        if not is_valid_action:
            reward -= 10.0
            return reward

        # Calculate progress deltas
        current_dirt_total = 0.0
        current_disasters_completed = 0

        for disaster in self.engine.disaster_store.items:
            if isinstance(disaster, Landslide):
                # Track remaining dirt
                current_dirt_total += disaster.dirt.level

        # Count disasters that were resolved
        if hasattr(self, "_prev_dirt_total") and self._prev_dirt_total is not None:
            dirt_removed = max(0, self._prev_dirt_total - current_dirt_total)
            # Reward proportional to dirt removed (0.5 per unit)
            reward += dirt_removed * 0.5

        # Store current state for next comparison
        self._prev_dirt_total = current_dirt_total

        # Reward when excavators and trucks are at the same site
        for disaster in self.engine.disaster_store.items:
            if isinstance(disaster, Landslide):
                num_excavators = len(disaster.roster[ResourceType.EXCAVATOR])
                num_trucks = len(disaster.roster[ResourceType.TRUCK])

                # Bonus for having both types (coordination)
                if num_excavators > 0 and num_trucks > 0:
                    # More bonus when ratio is balanced (1:1 to 3:1 trucks:excavators)
                    ratio = num_trucks / num_excavators if num_excavators > 0 else 0
                    if 1 <= ratio <= 3:
                        coordination_bonus = 2.0
                    else:
                        coordination_bonus = 1.0
                    reward += coordination_bonus

        return reward

    def _get_total_resources(self) -> int:
        """Get total number of resources in the simulation."""
        if self.engine is None:
            return 0

        total = 0
        # Count in depot and at disasters
        for node in self.engine.resource_nodes:
            if hasattr(node, "roster"):
                for resources in node.roster.values():
                    total += len(resources)

        # Count idle resources
        for resources in self.engine.idle_resources.roster.values():
            total += len(resources)

        # Count resources at disasters
        for disaster in self.engine.disaster_store.items:
            for resources in disaster.roster.values():
                total += len(resources)

        return max(total, 1)

    @override
    def reset(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, *, seed: int | None = None, options: dict[str, str] | None = None
    ) -> tuple[ObsType, InfoType]:
        """Resets the SimPy engine and runs until the first resource is ready."""
        super().reset(seed=seed)
        if seed is None:
            seed = np.random.randint(0, 2**31 - 1)

        gym_policy = Policy("gym_driver", lambda r, ds, env: ds[0])

        self.engine = SimPySimulationEngine(
            policy=gym_policy,
            seed=seed,
            scenario_config=self.scenario_config,
        )
        self.engine.initialize_world()

        self.current_resource = None
        self.decision_needed = False
        self.visible_disaster_mapping = {}
        self._prev_dirt_total = None

        self.engine.run_in_gym(self.loop)

        self._advance_to_decision()

        return self._get_obs(), self._get_info()

    def _advance_to_decision(self):
        """Helper to loop SimPy events until the simulation flags that an agent action is needed."""
        if self.engine is None:
            return

        try:
            while not self.decision_needed:
                self.engine.env.step()
        except EmptySchedule:
            pass

    @override
    def step(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, action: ActType
    ) -> tuple[ObsType, float, bool, bool, InfoType]:
        """Takes an action (disaster index) and advances SimPy to the next decision point."""
        if self.engine is None:
            raise RuntimeError("Environment must be reset before calling step().")

        self.current_action = int(action)
        is_valid_action = self.current_action in self.visible_disaster_mapping
        self.current_resource = None
        self.decision_needed = False

        terminated = False  # This is when we are done
        truncated = False  # This is when we can't make progress
        reward = 0.0

        last_time = self.engine.env.now

        try:
            # while something until the next decision is needed
            while not self.decision_needed:
                self.engine.env.step()
        except EmptySchedule:
            # CASE 1: The schedule is empty.
            # Did we finish?
            if (
                self.engine.disasters_process
                and self.engine.disasters_process.triggered
                and len(self.engine.disaster_store.items) == 0
            ):
                terminated = True
            else:
                # We ran out of events but disasters remain -> FAILURE
                truncated = True

                # print(f"TRUNCATE DEBUG:")
                # print(f"  Disasters remaining: {len(self.engine.disaster_store.items)}")
                # roster = self.engine.idle_resources.roster
                # print(f"  Idle excavators: {len(roster[ResourceType.EXCAVATOR])}")
                # print(f"  Idle trucks: {len(roster[ResourceType.TRUCK])}")
                # print(f"  Sim time: {self.engine.env.now}")

        if terminated or truncated:
            obs: ObsType = {
                "current_resource_type": 0,
                "visible_disasters": np.zeros((self.max_slots, self.num_disaster_features), dtype=np.float32),
                "valid_actions": np.zeros(self.max_slots, dtype=np.int8),
                "idle_trucks": (0.0,),
                "idle_excavators": (0.0,),
            }
        else:
            obs = self._get_obs()

        info = self._get_info()

        current_time = self.engine.env.now

        # Calculate comprehensive reward
        reward = self._calculate_reward(
            last_time=last_time,
            current_time=current_time,
            terminated=terminated,
            truncated=truncated,
            is_valid_action=is_valid_action,
        )

        return obs, float(reward), terminated, truncated, info

    def loop(self) -> Generator[Event, Resource, Never]:
        """
        The main orchestrator. It waits for ANY resource to become available,
        then asks the policy where to send it.
        """
        if self.engine is None:
            raise RuntimeError("Environment must be reset before calling step().")

        while True:
            resource = yield self.engine.env.process(self.engine.idle_resources.get_any_resource())
            yield self.engine.env.process(self.engine.disaster_store.wait_for_any())

            self.decision_needed = True
            self.current_resource = resource

            # Release control back to the main step loop
            yield self.engine.env.timeout(0)
            # Control returns when a disaster is selected and step is called again

            # Parse current actions here
            if self.current_action is None:
                raise Exception("No action was taken.")

            if self.current_action in self.visible_disaster_mapping:
                disaster_id = self.visible_disaster_mapping[self.current_action]
                target_disaster = None
                for ls in list(self.engine.disaster_store.items):
                    if ls.id == disaster_id:
                        target_disaster = ls
                        break

                if target_disaster is not None:
                    target_disaster.transfer_resource(resource)

            self.current_action = None

    def update_scenario_config(self, new_config: ScenarioConfig):
        """Update the scenario configuration for curriculum learning. This will take effect on next reset()"""
        self.scenario_config = new_config
        print(f"Environment scenario config updated to: {new_config}")

    def update_max_visible_disasters(self, new_max: int):
        """Update the maximum visible disasters for curriculum learning."""
        self.max_slots = new_max
        # Update action space
        self.action_space = spaces.Discrete(self.max_slots)
        # Update observation space
        self.observation_space = cast(
            spaces.Space[ObsType],
            spaces.Dict(
                {
                    "current_resource_type": spaces.Discrete(self.num_resource_types),
                    "visible_disasters": spaces.Box(
                        low=0,
                        high=1,
                        shape=(self.max_slots, self.num_disaster_features),
                        dtype=np.float32,
                    ),
                    "valid_actions": spaces.Box(low=0, high=1, shape=(self.max_slots,), dtype=np.int8),
                    "idle_trucks": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                    "idle_excavators": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                }
            ),
        )
        print(f"Environment max_visible_disasters updated to: {new_max}")
