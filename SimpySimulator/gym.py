from __future__ import annotations
from simpy.core import EmptySchedule
from .simulation import Disaster, Resource, ResourceType

from simpy.events import Event
from typing import Literal, Never, TypedDict, cast
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
        # 3. Distance
        self.num_disaster_features: int = self.num_disaster_types + 2

        # Observation:
        # 1. Current Resource Type as an index
        # 2. Visible Disasters as a matrix of [TypeID, Progress, Distance]
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
                return 0
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
        self.visible_disaster_mapping = {}

        for i, d in enumerate(candidates):
            self.visible_disaster_mapping[i] = d.id

            # Disaster Type
            type_one_hot = np.zeros(self.num_disaster_types, dtype=np.float32)
            type_one_hot[d.one_hot_index] = 1.0

            # Progress
            progress = d.percent_remaining()

            # Distance
            dist = self.engine.get_distance(self.current_resource, d)

            features[i] = np.concatenate((type_one_hot, [progress, dist]))
            valid_actions[i] = 1.0

        return {
            "current_resource_type": self.current_resource.resource_type.value,
            "visible_disasters": features,
            "valid_actions": valid_actions,
        }

    def _get_info(self) -> InfoType:
        return {
            "sim_time": self.engine.env.now if self.engine else 0,
            "active_disasters": len(self.engine.disaster_store.items) if self.engine else 0,
        }

    @override
    def reset(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, *, seed: int | None = None, options: dict[str, str] | None = None
    ) -> tuple[ObsType, InfoType]:
        """Resets the SimPy engine and runs until the first resource is ready."""
        super().reset(seed=seed)

        gym_policy = Policy("gym_driver", lambda r, ds, env: ds[0])

        self.engine = SimPySimulationEngine(
            policy=gym_policy,
            seed=seed if seed is not None else 0,
            scenario_config=self.scenario_config,
        )
        self.engine.initialize_world()

        self.current_resource = None
        self.decision_needed = False
        self.visible_disaster_mapping = {}

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

        self.current_action = action
        self.current_resource = None
        self.decision_needed = False

        terminated = False  # This is when we are done
        truncated = False  # This is when we can't make progress
        reward = 0.0

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

        if terminated or truncated:
            obs: ObsType = {
                "current_resource_type": 0,
                "visible_disasters": np.zeros((self.max_slots, self.num_disaster_features), dtype=np.float32),
                "valid_actions": np.zeros(self.max_slots, dtype=np.int8),
            }
        else:
            obs = self._get_obs()

        return obs, float(reward), terminated, truncated, self._get_info()

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

            disaster_id = self.visible_disaster_mapping[self.current_action]
            target_disaster = None
            for ls in list(self.engine.disaster_store.items):
                if ls.id == disaster_id:
                    target_disaster = ls
                    break

            if target_disaster is None:
                raise Exception("No disaster found with this ID.")

            target_disaster.transfer_resource(resource)

            self.current_action = None
