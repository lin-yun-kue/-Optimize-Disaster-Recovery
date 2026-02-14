"""
Example usage of the GIS-integrated disaster response simulation.

Has:
Running a simulation with simple coordinates (no GIS)
Running a simulation with GIS-based road networks
Using the Gym wrapper with GIS data
"""

import os
from SimPyTest.engine import SimPySimulationEngine, ScenarioConfig
from SimPyTest.policies import POLICIES
from SimPyTest.gym import DisasterResponseEnv
from SimPyTest.gis_utils import Depot, GISConfig, Landfill, build_road_graph, load_roads, load_and_prune_roads


depots: list[Depot] = [
    {
        "Longitude": -123.92616052570274,
        "Latitude": 46.16262226957932,
        "Name": "ODOT Warrenton",
        "numExc": 5,
        "numTrucks": 50,
        "color": "green",
    },
    {
        "Longitude": -123.92250095724509,
        "Latitude": 45.984013191819535,
        "Name": "ODOT Seaside",
        "numExc": 3,
        "numTrucks": 40,
        "color": "yellow",
    },
    {
        "Longitude": -123.81788435312336,
        "Latitude": 45.91011185257814,
        "Name": "ODOT Necanium",
        "numExc": 4,
        "numTrucks": 30,
        "color": "purple",
    },
]

landfills: list[Landfill] = [
    {"Label": "A", "Longitude": -123.70105856996436, "Latitude": 45.91375387223106, "Name": "Random Landfill"},
    {"Label": "B", "Longitude": -123.80828890576535, "Latitude": 46.17804487993376, "Name": "Astoria Recology"},
    {
        "Label": "C",
        "Longitude": -123.90783452733942,
        "Latitude": 45.95217222679762,
        "Name": "Seaside Knife River Quarry",
    },
]

# Path to road shapefile - update this to your actual path
ROAD_SHAPEFILE = "maps/tl_2024_41007_roads/tl_2024_41007_roads.shp"

# Create GIS configuration with pruned road network (major roads only)
# Uses Interstate (I), US Highway (U), and State (S) roads for faster simulation
# Automatically ensures all depots and landfills remain connected to the network
roads_gdf = load_and_prune_roads(ROAD_SHAPEFILE, depots=depots, landfills=landfills, enabled_types=['I', 'U', 'S'])
road_graph = build_road_graph(roads_gdf)
gis_config = GISConfig(roads_gdf=roads_gdf, road_graph=road_graph, depots=depots, landfills=landfills)


def example_1_basic_simulation():
    """
    Example 1: Basic simulation without GIS (uses simple coordinates).
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Simulation (No GIS)")
    print("=" * 70)

    # Use a simple policy
    policy = [p for p in POLICIES if p.name == "balanced_ratio"][0]

    # Create scenario configuration
    scenario_config = ScenarioConfig(
        num_trucks=50, num_excavators=10, num_landslides=15, landslide_size_range=(150, 250)
    )

    # Create and run simulation
    engine = SimPySimulationEngine(policy=policy, seed=42, scenario_config=scenario_config)
    engine.initialize_world()
    success = engine.run()

    duration = engine.get_summary()["non_idle_time"]
    print(f"\nSimulation {'succeeded' if success else 'failed'}")
    print(f"Non-idle time: {duration:.2f} time units")


def example_2_gis_simulation():
    """
    Example 2: Simulation with GIS road network.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: GIS-Based Simulation")
    print("=" * 70)

    # Check if road shapefile exists
    road_shapefile = "maps/tl_2024_41007_roads/tl_2024_41007_roads.shp"

    if not os.path.exists(road_shapefile):
        print(f"\nSkipping GIS example - road shapefile not found at: {road_shapefile}")
        print("To run this example, update the road_shapefile path in the code.")
        return

    # Use a simple policy
    policy = [p for p in POLICIES if p.name == "balanced_ratio"][0]

    # Create scenario configuration with GIS
    scenario_config = ScenarioConfig(
        num_trucks=depots[0]["numTrucks"],  # Use truck count from first depot
        num_excavators=depots[0]["numExc"],  # Use excavator count from first depot
        num_landslides=15,
        landslide_size_range=(500, 5000),  # Larger landslides like in GIS example
        gis_config=gis_config,
    )

    # Create and run simulation
    engine = SimPySimulationEngine(policy=policy, seed=42, scenario_config=scenario_config)

    print("\nLoading road network...")
    engine.initialize_world()

    if engine.road_graph is not None:
        print(f"Road graph loaded with {len(engine.road_graph.nodes)} nodes")
        print(f"Road graph loaded with {len(engine.road_graph.edges)} edges")

    print("\nRunning simulation...")
    success = engine.run()

    duration = engine.get_summary()["non_idle_time"]
    print(f"\nSimulation {'succeeded' if success else 'failed'}")
    print(f"Non-idle time: {duration:.2f} time units")


def example_3_gym_basic():
    """
    Example 3: Using the Gym environment without GIS.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Gym Environment (No GIS)")
    print("=" * 70)

    # Create scenario configuration
    scenario_config = ScenarioConfig(num_trucks=30, num_excavators=5, num_landslides=5, landslide_size_range=(150, 250))

    # Create Gym environment
    env = DisasterResponseEnv(max_visible_disasters=5, sorting_strategy="nearest", scenario_config=scenario_config)

    # Reset environment
    obs, info = env.reset(seed=42)

    print(f"\nEnvironment initialized")
    print(f"Current resource type: {obs['current_resource_type']}")
    print(f"Visible disasters shape: {obs['visible_disasters'].shape}")
    print(f"Valid actions: {obs['valid_actions']}")

    # Take a few random actions
    done = False
    steps = 0
    max_steps = 10
    terminated = False
    truncated = False

    print("\nTaking random actions...")
    while not done and steps < max_steps:
        # Choose a random valid action
        valid_actions = [i for i, v in enumerate(obs["valid_actions"]) if v == 1]
        if not valid_actions:
            print("No valid actions available!")
            break

        action = env.action_space.sample()
        if action not in valid_actions:
            action = valid_actions[0]

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

        print(
            f"Step {steps}: Action={action}, Active disasters={info['active_disasters']}, Time={info['sim_time']:.2f}"
        )

    if terminated:
        print("\nSimulation completed successfully!")
    elif truncated:
        print("\nSimulation truncated (failed to complete)")
    else:
        print(f"\nStopped after {max_steps} steps")


def example_4_gym_with_gis():
    """
    Example 4: Using the Gym environment with GIS road network.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Gym Environment with GIS")
    print("=" * 70)

    # Check if road shapefile exists
    road_shapefile = "maps/tl_2024_41007_roads/tl_2024_41007_roads.shp"

    if not os.path.exists(road_shapefile):
        print(f"\nSkipping GIS Gym example - road shapefile not found at: {road_shapefile}")
        print("To run this example, update the road_shapefile path in the code.")
        return

    # Create scenario configuration with GIS
    scenario_config = ScenarioConfig(
        num_trucks=depots[0]["numTrucks"],
        num_excavators=depots[0]["numExc"],
        num_landslides=5,
        landslide_size_range=(500, 5000),
        gis_config=gis_config,
    )

    # Create Gym environment
    env = DisasterResponseEnv(max_visible_disasters=5, sorting_strategy="nearest", scenario_config=scenario_config)

    print("\nLoading road network...")
    obs, info = env.reset(seed=42)

    print(f"\nEnvironment initialized with GIS")
    if env.engine is not None and env.engine.road_graph is not None:
        print(f"Road graph nodes: {len(env.engine.road_graph.nodes)}")
        print(f"Road graph edges: {len(env.engine.road_graph.edges)}")
    print(f"Current resource type: {obs['current_resource_type']}")
    print(f"Visible disasters: {obs['visible_disasters'].shape}")

    # Take a few random actions
    done = False
    steps = 0
    max_steps = 10
    terminated = False
    truncated = False

    print("\nTaking random actions...")
    while not done and steps < max_steps:
        valid_actions = [i for i, v in enumerate(obs["valid_actions"]) if v == 1]
        if not valid_actions:
            print("No valid actions available!")
            break

        action = env.action_space.sample()
        if action not in valid_actions:
            action = valid_actions[0]

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

        print(
            f"Step {steps}: Action={action}, Active disasters={info['active_disasters']}, Time={info['sim_time']:.2f}"
        )

    if terminated:
        print("\nSimulation completed successfully!")
    elif truncated:
        print("\nSimulation truncated (failed to complete)")
    else:
        print(f"\nStopped after {max_steps} steps")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("GIS-Integrated Disaster Response Simulation Examples")
    print("=" * 70)

    # Run examples
    example_1_basic_simulation()
    example_2_gis_simulation()
    example_3_gym_basic()
    example_4_gym_with_gis()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70 + "\n")
