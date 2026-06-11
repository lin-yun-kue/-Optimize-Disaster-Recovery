import time
from SimPyTest.benchmark_catalog import create_scenario_config
from SimPyTest.policies import Policy, first_priority_policy

# Import the engine we just built
from Generator.ConstrobeEngine import ConStrobeSimulationEngine
from SimPyTest.scenario_types import ResourceCounts, ScenarioConfig, SeasonalDisasterConfig


def main():
    print("=== ConStrobe Disaster Response Simulation ===")

    # 1. Load a benchmark scenario (using None for GIS config to use Euclidean distance initially)
    scenario_name = "easy-winter"
    print(f"1. Loading Scenario: {scenario_name}...")
    scenario_config: ScenarioConfig = ScenarioConfig(
        resource_counts=ResourceCounts(trucks=15, excavators=8),
        seasonal_spawn={
            "landslide": SeasonalDisasterConfig(
                event_count_range_by_season={"winter": (2, 2), "spring": (2, 2), "summer": (2, 2), "fall": (2, 2)},
                size_range_by_season={
                    "winter": (100, 100),
                    "spring": (100, 100),
                    "summer": (100, 100),
                    "fall": (100, 100),
                },
            )
        },
        time_variance=0.1,
        calendar_start_date=0.0,
        calendar_duration_years=0.5,
    )

    # 2. Setup the Policy
    print("2. Setting up First Priority Policy...")
    policy = Policy(name="first_priority", func=first_priority_policy)

    # 3. Initialize the ConStrobe Engine
    print("3. Initializing Engine...")
    engine = ConStrobeSimulationEngine(
        policy=policy,
        scenario_config=scenario_config,
        seed=42,  # Deterministic seed
        track_metrics=True,  # Enable detailed metric tracking
    )

    # 4. Run the simulation
    print("4. Running Simulation (Building JSTRX and booting ConStrobe)...")
    start_time = time.time()

    # This will block while ConStrobe evaluates the JSTRX file via IPC
    success = engine.run()

    elapsed = time.time() - start_time

    # 5. Output the standard Evaluation Metrics
    print(f"\nSimulation finished in {elapsed:.2f} seconds. Success: {success}")

    summary = engine.summary()
    print("\n" + "=" * 40)
    print("         SIMULATION SUMMARY")
    print("=" * 40)
    print(f"Terminal Outcome:         {summary.terminal_outcome}")
    print(f"Disasters Created:        {summary.disasters_created}")
    print(f"Disasters Resolved:       {summary.disasters_resolved}")
    print(f"Resolution Rate:          {summary.resolution_rate * 100:.1f}%")
    print("-" * 40)
    print(f"Total Time w/ Disasters:  {summary.time_with_disasters:.2f} mins")
    print(f"Avg Response Time:        {summary.avg_response_time:.2f} mins")
    print(f"Avg Resolution Time:      {summary.avg_resolution_time:.2f} mins")
    print("-" * 40)
    print(f"Total Resource Hours:     {summary.total_resource_hours:.2f} hrs")
    print(f"Total Operating Cost:     ${summary.total_operating_cost:,.2f}")
    print(f"Total Spent:              ${summary.total_spent:,.2f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
