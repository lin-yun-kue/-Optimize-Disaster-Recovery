import numpy as np
from SimPyTest import DisasterResponseEnv, ScenarioConfig


def run_test():
    config = ScenarioConfig(num_trucks=(20, 40), num_landslides=5)
    env = DisasterResponseEnv(8, "nearest", scenario_config=config)

    print("--- Starting Test: Random Agent with Action Masking ---")
    obs, info = env.reset(seed=42)

    terminated = False
    truncated = False
    step_count = 0
    total_reward = 0

    while not (terminated or truncated):
        # The Observation is a Dict (The Flexible Contract)
        disaster_matrix = obs["visible_disasters"]
        mask = obs["valid_actions"]
        # resource_data = obs["current_resource_type"]

        # 2. DECISION LOGIC (The Agent)
        # Find all indices where the mask is 1 (valid targets)
        valid_indices = np.where(mask == 1)[0]

        if len(valid_indices) == 0:
            # If no disasters are active, we might just have to wait
            # (though the env shouldn't prompt for a decision if none exist)
            action = 0
        else:
            # NOVEL AGENT LOGIC:
            # Instead of just random, let's pick the disaster with the most 'partners'
            # Row index 7 in our matrix was 'PartnerCount'
            partner_counts = disaster_matrix[valid_indices, 2]
            best_sub_idx = np.argmax(partner_counts)
            action = valid_indices[best_sub_idx]

        # 3. APPLY ACTION
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        step_count += 1

        if step_count % 10 == 0:
            print(f"Step: {step_count} | Sim Time: {info['sim_time']:.2f} | Reward: {reward:.2f}")

    print(f"--- Episode Finished ---")
    print(f"Total Steps: {step_count}")
    print(f"Final Reward: {total_reward:.2f}")
    print(f"Final Sim Time: {info['sim_time']:.2f}")

    env.close()


if __name__ == "__main__":
    # Test 1: Full run with visualization
    try:
        run_test()
    except Exception as e:
        print(f"Full run failed: {e}")

    # Test 2: Quick check of agnostic logic
    print("\n--- Running Manual Heuristic Check ---")
    # test_manual_heuristic()
