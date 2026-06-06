import numpy as np
from SimPyTest import DisasterResponseEnv, ScenarioConfig
from DecisionAgent import ContextualBanditAgent


def run_contextual_bandit():
    config = ScenarioConfig(num_trucks=(20, 40), num_landslides=5)
    env = DisasterResponseEnv(8, "nearest", scenario_config=config)
    agent = ContextualBanditAgent(lr=1e-3, epsilon=0.2, seed=0)

    print("--- Starting Test: Random Agent with Action Masking ---")
    obs, info = env.reset(seed=42)

    terminated = False
    truncated = False
    step_count = 0
    total_reward = 0

    while not (terminated or truncated):
        action = agent.select_action(obs)

        if obs["valid_actions"][action] != 1:
            valid = np.where(obs["valid_actions"] == 1)[0]
            action = int(np.random.choice(valid)) if valid.size > 0 else 0

        # 3. APPLY ACTION
        next_obs, reward, terminated, truncated, info = env.step(action)
        print("reward", reward)

        agent.update(obs, action, reward)
        obs = next_obs
        
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
    run_contextual_bandit()
    # try:
    #     run_contextual_bandit()
    # except Exception as e:
    #     print(f"Full run failed: {e}")

    # Test 2: Quick check of agnostic logic
    print("\n--- Running Manual Heuristic Check ---")
    # test_manual_heuristic()


