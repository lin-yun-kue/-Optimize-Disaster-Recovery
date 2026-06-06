from __future__ import annotations

import json
from pathlib import Path

import torch

from ppo_init.train_critic import PPOAgent, PPOConfig, evaluate_agent

RUN_DIR = Path("experiment_results\\ppo_init_critic\\20260506_082908\\t11000_freeze4\\20260506_083437")
CHECKPOINT_PATH = RUN_DIR / "checkpoint.pt"
TRAINING_METRICS_PATH = RUN_DIR / "training_metrics.json"
DEVICE = "cuda"  # cpu / cuda / mps


def load_config_from_training_metrics(training_metrics_path: Path) -> PPOConfig:
    payload = json.loads(training_metrics_path.read_text(encoding="utf-8"))
    return PPOConfig(**payload["config"])


def load_agent(checkpoint_path: Path, config: PPOConfig, device: torch.device) -> PPOAgent:
    payload = torch.load(checkpoint_path, map_location=device)
    agent = PPOAgent(config).to(device)
    agent.actor.load_state_dict(payload["actor_state_dict"])
    agent.critic.load_state_dict(payload["critic_state_dict"])
    agent.eval()
    return agent


def main() -> None:
    device = torch.device(DEVICE)
    config = load_config_from_training_metrics(TRAINING_METRICS_PATH)
    agent = load_agent(CHECKPOINT_PATH, config, device)

    checkpoint_metrics = evaluate_agent(agent, config, device, deterministic=True)

    print(f"checkpoint: {CHECKPOINT_PATH.resolve()}")
    print(f"training_metrics: {TRAINING_METRICS_PATH.resolve()}")
    print(
        "Checkpoint eval | "
        f"obj={checkpoint_metrics['avg_objective_score']:.2f} | "
        f"success={checkpoint_metrics['success_rate'] * 100:.1f}% | "
        f"reward={checkpoint_metrics['avg_total_reward']:.2f}"
    )
    # print(json.dumps(checkpoint_metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
