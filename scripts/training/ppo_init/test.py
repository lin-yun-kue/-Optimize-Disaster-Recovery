from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt

from scripts.training.ppo_init.train_critic import PPOConfig, train

def load_metrics(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "training_metrics.json"
    return json.loads(metrics_path.read_text(encoding="utf-8"))

DEFAULT_ACTOR_CHECKPOINT = "experiment_results/dispatch_ml/20260407_233420/dispatch_model.pt"
DEFAULT_PPO_INIT_CHECKPOINT = "experiment_results/ppo_init_critic/20260513_063620/checkpoint.pt"

total_episodes = 800
freeze_updates = 0
seed = 0

sweep_started_at = datetime.now(timezone.utc)
sweep_dir = Path("experiment_results/ppo_init_critic")
sweep_dir.mkdir(parents=True, exist_ok=True)
run_save_dir = sweep_dir

config = PPOConfig(
                total_episodes=total_episodes,
                freeze_actor_updates=freeze_updates,
                seed=seed,
                device="cuda",
                save_dir=str(run_save_dir),
                # actor_checkpoint=DEFAULT_ACTOR_CHECKPOINT
                ppo_init_checkpoint=DEFAULT_PPO_INIT_CHECKPOINT
            )

run_dir = train(config)
metrics = load_metrics(run_dir)
checkpoint_metrics = metrics["checkpoint_metrics"]

result = {
    "timesteps": total_episodes,
    "freeze_actor_updates": freeze_updates,
    "run_dir": str(run_dir),
    "success_rate": checkpoint_metrics.get("success_rate", 0.0),
    "avg_objective_score": checkpoint_metrics.get("avg_objective_score", 0.0),
    "avg_total_reward": checkpoint_metrics.get("avg_total_reward", 0.0),
}
print(result)