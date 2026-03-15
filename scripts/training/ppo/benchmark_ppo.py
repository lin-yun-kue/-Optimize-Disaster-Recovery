from __future__ import annotations

import json
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from scripts.training.ppo.ppo_dispatch import CHECKPOINT_SEEDS, evaluate_model, load_model, select_device


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python3 scripts/training/ppo/benchmark_ppo.py <model-path>")

    model_path = Path(sys.argv[1])
    model = load_model(model_path, select_device())
    metrics = evaluate_model(model, list(CHECKPOINT_SEEDS), True, "ppo_benchmark")

    print(f"Model: {model_path}")
    print(f"Success rate: {metrics['success_rate'] * 100:.1f}%")
    print(f"Average objective score: {metrics['avg_objective_score']:.2f}")
    print(f"Average total reward: {metrics['avg_total_reward']:.2f}")
    print(f"Average sim time: {metrics['avg_sim_time']:.2f}")
    print(f"Average wall time: {metrics['avg_wall_time_s']:.2f}s")
    print("\nPer-seed results:")
    for episode in metrics["episodes"]:
        status = "SUCCESS" if episode["success"] else "FAIL"
        print(
            f"  seed={episode['seed']:<3} | "
            f"{status:<7} | "
            f"Obj: {episode['objective_score']:<10.2f} | "
            f"Reward: {episode['total_reward']:<10.2f} | "
            f"Sim: {episode['sim_time']:<10.2f} | "
            f"Wall: {episode['wall_time_s']:<10.2f}s"
        )

    output_path = model_path.parent / "benchmark_ppo_metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nWrote metrics to {output_path}")
