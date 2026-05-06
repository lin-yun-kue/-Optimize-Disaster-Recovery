from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt

from scripts.training.ppo_init.train_critic import PPOConfig, train

DEFAULT_ACTOR_CHECKPOINT = "experiment_results/dispatch_ml/20260407_233420/dispatch_model.pt"

def load_metrics(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "training_metrics.json"
    return json.loads(metrics_path.read_text(encoding="utf-8"))

def plot_results(results: list[dict[str, Any]], sweep_dir: Path) -> Path:
    freeze_updates_values = sorted({item["freeze_actor_updates"] for item in results})
    timesteps_values = sorted({item["timesteps"] for item in results})

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    metric_specs = [
        ("success_rate", "Success Rate"),
        ("avg_objective_score", "Avg Objective Score"),
        ("avg_total_reward", "Avg Total Reward"),
    ]

    for axis, (metric_key, metric_label) in zip(axes, metric_specs):
        for freeze_updates in freeze_updates_values:
            filtered = [
                item for item in results if item["freeze_actor_updates"] == freeze_updates
            ]
            x_values = [item["timesteps"] for item in sorted(filtered, key=lambda row: row["timesteps"])]
            y_values = [item[metric_key] for item in sorted(filtered, key=lambda row: row["timesteps"])]
            axis.plot(
                x_values,
                y_values,
                marker="o",
                label=f"freeze={freeze_updates}",
            )

        axis.set_title(metric_label)
        axis.set_xlabel("Total Timesteps")
        axis.grid(True, alpha=0.3)
        axis.set_xticks(timesteps_values[::2])
        axis.ticklabel_format(style="plain", axis="x")

    axes[0].set_ylabel("Metric Value")
    axes[2].legend(loc="best")

    fig.suptitle("train_critic Sweep Results", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    chart_path = sweep_dir / "sweep_results.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    return chart_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep train_critic hyperparameters (total_timesteps, freeze_actor_updates)."
    )
    parser.add_argument("--base-save-dir", type=str, default="experiment_results/ppo_init_critic")
    parser.add_argument("--seed", type=int, default=PPOConfig.seed)
    args = parser.parse_args()

    timesteps_list = list(range(10000, 11001, 1000))
    freeze_list = [4]

    sweep_started_at = datetime.now(timezone.utc)
    sweep_dir = Path(args.base_save_dir) / sweep_started_at.strftime("%Y%m%d_%H%M%S")
    sweep_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []

    for timesteps in timesteps_list:
        for freeze_updates in freeze_list:
            run_save_dir = sweep_dir / f"t{timesteps}_freeze{freeze_updates}"
            config = PPOConfig(
                total_timesteps=timesteps,
                freeze_actor_updates=freeze_updates,
                seed=args.seed,
                save_dir=str(run_save_dir),
                actor_checkpoint=DEFAULT_ACTOR_CHECKPOINT
            )
            print(f"[SWEEP] Start timesteps={timesteps}, freeze_actor_updates={freeze_updates}")
            run_dir = train(config)
            metrics = load_metrics(run_dir)
            checkpoint_metrics = metrics["checkpoint_metrics"]

            result = {
                "timesteps": timesteps,
                "freeze_actor_updates": freeze_updates,
                "run_dir": str(run_dir),
                "success_rate": checkpoint_metrics.get("success_rate", 0.0),
                "avg_objective_score": checkpoint_metrics.get("avg_objective_score", 0.0),
                "avg_total_reward": checkpoint_metrics.get("avg_total_reward", 0.0),
            }
            results.append(result)
            print(
                "[SWEEP] Done "
                f"timesteps={timesteps}, freeze_actor_updates={freeze_updates}, "
                f"success={result['success_rate']:.3f}, obj={result['avg_objective_score']:.3f}, reward={result['avg_total_reward']:.3f}"
            )

    best = max(
        results,
        key=lambda item: (item["success_rate"], item["avg_objective_score"], item["avg_total_reward"]),
    )

    summary = {
        "sweep_started_at_utc": sweep_started_at.isoformat(),
        "sweep_finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_config": asdict(PPOConfig(seed=args.seed)),
        "timesteps_list": timesteps_list,
        "freeze_updates_list": freeze_list,
        "results": results,
        "best": best,
        "ranking_policy": ["success_rate", "avg_objective_score", "avg_total_reward"],
    }

    summary_path = sweep_dir / "sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    chart_path = plot_results(results, sweep_dir)
    print(f"[SWEEP] Best combo: {best}")
    print(f"[SWEEP] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
