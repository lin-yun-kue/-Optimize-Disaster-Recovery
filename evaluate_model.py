from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from scripts.training.ppo_init.train_critic import (
    PPOAgent,
    PPOConfig,
    evaluate_agent,
    make_env,
    select_device,
)
from scripts.training.mlp.ml_dispatch import TrainedDispatchPolicy

DEFAULT_SCENARIOS = ["medium-winter"]
# DEFAULT_SCENARIOS = ["hard-winter"]
DEFAULT_SEEDS = list(range(200000, 200010))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def save_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = [
        "scenario",
        "seed",
        "success",
        "objective_score",
        "total_reward",
        "terminated",
        "truncated",
        "last_outcome",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_config(
    scenario_name: str,
    checkpoint_config: dict[str, Any] | None,
    device: str,
    save_dir: Path,
    max_visible_disasters: int,
    actor_hidden_dim: int,
    actor_depth: int,
    actor_dropout: float,
    critic_hidden_dim: int,
    critic_depth: int,
) -> PPOConfig:
    config = PPOConfig(
        scenario_name=scenario_name,
        max_visible_disasters=max_visible_disasters,
        device=device,
        save_dir=str(save_dir),
        actor_hidden_dim=actor_hidden_dim,
        actor_depth=actor_depth,
        actor_dropout=actor_dropout,
        critic_hidden_dim=critic_hidden_dim,
        critic_depth=critic_depth,
    )

    if checkpoint_config is not None:
        # If users did not override key architecture values, use checkpoint config.
        for key in [
            "max_visible_disasters",
            "actor_hidden_dim",
            "actor_depth",
            "actor_dropout",
            "critic_hidden_dim",
            "critic_depth",
        ]:
            if getattr(config, key) is None and key in checkpoint_config:
                setattr(config, key, checkpoint_config[key])

    return config


def detect_checkpoint_type(checkpoint_path: Path, preferred_type: str = "auto") -> str:
    if preferred_type != "auto":
        return preferred_type
    payload = torch.load(checkpoint_path, map_location="cpu")
    if "actor_state_dict" in payload and "critic_state_dict" in payload:
        return "ppo"
    if "state_dict" in payload and "metadata" in payload:
        return "ml"
    raise ValueError(
        f"Unable to detect checkpoint type from {checkpoint_path}. "
        "Use --model-type ppo or --model-type ml."
    )


def load_ppo_checkpoint(
    checkpoint_path: Path,
    config: PPOConfig,
    device: torch.device,
) -> tuple[PPOAgent, dict[str, Any]]:
    agent = PPOAgent(config).to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    agent.actor.load_state_dict(payload["actor_state_dict"])
    agent.critic.load_state_dict(payload["critic_state_dict"])
    checkpoint_config = payload.get("config", {})
    return agent, checkpoint_config


def evaluate_dispatch_policy(
    policy: TrainedDispatchPolicy,
    config: PPOConfig,
    device: torch.device,
    deterministic: bool,
    eval_seeds: list[int] | None = None,
) -> dict[str, Any]:
    policy.model.eval()
    episodes: list[dict[str, float | bool | int | str | None]] = []
    seeds = eval_seeds if eval_seeds is not None else config.test_seeds
    for seed in seeds:
        env = make_env(config, controller_name="ml_dispatch_checkpoint")
        observation, _ = env.reset(seed=seed)
        terminated = False
        truncated = False
        total_reward = 0.0
        info: dict[str, Any] = {}

        while not terminated and not truncated:
            action = policy.predict(observation, deterministic=deterministic)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
        print(
            f"Seed {seed}: success={info['is_success']}, objective_score={info.get('objective_score', 0.0)}, "
            f"total_reward={total_reward}, terminated={terminated}, truncated={truncated}, last_outcome={env.engine.last_terminal_outcome}    "
        )
        episodes.append(
            {
                "seed": int(seed),
                "success": bool(info["is_success"]),
                "objective_score": float(info.get("objective_score", 0.0)),
                "total_reward": total_reward,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "last_outcome": env.engine.last_terminal_outcome,
            }
        )

    successes = [episode for episode in episodes if bool(episode["success"])]
    scores = [float(episode.get("objective_score", 0)) for episode in episodes if episode.get("success")]
    rewards = [float(episode.get("total_reward", 0)) for episode in episodes if episode.get("success")]
    return {
        "episodes": episodes,
        "success_rate": len(successes) / len(episodes) if episodes else 0.0,
        "avg_objective_score": statistics.mean(scores) if scores else 0,
        "avg_total_reward": statistics.mean(rewards) if rewards else 0,
    }


def run_evaluation(
    checkpoint_path: Path,
    model_type: str,
    scenarios: list[str],
    seeds: list[int],
    device_name: str,
    output_dir: Path,
    deterministic: bool,
    max_visible_disasters: int,
    actor_hidden_dim: int,
    actor_depth: int,
    actor_dropout: float,
    critic_hidden_dim: int,
    critic_depth: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name if device_name != "auto" else select_device())

    summary_rows: list[dict[str, Any]] = []
    scenario_results: list[dict[str, Any]] = []

    checkpoint_type = detect_checkpoint_type(checkpoint_path, model_type)
    for scenario_name in scenarios:
        config = PPOConfig(
            scenario_name=scenario_name,
            max_visible_disasters=max_visible_disasters,
            device=device_name,
            save_dir=str(output_dir),
            actor_hidden_dim=actor_hidden_dim,
            actor_depth=actor_depth,
            actor_dropout=actor_dropout,
            critic_hidden_dim=critic_hidden_dim,
            critic_depth=critic_depth,
            sorting_strategy="nearest"
        )

        if checkpoint_type == "ppo":
            agent, checkpoint_config = load_ppo_checkpoint(checkpoint_path, config, device)
            if checkpoint_config:
                for key in [
                    "max_visible_disasters",
                    "actor_hidden_dim",
                    "actor_depth",
                    "actor_dropout",
                    "critic_hidden_dim",
                    "critic_depth",
                ]:
                    if key in checkpoint_config:
                        setattr(config, key, checkpoint_config[key])

            evaluation = evaluate_agent(
                agent,
                config,
                device,
                deterministic=deterministic,
                eval_seeds=seeds,
            )
        elif checkpoint_type == "ml":
            policy = TrainedDispatchPolicy.load(
                checkpoint_path,
                device=device_name if device_name != "auto" else select_device(),
            )
            config.max_visible_disasters = policy.max_visible_disasters
            evaluation = evaluate_dispatch_policy(
                policy,
                config,
                device,
                deterministic=deterministic,
                eval_seeds=seeds,
            )
        else:
            raise ValueError(f"Unsupported model type: {checkpoint_type}")

        scenario_payload = {
            "scenario": scenario_name,
            "checkpoint": str(checkpoint_path),
            "checkpoint_type": checkpoint_type,
            "device": device_name,
            "seeds": seeds,
            "success_rate": float(evaluation["success_rate"]),
            "avg_objective_score": float(evaluation["avg_objective_score"]),
            "avg_total_reward": float(evaluation["avg_total_reward"]),
            "episodes": evaluation["episodes"],
        }
        scenario_results.append(scenario_payload)

        for episode in evaluation["episodes"]:
            summary_rows.append(
                {
                    "scenario": scenario_name,
                    "seed": int(episode["seed"]),
                    "success": bool(episode["success"]),
                    "objective_score": float(episode.get("objective_score", 0.0)),
                    "total_reward": float(episode.get("total_reward", 0.0)),
                    "terminated": bool(episode.get("terminated", False)),
                    "truncated": bool(episode.get("truncated", False)),
                    "last_outcome": episode.get("last_outcome"),
                }
            )

    save_json(output_dir / "evaluation_summary.json", {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_type": model_type,
        "device": device_name,
        "deterministic": deterministic,
        "scenarios": scenario_results,
    })
    save_summary_csv(output_dir / "evaluation_summary.csv", summary_rows)

    for scenario_payload in scenario_results:
        scenario_name = scenario_payload["scenario"]
        save_json(output_dir / f"evaluation_{scenario_name}.json", scenario_payload)

    return {
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint_path),
        "scenarios": [s["scenario"] for s in scenario_results],
        "results": scenario_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a PPO critic checkpoint or a ml_dispatch model on medium-winter and hard-winter scenarios."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained checkpoint.pt file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run evaluation on (cuda, mps, cpu, or auto).",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=DEFAULT_SCENARIOS,
        help="Scenario names to evaluate.",
    )
    parser.add_argument(
        "--eval-seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Seeds to run for each scenario.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiment_results/evaluations",
        help="Directory to save evaluation results.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic action selection during evaluation.",
        default=True
    )
    parser.add_argument(
        "--max-visible-disasters",
        type=int,
        default=5,
        help="Maximum visible disasters for the evaluation environment.",
    )
    parser.add_argument(
        "--actor-hidden-dim",
        type=int,
        default=256,
        help="Actor hidden dimension, used if checkpoint config is unavailable.",
    )
    parser.add_argument(
        "--actor-depth",
        type=int,
        default=3,
        help="Actor depth, used if checkpoint config is unavailable.",
    )
    parser.add_argument(
        "--actor-dropout",
        type=float,
        default=0.1,
        help="Actor dropout, used if checkpoint config is unavailable.",
    )
    parser.add_argument(
        "--critic-hidden-dim",
        type=int,
        default=256,
        help="Critic hidden dimension, used if checkpoint config is unavailable.",
    )
    parser.add_argument(
        "--critic-depth",
        type=int,
        default=3,
        help="Critic depth, used if checkpoint config is unavailable.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["auto", "ppo", "ml"],
        default="auto",
        help="Model type to evaluate. Use auto to detect checkpoint format.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print()
    output_dir = Path(args.output_dir) / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    result = run_evaluation(
        checkpoint_path=checkpoint_path,
        model_type=args.model_type,
        scenarios=args.scenarios,
        seeds=args.eval_seeds,
        device_name=args.device,
        output_dir=output_dir,
        deterministic=args.deterministic,
        max_visible_disasters=args.max_visible_disasters,
        actor_hidden_dim=args.actor_hidden_dim,
        actor_depth=args.actor_depth,
        actor_dropout=args.actor_dropout,
        critic_hidden_dim=args.critic_hidden_dim,
        critic_depth=args.critic_depth,
    )
    # print(json.dumps(result, indent=2, ensure_ascii=False))
    print_summary_table(result)


def print_summary_table(result: dict[str, Any]) -> None:
    rows = []
    for scenario_payload in result["results"]:
        rows.append(
            {
                "scenario": scenario_payload["scenario"],
                "success_rate": f"{scenario_payload['success_rate']:.3f}",
                "avg_objective_score": f"{scenario_payload['avg_objective_score']:.2f}",
                "avg_total_reward": f"{scenario_payload['avg_total_reward']:.2f}",
                "episodes": len(scenario_payload["episodes"]),
            }
        )

    if not rows:
        print("No evaluation results to display.")
        return

    headers = ["scenario", "success_rate", "avg_objective_score", "avg_total_reward", "episodes"]
    col_widths = {header: max(len(header), max(len(str(row[header])) for row in rows)) for header in headers}

    header_line = " | ".join(header.ljust(col_widths[header]) for header in headers)
    separator_line = "-+-".join("-" * col_widths[header] for header in headers)
    print("\nEvaluation summary by scenario:")
    print(header_line)
    print(separator_line)
    for row in rows:
        print(" | ".join(str(row[header]).ljust(col_widths[header]) for header in headers))


if __name__ == "__main__":
    main()
