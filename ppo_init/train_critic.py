from __future__ import annotations
from SimPyTest.gym import CURRENT_RESOURCE_FEATURES, DISASTER_FEATURES, GLOBAL_STATE_FEATURES, DisasterResponseGym, ObsType

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import time
from typing import Any
import statistics

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from benchmark import create_scenario_config
from mlp.ml_dispatch import DispatchScoringConfig, DispatchScoringModel
import matplotlib.pyplot as plt

ROLLOUT_RESET_SEED_INDEX = 0


@dataclass
class PPOConfig:
    scenario_name: str = "clatsop_landslide_ops"
    max_visible_disasters: int = 5
    sorting_strategy: str = "most_progress"
    total_episodes: int = 50
    rollout_steps: int = 1024
    epochs: int = 3
    minibatch_size: int = 256
    gamma: float = 0.985
    gae_lambda: float = 0.92
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.005
    value_coef: float = 0.5
    max_grad_norm: float = 0.3
    learning_rate: float = 1e-4
    scheduler_step_size: int = 40
    scheduler_gamma: float = 0.9
    seed: int = 0
    device: str = "cuda"
    actor_hidden_dim: int = 256
    actor_depth: int = 3
    actor_dropout: float = 0.1
    critic_hidden_dim: int = 256
    critic_depth: int = 3
    freeze_actor_updates: int = 25
    log_interval: int = 2
    save_dir: str = "experiment_results/init_critic"
    actor_checkpoint: str | None = None
    ppo_init_checkpoint: str | None = None
    test_seeds: list[int] = field(default_factory=lambda: list(range(80, 90)))
    evaluation_interval: int = 5
    evaluation_seeds: list[int] = field(default_factory=lambda: [101, 102, 103, 104])
    rollout_reset_seeds: list[int] = field(default_factory=lambda: list(range(1000, 9000)))
    normalize_returns: bool = True


@dataclass
class RolloutBatch:
    current_resource: torch.Tensor
    global_state: torch.Tensor
    candidate_disasters: torch.Tensor
    valid_actions: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


class DispatchActor(nn.Module):
    """Candidate-scoring actor compatible with BC dispatch model checkpoints."""

    def __init__(self, hidden_dim: int, depth: int, dropout: float):
        super().__init__()
        config: DispatchScoringConfig = DispatchScoringConfig(
            current_dim=CURRENT_RESOURCE_FEATURES,
            global_dim=GLOBAL_STATE_FEATURES,
            candidate_dim=DISASTER_FEATURES,
            output_dim=1,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
        )
        self.scorer = DispatchScoringModel(config)

    def forward(self, current_resource: torch.Tensor, global_state: torch.Tensor, candidate_disasters: torch.Tensor) -> torch.Tensor:
        return self.scorer(current_resource, global_state, candidate_disasters)


class Critic(nn.Module):
    def __init__(self, hidden_dim: int, depth: int, max_visible_disasters: int):
        super().__init__()
        input_dim = CURRENT_RESOURCE_FEATURES + GLOBAL_STATE_FEATURES + (max_visible_disasters * DISASTER_FEATURES) + max_visible_disasters
        layers: list[nn.Module] = []
        next_in = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(next_in, hidden_dim))
            layers.append(nn.ReLU())
            next_in = hidden_dim
        layers.append(nn.Linear(next_in, 1))
        self.model = nn.Sequential(*layers)

    def forward(
        self,
        current_resource: torch.Tensor,
        global_state: torch.Tensor,
        candidate_disasters: torch.Tensor,
        valid_actions: torch.Tensor,
    ) -> torch.Tensor:
        flat_candidates = candidate_disasters.reshape(candidate_disasters.shape[0], -1)
        features = torch.cat((current_resource, global_state, flat_candidates, valid_actions), dim=1)
        return self.model(features).squeeze(-1)


class PPOAgent(nn.Module):
    def __init__(self, config: PPOConfig):
        super().__init__()
        self.actor = DispatchActor(config.actor_hidden_dim, config.actor_depth, config.actor_dropout)
        self.critic = Critic(config.critic_hidden_dim, config.critic_depth, config.max_visible_disasters)

    def action_distribution(
        self,
        current_resource: torch.Tensor,
        global_state: torch.Tensor,
        candidate_disasters: torch.Tensor,
        valid_actions: torch.Tensor,
    ) -> Categorical:
        logits = self.actor(current_resource, global_state, candidate_disasters)
        masked_logits = logits.masked_fill(valid_actions <= 0, -1e9)
        return Categorical(logits=masked_logits)

    def value(
        self,
        current_resource: torch.Tensor,
        global_state: torch.Tensor,
        candidate_disasters: torch.Tensor,
        valid_actions: torch.Tensor,
    ) -> torch.Tensor:
        return self.critic(current_resource, global_state, candidate_disasters, valid_actions)


class TransitionBuffer:
    def __init__(self) -> None:
        self.current_resource: list[np.ndarray[Any, np.dtype[np.float32]]] = []
        self.global_state: list[np.ndarray[Any, np.dtype[np.float32]]] = []
        self.candidate_disasters: list[np.ndarray[Any, np.dtype[np.float32]]] = []
        self.valid_actions: list[np.ndarray[Any, np.dtype[np.float32]]] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.dones: list[float] = []
        self.values: list[float] = []
        self.log_probs: list[float] = []

    def clear(self) -> None:
        self.__init__()


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def obs_to_tensors(observation: ObsType, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    current_resource = torch.from_numpy(observation["current_resource"]).float().unsqueeze(0).to(device)
    global_state = torch.from_numpy(observation["global_state"]).float().unsqueeze(0).to(device)
    candidate_disasters = torch.from_numpy(observation["candidate_disasters"]).float().unsqueeze(0).to(device)
    valid_actions = torch.from_numpy(observation["valid_actions"].astype(np.float32)).float().unsqueeze(0).to(device)
    return current_resource, global_state, candidate_disasters, valid_actions


def make_env(config: PPOConfig, controller_name: str) -> DisasterResponseGym:
    return DisasterResponseGym(
        max_visible_disasters=config.max_visible_disasters,
        sorting_strategy=config.sorting_strategy,
        scenario_config=create_scenario_config(config.scenario_name, gis_config=None),
        controller_name=controller_name,
        scenario_name=config.scenario_name,
    )


def load_actor_checkpoint(actor: DispatchActor, checkpoint_path: str, device: torch.device) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = payload["state_dict"]
    actor.scorer.load_state_dict(state_dict)
    return dict(payload.get("metadata", {}))


def load_train_critic_checkpoint(agent: PPOAgent, checkpoint_path: str, device: torch.device) -> dict[str, object]:
    payload = torch.load(checkpoint_path, map_location=device)
    agent.actor.load_state_dict(payload["actor_state_dict"])
    agent.critic.load_state_dict(payload["critic_state_dict"])
    return {
        "episodes_completed": payload.get("episodes_completed"),
        # "actor_metadata": payload.get("actor_metadata", {}),
        "config": payload.get("config", {}),
    }


def predict_action(agent: PPOAgent, observation: ObsType, device: torch.device, deterministic: bool) -> int:
    with torch.no_grad():
        cr, gs, cd, va = obs_to_tensors(observation, device)
        dist = agent.action_distribution(cr, gs, cd, va)
        if deterministic:
            action = int(torch.argmax(dist.logits, dim=-1).item())
        else:
            action = int(dist.sample().item())
    return action


def evaluate_agent(
    agent: PPOAgent,
    config: PPOConfig,
    device: torch.device,
    deterministic: bool = True,
    eval_seeds: list[int] | None = None,
) -> dict[str, Any]:
    agent.eval()
    episodes: list[dict[str, float | bool | int | str | None]] = []
    seeds = eval_seeds if eval_seeds is not None else config.test_seeds
    for seed in seeds:
        env = make_env(config, controller_name="ppo_custom_checkpoint")
        observation, _ = env.reset(seed=seed)
        terminated = False
        truncated = False
        total_reward = 0.0
        info: dict[str, Any] = {}

        while not terminated and not truncated:
            action = predict_action(agent, observation, device, deterministic)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
        episodes.append(
            {
                "seed": int(seed),
                "success": bool(info["is_success"]),
                "objective_score": float(info.get("objective_score", 0.0)),
                "total_reward": total_reward,
                # "reward": total_reward,
                # "terminal_outcome": info["terminal_outcome"],
            }
        )
        # print(f"Checkpoint eval seed={seed} success={bool(info['is_success'])}")

    successes = [episode for episode in episodes if bool(episode["success"])]
    scores = [float(episode.get("objective_score", 0)) for episode in episodes if episode.get("success")]

    rewards = [float(episode.get("total_reward", 0)) for episode in episodes if episode.get("success")]
    agent.train()
    return {
        "episodes": episodes,
        "success_rate": len(successes) / len(episodes) if episodes else 0.0,
        "avg_objective_score": statistics.mean(scores) if scores else 0,
        "avg_total_reward": statistics.mean(rewards) if rewards else 0,
    }


def collect_rollout(
    agent: PPOAgent,
    env: DisasterResponseGym,
    buffer: TransitionBuffer,
    config: PPOConfig,
    device: torch.device,
    current_obs: ObsType,
) -> tuple[ObsType, float, dict[str, float]]:
    global ROLLOUT_RESET_SEED_INDEX

    for _ in range(config.rollout_steps):
        with torch.no_grad():
            cr, gs, cd, va = obs_to_tensors(current_obs, device)
            dist = agent.action_distribution(cr, gs, cd, va)
            value = agent.value(cr, gs, cd, va)
            action = int(dist.sample().item())
            log_prob = float(dist.log_prob(torch.tensor([action], device=device)).item())
            value_scalar = float(value.item())

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        buffer.current_resource.append(current_obs["current_resource"].copy())
        buffer.global_state.append(current_obs["global_state"].copy())
        buffer.candidate_disasters.append(current_obs["candidate_disasters"].copy())
        buffer.valid_actions.append(current_obs["valid_actions"].astype(np.float32))
        buffer.actions.append(action)
        buffer.rewards.append(float(reward))
        buffer.dones.append(float(done))
        buffer.values.append(value_scalar)
        buffer.log_probs.append(log_prob)

        current_obs = next_obs

        if done:
            reset_seed = int(config.rollout_reset_seeds[ROLLOUT_RESET_SEED_INDEX])
            ROLLOUT_RESET_SEED_INDEX += 1
            current_obs, _ = env.reset(seed=int(reset_seed))

    return current_obs


def build_training_batch(buffer: TransitionBuffer, agent: PPOAgent, last_obs: ObsType, device: torch.device, config: PPOConfig) -> RolloutBatch:
    with torch.no_grad():
        cr, gs, cd, va = obs_to_tensors(last_obs, device)
        last_value = float(agent.value(cr, gs, cd, va).item())

    rewards = np.asarray(buffer.rewards, dtype=np.float32)
    dones = np.asarray(buffer.dones, dtype=np.float32)
    values = np.asarray(buffer.values + [last_value], dtype=np.float32)

    advantages = np.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + config.gamma * values[t + 1] * non_terminal - values[t]
        gae = delta + config.gamma * config.gae_lambda * non_terminal * gae
        advantages[t] = gae

    returns = advantages + values[:-1]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return RolloutBatch(
        current_resource=torch.tensor(np.asarray(buffer.current_resource), dtype=torch.float32, device=device),
        global_state=torch.tensor(np.asarray(buffer.global_state), dtype=torch.float32, device=device),
        candidate_disasters=torch.tensor(np.asarray(buffer.candidate_disasters), dtype=torch.float32, device=device),
        valid_actions=torch.tensor(np.asarray(buffer.valid_actions), dtype=torch.float32, device=device),
        actions=torch.tensor(buffer.actions, dtype=torch.long, device=device),
        old_log_probs=torch.tensor(buffer.log_probs, dtype=torch.float32, device=device),
        returns=torch.tensor(returns, dtype=torch.float32, device=device),
        advantages=torch.tensor(advantages, dtype=torch.float32, device=device),
    )


def ppo_update(
    agent: PPOAgent,
    optimizer: Adam,
    batch: RolloutBatch,
    config: PPOConfig,
    updates_done: int,
) -> dict[str, float]:
    total_samples = batch.actions.shape[0]
    indices = np.arange(total_samples)
    policy_loss_total = 0.0
    value_loss_total = 0.0
    entropy_total = 0.0
    batches = 0

    return_mean = batch.returns.mean()
    return_std = batch.returns.std().clamp_min(1e-8)

    for _epoch in range(config.epochs):
        np.random.shuffle(indices)
        for start in range(0, total_samples, config.minibatch_size):
            end = min(start + config.minibatch_size, total_samples)
            mb_idx = indices[start:end]

            dist = agent.action_distribution(
                batch.current_resource[mb_idx],
                batch.global_state[mb_idx],
                batch.candidate_disasters[mb_idx],
                batch.valid_actions[mb_idx],
            )
            new_log_probs = dist.log_prob(batch.actions[mb_idx])
            entropy = dist.entropy().mean()
            values = agent.value(
                batch.current_resource[mb_idx],
                batch.global_state[mb_idx],
                batch.candidate_disasters[mb_idx],
                batch.valid_actions[mb_idx],
            )

            ratio = torch.exp(new_log_probs - batch.old_log_probs[mb_idx])
            surr1 = ratio * batch.advantages[mb_idx]
            surr2 = torch.clamp(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon) * batch.advantages[mb_idx]
            policy_loss = -torch.min(surr1, surr2).mean()

            value_targets = batch.returns[mb_idx]
            if config.normalize_returns:
                value_targets = (value_targets - return_mean) / return_std
                values_for_loss = (values - return_mean) / return_std
            else:
                values_for_loss = values
            value_loss = torch.nn.functional.mse_loss(values_for_loss, value_targets)

            if updates_done <= config.freeze_actor_updates:
                loss = config.value_coef * value_loss
            else:
                loss = policy_loss + (config.value_coef * value_loss) - (config.entropy_coef * entropy)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
            optimizer.step()

            policy_loss_total += float(policy_loss.item())
            value_loss_total += float(value_loss.item())
            entropy_total += float(entropy.item())
            batches += 1

    denom = max(batches, 1)
    return {
        "policy_loss": policy_loss_total / denom,
        "value_loss": value_loss_total / denom,
        "entropy": entropy_total / denom,
    }


def save_checkpoint(
    output_dir: Path,
    agent: PPOAgent,
    optimizer: Adam,
    config: PPOConfig,
    step: int,
    actor_metadata: dict[str, Any],
) -> None:
    payload = {
        "step": step,
        "config": asdict(config),
        "actor_state_dict": agent.actor.state_dict(),
        "critic_state_dict": agent.critic.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "actor_metadata": actor_metadata,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    torch.save(payload, output_dir / "checkpoint.pt")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def save_loss_plot(
    training_log: list[dict[str, Any]],
    loss_key: str,
    title: str,
    output_path: Path,
    color: str,
) -> None:
    iterations = [int(record["episodes"]) for record in training_log]
    losses = [float(record[loss_key]) for record in training_log]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iterations, losses, label=loss_key, color=color, linewidth=1.4)
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_evaluation_plot(
    evaluation_log: list[dict[str, Any]],
    value_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
    color: str,
) -> None:
    if not evaluation_log:
        return

    iterations = [int(record["iteration"]) for record in evaluation_log]
    values = [float(record[value_key]) for record in evaluation_log]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iterations, values, label=value_key, color=color, linewidth=1.4, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def train(config: PPOConfig) -> Path:
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    train_started_at_utc = datetime.now(timezone.utc)
    train_started_perf = time.perf_counter()
    device_name = config.device if config.device != "auto" else select_device()
    device = torch.device(device_name)

    run_dir = Path(config.save_dir) / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(config, controller_name="ppo_custom")
    obs, _ = env.reset(seed=config.seed)

    agent = PPOAgent(config).to(device)
    actor_metadata: dict[str, Any] = {}
    if config.actor_checkpoint:
        actor_metadata = load_actor_checkpoint(agent.actor, config.actor_checkpoint, device)

    if config.ppo_init_checkpoint:
        checkpoint_info = load_train_critic_checkpoint(agent, config.ppo_init_checkpoint, device)
        # print(f"Initialized PPO agent from checkpoint with metadata: {checkpoint_info.get('config', {})}")

    optimizer = Adam(agent.parameters(), lr=config.learning_rate)
    scheduler = StepLR(optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)
    buffer = TransitionBuffer()

    # timestep = 0
    episodes_completed = 0
    training_log: list[dict[str, float]] = []
    evaluation_log: list[dict[str, Any]] = []

    while episodes_completed < config.total_episodes:
        buffer.clear()
        obs = collect_rollout(agent, env, buffer, config, device, obs)
        # timestep += config.rollout_steps
        episodes_completed += 1

        batch = build_training_batch(buffer, agent, obs, device, config)
        losses = ppo_update(agent, optimizer, batch, config, episodes_completed)
        scheduler.step()

        record = {
            "episodes": float(episodes_completed),
            "policy_loss": losses["policy_loss"],
            "value_loss": losses["value_loss"],
            "entropy": losses["entropy"],
        }
        training_log.append(record)

        if episodes_completed % config.evaluation_interval == 0:
            periodic_eval = evaluate_agent(
                agent,
                config,
                device,
                deterministic=True,
                eval_seeds=config.evaluation_seeds,
            )
            evaluation_log.append(
                {
                    "iteration": episodes_completed,
                    "seeds": list(config.evaluation_seeds),
                    "episodes": [
                        {
                            "seed": int(ep["seed"]),
                            "object_score": float(ep["objective_score"]),
                            "reward": float(ep["total_reward"]),
                            "success": bool(ep["success"]),
                        }
                        for ep in periodic_eval["episodes"]
                    ],
                    "avg_object_score": float(periodic_eval["avg_objective_score"]),
                    "avg_reward": float(periodic_eval["avg_total_reward"]),
                    "success_rate": float(periodic_eval["success_rate"]),
                }
            )
            print(
                f"Evaluation {episodes_completed}| "
                f"obj={periodic_eval['avg_objective_score']:.2f} | "
                f"success={periodic_eval['success_rate'] * 100:.1f}% | "
                f"reward={periodic_eval['avg_total_reward']:.2f}"
            )

        if episodes_completed % config.log_interval == 0:
            print(f"PPO custom| " f"episodes={episodes_completed} | " f"pi_loss={record['policy_loss']:.4f} | " f"vf_loss={record['value_loss']:.4f} | " f"entropy={record['entropy']:.4f} | ")

    total_wall_time_s = time.perf_counter() - train_started_perf
    train_finished_at_utc = datetime.now(timezone.utc)

    checkpoint_metrics = evaluate_agent(agent, config, device, deterministic=True)
    save_checkpoint(run_dir, agent, optimizer, config, episodes_completed, actor_metadata)

    save_loss_plot(
        training_log=training_log,
        loss_key="policy_loss",
        title="Policy loss trend during training",
        output_path=run_dir / "policy_loss_curve.png",
        color="tab:orange",
    )
    save_loss_plot(
        training_log=training_log,
        loss_key="value_loss",
        title="Value loss trend during training",
        output_path=run_dir / "value_loss_curve.png",
        color="tab:green",
    )
    save_evaluation_plot(
        evaluation_log=evaluation_log,
        value_key="avg_reward",
        title="Average evaluation reward during training",
        ylabel="Average reward",
        output_path=run_dir / "evaluation_avg_reward_curve.png",
        color="tab:red",
    )
    save_evaluation_plot(
        evaluation_log=evaluation_log,
        value_key="avg_object_score",
        title="Average evaluation object score during training",
        ylabel="Average object score",
        output_path=run_dir / "evaluation_avg_object_curve.png",
        color="tab:purple",
    )
    save_evaluation_plot(
        evaluation_log=evaluation_log,
        value_key="success_rate",
        title="Evaluation success rate during training",
        ylabel="Success rate",
        output_path=run_dir / "evaluation_success_rate_curve.png",
        color="tab:blue",
    )
    write_json(
        run_dir / "training_metrics.json",
        {
            "config": asdict(config),
            "actor_checkpoint": config.actor_checkpoint,
            "device": device_name,
            "train_started_at_utc": train_started_at_utc.isoformat(),
            "train_finished_at_utc": train_finished_at_utc.isoformat(),
            "total_wall_time_s": total_wall_time_s,
            "total_wall_time_hms": format_duration(total_wall_time_s),
            "checkpoint_metrics": checkpoint_metrics,
            "evaluation_log": evaluation_log,
            "training_log": training_log,
        },
    )
    print(
        "Checkpoint | " f"obj={checkpoint_metrics['avg_objective_score']:.2f} | " f"success={checkpoint_metrics['success_rate'] * 100:.1f}% | " f"reward={checkpoint_metrics['avg_total_reward']:.2f}"
    )
    print(f"Saved custom PPO artifacts to {run_dir}")
    print(f"Training wall time: {total_wall_time_s:.2f}s ({format_duration(total_wall_time_s)})")
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a custom PPO agent on DisasterResponseGym with optional actor initialization.")
    parser.add_argument("--scenario-name", type=str, default=PPOConfig.scenario_name)
    parser.add_argument("--max-visible-disasters", type=int, default=PPOConfig.max_visible_disasters)
    parser.add_argument("--sorting-strategy", type=str, default=PPOConfig.sorting_strategy)
    parser.add_argument("--episodes", type=int, default=PPOConfig.total_episodes)
    parser.add_argument("--rollout-steps", type=int, default=PPOConfig.rollout_steps)
    parser.add_argument("--epochs", type=int, default=PPOConfig.epochs)
    parser.add_argument("--minibatch-size", type=int, default=PPOConfig.minibatch_size)
    parser.add_argument("--learning-rate", type=float, default=PPOConfig.learning_rate)
    parser.add_argument("--gamma", type=float, default=PPOConfig.gamma)
    parser.add_argument("--gae-lambda", type=float, default=PPOConfig.gae_lambda)
    parser.add_argument("--clip-epsilon", type=float, default=PPOConfig.clip_epsilon)
    parser.add_argument("--entropy-coef", type=float, default=PPOConfig.entropy_coef)
    parser.add_argument("--value-coef", type=float, default=PPOConfig.value_coef)
    parser.add_argument("--max-grad-norm", type=float, default=PPOConfig.max_grad_norm)
    parser.add_argument("--seed", type=int, default=PPOConfig.seed)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--actor-hidden-dim", type=int, default=PPOConfig.actor_hidden_dim)
    parser.add_argument("--actor-depth", type=int, default=PPOConfig.actor_depth)
    parser.add_argument("--actor-dropout", type=float, default=PPOConfig.actor_dropout)
    parser.add_argument("--critic-hidden-dim", type=int, default=PPOConfig.critic_hidden_dim)
    parser.add_argument("--critic-depth", type=int, default=PPOConfig.critic_depth)
    parser.add_argument("--freeze-actor-updates", type=int, default=PPOConfig.freeze_actor_updates)
    parser.add_argument("--log-interval", type=int, default=PPOConfig.log_interval)
    parser.add_argument("--save-dir", type=str, default=PPOConfig.save_dir)
    parser.add_argument("--actor-checkpoint", type=str, default=None, help="Path to dispatch_model.pt from classification training.")
    parser.add_argument("--evaluation-interval", type=int, default=PPOConfig.evaluation_interval)
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", default=PPOConfig.evaluation_seeds)
    args = parser.parse_args()

    config = PPOConfig(
        scenario_name=args.scenario_name,
        max_visible_disasters=args.max_visible_disasters,
        sorting_strategy=args.sorting_strategy,
        total_episodes=args.episodes,
        rollout_steps=args.rollout_steps,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        device=select_device() if args.device == "auto" else args.device,
        actor_hidden_dim=args.actor_hidden_dim,
        actor_depth=args.actor_depth,
        actor_dropout=args.actor_dropout,
        critic_hidden_dim=args.critic_hidden_dim,
        critic_depth=args.critic_depth,
        freeze_actor_updates=args.freeze_actor_updates,
        log_interval=args.log_interval,
        save_dir=args.save_dir,
        actor_checkpoint=args.actor_checkpoint,
        evaluation_interval=args.evaluation_interval,
        evaluation_seeds=args.evaluation_seeds,
    )

    train(config)
