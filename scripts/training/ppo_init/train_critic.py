from __future__ import annotations
from SimPyTest.gym import CURRENT_RESOURCE_FEATURES, DISASTER_FEATURES, GLOBAL_STATE_FEATURES, DisasterResponseGym, InfoType, ObsType

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

from benchmark import create_scenario_config
from SimPyTest.gym import (
    CURRENT_RESOURCE_FEATURES,
    DISASTER_FEATURES,
    GLOBAL_STATE_FEATURES,
    DisasterResponseGym,
    ObsType,
)
from scripts.training.mlp.ml_dispatch import DispatchScoringConfig, DispatchScoringModel


@dataclass
class PPOConfig:
    scenario_name: str = "clatsop_landslide_curriculum"
    max_visible_disasters: int = 5
    sorting_strategy: str = "most_progress"
    total_timesteps: int = 20_000
    rollout_steps: int = 512
    epochs: int = 10
    minibatch_size: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.005
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 1e-3
    seed: int = 0
    device: str = "cpu"
    actor_hidden_dim: int = 256
    actor_depth: int = 3
    actor_dropout: float = 0.1
    critic_hidden_dim: int = 256
    critic_depth: int = 3
    freeze_actor_updates: int = 0
    log_interval: int = 1024
    save_dir: str = "experiment_results/ppo_custom"
    actor_checkpoint: str | None = None
    checkpoint_seeds: list[int] = field(default_factory=lambda: list(range(80, 100)))


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
        config = DispatchScoringConfig(
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


def predict_action(agent: PPOAgent, observation: ObsType, device: torch.device, deterministic: bool) -> int:
    with torch.no_grad():
        cr, gs, cd, va = obs_to_tensors(observation, device)
        dist = agent.action_distribution(cr, gs, cd, va)
        if deterministic:
            action = int(torch.argmax(dist.logits, dim=-1).item())
        else:
            action = int(dist.sample().item())
    return action


def evaluate_agent(agent: PPOAgent, config: PPOConfig, device: torch.device, deterministic: bool = True) -> dict[str, Any]:
    episodes: list[dict[str, float | bool | int | str | None]] = []
    for seed in config.checkpoint_seeds:
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
                "objective_score": float(info["objective_score"]),
                "total_reward": total_reward,
                "terminal_outcome": info["terminal_outcome"],
            }
        )
        print(f"Checkpoint eval seed={seed} success={bool(info['is_success'])}")

    successes = [episode for episode in episodes if bool(episode["success"])]
    return {
        "episodes": episodes,
        "success_rate": len(successes) / len(episodes) if episodes else 0.0,
        "avg_objective_score": statistics.mean(float(episode["objective_score"]) for episode in episodes) if episodes else 0.0,
        "avg_total_reward": statistics.mean(float(episode["total_reward"]) for episode in episodes) if episodes else 0.0,
    }


def collect_rollout(
    agent: PPOAgent,
    env: DisasterResponseGym,
    buffer: TransitionBuffer,
    config: PPOConfig,
    device: torch.device,
    current_obs: ObsType,
    episode_reward: float,
) -> tuple[ObsType, float, dict[str, float]]:
    episodes_finished = 0
    episode_rewards: list[float] = []

    for _ in range(config.rollout_steps):
        with torch.no_grad():
            cr, gs, cd, va = obs_to_tensors(current_obs, device)
            dist = agent.action_distribution(cr, gs, cd, va)
            value = agent.value(cr, gs, cd, va)
            action = int(dist.sample().item())
            log_prob = float(dist.log_prob(torch.tensor([action], device=device)).item())
            value_scalar = float(value.item())

        next_obs, reward, terminated, truncated, _info = env.step(action)
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

        episode_reward += float(reward)
        current_obs = next_obs

        if done:
            episodes_finished += 1
            episode_rewards.append(episode_reward)
            current_obs, _ = env.reset(seed=int(np.random.randint(0, 2**31 - 1)))
            episode_reward = 0.0

    stats = {
        "episodes_finished": float(episodes_finished),
        "mean_episode_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
    }
    return current_obs, episode_reward, stats


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
            value_loss = torch.nn.functional.mse_loss(values, batch.returns[mb_idx])

            if updates_done < config.freeze_actor_updates:
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

    optimizer = Adam(agent.parameters(), lr=config.learning_rate)
    buffer = TransitionBuffer()

    timestep = 0
    update_idx = 0
    episode_reward = 0.0
    training_log: list[dict[str, float]] = []

    while timestep < config.total_timesteps:
        buffer.clear()
        obs, episode_reward, rollout_stats = collect_rollout(agent, env, buffer, config, device, obs, episode_reward)
        timestep += config.rollout_steps

        batch = build_training_batch(buffer, agent, obs, device, config)
        losses = ppo_update(agent, optimizer, batch, config, update_idx)
        update_idx += 1

        record = {
            "timestep": float(timestep),
            "update": float(update_idx),
            "policy_loss": losses["policy_loss"],
            "value_loss": losses["value_loss"],
            "entropy": losses["entropy"],
            "episodes_finished": rollout_stats["episodes_finished"],
            "mean_episode_reward": rollout_stats["mean_episode_reward"],
        }
        training_log.append(record)

        if timestep % config.log_interval == 0:
            print(
                "PPO custom | "
                f"steps={timestep} | "
                f"update={update_idx} | "
                f"pi_loss={record['policy_loss']:.4f} | "
                f"vf_loss={record['value_loss']:.4f} | "
                f"entropy={record['entropy']:.4f} | "
                f"episodes={int(record['episodes_finished'])} | "
                f"mean_ep_reward={record['mean_episode_reward']:.2f}"
            )

    total_wall_time_s = time.perf_counter() - train_started_perf
    train_finished_at_utc = datetime.now(timezone.utc)
    checkpoint_metrics = evaluate_agent(agent, config, device, deterministic=True)
    save_checkpoint(run_dir, agent, optimizer, config, timestep, actor_metadata)
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
            "training_log": training_log,
        },
    )
    print(
        "Checkpoint | "
        f"obj={checkpoint_metrics['avg_objective_score']:.2f} | "
        f"success={checkpoint_metrics['success_rate'] * 100:.1f}% | "
        f"reward={checkpoint_metrics['avg_total_reward']:.2f}"
    )
    print(f"Saved custom PPO artifacts to {run_dir}")
    print(f"Training wall time: {total_wall_time_s:.2f}s ({format_duration(total_wall_time_s)})")
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a custom PPO agent on DisasterResponseGym with optional actor initialization.")
    parser.add_argument("--scenario-name", type=str, default=PPOConfig.scenario_name)
    parser.add_argument("--max-visible-disasters", type=int, default=PPOConfig.max_visible_disasters)
    parser.add_argument("--sorting-strategy", type=str, default=PPOConfig.sorting_strategy)
    parser.add_argument("--timesteps", type=int, default=PPOConfig.total_timesteps)
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
    args = parser.parse_args()

    config = PPOConfig(
        scenario_name=args.scenario_name,
        max_visible_disasters=args.max_visible_disasters,
        sorting_strategy=args.sorting_strategy,
        total_timesteps=args.timesteps,
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
    )
    train(config)
