from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import statistics
import time
from typing import Any, TypedDict, cast

from sb3_contrib import MaskablePPO
import torch
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from typing_extensions import override

from SimPyTest.evaluation import EVALUATION_PROTOCOL_VERSION, KPIBundle, compute_kpi_bundle
from SimPyTest.gym import CURRENT_RESOURCE_FEATURES, DISASTER_FEATURES, GLOBAL_STATE_FEATURES, DisasterResponseGym, InfoType, ObsType
from SimPyTest.simulation import Disaster

PPO_CHECKPOINT_VERSION = "v1"
TRAINING_SEEDS = list(range(64))
VALIDATION_SEEDS = list(range(80, 100))
CHECKPOINT_SEEDS = list(range(80, 100))
SCENARIO_NAME = "clatsop_landslide_curriculum"
MAX_VISIBLE_DISASTERS = 5
SORTING_STRATEGY = "most_progress"
TOTAL_TIMESTEPS = 20_000
EVAL_FREQUENCY = 10_000
EVAL_MAX_DECISIONS = 2_000
EVAL_MAX_WALL_TIME_S = 120.0


class EpisodeResult(TypedDict):
    seed: int
    success: bool
    sim_time: float
    wall_time_s: float
    total_reward: float
    objective_score: float
    terminal_outcome: str | None
    invalid_action_count: int
    invalid_action_remaps: int
    valid_action_ratio: float
    kpis: KPIBundle


@dataclass(frozen=True)
class PPOTrainingConfig:
    total_timesteps: int
    eval_frequency: int
    training_seeds: list[int]
    validation_seeds: list[int]
    checkpoint_seeds: list[int]
    scenario_name: str
    max_visible_disasters: int
    sorting_strategy: str
    device: str
    policy_kwargs: dict[str, Any]
    learning_rate: float
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    ent_coef: float
    vf_coef: float
    clip_range: float
    curriculum_stages: list["CurriculumStage"]


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    scenario_name: str
    timesteps: int


class SeedCyclingEnv(DisasterResponseGym):
    training_seeds: list[int]
    seed_index: int

    def __init__(self, training_seeds: list[int], controller_name: str, scenario_name: str):
        from benchmark import create_scenario_config

        self.training_seeds = list(training_seeds)
        self.seed_index = 0
        super().__init__(
            max_visible_disasters=MAX_VISIBLE_DISASTERS,
            sorting_strategy=SORTING_STRATEGY,
            scenario_config=create_scenario_config(scenario_name, gis_config=None),
            controller_name=controller_name,
            scenario_name=scenario_name,
        )
        self.action_space = spaces.Discrete(self.max_slots)
        self.observation_space = cast(
            spaces.Space[ObsType],
            spaces.Dict(
                {
                    "current_resource": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(CURRENT_RESOURCE_FEATURES,),
                        dtype=np.float32,
                    ),
                    "global_state": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(GLOBAL_STATE_FEATURES,),
                        dtype=np.float32,
                    ),
                    "candidate_disasters": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(self.max_slots, DISASTER_FEATURES),
                        dtype=np.float32,
                    ),
                    "valid_actions": spaces.Box(
                        low=0,
                        high=1,
                        shape=(self.max_slots,),
                        dtype=np.int8,
                    ),
                }
            ),
        )

    @override
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[ObsType, InfoType]:
        if seed is None:
            seed = self.training_seeds[self.seed_index % len(self.training_seeds)]
            self.seed_index += 1
        return super().reset(seed=seed, options=options)

    def _valid_actions(self) -> np.ndarray[tuple[int], np.dtype[np.int8]]:
        valid_actions = np.zeros(self.max_slots, dtype=np.int8)
        valid_actions[: len(self.current_candidates)] = 1
        return valid_actions

    def _zero_obs(self) -> ObsType:
        return {
            "current_resource": np.zeros(CURRENT_RESOURCE_FEATURES, dtype=np.float32),
            "global_state": np.zeros(GLOBAL_STATE_FEATURES, dtype=np.float32),
            "candidate_disasters": np.zeros((self.max_slots, DISASTER_FEATURES), dtype=np.float32),
            "valid_actions": np.zeros(self.max_slots, dtype=np.int8),
        }

    def _resolve_action(self, action: int) -> tuple[str, Disaster | None, bool]:
        if 0 <= action < len(self.current_candidates):
            return "DISPATCH", self.current_candidates[action], False
        return "INVALID", None, True


def build_training_config() -> PPOTrainingConfig:
    curriculum_stages = [
        CurriculumStage(name="stage_1_easy", scenario_name="clatsop_landslide_curriculum", timesteps=8_000),
        CurriculumStage(name="stage_2_small_ramp", scenario_name="clatsop_landslide_curriculum_stage2", timesteps=12_000),
    ]
    return PPOTrainingConfig(
        total_timesteps=TOTAL_TIMESTEPS,
        eval_frequency=EVAL_FREQUENCY,
        training_seeds=list(TRAINING_SEEDS),
        validation_seeds=list(VALIDATION_SEEDS),
        checkpoint_seeds=list(CHECKPOINT_SEEDS),
        scenario_name=SCENARIO_NAME,
        max_visible_disasters=MAX_VISIBLE_DISASTERS,
        sorting_strategy=SORTING_STRATEGY,
        device=select_device(),
        policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
        learning_rate=1e-3,
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.005,
        vf_coef=0.5,
        clip_range=0.2,
        curriculum_stages=curriculum_stages,
    )


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def create_env(seed: int, controller_name: str, scenario_name: str = SCENARIO_NAME) -> SeedCyclingEnv:
    return SeedCyclingEnv([seed], controller_name, scenario_name)


def create_training_env(training_seeds: list[int], controller_name: str, scenario_name: str) -> SeedCyclingEnv:
    return SeedCyclingEnv(training_seeds, controller_name, scenario_name)


def predict_action(model: object, env: SeedCyclingEnv, observation: ObsType, deterministic: bool) -> int:
    predictor = cast(Any, model).predict
    try:
        predicted = predictor(observation, deterministic=deterministic, action_masks=env.action_masks())
    except TypeError:
        predicted = predictor(observation, deterministic=deterministic)
    return int(predicted[0])


def run_policy_episode(model: object, seed: int, deterministic: bool, controller_name: str, scenario_name: str = SCENARIO_NAME) -> EpisodeResult:
    env = create_env(seed, controller_name, scenario_name)
    observation, info = env.reset(seed=seed)
    terminated = False
    truncated = False
    total_reward = 0.0
    total_valid_action_ratio = 0.0
    started_at = time.perf_counter()
    decisions = 0

    while not terminated and not truncated:
        action = predict_action(model, env, observation, deterministic)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        total_valid_action_ratio += float(info["valid_action_count"]) / float(env.max_slots)
        decisions += 1

        elapsed_wall_time_s = time.perf_counter() - started_at
        if decisions >= EVAL_MAX_DECISIONS or elapsed_wall_time_s >= EVAL_MAX_WALL_TIME_S:
            print("Evaluation cutoff | " f"seed={seed} | " f"decisions={decisions} | " f"wall={elapsed_wall_time_s:.1f}s | " f"objective={info['objective_score']:.2f}")
            terminated = True
            info["terminal_outcome"] = "EVAL_CUTOFF"
            info["is_success"] = False

    return {
        "seed": seed,
        "success": bool(info["is_success"]),
        "sim_time": float(info["summary"].non_idle_time),
        "wall_time_s": time.perf_counter() - started_at,
        "total_reward": total_reward,
        "objective_score": float(info["objective_score"]),
        "terminal_outcome": info["terminal_outcome"],
        "invalid_action_count": int(info["invalid_action_count"]),
        "invalid_action_remaps": int(info["invalid_action_remaps"]),
        "valid_action_ratio": total_valid_action_ratio / decisions if decisions > 0 else 0.0,
        "kpis": compute_kpi_bundle(info["summary"]),
    }


def evaluate_model(model: object, seeds: list[int], deterministic: bool, controller_name: str, scenario_name: str = SCENARIO_NAME) -> dict[str, Any]:
    # episodes = [run_policy_episode(model, seed, deterministic, controller_name) for seed in seeds]
    episodes = []
    for seed in seeds:
        episode = run_policy_episode(model, seed, deterministic, controller_name, scenario_name)
        episodes.append(episode)
        print(f"Evaluation seed {seed} success: {episode['success']}")
    successes = [episode for episode in episodes if episode["success"]]
    return {
        "episodes": episodes,
        "success_rate": len(successes) / len(episodes) if episodes else 0.0,
        "avg_objective_score": statistics.mean(episode["objective_score"] for episode in episodes) if episodes else 0.0,
        "avg_total_reward": statistics.mean(episode["total_reward"] for episode in episodes) if episodes else 0.0,
        "avg_sim_time": statistics.mean(episode["sim_time"] for episode in episodes) if episodes else 0.0,
        "avg_wall_time_s": statistics.mean(episode["wall_time_s"] for episode in episodes) if episodes else 0.0,
        "avg_invalid_action_count": statistics.mean(episode["invalid_action_count"] for episode in episodes) if episodes else 0.0,
        "avg_invalid_action_remaps": statistics.mean(episode["invalid_action_remaps"] for episode in episodes) if episodes else 0.0,
        "avg_valid_action_ratio": statistics.mean(episode["valid_action_ratio"] for episode in episodes) if episodes else 0.0,
        "successes": len(successes),
        "runs": len(episodes),
    }


def evaluate_curriculum_stages(model: object, config: PPOTrainingConfig, seeds: list[int], deterministic: bool, controller_name: str) -> dict[str, dict[str, Any]]:
    stage_metrics: dict[str, dict[str, Any]] = {}
    for stage in config.curriculum_stages:
        metrics = evaluate_model(
            model,
            seeds,
            deterministic,
            controller_name,
            stage.scenario_name,
        )
        metrics["stage_name"] = stage.name
        metrics["stage_scenario_name"] = stage.scenario_name
        stage_metrics[stage.name] = metrics
    return stage_metrics


class ValidationCallback(BaseCallback):
    def __init__(self, config: PPOTrainingConfig, run_dir: Path):
        super().__init__()
        self.config = config
        self.run_dir = run_dir
        self.best_success_rate = float("-inf")
        self.best_objective = float("-inf")
        self.validation_history: list[dict[str, Any]] = []
        self.stage_name = config.curriculum_stages[0].name
        self.stage_scenario_name = config.curriculum_stages[0].scenario_name

    def set_stage(self, stage: CurriculumStage) -> None:
        self.stage_name = stage.name
        self.stage_scenario_name = stage.scenario_name

    def run_validation(self) -> dict[str, Any]:
        print("Validation starting | " f"steps={self.num_timesteps} | " f"seeds={self.config.validation_seeds[0]}..{self.config.validation_seeds[-1]} | " f"runs={len(self.config.validation_seeds)}")
        metrics = evaluate_model(
            self.model,
            self.config.validation_seeds,
            True,
            "ppo_validation",
            self.stage_scenario_name,
        )
        metrics["timesteps"] = self.num_timesteps
        metrics["stage_name"] = self.stage_name
        metrics["stage_scenario_name"] = self.stage_scenario_name
        self.validation_history.append(metrics)
        print(
            "Validation | "
            f"stage={self.stage_name} | "
            f"steps={self.num_timesteps} | "
            f"obj={metrics['avg_objective_score']:.2f} | "
            f"success={metrics['success_rate'] * 100:.1f}% | "
            f"reward={metrics['avg_total_reward']:.2f} | "
            f"invalid={metrics['avg_invalid_action_count']:.2f} | "
            f"remaps={metrics['avg_invalid_action_remaps']:.2f} | "
            f"valid_ratio={metrics['avg_valid_action_ratio']:.3f}"
        )

        success_rate = float(metrics["success_rate"])
        objective_score = float(metrics["avg_objective_score"])
        if success_rate > self.best_success_rate or (success_rate == self.best_success_rate and objective_score > self.best_objective):
            self.best_success_rate = success_rate
            self.best_objective = objective_score
            self.model.save(self.run_dir / "best_model.zip")

        write_json(self.run_dir / "validation_metrics.json", {"history": self.validation_history})
        return metrics

    def _on_step(self) -> bool:
        if self.n_calls % self.config.eval_frequency != 0:
            return True

        self.run_validation()
        return True


class ActionDebugCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.action_counts: dict[int, int] = {}
        self.invalid_count = 0
        self.remap_events = 0
        self.valid_action_total = 0
        self.samples = 0
        self.stage_name = "stage_1_easy"

    def set_stage(self, stage: CurriculumStage) -> None:
        self.stage_name = stage.name

    def _on_step(self) -> bool:
        actions = self.locals.get("actions")
        infos = self.locals.get("infos")
        if actions is None or infos is None:
            return True

        for action, info in zip(actions, infos, strict=False):
            action_idx = int(action)
            self.action_counts[action_idx] = self.action_counts.get(action_idx, 0) + 1
            self.invalid_count += int(bool(info.get("invalid_action")))
            self.remap_events += int(bool(info.get("invalid_action")) and bool(info.get("invalid_action_remaps")))
            self.valid_action_total += int(info.get("valid_action_count", 0))
            self.samples += 1

        if self.num_timesteps % 1_000 == 0 and self.samples > 0:
            total = sum(self.action_counts.values())
            distribution = ", ".join(f"{action}:{count / total:.2f}" for action, count in sorted(self.action_counts.items()))
            print(
                "Action debug | "
                f"stage={self.stage_name} | "
                f"steps={self.num_timesteps} | "
                f"invalid={self.invalid_count / self.samples:.3f} | "
                f"remap={self.remap_events / self.samples:.3f} | "
                f"avg_valid_actions={self.valid_action_total / self.samples:.2f} | "
                f"dist=[{distribution}]"
            )
            self.action_counts = {}
            self.invalid_count = 0
            self.remap_events = 0
            self.valid_action_total = 0
            self.samples = 0
        return True


def build_model(config: PPOTrainingConfig) -> MaskablePPO:
    initial_stage = config.curriculum_stages[0]
    env = create_training_env(config.training_seeds, "ppo_train", initial_stage.scenario_name)
    return MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        device=config.device,
        policy_kwargs=config.policy_kwargs,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        clip_range=config.clip_range,
        seed=config.training_seeds[0],
    )


def create_run_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path("experiment_results/ppo") / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_metadata(run_dir: Path, config: PPOTrainingConfig, extra_metadata: dict[str, Any]) -> None:
    payload = {
        "checkpoint_version": PPO_CHECKPOINT_VERSION,
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        **asdict(config),
        **extra_metadata,
    }
    write_json(run_dir / "metadata.json", payload)


def train_with_curriculum(model: MaskablePPO, config: PPOTrainingConfig, validation_callback: ValidationCallback, action_debug_callback: ActionDebugCallback, run_dir: Path) -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    for index, stage in enumerate(config.curriculum_stages):
        stage_env = create_training_env(config.training_seeds, "ppo_train", stage.scenario_name)
        model.set_env(stage_env)
        validation_callback.set_stage(stage)
        action_debug_callback.set_stage(stage)

        transition = {
            "stage_index": index,
            "stage_name": stage.name,
            "scenario_name": stage.scenario_name,
            "timesteps": stage.timesteps,
            "started_total_timesteps": model.num_timesteps,
        }
        transitions.append(transition)
        write_json(run_dir / "curriculum_transitions.json", {"history": transitions})
        print("Curriculum stage | " f"index={index + 1}/{len(config.curriculum_stages)} | " f"name={stage.name} | " f"scenario={stage.scenario_name} | " f"timesteps={stage.timesteps}")
        model.learn(total_timesteps=stage.timesteps, callback=[validation_callback, action_debug_callback], progress_bar=False, reset_num_timesteps=False)
        latest_metrics = validation_callback.validation_history[-1] if validation_callback.validation_history else None
        if latest_metrics is None or int(latest_metrics.get("timesteps", -1)) != int(model.num_timesteps):
            validation_callback.run_validation()
        transition["completed_total_timesteps"] = model.num_timesteps
        write_json(run_dir / "curriculum_transitions.json", {"history": transitions})
    return transitions


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_model(model_path: str | Path, device: str) -> MaskablePPO:
    return MaskablePPO.load(Path(model_path), device=device)
