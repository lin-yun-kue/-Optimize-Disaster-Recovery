from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time
from typing import Any, TypedDict, cast

from sb3_contrib import MaskablePPO
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from typing_extensions import override

from SimPyTest.evaluation import KPIBundle, compute_kpi_bundle
from SimPyTest.gym import CURRENT_RESOURCE_FEATURES, DISASTER_FEATURES, GLOBAL_STATE_FEATURES, DisasterResponseGym, InfoType, ObsType
from SimPyTest.scenario_types import ScenarioConfig, SeasonalDisasterConfig
from SimPyTest.simulation import Disaster
import random

PPO_CHECKPOINT_VERSION = "v2"
TRAINING_SEEDS = list(range(128))
VALIDATION_SEEDS = list(range(-20, -10))
SCENARIO_NAME = "landslide_winter_curriculum"
MAX_VISIBLE_DISASTERS = 5
SORTING_STRATEGY = "random"
EVAL_FREQUENCY = 10_000
EVAL_MAX_WALL_TIME_S = 240.0


class EpisodeResult(TypedDict):
    seed: int
    success: bool
    sim_time: float
    time_with_disasters: float
    wall_time_s: float
    total_reward: float
    objective_score: float
    terminal_outcome: str | None
    invalid_action_count: int
    invalid_action_remaps: int
    valid_action_ratio: float
    kpis: KPIBundle
    decisions: int


class EvaluationMetrics(TypedDict):
    episodes: list[EpisodeResult]
    success_rate: float
    avg_objective_score: float
    avg_total_reward: float
    avg_sim_time: float
    avg_time_with_disasters: float
    avg_wall_time_s: float
    avg_invalid_action_count: float
    avg_invalid_action_remaps: float
    avg_valid_action_ratio: float
    successes: int
    runs: int


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    benchmark_preset: str
    timesteps: int
    scenario_config: ScenarioConfig


@dataclass(frozen=True)
class ValidationTarget:
    log_name: str
    scenario_name: str
    benchmark_preset: str
    scenario_config: ScenarioConfig
    deterministic: bool
    save_best: bool = False


@dataclass(frozen=True)
class PPOTrainingConfig:
    total_timesteps: int
    eval_frequency: int
    training_seeds: list[int]
    validation_seeds: list[int]
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
    curriculum_stages: list[CurriculumStage]


class SeedCyclingEnv(DisasterResponseGym):
    training_seeds: list[int]
    seed_index: int

    def __init__(self, training_seeds: list[int], controller_name: str, stage: CurriculumStage):
        self.training_seeds = list(training_seeds)
        random.shuffle(self.training_seeds)
        self.seed_index = 0
        super().__init__(
            max_visible_disasters=MAX_VISIBLE_DISASTERS,
            sorting_strategy=SORTING_STRATEGY,
            scenario_config=stage.scenario_config,
            controller_name=controller_name,
        )
        self.action_space = spaces.Discrete(self.max_slots)
        self.observation_space = cast(
            spaces.Space[ObsType],
            spaces.Dict(
                {
                    "current_resource": spaces.Box(low=0.0, high=1.0, shape=(CURRENT_RESOURCE_FEATURES,), dtype=np.float32),
                    "global_state": spaces.Box(low=0.0, high=1.0, shape=(GLOBAL_STATE_FEATURES,), dtype=np.float32),
                    "candidate_disasters": spaces.Box(low=0.0, high=1.0, shape=(self.max_slots, DISASTER_FEATURES), dtype=np.float32),
                    "valid_actions": spaces.Box(low=0, high=1, shape=(self.max_slots,), dtype=np.int8),
                }
            ),
        )

    @override
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[ObsType, InfoType]:
        if seed is None:
            seed = self.training_seeds[self.seed_index % len(self.training_seeds)]
            self.seed_index += 1
        return super().reset(seed=seed, options=options)

    @override
    def _valid_actions(self) -> np.ndarray[tuple[int], np.dtype[np.int8]]:
        valid_actions = np.zeros(self.max_slots, dtype=np.int8)
        valid_actions[: len(self.current_candidates)] = 1
        return valid_actions

    @override
    def _zero_obs(self) -> ObsType:
        return {
            "current_resource": np.zeros(CURRENT_RESOURCE_FEATURES, dtype=np.float32),
            "global_state": np.zeros(GLOBAL_STATE_FEATURES, dtype=np.float32),
            "candidate_disasters": np.zeros((self.max_slots, DISASTER_FEATURES), dtype=np.float32),
            "valid_actions": np.zeros(self.max_slots, dtype=np.int8),
        }

    @override
    def _resolve_action(self, action: int) -> tuple[str, Disaster | None, bool]:
        if 0 <= action < len(self.current_candidates):
            return "DISPATCH", self.current_candidates[action], False
        return "INVALID", None, True


def _seasonal_counts(winter: tuple[int, int], default: tuple[int, int] = (0, 0)) -> dict[str, tuple[int, int]]:
    return {
        "winter": winter,
        "spring": default,
        "summer": default,
        "fall": default,
    }


def _lerp_int(start: int, end: int, numerator: int, denominator: int) -> int:
    if denominator <= 0:
        return end
    return int(round(start + (end - start) * (numerator / denominator)))


def _landslide_only_stage_config(
    base: ScenarioConfig,
    landslide_count: tuple[int, int],
    landslide_size: tuple[int, int],
) -> ScenarioConfig:
    seasonal_spawn = deepcopy(base.seasonal_spawn)
    seasonal_spawn["landslide"] = SeasonalDisasterConfig(
        event_count_range_by_season=_seasonal_counts(landslide_count),
        size_range_by_season=_seasonal_counts(landslide_size, landslide_size),
    )
    seasonal_spawn["wildfire_debris"] = SeasonalDisasterConfig(
        event_count_range_by_season=_seasonal_counts((0, 0)),
        size_range_by_season=_seasonal_counts((1, 1), (1, 1)),
    )
    return ScenarioConfig(
        resource_counts=base.resource_counts,
        seasonal_spawn=seasonal_spawn,
        time_variance=base.time_variance,
        calendar_start_date=base.calendar_start_date,
        calendar_duration_years=base.calendar_duration_years,
        gis_config=None,
    )


def build_curriculum_stages() -> list[CurriculumStage]:
    from benchmark import create_scenario_config

    anchors = [
        ("easy-winter", create_scenario_config("easy-winter", gis_config=None), 3, 12_000),
        ("medium-winter", create_scenario_config("medium-winter", gis_config=None), 3, 16_000),
        ("hard-winter", create_scenario_config("hard-winter", gis_config=None), 4, 20_000),
    ]
    previous_count = (1, 2)
    previous_size = (100, 800)
    stages: list[CurriculumStage] = []

    for preset, scenario, ramp_steps, stage_timesteps in anchors:
        landslide = scenario.seasonal_spawn["landslide"]
        target_count = landslide.event_count_range_by_season["winter"]
        target_size = landslide.size_range_by_season["winter"]
        for step_index in range(ramp_steps):
            numerator = step_index + 1
            count_range = (
                _lerp_int(previous_count[0], target_count[0], numerator, ramp_steps),
                _lerp_int(previous_count[1], target_count[1], numerator, ramp_steps),
            )
            size_range = (
                _lerp_int(previous_size[0], target_size[0], numerator, ramp_steps),
                _lerp_int(previous_size[1], target_size[1], numerator, ramp_steps),
            )
            stage_name = f"{preset.replace('-', '_')}r{step_index + 1:02d}"
            stages.append(
                CurriculumStage(
                    name=stage_name,
                    benchmark_preset=preset,
                    timesteps=stage_timesteps,
                    scenario_config=_landslide_only_stage_config(scenario, count_range, size_range),
                )
            )
        previous_count = target_count
        previous_size = target_size
    return stages


def build_training_config() -> PPOTrainingConfig:
    curriculum_stages = build_curriculum_stages()
    total_timesteps = sum(stage.timesteps for stage in curriculum_stages)
    return PPOTrainingConfig(
        total_timesteps=total_timesteps,
        eval_frequency=EVAL_FREQUENCY,
        training_seeds=list(TRAINING_SEEDS),
        validation_seeds=list(VALIDATION_SEEDS),
        max_visible_disasters=MAX_VISIBLE_DISASTERS,
        sorting_strategy=SORTING_STRATEGY,
        device=select_device(),
        policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
        learning_rate=5e-4,
        n_steps=1024,
        batch_size=256,
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


def create_env(seed: int, controller_name: str, stage: CurriculumStage) -> SeedCyclingEnv:
    return SeedCyclingEnv([seed], controller_name, stage)


def create_training_env(training_seeds: list[int], controller_name: str, stage: CurriculumStage) -> SeedCyclingEnv:
    return SeedCyclingEnv(training_seeds, controller_name, stage)


def predict_action(model: object, env: SeedCyclingEnv, observation: ObsType, deterministic: bool) -> int:
    predictor = cast(Any, model).predict
    try:
        predicted = predictor(observation, deterministic=deterministic, action_masks=env.action_masks())
    except TypeError:
        predicted = predictor(observation, deterministic=deterministic)
    return int(predicted[0])


def run_policy_episode(model: object, seed: int, deterministic: bool, controller_name: str, scenario_name: str, scenario_config: ScenarioConfig) -> EpisodeResult:
    stage = CurriculumStage(
        name=scenario_name,
        benchmark_preset=scenario_name,
        timesteps=0,
        scenario_config=scenario_config,
    )
    env = create_env(seed, controller_name, stage)
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
        if elapsed_wall_time_s >= EVAL_MAX_WALL_TIME_S:
            print("Evaluation cutoff | " f"seed={seed} | " f"decisions={decisions} | " f"wall={elapsed_wall_time_s:.1f}s | " f"training_obj={info['objective_score']:.2f}")
            terminated = True
            info["terminal_outcome"] = "EVAL_CUTOFF"
            info["is_success"] = False

    return {
        "seed": seed,
        "success": bool(info["is_success"]),
        "sim_time": float(info["sim_time"]),
        "time_with_disasters": float(info["summary"].time_with_disasters),
        "wall_time_s": time.perf_counter() - started_at,
        "total_reward": total_reward,
        "objective_score": float(info["objective_score"]),
        "terminal_outcome": info["terminal_outcome"],
        "invalid_action_count": int(info["invalid_action_count"]),
        "invalid_action_remaps": int(info["invalid_action_remaps"]),
        "valid_action_ratio": total_valid_action_ratio / decisions if decisions > 0 else 0.0,
        "kpis": compute_kpi_bundle(info["summary"]),
        "decisions": decisions,
    }


def evaluate_model(
    model: object,
    seeds: list[int],
    deterministic: bool,
    controller_name: str,
    scenario_name: str,
    scenario_config: ScenarioConfig,
) -> EvaluationMetrics:
    episodes: list[EpisodeResult] = []
    for seed in seeds:
        episode = run_policy_episode(model, seed, deterministic, controller_name, scenario_name, scenario_config)
        episodes.append(episode)
        print(f"Evaluation seed {seed} success: {episode['success']}")

    successes = [episode for episode in episodes if episode["success"]]
    return {
        "episodes": episodes,
        "success_rate": len(successes) / len(episodes) if episodes else 0.0,
        "avg_objective_score": statistics.mean(episode["objective_score"] for episode in episodes) if episodes else 0.0,
        "avg_total_reward": statistics.mean(episode["total_reward"] for episode in episodes) if episodes else 0.0,
        "avg_sim_time": statistics.mean(episode["sim_time"] for episode in episodes) if episodes else 0.0,
        "avg_time_with_disasters": statistics.mean(episode["time_with_disasters"] for episode in episodes) if episodes else 0.0,
        "avg_wall_time_s": statistics.mean(episode["wall_time_s"] for episode in episodes) if episodes else 0.0,
        "avg_invalid_action_count": statistics.mean(episode["invalid_action_count"] for episode in episodes) if episodes else 0.0,
        "avg_invalid_action_remaps": statistics.mean(episode["invalid_action_remaps"] for episode in episodes) if episodes else 0.0,
        "avg_valid_action_ratio": statistics.mean(episode["valid_action_ratio"] for episode in episodes) if episodes else 0.0,
        "successes": len(successes),
        "runs": len(episodes),
    }


# def evaluate_curriculum_stages(model: object, config: PPOTrainingConfig, seeds: list[int], deterministic: bool, controller_name: str) -> dict[str, dict[str, Any]]:
#     stage_metrics: dict[str, dict[str, Any]] = {}
#     for stage in config.curriculum_stages:
#         metrics = evaluate_model(
#             model,
#             seeds,
#             deterministic,
#             controller_name,
#             stage.scenario_config,
#         )
#         metrics["stage_name"] = stage.name
#         metrics["benchmark_preset"] = stage.benchmark_preset
#         stage_metrics[stage.name] = metrics
#     return stage_metrics


class ValidationCallback(BaseCallback):
    def __init__(self, config: PPOTrainingConfig, run_dir: Path):
        super().__init__()
        self.config = config
        self.run_dir = run_dir
        self.best_target_success_rate = float("-inf")
        self.best_target_objective = float("-inf")
        self.validation_history: list[dict[str, Any]] = []
        self.stage = config.curriculum_stages[0]

    def set_stage(self, stage: CurriculumStage) -> None:
        self.stage = stage

    def _get_stage_index(self) -> int:
        for i, s in enumerate(self.config.curriculum_stages):
            if s.name == self.stage.name:
                return i
        return 0

    def _build_validation_targets(self) -> list[ValidationTarget]:
        from benchmark import create_scenario_config

        return [
            ValidationTarget(
                log_name="easy_benchmark",
                scenario_name="easy_benchmark",
                benchmark_preset="easy-winter",
                scenario_config=create_scenario_config("easy-winter", None),
                deterministic=True,
            ),
            ValidationTarget(
                log_name="medium_benchmark",
                scenario_name="medium_benchmark",
                benchmark_preset="medium-winter",
                scenario_config=create_scenario_config("medium-winter", None),
                deterministic=True,
            ),
            ValidationTarget(
                log_name="curriculum_det",
                scenario_name=self.stage.name,
                benchmark_preset=self.stage.benchmark_preset,
                scenario_config=self.stage.scenario_config,
                deterministic=True,
                save_best=True,
            ),
            ValidationTarget(
                log_name="curriculum_stoch",
                scenario_name=self.stage.name,
                benchmark_preset=self.stage.benchmark_preset,
                scenario_config=self.stage.scenario_config,
                deterministic=False,
            ),
        ]

    def _save_if_best(self, metrics: EvaluationMetrics) -> None:
        success_rate = float(metrics["success_rate"])
        objective_score = float(metrics["avg_objective_score"])
        if success_rate > self.best_target_success_rate or (success_rate == self.best_target_success_rate and objective_score > self.best_target_objective):
            self.best_target_success_rate = success_rate
            self.best_target_objective = objective_score
            self.model.save(self.run_dir / "best_model.zip")

    def run_validation(self) -> None:
        print("Validation starting | " f"steps={self.num_timesteps} | " f"seeds={self.config.validation_seeds[0]}..{self.config.validation_seeds[-1]} | " f"runs={len(self.config.validation_seeds)}")

        stage_index = self._get_stage_index()
        self.logger.record("validation/curriculum_stage_index", stage_index)
        self.logger.record("validation/current_stage_timesteps", self.stage.timesteps)

        for target in self._build_validation_targets():
            metrics = evaluate_model(
                self.model,
                self.config.validation_seeds,
                target.deterministic,
                "ppo_validation",
                target.scenario_name,
                target.scenario_config,
            )
            row = {
                **metrics,
                "timesteps": self.num_timesteps,
                "active_stage_name": self.stage.name,
                "active_stage_benchmark_preset": self.stage.benchmark_preset,
                "active_stage_index": stage_index,
                "evaluation_log_name": target.log_name,
                "evaluated_scenario_name": target.scenario_name,
                "evaluated_benchmark_preset": target.benchmark_preset,
                "deterministic": target.deterministic,
                "save_best_target": target.save_best,
            }
            self.validation_history.append(row)

            self.logger.record(f"validation/{target.log_name}-success_rate", float(metrics["success_rate"]))
            self.logger.record(f"validation/{target.log_name}-avg_obj_score", float(metrics["avg_objective_score"]))
            self.logger.record(f"validation/{target.log_name}-avg_tot_rew", float(metrics["avg_total_reward"]))
            self.logger.record(f"validation/{target.log_name}-avg_time_w_dis", float(metrics["avg_time_with_disasters"]))
            self.logger.record(f"validation/{target.log_name}-avg_inv_act_count", float(metrics["avg_invalid_action_count"]))
            self.logger.record(f"validation/{target.log_name}-avg_inv_act_remaps", float(metrics["avg_invalid_action_remaps"]))
            self.logger.record(f"validation/{target.log_name}-avg_val_act_ratio", float(metrics["avg_valid_action_ratio"]))
            self.logger.dump(self.num_timesteps)

            print(
                "Validation | "
                f"active_stage={self.stage.name} | "
                f"eval={target.log_name} | "
                f"preset={target.benchmark_preset} | "
                f"deterministic={target.deterministic} | "
                f"steps={self.num_timesteps} | "
                f"train_obj={metrics['avg_objective_score']:.2f} | "
                f"success={metrics['success_rate'] * 100:.1f}% | "
                f"reward={metrics['avg_total_reward']:.2f} | "
                f"time_with_disasters={metrics['avg_time_with_disasters']:.2f} | "
                f"invalid={metrics['avg_invalid_action_count']:.2f} | "
                f"remaps={metrics['avg_invalid_action_remaps']:.2f} | "
                f"valid_ratio={metrics['avg_valid_action_ratio']:.3f}"
            )

            if target.save_best:
                self._save_if_best(metrics)

        write_json(self.run_dir / "validation_metrics.json", {"history": self.validation_history})

    @override
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
        self.stage_name = "unknown"

    def set_stage(self, stage: CurriculumStage) -> None:
        self.stage_name = stage.name
        self.action_counts = {}
        self.invalid_count = 0
        self.remap_events = 0
        self.valid_action_total = 0
        self.samples = 0

    @override
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
            invalid_rate = self.invalid_count / self.samples
            remap_rate = self.remap_events / self.samples
            avg_valid_actions = self.valid_action_total / self.samples
            print(
                "Action debug | "
                f"stage={self.stage_name} | "
                f"steps={self.num_timesteps} | "
                f"invalid={invalid_rate:.3f} | "
                f"remap={remap_rate:.3f} | "
                f"avg_valid_actions={avg_valid_actions:.2f} | "
                f"dist=[{distribution}]"
            )
            self.logger.record("action_debug/invalid_rate", invalid_rate)
            self.logger.record("action_debug/remap_rate", remap_rate)
            self.logger.record("action_debug/avg_valid_actions", avg_valid_actions)
            self.logger.dump(self.num_timesteps)
            self.action_counts = {}
            self.invalid_count = 0
            self.remap_events = 0
            self.valid_action_total = 0
            self.samples = 0
        return True


def build_model(config: PPOTrainingConfig, run_dir: Path | None = None) -> MaskablePPO:
    initial_stage = config.curriculum_stages[0]
    env = create_training_env(config.training_seeds, "ppo_train", initial_stage)
    return MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        device=config.device,
        tensorboard_log=str((run_dir / "tensorboard").resolve()) if run_dir is not None else None,
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
    (run_dir / "tensorboard").mkdir(parents=True, exist_ok=True)
    return run_dir


def serialize_stage(stage: CurriculumStage) -> dict[str, Any]:
    landslide = stage.scenario_config.seasonal_spawn["landslide"]
    wildfire = stage.scenario_config.seasonal_spawn["wildfire_debris"]
    return {
        "name": stage.name,
        "scenario_name": stage.name,
        "benchmark_preset": stage.benchmark_preset,
        "timesteps": stage.timesteps,
        "resource_counts": {
            "trucks": stage.scenario_config.resource_counts.trucks,
            "excavators": stage.scenario_config.resource_counts.excavators,
        },
        "landslide_event_count_winter": landslide.event_count_range_by_season["winter"],
        "landslide_size_winter": landslide.size_range_by_season["winter"],
        "wildfire_event_count_winter": wildfire.event_count_range_by_season["winter"],
        "calendar_start_date": stage.scenario_config.calendar_start_date,
        "calendar_duration_years": stage.scenario_config.calendar_duration_years,
        "time_variance": stage.scenario_config.time_variance,
    }


def save_run_metadata(run_dir: Path, config: PPOTrainingConfig, extra_metadata: dict[str, Any]) -> None:
    payload = {
        "checkpoint_version": PPO_CHECKPOINT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_timesteps": config.total_timesteps,
        "eval_frequency": config.eval_frequency,
        "training_seeds": config.training_seeds,
        "validation_seeds": config.validation_seeds,
        "max_visible_disasters": config.max_visible_disasters,
        "sorting_strategy": config.sorting_strategy,
        "device": config.device,
        "policy_kwargs": config.policy_kwargs,
        "learning_rate": config.learning_rate,
        "n_steps": config.n_steps,
        "batch_size": config.batch_size,
        "n_epochs": config.n_epochs,
        "gamma": config.gamma,
        "gae_lambda": config.gae_lambda,
        "ent_coef": config.ent_coef,
        "vf_coef": config.vf_coef,
        "clip_range": config.clip_range,
        "curriculum_stages": [serialize_stage(stage) for stage in config.curriculum_stages],
        "best_model_selection_target": {
            "validation_log_name": "curriculum_det",
            "deterministic": True,
            "selection_rule": "maximize success_rate, tie-break on avg_objective_score",
        },
        **extra_metadata,
    }
    write_json(run_dir / "metadata.json", payload)


def reset_rollout_tracking(model: MaskablePPO) -> None:
    ep_info_buffer = getattr(model, "ep_info_buffer", None)
    if ep_info_buffer is not None:
        ep_info_buffer.clear()
    ep_success_buffer = getattr(model, "ep_success_buffer", None)
    if ep_success_buffer is not None:
        ep_success_buffer.clear()


def train_with_curriculum(
    model: MaskablePPO,
    config: PPOTrainingConfig,
    validation_callback: ValidationCallback,
    action_debug_callback: ActionDebugCallback,
    run_dir: Path,
) -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    for index, stage in enumerate(config.curriculum_stages):
        stage_env = create_training_env(config.training_seeds, "ppo_train", stage)
        model.set_env(stage_env)
        reset_rollout_tracking(model)
        validation_callback.set_stage(stage)
        action_debug_callback.set_stage(stage)

        transition = {
            "stage_index": index,
            "stage_name": stage.name,
            "benchmark_preset": stage.benchmark_preset,
            "timesteps": stage.timesteps,
            "started_total_timesteps": model.num_timesteps,
        }
        transitions.append(transition)
        write_json(run_dir / "curriculum_transitions.json", {"history": transitions})
        print("Curriculum stage | " f"index={index + 1}/{len(config.curriculum_stages)} | " f"name={stage.name} | " f"preset={stage.benchmark_preset} | " f"timesteps={stage.timesteps}")
        model.learn(
            total_timesteps=stage.timesteps,
            callback=[validation_callback, action_debug_callback],
            progress_bar=False,
            reset_num_timesteps=False,
            tb_log_name="ppo_dispatch",
        )
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
