from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, TextIO, override

import numpy as np
import numpy.typing as npt
from simpy.core import EmptySchedule
import torch
from torch import nn
from torch.nn.modules.container import Sequential
from torch.utils.data import DataLoader, TensorDataset

from SimPyTest.engine import SimPySimulationEngine
from SimPyTest.gym import DisasterResponseGym, ObsType
from SimPyTest.policies import Policy
from benchmark import create_scenario_config


def flatten_observation(observation: ObsType) -> npt.NDArray[np.float32]:
    return np.concatenate(
        (
            observation["current_resource"],
            observation["global_state"],
            observation["candidate_disasters"].reshape(-1),
            observation["valid_actions"].astype(np.float32),
        )
    ).astype(np.float32)


@dataclass
class DemonstrationBatch:
    current_resource: npt.NDArray[np.float32]
    global_state: npt.NDArray[np.float32]
    candidate_disasters: npt.NDArray[np.float32]
    observations: npt.NDArray[np.float32]
    actions: npt.NDArray[np.int64]
    action_masks: npt.NDArray[np.float32]
    metadata: dict[str, Any]

    def num_examples(self) -> int:
        return int(self.actions.shape[0])

    def validate_for_training(self) -> None:
        if self.num_examples() == 0:
            raise ValueError("Demonstration dataset is empty. Regenerate it with a collector that produced at least one decision example.")

        expected_shapes = {
            "current_resource": 2,
            "global_state": 2,
            "candidate_disasters": 3,
            "observations": 2,
            "actions": 1,
            "action_masks": 2,
        }
        actual_shapes = {
            "current_resource": self.current_resource.ndim,
            "global_state": self.global_state.ndim,
            "candidate_disasters": self.candidate_disasters.ndim,
            "observations": self.observations.ndim,
            "actions": self.actions.ndim,
            "action_masks": self.action_masks.ndim,
        }
        bad = [name for name, ndim in actual_shapes.items() if ndim != expected_shapes[name]]
        if bad:
            details = ", ".join(f"{name} ndim={actual_shapes[name]}" for name in bad)
            raise ValueError(f"Demonstration dataset has invalid shapes for training: {details}")

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            current_resource=self.current_resource,
            global_state=self.global_state,
            candidate_disasters=self.candidate_disasters,
            observations=self.observations,
            actions=self.actions,
            action_masks=self.action_masks,
            metadata_json=np.asarray(json.dumps(self.metadata), dtype=np.str_),
        )

    @classmethod
    def load(cls, path: str | Path) -> "DemonstrationBatch":
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(
                current_resource=np.asarray(data["current_resource"], dtype=np.float32),
                global_state=np.asarray(data["global_state"], dtype=np.float32),
                candidate_disasters=np.asarray(data["candidate_disasters"], dtype=np.float32),
                observations=np.asarray(data["observations"], dtype=np.float32),
                actions=np.asarray(data["actions"], dtype=np.int64),
                action_masks=np.asarray(data["action_masks"], dtype=np.float32),
                metadata=json.loads(str(data["metadata_json"].item())),
            )


@dataclass
class TrainingHistory:
    epochs: list[dict[str, float]]


@dataclass
class _ConsoleProgressBar:
    total: int
    label: str
    stream: TextIO = sys.stderr
    current: int = 0
    _isatty: bool = False

    def __post_init__(self) -> None:
        self.current = 0
        self.total = max(int(self.total), 1)
        self._isatty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.render()

    def update(self, step: int = 1, *, status: str | None = None) -> None:
        self.current = min(self.total, self.current + step)
        self.render(status=status)

    def render(self, *, status: str | None = None) -> None:
        width = 28
        ratio = self.current / self.total
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        message = f"{self.label:<24} [{bar}] {self.current:>3}/{self.total:<3} {ratio * 100:>6.2f}%"
        if status:
            message = f"{message} {status}"
        if self._isatty:
            print(f"\r{message}", end="", file=self.stream, flush=True)
        else:
            print(message, file=self.stream, flush=True)

    def close(self, *, status: str | None = None) -> None:
        self.current = self.total
        self.render(status=status)
        if self._isatty:
            print(file=self.stream, flush=True)


@dataclass
class DispatchScoringConfig:
    current_dim: int
    global_dim: int
    candidate_dim: int
    output_dim: int
    hidden_dim: int = 128
    depth: int = 2
    dropout: float = 0.1


class DispatchScoringModel(nn.Module):
    def __init__(self, config: DispatchScoringConfig):
        super().__init__()
        # Example shapes for one batch:
        #  - current_resource: [B, current_dim] e.g. [32, 3]
        #  - global_state:    [B, global_dim]  e.g. [32, 4]
        #  - candidate_disasters (per decision): [B, max_visible_disasters, candidate_dim]
        #    e.g. [32, 5, 6]
        # The model scores each candidate as a scalar, then picks one index.
        context_layers: list[nn.Module] = []
        context_in = config.current_dim + config.global_dim
        for _ in range(config.depth):
            context_layers.append(nn.Linear(context_in, config.hidden_dim))
            context_layers.append(nn.ReLU())
            context_layers.append(nn.Dropout(config.dropout))
            context_in = config.hidden_dim
        self.context_encoder: Sequential = nn.Sequential(*context_layers)

        candidate_layers: list[nn.Module] = []
        candidate_in = config.hidden_dim + config.candidate_dim
        for _ in range(config.depth):
            candidate_layers.append(nn.Linear(candidate_in, config.hidden_dim))
            candidate_layers.append(nn.ReLU())
            candidate_layers.append(nn.Dropout(config.dropout))
            candidate_in = config.hidden_dim
        candidate_layers.append(nn.Linear(candidate_in, 1))
        self.candidate_head: Sequential = nn.Sequential(*candidate_layers)

    @override
    def forward(self, current_resource: torch.Tensor, global_state: torch.Tensor, candidate_disasters: torch.Tensor) -> torch.Tensor:
        # Input shapes:
        # current_resource:   [B, current_dim]
        # global_state:       [B, global_dim]
        # candidate_disasters:[B, max_visible_disasters, candidate_dim]
        # Example: [32, 3], [32, 4], [32, 5, 6]

        # Concatenate state context -> [B, current_dim + global_dim]
        context = torch.cat((current_resource, global_state), dim=1)
        # Encode context once -> [B, hidden_dim]
        context_features = self.context_encoder(context)
        # Expand context per candidate for shared scoring:
        # [B, hidden_dim] -> [B, 5, hidden_dim] when max_visible_disasters=5
        repeated_context = context_features.unsqueeze(1).expand(-1, candidate_disasters.shape[1], -1)
        # For each candidate, append candidate features:
        # [B, 5, hidden_dim] + [B, 5, candidate_dim] -> [B, 5, hidden_dim + candidate_dim]
        candidate_inputs = torch.cat((repeated_context, candidate_disasters), dim=2)
        # Score each candidate with shared head -> [B, 5, 1] -> squeeze -> [B, 5]
        # e.g. [32, 5]
        return self.candidate_head(candidate_inputs).squeeze(-1)


@dataclass
class TrainedDispatchPolicy:
    model: nn.Module
    device: str
    max_visible_disasters: int
    metadata: dict[str, Any]

    def predict(self, observation: ObsType, deterministic: bool = True) -> int:
        valid_actions = observation["valid_actions"].astype(np.float32)
        with torch.no_grad():
            current_resource = torch.from_numpy(observation["current_resource"]).float().unsqueeze(0).to(self.device)
            global_state = torch.from_numpy(observation["global_state"]).float().unsqueeze(0).to(self.device)
            candidate_disasters = torch.from_numpy(observation["candidate_disasters"]).float().unsqueeze(0).to(self.device)
            logits = self.model(current_resource, global_state, candidate_disasters).squeeze(0).cpu().numpy()

        masked = np.where(valid_actions > 0, logits, -1e9)
        if np.all(valid_actions <= 0):
            return 0
        if deterministic:
            return int(np.argmax(masked))

        exp = np.exp(masked - np.max(masked))
        probs = exp / np.sum(exp)
        return int(np.random.choice(len(probs), p=probs))

    def save(self, output_dir: str | Path) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "metadata": self.metadata,
                "max_visible_disasters": self.max_visible_disasters,
            },
            path / "dispatch_model.pt",
        )
        with (path / "dispatch_model_meta.json").open("w", encoding="utf-8") as handle:
            json.dump(self.metadata, handle, indent=2, sort_keys=True)

    @classmethod
    def load(cls, model_path: str | Path, device: str = "cpu") -> "TrainedDispatchPolicy":
        payload = torch.load(model_path, map_location=device)
        metadata = dict(payload["metadata"])
        config = DispatchScoringConfig(
            current_dim=int(metadata["current_dim"]),
            global_dim=int(metadata["global_dim"]),
            candidate_dim=int(metadata["candidate_dim"]),
            output_dim=int(metadata["output_dim"]),
            hidden_dim=int(metadata["hidden_dim"]),
            depth=int(metadata["depth"]),
            dropout=float(metadata["dropout"]),
        )
        model = DispatchScoringModel(config)
        model.load_state_dict(payload["state_dict"])
        model.to(device)
        model.eval()
        return cls(
            model=model,
            device=device,
            max_visible_disasters=int(payload["max_visible_disasters"]),
            metadata=metadata,
        )


def teacher_action_from_env(env: DisasterResponseGym, teacher_policy: Policy) -> int:
    if env.current_resource is None:
        raise RuntimeError("Teacher requested action outside a decision point")
    if not env.current_candidates or not env.engine:
        return 0

    target = teacher_policy.func(env.current_resource, list(env.current_candidates), env.engine.env)
    if target is None:
        return 0

    for idx, disaster in enumerate(env.current_candidates):
        if disaster.id == target.id:
            return idx
    return 0


def has_single_valid_action(observation: ObsType) -> bool:
    return int(np.count_nonzero(observation["valid_actions"])) <= 1


def _advance_engine_to_policy_decision(engine: SimPySimulationEngine) -> bool:
    while engine.pending_decision_resource is None and engine.last_terminal_outcome is None:
        try:
            engine.advance_to_next_event()
        except EmptySchedule:
            engine.last_terminal_outcome = engine.infer_terminal_outcome(schedule_exhausted=True)
        except Exception as exc:
            engine.last_terminal_error = repr(exc)
            engine.last_terminal_outcome = SimPySimulationEngine.TERMINAL_FAIL_INVALID_STATE
    return engine.pending_decision_resource is not None


def _apply_pending_decision(engine: SimPySimulationEngine, target_disaster: Any) -> None:
    resource = engine.pending_decision_resource
    if resource is None:
        raise RuntimeError("No pending decision resource to dispatch")

    if engine.branch_decision is None:
        engine.branch_decision = target_disaster.id
    engine.decision_log.append(target_disaster.id)
    engine.decisions_made += 1

    dispatch_delay = engine._consume_dispatch_delay_minutes(target_disaster)
    if dispatch_delay > 0:
        engine.metrics.record_dispatch_delay(target_disaster.id, dispatch_delay)
        engine.total_dispatch_delay += dispatch_delay
        engine.dispatch_delay_events += 1
        engine.max_dispatch_delay = max(engine.max_dispatch_delay, dispatch_delay)
        engine.env.run(until=engine.env.timeout(dispatch_delay))
        if not target_disaster.active or target_disaster not in engine.disaster_store.items:
            resource.assigned_node = engine.idle_resources
            engine.idle_resources.mark_resource_available(resource)
            engine.pending_decision_resource = None
            return

    target_disaster.transfer_resource(resource)
    engine.pending_decision_resource = None
    engine._main_loop_process = engine.env.process(engine.loop())


def _sync_env_to_decision_point(env: DisasterResponseGym) -> bool:
    if env.engine is None:
        return False
    if env.decision_needed and env.current_resource is not None:
        return True
    env._advance_to_decision_or_terminal()
    return env.decision_needed and env.current_resource is not None


def collect_policy_demonstrations(
    *,
    teacher_policies: list[Policy],
    difficulty: str,
    seed_values: list[int],
    max_visible_disasters: int,
    max_steps_per_seed: int = 25_000,
    successful_episodes_only: bool = False,
) -> DemonstrationBatch:
    observations: list[npt.NDArray[np.float32]] = []
    current_resource_rows: list[npt.NDArray[np.float32]] = []
    global_state_rows: list[npt.NDArray[np.float32]] = []
    candidate_rows: list[npt.NDArray[np.float32]] = []
    actions: list[int] = []
    masks: list[npt.NDArray[np.float32]] = []
    teacher_counts: dict[str, int] = {policy.name: 0 for policy in teacher_policies}
    episode_rows: list[dict[str, Any]] = []
    progress = _ConsoleProgressBar(
        total=len(teacher_policies) * max(len(seed_values), 1),
        label=f"Dataset {difficulty}",
    )

    for teacher_policy in teacher_policies:
        for seed in seed_values:
            cfg = create_scenario_config(difficulty, gis_config=None)
            env = DisasterResponseGym(
                max_visible_disasters=max_visible_disasters,
                sorting_strategy="nearest",
                scenario_config=cfg,
                controller_name=f"teacher_{teacher_policy.name}",
                scenario_name=difficulty,
            )
            teacher_driver = Policy("collector_noop", lambda resource, disasters, env: None)
            engine = SimPySimulationEngine(policy=teacher_driver, seed=seed, scenario_config=cfg)
            engine.initialize_world()
            engine.stop_before_policy_decision = True
            engine.run_in_gym(engine.loop)
            env.engine = engine
            env.current_resource = None
            env.current_candidates = []
            env.current_action = None
            env.decision_needed = False
            steps = 0
            episode_observations: list[npt.NDArray[np.float32]] = []
            episode_current: list[npt.NDArray[np.float32]] = []
            episode_global: list[npt.NDArray[np.float32]] = []
            episode_candidates: list[npt.NDArray[np.float32]] = []
            episode_actions: list[int] = []
            episode_masks: list[npt.NDArray[np.float32]] = []
            while True:
                if not _advance_engine_to_policy_decision(engine):
                    terminal_outcome = engine.last_terminal_outcome or SimPySimulationEngine.TERMINAL_FAIL_INVALID_STATE
                    success = bool(terminal_outcome == SimPySimulationEngine.TERMINAL_SUCCESS)
                    if success or not successful_episodes_only:
                        observations.extend(episode_observations)
                        current_resource_rows.extend(episode_current)
                        global_state_rows.extend(episode_global)
                        candidate_rows.extend(episode_candidates)
                        actions.extend(episode_actions)
                        masks.extend(episode_masks)
                        teacher_counts[teacher_policy.name] += len(episode_actions)
                    episode_rows.append(
                        {
                            "teacher_policy": teacher_policy.name,
                            "seed": seed,
                            "steps": steps,
                            "terminal_outcome": terminal_outcome,
                            "success": success,
                            "included_in_dataset": success or not successful_episodes_only,
                        }
                    )
                    progress.update(status=f"teacher={teacher_policy.name} seed={seed} success={'yes' if success else 'no'}")
                    break
                resource = engine.pending_decision_resource
                if resource is None:
                    raise RuntimeError("Engine reached a policy decision without a pending resource")
                actionable = [disaster for disaster in engine.disaster_store.items if resource.resource_type in disaster.needed_resources()]
                env.current_resource = resource
                env.current_candidates = env._sort_candidates(resource, actionable)[: env.max_slots]
                env.decision_needed = True
                obs = env._get_obs()

                action = teacher_action_from_env(env, teacher_policy)
                episode_observations.append(flatten_observation(obs))
                episode_current.append(obs["current_resource"].copy())
                episode_global.append(obs["global_state"].copy())
                episode_candidates.append(obs["candidate_disasters"].copy())
                episode_actions.append(int(action))
                episode_masks.append(obs["valid_actions"].astype(np.float32))
                _apply_pending_decision(engine, env.current_candidates[action])
                env.current_resource = None
                env.current_candidates = []
                env.decision_needed = False
                steps += 1
                if steps >= max_steps_per_seed:
                    terminal_outcome = engine.last_terminal_outcome or SimPySimulationEngine.TERMINAL_FAIL_TIMEOUT
                    success = bool(terminal_outcome == SimPySimulationEngine.TERMINAL_SUCCESS)
                    if success or not successful_episodes_only:
                        observations.extend(episode_observations)
                        current_resource_rows.extend(episode_current)
                        global_state_rows.extend(episode_global)
                        candidate_rows.extend(episode_candidates)
                        actions.extend(episode_actions)
                        masks.extend(episode_masks)
                        teacher_counts[teacher_policy.name] += len(episode_actions)
                    episode_rows.append(
                        {
                            "teacher_policy": teacher_policy.name,
                            "seed": seed,
                            "steps": steps,
                            "terminal_outcome": terminal_outcome,
                            "success": success,
                            "included_in_dataset": success,
                        }
                    )
                    progress.update(status=f"teacher={teacher_policy.name} seed={seed} success={'yes' if success else 'no'}")
                    break
    progress.close(status=f"examples={len(actions)}")

    return DemonstrationBatch(
        current_resource=np.asarray(current_resource_rows, dtype=np.float32),
        global_state=np.asarray(global_state_rows, dtype=np.float32),
        candidate_disasters=np.asarray(candidate_rows, dtype=np.float32),
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        action_masks=np.asarray(masks, dtype=np.float32),
        metadata={
            "difficulty": difficulty,
            "seed_values": list(seed_values),
            "teacher_counts": teacher_counts,
            "episodes": episode_rows,
            "successful_episodes_only": successful_episodes_only,
        },
    )


def append_demonstration_batches(*batches: DemonstrationBatch) -> DemonstrationBatch:
    if not batches:
        raise ValueError("At least one demonstration batch is required")
    if len(batches) == 1:
        return batches[0]

    teacher_counts: dict[str, int] = {}
    merged_from: list[dict[str, Any]] = []
    for batch in batches:
        merged_from.append(batch.metadata)
        for key, value in batch.metadata.get("teacher_counts", {}).items():
            teacher_counts[key] = teacher_counts.get(key, 0) + int(value)

    return DemonstrationBatch(
        current_resource=np.concatenate([batch.current_resource for batch in batches], axis=0),
        global_state=np.concatenate([batch.global_state for batch in batches], axis=0),
        candidate_disasters=np.concatenate([batch.candidate_disasters for batch in batches], axis=0),
        observations=np.concatenate([batch.observations for batch in batches], axis=0),
        actions=np.concatenate([batch.actions for batch in batches], axis=0),
        action_masks=np.concatenate([batch.action_masks for batch in batches], axis=0),
        metadata={
            "merged_from": merged_from,
            "teacher_counts": teacher_counts,
        },
    )


def _split_indices(size: int, validation_fraction: float, seed: int) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(size)
    rng.shuffle(indices)
    val_size = max(1, int(size * validation_fraction))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    if len(train_indices) == 0:
        train_indices = val_indices
    return train_indices.astype(np.int64), val_indices.astype(np.int64)


def train_behavior_cloning(
    *,
    dataset: DemonstrationBatch,
    max_visible_disasters: int,
    seed: int = 0,
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    validation_fraction: float = 0.1,
    hidden_dim: int = 256,
    depth: int = 3,
    dropout: float = 0.1,
    device: str = "cpu",
    model_type: str = "candidate_scorer",
) -> tuple[TrainedDispatchPolicy, TrainingHistory]:
    output_dim = int(max_visible_disasters)
    current_dim = int(dataset.current_resource.shape[1])
    global_dim = int(dataset.global_state.shape[1])
    candidate_dim = int(dataset.candidate_disasters.shape[2])
    input_dim = int(dataset.observations.shape[1])

    config = DispatchScoringConfig(
        current_dim=current_dim,
        global_dim=global_dim,
        candidate_dim=candidate_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        depth=depth,
        dropout=dropout,
    )
    model = DispatchScoringModel(config).to(device)

    train_indices, val_indices = _split_indices(len(dataset.actions), validation_fraction, seed)
    x = torch.from_numpy(dataset.observations)
    x_current = torch.from_numpy(dataset.current_resource)
    x_global = torch.from_numpy(dataset.global_state)
    x_candidates = torch.from_numpy(dataset.candidate_disasters)
    y = torch.from_numpy(dataset.actions)
    mask = torch.from_numpy(dataset.action_masks)

    train_loader = DataLoader(
        TensorDataset(
            x[train_indices],
            x_current[train_indices],
            x_global[train_indices],
            x_candidates[train_indices],
            y[train_indices],
            mask[train_indices],
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            x[val_indices],
            x_current[val_indices],
            x_global[val_indices],
            x_candidates[val_indices],
            y[val_indices],
            mask[val_indices],
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    class_counts = np.bincount(dataset.actions, minlength=output_dim).astype(np.float32)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).float().to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history_rows: list[dict[str, float]] = []
    progress = _ConsoleProgressBar(total=epochs, label="Training epochs")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for batch_x, batch_current, batch_global, batch_candidates, batch_y, batch_mask in train_loader:
            batch_x = batch_x.to(device)
            batch_current = batch_current.to(device)
            batch_global = batch_global.to(device)
            batch_candidates = batch_candidates.to(device)
            batch_y = batch_y.to(device)
            batch_mask = batch_mask.to(device)

            if model_type == "candidate_scorer":
                logits = model(batch_current, batch_global, batch_candidates)
            else:
                logits = model(batch_x)
            masked_logits = torch.where(batch_mask > 0, logits, torch.full_like(logits, -1e9))
            loss = criterion(masked_logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item()) * int(batch_y.shape[0])
            preds = torch.argmax(masked_logits, dim=1)
            train_correct += int((preds == batch_y).sum().item())
            train_total += int(batch_y.shape[0])

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_current, batch_global, batch_candidates, batch_y, batch_mask in val_loader:
                batch_x = batch_x.to(device)
                batch_current = batch_current.to(device)
                batch_global = batch_global.to(device)
                batch_candidates = batch_candidates.to(device)
                batch_y = batch_y.to(device)
                batch_mask = batch_mask.to(device)
                if model_type == "candidate_scorer":
                    logits = model(batch_current, batch_global, batch_candidates)
                else:
                    logits = model(batch_x)
                masked_logits = torch.where(batch_mask > 0, logits, torch.full_like(logits, -1e9))
                loss = criterion(masked_logits, batch_y)
                val_loss += float(loss.item()) * int(batch_y.shape[0])
                preds = torch.argmax(masked_logits, dim=1)
                val_correct += int((preds == batch_y).sum().item())
                val_total += int(batch_y.shape[0])

        history_rows.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": train_loss / max(train_total, 1),
                "train_accuracy": train_correct / max(train_total, 1),
                "val_loss": val_loss / max(val_total, 1),
                "val_accuracy": val_correct / max(val_total, 1),
            }
        )
        latest = history_rows[-1]
        progress.update(status=(f"train_loss={latest['train_loss']:.4f} " f"val_loss={latest['val_loss']:.4f} " f"val_acc={latest['val_accuracy']:.3f}"))
    progress.close()

    metadata = {
        "model_type": model_type,
        "input_dim": input_dim,
        "current_dim": current_dim,
        "global_dim": global_dim,
        "candidate_dim": candidate_dim,
        "output_dim": output_dim,
        "hidden_dim": hidden_dim,
        "depth": depth,
        "dropout": dropout,
        "max_visible_disasters": max_visible_disasters,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "final_train_accuracy": history_rows[-1]["train_accuracy"] if history_rows else 0.0,
        "final_val_accuracy": history_rows[-1]["val_accuracy"] if history_rows else 0.0,
    }
    trained = TrainedDispatchPolicy(
        model=model.eval(),
        device=device,
        max_visible_disasters=max_visible_disasters,
        metadata=metadata,
    )
    return trained, TrainingHistory(epochs=history_rows)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
