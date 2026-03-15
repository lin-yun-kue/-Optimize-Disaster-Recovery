from __future__ import annotations

from argparse import Namespace
import argparse
from pathlib import Path
import time
from typing import cast

from .generate_dispatch_training_data import collect_demonstration_dataset
from .ml_dispatch import DemonstrationBatch, append_demonstration_batches, train_behavior_cloning, write_json


class MyNamespace(Namespace):
    difficulties: str | None = None
    teachers: str = ""
    train_seeds: int = 0
    eval_seeds: int = 0
    seed: int = 0
    epochs: int = 0
    batch_size: int = 0
    learning_rate: float = 0.0
    weight_decay: float = 0.0
    hidden_dim: int = 0
    depth: int = 0
    dropout: float = 0.0
    max_visible_disasters: int = 0
    save_dir: str = ""
    dataset: str | None = None


def parse_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> MyNamespace:
    parser = argparse.ArgumentParser(description="Train a generic dispatch ML policy from teacher demonstrations.")
    parser.add_argument(
        "--difficulties",
        type=str,
        default=None,
        help="Comma-separated difficulties to train across when collecting fresh demonstrations.",
    )
    parser.add_argument("--teachers", type=str, default="tournament")
    parser.add_argument("--train-seeds", type=int, default=10)
    parser.add_argument("--eval-seeds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-visible-disasters", type=int, default=5)
    parser.add_argument("--save-dir", type=str, default="experiment_results/dispatch_ml")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to a saved demonstration dataset (.npz), or a comma-separated list of dataset paths. If provided, teacher rollout collection is skipped.",
    )
    return cast(MyNamespace, parser.parse_args())


def load_or_collect_dataset(args: MyNamespace) -> DemonstrationBatch:
    if args.dataset:
        dataset_paths = parse_csv(args.dataset)
        batches = [DemonstrationBatch.load(path) for path in dataset_paths]
        if not batches:
            raise ValueError("At least one dataset path is required when --dataset is provided")
        return append_demonstration_batches(*batches)
    return collect_demonstration_dataset(
        difficulties_csv=args.difficulties,
        teachers_csv=args.teachers,
        train_seeds=args.train_seeds,
        max_visible_disasters=args.max_visible_disasters,
    )


if __name__ == "__main__":
    args = parse_args()

    name = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.save_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"args: {args.save_dir}")
    print(f"name: {name}")
    print(f"run_dir: {run_dir}")

    dataset = load_or_collect_dataset(args)
    dataset.validate_for_training()
    policy, history = train_behavior_cloning(
        dataset=dataset,
        max_visible_disasters=args.max_visible_disasters,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
        device="mps",
    )
    policy.metadata.update(
        {
            "dataset_path": args.dataset,
            "dataset_paths": parse_csv(args.dataset),
            "train_seeds": args.train_seeds,
            "eval_seeds": args.eval_seeds,
            "dataset_metadata": dataset.metadata,
        }
    )
    policy.save(run_dir)
    write_json(run_dir / "training_history.json", {"epochs": history.epochs})

    print(f"Saved generic dispatch model to {run_dir}")
