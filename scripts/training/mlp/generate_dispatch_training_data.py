from __future__ import annotations

# python -m scripts.training.mlp.generate_dispatch_training_data --difficulties clatsop_winter_ops --teachers tournament --train-seeds 50 --name clatsop_winter

from argparse import Namespace
import argparse
from pathlib import Path
import time

from SimPyTest.policies import BENCHMARK_POLICIES

from .ml_dispatch import DemonstrationBatch, append_demonstration_batches, collect_policy_demonstrations, write_json


class MyNamespace(Namespace):
    difficulties: str | None
    teachers: str
    train_seeds: int
    max_visible_disasters: int
    output_dir: str
    name: str | None


def parse_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def collect_demonstration_dataset(
    *,
    difficulties_csv: str | None,
    teachers_csv: str,
    train_seeds: int,
    max_visible_disasters: int,
) -> DemonstrationBatch:
    difficulties = parse_csv(difficulties_csv)
    teacher_names = parse_csv(teachers_csv)
    if not difficulties:
        raise ValueError("At least one difficulty is required when collecting fresh demonstrations")
    if not teacher_names:
        raise ValueError("At least one teacher policy is required")

    teacher_lookup = {policy.name: policy for policy in BENCHMARK_POLICIES}
    teacher_policies = [teacher_lookup[name] for name in teacher_names]
    seed_values = list(range(train_seeds))

    batches: list[DemonstrationBatch] = []
    for difficulty in difficulties:
        batches.append(
            collect_policy_demonstrations(
                teacher_policies=teacher_policies,
                difficulty=difficulty,
                seed_values=seed_values,
                max_visible_disasters=max_visible_disasters,
            )
        )

    dataset = append_demonstration_batches(*batches)
    dataset.metadata.update(
        {
            "difficulties": difficulties,
            "teachers": teacher_names,
            "train_seeds": train_seeds,
            "max_visible_disasters": max_visible_disasters,
            "num_examples": int(dataset.actions.shape[0]),
        }
    )
    dataset.validate_for_training()
    return dataset


def parse_args() -> MyNamespace:
    parser = argparse.ArgumentParser(description="Generate and save dispatch teacher demonstrations for later training.")
    parser.add_argument("--difficulties", type=str, default=None)
    parser.add_argument("--teachers", type=str, default="tournament")
    parser.add_argument("--train-seeds", type=int, default=10)
    parser.add_argument("--max-visible-disasters", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="experiment_results/dispatch_datasets")
    parser.add_argument("--name", type=str, default=None)
    return parser.parse_args(namespace=MyNamespace())


if __name__ == "__main__":
    args = parse_args()
    dataset = collect_demonstration_dataset(
        difficulties_csv=args.difficulties,
        teachers_csv=args.teachers,
        train_seeds=args.train_seeds,
        max_visible_disasters=args.max_visible_disasters,
    )

    name = time.strftime("%Y%m%d_%H%M%S")
    if args.name:
        name = args.name + "_" + name
    output_dir = Path(args.output_dir) / name
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = output_dir / "demonstrations.npz"
    dataset.save(dataset_path)
    write_json(output_dir / "demonstrations_meta.json", dataset.metadata)

    print(f"Saved demonstration dataset to {dataset_path}")
    print(f"Train using: `python -m scripts.training.mlp.train_dispatch_ml --dataset experiment_results/dispatch_datasets/{name}/demonstrations.npz`")
