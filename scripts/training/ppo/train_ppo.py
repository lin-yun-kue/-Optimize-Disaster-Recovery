from __future__ import annotations

# python -m scripts.training.ppo.train_ppo

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from scripts.training.ppo.ppo_dispatch import (
    ActionDebugCallback,
    ValidationCallback,
    build_model,
    build_training_config,
    create_run_dir,
    evaluate_curriculum_stages,
    evaluate_model,
    load_model,
    save_run_metadata,
    train_with_curriculum,
    write_json,
)


if __name__ == "__main__":
    config = build_training_config()
    run_dir = create_run_dir()
    model = build_model(config)
    callback = ValidationCallback(config, run_dir)
    action_debug_callback = ActionDebugCallback()

    print(
        "Training PPO checkpoint baseline | "
        f"scenario={config.scenario_name} | "
        f"device={config.device} | "
        f"timesteps={config.total_timesteps} | "
        f"stages={len(config.curriculum_stages)}"
    )

    transitions = train_with_curriculum(model, config, callback, action_debug_callback, run_dir)
    model.save(run_dir / "last_model.zip")
    best_model = load_model(run_dir / "best_model.zip", config.device)

    final_stage = config.curriculum_stages[-1]

    checkpoint_metrics = evaluate_model(
        best_model,
        config.checkpoint_seeds,
        True,
        "ppo_checkpoint",
        final_stage.scenario_name,
    )
    write_json(run_dir / "checkpoint_metrics.json", checkpoint_metrics)
    stage_checkpoint_metrics = evaluate_curriculum_stages(
        best_model,
        config,
        config.checkpoint_seeds,
        True,
        "ppo_checkpoint_stage",
    )
    write_json(run_dir / "checkpoint_stage_metrics.json", stage_checkpoint_metrics)
    save_run_metadata(
        run_dir,
        config,
        {
            "checkpoint_target": "Beat smallest_job_first on held-out winter seeds by objective score with equal or better success rate.",
            "validation_history_path": str((run_dir / "validation_metrics.json").resolve()),
            "checkpoint_metrics_path": str((run_dir / "checkpoint_metrics.json").resolve()),
            "checkpoint_stage_metrics_path": str((run_dir / "checkpoint_stage_metrics.json").resolve()),
            "best_model_path": str((run_dir / "best_model.zip").resolve()),
            "last_model_path": str((run_dir / "last_model.zip").resolve()),
            "curriculum_transitions_path": str((run_dir / "curriculum_transitions.json").resolve()),
            "curriculum_stage_count": len(config.curriculum_stages),
            "curriculum_transitions": transitions,
            "final_checkpoint_stage_name": final_stage.name,
            "final_checkpoint_scenario_name": final_stage.scenario_name,
        },
    )

    print(
        "Checkpoint | "
        f"stage={final_stage.name} | "
        f"obj={checkpoint_metrics['avg_objective_score']:.2f} | "
        f"success={checkpoint_metrics['success_rate'] * 100:.1f}% | "
        f"reward={checkpoint_metrics['avg_total_reward']:.2f}"
    )
    print(f"Saved PPO run to {run_dir}")
