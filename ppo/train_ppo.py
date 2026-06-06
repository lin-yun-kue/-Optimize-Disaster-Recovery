from __future__ import annotations

# python -m ppo.train_ppo

from ppo.ppo_dispatch import (
    ActionDebugCallback,
    ValidationCallback,
    build_model,
    build_training_config,
    create_run_dir,
    save_run_metadata,
    train_with_curriculum,
)

if __name__ == "__main__":
    config = build_training_config()
    run_dir = create_run_dir()
    model = build_model(config, run_dir)
    callback = ValidationCallback(config, run_dir)
    action_debug_callback = ActionDebugCallback()

    print("Training PPO baseline | " f"device={config.device} | " f"timesteps={config.total_timesteps} | " f"stages={len(config.curriculum_stages)}")

    transitions = train_with_curriculum(model, config, callback, action_debug_callback, run_dir)
    model.save(run_dir / "last_model.zip")
    save_run_metadata(
        run_dir,
        config,
        {
            "benchmark_target": "Benchmark PPO through benchmark.py against easy-winter, medium-winter, and hard-winter using holdout seeds.",
            "validation_history_path": str((run_dir / "validation_metrics.json").resolve()),
            "best_model_path": str((run_dir / "best_model.zip").resolve()),
            "last_model_path": str((run_dir / "last_model.zip").resolve()),
            "tensorboard_log_dir": str((run_dir / "tensorboard").resolve()),
            "curriculum_transitions_path": str((run_dir / "curriculum_transitions.json").resolve()),
            "curriculum_stage_count": len(config.curriculum_stages),
            "curriculum_transitions": transitions,
            "final_stage_name": config.curriculum_stages[-1].name,
        },
    )
    print(f"Saved PPO run to {run_dir}")
