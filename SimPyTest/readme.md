# SimPyTest Package

Package-level documentation for the simulation core used by this repository.

This file is intentionally scoped to what is implemented in the current tree.

## What This Package Contains

- `engine.py`
  - `ScenarioConfig`
  - `SimPySimulationEngine`
- `simulation.py`
  - domain objects/resources/disasters (`Resource`, `ResourceType`, `Disaster`, `Landslide`, `SnowEvent`, `WildfireDebris`, `FloodEvent`)
- `policies.py`
  - heuristic and tournament dispatch policies
- `gym.py`
  - `DisasterResponseEnv` gymnasium environment
- `gis_utils.py`
  - GIS loading/routing/pruning utilities
- `metrics_tracker.py`
  - simulation metrics and impact estimation helpers
- `calendar.py`
  - seasonality/weather process utilities
- `real_world_params.py`
  - prior/default conversion helpers

## Entry Points

From repository root:

```bash
python -m SimPyTest.main --seeds 5
python -m SimPyTest.main --policy balanced_ratio --seeds 5
python benchmark.py --difficulty clatsop_winter_ops --seed-set smoke
python benchmark.py --difficulty clatsop_winter_ops --seed-set smoke --check-invariants
python scripts/testing/run_invariant_smoke.py
```

## Exported API (`SimPyTest/__init__.py`)

```python
from SimPyTest import SimPySimulationEngine, ScenarioConfig
```

`__init__.py` also re-exports key simulation, gym, and GIS symbols used by examples.

## Available Policies (current)

- `first_priority`
- `split_excavator`
- `smallest_job_first`
- `balanced_ratio`
- `smart_split`
- `cost_function`
- `chain_gang`
- `tournament`
- `tournament_recursive`
- `seasonal_priority`
- `resource_season_match`
- `cost_efficient`
- `population_impact`
- `budget_aware`
- `resource_efficiency`

## Notes

- Canonical project-level docs live in repository root `README.md`, `PROJECT.md`, and `TODO.md`.
- Treat this package doc as implementation-facing and keep it synchronized with code paths/CLI flags.
