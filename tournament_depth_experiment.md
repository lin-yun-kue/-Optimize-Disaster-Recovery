# Tournament Depth Experiment Results

- Generated: `2026-02-25T04:13:19`
- Tournament policy under test: `tournament_recursive`
- Enabled baseline policies: `first_priority, split_excavator, smallest_job_first, balanced_ratio, smart_split, cost_function, chain_gang, seasonal_priority, resource_season_match, cost_efficient, population_impact, budget_aware, resource_efficiency`
- Tournament candidate policies: `first_priority, split_excavator, smallest_job_first, balanced_ratio, smart_split, cost_function, chain_gang, seasonal_priority, resource_season_match, cost_efficient, population_impact, budget_aware, resource_efficiency`
- Tournament multiprocessing disabled: `True`
- Tournament candidate whitelist: `None`
- Tournament depths tested: `[1, 2, 3]`
- Total runs: `1280`
- Total wall time: `6067.38s`

## Overall Summary

| Scenario                |     Best Baseline (success-first) |   T d=1 |   T d=2 |   T d=3 | Best Tournament Depth |
| ----------------------- | --------------------------------: | ------: | ------: | ------: | --------------------- |
| `easy_nonseasonal`      |  170.93 (first_priority, 100.00%) |  170.58 |  170.58 |  170.58 | d=1                   |
| `medium_nonseasonal`    |  3234.44 (cost_function, 100.00%) | 3230.98 | 3230.81 | 3230.52 | d=3                   |
| `seasonal_winter_light` | 1977.94 (first_priority, 100.00%) | 1943.46 | 1681.20 | 1883.54 | d=2                   |
| `seasonal_summer_light` |  276.55 (balanced_ratio, 100.00%) |  275.64 |  277.21 |  277.89 | d=1                   |

## Per-Scenario Details

### `easy_nonseasonal`

- Small non-seasonal scenario; includes depth 3 for trend checks.
- Seeds: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]`
- Seed count: `20`
- Tournament depths: `[1, 2, 3]`
- Tournament policy: `tournament_recursive`
- Scenario config:
  - `num_trucks`: `[18, 24]`
  - `num_excavators`: `[10, 14]`
  - `num_snowplows`: `0`
  - `num_landslides`: `[2, 3]`
  - `landslide_size_range`: `[100, 150]`
  - `landslide_distance_range`: `[500, 1000]`
  - `calendar_start_date`: `None`
  - `calendar_duration_years`: `1`
  - `use_seasonal_disasters`: `False`
  - `use_weather_modifier`: `False`
  - `annual_budget`: `10000000`
  - `track_costs`: `False`
  - `time_variance`: `0.0`
  - `custom_setup_fn`: `None`
  - `gis_config`: `None`

**Baselines (enabled policies)**

| Policy                  | Success % | Avg Sim | Avg Wall (s) | Sim Stdev |
| ----------------------- | --------: | ------: | -----------: | --------: |
| `first_priority`        |   100.00% |  170.93 |         0.00 |     69.68 |
| `population_impact`     |   100.00% |  171.24 |         0.00 |     70.25 |
| `chain_gang`            |   100.00% |  171.24 |         0.00 |     70.25 |
| `resource_season_match` |   100.00% |  171.32 |         0.00 |     70.24 |
| `budget_aware`          |   100.00% |  171.32 |         0.00 |     70.24 |
| `cost_function`         |   100.00% |  171.32 |         0.00 |     70.24 |
| `seasonal_priority`     |   100.00% |  171.32 |         0.00 |     70.24 |
| `smallest_job_first`    |   100.00% |  171.32 |         0.00 |     70.24 |
| `cost_efficient`        |   100.00% |  171.32 |         0.00 |     70.24 |
| `resource_efficiency`   |   100.00% |  171.64 |         0.00 |     70.80 |
| `balanced_ratio`        |   100.00% |  171.64 |         0.00 |     70.80 |
| `smart_split`           |   100.00% |  171.64 |         0.00 |     70.80 |
| `split_excavator`       |   100.00% |  171.64 |         0.00 |     70.80 |

**Tournament Depth Sweep**

| Depth | Success % | Avg Sim | Avg Wall (s) | Sim Stdev | Delta vs Best Baseline (Sim) |
| ----: | --------: | ------: | -----------: | --------: | ---------------------------: |
|     1 |   100.00% |  170.58 |         0.47 |     69.24 |                        -0.35 |
|     2 |   100.00% |  170.58 |         1.26 |     69.24 |                        -0.35 |
|     3 |   100.00% |  170.58 |         2.33 |     69.24 |                        -0.35 |

**Paired Seed Comparisons (Tournament vs Best Baseline)**

| Depth | Shared Seeds | Both Success | T faster | Baseline faster | T-only success | Baseline-only success | Avg Sim Delta (T-B) | Avg Wall Delta s (T-B) |
| ----: | -----------: | -----------: | -------: | --------------: | -------------: | --------------------: | ------------------: | ---------------------: |
|     1 |           20 |           20 |        2 |               0 |              0 |                     0 |               -0.35 |                   0.47 |
|     2 |           20 |           20 |        2 |               0 |              0 |                     0 |               -0.35 |                   1.26 |
|     3 |           20 |           20 |        2 |               0 |              0 |                     0 |               -0.35 |                   2.32 |

**Paired Depth Comparisons (Tournament, seed-matched)**

| Comparison | Shared Seeds | Both Success | Left faster | Right faster | Avg Sim Delta (L-R) | Avg Wall Delta s (L-R) |
| ---------- | -----------: | -----------: | ----------: | -----------: | ------------------: | ---------------------: |
| `d1_vs_d2` |           20 |           20 |           0 |            0 |                0.00 |                  -0.79 |
| `d2_vs_d3` |           20 |           20 |           0 |            0 |                0.00 |                  -1.07 |

Depth trend notes:

- d=1 -> d=2: sim delta +0.00, wall delta +0.79s
- d=2 -> d=3: sim delta +0.00, wall delta +1.07s

### `medium_nonseasonal`

- Benchmark-like medium scenario with one seed (expensive for depth>1).
- Seeds: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]`
- Seed count: `20`
- Tournament depths: `[1, 2, 3]`
- Tournament policy: `tournament_recursive`
- Scenario config:
  - `num_trucks`: `[15, 25]`
  - `num_excavators`: `[8, 12]`
  - `num_snowplows`: `0`
  - `num_landslides`: `[8, 15]`
  - `landslide_size_range`: `[200, 400]`
  - `landslide_distance_range`: `[1000, 2000]`
  - `calendar_start_date`: `None`
  - `calendar_duration_years`: `1`
  - `use_seasonal_disasters`: `False`
  - `use_weather_modifier`: `False`
  - `annual_budget`: `10000000`
  - `track_costs`: `False`
  - `time_variance`: `0.0`
  - `custom_setup_fn`: `None`
  - `gis_config`: `None`

**Baselines (enabled policies)**

| Policy                  | Success % | Avg Sim | Avg Wall (s) | Sim Stdev |
| ----------------------- | --------: | ------: | -----------: | --------: |
| `cost_function`         |   100.00% | 3234.44 |         0.03 |    748.72 |
| `balanced_ratio`        |   100.00% | 3235.17 |         0.02 |    747.30 |
| `smart_split`           |   100.00% | 3235.17 |         0.03 |    747.30 |
| `split_excavator`       |   100.00% | 3235.44 |         0.03 |    747.26 |
| `chain_gang`            |   100.00% | 3253.85 |         0.03 |    761.01 |
| `population_impact`     |   100.00% | 3253.85 |         0.03 |    761.01 |
| `budget_aware`          |   100.00% | 3253.99 |         0.03 |    754.67 |
| `seasonal_priority`     |   100.00% | 3253.99 |         0.03 |    754.67 |
| `resource_season_match` |   100.00% | 3253.99 |         0.03 |    754.67 |
| `cost_efficient`        |   100.00% | 3253.99 |         0.04 |    754.67 |
| `resource_efficiency`   |   100.00% | 3260.01 |         0.03 |    757.21 |
| `first_priority`        |   100.00% | 3262.36 |         0.03 |    755.99 |
| `smallest_job_first`    |    95.00% | 3245.51 |         0.03 |    774.37 |

**Tournament Depth Sweep**

| Depth | Success % | Avg Sim | Avg Wall (s) | Sim Stdev | Delta vs Best Baseline (Sim) |
| ----: | --------: | ------: | -----------: | --------: | ---------------------------: |
|     1 |   100.00% | 3230.98 |       140.53 |    748.43 |                        -3.46 |
|     2 |   100.00% | 3230.81 |       445.76 |    748.32 |                        -3.62 |
|     3 |   100.00% | 3230.52 |      1559.73 |    748.18 |                        -3.92 |

**Paired Seed Comparisons (Tournament vs Best Baseline)**

| Depth | Shared Seeds | Both Success | T faster | Baseline faster | T-only success | Baseline-only success | Avg Sim Delta (T-B) | Avg Wall Delta s (T-B) |
| ----: | -----------: | -----------: | -------: | --------------: | -------------: | --------------------: | ------------------: | ---------------------: |
|     1 |           20 |           20 |       18 |               0 |              0 |                     0 |               -3.46 |                 140.51 |
|     2 |           20 |           20 |       19 |               0 |              0 |                     0 |               -3.62 |                 445.74 |
|     3 |           20 |           20 |       19 |               0 |              0 |                     0 |               -3.92 |                1559.70 |

**Paired Depth Comparisons (Tournament, seed-matched)**

| Comparison | Shared Seeds | Both Success | Left faster | Right faster | Avg Sim Delta (L-R) | Avg Wall Delta s (L-R) |
| ---------- | -----------: | -----------: | ----------: | -----------: | ------------------: | ---------------------: |
| `d1_vs_d2` |           20 |           20 |           3 |            3 |                0.16 |                -305.23 |
| `d2_vs_d3` |           20 |           20 |           0 |            5 |                0.30 |               -1113.97 |

Depth trend notes:

- d=1 -> d=2: sim delta -0.16, wall delta +305.23s
- d=2 -> d=3: sim delta -0.30, wall delta +1113.97s

### `seasonal_winter_light`

- Winter seasonal/weather scenario (lighter than benchmark seasonal).
- Seeds: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]`
- Seed count: `20`
- Tournament depths: `[1, 2, 3]`
- Tournament policy: `tournament_recursive`
- Scenario config:
  - `num_trucks`: `[12, 16]`
  - `num_excavators`: `[6, 8]`
  - `num_snowplows`: `[2, 4]`
  - `num_landslides`: `[5, 8]`
  - `landslide_size_range`: `[150, 300]`
  - `landslide_distance_range`: `[900, 1600]`
  - `calendar_start_date`: `2024-01-01T00:00:00`
  - `calendar_duration_years`: `1`
  - `use_seasonal_disasters`: `True`
  - `use_weather_modifier`: `True`
  - `annual_budget`: `10000000`
  - `track_costs`: `False`
  - `time_variance`: `0.0`
  - `custom_setup_fn`: `None`
  - `gis_config`: `None`

**Baselines (enabled policies)**

| Policy                  | Success % | Avg Sim | Avg Wall (s) | Sim Stdev |
| ----------------------- | --------: | ------: | -----------: | --------: |
| `first_priority`        |   100.00% | 1977.94 |         0.00 |   1759.04 |
| `cost_function`         |   100.00% | 2068.31 |         0.01 |   1896.52 |
| `smart_split`           |   100.00% | 2068.31 |         0.01 |   1896.52 |
| `seasonal_priority`     |   100.00% | 2068.81 |         0.00 |   1896.07 |
| `cost_efficient`        |   100.00% | 2068.81 |         0.01 |   1896.07 |
| `population_impact`     |   100.00% | 2068.81 |         0.01 |   1896.07 |
| `resource_season_match` |   100.00% | 2068.97 |         0.00 |   1895.92 |
| `budget_aware`          |   100.00% | 2068.97 |         0.01 |   1895.92 |
| `resource_efficiency`   |   100.00% | 2069.81 |         0.01 |   1896.00 |
| `chain_gang`            |   100.00% | 2070.31 |         0.01 |   1894.58 |
| `balanced_ratio`        |   100.00% | 2070.47 |         0.01 |   1895.40 |
| `split_excavator`       |   100.00% | 2076.40 |         0.00 |   1889.45 |
| `smallest_job_first`    |    95.00% | 2070.62 |         0.00 |   1947.86 |

**Tournament Depth Sweep**

| Depth | Success % | Avg Sim | Avg Wall (s) | Sim Stdev | Delta vs Best Baseline (Sim) |
| ----: | --------: | ------: | -----------: | --------: | ---------------------------: |
|     1 |   100.00% | 1943.46 |         1.82 |   1771.16 |                       -34.48 |
|     2 |   100.00% | 1681.20 |         4.82 |   1403.20 |                      -296.74 |
|     3 |   100.00% | 1883.54 |        10.39 |   1793.68 |                       -94.40 |

**Paired Seed Comparisons (Tournament vs Best Baseline)**

| Depth | Shared Seeds | Both Success | T faster | Baseline faster | T-only success | Baseline-only success | Avg Sim Delta (T-B) | Avg Wall Delta s (T-B) |
| ----: | -----------: | -----------: | -------: | --------------: | -------------: | --------------------: | ------------------: | ---------------------: |
|     1 |           20 |           20 |        3 |               0 |              0 |                     0 |              -34.48 |                   1.81 |
|     2 |           20 |           20 |        5 |               0 |              0 |                     0 |             -296.74 |                   4.81 |
|     3 |           20 |           20 |        5 |               0 |              0 |                     0 |              -94.40 |                  10.38 |

**Paired Depth Comparisons (Tournament, seed-matched)**

| Comparison | Shared Seeds | Both Success | Left faster | Right faster | Avg Sim Delta (L-R) | Avg Wall Delta s (L-R) |
| ---------- | -----------: | -----------: | ----------: | -----------: | ------------------: | ---------------------: |
| `d1_vs_d2` |           20 |           20 |           1 |            3 |              262.26 |                  -3.00 |
| `d2_vs_d3` |           20 |           20 |           2 |            4 |             -202.33 |                  -5.57 |

Depth trend notes:

- d=1 -> d=2: sim delta -262.26, wall delta +3.00s
- d=2 -> d=3: sim delta +202.33, wall delta +5.57s

### `seasonal_summer_light`

- Summer seasonal/weather scenario (lighter than benchmark seasonal).
- Seeds: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]`
- Seed count: `20`
- Tournament depths: `[1, 2, 3]`
- Tournament policy: `tournament_recursive`
- Scenario config:
  - `num_trucks`: `[12, 16]`
  - `num_excavators`: `[6, 8]`
  - `num_snowplows`: `[0, 1]`
  - `num_landslides`: `[5, 8]`
  - `landslide_size_range`: `[150, 300]`
  - `landslide_distance_range`: `[900, 1600]`
  - `calendar_start_date`: `2024-07-01T00:00:00`
  - `calendar_duration_years`: `1`
  - `use_seasonal_disasters`: `True`
  - `use_weather_modifier`: `True`
  - `annual_budget`: `10000000`
  - `track_costs`: `False`
  - `time_variance`: `0.0`
  - `custom_setup_fn`: `None`
  - `gis_config`: `None`

**Baselines (enabled policies)**

| Policy                  | Success % | Avg Sim | Avg Wall (s) | Sim Stdev |
| ----------------------- | --------: | ------: | -----------: | --------: |
| `balanced_ratio`        |   100.00% |  276.55 |         0.00 |     94.90 |
| `chain_gang`            |   100.00% |  279.45 |         0.01 |     95.45 |
| `cost_efficient`        |   100.00% |  279.70 |         0.02 |     95.14 |
| `cost_function`         |   100.00% |  279.97 |         0.00 |     96.64 |
| `population_impact`     |   100.00% |  280.51 |         0.01 |     97.14 |
| `smart_split`           |   100.00% |  280.61 |         0.00 |     95.64 |
| `split_excavator`       |   100.00% |  281.57 |         0.00 |    100.47 |
| `first_priority`        |   100.00% |  282.83 |         0.00 |     96.64 |
| `seasonal_priority`     |   100.00% |  282.91 |         0.01 |     90.25 |
| `resource_efficiency`   |   100.00% |  283.46 |         0.01 |    101.06 |
| `budget_aware`          |   100.00% |  283.79 |         0.01 |    100.82 |
| `resource_season_match` |   100.00% |  284.70 |         0.00 |    103.32 |
| `smallest_job_first`    |    40.00% |  248.68 |         0.00 |    105.55 |

**Tournament Depth Sweep**

| Depth | Success % | Avg Sim | Avg Wall (s) | Sim Stdev | Delta vs Best Baseline (Sim) |
| ----: | --------: | ------: | -----------: | --------: | ---------------------------: |
|     1 |   100.00% |  275.64 |         0.88 |     93.27 |                        -0.91 |
|     2 |   100.00% |  277.21 |         2.78 |     94.41 |                         0.66 |
|     3 |   100.00% |  277.89 |         4.90 |     93.90 |                         1.34 |

**Paired Seed Comparisons (Tournament vs Best Baseline)**

| Depth | Shared Seeds | Both Success | T faster | Baseline faster | T-only success | Baseline-only success | Avg Sim Delta (T-B) | Avg Wall Delta s (T-B) |
| ----: | -----------: | -----------: | -------: | --------------: | -------------: | --------------------: | ------------------: | ---------------------: |
|     1 |           20 |           20 |        4 |               4 |              0 |                     0 |               -0.91 |                   0.88 |
|     2 |           20 |           20 |        6 |               5 |              0 |                     0 |                0.66 |                   2.77 |
|     3 |           20 |           20 |        3 |               6 |              0 |                     0 |                1.34 |                   4.90 |

**Paired Depth Comparisons (Tournament, seed-matched)**

| Comparison | Shared Seeds | Both Success | Left faster | Right faster | Avg Sim Delta (L-R) | Avg Wall Delta s (L-R) |
| ---------- | -----------: | -----------: | ----------: | -----------: | ------------------: | ---------------------: |
| `d1_vs_d2` |           20 |           20 |           4 |            4 |               -1.57 |                  -1.90 |
| `d2_vs_d3` |           20 |           20 |           6 |            2 |               -0.68 |                  -2.13 |

Depth trend notes:

- d=1 -> d=2: sim delta +1.57, wall delta +1.90s
- d=2 -> d=3: sim delta +0.68, wall delta +2.13s

## Notes

- `Avg Sim` is the simulation's `non_idle_time` metric (not wall-clock runtime).
- `Avg Wall` is measured with `time.perf_counter()` around each full run.
- Tournament depth cost scales quickly with number of decision points; wall-time trends matter more than sim-time trends for usability.
