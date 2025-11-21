Some run stats:

```bash
python main.py --seeds=1
```

```python
class SimulationConfig:
    DROP_OFF_LOCATION = (75, 75)
    DEPOT_LOCATION = (0, 0)
    HOSPITAL_LOCATION = (-75, -75)

    # Ticks should be in minutes

    # Disaster Specifics
    LANDSLIDE_MIN_SIZE = 150
    LANDSLIDE_MAX_SIZE = 250
    NUM_STARTING_LANDSLIDES = 10
    NUM_LANDSLIDES = 15

    NUM_TRUCKS = 100
    NUM_EXCAVATORS = 15
    NUM_FIRE_TRUCKS = 4
    NUM_AMBULANCES = 5
```

> Tournament is the policy switching that runs all policies for every decision and picks the best one.

| POLICY               | SUCCESS %  | TIME       |
|----------------------|------------|------------|
| tournament           | 100.0    % | 2324.04    |
| balanced_ratio       | 100.0    % | 3022.78    |
| random               | 100.0    % | 3134.17    |
| first_priority       | 100.0    % | 3478.69    |
| split_excavator      | 100.0    % | 3482.48    |
| smart_split          | 100.0    % | 3489.35    |
| cost_function        | 100.0    % | 3611.96    |
| chain_gang           | 100.0    % | 4154.77    |
| smallest_job_first   | 100.0    % | 4535.75    |
| closest_neighbor     | 0.0      % | DNF        |
| gravity              | 0.0      % | DNF        |


